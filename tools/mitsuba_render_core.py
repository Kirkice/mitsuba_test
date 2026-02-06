#!/usr/bin/env python3
"""Shared rendering core for Mitsuba soft rasterization.

This module contains all shared code between mitsuba_soft_render.py and
mitsuba_raster_fit_disney.py, including:
- Scene parsing
- Geometry generation
- Camera matrices
- Disney BRDF implementation
- Rendering pipeline

The main difference between the two scripts is:
- mitsuba_soft_render.py: Single-pass preview rendering
- mitsuba_raster_fit_disney.py: Multi-iteration material optimization
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ============================================================================
# Utility Functions
# ============================================================================

def _require(pkg: str, import_name: Optional[str] = None):
    """Import and return a required package, with a helpful error message if missing."""
    try:
        return __import__(import_name or pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg}'. Install it in your training env. Original error: {e}"
        )


def _parse_vec3(s: str) -> np.ndarray:
    """Parse a vec3 string like '1.0, 2.0, 3.0' into numpy array."""
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected vec3, got: {s!r}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)


def _xml_find_first(elem, path: str):
    """Find first XML element matching path, raising error if not found."""
    x = elem.find(path)
    if x is None:
        raise ValueError(f"XML missing required element: {path}")
    return x


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Camera:
    """Camera specification"""
    origin: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_deg: float
    width: int
    height: int


@dataclass
class ObjectSpec:
    """Object geometry specification"""
    kind: str
    translate: np.ndarray
    scale: np.ndarray
    radius: float
    filename: str


@dataclass
class LightSpec:
    """Light source specification supporting multiple types"""
    light_type: str  # 'area', 'point', 'spot', 'directional', 'envmap'
    radiance: np.ndarray  # RGB color/intensity

    # Area light
    position: np.ndarray | None = None  # Center position for area light
    scale: float = 1.0  # Area light size

    # Point light
    point_position: np.ndarray | None = None

    # Spot light
    spot_position: np.ndarray | None = None
    spot_direction: np.ndarray | None = None
    spot_cutoff_angle: float = 30.0
    spot_beam_width: float = 20.0

    # Directional light
    directional_direction: np.ndarray | None = None

    # Envmap
    envmap_filename: str | None = None
    envmap_scale: float = 1.0
    envmap_rotation: float = 0.0  # Rotation in degrees


@dataclass
class WallSpec:
    """Rectangle wall with transform and material"""
    rotate_axis: np.ndarray  # (x, y, z)
    rotate_angle: float      # degrees
    translate: np.ndarray    # (x, y, z)
    scale: np.ndarray        # (x, y, z)
    reflectance: np.ndarray  # RGB color


# ============================================================================
# Scene Parsing
# ============================================================================

def parse_scene_xml(scene_path: Path) -> Tuple[Camera, ObjectSpec, list[LightSpec], list[WallSpec]]:
    """Parse scene XML file to extract camera, objects, lights, and walls"""
    import xml.etree.ElementTree as ET

    root = ET.parse(scene_path).getroot()

    # Parse camera
    sensor = _xml_find_first(root, "./sensor")
    fov = float(_xml_find_first(sensor, "./float[@name='fov']").attrib["value"])

    lookat = _xml_find_first(sensor, "./transform[@name='to_world']/lookat")
    origin = _parse_vec3(lookat.attrib["origin"])
    target = _parse_vec3(lookat.attrib["target"])
    up = _parse_vec3(lookat.attrib["up"])

    film = _xml_find_first(sensor, "./film")
    width = int(_xml_find_first(film, "./integer[@name='width']").attrib["value"])
    height = int(_xml_find_first(film, "./integer[@name='height']").attrib["value"])

    # Parse all lights
    lights = []

    # Parse area lights (emitters attached to shapes)
    for shape in root.findall("./shape"):
        emitter = shape.find("./emitter[@type='area']")
        if emitter is None:
            continue

        light_pos = np.array([0.0, 1.99, 0.0], dtype=np.float32)
        light_rad = np.array([18.0, 18.0, 18.0], dtype=np.float32)
        light_scale = 0.35

        rgb = emitter.find("./rgb[@name='radiance']")
        if rgb is not None:
            light_rad = _parse_vec3(rgb.attrib["value"])

        tr = shape.find("./transform[@name='to_world']/translate")
        if tr is not None:
            light_pos = np.array(
                [float(tr.attrib.get("x", 0.0)), float(tr.attrib.get("y", 0.0)), float(tr.attrib.get("z", 0.0))],
                dtype=np.float32,
            )

        scale_elem = shape.find("./transform[@name='to_world']/scale")
        if scale_elem is not None:
            light_scale = float(scale_elem.attrib.get("x", 0.35))

        lights.append(LightSpec(
            light_type='area',
            radiance=light_rad,
            position=light_pos,
            scale=light_scale,
        ))

    # Parse point lights
    for emitter in root.findall("./emitter[@type='point']"):
        light_rad = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        light_pos = np.array([0.0, 1.5, 0.0], dtype=np.float32)

        rgb = emitter.find("./rgb[@name='intensity']")
        if rgb is not None:
            light_rad = _parse_vec3(rgb.attrib["value"])

        point_elem = emitter.find("./point[@name='position']")
        if point_elem is not None:
            light_pos = np.array(
                [float(point_elem.attrib.get("x", 0.0)),
                 float(point_elem.attrib.get("y", 0.0)),
                 float(point_elem.attrib.get("z", 0.0))],
                dtype=np.float32,
            )

        lights.append(LightSpec(
            light_type='point',
            radiance=light_rad,
            point_position=light_pos,
        ))

    # Parse spot lights
    for emitter in root.findall("./emitter[@type='spot']"):
        light_rad = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        spot_pos = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        spot_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        cutoff_angle = 30.0
        beam_width = 20.0

        rgb = emitter.find("./rgb[@name='intensity']")
        if rgb is not None:
            light_rad = _parse_vec3(rgb.attrib["value"])

        lookat = emitter.find("./transform[@name='to_world']/lookat")
        if lookat is not None:
            spot_pos = _parse_vec3(lookat.attrib["origin"])
            spot_target = _parse_vec3(lookat.attrib["target"])

        cutoff_elem = emitter.find("./float[@name='cutoff_angle']")
        if cutoff_elem is not None:
            cutoff_angle = float(cutoff_elem.attrib["value"])

        beam_elem = emitter.find("./float[@name='beam_width']")
        if beam_elem is not None:
            beam_width = float(beam_elem.attrib["value"])

        spot_dir = spot_target - spot_pos
        spot_dir = spot_dir / (np.linalg.norm(spot_dir) + 1e-8)

        lights.append(LightSpec(
            light_type='spot',
            radiance=light_rad,
            spot_position=spot_pos,
            spot_direction=spot_dir,
            spot_cutoff_angle=cutoff_angle,
            spot_beam_width=beam_width,
        ))

    # Parse directional lights
    for emitter in root.findall("./emitter[@type='directional']"):
        light_rad = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        light_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        rgb = emitter.find("./rgb[@name='irradiance']")
        if rgb is not None:
            light_rad = _parse_vec3(rgb.attrib["value"])

        vec_elem = emitter.find("./vector[@name='direction']")
        if vec_elem is not None:
            light_dir = np.array(
                [float(vec_elem.attrib.get("x", 0.0)),
                 float(vec_elem.attrib.get("y", -1.0)),
                 float(vec_elem.attrib.get("z", 0.0))],
                dtype=np.float32,
            )
            light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)

        lights.append(LightSpec(
            light_type='directional',
            radiance=light_rad,
            directional_direction=light_dir,
        ))

    # Parse envmap lights
    for emitter in root.findall("./emitter[@type='envmap']"):
        filename = "textures/envmap.exr"
        scale = 1.0

        filename_elem = emitter.find("./string[@name='filename']")
        if filename_elem is not None:
            filename = filename_elem.attrib["value"]

        scale_elem = emitter.find("./float[@name='scale']")
        if scale_elem is not None:
            scale = float(scale_elem.attrib["value"])

        lights.append(LightSpec(
            light_type='envmap',
            radiance=np.array([1.0, 1.0, 1.0], dtype=np.float32),  # Placeholder
            envmap_filename=filename,
            envmap_scale=scale,
        ))

    # Parse all rectangle walls (skip those with emitters - they're lights)
    walls = []
    for shape in root.findall("./shape"):
        if shape.attrib.get("type") != "rectangle":
            continue
        # Skip light rectangles
        if shape.find("./emitter[@type='area']") is not None:
            continue

        # Parse transforms
        transform = shape.find("./transform[@name='to_world']")
        rotate_axis = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        rotate_angle = 0.0
        translate = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        if transform is not None:
            rot = transform.find("./rotate")
            if rot is not None:
                rotate_axis = np.array(
                    [float(rot.attrib.get("x", 0.0)), float(rot.attrib.get("y", 0.0)), float(rot.attrib.get("z", 0.0))],
                    dtype=np.float32,
                )
                rotate_angle = float(rot.attrib.get("angle", 0.0))

            tr = transform.find("./translate")
            if tr is not None:
                translate = np.array(
                    [float(tr.attrib.get("x", 0.0)), float(tr.attrib.get("y", 0.0)), float(tr.attrib.get("z", 0.0))],
                    dtype=np.float32,
                )

            sc = transform.find("./scale")
            if sc is not None:
                if "value" in sc.attrib:
                    v = float(sc.attrib["value"])
                    scale = np.array([v, v, v], dtype=np.float32)
                else:
                    scale = np.array(
                        [float(sc.attrib.get("x", 1.0)), float(sc.attrib.get("y", 1.0)), float(sc.attrib.get("z", 1.0))],
                        dtype=np.float32,
                    )

        # Parse material reflectance
        reflectance = np.array([0.8, 0.8, 0.8], dtype=np.float32)
        bsdf = shape.find("./bsdf[@type='diffuse']")
        if bsdf is not None:
            rgb = bsdf.find("./rgb[@name='reflectance']")
            if rgb is not None:
                reflectance = _parse_vec3(rgb.attrib["value"])

        walls.append(WallSpec(
            rotate_axis=rotate_axis,
            rotate_angle=rotate_angle,
            translate=translate,
            scale=scale,
            reflectance=reflectance
        ))

    # Object parsing
    obj_kind = None
    obj_elem = None
    for shape in root.findall("./shape"):
        t = shape.attrib.get("type", "")
        if t in {"sphere", "cube", "obj", "ply"}:
            obj_kind = t
            obj_elem = shape
            break

    if obj_kind is None or obj_elem is None:
        raise ValueError("Could not find an object shape of type sphere/cube/obj/ply in the XML")

    translate = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    radius = 0.35
    filename = ""

    tr = obj_elem.find("./transform[@name='to_world']/translate")
    if tr is not None:
        translate = np.array(
            [float(tr.attrib.get("x", 0.0)), float(tr.attrib.get("y", 0.0)), float(tr.attrib.get("z", 0.0))],
            dtype=np.float32,
        )

    sc = obj_elem.find("./transform[@name='to_world']/scale")
    if sc is not None:
        if "value" in sc.attrib:
            v = float(sc.attrib["value"])
            scale = np.array([v, v, v], dtype=np.float32)
        else:
            scale = np.array(
                [float(sc.attrib.get("x", 1.0)), float(sc.attrib.get("y", 1.0)), float(sc.attrib.get("z", 1.0))],
                dtype=np.float32,
            )

    if obj_kind == "sphere":
        fl = obj_elem.find("./float[@name='radius']")
        if fl is not None:
            radius = float(fl.attrib["value"])

    if obj_kind in {"obj", "ply"}:
        fn = obj_elem.find("./string[@name='filename']")
        if fn is not None:
            filename = fn.attrib["value"]

    cam = Camera(origin=origin, target=target, up=up, fov_deg=fov, width=width, height=height)
    obj = ObjectSpec(kind=obj_kind, translate=translate, scale=scale, radius=radius, filename=filename)

    return cam, obj, lights, walls


# ============================================================================
# Camera Matrices
# ============================================================================

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Create a look-at view matrix"""
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-8)
    u = up / (np.linalg.norm(up) + 1e-8)
    s = np.cross(f, u)
    s = s / (np.linalg.norm(s) + 1e-8)
    u2 = np.cross(s, f)

    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u2
    m[2, :3] = -f
    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = -eye
    return m @ t


def perspective(fov_y_deg: float, aspect: float, z_near: float = 0.01, z_far: float = 100.0) -> np.ndarray:
    """Create a perspective projection matrix"""
    f = 1.0 / math.tan(math.radians(fov_y_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = -f  # Flip Y to match image coordinates (top-left origin)
    m[2, 2] = (z_far + z_near) / (z_near - z_far)
    m[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    m[3, 2] = -1.0
    return m


def rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Create a 3x3 rotation matrix around axis by angle (degrees)"""
    angle = math.radians(angle_deg)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    x, y, z = axis
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c

    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ], dtype=np.float32)


# ============================================================================
# Geometry Generation
# ============================================================================

def make_rectangle_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 2x2 rectangle in XY plane centered at origin, facing +Z"""
    v = np.array([
        [-1.0, -1.0, 0.0],
        [ 1.0, -1.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [-1.0,  1.0, 0.0],
    ], dtype=np.float32)

    f = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    # Normals all point in +Z direction
    n = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    return v, f, n


def make_cube_mesh() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a unit cube mesh"""
    v = np.array(
        [
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )
    f = np.array(
        [
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ],
        dtype=np.int32,
    )
    n = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
    return v, f, n


def make_uv_sphere_mesh(radius: float, n_lat: int = 32, n_lon: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a UV sphere mesh"""
    verts = []
    norms = []
    for i in range(n_lat + 1):
        v = i / n_lat
        theta = v * math.pi
        y = math.cos(theta)
        r = math.sin(theta)
        for j in range(n_lon + 1):
            u = j / n_lon
            phi = u * 2.0 * math.pi
            x = r * math.cos(phi)
            z = r * math.sin(phi)
            norms.append([x, y, z])
            verts.append([radius * x, radius * y, radius * z])

    verts = np.array(verts, dtype=np.float32)
    norms = np.array(norms, dtype=np.float32)

    faces = []
    def idx(i: int, j: int) -> int:
        return i * (n_lon + 1) + j

    for i in range(n_lat):
        for j in range(n_lon):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i + 1, j + 1)
            d = idx(i, j + 1)
            if i != 0:
                faces.append([a, b, d])
            if i != n_lat - 1:
                faces.append([d, b, c])

    faces = np.array(faces, dtype=np.int32)
    return verts, faces, norms


# ============================================================================
# Mesh Processing
# ============================================================================

def transform_wall_mesh(wall: WallSpec) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate and transform a rectangle wall mesh according to WallSpec"""
    v, f, n = make_rectangle_mesh()

    # Apply rotation if any
    if np.linalg.norm(wall.rotate_axis) > 1e-6 and abs(wall.rotate_angle) > 1e-6:
        rot_mat = rotation_matrix(wall.rotate_axis, wall.rotate_angle)
        v = v @ rot_mat.T
        n = n @ rot_mat.T

    # Apply scale and translation
    v = v * wall.scale.reshape(1, 3) + wall.translate.reshape(1, 3)

    # Normalize normals
    n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-8)

    return v, f, n


def combine_meshes(mesh_list: list[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine multiple meshes into a single mesh"""
    if not mesh_list:
        raise ValueError("No meshes to combine")

    all_verts = []
    all_faces = []
    all_norms = []
    vertex_offset = 0

    for v, f, n in mesh_list:
        all_verts.append(v)
        all_faces.append(f + vertex_offset)
        all_norms.append(n)
        vertex_offset += len(v)

    combined_verts = np.vstack(all_verts)
    combined_faces = np.vstack(all_faces)
    combined_norms = np.vstack(all_norms)

    return combined_verts, combined_faces, combined_norms


def load_mesh_any(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load mesh from OBJ or PLY file"""
    try:
        import trimesh

        m = trimesh.load(path, force="mesh")
        if hasattr(m, "geometry"):
            geoms = list(m.geometry.values())
            if not geoms:
                raise ValueError(f"No geometry in {path}")
            m = geoms[0]
        v = np.asarray(m.vertices, dtype=np.float32)
        f = np.asarray(m.faces, dtype=np.int32)
        if m.vertex_normals is None or len(m.vertex_normals) != len(v):
            m.compute_vertex_normals()
        n = np.asarray(m.vertex_normals, dtype=np.float32)
        return v, f, n
    except Exception:
        pass

    if path.suffix.lower() == ".obj":
        vs = []
        fs = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    vs.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    parts = line.strip().split()[1:]
                    idxs = []
                    for p in parts:
                        idxs.append(int(p.split("/")[0]) - 1)
                    if len(idxs) == 3:
                        fs.append(idxs)
                    elif len(idxs) == 4:
                        fs.append([idxs[0], idxs[1], idxs[2]])
                        fs.append([idxs[0], idxs[2], idxs[3]])
        v = np.array(vs, dtype=np.float32)
        f = np.array(fs, dtype=np.int32)
        n = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        return v, f, n

    raise RuntimeError(f"Could not load mesh {path}. Install trimesh for support.")


# ============================================================================
# Image Saving
# ============================================================================

def save_image_u8(path: Path, img_linear: np.ndarray, gamma: float = 2.2):
    """Save image with gamma correction"""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.clip(img_linear, 0.0, 1.0) ** (1.0 / gamma)
    u8 = (img * 255.0 + 0.5).astype(np.uint8)

    try:
        import imageio.v3 as iio
        iio.imwrite(path, u8)
        return
    except Exception:
        pass

    try:
        from PIL import Image
        Image.fromarray(u8).save(path)
        return
    except Exception as e:
        raise RuntimeError(f"Failed to save image. Install imageio or pillow. Original error: {e}")


# ============================================================================
# Disney BRDF Functions (Simplified Version)
# ============================================================================

def schlick_weight(cos_theta):
    """Schlick's weight: (1 - cos_theta)^5"""
    import torch
    m = torch.clamp(1.0 - cos_theta, 0.0, 1.0)
    return m * m * m * m * m


def fresnel_schlick(cos_theta, F0):
    """Fresnel-Schlick approximation"""
    import torch
    return F0 + (1.0 - F0) * schlick_weight(cos_theta)


def geometric_smith_ggx(ndotl, ndotv, roughness, eps=1e-8):
    """Smith geometric shadowing-masking function with GGX distribution"""
    import torch

    alpha = roughness * roughness
    alpha2 = alpha * alpha

    # Lambda for light direction
    gl = ndotl * torch.sqrt(alpha2 + (1.0 - alpha2) * ndotv * ndotv)
    # Lambda for view direction
    gv = ndotv * torch.sqrt(alpha2 + (1.0 - alpha2) * ndotl * ndotl)

    return 0.5 / (gl + gv + eps)


def disney_brdf_simple(
    n, l, v, h,
    base_color, roughness, metallic, specular,
    eps=1e-8
):
    """Disney principled BRDF (simplified implementation without anisotropic).

    This is the simplified version used in mitsuba_soft_render.py

    Args:
        n: surface normal [B, H, W, 3]
        l: light direction [B, H, W, 3]
        v: view direction [B, H, W, 3]
        h: half vector [B, H, W, 3]
        base_color: base color [B, H, W, 3]
        roughness: roughness [B, H, W, 1]
        metallic: metallic [B, H, W, 1]
        specular: specular reflectance [B, H, W, 1]
        eps: small epsilon for numerical stability

    Returns:
        BRDF value [B, H, W, 3]
    """
    import torch

    ndotl = torch.sum(n * l, dim=-1, keepdim=True).clamp_min(0.0)
    ndotv = torch.sum(n * v, dim=-1, keepdim=True).clamp_min(eps)
    ndoth = torch.sum(n * h, dim=-1, keepdim=True).clamp_min(0.0)
    ldoth = torch.sum(l * h, dim=-1, keepdim=True).clamp_min(0.0)

    # --- Diffuse component (Disney diffuse) ---
    # Simplified Lambertian with roughness-based retro-reflection
    # fd90 = 0.5 + 2.0 * ldoth * ldoth * roughness
    # fl = schlick_weight(ndotl)
    # fv = schlick_weight(ndotv)
    #fd = torch.lerp(torch.ones_like(fd90), fd90, fl) * torch.lerp(torch.ones_like(fd90), fd90, fv)

    # Base diffuse
    diffuse = base_color / torch.pi

    # --- Specular component (GGX + Smith + Fresnel) ---
    # Normal distribution function (GGX/Trowbridge-Reitz)
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    denom = ndoth * ndoth * (alpha2 - 1.0) + 1.0
    D = alpha2 / (torch.pi * denom * denom + eps)

    # Geometric attenuation (Smith GGX)
    G = geometric_smith_ggx(ndotl, ndotv, roughness, eps)

    # Fresnel (Schlick approximation)
    # F0 = dielectric reflectance for metals, base_color for metals
    f0_dielectric = 0.08 * specular  # Default dielectric F0 scaled by specular param
    F0 = torch.lerp(
        f0_dielectric.expand_as(base_color),
        base_color,
        metallic
    )
    F = fresnel_schlick(ldoth, F0)

    # Cook-Torrance specular BRDF
    
    specular_brdf = (D * G * F) / (4.0 * ndotl * ndotv + eps)

    # --- Combine diffuse and specular ---
    # Metals have no diffuse component
    kd = (1.0 - F) * (1.0 - metallic)
    brdf = kd * diffuse + specular_brdf

    return brdf


# ============================================================================
# Environment Map Loading
# ============================================================================

def load_envmap(envmap_path: Path, working_directory: Path):
    """Load environment map from file, supporting multiple formats.

    Args:
        envmap_path: Path to the envmap file (can be relative)
        working_directory: Working directory to resolve relative paths

    Returns:
        numpy array of shape (H, W, 3) with float32 values, or None if loading fails
    """
    if not envmap_path.is_absolute():
        envmap_path = working_directory / envmap_path

    if not envmap_path.exists():
        print(f"  Warning: Envmap file not found: {envmap_path}")
        return None

    envmap_np = None
    file_ext = str(envmap_path).lower()

    # Method 1: Try OpenEXR (best for EXR files)
    if file_ext.endswith('.exr'):
        try:
            import OpenEXR
            import Imath

            exr_file = OpenEXR.InputFile(str(envmap_path))
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1

            # Read RGB channels
            FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
            channels = ['R', 'G', 'B']
            rgb = [np.frombuffer(exr_file.channel(c, FLOAT), dtype=np.float32) for c in channels]
            envmap_np = np.stack(rgb, axis=-1).reshape(height, width, 3)
            print(f"  Loaded EXR using OpenEXR library")
        except ImportError:
            print("  Note: OpenEXR library not found, trying imageio...")
        except Exception as e:
            print(f"  Warning: OpenEXR failed: {e}, trying imageio...")

    # Method 2: Try DDS format (DirectDraw Surface)
    if envmap_np is None and file_ext.endswith('.dds'):
        try:
            from PIL import Image
            from PIL import ImageFile
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            img = Image.open(envmap_path)
            envmap_np = np.array(img)
            print(f"  Loaded DDS using Pillow")
        except ImportError:
            print("  Note: Pillow not available for DDS, trying imageio...")
        except Exception as e:
            print(f"  Warning: Pillow DDS failed: {e}, trying imageio...")

        # Fallback: Try imageio for DDS
        if envmap_np is None:
            try:
                import imageio.v3 as iio
                envmap_np = iio.imread(envmap_path)
                print(f"  Loaded DDS using imageio")
            except Exception as e:
                print(f"  Warning: imageio DDS failed: {e}")

    # Method 3: Try imageio for other formats (PNG, JPG, HDR, etc.)
    if envmap_np is None:
        try:
            import imageio.v3 as iio
            envmap_np = iio.imread(envmap_path)
            print(f"  Loaded envmap using imageio")
        except Exception as e:
            print(f"  Warning: imageio failed: {e}")

    # Process loaded image
    if envmap_np is not None:
        if envmap_np.dtype == np.uint8:
            envmap_np = envmap_np.astype(np.float32) / 255.0
        else:
            envmap_np = envmap_np.astype(np.float32)

        # Ensure 3 channels
        if envmap_np.ndim == 2:
            envmap_np = np.stack([envmap_np] * 3, axis=-1)
        elif envmap_np.shape[-1] == 4:
            envmap_np = envmap_np[..., :3]

        return envmap_np

    return None


# ============================================================================
# Low-discrepancy Sampling (for IBL)
# ============================================================================

def radical_inverse_vdc(bits):
    """Van der Corput sequence for low-discrepancy sampling"""
    bits = (bits << 16) | (bits >> 16)
    bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
    bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
    bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
    bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
    return float(bits) * 2.3283064365386963e-10


def hammersley(i, N, device):
    """Generate 2D Hammersley point"""
    import torch
    return torch.tensor([float(i) / float(N), radical_inverse_vdc(i)], dtype=torch.float32, device=device)


# ============================================================================
# Disney BRDF (Full Version with Anisotropic)
# ============================================================================

def fresnel_schlick_scalar(f0, u):
    """Scalar Schlick Fresnel matching:
    F_Schlick(f0, u) = (1 - f0) * (1 - u)^5 + f0

    Args:
        f0: scalar reflectance at normal incidence [..., 1]
        u: cosine term (typically L·H or V·H) [..., 1]
    """
    import torch
    f0 = torch.clamp(f0, 0.0, 1.0)
    u = torch.clamp(u, 0.0, 1.0)
    x = 1.0 - u
    x2 = x * x
    x5 = x * x2 * x2
    return (1.0 - f0) * x5 + f0


def fresnel_schlick_ue(specular_color, voh):
    """Unreal Engine modified Schlick Fresnel.

    Fc = Pow5(1 - VoH)
    F = saturate(50 * SpecularColor.g) * Fc + (1 - Fc) * SpecularColor

    Args:
        specular_color: RGB F0 [..., 3]
        voh: cosine term V·H [..., 1]
    """
    import torch

    voh = torch.clamp(voh, 0.0, 1.0)
    fc = schlick_weight(voh)
    damp = torch.clamp(50.0 * specular_color[..., 1:2], 0.0, 1.0)
    return damp * fc + (1.0 - fc) * specular_color


def fresnel_dielectric(cos_theta_i, eta, eps=1e-8):
    """Exact unpolarized Fresnel for dielectrics.

    This is used to avoid the Schlick edge case where F0=0 still yields
    strong grazing reflections. With eta=1 (air), Fresnel is 0 for all angles.
    """
    import torch

    cos_theta_i = torch.clamp(cos_theta_i, 0.0, 1.0)
    eta = eta.clamp_min(1.0)

    sin2_theta_i = torch.clamp(1.0 - cos_theta_i * cos_theta_i, 0.0, 1.0)
    sin2_theta_t = sin2_theta_i / (eta * eta + eps)

    tir = sin2_theta_t > 1.0
    cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin2_theta_t, 0.0, 1.0))

    r_parl = ((eta * cos_theta_i) - cos_theta_t) / ((eta * cos_theta_i) + cos_theta_t + eps)
    r_perp = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t + eps)
    F = 0.5 * (r_parl * r_parl + r_perp * r_perp)

    return torch.where(tir, torch.ones_like(F), F)


def disney_brdf_full(
    n, l, v, h,
    base_color, roughness, metallic, specular,
    specular_tint, anisotropic,
    eps=1e-8
):
    """Disney principled BRDF (full implementation with anisotropic).

    Based on Burley 2012 "Physically-Based Shading at Disney"
    This is the full version used in mitsuba_raster_fit_disney.py

    Args:
        n: surface normal [B, H, W, 3]
        l: light direction [B, H, W, 3]
        v: view direction [B, H, W, 3]
        h: half vector [B, H, W, 3]
        base_color: base color [B, H, W, 3]
        roughness: roughness [B, H, W, 1]
        metallic: metallic [B, H, W, 1]
        specular: specular reflectance [B, H, W, 1]
        specular_tint: specular tint [B, H, W, 1]
        anisotropic: anisotropic [B, H, W, 1]
        eps: small epsilon for numerical stability

    Returns:
        BRDF value [B, H, W, 3]
    """
    import torch

    ndotl = torch.sum(n * l, dim=-1, keepdim=True).clamp_min(0.0)
    ndotv = torch.sum(n * v, dim=-1, keepdim=True).clamp_min(eps)
    ndoth = torch.sum(n * h, dim=-1, keepdim=True).clamp_min(0.0)
    ldoth = torch.sum(l * h, dim=-1, keepdim=True).clamp_min(0.0)
    vdoth = torch.sum(v * h, dim=-1, keepdim=True).clamp_min(0.0)

    # --- Diffuse component (Disney diffuse) ---
    # Simplified Lambertian with roughness-based retro-reflection
    fd90 = 0.5 + 2.0 * ldoth * ldoth * roughness
    fl = schlick_weight(ndotl)
    fv = schlick_weight(ndotv)
    #fd = torch.lerp(torch.ones_like(fd90), fd90, fl) * torch.lerp(torch.ones_like(fd90), fd90, fv)

    # Base diffuse
    diffuse = base_color / torch.pi

    # --- Specular component (GGX + Smith + Fresnel) ---
    # Disney-like specular tint: tint the dielectric specular color towards base_color hue.
    # Avoid division blow-ups for very dark base_color.
    lum = (0.3 * base_color[..., 0:1] + 0.6 * base_color[..., 1:2] + 0.1 * base_color[..., 2:3])
    tint_color = torch.where(
        lum > 1e-4,
        base_color / lum.clamp_min(1e-4),
        torch.ones_like(base_color),
    )
    tint_color = torch.clamp(tint_color, 0.0, 1.0)
    spec_color = torch.lerp(torch.ones_like(base_color), tint_color, specular_tint)

    # Disney anisotropic GGX parameters
    # (matches the common Disney mapping; anisotropic=0 => isotropic)
    a = (roughness * roughness).clamp_min(1e-4)
    aspect = torch.sqrt((1.0 - 0.9 * anisotropic).clamp_min(1e-4))
    alpha_x = (a / aspect).clamp_min(1e-4)
    alpha_y = (a * aspect).clamp_min(1e-4)

    # Build a stable tangent frame from the normal (no UVs available)
    up = torch.where(
        (torch.abs(n[..., 2:3]) < 0.999).expand_as(n),
        torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype).view(1, 1, 1, 3).expand_as(n),
        torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype).view(1, 1, 1, 3).expand_as(n),
    )
    t = torch.nn.functional.normalize(torch.cross(up, n, dim=-1), dim=-1)
    b = torch.cross(n, t, dim=-1)

    def to_tangent(w):
        wx = torch.sum(w * t, dim=-1, keepdim=True)
        wy = torch.sum(w * b, dim=-1, keepdim=True)
        wz = torch.sum(w * n, dim=-1, keepdim=True)
        return wx, wy, wz

    hx, hy, hz = to_tangent(h)

    # Anisotropic GGX NDF (Heitz)
    denom = (hx * hx) / (alpha_x * alpha_x) + (hy * hy) / (alpha_y * alpha_y) + (hz * hz)
    D = 1.0 / (torch.pi * alpha_x * alpha_y * denom * denom + eps)

    # Anisotropic Smith masking-shadowing (approx)
    def G1_aniso(w):
        wx, wy, wz = to_tangent(w)
        wz = wz.clamp_min(eps)
        tan2 = ((wx * wx) * (alpha_x * alpha_x) + (wy * wy) * (alpha_y * alpha_y)) / (wz * wz)
        lam = 0.5 * (torch.sqrt(1.0 + tan2) - 1.0)
        return 1.0 / (1.0 + lam)

    G = G1_aniso(l) * G1_aniso(v)

    # Fresnel
    # Unreal Engine modified Schlick using VoH and a damped grazing term.
    # Disney's 'specular' parameter maps to F0 via 0.08 * specular.
    f0_scalar = (0.08 * specular).clamp(0.0, 1.0)
    F0_diel_rgb = f0_scalar * spec_color
    F_diel_rgb = fresnel_schlick_ue(F0_diel_rgb, vdoth)

    # Metals: Schlick with base_color as RGB F0
    F_metal = fresnel_schlick(vdoth, base_color)

    F = torch.lerp(F_diel_rgb, F_metal, metallic)

    # Cook-Torrance specular BRDF
    specular_brdf = (D * G * F) / (4.0 * ndotl * ndotv + eps)

    # --- Combine diffuse and specular ---
    # Metals have no diffuse component
    kd = (1.0 - F) * (1.0 - metallic)
    brdf = kd * diffuse + specular_brdf

    return brdf
