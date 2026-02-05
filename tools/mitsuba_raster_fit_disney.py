#!/usr/bin/env python3
"""Disney BRDF material fitting with nvdiffrast.

This script fits a Disney principled BRDF to Mitsuba path-traced ground truth.
It optimizes multiple PBR parameters including:
- Base color (albedo)
- Roughness
- Metallic
- Specular reflectance

Features:
- Disney BRDF implementation (based on Burley 2012)
- Multi-light support
- Environment map support (optional)
- Real-time progress logging for GUI integration

Example:
  python tools/mitsuba_raster_fit_disney.py \
    --scene scenes/cbox.xml \
    --gt-variant scalar_rgb \
    --gt-spp 256 \
    --steps 400 \
    --lr 0.01 \
    --out-dir renders/fit_disney
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def _require(pkg: str, import_name: Optional[str] = None):
    try:
        return __import__(import_name or pkg)
    except Exception as e:
        raise RuntimeError(
            f"Missing dependency '{pkg}'. Install it in your training env. Original error: {e}"
        )


def _parse_vec3(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.replace(",", " ").split() if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"Expected vec3, got: {s!r}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)


def _xml_find_first(elem, path: str):
    x = elem.find(path)
    if x is None:
        raise ValueError(f"XML missing required element: {path}")
    return x


@dataclass
class Camera:
    origin: np.ndarray
    target: np.ndarray
    up: np.ndarray
    fov_deg: float
    width: int
    height: int


@dataclass
class ObjectSpec:
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


def parse_scene_xml(scene_path: Path) -> Tuple[Camera, ObjectSpec, list[LightSpec], list[WallSpec]]:
    import xml.etree.ElementTree as ET

    root = ET.parse(scene_path).getroot()

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

    # Object parsing (same as before)
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


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
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


def save_image_u8(path: Path, img_linear: np.ndarray, gamma: float = 2.2):
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


def disney_brdf(
    n, l, v, h,
    base_color, roughness, metallic, specular,
    specular_tint, anisotropic,
    eps=1e-8
):
    """Disney principled BRDF (simplified implementation).

    Based on Burley 2012 "Physically-Based Shading at Disney"
    This is a simplified version focusing on the most important terms.

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
    fd = torch.lerp(torch.ones_like(fd90), fd90, fl) * torch.lerp(torch.ones_like(fd90), fd90, fv)

    # Base diffuse
    diffuse = base_color * fd / torch.pi

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


def schlick_weight(cos_theta):
    """Schlick's weight: (1 - cos_theta)^5"""
    import torch
    m = torch.clamp(1.0 - cos_theta, 0.0, 1.0)
    return m * m * m * m * m


def fresnel_schlick(cos_theta, F0):
    """Fresnel-Schlick approximation"""
    import torch
    return F0 + (1.0 - F0) * schlick_weight(cos_theta)

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Mitsuba XML scene path")
    ap.add_argument("--gt-image", type=str, help="Ground truth image path (if provided, skip rendering)")
    ap.add_argument("--gt-variant", default="scalar_rgb", help="Mitsuba variant for ground truth")
    ap.add_argument("--gt-spp", type=int, default=256, help="SPP for ground truth (only if --gt-image not provided)")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out-dir", default="renders/fit_disney", help="Output directory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Torch device")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", type=str, help="JSON config file path (.mitsuba_studio_state.json)")
    ap.add_argument("--ibl-samples", type=int, default=128, help="Number of samples for IBL importance sampling")

    # Material parameter initialization
    ap.add_argument("--init-base-color", type=str, default="0.8,0.8,0.8", help="Initial base color RGB")
    ap.add_argument("--init-roughness", type=float, default=0.5, help="Initial roughness")
    ap.add_argument("--init-metallic", type=float, default=0.0, help="Initial metallic")
    ap.add_argument("--init-specular", type=float, default=0.5, help="Initial specular")
    ap.add_argument("--init-specular-tint", type=float, default=0.0, help="Initial specular tint")
    ap.add_argument("--init-anisotropic", type=float, default=0.0, help="Initial anisotropic")
    ap.add_argument("--init-sheen", type=float, default=0.0, help="Initial sheen")
    ap.add_argument("--init-sheen-tint", type=float, default=0.5, help="Initial sheen tint")
    ap.add_argument("--init-clearcoat", type=float, default=0.0, help="Initial clearcoat")
    ap.add_argument("--init-clearcoat-gloss", type=float, default=1.0, help="Initial clearcoat gloss")

    args = ap.parse_args()

    ws = Path.cwd()
    scene_path = Path(args.scene)
    if not scene_path.is_absolute():
        scene_path = ws / scene_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ws / out_dir

    # Load Disney material config from JSON if provided
    disney_material_config = None
    if args.config:
        import json
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ws / config_path

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
                if 'config' in full_config and 'disney_material' in full_config['config']:
                    disney_material_config = full_config['config']['disney_material']
                    print(f"Loaded Disney material config from {config_path}")
        else:
            print(f"Warning: Config file not found: {config_path}")

    # Mitsuba GT
    mi = _require("mitsuba")
    dr = _require("drjit")
    mi.set_variant(args.gt_variant)

    scene = mi.load_file(str(scene_path))
    sensor = scene.sensors()[0]
    film = sensor.film()
    size = film.size()
    width, height = int(size[0]), int(size[1])

    try:
        mi.set_log_level(mi.LogLevel.Warn)
    except Exception:
        pass

    # Load or render ground truth
    if args.gt_image:
        gt_path = Path(args.gt_image)
        if not gt_path.is_absolute():
            gt_path = ws / gt_path
        print(f"Loading ground truth from: {gt_path}")
        from PIL import Image
        img_gt_pil = Image.open(gt_path).convert("RGB")
        img_gt_np = np.array(img_gt_pil, dtype=np.float32) / 255.0
        if img_gt_np.shape[:2] != (height, width):
            print(f"Warning: GT image size {img_gt_np.shape[:2]} != film size ({height}, {width})")
    else:
        print(f"Rendering ground truth with {args.gt_spp} spp...")
        img_gt = mi.render(scene, spp=args.gt_spp)
        img_gt_np = np.array(img_gt, dtype=np.float32).reshape(height, width, 3)

    # Torch + nvdiffrast
    torch = _require("torch")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available. Run this on a CUDA machine or pass --device cpu.")

    try:
        import nvdiffrast.torch as drt
    except Exception as e:
        raise RuntimeError(
            f"nvdiffrast not available. Install it in a CUDA environment. Original error: {e}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cam, obj, lights, walls = parse_scene_xml(scene_path)

    # Override light config from JSON if provided
    if args.config:
        import json
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = ws / config_path
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
                if 'config' in full_config and 'light' in full_config['config']:
                    light_config = full_config['config']['light']
                    # Update envmap rotation if light is envmap type
                    for light in lights:
                        if light.light_type == 'envmap':
                            if 'envmap_rotation' in light_config:
                                light.envmap_rotation = light_config['envmap_rotation']
                                print(f"Loaded envmap rotation from config: {light.envmap_rotation} degrees")

    print(f"Parsed {len(lights)} light(s) and {len(walls)} wall(s) from scene")

    # --- Material parameter initialization (optionally from .mitsuba_studio_state.json) ---
    if disney_material_config is not None:
        init_base_color = np.array(disney_material_config['base_color']['value'], dtype=np.float32)
        init_roughness = float(disney_material_config['roughness']['value'])
        init_metallic = float(disney_material_config['metallic']['value'])
        init_specular = float(disney_material_config['specular']['value'])
        init_specular_tint = float(disney_material_config['specular_tint']['value'])
        init_anisotropic = float(disney_material_config['anisotropic']['value'])
        init_sheen = float(disney_material_config['sheen']['value'])
        init_sheen_tint = float(disney_material_config['sheen_tint']['value'])
        init_clearcoat = float(disney_material_config['clearcoat']['value'])
        init_clearcoat_gloss = float(disney_material_config['clearcoat_gloss']['value'])

        diff_base_color = bool(disney_material_config['base_color']['differentiable'])
        diff_roughness = bool(disney_material_config['roughness']['differentiable'])
        diff_metallic = bool(disney_material_config['metallic']['differentiable'])
        diff_specular = bool(disney_material_config['specular']['differentiable'])
        diff_specular_tint = bool(disney_material_config['specular_tint']['differentiable'])
        diff_anisotropic = bool(disney_material_config['anisotropic']['differentiable'])
        diff_sheen = bool(disney_material_config['sheen']['differentiable'])
        diff_sheen_tint = bool(disney_material_config['sheen_tint']['differentiable'])
        diff_clearcoat = bool(disney_material_config['clearcoat']['differentiable'])
        diff_clearcoat_gloss = bool(disney_material_config['clearcoat_gloss']['differentiable'])

        print("Using Disney material config:")
        print(f"  base_color: {init_base_color.tolist()} (diff={diff_base_color})")
        print(f"  roughness: {init_roughness} (diff={diff_roughness})")
        print(f"  metallic: {init_metallic} (diff={diff_metallic})")
        print(f"  specular: {init_specular} (diff={diff_specular})")
        print(f"  specular_tint: {init_specular_tint} (diff={diff_specular_tint})")
        print(f"  anisotropic: {init_anisotropic} (diff={diff_anisotropic})")
        print(f"  sheen: {init_sheen} (diff={diff_sheen})")
        print(f"  sheen_tint: {init_sheen_tint} (diff={diff_sheen_tint})")
        print(f"  clearcoat: {init_clearcoat} (diff={diff_clearcoat})")
        print(f"  clearcoat_gloss: {init_clearcoat_gloss} (diff={diff_clearcoat_gloss})")
    else:
        init_base_color = np.array([float(x) for x in args.init_base_color.split(",")], dtype=np.float32)
        if init_base_color.shape[0] != 3:
            raise ValueError("--init-base-color must have 3 values")
        init_roughness = args.init_roughness
        init_metallic = args.init_metallic
        init_specular = args.init_specular
        init_specular_tint = args.init_specular_tint
        init_anisotropic = args.init_anisotropic
        init_sheen = args.init_sheen
        init_sheen_tint = args.init_sheen_tint
        init_clearcoat = args.init_clearcoat
        init_clearcoat_gloss = args.init_clearcoat_gloss

        # Default: optimize base_color and roughness only
        diff_base_color = True
        diff_roughness = True
        diff_metallic = False
        diff_specular = False
        diff_specular_tint = False
        diff_anisotropic = False
        diff_sheen = False
        diff_sheen_tint = False
        diff_clearcoat = False
        diff_clearcoat_gloss = False

    # Load/build object mesh
    if obj.kind == "sphere":
        v_obj, f_obj, n_obj = make_uv_sphere_mesh(radius=obj.radius)
    elif obj.kind == "cube":
        v_obj, f_obj, n_obj = make_cube_mesh()
    else:
        mesh_path = Path(obj.filename)
        if not mesh_path.is_absolute():
            mesh_path = scene_path.parent / mesh_path
        v_obj, f_obj, n_obj = load_mesh_any(mesh_path)

    # Transform object mesh
    v_obj = v_obj * obj.scale.reshape(1, 3) + obj.translate.reshape(1, 3)

    # Generate wall meshes and track vertex colors
    meshes_to_combine = [(v_obj, f_obj, n_obj)]
    vertex_colors = []  # Store base color per vertex

    # Object vertices will use optimized material (mark with special color for now)
    # We'll use -1 to indicate "use optimized color"
    obj_colors = np.full((len(v_obj), 3), -1.0, dtype=np.float32)
    vertex_colors.append(obj_colors)

    for wall in walls:
        v_wall, f_wall, n_wall = transform_wall_mesh(wall)
        meshes_to_combine.append((v_wall, f_wall, n_wall))

        # Wall vertices use fixed reflectance colors
        wall_colors = np.tile(wall.reflectance.reshape(1, 3), (len(v_wall), 1))
        vertex_colors.append(wall_colors)

    # Combine all meshes into single scene mesh
    v, f, n = combine_meshes(meshes_to_combine)
    vertex_base_colors = np.vstack(vertex_colors)

    device = torch.device(args.device)

    v_t = torch.from_numpy(v).to(device=device, dtype=torch.float32)
    n_t = torch.from_numpy(n).to(device=device, dtype=torch.float32)
    f_t = torch.from_numpy(f).to(device=device, dtype=torch.int32)
    vertex_base_colors_t = torch.from_numpy(vertex_base_colors).to(device=device, dtype=torch.float32)

    # Camera matrices
    aspect = cam.width / max(1.0, float(cam.height))
    view = look_at(cam.origin, cam.target, cam.up)
    proj = perspective(cam.fov_deg, aspect)
    mvp = (proj @ view).astype(np.float32)

    mvp_t = torch.from_numpy(mvp).to(device=device, dtype=torch.float32)

    # Homogeneous coordinates
    ones = torch.ones((v_t.shape[0], 1), device=device, dtype=torch.float32)
    v_h = torch.cat([v_t, ones], dim=1)
    v_clip = (v_h @ mvp_t.t())

    # Helper function to safely compute logit transform
    def safe_logit(value, eps=1e-4):
        """Compute logit while clamping input to safe range [eps, 1-eps]"""
        clamped = max(eps, min(1.0 - eps, value))
        return math.log(clamped / (1.0 - clamped))

    # Material parameters (logit parameterization for bounded values)
    # Base color
    init_base_color_safe = np.clip(init_base_color, 1e-6, 1.0 - 1e-6)
    base_color_logit = torch.from_numpy(
        np.log(init_base_color_safe / (1.0 - init_base_color_safe))
    ).to(device=device, dtype=torch.float32)
    base_color_logit.requires_grad = diff_base_color

    # Roughness
    roughness_logit = torch.tensor(
        [safe_logit(init_roughness)],
        device=device, dtype=torch.float32, requires_grad=diff_roughness
    )

    # Metallic
    metallic_logit = torch.tensor(
        [safe_logit(init_metallic)],
        device=device, dtype=torch.float32, requires_grad=diff_metallic
    )

    # Specular
    specular_logit = torch.tensor(
        [safe_logit(init_specular)],
        device=device, dtype=torch.float32, requires_grad=diff_specular
    )

    # Specular Tint
    specular_tint_logit = torch.tensor(
        [safe_logit(init_specular_tint)],
        device=device, dtype=torch.float32, requires_grad=diff_specular_tint
    )

    # Anisotropic
    anisotropic_logit = torch.tensor(
        [safe_logit(init_anisotropic)],
        device=device, dtype=torch.float32, requires_grad=diff_anisotropic
    )

    # Sheen
    sheen_logit = torch.tensor(
        [safe_logit(init_sheen)],
        device=device, dtype=torch.float32, requires_grad=diff_sheen
    )

    # Sheen Tint
    sheen_tint_logit = torch.tensor(
        [safe_logit(init_sheen_tint)],
        device=device, dtype=torch.float32, requires_grad=diff_sheen_tint
    )

    # Clearcoat
    clearcoat_logit = torch.tensor(
        [safe_logit(init_clearcoat)],
        device=device, dtype=torch.float32, requires_grad=diff_clearcoat
    )

    # Clearcoat Gloss
    clearcoat_gloss_logit = torch.tensor(
        [safe_logit(init_clearcoat_gloss)],
        device=device, dtype=torch.float32, requires_grad=diff_clearcoat_gloss
    )

    # Collect parameters that need optimization
    opt_params = []
    if diff_base_color:
        opt_params.append(base_color_logit)
    if diff_roughness:
        opt_params.append(roughness_logit)
    if diff_metallic:
        opt_params.append(metallic_logit)
    if diff_specular:
        opt_params.append(specular_logit)
    if diff_specular_tint:
        opt_params.append(specular_tint_logit)
    if diff_anisotropic:
        opt_params.append(anisotropic_logit)
    if diff_sheen:
        opt_params.append(sheen_logit)
    if diff_sheen_tint:
        opt_params.append(sheen_tint_logit)
    if diff_clearcoat:
        opt_params.append(clearcoat_logit)
    if diff_clearcoat_gloss:
        opt_params.append(clearcoat_gloss_logit)

    if not opt_params:
        raise ValueError("No parameters marked as differentiable. Enable at least one parameter for optimization.")

    opt = torch.optim.Adam(opt_params, lr=args.lr)

    # GT to torch
    gt = torch.from_numpy(img_gt_np).to(device=device, dtype=torch.float32)

    # Camera position for view vector
    cam_pos_t = torch.from_numpy(cam.origin).to(device=device, dtype=torch.float32)

    # Prepare light sources for rendering
    point_light_positions = []
    point_light_radiances = []

    spot_light_positions = []
    spot_light_directions = []
    spot_light_radiances = []
    spot_light_cutoffs = []
    spot_light_beam_widths = []

    directional_light_directions = []
    directional_light_irradiances = []

    envmap_data = None
    envmap_scale_val = 1.0
    envmap_rotation_val = 0.0  # Rotation in degrees

    for light in lights:
        if light.light_type == 'area':
            # Sample area light with a grid to simulate soft shadows
            num_samples = 5
            light_size = light.scale
            base_pos = light.position

            for i in range(num_samples):
                for j in range(num_samples):
                    offset_x = (i / (num_samples - 1) - 0.5) * light_size
                    offset_z = (j / (num_samples - 1) - 0.5) * light_size
                    sample_pos = base_pos + np.array([offset_x, 0.0, offset_z], dtype=np.float32)
                    point_light_positions.append(torch.from_numpy(sample_pos).to(device=device, dtype=torch.float32))
                    # Divide radiance by number of samples
                    point_light_radiances.append(torch.from_numpy(light.radiance / (num_samples * num_samples)).to(device=device, dtype=torch.float32))

        elif light.light_type == 'point':
            point_light_positions.append(torch.from_numpy(light.point_position).to(device=device, dtype=torch.float32))
            point_light_radiances.append(torch.from_numpy(light.radiance).to(device=device, dtype=torch.float32))

        elif light.light_type == 'spot':
            spot_light_positions.append(torch.from_numpy(light.spot_position).to(device=device, dtype=torch.float32))
            spot_light_directions.append(torch.from_numpy(light.spot_direction).to(device=device, dtype=torch.float32))
            spot_light_radiances.append(torch.from_numpy(light.radiance).to(device=device, dtype=torch.float32))
            spot_light_cutoffs.append(light.spot_cutoff_angle)
            spot_light_beam_widths.append(light.spot_beam_width)

        elif light.light_type == 'directional':
            directional_light_directions.append(torch.from_numpy(light.directional_direction).to(device=device, dtype=torch.float32))
            directional_light_irradiances.append(torch.from_numpy(light.radiance).to(device=device, dtype=torch.float32))

        elif light.light_type == 'envmap':
            # Load environment map
            envmap_path = Path(light.envmap_filename)
            if not envmap_path.is_absolute():
                # Resolve relative to working directory, not scene file directory
                envmap_path = ws / envmap_path

            if envmap_path.exists():
                envmap_np = None
                file_ext = str(envmap_path).lower()

                # Try multiple methods to load different formats
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
                        print(f"[OK] Loaded EXR using OpenEXR library")
                    except ImportError:
                        print("Note: OpenEXR library not found, trying imageio...")
                    except Exception as e:
                        print(f"Warning: OpenEXR failed: {e}, trying imageio...")

                # Method 2: Try DDS format (DirectDraw Surface)
                if envmap_np is None and file_ext.endswith('.dds'):
                    try:
                        # Try using Pillow (supports DDS with plugin)
                        from PIL import Image
                        from PIL import ImageFile
                        ImageFile.LOAD_TRUNCATED_IMAGES = True

                        img = Image.open(envmap_path)
                        envmap_np = np.array(img)
                        print(f"[OK] Loaded DDS using Pillow")
                    except ImportError:
                        print("Note: Pillow not available for DDS, trying imageio...")
                    except Exception as e:
                        print(f"Warning: Pillow DDS failed: {e}, trying imageio...")

                    # Fallback: Try imageio for DDS
                    if envmap_np is None:
                        try:
                            import imageio.v3 as iio
                            envmap_np = iio.imread(envmap_path)
                            print(f"[OK] Loaded DDS using imageio")
                        except Exception as e:
                            print(f"Warning: imageio DDS failed: {e}")

                # Method 3: Try imageio for other formats (PNG, JPG, HDR, etc.)
                if envmap_np is None:
                    try:
                        import imageio.v3 as iio
                        envmap_np = iio.imread(envmap_path)
                        print(f"[OK] Loaded using imageio")
                    except Exception as e:
                        print(f"Warning: imageio failed: {e}")
                        if file_ext.endswith('.exr'):
                            print("\nTo load EXR files, install one of:")
                            print("  pip install OpenEXR")
                            print("  pip install imageio[pyav]")
                            print("  pip install imageio[opencv]")
                        elif file_ext.endswith('.dds'):
                            print("\nTo load DDS files, install:")
                            print("  pip install Pillow")
                            print("  or: pip install imageio[pyav]")

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

                    envmap_data = torch.from_numpy(envmap_np).to(device=device, dtype=torch.float32)
                    envmap_scale_val = light.envmap_scale
                    envmap_rotation_val = light.envmap_rotation
                    print(f"  Path: {envmap_path}")
                    print(f"  Shape: {envmap_data.shape}")
                    print(f"  Value range: [{envmap_data.min().item():.4f}, {envmap_data.max().item():.4f}]")
                    print(f"  Rotation: {envmap_rotation_val} degrees")
                    print(f"  Mean: {envmap_data.mean().item():.4f}")
                    print(f"  Scale: {envmap_scale_val}")
                else:
                    print("Using constant ambient fallback")
            else:
                print(f"Warning: Envmap file not found: {envmap_path}")
                print("Using constant ambient fallback")

    # Raster context
    ctx = drt.RasterizeCudaContext()

    # Constant ambient fallback (used when no envmap lighting is available)
    ambient_color = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)

    # ===== GGX Importance Sampling Functions =====
    def radical_inverse_vdc(bits):
        """Van der Corput sequence for low-discrepancy sampling"""
        bits = (bits << 16) | (bits >> 16)
        bits = ((bits & 0x55555555) << 1) | ((bits & 0xAAAAAAAA) >> 1)
        bits = ((bits & 0x33333333) << 2) | ((bits & 0xCCCCCCCC) >> 2)
        bits = ((bits & 0x0F0F0F0F) << 4) | ((bits & 0xF0F0F0F0) >> 4)
        bits = ((bits & 0x00FF00FF) << 8) | ((bits & 0xFF00FF00) >> 8)
        return float(bits) * 2.3283064365386963e-10

    def hammersley(i, N):
        """Generate 2D Hammersley point"""
        return torch.tensor([float(i) / float(N), radical_inverse_vdc(i)], dtype=torch.float32, device=device)

    def importance_sample_ggx(xi, N, roughness):
        """
        Generate sample direction based on GGX distribution
        xi: 2D random sample [0,1]^2
        N: normal vector
        roughness: material roughness [0,1]
        Returns: sample direction in world space
        """
        a = roughness * roughness

        phi = 2.0 * math.pi * xi[0]
        cos_theta = torch.sqrt((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1]))
        sin_theta = torch.sqrt(1.0 - cos_theta * cos_theta)

        # Spherical to Cartesian (tangent space)
        H = torch.stack([
            torch.cos(phi) * sin_theta,
            torch.sin(phi) * sin_theta,
            cos_theta
        ])

        # Tangent space to world space
        up = torch.where(
            torch.abs(N[2]) < 0.999,
            torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32),
            torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float32)
        )
        tangent = torch.nn.functional.normalize(torch.cross(up, N, dim=0), dim=0)
        bitangent = torch.cross(N, tangent, dim=0)

        sample_vec = tangent * H[0] + bitangent * H[1] + N * H[2]
        return torch.nn.functional.normalize(sample_vec, dim=0)

    def render_raster(base_color, roughness, metallic, specular,
                      specular_tint, anisotropic, sheen, sheen_tint,
                      clearcoat, clearcoat_gloss):
        rast, _ = drt.rasterize(ctx, v_clip[None, ...], f_t, (height, width))
        mask = rast[..., 3:4] > 0

        # Interpolate world position, normals, and vertex colors
        pos, _ = drt.interpolate(v_t[None, ...], rast, f_t)
        nor, _ = drt.interpolate(n_t[None, ...], rast, f_t)
        nor = torch.nn.functional.normalize(nor, dim=-1)

        # Interpolate per-vertex base colors
        vertex_colors, _ = drt.interpolate(vertex_base_colors_t[None, ...], rast, f_t)

        # For object vertices (marked with negative values), use optimized base_color
        # For wall vertices, use their fixed colors
        is_object = vertex_colors[..., 0:1] < 0
        per_pixel_base_color = torch.where(is_object, base_color[None, None, None, :], vertex_colors)

        # For walls, use simple diffuse material (metallic=0, high roughness)
        # For object, use optimized material
        per_pixel_roughness = torch.where(is_object, roughness[None, None, None, :], torch.ones_like(roughness[None, None, None, :]))
        per_pixel_metallic = torch.where(is_object, metallic[None, None, None, :], torch.zeros_like(metallic[None, None, None, :]))
        per_pixel_specular = torch.where(is_object, specular[None, None, None, :], torch.ones_like(specular[None, None, None, :]) * 0.5)
        per_pixel_specular_tint = torch.where(is_object, specular_tint[None, None, None, :], torch.zeros_like(specular_tint[None, None, None, :]))
        per_pixel_anisotropic = torch.where(is_object, anisotropic[None, None, None, :], torch.zeros_like(anisotropic[None, None, None, :]))

        # View direction
        view_dir = torch.nn.functional.normalize(cam_pos_t[None, None, None, :] - pos, dim=-1)

        # Accumulate lighting from all light sources
        col = torch.zeros_like(pos)

        # Point lights (includes sampled area lights)
        for light_pos, light_rad in zip(point_light_positions, point_light_radiances):
            l_vec = light_pos[None, None, None, :] - pos
            dist2 = torch.sum(l_vec * l_vec, dim=-1, keepdim=True).clamp_min(1e-4)
            l_dir = l_vec / torch.sqrt(dist2)
            h = torch.nn.functional.normalize(l_dir + view_dir, dim=-1)

            brdf = disney_brdf(
                nor, l_dir, view_dir, h,
                per_pixel_base_color,
                per_pixel_roughness,
                per_pixel_metallic,
                per_pixel_specular,
                per_pixel_specular_tint,
                per_pixel_anisotropic,
            )

            ndotl = torch.sum(nor * l_dir, dim=-1, keepdim=True).clamp_min(0.0)
            light_contrib = brdf * ndotl * (light_rad[None, None, None, :] / dist2)
            col = col + light_contrib

        # Spot lights
        for spot_pos, spot_dir, spot_rad, spot_cutoff, spot_beam in zip(
            spot_light_positions, spot_light_directions, spot_light_radiances,
            spot_light_cutoffs, spot_light_beam_widths
        ):
            l_vec = spot_pos[None, None, None, :] - pos
            dist2 = torch.sum(l_vec * l_vec, dim=-1, keepdim=True).clamp_min(1e-4)
            l_dir = l_vec / torch.sqrt(dist2)

            # Compute spotlight cone falloff
            spot_dir_normalized = spot_dir[None, None, None, :]
            cos_angle = torch.sum(-l_dir * spot_dir_normalized, dim=-1, keepdim=True)
            cos_cutoff = math.cos(math.radians(spot_cutoff))
            cos_beam = math.cos(math.radians(spot_beam))

            # Smooth falloff between beam width and cutoff angle
            falloff = torch.clamp((cos_angle - cos_cutoff) / (cos_beam - cos_cutoff + 1e-6), 0.0, 1.0)
            falloff = falloff * falloff  # Quadratic falloff

            h = torch.nn.functional.normalize(l_dir + view_dir, dim=-1)
            brdf = disney_brdf(
                nor, l_dir, view_dir, h,
                per_pixel_base_color,
                per_pixel_roughness,
                per_pixel_metallic,
                per_pixel_specular,
                per_pixel_specular_tint,
                per_pixel_anisotropic,
            )

            ndotl = torch.sum(nor * l_dir, dim=-1, keepdim=True).clamp_min(0.0)
            light_contrib = brdf * ndotl * (spot_rad[None, None, None, :] / dist2) * falloff
            col = col + light_contrib

        # Directional lights
        for dir_light_dir, dir_light_irr in zip(directional_light_directions, directional_light_irradiances):
            l_dir = -dir_light_dir[None, None, None, :]  # Light direction (opposite of light vector)
            h = torch.nn.functional.normalize(l_dir + view_dir, dim=-1)

            brdf = disney_brdf(
                nor, l_dir, view_dir, h,
                per_pixel_base_color,
                per_pixel_roughness,
                per_pixel_metallic,
                per_pixel_specular,
                per_pixel_specular_tint,
                per_pixel_anisotropic,
            )

            ndotl = torch.sum(nor * l_dir, dim=-1, keepdim=True).clamp_min(0.0)
            light_contrib = brdf * ndotl * dir_light_irr[None, None, None, :]
            col = col + light_contrib

        # Environment map lighting (IBL)
        if envmap_data is not None:
            h_env, w_env = envmap_data.shape[0], envmap_data.shape[1]

            # Helper function to sample envmap from direction with bilinear interpolation
            def sample_envmap(direction):
                """Sample envmap using equirectangular projection with bilinear filtering"""
                dir_normalized = torch.nn.functional.normalize(direction, dim=-1)
                # Flip both U and V to match orientation
                u = 1.0 - (torch.atan2(dir_normalized[..., 0:1], dir_normalized[..., 2:3]) / (2.0 * math.pi) + 0.5)
                v = 1.0 - (torch.asin(torch.clamp(dir_normalized[..., 1:2], -1.0, 1.0)) / math.pi + 0.5)

                # Apply rotation (rotate around vertical axis)
                u = u + (envmap_rotation_val / 360.0)
                u = u - torch.floor(u)  # Wrap to [0, 1]

                # Bilinear interpolation
                # Convert UV [0,1] to pixel coordinates [0, w-1] and [0, h-1]
                x = u.squeeze(-1) * (w_env - 1)
                y = v.squeeze(-1) * (h_env - 1)

                # Get integer coordinates
                x0 = torch.floor(x).long().clamp(0, w_env - 1)
                x1 = (x0 + 1).clamp(0, w_env - 1)
                y0 = torch.floor(y).long().clamp(0, h_env - 1)
                y1 = (y0 + 1).clamp(0, h_env - 1)

                # Get fractional parts
                fx = (x - x0.float()).unsqueeze(-1)
                fy = (y - y0.float()).unsqueeze(-1)

                # Sample 4 corners
                c00 = envmap_data[y0, x0]
                c01 = envmap_data[y0, x1]
                c10 = envmap_data[y1, x0]
                c11 = envmap_data[y1, x1]

                # Bilinear interpolation
                c0 = c00 * (1 - fx) + c01 * fx
                c1 = c10 * (1 - fx) + c11 * fx
                result = c0 * (1 - fy) + c1 * fy

                return result * envmap_scale_val

            # Diffuse IBL: sample using normal direction
            # Note: This is a simplified approximation. For better results, should use:
            # 1. Pre-computed irradiance map (convolved with cosine-weighted hemisphere)
            # 2. Or use a low-resolution mipmap level to approximate hemisphere integral
            diffuse_envmap = sample_envmap(nor)
            diffuse_ibl = per_pixel_base_color * diffuse_envmap[None, ...] * (1.0 - per_pixel_metallic) / math.pi
            col = col + diffuse_ibl

            # Specular IBL: GGX importance sampling
            # Use importance sampling to properly handle roughness-based filtering
            batch, h, w, _ = nor.shape

            # Flatten spatial dimensions for processing
            nor_flat = nor.reshape(-1, 3)  # [N, 3]
            view_dir_flat = view_dir.reshape(-1, 3)  # [N, 3]
            roughness_flat = per_pixel_roughness.reshape(-1)  # [N]

            num_pixels = nor_flat.shape[0]
            num_samples = args.ibl_samples

            # Pre-generate all Hammersley samples
            xi_samples = torch.stack([hammersley(i, num_samples) for i in range(num_samples)], dim=0)  # [S, 2]

            # Accumulate for all pixels and samples
            prefiltered_color = torch.zeros((num_pixels, 3), device=device, dtype=torch.float32)
            total_weight = torch.zeros((num_pixels,), device=device, dtype=torch.float32)

            # Process samples in batches to avoid memory explosion
            for sample_idx in range(num_samples):
                xi = xi_samples[sample_idx]  # [2]

                # Vectorized importance sampling for all pixels
                # This is approximate: uses same random sample xi for all pixels
                # but different roughness values
                a = roughness_flat * roughness_flat

                phi = 2.0 * math.pi * xi[0]
                cos_theta = torch.sqrt((1.0 - xi[1]) / (1.0 + (a * a - 1.0) * xi[1]))
                sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta * cos_theta, min=0.0))

                # Half vector in tangent space [N, 3]
                H_tangent = torch.stack([
                    torch.full_like(roughness_flat, math.cos(phi)) * sin_theta,
                    torch.full_like(roughness_flat, math.sin(phi)) * sin_theta,
                    cos_theta
                ], dim=1)

                # Transform to world space (per-pixel tangent space)
                # Build tangent frames for all pixels at once
                up = torch.where(
                    (torch.abs(nor_flat[:, 2:3]) < 0.999).expand(-1, 3),
                    torch.tensor([[0.0, 0.0, 1.0]], device=device, dtype=torch.float32).expand(num_pixels, -1),
                    torch.tensor([[1.0, 0.0, 0.0]], device=device, dtype=torch.float32).expand(num_pixels, -1)
                )
                tangent = torch.nn.functional.normalize(
                    torch.cross(up, nor_flat, dim=1),
                    dim=1
                )
                bitangent = torch.cross(nor_flat, tangent, dim=1)

                # Transform H from tangent to world space [N, 3]
                H_world = (
                    tangent * H_tangent[:, 0:1] +
                    bitangent * H_tangent[:, 1:2] +
                    nor_flat * H_tangent[:, 2:3]
                )
                H_world = torch.nn.functional.normalize(H_world, dim=1)

                # Compute light direction L = reflect(-V, H) = 2(V·H)H - V
                VdotH = torch.sum(view_dir_flat * H_world, dim=1, keepdim=True)
                L = torch.nn.functional.normalize(
                    2.0 * VdotH * H_world - view_dir_flat,
                    dim=1
                )

                # NdotL for weighting
                NdotL = torch.sum(nor_flat * L, dim=1).clamp(min=0.0)  # [N]

                # Sample environment map for all light directions at once
                # Reshape L to [N, 1, 1, 3] to match sample_envmap input format
                L_reshaped = L.unsqueeze(1).unsqueeze(1)  # [N, 1, 1, 3]
                envmap_colors = sample_envmap(L_reshaped).squeeze(1).squeeze(1)  # [N, 3]

                # Accumulate weighted samples
                prefiltered_color += envmap_colors * NdotL.unsqueeze(1)
                total_weight += NdotL

            # Normalize by total weight
            prefiltered_color = prefiltered_color / (total_weight.unsqueeze(1) + 1e-8)

            # Reshape back to spatial dimensions
            prefiltered_color = prefiltered_color.reshape(batch, h, w, 3)

            # Apply Fresnel
            lum = (0.3 * per_pixel_base_color[..., 0:1] + 0.6 * per_pixel_base_color[..., 1:2] + 0.1 * per_pixel_base_color[..., 2:3])
            tint_color = torch.where(
                lum > 1e-4,
                per_pixel_base_color / lum.clamp_min(1e-4),
                torch.ones_like(per_pixel_base_color),
            )
            tint_color = torch.clamp(tint_color, 0.0, 1.0)
            spec_color = torch.lerp(torch.ones_like(per_pixel_base_color), tint_color, per_pixel_specular_tint)

            cos_theta = torch.sum(nor * view_dir, dim=-1, keepdim=True).clamp(0.0, 1.0)
            # IBL doesn't have a single V·H, so we approximate using N·V.
            f0_scalar = (0.08 * per_pixel_specular).clamp(0.0, 1.0)
            F0_diel_rgb = f0_scalar * spec_color
            F_diel_rgb = fresnel_schlick_ue(F0_diel_rgb, cos_theta)

            F_metal = fresnel_schlick(cos_theta, per_pixel_base_color)
            fresnel = torch.lerp(F_diel_rgb, F_metal, per_pixel_metallic)

            specular_ibl = fresnel * prefiltered_color
            col = col + specular_ibl
        else:
            # Fallback to ambient lighting
            col = col + per_pixel_base_color * ambient_color[None, None, None, :] * (1.0 - per_pixel_metallic)

        # Handle background
        if envmap_data is not None:
            # Compute camera ray directions for all pixels
            # Create pixel coordinate grid
            h, w = height, width
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing='ij'
            )

            # Convert to NDC coordinates [-1, 1]
            ndc_x = (x_coords / (w - 1)) * 2.0 - 1.0
            ndc_y = -((y_coords / (h - 1)) * 2.0 - 1.0)  # Flip Y

            # Compute ray directions in camera space
            aspect = w / float(h)
            fov_rad = cam.fov_deg * (math.pi / 180.0)
            tan_half_fov = math.tan(fov_rad / 2.0)

            # Camera space ray directions
            ray_x = ndc_x * tan_half_fov * aspect
            ray_y = ndc_y * tan_half_fov
            ray_z = -torch.ones_like(ray_x)  # Forward is -Z

            # Stack and normalize
            ray_dirs_cam = torch.stack([ray_x, ray_y, ray_z], dim=-1)
            ray_dirs_cam = torch.nn.functional.normalize(ray_dirs_cam, dim=-1)

            # Transform to world space using view matrix inverse
            # view matrix transforms world to camera, so we need its rotation part inversed
            cam_forward = torch.nn.functional.normalize(
                torch.tensor(cam.target, device=device) - torch.tensor(cam.origin, device=device),
                dim=0
            )
            cam_right = torch.nn.functional.normalize(
                torch.cross(cam_forward, torch.tensor(cam.up, device=device), dim=0),
                dim=0
            )
            cam_up = torch.cross(cam_right, cam_forward, dim=0)

            # Rotation matrix from camera to world
            # [right, up, -forward] as column vectors
            rot_cam_to_world = torch.stack([cam_right, cam_up, -cam_forward], dim=1)

            # Apply rotation: ray_world = rot * ray_cam
            ray_dirs_world = torch.matmul(ray_dirs_cam, rot_cam_to_world.T)
            ray_dirs_world = torch.nn.functional.normalize(ray_dirs_world, dim=-1)

            # Sample envmap for background
            bg_color = sample_envmap(ray_dirs_world)
            col = torch.where(mask, col, bg_color)
        else:
            # Black background
            col = torch.where(mask, col, torch.zeros_like(col))

        return col[0], mask[0]

    eps = 1e-3

    print(f"\nStarting optimization for {args.steps} steps...")
    print(f"Initial parameters:")
    print(f"  base_color: {init_base_color}")
    print(f"  roughness: {args.init_roughness:.3f}")
    print(f"  metallic: {args.init_metallic:.3f}")
    print(f"  specular: {args.init_specular:.3f}\n")

    # Prepare progress preview path for GUI
    progress_preview_path = out_dir / "progress.png"

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)

        # Apply sigmoid to get [0, 1] range
        base_color = torch.sigmoid(base_color_logit)
        roughness = torch.sigmoid(roughness_logit)
        metallic = torch.sigmoid(metallic_logit)
        specular = torch.sigmoid(specular_logit)
        specular_tint = torch.sigmoid(specular_tint_logit)
        anisotropic = torch.sigmoid(anisotropic_logit)
        sheen = torch.sigmoid(sheen_logit)
        sheen_tint = torch.sigmoid(sheen_tint_logit)
        clearcoat = torch.sigmoid(clearcoat_logit)
        clearcoat_gloss = torch.sigmoid(clearcoat_gloss_logit)

        pred, mask = render_raster(
            base_color, roughness, metallic, specular,
            specular_tint, anisotropic, sheen, sheen_tint,
            clearcoat, clearcoat_gloss
        )

        # Compare only where raster has geometry coverage
        # Compute loss only on masked regions
        diff = torch.abs(torch.log(pred + eps) - torch.log(gt + eps))
        masked_diff = diff * mask  # mask is [H, W, 1], will broadcast to [H, W, 3]
        loss = torch.sum(masked_diff) / (torch.sum(mask) * 3 + 1e-8)  # Average over valid pixels and channels

        loss.backward()

        # If a parameter is marked differentiable but doesn't affect the rendered image,
        # its gradient will be ~0 and it will not update. Emit a warning to make this obvious.
        if step % 25 == 0 or step == args.steps - 1:
            grad_warn_eps = 1e-12

            def warn_if_no_grad(name: str, t):
                if not getattr(t, "requires_grad", False):
                    return
                g = t.grad
                if g is None:
                    print(f"[warn] grad[{name}] is None (parameter may be unused in the forward pass)")
                    return
                gmax = float(g.detach().abs().max().item())
                if gmax < grad_warn_eps:
                    print(
                        f"[warn] grad[{name}] is ~0 (max|grad|={gmax:.2e}). "
                        "Parameter won't update; it may be unused or the scene/lighting doesn't constrain it."
                    )

            warn_if_no_grad("base_color", base_color_logit)
            warn_if_no_grad("roughness", roughness_logit)
            warn_if_no_grad("metallic", metallic_logit)
            warn_if_no_grad("specular", specular_logit)
            warn_if_no_grad("specular_tint", specular_tint_logit)
            warn_if_no_grad("anisotropic", anisotropic_logit)
            warn_if_no_grad("sheen", sheen_logit)
            warn_if_no_grad("sheen_tint", sheen_tint_logit)
            warn_if_no_grad("clearcoat", clearcoat_logit)
            warn_if_no_grad("clearcoat_gloss", clearcoat_gloss_logit)

        opt.step()

        if step % 25 == 0 or step == args.steps - 1:
            with torch.no_grad():
                bc = base_color.detach().cpu().numpy()
                r = roughness.item()
                m_val = metallic.item()
                s = specular.item()
                st = specular_tint.item()
                an = anisotropic.item()
                sh = sheen.item()
                sht = sheen_tint.item()
                cc = clearcoat.item()
                ccg = clearcoat_gloss.item()

                # Format for GUI parsing
                print(f"step={step:04d} loss={loss.item():.6f} "
                      f"baseColor=[{bc[0]:.6f} {bc[1]:.6f} {bc[2]:.6f}] "
                      f"roughness={r:.6f} metallic={m_val:.6f} specular={s:.6f} "
                      f"specularTint={st:.6f} anisotropic={an:.6f} "
                      f"sheen={sh:.6f} sheenTint={sht:.6f} "
                      f"clearcoat={cc:.6f} clearcoatGloss={ccg:.6f}")

                # Save progress preview for GUI (2x2 grid layout)
                pred_np = pred.detach().cpu().numpy().astype(np.float32)
                # Apply a display power curve directly to the shading RGB.
                # This is intentionally done here (not in the file saver) to keep the rest of the
                # pipeline unchanged while making the preview match the expected appearance.
                display_power = 2.2
                pred_clipped = np.clip(pred_np, 0.0, 1.0) ** display_power
                gt_clipped = np.clip(img_gt_np, 0.0, 1.0) ** display_power
                diff = np.clip(np.abs(gt_clipped - pred_clipped) * 3.0, 0.0, 1.0)

                # Create 2x2 grid: [GT | Current]
                #                  [Diff | Info  ]
                h, w = gt_clipped.shape[:2]
                canvas = np.zeros((h * 2, w * 2, 3), dtype=np.float32)

                # Top row: GT | Current
                canvas[:h, :w, :] = gt_clipped
                canvas[:h, w:, :] = pred_clipped

                # Bottom left: Diff
                canvas[h:, :w, :] = diff

                # Bottom right: Info panel
                canvas[h:, w:, :] = np.ones((h, w, 3), dtype=np.float32) * 0.1

                # Add text labels
                try:
                    from PIL import Image, ImageDraw, ImageFont
                    img_pil = Image.fromarray((canvas * 255).astype(np.uint8))
                    draw = ImageDraw.Draw(img_pil)

                    # Use default font
                    try:
                        font_large = ImageFont.truetype("arial.ttf", 22)
                        font_small = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font_large = ImageFont.load_default()
                        font_small = font_large

                    # Draw labels on top row
                    draw.text((10, 10), "Ground Truth", fill=(255, 255, 0), font=font_large)
                    draw.text((w + 10, 10), f"Current (Step {step}/{args.steps})", fill=(255, 255, 0), font=font_large)

                    # Draw label on bottom left
                    draw.text((10, h + 10), "Difference x3", fill=(255, 255, 0), font=font_large)

                    # Draw parameters on bottom right
                    draw.text((w + 10, h + 10), "Parameters:", fill=(255, 255, 255), font=font_large)

                    # Compact multi-line table so all Disney params are visible.
                    def fmt(name: str, value: str, diff: bool | None):
                        tag = " [Diff]" if diff else ""
                        return f"{name}: {value}{tag}"

                    lines = [
                        fmt("Loss", f"{loss.item():.6f}", None),
                        fmt("base_color", f"[{bc[0]:.3f} {bc[1]:.3f} {bc[2]:.3f}]", diff_base_color),
                        fmt("roughness", f"{r:.4f}", diff_roughness),
                        fmt("metallic", f"{m_val:.4f}", diff_metallic),
                        fmt("specular", f"{s:.4f}", diff_specular),
                        fmt("specular_tint", f"{st:.4f}", diff_specular_tint),
                        fmt("anisotropic", f"{an:.4f}", diff_anisotropic),
                        fmt("sheen", f"{sh:.4f}", diff_sheen),
                        fmt("sheen_tint", f"{sht:.4f}", diff_sheen_tint),
                        fmt("clearcoat", f"{cc:.4f}", diff_clearcoat),
                        fmt("clearcoat_gloss", f"{ccg:.4f}", diff_clearcoat_gloss),
                    ]

                    # Estimate a safe line height.
                    try:
                        line_h = int((font_small.getbbox("Ag")[3] - font_small.getbbox("Ag")[1]) * 1.25)
                    except Exception:
                        line_h = 18

                    y_offset = h + 45
                    for line in lines:
                        if y_offset + line_h > (h + h - 6):
                            break
                        color = (200, 200, 255)
                        if line.startswith("Loss"):
                            color = (255, 200, 200)
                        draw.text((w + 10, y_offset), line, fill=color, font=font_small)
                        y_offset += line_h

                    img_pil.save(str(progress_preview_path))
                except Exception as e:
                    # Fallback: save without labels
                    save_image_u8(progress_preview_path, canvas)

    # Final render and save
    with torch.no_grad():
        base_color = torch.sigmoid(base_color_logit)
        roughness = torch.sigmoid(roughness_logit)
        metallic = torch.sigmoid(metallic_logit)
        specular = torch.sigmoid(specular_logit)
        specular_tint = torch.sigmoid(specular_tint_logit)
        anisotropic = torch.sigmoid(anisotropic_logit)
        sheen = torch.sigmoid(sheen_logit)
        sheen_tint = torch.sigmoid(sheen_tint_logit)
        clearcoat = torch.sigmoid(clearcoat_logit)
        clearcoat_gloss = torch.sigmoid(clearcoat_gloss_logit)

        pred, mask = render_raster(
            base_color, roughness, metallic, specular,
            specular_tint, anisotropic, sheen, sheen_tint,
            clearcoat, clearcoat_gloss
        )

        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        mask_np = mask.detach().cpu().numpy().astype(np.float32)

    # Save outputs
    save_image_u8(out_dir / "gt.png", np.clip(img_gt_np, 0.0, 1.0))
    save_image_u8(out_dir / "pred.png", np.clip(pred_np, 0.0, 1.0))

    diff = np.abs(np.clip(img_gt_np, 0.0, 1.0) - np.clip(pred_np, 0.0, 1.0))
    save_image_u8(out_dir / "diff.png", np.clip(diff * 4.0, 0.0, 1.0))

    # Save fitted parameters
    bc_final = base_color.detach().cpu().numpy()
    params = {
        "base_color": [float(bc_final[0]), float(bc_final[1]), float(bc_final[2])],
        "roughness": float(roughness.item()),
        "metallic": float(metallic.item()),
        "specular": float(specular.item()),
        "specular_tint": float(specular_tint.item()),
        "anisotropic": float(anisotropic.item()),
        "sheen": float(sheen.item()),
        "sheen_tint": float(sheen_tint.item()),
        "clearcoat": float(clearcoat.item()),
        "clearcoat_gloss": float(clearcoat_gloss.item()),
        "steps": args.steps,
        "lr": args.lr,
        "final_loss": float(loss.item()),
    }

    params_path = out_dir / "fit_params.json"
    params_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")

    print(f"\nOptimization complete!")
    print(f"Final parameters:")
    print(f"  base_color: [{bc_final[0]:.3f}, {bc_final[1]:.3f}, {bc_final[2]:.3f}]")
    print(f"  roughness: {roughness.item():.3f}")
    print(f"  metallic: {metallic.item():.3f}")
    print(f"  specular: {specular.item():.3f}")
    print(f"  specular_tint: {specular_tint.item():.3f}")
    print(f"  anisotropic: {anisotropic.item():.3f}")
    print(f"  sheen: {sheen.item():.3f}")
    print(f"  sheen_tint: {sheen_tint.item():.3f}")
    print(f"  clearcoat: {clearcoat.item():.3f}")
    print(f"  clearcoat_gloss: {clearcoat_gloss.item():.3f}")
    print(f"  final_loss: {loss.item():.6f}")
    print(f"\nWrote results to: {out_dir}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise
