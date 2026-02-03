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
    position: np.ndarray
    radiance: np.ndarray


def parse_scene_xml(scene_path: Path) -> Tuple[Camera, ObjectSpec, list[LightSpec]]:
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
    for shape in root.findall("./shape"):
        emitter = shape.find("./emitter[@type='area']")
        if emitter is None:
            continue

        light_pos = np.array([0.0, 1.99, 0.0], dtype=np.float32)
        light_rad = np.array([18.0, 18.0, 18.0], dtype=np.float32)

        rgb = emitter.find("./rgb[@name='radiance']")
        if rgb is not None:
            light_rad = _parse_vec3(rgb.attrib["value"])

        tr = shape.find("./transform[@name='to_world']/translate")
        if tr is not None:
            light_pos = np.array(
                [float(tr.attrib.get("x", 0.0)), float(tr.attrib.get("y", 0.0)), float(tr.attrib.get("z", 0.0))],
                dtype=np.float32,
            )

        lights.append(LightSpec(position=light_pos, radiance=light_rad))

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

    return cam, obj, lights


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
    m[1, 1] = f
    m[2, 2] = (z_far + z_near) / (z_near - z_far)
    m[2, 3] = (2.0 * z_far * z_near) / (z_near - z_far)
    m[3, 2] = -1.0
    return m


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
    fd90 = 0.5 + 2.0 * ldoth * ldoth * roughness
    fl = schlick_weight(ndotl)
    fv = schlick_weight(ndotv)
    fd = torch.lerp(torch.ones_like(fd90), fd90, fl) * torch.lerp(torch.ones_like(fd90), fd90, fv)

    # Base diffuse
    diffuse = base_color * fd / torch.pi

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Mitsuba XML scene path")
    ap.add_argument("--gt-variant", default="scalar_rgb", help="Mitsuba variant for ground truth")
    ap.add_argument("--gt-spp", type=int, default=256, help="SPP for ground truth")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--out-dir", default="renders/fit_disney", help="Output directory")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Torch device")
    ap.add_argument("--seed", type=int, default=0)

    # Material parameter initialization
    ap.add_argument("--init-base-color", type=str, default="0.8,0.8,0.8", help="Initial base color RGB")
    ap.add_argument("--init-roughness", type=float, default=0.5, help="Initial roughness")
    ap.add_argument("--init-metallic", type=float, default=0.0, help="Initial metallic")
    ap.add_argument("--init-specular", type=float, default=0.5, help="Initial specular")

    args = ap.parse_args()

    ws = Path.cwd()
    scene_path = Path(args.scene)
    if not scene_path.is_absolute():
        scene_path = ws / scene_path

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ws / out_dir

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

    cam, obj, lights = parse_scene_xml(scene_path)

    print(f"Parsed {len(lights)} light(s) from scene")

    # Load/build object mesh
    if obj.kind == "sphere":
        v, f, n = make_uv_sphere_mesh(radius=obj.radius)
    elif obj.kind == "cube":
        v, f, n = make_cube_mesh()
    else:
        mesh_path = Path(obj.filename)
        if not mesh_path.is_absolute():
            mesh_path = scene_path.parent / mesh_path
        v, f, n = load_mesh_any(mesh_path)

    v = v * obj.scale.reshape(1, 3) + obj.translate.reshape(1, 3)

    device = torch.device(args.device)

    v_t = torch.from_numpy(v).to(device=device, dtype=torch.float32)
    n_t = torch.from_numpy(n).to(device=device, dtype=torch.float32)
    f_t = torch.from_numpy(f).to(device=device, dtype=torch.int32)

    # Camera matrices
    aspect = cam.width / max(1.0, float(cam.height))
    view = look_at(cam.origin, cam.target, cam.up)
    proj = perspective(cam.fov_deg, aspect)
    mvp = (proj @ view).astype(np.float32)

    mvp_t = torch.from_numpy(mvp).to(device=device, dtype=torch.float32)

    # Homogeneous coordinates
    ones = torch.ones((v_t.shape[0], 1), device=device, dtype=torch.float32)
    v_h = torch.cat([v_t, ones], dim=1)
    v_clip = (v_h @ mvp_t.T).contiguous()

    # Parse initial values
    init_base_color = np.array([float(x) for x in args.init_base_color.split(",")], dtype=np.float32)
    if init_base_color.shape[0] != 3:
        raise ValueError("--init-base-color must have 3 values")

    # Material parameters (logit parameterization for bounded values)
    base_color_logit = torch.from_numpy(
        np.log(init_base_color / (1.0 - init_base_color + 1e-8))
    ).to(device=device, dtype=torch.float32)
    base_color_logit.requires_grad = True

    roughness_logit = torch.tensor(
        [math.log(args.init_roughness / (1.0 - args.init_roughness + 1e-8))],
        device=device, dtype=torch.float32, requires_grad=True
    )

    metallic_logit = torch.tensor(
        [math.log(args.init_metallic / (1.0 - args.init_metallic + 1e-8))],
        device=device, dtype=torch.float32, requires_grad=True
    )

    specular_logit = torch.tensor(
        [math.log(args.init_specular / (1.0 - args.init_specular + 1e-8))],
        device=device, dtype=torch.float32, requires_grad=True
    )

    opt = torch.optim.Adam([base_color_logit, roughness_logit, metallic_logit, specular_logit], lr=args.lr)

    # GT to torch
    gt = torch.from_numpy(img_gt_np).to(device=device, dtype=torch.float32)

    # Camera position for view vector
    cam_pos_t = torch.from_numpy(cam.origin).to(device=device, dtype=torch.float32)

    # Lights to torch
    lights_pos = []
    lights_rad = []
    for light in lights:
        lights_pos.append(torch.from_numpy(light.position).to(device=device, dtype=torch.float32))
        lights_rad.append(torch.from_numpy(light.radiance).to(device=device, dtype=torch.float32))

    # Raster context
    ctx = drt.RasterizeGLContext()

    # Ambient light (simple environment approximation)
    ambient_color = torch.tensor([0.05, 0.05, 0.05], device=device, dtype=torch.float32)

    def render_raster(base_color, roughness, metallic, specular):
        rast, _ = drt.rasterize(ctx, v_clip[None, ...], f_t, (height, width))
        mask = rast[..., 3:4] > 0

        # Interpolate world position and normals
        pos, _ = drt.interpolate(v_t[None, ...], rast, f_t)
        nor, _ = drt.interpolate(n_t[None, ...], rast, f_t)
        nor = torch.nn.functional.normalize(nor, dim=-1)

        # View direction
        view_dir = torch.nn.functional.normalize(cam_pos_t[None, None, None, :] - pos, dim=-1)

        # Accumulate lighting from all lights
        col = torch.zeros_like(pos)

        for light_pos, light_rad in zip(lights_pos, lights_rad):
            # Light direction and distance
            l_vec = light_pos[None, None, None, :] - pos
            dist2 = torch.sum(l_vec * l_vec, dim=-1, keepdim=True).clamp_min(1e-4)
            l_dir = l_vec / torch.sqrt(dist2)

            # Half vector
            h = torch.nn.functional.normalize(l_dir + view_dir, dim=-1)

            # Evaluate Disney BRDF
            brdf = disney_brdf(
                nor, l_dir, view_dir, h,
                base_color[None, None, None, :],
                roughness[None, None, None, :],
                metallic[None, None, None, :],
                specular[None, None, None, :]
            )

            # Lighting equation
            ndotl = torch.sum(nor * l_dir, dim=-1, keepdim=True).clamp_min(0.0)
            light_contrib = brdf * ndotl * (light_rad[None, None, None, :] / dist2)
            col = col + light_contrib

        # Add ambient lighting (very simple approximation)
        col = col + base_color[None, None, None, :] * ambient_color[None, None, None, :] * (1.0 - metallic[None, None, None, :])

        # Background = 0
        col = torch.where(mask, col, torch.zeros_like(col))
        return col[0], mask[0]

    eps = 1e-3

    print(f"\nStarting optimization for {args.steps} steps...")
    print(f"Initial parameters:")
    print(f"  base_color: {init_base_color}")
    print(f"  roughness: {args.init_roughness:.3f}")
    print(f"  metallic: {args.init_metallic:.3f}")
    print(f"  specular: {args.init_specular:.3f}\n")

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)

        # Apply sigmoid to get [0, 1] range
        base_color = torch.sigmoid(base_color_logit)
        roughness = torch.sigmoid(roughness_logit)
        metallic = torch.sigmoid(metallic_logit)
        specular = torch.sigmoid(specular_logit)

        pred, mask = render_raster(base_color, roughness, metallic, specular)

        # Compare only where raster has geometry coverage
        m = mask.expand_as(pred)
        pred_m = pred[m]
        gt_m = gt[m]

        # HDR-friendly loss (log + L1)
        loss = torch.mean(torch.abs(torch.log(pred_m + eps) - torch.log(gt_m + eps)))

        loss.backward()
        opt.step()

        if step % 25 == 0 or step == args.steps - 1:
            with torch.no_grad():
                bc = base_color.detach().cpu().numpy()
                r = roughness.item()
                m_val = metallic.item()
                s = specular.item()

                # Format for GUI parsing
                print(f"step={step:04d} loss={loss.item():.6f} "
                      f"baseColor=[{bc[0]:.6f} {bc[1]:.6f} {bc[2]:.6f}] "
                      f"roughness={r:.6f} metallic={m_val:.6f} specular={s:.6f}")

    # Final render and save
    with torch.no_grad():
        base_color = torch.sigmoid(base_color_logit)
        roughness = torch.sigmoid(roughness_logit)
        metallic = torch.sigmoid(metallic_logit)
        specular = torch.sigmoid(specular_logit)

        pred, mask = render_raster(base_color, roughness, metallic, specular)

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
