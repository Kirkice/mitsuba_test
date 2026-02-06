#!/usr/bin/env python3
"""Disney BRDF material fitting with nvdiffrast.

This script fits a Disney principled BRDF to Mitsuba path-traced ground truth.
It optimizes multiple PBR parameters including:
- Base color (albedo)
- Roughness
- Metallic
- Specular reflectance
- Specular tint
- Anisotropic

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
import sys
from pathlib import Path

import numpy as np

# Import shared rendering core
from mitsuba_render_core import (
    _require,
    parse_scene_xml,
    look_at,
    perspective,
    make_uv_sphere_mesh,
    make_cube_mesh,
    transform_wall_mesh,
    combine_meshes,
    load_mesh_any,
    save_image_u8,
    disney_brdf_full,
    load_envmap,
    hammersley,
)


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
    if args.config and disney_material_config is not None:
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

    # Transform normals to world space
    # Normals use inverse transpose of scale matrix (for non-uniform scaling)
    # Since we only have scale (no rotation), inverse transpose = scale inverse
    n_obj = n_obj * (1.0 / obj.scale.reshape(1, 3))
    n_obj = n_obj / (np.linalg.norm(n_obj, axis=1, keepdims=True) + 1e-8)

    # Generate wall meshes and track vertex colors
    meshes_to_combine = [(v_obj, f_obj, n_obj)]
    vertex_colors = []

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
    envmap_rotation_val = 0.0

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
            envmap_np = load_envmap(Path(light.envmap_filename), ws)
            if envmap_np is not None:
                envmap_data = torch.from_numpy(envmap_np).to(device=device, dtype=torch.float32)
                envmap_scale_val = light.envmap_scale
                envmap_rotation_val = light.envmap_rotation
                print(f"  Path: {light.envmap_filename}")
                print(f"  Shape: {envmap_data.shape}")
                print(f"  Value range: [{envmap_data.min().item():.4f}, {envmap_data.max().item():.4f}]")
                print(f"  Rotation: {envmap_rotation_val} degrees")
                print(f"  Mean: {envmap_data.mean().item():.4f}")
                print(f"  Scale: {envmap_scale_val}")
            else:
                print("Using constant ambient fallback")

    # Raster context
    ctx = drt.RasterizeCudaContext()

    # Constant ambient fallback (used when no envmap lighting is available)
    ambient_color = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)

    # Render function - this is the key difference from soft_render.py
    # This version will be called multiple times during optimization
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

            brdf = disney_brdf_full(
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
            brdf = disney_brdf_full(
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

            brdf = disney_brdf_full(
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

        # Environment map lighting (IBL) - similar implementation to soft_render.py
        if envmap_data is not None:
            h_env, w_env = envmap_data.shape[0], envmap_data.shape[1]

            # Helper function to sample envmap from direction with bilinear interpolation
            def sample_envmap(direction):
                """Sample envmap using equirectangular projection with bilinear filtering"""
                dir_normalized = torch.nn.functional.normalize(direction, dim=-1)
                u = 1.0 - (torch.atan2(dir_normalized[..., 0:1], dir_normalized[..., 2:3]) / (2.0 * math.pi) + 0.5)
                v = 1.0 - (torch.asin(torch.clamp(dir_normalized[..., 1:2], -1.0, 1.0)) / math.pi + 0.5)

                # Apply rotation (rotate around vertical axis)
                u = u + (envmap_rotation_val / 360.0)
                u = u - torch.floor(u)  # Wrap to [0, 1]

                # Bilinear interpolation
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
            diffuse_envmap = sample_envmap(nor)
            diffuse_ibl = per_pixel_base_color * diffuse_envmap[None, ...] * (1.0 - per_pixel_metallic) / math.pi
            col = col + diffuse_ibl

            # Specular IBL: GGX importance sampling (same implementation as soft_render.py)
            batch, h, w, _ = nor.shape

            # Flatten spatial dimensions for processing
            nor_flat = nor.reshape(-1, 3)
            view_dir_flat = view_dir.reshape(-1, 3)
            roughness_flat = per_pixel_roughness.reshape(-1)

            num_pixels = nor_flat.shape[0]
            num_samples = args.ibl_samples

            # Pre-generate all Hammersley samples
            xi_samples = torch.stack([hammersley(i, num_samples, device) for i in range(num_samples)], dim=0)

            # Accumulate for all pixels and samples
            prefiltered_color = torch.zeros((num_pixels, 3), device=device, dtype=torch.float32)
            total_weight = torch.zeros((num_pixels,), device=device, dtype=torch.float32)

            # Process samples in batches to avoid memory explosion
            for sample_idx in range(num_samples):
                xi = xi_samples[sample_idx]

                # Vectorized importance sampling for all pixels
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
                NdotL = torch.sum(nor_flat * L, dim=1).clamp(min=0.0)

                # Sample environment map for all light directions at once
                L_reshaped = L.unsqueeze(1).unsqueeze(1)
                envmap_colors = sample_envmap(L_reshaped).squeeze(1).squeeze(1)

                # Accumulate weighted samples
                prefiltered_color += envmap_colors * NdotL.unsqueeze(1)
                total_weight += NdotL

            # Normalize by total weight
            prefiltered_color = prefiltered_color / (total_weight.unsqueeze(1) + 1e-8)

            # Reshape back to spatial dimensions
            prefiltered_color = prefiltered_color.reshape(batch, h, w, 3)

            # Apply Fresnel (using specular_tint for tinted Fresnel)
            from mitsuba_render_core import fresnel_schlick_ue, fresnel_schlick

            lum = (0.3 * per_pixel_base_color[..., 0:1] + 0.6 * per_pixel_base_color[..., 1:2] + 0.1 * per_pixel_base_color[..., 2:3])
            tint_color = torch.where(
                lum > 1e-4,
                per_pixel_base_color / lum.clamp_min(1e-4),
                torch.ones_like(per_pixel_base_color),
            )
            tint_color = torch.clamp(tint_color, 0.0, 1.0)
            spec_color = torch.lerp(torch.ones_like(per_pixel_base_color), tint_color, per_pixel_specular_tint)

            cos_theta = torch.sum(nor * view_dir, dim=-1, keepdim=True).clamp(0.0, 1.0)
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

        # Handle background (same as soft_render.py)
        if envmap_data is not None:
            # Compute camera ray directions for all pixels
            h, w = height, width
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=device, dtype=torch.float32),
                torch.arange(w, device=device, dtype=torch.float32),
                indexing='ij'
            )

            # Convert to NDC coordinates [-1, 1]
            ndc_x = (x_coords / (w - 1)) * 2.0 - 1.0
            ndc_y = -((y_coords / (h - 1)) * 2.0 - 1.0)

            # Compute ray directions in camera space
            aspect = w / float(h)
            fov_rad = cam.fov_deg * (math.pi / 180.0)
            tan_half_fov = math.tan(fov_rad / 2.0)

            # Camera space ray directions
            ray_x = ndc_x * tan_half_fov * aspect
            ray_y = ndc_y * tan_half_fov
            ray_z = -torch.ones_like(ray_x)

            # Stack and normalize
            ray_dirs_cam = torch.stack([ray_x, ray_y, ray_z], dim=-1)
            ray_dirs_cam = torch.nn.functional.normalize(ray_dirs_cam, dim=-1)

            # Transform to world space
            cam_forward = torch.nn.functional.normalize(
                torch.tensor(cam.target, device=device) - torch.tensor(cam.origin, device=device),
                dim=0
            )
            cam_right = torch.nn.functional.normalize(
                torch.cross(cam_forward, torch.tensor(cam.up, device=device), dim=0),
                dim=0
            )
            cam_up = torch.cross(cam_right, cam_forward, dim=0)

            rot_cam_to_world = torch.stack([cam_right, cam_up, -cam_forward], dim=1)
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

    # Optimization loop
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

        # Compute loss only on masked regions
        diff = torch.abs(torch.log(pred + eps) - torch.log(gt + eps))
        masked_diff = diff * mask
        loss = torch.sum(masked_diff) / (torch.sum(mask) * 3 + 1e-8)

        loss.backward()

        # Gradient warnings (check if gradients are too small)
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

        # Print progress and save preview
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
                display_power = 2.2
                pred_clipped = np.clip(pred_np, 0.0, 1.0) ** display_power
                gt_clipped = np.clip(img_gt_np, 0.0, 1.0) ** display_power
                diff_img = np.clip(np.abs(gt_clipped - pred_clipped) * 3.0, 0.0, 1.0)

                # Create 2x2 grid: [GT | Current]
                #                  [Diff | Info  ]
                h, w = gt_clipped.shape[:2]
                canvas = np.zeros((h * 2, w * 2, 3), dtype=np.float32)

                # Top row: GT | Current
                canvas[:h, :w, :] = gt_clipped
                canvas[:h, w:, :] = pred_clipped

                # Bottom left: Diff
                canvas[h:, :w, :] = diff_img

                # Bottom right: Info panel
                canvas[h:, w:, :] = np.ones((h, w, 3), dtype=np.float32) * 0.1

                # Add text labels using PIL
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

                    # Draw labels
                    draw.text((10, 10), "Ground Truth", fill=(255, 255, 0), font=font_large)
                    draw.text((w + 10, 10), f"Current (Step {step}/{args.steps})", fill=(255, 255, 0), font=font_large)
                    draw.text((10, h + 10), "Difference x3", fill=(255, 255, 0), font=font_large)
                    draw.text((w + 10, h + 10), "Parameters:", fill=(255, 255, 255), font=font_large)

                    # Compact multi-line table
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

                    # Estimate line height
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
