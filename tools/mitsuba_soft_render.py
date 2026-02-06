#!/usr/bin/env python3
"""Single-pass soft rasterization renderer using nvdiffrast and Disney BRDF.

This script performs a single forward render pass using material parameters
from the GUI configuration file. It provides a fast preview of the current
Disney BRDF material settings without optimization.

Features:
- Disney BRDF implementation (based on Burley 2012)
- Multi-light support (area, point, spot, directional)
- Environment map support (IBL with importance sampling)
- Reads material parameters from .mitsuba_studio_state.json

Example:
  python tools/mitsuba_soft_render.py \
    --scene scenes/cbox.xml \
    --config .mitsuba_studio_state.json \
    --out renders/soft_preview.png
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
    disney_brdf_simple,
    load_envmap,
    hammersley,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="Mitsuba XML scene path")
    ap.add_argument("--out", required=True, help="Output image path")
    ap.add_argument("--config", required=True, help="JSON config file path (.mitsuba_studio_state.json)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Torch device")
    ap.add_argument("--ibl-samples", type=int, default=128, help="Number of samples for IBL importance sampling")

    args = ap.parse_args()

    ws = Path.cwd()
    scene_path = Path(args.scene)
    if not scene_path.is_absolute():
        scene_path = ws / scene_path

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ws / out_path

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ws / config_path

    # Load Disney material config from JSON
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = json.load(f)
        if 'config' not in full_config or 'disney_material' not in full_config['config']:
            raise ValueError("Config file missing 'config.disney_material' section")
        disney_material_config = full_config['config']['disney_material']

    # Load material parameters from config
    base_color = np.array(disney_material_config['base_color']['value'], dtype=np.float32)
    roughness = disney_material_config['roughness']['value']
    metallic = disney_material_config['metallic']['value']
    specular = disney_material_config['specular']['value']

    print("Disney Material Parameters:")
    print(f"  base_color: [{base_color[0]:.3f}, {base_color[1]:.3f}, {base_color[2]:.3f}]")
    print(f"  roughness: {roughness:.3f}")
    print(f"  metallic: {metallic:.3f}")
    print(f"  specular: {specular:.3f}")

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

    # Parse scene
    cam, obj, lights, walls = parse_scene_xml(scene_path)

    # Override light config from JSON if provided
    if 'light' in full_config['config']:
        light_config = full_config['config']['light']
        # Update envmap rotation if light is envmap type
        for light in lights:
            if light.light_type == 'envmap':
                if 'envmap_rotation' in light_config:
                    light.envmap_rotation = light_config['envmap_rotation']
                    print(f"  envmap_rotation: {light.envmap_rotation} degrees")

    print(f"\nScene Info:")
    print(f"  Parsed {len(lights)} light(s) and {len(walls)} wall(s) from scene")

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

    # Object vertices will use Disney material (mark with special color for now)
    # We'll use -1 to indicate "use Disney material"
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
    v_clip = (v_h @ mvp_t.T).contiguous()

    # Convert material parameters to torch tensors
    base_color_t = torch.from_numpy(base_color).to(device=device, dtype=torch.float32)
    roughness_t = torch.tensor([roughness], device=device, dtype=torch.float32)
    metallic_t = torch.tensor([metallic], device=device, dtype=torch.float32)
    specular_t = torch.tensor([specular], device=device, dtype=torch.float32)

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
                print(f"  Envmap shape: {envmap_data.shape}")
                print(f"  Envmap value range: [{envmap_data.min().item():.4f}, {envmap_data.max().item():.4f}]")
                print(f"  Envmap scale: {envmap_scale_val}")
            else:
                print("  Warning: Using constant ambient fallback")

    # Raster context
    ctx = drt.RasterizeCudaContext()

    # Increase ambient light to approximate indirect illumination from colored walls
    ambient_color = torch.tensor([0.15, 0.15, 0.15], device=device, dtype=torch.float32)

    print(f"\nRendering soft rasterization preview...")

    # Render function
    def render_raster(base_color, roughness, metallic, specular):
        rast, _ = drt.rasterize(ctx, v_clip[None, ...], f_t, (cam.height, cam.width))
        mask = rast[..., 3:4] > 0

        # Interpolate world position, normals, and vertex colors
        pos, _ = drt.interpolate(v_t[None, ...], rast, f_t)
        nor, _ = drt.interpolate(n_t[None, ...], rast, f_t)
        nor = torch.nn.functional.normalize(nor, dim=-1)

        # Interpolate per-vertex base colors
        vertex_colors, _ = drt.interpolate(vertex_base_colors_t[None, ...], rast, f_t)

        # For object vertices (marked with negative values), use Disney base_color
        # For wall vertices, use their fixed colors
        is_object = vertex_colors[..., 0:1] < 0
        per_pixel_base_color = torch.where(is_object, base_color[None, None, None, :], vertex_colors)

        # For walls, use simple diffuse material (metallic=0, high roughness)
        # For object, use Disney material
        per_pixel_roughness = torch.where(is_object, roughness[None, None, None, :], torch.ones_like(roughness[None, None, None, :]))
        per_pixel_metallic = torch.where(is_object, metallic[None, None, None, :], torch.zeros_like(metallic[None, None, None, :]))
        per_pixel_specular = torch.where(is_object, specular[None, None, None, :], torch.ones_like(specular[None, None, None, :]) * 0.5)

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

            brdf = disney_brdf_simple(
                nor, l_dir, view_dir, h,
                per_pixel_base_color,
                per_pixel_roughness,
                per_pixel_metallic,
                per_pixel_specular
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
            brdf = disney_brdf_simple(
                nor, l_dir, view_dir, h,
                per_pixel_base_color,
                per_pixel_roughness,
                per_pixel_metallic,
                per_pixel_specular
            )

            ndotl = torch.sum(nor * l_dir, dim=-1, keepdim=True).clamp_min(0.0)
            light_contrib = brdf * ndotl * (spot_rad[None, None, None, :] / dist2) * falloff
            col = col + light_contrib

        # Directional lights
        for dir_light_dir, dir_light_irr in zip(directional_light_directions, directional_light_irradiances):
            l_dir = -dir_light_dir[None, None, None, :]  # Light direction (opposite of light vector)
            h = torch.nn.functional.normalize(l_dir + view_dir, dim=-1)

            brdf = disney_brdf_simple(
                nor, l_dir, view_dir, h,
                per_pixel_base_color,
                per_pixel_roughness,
                per_pixel_metallic,
                per_pixel_specular
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
            diffuse_envmap = sample_envmap(nor)
            diffuse_ibl = per_pixel_base_color * diffuse_envmap[None, ...] * (1.0 - per_pixel_metallic) / math.pi
            col = col + diffuse_ibl

            # Specular IBL: GGX importance sampling
            batch, h, w, _ = nor.shape

            # Flatten spatial dimensions for processing
            nor_flat = nor.reshape(-1, 3)  # [N, 3]
            view_dir_flat = view_dir.reshape(-1, 3)  # [N, 3]
            roughness_flat = per_pixel_roughness.reshape(-1)  # [N]

            num_pixels = nor_flat.shape[0]
            num_samples = args.ibl_samples

            # Pre-generate all Hammersley samples
            xi_samples = torch.stack([hammersley(i, num_samples, device) for i in range(num_samples)], dim=0)  # [S, 2]

            # Accumulate for all pixels and samples
            prefiltered_color = torch.zeros((num_pixels, 3), device=device, dtype=torch.float32)
            total_weight = torch.zeros((num_pixels,), device=device, dtype=torch.float32)

            # Process samples in batches to avoid memory explosion
            for sample_idx in range(num_samples):
                xi = xi_samples[sample_idx]  # [2]

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
            f0_dielectric = 0.08 * per_pixel_specular
            f0 = torch.where(
                per_pixel_metallic > 0.5,
                per_pixel_base_color,
                torch.full_like(per_pixel_base_color, 1.0) * f0_dielectric
            )

            cos_theta = torch.sum(nor * view_dir, dim=-1, keepdim=True).clamp_min(0.0)
            fresnel = f0 + (1.0 - f0) * torch.pow(1.0 - cos_theta, 5.0)

            # Apply Fresnel and specular parameter
            specular_attenuation = per_pixel_specular
            specular_ibl = fresnel * specular_attenuation * prefiltered_color
            col = col + specular_ibl
        else:
            # Fallback to ambient lighting
            col = col + per_pixel_base_color * ambient_color[None, None, None, :] * (1.0 - per_pixel_metallic)

        # Handle background
        if envmap_data is not None:
            # Compute camera ray directions for all pixels
            h, w = cam.height, cam.width
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

    # Perform single forward render pass
    with torch.no_grad():
        pred, mask = render_raster(base_color_t, roughness_t, metallic_t, specular_t)
        pred_np = pred.detach().cpu().numpy().astype(np.float32)

    # Save output
    # Match the training GUI preview tone curve: apply a power(2.2) to the shading RGB.
    # Note: save_image_u8 applies x ** (1/gamma). Setting gamma=1/2.2 results in x ** 2.2.
    save_image_u8(out_path, np.clip(pred_np, 0.0, 1.0), gamma=(1.0 / 2.2))

    print(f"\nRender complete!")
    print(f"Saved to: {out_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        raise
