#!/usr/bin/env python3
"""Minimal differentiable rendering demo: optimize a BSDF parameter.

Why this exists
---------------
If you only call `mi.render()` once, a differentiable variant (e.g. `llvm_ad_rgb`)
looks just like a normal path tracer: it produces an image.

The *difference* is that you can additionally compute gradients of an objective
(loss) w.r.t. scene parameters (material / geometry / lights), and then run
optimization (inverse rendering).

This script:
- Loads a Mitsuba XML scene.
- Picks a `reflectance` parameter (auto-detect or --param).
- Renders a target image with a known reflectance.
- Starts from a different reflectance and runs gradient descent to match target.

Notes
-----
- This uses the standard rendering equation / BSDFs. No neural networks.
- The optimization is noisy (Monte Carlo), so keep steps small and use enough spp.

Example
-------
.venv/bin/python tools/mitsuba_diff_optimize_reflectance.py \
  --scene scenes/cbox.xml \
  --param "reflectance" \
  --steps 20 --lr 0.3 \
  --out renders/diff_opt

"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


def _pick_param_key(keys: list[str], pattern: str | None) -> str | None:
    if not keys:
        return None

    if pattern:
        rx = re.compile(pattern)
        for k in keys:
            if rx.search(k):
                return k
        return None

    # Heuristic: prefer reflectance-like values.
    preferred = [
        r"reflectance\\.value$",
        r"reflectance$",
        r"base_color\\.value$",
        r"albedo\\.value$",
    ]
    for p in preferred:
        rx = re.compile(p)
        for k in keys:
            if rx.search(k):
                return k

    # Fallback: any key that mentions reflectance.
    for k in keys:
        if "reflectance" in k:
            return k

    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Path to Mitsuba XML scene")
    parser.add_argument("--variant", default="llvm_ad_rgb", help="Differentiable variant")
    parser.add_argument(
        "--param",
        default=None,
        help=(
            "Regex to select a parameter key from mi.traverse(scene).keys(). "
            "If omitted, we auto-detect a reflectance-like parameter. "
            "Examples: 'reflectance', 'object.*reflectance', 'bsdf.*reflectance\\.value$'"
        ),
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--spp_target", type=int, default=64)
    parser.add_argument("--spp_iter", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="renders/diff_opt", help="Output directory")
    args = parser.parse_args()

    os.environ.setdefault("DRJIT_LOG_LEVEL", "warn")

    try:
        import mitsuba as mi
        import drjit as dr
    except Exception as exc:
        print(f"Failed to import mitsuba/drjit: {exc}", file=sys.stderr)
        return 2

    try:
        mi.set_variant(args.variant)
    except Exception as exc:
        print(f"Failed to enable variant '{args.variant}': {exc}", file=sys.stderr)
        return 3

    scene = mi.load_file(args.scene)
    params = mi.traverse(scene)

    keys = sorted([str(k) for k in params.keys()])
    key = _pick_param_key(keys, args.param)
    if key is None:
        print("Could not find a parameter to optimize.")
        print("Available parameter keys:")
        for k in keys:
            print("  ", k)
        return 4

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Render a target image with a known reflectance.
    target_color = mi.Color3f(0.10, 0.65, 0.90)
    params[key] = target_color
    params.update()
    target = mi.render(scene, params=params, spp=args.spp_target, seed=args.seed)
    mi.util.convert_to_bitmap(target, uint8_srgb=True).write(str(out_dir / "target.png"))

    # Initialize from a different value and optimize.
    x = mi.Color3f(0.85, 0.20, 0.20)
    x = dr.clip(x, 0.0, 1.0)
    dr.enable_grad(x)
    params[key] = x
    params.update()

    print(f"Optimizing parameter: {key}")
    print(f"Target reflectance: {target_color}")
    print(f"Init reflectance:   {x}")

    for it in range(args.steps):
        img = mi.render(scene, params=params, spp=args.spp_iter, seed=args.seed + 1 + it)
        loss = dr.mean(dr.square(img - target))

        dr.backward(loss)
        g = dr.grad(x)

        # Gradient descent step.
        x = dr.detach(dr.clip(x - args.lr * g, 0.0, 1.0))
        dr.enable_grad(x)
        params[key] = x
        params.update()

        print(f"iter {it:03d}  loss={loss}  x={x}")

        if (it + 1) % 5 == 0 or it == args.steps - 1:
            mi.util.convert_to_bitmap(img, uint8_srgb=True).write(str(out_dir / f"iter_{it:03d}.png"))

    final = mi.render(scene, params=params, spp=args.spp_target, seed=args.seed + 999)
    mi.util.convert_to_bitmap(final, uint8_srgb=True).write(str(out_dir / "final.png"))

    print(f"OK: wrote {out_dir}/target.png and {out_dir}/final.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
