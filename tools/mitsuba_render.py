#!/usr/bin/env python3
"""Render a Mitsuba scene from an XML file.

This is meant to be called from the Rust egui tool (spawned as a subprocess).

Examples:
  .venv/bin/python tools/mitsuba_render.py --scene scenes/cbox.xml --variant scalar_rgb --spp 64 --out renders/preview.png

Notes:
- For LDR formats (png/jpg), we convert to 8-bit sRGB (includes gamma).
- For HDR formats (exr/hdr/pfm), we write linear values (no gamma).
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="Path to Mitsuba XML scene")
    parser.add_argument("--variant", default="scalar_rgb", help="Mitsuba variant (e.g., scalar_rgb, llvm_ad_rgb)")
    parser.add_argument("--spp", type=int, default=64, help="Samples per pixel")
    parser.add_argument("--out", required=True, help="Output image path (e.g., renders/preview.png)")
    args = parser.parse_args()

    # Reduce noise; the GUI will show stdout/stderr anyway.
    os.environ.setdefault("DRJIT_LOG_LEVEL", "warn")

    try:
        import mitsuba as mi
    except Exception as exc:  # pragma: no cover
        print(f"Failed to import mitsuba: {exc}", file=sys.stderr)
        return 2

    try:
        mi.set_variant(args.variant)
    except Exception as exc:
        print(f"Failed to set variant '{args.variant}': {exc}", file=sys.stderr)
        return 3

    scene = mi.load_file(args.scene)
    image = mi.render(scene, spp=args.spp)

    out_lower = args.out.lower()
    if out_lower.endswith((".exr", ".hdr", ".pfm")):
        # HDR output: keep linear values.
        mi.util.write_bitmap(args.out, image)
    else:
        # LDR output: convert to display-friendly 8-bit sRGB (includes gamma).
        bmp = mi.util.convert_to_bitmap(image, uint8_srgb=True)
        bmp.write(args.out)
    print(f"OK: wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
