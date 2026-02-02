#!/usr/bin/env python3
"""Differentiable rendering smoketest.

This script exists mainly to validate that the chosen AD backend works on this
machine (e.g. `llvm_ad_rgb` or `cuda_ad_rgb`).

Right now, your environment reports:
  ImportError: libLLVM.dylib could not be found
which means `llvm_ad_*` variants cannot run until LLVM is discoverable.

We still keep this script so the egui tool can run it and display a friendly
error message.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--variant", default="llvm_ad_rgb")
    parser.add_argument("--spp", type=int, default=4)
    args = parser.parse_args()

    os.environ.setdefault("DRJIT_LOG_LEVEL", "warn")

    try:
        import mitsuba as mi
    except Exception as exc:
        print(f"Failed to import mitsuba: {exc}", file=sys.stderr)
        return 2

    try:
        mi.set_variant(args.variant)
    except Exception as exc:
        print(f"Failed to enable differentiable variant '{args.variant}'.", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(
            "\nHint: For llvm_ad_* on macOS, ensure libLLVM is installed and set DRJIT_LIBLLVM_PATH.\n"
            "Example (Homebrew):\n"
            "  export DRJIT_LIBLLVM_PATH=/opt/homebrew/opt/llvm/lib/libLLVM.dylib\n"
            "or:\n"
            "  export DRJIT_LIBLLVM_PATH=/usr/local/opt/llvm/lib/libLLVM.dylib\n",
            file=sys.stderr,
        )
        return 3

    scene = mi.load_file(args.scene)
    _ = mi.render(scene, spp=args.spp)
    print("OK: rendered once with differentiable variant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
