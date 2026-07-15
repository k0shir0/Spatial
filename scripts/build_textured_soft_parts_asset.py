#!/usr/bin/env python3
"""Bake reviewed video detail onto a fitted soft-parts mesh."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.texture_bake import bake_textured_soft_parts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="checked-in deterministic texture config")
    parser.add_argument("--output", type=Path, required=True, help="self-contained GLB destination")
    parser.add_argument(
        "--no-usdz",
        action="store_true",
        help="skip the parallel USD/Quick Look package",
    )
    args = parser.parse_args()
    report = bake_textured_soft_parts(
        args.config,
        args.output,
        write_usdz=not args.no_usdz,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
