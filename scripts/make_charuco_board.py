#!/usr/bin/env python3
"""Render the printable ChArUco capture board (deterministic, no models).

The PNG embeds its DPI so printing at 100% scale reproduces the configured
physical square size, which is what makes board poses metric.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image  # noqa: E402

from local3d.charuco_pose import BoardSpec, render_board_image  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="output PNG path")
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--marker-mm", type=float, default=22.0)
    parser.add_argument("--dictionary", default="DICT_4X4_100")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if not 72 <= args.dpi <= 1200:
        parser.error("--dpi must be between 72 and 1200")
    spec = BoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_mm=args.square_mm,
        marker_mm=args.marker_mm,
        dictionary=args.dictionary,
    )
    pixels_per_square = max(20, int(round(spec.square_mm / 25.4 * args.dpi)))
    image = render_board_image(spec, pixels_per_square=pixels_per_square)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output, dpi=(args.dpi, args.dpi))

    report = {
        "output": str(args.output),
        "board": {
            "squares_x": spec.squares_x,
            "squares_y": spec.squares_y,
            "square_mm": spec.square_mm,
            "marker_mm": spec.marker_mm,
            "dictionary": spec.dictionary,
        },
        "dpi": args.dpi,
        "print_instructions": (
            f"Print at 100% scale ({args.dpi} DPI). Verify with a ruler: each "
            f"square must measure {spec.square_mm:g} mm. Tape the sheet flat; "
            "any curl bends the recovered poses."
        ),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
