#!/usr/bin/env python3
"""Create local RGBA object cutouts from point-prompted images.

The prompt file maps each input filename to normalized positive and negative
points.  Inference is entirely local through rembg's cached SAM ONNX model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--padding", type=float, default=0.12)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        help="Fill internal alpha holes, useful for transparent windows in otherwise solid objects.",
    )
    return parser.parse_args()


def point_pixels(points: list[list[float]], width: int, height: int) -> list[list[int]]:
    return [[round(x * width), round(y * height)] for x, y in points]


def square_crop(image: Image.Image, padding: float, size: int) -> tuple[Image.Image, list[int]]:
    alpha = np.asarray(image.getchannel("A"))
    ys, xs = np.nonzero(alpha > 8)
    if len(xs) == 0:
        raise ValueError("segmentation produced an empty mask")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    side = max(x1 - x0, y1 - y0)
    side = max(2, round(side * (1.0 + 2.0 * padding)))
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    left, top = round(cx - side / 2), round(cy - side / 2)
    right, bottom = left + side, top + side

    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    src_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    dst_xy = (max(0, -left), max(0, -top))
    canvas.alpha_composite(image.crop(src_box), dst_xy)
    return canvas.resize((size, size), Image.Resampling.LANCZOS), [left, top, right, bottom]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = json.loads(args.prompts.read_text())
    session = new_session("sam")
    report: dict[str, object] = {"session": "sam", "images": []}

    for name, config in prompts.items():
        source = args.input_dir / name
        image = Image.open(source).convert("RGB")
        positives = point_pixels(config.get("positive", []), image.width, image.height)
        negatives = point_pixels(config.get("negative", []), image.width, image.height)
        if not positives:
            raise ValueError(f"{name}: at least one positive point is required")

        sam_prompt: list[dict[str, object]] = []
        if box := config.get("box"):
            x0, y0 = point_pixels([box[:2]], image.width, image.height)[0]
            x1, y1 = point_pixels([box[2:]], image.width, image.height)[0]
            sam_prompt.append({"type": "rectangle", "label": 1, "data": [x0, y0, x1, y1]})
        sam_prompt.extend({"type": "point", "label": 1, "data": point} for point in positives)
        sam_prompt.extend({"type": "point", "label": 0, "data": point} for point in negatives)

        rgba = remove(image, session=session, sam_prompt=sam_prompt).convert("RGBA")
        if args.fill_holes:
            raw_mask = np.asarray(rgba.getchannel("A")) > 8
            labels, count = ndimage.label(raw_mask)
            if count:
                sizes = ndimage.sum(raw_mask, labels, range(1, count + 1))
                raw_mask = labels == (int(np.argmax(sizes)) + 1)
            filled_mask = ndimage.binary_fill_holes(raw_mask)
            rgba = image.convert("RGBA")
            rgba.putalpha(Image.fromarray((filled_mask * 255).astype(np.uint8), mode="L"))
        isolated, crop = square_crop(rgba, args.padding, args.size)
        isolated_path = args.output_dir / name
        isolated.save(isolated_path)

        white = Image.new("RGBA", isolated.size, (255, 255, 255, 255))
        white.alpha_composite(isolated)
        white.convert("RGB").save(args.output_dir / f"{Path(name).stem}_white.jpg", quality=96)

        alpha = np.asarray(isolated.getchannel("A"))
        report["images"].append(
            {
                "source": str(source),
                "output": str(isolated_path),
                "crop_xyxy": crop,
                "foreground_fraction": round(float((alpha > 8).mean()), 5),
                "positive_points_px": positives,
                "negative_points_px": negatives,
            }
        )
        print(f"isolated {name} -> {isolated_path}", flush=True)

    (args.output_dir / "segmentation_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
