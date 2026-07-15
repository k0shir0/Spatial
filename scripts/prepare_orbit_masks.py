#!/usr/bin/env python3
"""Crop, repair, and center reviewed masks for an assumed-orbit visual hull."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON array of view entries")
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--fit", type=float, default=0.84)
    parser.add_argument("--close-radius", type=int, default=9)
    args = parser.parse_args()
    entries = json.loads(args.config.read_text(encoding="utf-8"))
    if not isinstance(entries, list) or len(entries) < 3:
        parser.error("config must contain at least three view entries")
    args.output.mkdir(parents=True, exist_ok=True)
    image_output = args.output / "images"
    mask_output = args.output / "masks"
    mask_output.mkdir(exist_ok=True)
    if args.image_dir:
        image_output.mkdir(exist_ok=True)
    output_views = []

    for index, entry in enumerate(entries):
        name = str(entry["file"])
        source = cv2.imread(str(args.mask_dir / name), cv2.IMREAD_GRAYSCALE)
        if source is None:
            raise SystemExit(f"could not decode mask {name}")
        height, width = source.shape
        x0, y0, x1, y1 = [float(v) for v in entry.get("bbox", [0, 0, 1, 1])]
        left, top = max(0, round(x0 * width)), max(0, round(y0 * height))
        right, bottom = min(width, round(x1 * width)), min(height, round(y1 * height))
        clipped = np.zeros_like(source)
        clipped[top:bottom, left:right] = source[top:bottom, left:right]
        if args.close_radius:
            radius = args.close_radius
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            clipped = cv2.morphologyEx(clipped, cv2.MORPH_CLOSE, kernel)
        points = cv2.findNonZero(clipped)
        if points is None:
            raise SystemExit(f"mask became empty: {name}")
        bx, by, bw, bh = cv2.boundingRect(points)
        crop = clipped[by:by + bh, bx:bx + bw]
        scale = args.size * args.fit / max(bw, bh)
        resized = cv2.resize(crop, (max(1, round(bw * scale)), max(1, round(bh * scale))), interpolation=cv2.INTER_NEAREST)
        canvas = np.zeros((args.size, args.size), np.uint8)
        oy, ox = (args.size - resized.shape[0]) // 2, (args.size - resized.shape[1]) // 2
        canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
        output_name = f"view_{index:02d}.png"
        cv2.imwrite(str(mask_output / output_name), canvas)

        row = {"file": output_name, "yaw_degrees": float(entry["yaw_degrees"]),
               "elevation_degrees": float(entry.get("elevation_degrees", 0)),
               "roll_degrees": float(entry.get("roll_degrees", 0)), "source_mask": name}
        if args.image_dir:
            image_name = Path(name).with_suffix(".jpg").name
            image = cv2.imread(str(args.image_dir / image_name))
            if image is None:
                raise SystemExit(f"could not decode image {image_name}")
            image_crop = image[by:by + bh, bx:bx + bw]
            image_resized = cv2.resize(image_crop, (resized.shape[1], resized.shape[0]), interpolation=cv2.INTER_AREA)
            image_canvas = np.zeros((args.size, args.size, 3), np.uint8)
            image_canvas[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = image_resized
            aligned_image_name = f"view_{index:02d}.jpg"
            cv2.imwrite(str(image_output / aligned_image_name), image_canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
            row["image"] = aligned_image_name
        output_views.append(row)

    (args.output / "views.json").write_text(json.dumps(output_views, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"views": len(output_views), "size": args.size, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
