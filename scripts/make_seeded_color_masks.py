#!/usr/bin/env python3
"""Create model-free object masks from a persistent normalized color seed.

This is intended for mostly uniform objects on a contrasting background, such
as plush toys.  It learns color independently per frame from a small seed patch,
keeps only the connected region touching that seed, fills internal details, and
fails on implausible coverage or abrupt temporal changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def fill_holes(mask: np.ndarray) -> np.ndarray:
    flood = mask.copy()
    padded = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, None, (0, 0), 255)
    background = padded[1:-1, 1:-1]
    return cv2.bitwise_or(mask, cv2.bitwise_not(background))


def component_at_seed(mask: np.ndarray, x: int, y: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    identifier = int(labels[y, x])
    if identifier == 0:
        candidates = []
        for label in range(1, count):
            left, top, width, height, area = stats[label]
            cx = min(max(x, left), left + width - 1)
            cy = min(max(y, top), top + height - 1)
            distance = (cx - x) ** 2 + (cy - y) ** 2
            candidates.append((distance, -int(area), label))
        if not candidates or min(candidates)[0] > max(mask.shape) ** 2 * 0.01:
            return np.zeros_like(mask)
        identifier = min(candidates)[2]
    return np.where(labels == identifier, 255, 0).astype(np.uint8)


def make_mask(image: np.ndarray, seed_x: float, seed_y: float, threshold: float) -> tuple[np.ndarray, dict]:
    height, width = image.shape[:2]
    x, y = int(seed_x * width), int(seed_y * height)
    radius = max(5, round(min(width, height) * 0.018))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    patch = lab[max(0, y - radius):min(height, y + radius + 1), max(0, x - radius):min(width, x + radius + 1)]
    median = np.median(patch.reshape(-1, 3), axis=0)
    distance = np.linalg.norm(lab - median, axis=2)
    mask = np.where(distance <= threshold, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = component_at_seed(mask, x, y)
    mask = fill_holes(mask)
    # Preserve wispy silhouette detail while removing one-pixel compression noise.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    area = cv2.countNonZero(mask) / mask.size
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None
    perimeter = cv2.arcLength(contour, True) if contour is not None else 0
    return mask, {
        "seed_lab": [round(float(v), 3) for v in median],
        "area_fraction": round(float(area), 6),
        "perimeter_pixels": round(float(perimeter), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-output", type=Path)
    parser.add_argument("--seed", default="0.5,0.45", help="normalized x,y")
    parser.add_argument("--threshold", type=float, default=28.0, help="Lab distance")
    parser.add_argument("--minimum-area", type=float, default=0.03)
    parser.add_argument("--maximum-area", type=float, default=0.55)
    parser.add_argument("--max-area-ratio", type=float, default=2.2)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        seed_x, seed_y = (float(value) for value in args.seed.split(","))
    except ValueError:
        parser.error("--seed must be normalized x,y")
    if not 0 <= seed_x <= 1 or not 0 <= seed_y <= 1:
        parser.error("--seed values must be between 0 and 1")

    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        parser.error("input contains no images")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.debug_output:
        args.debug_output.mkdir(parents=True, exist_ok=True)

    rows = []
    previous_area: float | None = None
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            rows.append({"image": path.name, "accepted": False, "reason": "decode_failed"})
            continue
        mask, metrics = make_mask(image, seed_x, seed_y, args.threshold)
        area = metrics["area_fraction"]
        accepted = args.minimum_area <= area <= args.maximum_area
        reason = None if accepted else "unsafe_area"
        if accepted and previous_area is not None:
            ratio = max(area, previous_area) / max(min(area, previous_area), 1e-9)
            if ratio > args.max_area_ratio:
                accepted, reason = False, "abrupt_area_change"
        if accepted:
            previous_area = area
            cv2.imwrite(str(args.output / f"{path.stem}.png"), mask)
        if args.debug_output:
            overlay = image.copy()
            overlay[mask > 0] = (0.65 * overlay[mask > 0] + 0.35 * np.array([40, 220, 40])).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 255), 2)
            cv2.putText(overlay, "ACCEPT" if accepted else f"REJECT {reason}", (20, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 220, 30) if accepted else (0, 0, 255), 2)
            cv2.imwrite(str(args.debug_output / f"{path.stem}.jpg"), overlay)
        rows.append({"image": path.name, "mask": f"{path.stem}.png", "accepted": accepted, "reason": reason, **metrics})
        print(("ACCEPT" if accepted else "REJECT"), path.name, f"area={area:.3f}", reason or "")

    report = {"backend": "seeded_lab_color", "threshold": args.threshold, "seed": [seed_x, seed_y],
              "accepted": sum(row["accepted"] for row in rows), "rejected": sum(not row["accepted"] for row in rows),
              "frames": rows}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["accepted"] >= 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
