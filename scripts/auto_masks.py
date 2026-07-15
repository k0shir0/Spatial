#!/usr/bin/env python3
"""Automatic review-ready object masks for extracted frames (CPU-only).

Runs a small local matting model (u2netp by default, 4.6 MB) through rembg on
onnxruntime's CPU provider and writes, for every frame: a binary mask, an
overlay for human review, and per-frame quality metrics with fail-closed
flags.  Nothing here dispatches reconstruction; the output is review input.

Requires the ``segmentation`` extra.  The model file is fetched once into the
local rembg cache and hash-recorded in the report; runs are network-free
afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FRAME_SUFFIXES = (".png", ".jpg", ".jpeg")
MODELS = ("u2netp", "u2net", "isnet-general-use")


def frame_metrics(binary: np.ndarray) -> dict[str, object]:
    height, width = binary.shape
    coverage = float(binary.mean())
    border = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    border_touch = float(border.mean())
    component_count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    largest_fraction = 0.0
    if component_count > 1:
        sizes = np.bincount(labels.ravel())[1:]
        largest_fraction = float(sizes.max() / max(int(sizes.sum()), 1))
    flags = []
    if coverage < 0.02:
        flags.append("object_too_small_or_missed")
    if coverage > 0.85:
        flags.append("mask_covers_most_of_frame")
    if border_touch > 0.10:
        flags.append("mask_touches_frame_border")
    if component_count > 2 and largest_fraction < 0.90:
        flags.append("mask_is_fragmented")
    return {
        "coverage": round(coverage, 4),
        "border_touch": round(border_touch, 4),
        "components": int(component_count - 1),
        "largest_component_fraction": round(largest_fraction, 4),
        "flags": flags,
    }


def overlay_image(frame: np.ndarray, binary: np.ndarray) -> np.ndarray:
    dimmed = (frame * 0.35).astype(np.uint8)
    result = np.where(binary[..., None] > 0, frame, dimmed)
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(result, contours, -1, (80, 220, 80), 2)
    return result


def contact_sheet(overlays: list[np.ndarray], names: list[str], columns: int = 4, tile_width: int = 480) -> np.ndarray:
    tiles = []
    for image, name in zip(overlays, names):
        scale = tile_width / image.shape[1]
        tile = cv2.resize(image, (tile_width, int(round(image.shape[0] * scale))))
        cv2.putText(
            tile, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            tile, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        tiles.append(tile)
    height = max(tile.shape[0] for tile in tiles)
    tiles = [
        cv2.copyMakeBorder(tile, 0, height - tile.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        for tile in tiles
    ]
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.full_like(row[0], 24))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def model_provenance(session: object) -> dict[str, object]:
    cache = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))
    name = getattr(session, "model_name", None) or "unknown"
    candidate = cache / f"{name}.onnx"
    record: dict[str, object] = {"model": name}
    if candidate.is_file():
        record["model_file"] = str(candidate)
        record["model_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
        record["model_bytes"] = candidate.stat().st_size
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="directory of extracted frames")
    parser.add_argument("--output", type=Path, required=True, help="output directory")
    parser.add_argument("--model", choices=MODELS, default="u2netp")
    parser.add_argument("--threshold", type=int, default=128, help="alpha threshold for the binary mask")
    parser.add_argument("--max-frames", type=int, default=0, help="limit processed frames (0 = all)")
    args = parser.parse_args()

    cv2.setNumThreads(1)
    try:
        from rembg import new_session, remove
        from PIL import Image
    except ImportError:
        parser.error("rembg is not installed; install the segmentation extra: pip install -e '.[segmentation]'")

    paths = sorted(
        path for path in args.frames.iterdir() if path.suffix.lower() in FRAME_SUFFIXES
    )
    if args.max_frames:
        paths = paths[: args.max_frames]
    if not paths:
        parser.error(f"no frames found in {args.frames}")

    masks_dir = args.output / "masks"
    overlays_dir = args.output / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    session = new_session(args.model)
    frames_report = []
    overlays = []
    names = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            frames_report.append({"frame": path.name, "error": "could not decode"})
            continue
        alpha = remove(
            Image.open(path),
            session=session,
            only_mask=True,
            post_process_mask=False,
        )
        mask = np.asarray(alpha, dtype=np.uint8)
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        binary = (mask >= args.threshold).astype(np.uint8)
        metrics = frame_metrics(binary)
        cv2.imwrite(str(masks_dir / f"{path.stem}_mask.png"), binary * 255)
        overlay = overlay_image(frame, binary)
        cv2.imwrite(str(overlays_dir / f"{path.stem}_overlay.png"), overlay)
        overlays.append(overlay)
        names.append(path.stem)
        frames_report.append({"frame": path.name, **metrics})

    sheet = contact_sheet(overlays, names)
    sheet_path = args.output / "mask_contact_sheet.png"
    cv2.imwrite(str(sheet_path), sheet)

    flagged = [entry for entry in frames_report if entry.get("flags")]
    report = {
        "tool": "auto_masks (rembg on onnxruntime CPU, single-threaded)",
        **model_provenance(session),
        "threshold": args.threshold,
        "frame_count": len(frames_report),
        "flagged_count": len(flagged),
        "contact_sheet": str(sheet_path),
        "frames": frames_report,
        "review_note": (
            "Masks are advisory review input. Flagged frames need operator "
            "correction or recapture before any reconstruction step."
        ),
    }
    report_path = args.output / "auto_masks_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model", "frame_count", "flagged_count", "contact_sheet")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
