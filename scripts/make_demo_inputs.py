#!/usr/bin/env python3
"""Generate privacy-safe, deterministic images for the public phone demo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


WIDTH = 640
HEIGHT = 1280


def _write_png(path: Path, image: np.ndarray) -> None:
    parameters = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    if not cv2.imwrite(str(path), image, parameters):
        raise RuntimeError(f"could not write {path}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _front_image() -> np.ndarray:
    y = np.arange(HEIGHT, dtype=np.float32)[:, None]
    x = np.arange(WIDTH, dtype=np.float32)[None, :]
    glow = np.exp(
        -(
            ((x - WIDTH * 0.58) / (WIDTH * 0.52)) ** 2
            + ((y - HEIGHT * 0.42) / (HEIGHT * 0.48)) ** 2
        )
    )
    image = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[..., 0] = np.clip(18 + glow * 74 + y / HEIGHT * 20, 0, 255)
    image[..., 1] = np.clip(15 + glow * 47 + y / HEIGHT * 12, 0, 255)
    image[..., 2] = np.clip(23 + glow * 30, 0, 255)

    cv2.rectangle(image, (18, 18), (WIDTH - 19, HEIGHT - 19), (6, 8, 12), 18)
    cv2.putText(
        image,
        "09:41",
        (92, 440),
        cv2.FONT_HERSHEY_DUPLEX,
        2.8,
        (226, 232, 241),
        6,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "SPATIAL",
        (156, 525),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.35,
        (176, 194, 218),
        3,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (68, 925), (WIDTH - 69, 1045), (48, 59, 78), -1)
    cv2.putText(
        image,
        "DETERMINISTIC BUILD",
        (104, 998),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (222, 229, 238),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(image, (WIDTH // 2, 1175), 34, (214, 222, 235), 3, cv2.LINE_AA)
    cv2.line(image, (WIDTH // 2 - 46, 1234), (WIDTH // 2 + 46, 1234), (230, 234, 240), 4)
    return image


def _back_image() -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), (136, 124, 112), dtype=np.uint8)
    cv2.rectangle(image, (18, 18), (WIDTH - 19, HEIGHT - 19), (168, 158, 145), 18)
    cv2.circle(image, (WIDTH // 2, HEIGHT // 2 + 20), 220, (199, 212, 218), 34, cv2.LINE_AA)
    cv2.rectangle(image, (296, 866), (344, 1036), (199, 212, 218), -1)
    cv2.rectangle(image, (62, 65), (340, 405), (167, 153, 132), -1)
    cv2.circle(image, (190, 160), 58, (16, 11, 7), -1, cv2.LINE_AA)
    cv2.circle(image, (190, 305), 58, (16, 11, 7), -1, cv2.LINE_AA)
    cv2.circle(image, (190, 160), 69, (182, 171, 154), 10, cv2.LINE_AA)
    cv2.circle(image, (190, 305), 69, (182, 171, 154), 10, cv2.LINE_AA)
    cv2.circle(image, (295, 232), 20, (187, 217, 224), -1, cv2.LINE_AA)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("examples/demo"))
    arguments = parser.parse_args()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    front = output / "front.png"
    back = output / "back.png"
    _write_png(front, _front_image())
    _write_png(back, _back_image())
    report = {
        "schema_version": 1,
        "generated": True,
        "randomness": False,
        "dimensions_px": [WIDTH, HEIGHT],
        "files": {
            front.name: {"sha256": _sha256(front), "bytes": front.stat().st_size},
            back.name: {"sha256": _sha256(back), "bytes": back.stat().st_size},
        },
    }
    report_path = output / "manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
