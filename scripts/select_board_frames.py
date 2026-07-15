#!/usr/bin/env python3
"""Select capture frames by measured ChArUco pose coverage (zero-ML).

For every frame in a directory this estimates the printed board's pose with
classical OpenCV, gates on corner support and reprojection error, then picks
keyframes that spread measured azimuths around the orbit.  "Good frame"
becomes a measurement — corners detected, residual error, viewpoint novelty —
with no learned model and no heuristic scoring.

Outputs: per-frame poses.json, selected keyframes copied into keyframes/,
axis-overlay review images, and a summary report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from local3d.charuco_pose import (  # noqa: E402
    BoardSpec,
    approximate_camera_matrix,
    camera_azimuth_elevation,
    estimate_frame_pose,
    select_by_pose_coverage,
)

FRAME_SUFFIXES = (".png", ".jpg", ".jpeg")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="directory of capture frames")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16, help="keyframes to select")
    parser.add_argument("--min-azimuth-gap", type=float, default=12.0)
    parser.add_argument("--squares-x", type=int, default=7)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--marker-mm", type=float, default=22.0)
    parser.add_argument("--dictionary", default="DICT_4X4_100")
    parser.add_argument("--min-corners", type=int, default=6)
    parser.add_argument("--max-rms-px", type=float, default=3.0)
    parser.add_argument(
        "--fov-degrees", type=float, default=65.0,
        help="assumed horizontal FOV for approximate intrinsics (relative poses only)",
    )
    parser.add_argument(
        "--camera-json", type=Path,
        help="optional JSON with fx, fy, cx, cy (and optional dist) for calibrated intrinsics",
    )
    args = parser.parse_args()

    spec = BoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_mm=args.square_mm,
        marker_mm=args.marker_mm,
        dictionary=args.dictionary,
    )
    paths = sorted(path for path in args.frames.iterdir() if path.suffix.lower() in FRAME_SUFFIXES)
    if len(paths) < 3:
        parser.error(f"need at least three frames in {args.frames}")

    first = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    if first is None:
        parser.error(f"could not decode {paths[0]}")
    height, width = first.shape
    distortion = None
    if args.camera_json:
        intrinsics = json.loads(args.camera_json.read_text(encoding="utf-8"))
        camera = np.array(
            [
                [float(intrinsics["fx"]), 0.0, float(intrinsics["cx"])],
                [0.0, float(intrinsics["fy"]), float(intrinsics["cy"])],
                [0.0, 0.0, 1.0],
            ]
        )
        if "dist" in intrinsics:
            distortion = np.asarray(intrinsics["dist"], dtype=np.float64)
        intrinsics_source = f"calibrated ({args.camera_json.name})"
        scale_note = "translations are metric if the calibration and printed square size are correct"
    else:
        camera = approximate_camera_matrix(width, height, args.fov_degrees)
        intrinsics_source = f"approximate ({args.fov_degrees:g} degree horizontal FOV assumed)"
        scale_note = "poses are relative orbit geometry; do not treat translations as metric"

    overlays_dir = args.output / "pose_overlays"
    keyframes_dir = args.output / "keyframes"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    poses = []
    per_frame = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            poses.append(None)
            per_frame.append({"frame": path.name, "pose": None, "reason": "could not decode"})
            continue
        pose = estimate_frame_pose(
            image,
            spec,
            camera,
            distortion,
            min_corners=args.min_corners,
            max_rms_px=args.max_rms_px,
        )
        poses.append(pose)
        if pose is None:
            per_frame.append({"frame": path.name, "pose": None, "reason": "gates failed (board missing, occluded, or high residual)"})
            continue
        azimuth, elevation = camera_azimuth_elevation(pose)
        per_frame.append(
            {
                "frame": path.name,
                "pose": pose.as_json(),
                "azimuth_degrees": round(azimuth, 3),
                "elevation_degrees": round(elevation, 3),
            }
        )
        overlay = image.copy()
        cv2.drawFrameAxes(
            overlay,
            camera,
            distortion if distortion is not None else np.zeros((5, 1)),
            np.asarray(pose.rotation_rodrigues),
            np.asarray(pose.translation_mm),
            2.0 * spec.square_mm,
        )
        cv2.imwrite(str(overlays_dir / f"{path.stem}_axes.png"), overlay)

    selected = select_by_pose_coverage(
        poses, count=args.count, min_azimuth_gap_degrees=args.min_azimuth_gap
    )
    for index in selected:
        shutil.copy2(paths[index], keyframes_dir / paths[index].name)

    posed_count = sum(1 for pose in poses if pose is not None)
    selected_azimuths = sorted(
        round(camera_azimuth_elevation(poses[index])[0], 1) for index in selected
    )
    report = {
        "tool": "select_board_frames (OpenCV ChArUco, deterministic, zero-ML)",
        "board": {
            "squares_x": spec.squares_x,
            "squares_y": spec.squares_y,
            "square_mm": spec.square_mm,
            "marker_mm": spec.marker_mm,
            "dictionary": spec.dictionary,
        },
        "intrinsics_source": intrinsics_source,
        "scale_note": scale_note,
        "frame_count": len(paths),
        "posed_frame_count": posed_count,
        "selected_count": len(selected),
        "selected_frames": [paths[index].name for index in selected],
        "selected_azimuths_degrees": selected_azimuths,
        "frames": per_frame,
    }
    (args.output / "poses.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("frame_count", "posed_frame_count", "selected_count", "selected_azimuths_degrees")},
            indent=2,
        )
    )
    if posed_count == 0:
        print("no frame passed the pose gates; this capture cannot be used for board-based reconstruction", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
