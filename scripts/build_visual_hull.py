#!/usr/bin/env python3
"""Build a lightweight GLB visual hull from ordered binary orbit masks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.backends import write_glb_mesh  # noqa: E402
from local3d.visual_hull import (  # noqa: E402
    OrbitView, carve_visual_hull, load_binary_masks, occupancy_to_mesh, taubin_smooth,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("masks", type=Path, help="directory of ordered binary PNG masks")
    parser.add_argument("--output", type=Path, required=True, help="output GLB path")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--elevation", type=float, default=0.0, help="shared camera elevation in degrees")
    parser.add_argument(
        "--views-json", type=Path,
        help="optional JSON array of {file,yaw_degrees,elevation_degrees} camera entries",
    )
    parser.add_argument("--padding", type=float, default=0.04)
    parser.add_argument("--smooth-iterations", type=int, default=8)
    parser.add_argument(
        "--max-view-violations", type=int, default=0,
        help="number of inconsistent silhouettes a voxel may survive (default: 0)",
    )
    args = parser.parse_args()

    if args.views_json:
        entries = json.loads(args.views_json.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            parser.error("--views-json must contain a JSON array")
        try:
            paths = [args.masks / str(entry["file"]) for entry in entries]
            views = [
                OrbitView(
                    float(entry["yaw_degrees"]), float(entry.get("elevation_degrees", 0)),
                    float(entry.get("roll_degrees", 0)),
                )
                for entry in entries
            ]
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(f"invalid --views-json entry: {exc}")
    else:
        paths = sorted(args.masks.glob("*.png"))
        if len(paths) < 3:
            parser.error("at least three PNG masks are required")
        views = [OrbitView(index * 360.0 / len(paths), args.elevation) for index in range(len(paths))]
    if len(paths) < 3:
        parser.error("at least three PNG masks are required")
    masks = load_binary_masks(paths)
    started = time.perf_counter()
    occupancy = carve_visual_hull(
        masks, views, resolution=args.resolution, padding=args.padding,
        max_view_violations=args.max_view_violations,
    )
    vertices, faces = occupancy_to_mesh(occupancy)
    vertices = taubin_smooth(vertices, faces, iterations=args.smooth_iterations)
    write_glb_mesh(
        args.output, vertices.tolist(), faces.tolist(), generator="local3d-visual-hull",
        extras={
            "scale": "ambiguous", "source": "ordered_silhouettes",
            "geometry": "silhouette_intersection_with_unobserved_concavities",
        },
    )
    report = {
        "output": str(args.output), "mask_count": len(paths), "resolution": args.resolution,
        "occupied_voxels": int(occupancy.sum()), "vertices": len(vertices), "triangles": len(faces),
        "smooth_iterations": args.smooth_iterations,
        "max_view_violations": args.max_view_violations,
        "elapsed_seconds": round(time.perf_counter() - started, 3), "scale": "ambiguous",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
