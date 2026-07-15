#!/usr/bin/env python3
"""Build a lightweight fitted soft-object GLB from primitive parts JSON."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.backends import write_glb_material_parts  # noqa: E402
from local3d.soft_parts import (  # noqa: E402
    combine_parts,
    ellipsoid_mesh,
    superellipsoid_mesh,
    tube_mesh,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    started = time.perf_counter()
    parts = []
    for item in config["parts"]:
        if item["type"] == "ellipsoid":
            vertices, faces, normals = ellipsoid_mesh(
                item["center"], item["radii"], euler_degrees=item.get("rotation", [0, 0, 0]),
                rings=int(item.get("rings", 20)), segments=int(item.get("segments", 32)),
            )
        elif item["type"] == "superellipsoid":
            vertices, faces, normals = superellipsoid_mesh(
                item["center"],
                item["radii"],
                vertical_exponent=float(item.get("vertical_exponent", 0.65)),
                horizontal_exponent=float(item.get("horizontal_exponent", 0.75)),
                euler_degrees=item.get("rotation", [0, 0, 0]),
                rings=int(item.get("rings", 24)),
                segments=int(item.get("segments", 40)),
            )
        elif item["type"] == "tube":
            vertices, faces, normals = tube_mesh(
                item["points"], float(item["radius"]), segments=int(item.get("segments", 12))
            )
        else:
            raise SystemExit(f"unsupported part type: {item['type']}")
        parts.append((vertices, faces, normals, item["color_rgba"]))
    vertices, faces, normals, colors = combine_parts(parts)
    material_parts = [
        (vertices.tolist(), faces.tolist(), normals.tolist(), color)
        for vertices, faces, normals, color in parts
    ]
    write_glb_material_parts(
        args.output, material_parts, generator="local3d-soft-parts",
        extras={
            "source": config.get("source"), "method": "operator_fitted_soft_parts",
            "geometry": "inferred_from_observed_front_side_proportions", "scale": "ambiguous",
        },
    )
    report = {
        "output": str(args.output), "parts": len(parts), "vertices": len(vertices),
        "triangles": len(faces), "bytes": args.output.stat().st_size,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "geometry": "inferred_from_observed_front_side_proportions",
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
