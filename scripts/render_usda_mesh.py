#!/usr/bin/env python3
"""Render a small USD mesh with a CPU-only orthographic z-buffer.

This intentionally uses no 3D renderer, GPU API, model runtime, or network access.
It understands the simple USDA mesh arrays emitted by RealityKit Object Capture.
Binary USD/USDZ inputs are converted to temporary USDA text with Apple's usdcat.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _array(text: str, declaration: str) -> str:
    match = re.search(rf"\b{re.escape(declaration)}\s*=\s*\[(.*?)\]", text, re.S)
    if not match:
        raise ValueError(f"USD mesh has no {declaration!r} array")
    return match.group(1)


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 vertices and int32 triangles from USDA, USD, or USDZ."""
    suffix = path.suffix.lower()
    if suffix in {".usda", ".txt"}:
        text = path.read_text(encoding="utf-8")
    elif suffix == ".usd":
        raw = path.read_bytes()
        if raw.lstrip().startswith(b"#usda"):
            text = raw.decode("utf-8")
        else:
            text = _convert_with_usdcat(path)
    elif suffix in {".usdz", ".usdc"}:
        text = _convert_with_usdcat(path)
    else:
        raise ValueError(f"unsupported input extension: {suffix}")

    point_values = [float(value) for value in NUMBER.findall(_array(text, "point3f[] points"))]
    if len(point_values) % 3:
        raise ValueError("point array length is not divisible by three")
    points = np.asarray(point_values, dtype=np.float64).reshape(-1, 3)

    counts = [int(value) for value in NUMBER.findall(_array(text, "int[] faceVertexCounts"))]
    indices = [int(value) for value in NUMBER.findall(_array(text, "int[] faceVertexIndices"))]
    if sum(counts) != len(indices):
        raise ValueError("face counts do not match the index array")

    triangles: list[tuple[int, int, int]] = []
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        if count < 3:
            continue
        for index in range(1, count - 1):
            triangles.append((face[0], face[index], face[index + 1]))
    faces = np.asarray(triangles, dtype=np.int32)
    if faces.size == 0 or faces.max() >= len(points) or faces.min() < 0:
        raise ValueError("mesh contains no valid triangles")
    return points, faces


def _convert_with_usdcat(path: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="cpu-usd-preview-") as temp_dir:
        output = Path(temp_dir) / "mesh.usda"
        command = ["/usr/bin/usdcat", str(path), "--flatten", "-o", str(output)]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(f"usdcat failed: {detail}")
        return output.read_text(encoding="utf-8")


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def mesh_metrics(points: np.ndarray, faces: np.ndarray) -> dict[str, object]:
    edges: Counter[tuple[int, int]] = Counter()
    dsu = DisjointSet(len(points))
    for a, b, c in faces:
        for left, right in ((int(a), int(b)), (int(b), int(c)), (int(c), int(a))):
            edges[tuple(sorted((left, right)))] += 1
            dsu.union(left, right)

    used = set(int(value) for value in faces.ravel())
    components = len({dsu.find(index) for index in used})
    boundary = sum(count == 1 for count in edges.values())
    nonmanifold = sum(count > 2 for count in edges.values())
    boundary_adjacency: dict[int, set[int]] = {}
    for (left, right), count in edges.items():
        if count == 1:
            boundary_adjacency.setdefault(left, set()).add(right)
            boundary_adjacency.setdefault(right, set()).add(left)
    boundary_groups = 0
    visited: set[int] = set()
    for start in boundary_adjacency:
        if start in visited:
            continue
        boundary_groups += 1
        stack = [start]
        visited.add(start)
        while stack:
            for neighbor in boundary_adjacency[stack.pop()]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
    dimensions = np.ptp(points, axis=0)
    area = 0.5 * np.linalg.norm(
        np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]),
        axis=1,
    ).sum()
    return {
        "vertices": int(len(points)),
        "triangles": int(len(faces)),
        "connected_components": components,
        "boundary_edges": boundary,
        "boundary_loops": boundary_groups,
        "boundary_edge_fraction": boundary / len(edges) if edges else 0.0,
        "nonmanifold_edges": nonmanifold,
        "watertight_topology": boundary == 0 and nonmanifold == 0,
        "dimensions_m": {"x": float(dimensions[0]), "y": float(dimensions[1]), "z": float(dimensions[2])},
        "surface_area_m2": float(area),
    }


def camera_basis(direction: np.ndarray) -> np.ndarray:
    forward = direction.astype(np.float64)
    forward /= np.linalg.norm(forward)
    reference_up = np.array([0.0, 1.0, 0.0])
    if abs(float(np.dot(forward, reference_up))) > 0.98:
        reference_up = np.array([0.0, 0.0, -1.0])
    right = np.cross(reference_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack((right, up, forward))


def rasterize(
    points: np.ndarray, faces: np.ndarray, direction: np.ndarray, size: int,
    vertex_colors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rotation = camera_basis(direction)
    camera_points = (points - points.mean(axis=0)) @ rotation.T
    span = np.ptp(camera_points[:, :2], axis=0)
    scale = (size * 0.78) / max(float(span.max()), 1e-9)
    xy = np.empty((len(points), 2), dtype=np.float64)
    xy[:, 0] = camera_points[:, 0] * scale + size / 2
    xy[:, 1] = -camera_points[:, 1] * scale + size / 2
    depth = camera_points[:, 2]

    image = np.full((size, size, 3), (246, 244, 239), dtype=np.uint8)
    z_buffer = np.full((size, size), -np.inf, dtype=np.float64)
    light_camera = np.array([-0.35, 0.65, 1.0], dtype=np.float64)
    light_camera /= np.linalg.norm(light_camera)

    for face in faces:
        triangle = xy[face]
        min_x = max(0, int(math.floor(float(triangle[:, 0].min()))))
        max_x = min(size - 1, int(math.ceil(float(triangle[:, 0].max()))))
        min_y = max(0, int(math.floor(float(triangle[:, 1].min()))))
        max_y = min(size - 1, int(math.ceil(float(triangle[:, 1].max()))))
        if min_x > max_x or min_y > max_y:
            continue

        x0, y0 = triangle[0]
        x1, y1 = triangle[1]
        x2, y2 = triangle[2]
        denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(float(denominator)) < 1e-12:
            continue
        grid_x, grid_y = np.meshgrid(
            np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5,
            np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5,
        )
        weight0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
        weight1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
        weight2 = 1.0 - weight0 - weight1
        inside = (weight0 >= -1e-7) & (weight1 >= -1e-7) & (weight2 >= -1e-7)
        triangle_depth = weight0 * depth[face[0]] + weight1 * depth[face[1]] + weight2 * depth[face[2]]
        region_z = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
        visible = inside & (triangle_depth > region_z)
        if not np.any(visible):
            continue

        world = points[face]
        normal_world = np.cross(world[1] - world[0], world[2] - world[0])
        normal_camera = rotation @ normal_world
        normal_length = np.linalg.norm(normal_camera)
        if normal_length:
            normal_camera /= normal_length
        lambert = abs(float(np.dot(normal_camera, light_camera)))
        intensity = 0.38 + 0.62 * lambert
        if vertex_colors is not None:
            rgb = np.asarray(vertex_colors[face, :3], dtype=np.float64).mean(axis=0)
            base_bgr = rgb[::-1]
        else:
            base_bgr = np.array([147.0, 174.0, 103.0])
        color = np.clip(base_bgr * intensity + 35.0, 0, 255).astype(np.uint8)

        region_z[visible] = triangle_depth[visible]
        region_image = image[min_y : max_y + 1, min_x : max_x + 1]
        region_image[visible] = color

    mask = np.isfinite(z_buffer).astype(np.uint8) * 255
    contour_list, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contour_list:
        cv2.drawContours(image, contour_list, -1, (61, 68, 55), 2, cv2.LINE_AA)
    return image, mask


def silhouette_metrics(mask: np.ndarray) -> dict[str, float | int]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"components": 0, "area_px": 0, "rectangularity": 0.0, "convexity": 0.0}
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    rectangle_area = float(np.prod(cv2.minAreaRect(contour)[1]))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    return {
        "components": len(contours),
        "area_px": int(area),
        "rectangularity": area / rectangle_area if rectangle_area else 0.0,
        "convexity": area / hull_area if hull_area else 0.0,
    }


def reconstruction_gate(analysis: dict[str, object]) -> dict[str, object]:
    """Fail closed on topology defects that cannot be repaired by more detail.

    This gate is deliberately category-agnostic.  It does not decide what the
    object is or compare it with a template; it only checks whether Object
    Capture emitted one finite, closed, manifold surface.  Appearance and
    held-out-frame reprojection still require a separate review before an asset
    can be accepted.
    """
    failures: list[str] = []
    warnings: list[str] = []
    if int(analysis["connected_components"]) != 1:
        failures.append("mesh_has_multiple_connected_components")
    if int(analysis["boundary_edges"]) > 0:
        failures.append("mesh_has_open_boundary_edges")
    if int(analysis["nonmanifold_edges"]) > 0:
        failures.append("mesh_has_nonmanifold_edges")
    if int(analysis["triangles"]) < 100:
        failures.append("mesh_has_insufficient_geometry")

    dimensions = analysis["dimensions_m"]
    assert isinstance(dimensions, dict)
    extents = [float(dimensions[axis]) for axis in ("x", "y", "z")]
    if any(not math.isfinite(value) or value <= 0 for value in extents):
        failures.append("mesh_has_invalid_extent")
    else:
        aspect = max(extents) / min(extents)
        if aspect > 25:
            warnings.append("extreme_axis_ratio_requires_visual_review")

    warnings.append("metric_scale_unverified_without_depth_marker_or_measurement")
    status = "needs_recapture" if failures else "needs_visual_and_reprojection_review"
    return {
        "status": status,
        "may_advance_to_higher_detail": not failures,
        "failures": failures,
        "warnings": warnings,
    }


def add_label(image: np.ndarray, label: str) -> np.ndarray:
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (image.shape[1], 46), (246, 244, 239), -1)
    cv2.putText(labeled, label, (16, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (45, 49, 42), 2, cv2.LINE_AA)
    return labeled


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="RealityKit .usdz/.usd/.usda mesh")
    parser.add_argument("--output", type=Path, required=True, help="directory for PNG previews")
    parser.add_argument("--size", type=int, default=640, help="pixels per square view (default: 640)")
    args = parser.parse_args()
    if args.size < 128 or args.size > 2048:
        parser.error("--size must be between 128 and 2048")

    points, faces = load_mesh(args.input.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    views = {
        "top": np.array([0.0, 1.0, 0.0]),
        "front": np.array([0.0, 0.0, 1.0]),
        "side": np.array([1.0, 0.0, 0.0]),
        "isometric": np.array([1.0, 0.75, 1.0]),
    }

    rendered: list[np.ndarray] = []
    silhouettes: dict[str, object] = {}
    for name, direction in views.items():
        image, mask = rasterize(points, faces, direction, args.size)
        image = add_label(image, name.capitalize())
        output_path = args.output / f"mesh-{name}.png"
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"could not write {output_path}")
        rendered.append(image)
        silhouettes[name] = silhouette_metrics(mask)

    gutter = 16
    height, width = rendered[0].shape[:2]
    contact = np.full((height * 2 + gutter, width * 2 + gutter, 3), (225, 222, 214), dtype=np.uint8)
    for index, image in enumerate(rendered):
        row, column = divmod(index, 2)
        y, x = row * (height + gutter), column * (width + gutter)
        contact[y : y + height, x : x + width] = image
    contact_path = args.output / "mesh-contact.png"
    if not cv2.imwrite(str(contact_path), contact):
        raise RuntimeError(f"could not write {contact_path}")

    analysis = mesh_metrics(points, faces)
    analysis["silhouettes"] = silhouettes
    analysis["quality_gate"] = reconstruction_gate(analysis)
    analysis_path = args.output / "mesh-analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contact_sheet": str(contact_path), **analysis}, indent=2))


if __name__ == "__main__":
    main()
