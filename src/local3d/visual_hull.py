"""Lightweight orthographic visual-hull reconstruction from ordered masks.

This backend is intended for stationary objects on a turntable or captures that
approximate one.  It makes no semantic or learned-shape inference: geometry is
the intersection of supplied silhouettes under configured orbit cameras.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from skimage.measure import marching_cubes


@dataclass(frozen=True)
class OrbitView:
    yaw_degrees: float
    elevation_degrees: float = 0.0
    roll_degrees: float = 0.0


def _camera_axes(view: OrbitView) -> tuple[np.ndarray, np.ndarray]:
    yaw = np.deg2rad(view.yaw_degrees)
    elevation = np.deg2rad(view.elevation_degrees)
    forward = np.array(
        [np.sin(yaw) * np.cos(elevation), np.sin(elevation), np.cos(yaw) * np.cos(elevation)],
        dtype=np.float32,
    )
    forward /= np.linalg.norm(forward)
    reference_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(reference_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    if view.roll_degrees:
        roll = np.deg2rad(view.roll_degrees)
        rolled_right = np.cos(roll) * right + np.sin(roll) * up
        rolled_up = -np.sin(roll) * right + np.cos(roll) * up
        right, up = rolled_right, rolled_up
    return right, up


def carve_visual_hull(
    masks: Sequence[np.ndarray],
    views: Sequence[OrbitView],
    *,
    resolution: int = 96,
    padding: float = 0.04,
    max_view_violations: int = 0,
) -> np.ndarray:
    """Return a ``(resolution, resolution, resolution)`` boolean occupancy grid."""
    if len(masks) != len(views) or len(masks) < 3:
        raise ValueError("masks and views must have equal length with at least three views")
    if resolution < 24 or resolution > 256:
        raise ValueError("resolution must be between 24 and 256")
    if not 0 <= padding < 0.25:
        raise ValueError("padding must be in [0, 0.25)")
    if max_view_violations < 0 or max_view_violations >= len(masks):
        raise ValueError("max_view_violations must be nonnegative and smaller than the view count")

    axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    occupied = np.ones(len(points), dtype=bool)
    violations = np.zeros(len(points), dtype=np.uint8)

    for mask, view in zip(masks, views):
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        binary = np.asarray(mask) > 0
        if binary.ndim != 2 or not binary.any():
            raise ValueError("every mask must be a non-empty 2D silhouette")
        height, width = binary.shape
        right, up = _camera_axes(view)
        visible_points = points[occupied]
        u = visible_points @ right
        v = visible_points @ up
        # Preserve square pixels. Mapping x and y independently across a
        # non-square frame silently stretches the recovered object.
        scale_pixels = (min(width, height) - 1) / (2.0 * (1.0 + padding))
        px = np.rint(u * scale_pixels + (width - 1) * 0.5).astype(np.int32)
        py = np.rint(-v * scale_pixels + (height - 1) * 0.5).astype(np.int32)
        inside_image = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        survives = np.zeros(len(visible_points), dtype=bool)
        survives[inside_image] = binary[py[inside_image], px[inside_image]]
        occupied_indices = np.flatnonzero(occupied)
        failed = occupied_indices[~survives]
        violations[failed] += 1
        occupied[failed[violations[failed] > max_view_violations]] = False
        if not occupied.any():
            raise ValueError("silhouettes have an empty intersection; check ordering/camera assumptions")

    return occupied.reshape((resolution, resolution, resolution))


def occupancy_to_mesh(occupancy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract centered vertices and triangle indices using marching cubes."""
    if occupancy.ndim != 3 or min(occupancy.shape) < 3 or not occupancy.any():
        raise ValueError("occupancy must be a non-empty 3D grid")
    vertices_zyx, faces, _normals, _values = marching_cubes(
        np.pad(occupancy.astype(np.uint8), 1), level=0.5
    )
    shape = np.asarray(occupancy.shape, dtype=np.float32)
    vertices_zyx -= 1.0
    vertices_xyz = vertices_zyx[:, ::-1]
    vertices_xyz = vertices_xyz / (shape[::-1] - 1.0) * 2.0 - 1.0
    return vertices_xyz.astype(np.float32), faces.astype(np.int32)


def taubin_smooth(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    iterations: int = 8,
    lamb: float = 0.45,
    mu: float = -0.47,
) -> np.ndarray:
    """Smooth voxel stair-steps with a shrinkage-resistant Taubin filter."""
    if iterations < 0 or iterations > 50:
        raise ValueError("iterations must be between 0 and 50")
    result = np.asarray(vertices, dtype=np.float32).copy()
    neighbors: list[set[int]] = [set() for _ in range(len(result))]
    for a, b, c in np.asarray(faces, dtype=np.int32):
        neighbors[a].update((int(b), int(c)))
        neighbors[b].update((int(a), int(c)))
        neighbors[c].update((int(a), int(b)))

    def pass_once(weight: float) -> None:
        delta = np.zeros_like(result)
        for index, adjacent in enumerate(neighbors):
            if adjacent:
                delta[index] = result[list(adjacent)].mean(axis=0) - result[index]
        result[:] += weight * delta

    for _ in range(iterations):
        pass_once(lamb)
        pass_once(mu)
    return result


def load_binary_masks(paths: Sequence[Path]) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not decode mask: {path}")
        result.append(np.where(image > 127, 255, 0).astype(np.uint8))
    return result
