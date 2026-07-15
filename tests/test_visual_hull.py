from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.visual_hull import OrbitView, _camera_axes, carve_visual_hull, occupancy_to_mesh, taubin_smooth


def plush_occupancy(resolution: int) -> np.ndarray:
    axis = np.linspace(-1, 1, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")

    def ellipsoid(cx, cy, cz, rx, ry, rz):
        return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2 <= 1

    # Deliberately asymmetric soft-toy proxy: body, head, ears, muzzle and limbs.
    return (
        ellipsoid(0, -0.18, 0, 0.46, 0.58, 0.34)
        | ellipsoid(0, 0.43, 0.02, 0.39, 0.37, 0.33)
        | ellipsoid(-0.27, 0.75, 0, 0.16, 0.25, 0.13)
        | ellipsoid(0.25, 0.77, 0, 0.14, 0.23, 0.12)
        | ellipsoid(0, 0.39, 0.30, 0.22, 0.16, 0.13)
        | ellipsoid(-0.42, -0.13, 0, 0.18, 0.43, 0.16)
        | ellipsoid(0.43, -0.08, 0.03, 0.17, 0.39, 0.15)
        | ellipsoid(-0.22, -0.72, 0.02, 0.19, 0.31, 0.20)
        | ellipsoid(0.24, -0.70, 0.05, 0.18, 0.29, 0.19)
    )


def render_mask(occupancy: np.ndarray, view: OrbitView, size: int = 192, padding: float = 0.04) -> np.ndarray:
    resolution = occupancy.shape[0]
    axis = np.linspace(-1, 1, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.column_stack((x[occupancy], y[occupancy], z[occupancy]))
    right, up = _camera_axes(view)
    u, v = points @ right, points @ up
    scale = 1 + padding
    scale_pixels = (size - 1) / (2 * scale)
    px = np.rint(u * scale_pixels + (size - 1) * 0.5).astype(np.int32)
    py = np.rint(-v * scale_pixels + (size - 1) * 0.5).astype(np.int32)
    mask = np.zeros((size, size), dtype=np.uint8)
    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    mask[py[valid], px[valid]] = 255
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


class VisualHullTests(unittest.TestCase):
    def test_plush_proxy_accuracy_and_runtime(self):
        resolution = 72
        truth = plush_occupancy(resolution)
        views = [OrbitView(yaw, elevation) for elevation in (-18, 0, 18) for yaw in range(0, 360, 45)]
        masks = [render_mask(truth, view) for view in views]
        started = time.perf_counter()
        recovered = carve_visual_hull(masks, views, resolution=resolution, padding=0.04)
        elapsed = time.perf_counter() - started
        intersection = np.logical_and(truth, recovered).sum()
        union = np.logical_or(truth, recovered).sum()
        iou = intersection / union
        recall = intersection / truth.sum()
        self.assertGreater(iou, 0.72)
        self.assertGreater(recall, 0.97)
        self.assertLess(elapsed, 8.0)

        vertices, faces = occupancy_to_mesh(recovered)
        self.assertGreater(len(vertices), 1000)
        self.assertGreater(len(faces), 2000)
        self.assertTrue(np.isfinite(vertices).all())
        smoothed = taubin_smooth(vertices, faces, iterations=4)
        self.assertEqual(smoothed.shape, vertices.shape)
        self.assertTrue(np.isfinite(smoothed).all())
        # Taubin smoothing should not materially collapse the recovered body.
        self.assertGreater(np.ptp(smoothed, axis=0).min(), np.ptp(vertices, axis=0).min() * 0.95)

    def test_invalid_inputs_fail_closed(self):
        mask = np.ones((32, 32), dtype=np.uint8) * 255
        with self.assertRaises(ValueError):
            carve_visual_hull([mask, mask], [OrbitView(0), OrbitView(180)])

    def test_non_square_masks_preserve_square_pixel_geometry(self):
        # A circular silhouette in a 3:2 frame must not become 1.5x taller.
        mask = np.zeros((120, 180), dtype=np.uint8)
        cv2.circle(mask, (90, 60), 42, 255, -1)
        views = [OrbitView(yaw) for yaw in (0, 60, 120, 180, 240, 300)]
        recovered = carve_visual_hull([mask] * len(views), views, resolution=64, padding=0)
        occupied = np.argwhere(recovered)
        extents = np.ptp(occupied, axis=0)
        self.assertLess(abs(float(extents[1] / extents[2]) - 1.0), 0.08)

    def test_one_bad_mask_can_be_tolerated_explicitly(self):
        truth = plush_occupancy(64)
        views = [OrbitView(yaw) for yaw in range(0, 360, 45)]
        masks = [render_mask(truth, view, size=128) for view in views]
        shifted = np.roll(masks[3], 4, axis=1)
        strict = carve_visual_hull(masks[:3] + [shifted] + masks[4:], views, resolution=64)
        robust = carve_visual_hull(
            masks[:3] + [shifted] + masks[4:], views, resolution=64, max_view_violations=1
        )
        strict_recall = np.logical_and(truth, strict).sum() / truth.sum()
        robust_recall = np.logical_and(truth, robust).sum() / truth.sum()
        self.assertGreater(robust_recall, strict_recall + 0.03)


if __name__ == "__main__":
    unittest.main()
