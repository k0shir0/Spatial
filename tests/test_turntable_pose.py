"""Synthetic round-trip tests for silhouette-only turntable pose recovery.

A known ellipsoid is rendered from a true circular orbit; the module must
recover the sweep, elevation and a hull consistent with the ground truth from
the silhouettes alone, and do so deterministically.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.recon_common import rasterize_zbuffer
from local3d.turntable_pose import (
    camera_pose,
    carve_hull,
    fit_turntable_poses,
    hull_volume,
)

WIDTH, HEIGHT = 320, 240
INTRINSICS = (300.0, 160.0, 120.0, 0.0)
TEST_GRID = 64


def ellipsoid_mesh() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.vertices = mesh.vertices * np.array([1.0, 0.7, 0.55])
    return mesh


def render_orbit_masks(
    mesh: trimesh.Trimesh,
    *,
    n: int,
    distance: float,
    elevation: float,
    sweep_deg: float,
) -> list[np.ndarray]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    sweep = np.deg2rad(sweep_deg)
    masks: list[np.ndarray] = []
    for i in range(n):
        azimuth = sweep * i / n  # true circular orbit, evenly spaced
        rotation, translation = camera_pose(distance, elevation, azimuth)
        zbuffer, _ = rasterize_zbuffer(
            vertices, faces, rotation, translation, INTRINSICS, WIDTH, HEIGHT, scale=1.0
        )
        masks.append(np.isfinite(zbuffer))
    return masks


class TurntablePoseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mesh = ellipsoid_mesh()
        self.masks = render_orbit_masks(
            self.mesh, n=48, distance=3.2, elevation=0.2, sweep_deg=360.0
        )

    def test_masks_are_nonempty_and_in_frame(self) -> None:
        for mask in self.masks:
            self.assertTrue(mask.any())
            # Object stays clear of the frame border in a clean turntable capture.
            self.assertFalse(mask[0].any() or mask[-1].any())
            self.assertFalse(mask[:, 0].any() or mask[:, -1].any())

    def test_recovers_orbit_and_accepts(self) -> None:
        result = fit_turntable_poses(
            self.masks, INTRINSICS, grid=TEST_GRID, refine_grid=TEST_GRID
        )
        self.assertTrue(result["ok"])
        self.assertGreater(result["score"], 0.75)
        self.assertLess(abs(result["sweep_deg"] - 360.0), 25.0)
        self.assertLess(abs(result["report"]["elevation_rad"] - 0.2), 0.12)
        self.assertEqual(len(result["views"]), len(self.masks))
        # Every returned view is a valid COLMAP pose looking at the origin.
        for view in result["views"]:
            self.assertEqual(view["rotation"].shape, (3, 3))
            self.assertAlmostEqual(float(np.linalg.det(view["rotation"])), 1.0, places=5)

    def test_recovered_hull_volume_matches_ground_truth(self) -> None:
        result = fit_turntable_poses(
            self.masks, INTRINSICS, grid=TEST_GRID, refine_grid=TEST_GRID
        )
        occupancy = carve_hull(self.masks, result["views"], INTRINSICS, grid=TEST_GRID)
        recovered_volume = hull_volume(occupancy)
        truth_volume = float(self.mesh.volume)
        self.assertLess(
            abs(recovered_volume - truth_volume) / truth_volume,
            0.35,
            msg=f"hull {recovered_volume:.3f} vs truth {truth_volume:.3f}",
        )

    def test_is_deterministic(self) -> None:
        first = fit_turntable_poses(
            self.masks, INTRINSICS, grid=TEST_GRID, refine_grid=TEST_GRID
        )
        second = fit_turntable_poses(
            self.masks, INTRINSICS, grid=TEST_GRID, refine_grid=TEST_GRID
        )
        self.assertEqual(first["report"], second["report"])
        self.assertEqual(first["sweep_deg"], second["sweep_deg"])
        self.assertEqual(first["score"], second["score"])

    def test_too_few_masks_fails_closed(self) -> None:
        result = fit_turntable_poses(
            self.masks[:8], INTRINSICS, grid=TEST_GRID, refine_grid=TEST_GRID
        )
        self.assertFalse(result["ok"])
        self.assertTrue(
            any("too_few_masks" in reason for reason in result["report"]["fail_reasons"])
        )
        # Poses are still returned for every supplied frame.
        self.assertEqual(len(result["views"]), 8)

    def test_camera_pose_places_camera_at_distance_looking_at_origin(self) -> None:
        rotation, translation = camera_pose(3.0, 0.15, 0.7)
        center = -rotation.T @ translation
        self.assertAlmostEqual(float(np.linalg.norm(center)), 3.0, places=6)
        # The origin projects to camera-frame +Z (in front) near the principal point.
        in_camera = rotation @ np.zeros(3) + translation
        self.assertGreater(in_camera[2], 0.0)


if __name__ == "__main__":
    unittest.main()
