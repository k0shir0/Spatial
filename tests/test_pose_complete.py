"""Synthetic round-trip tests for silhouette pose completion.

A known ellipsoid is rendered from a ground-truth circular orbit of 36 poses.
The even-indexed 18 are handed to :func:`complete_poses` as "registered" SfM
views; the odd 18 are withheld with only their masks.  The module must recover
the missing poses from the silhouettes alone, agree with the ground-truth
orbit, keep its registered poses self-consistent, and do so deterministically.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.pose_complete import complete_poses
from local3d.recon_common import make_view, rasterize_zbuffer
from local3d.turntable_pose import camera_pose

WIDTH, HEIGHT = 320, 240
INTRINSICS = (300.0, 160.0, 120.0, 0.0)
TEST_GRID = 72
N_FRAMES = 36
DISTANCE = 3.2
ELEVATION = 0.2
SWEEP_DEG = 360.0


def ellipsoid_mesh() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=3)
    mesh.vertices = mesh.vertices * np.array([1.0, 0.7, 0.55])
    return mesh


def frame_name(index: int) -> str:
    # frame_%04d_...ms.jpg — string sort matches temporal order.
    return f"frame_{index:04d}_{index * 100:06d}ms.jpg"


def build_ground_truth() -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """Return (masks_by_name, poses_by_name (R, t), true_center_by_name)."""

    mesh = ellipsoid_mesh()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    sweep = np.deg2rad(SWEEP_DEG)

    masks: dict[str, np.ndarray] = {}
    poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    centers: dict[str, np.ndarray] = {}
    for index in range(N_FRAMES):
        azimuth = sweep * index / N_FRAMES
        rotation, translation = camera_pose(DISTANCE, ELEVATION, azimuth)
        zbuffer, _ = rasterize_zbuffer(
            vertices, faces, rotation, translation, INTRINSICS, WIDTH, HEIGHT, scale=1.0
        )
        name = frame_name(index)
        masks[name] = np.isfinite(zbuffer)
        poses[name] = (rotation, translation)
        centers[name] = -rotation.T @ translation
    return masks, poses, centers


def registered_views(
    masks: dict[str, np.ndarray], poses: dict[str, tuple[np.ndarray, np.ndarray]]
) -> list[dict]:
    views = []
    for index in range(0, N_FRAMES, 2):  # even indices are "registered"
        name = frame_name(index)
        rotation, translation = poses[name]
        views.append(
            make_view(
                name=name,
                image_path=Path(name),
                rotation=rotation,
                translation=translation,
                mask_tight=masks[name],
            )
        )
    return views


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(float(np.linalg.norm(a)), 1e-12)
    b = b / max(float(np.linalg.norm(b)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(float(a @ b), -1.0, 1.0))))


class PoseCompleteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.masks, self.poses, self.centers = build_ground_truth()
        self.views = registered_views(self.masks, self.poses)

    def test_masks_are_nonempty_and_in_frame(self) -> None:
        for mask in self.masks.values():
            self.assertTrue(mask.any())
            self.assertFalse(mask[0].any() or mask[-1].any())
            self.assertFalse(mask[:, 0].any() or mask[:, -1].any())

    def test_accepts_most_missing_frames(self) -> None:
        result = complete_poses(
            self.views, self.masks, INTRINSICS, grid=TEST_GRID, iterations=2
        )
        self.assertGreaterEqual(result["accepted"], 14)
        self.assertEqual(result["accepted"] + result["rejected"], 18)
        # views_all = 18 registered + accepted silhouette views, sorted by name.
        names = [view["name"] for view in result["views_all"]]
        self.assertEqual(names, sorted(names))
        self.assertEqual(len(result["views_all"]), 18 + result["accepted"])

    def test_completed_views_are_well_formed(self) -> None:
        result = complete_poses(
            self.views, self.masks, INTRINSICS, grid=TEST_GRID, iterations=2
        )
        sources = {"sfm": 0, "silhouette": 0}
        for view in result["views_all"]:
            sources[view["pose_source"]] += 1
            self.assertEqual(view["rotation"].shape, (3, 3))
            self.assertAlmostEqual(
                float(np.linalg.det(view["rotation"])), 1.0, places=5
            )
            if view["pose_source"] == "silhouette":
                self.assertIsNone(view["image_path"])
                self.assertIn("silhouette_iou", view)
                self.assertIsNotNone(view["mask_tight"])
        self.assertEqual(sources["sfm"], 18)
        self.assertEqual(sources["silhouette"], result["accepted"])

    def test_accepted_azimuth_error_is_small(self) -> None:
        result = complete_poses(
            self.views, self.masks, INTRINSICS, grid=TEST_GRID, iterations=2
        )
        errors = []
        for view in result["views_all"]:
            if view["pose_source"] != "silhouette":
                continue
            recovered = view["center"]
            truth = self.centers[view["name"]]
            errors.append(angle_between(recovered, truth))
        self.assertTrue(errors)
        self.assertLess(float(np.median(errors)), 10.0, msg=f"errors={errors}")

    def test_registered_views_are_self_consistent(self) -> None:
        # Step-4 check: rebuilding a registered pose from its own (az, el, r) must
        # reproject the hull to high IoU vs its own mask.
        result = complete_poses(
            self.views, self.masks, INTRINSICS, grid=TEST_GRID, iterations=2
        )
        self_ious = [
            view["self_iou"]
            for view in result["views_all"]
            if view["pose_source"] == "sfm"
        ]
        self.assertEqual(len(self_ious), 18)
        self.assertGreaterEqual(min(self_ious), 0.80, msg=f"self_ious={self_ious}")
        # Rebuilt poses must track the directly-posed registered IoU closely.
        self.assertGreaterEqual(
            result["report"]["self_consistency_iou_median"],
            0.85 * result["report"]["median_registered_iou"],
        )

    def test_is_deterministic(self) -> None:
        first = complete_poses(
            self.views, self.masks, INTRINSICS, grid=TEST_GRID, iterations=2
        )
        second = complete_poses(
            self.views, self.masks, INTRINSICS, grid=TEST_GRID, iterations=2
        )
        self.assertEqual(first["report"], second["report"])
        self.assertEqual(first["accepted"], second["accepted"])
        self.assertEqual(first["rejected"], second["rejected"])

    def test_runtime_under_budget(self) -> None:
        start = time.perf_counter()
        complete_poses(
            self.views, self.masks, INTRINSICS, grid=TEST_GRID, iterations=2
        )
        self.assertLess(time.perf_counter() - start, 90.0)


if __name__ == "__main__":
    unittest.main()
