"""Synthetic round-trip tests for deterministic ChArUco pose estimation."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from local3d.charuco_pose import (
    BoardSpec,
    FramePose,
    approximate_camera_matrix,
    camera_azimuth_elevation,
    estimate_frame_pose,
    estimate_poses,
    make_board,
    render_board_image,
    select_by_pose_coverage,
)


def pose_at_azimuth(azimuth_degrees: float, *, corners: int = 20, rms: float = 0.4) -> FramePose:
    """Identity-rotation pose whose camera center sits at a chosen azimuth."""

    radians = np.deg2rad(azimuth_degrees)
    center = np.array([np.cos(radians) * 500.0, np.sin(radians) * 500.0, 300.0])
    return FramePose(
        rotation_rodrigues=(0.0, 0.0, 0.0),
        translation_mm=tuple(-center),
        corner_count=corners,
        rms_reprojection_px=rms,
    )

IMAGE_SIZE = (1280, 720)
CAMERA = np.array(
    [[1000.0, 0.0, 639.5], [0.0, 1000.0, 359.5], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def synthetic_view(spec: BoardSpec, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Warp the flat board render into the view of a known synthetic camera.

    The mapping is built from actual detections on the flat render, so the
    test does not depend on OpenCV's board coordinate conventions.
    """

    flat = render_board_image(spec, pixels_per_square=80)
    board = make_board(spec)
    detector = cv2.aruco.CharucoDetector(board)
    corners, ids, _markers, _marker_ids = detector.detectBoard(flat)
    assert corners is not None and len(corners) >= 8, "flat render must be detectable"
    object_points, image_points = board.matchImagePoints(corners, ids)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, CAMERA, np.zeros((5, 1)))
    homography, _ = cv2.findHomography(image_points.reshape(-1, 2), projected.reshape(-1, 2))
    return cv2.warpPerspective(
        flat,
        homography,
        IMAGE_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


class CharucoPoseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = BoardSpec()

    def check_round_trip(self, rvec: np.ndarray, tvec: np.ndarray) -> None:
        view = synthetic_view(self.spec, rvec, tvec)
        pose = estimate_frame_pose(view, self.spec, CAMERA)
        self.assertIsNotNone(pose, "pose gates rejected a clean synthetic view")
        assert pose is not None
        recovered_rotation, _ = cv2.Rodrigues(np.asarray(pose.rotation_rodrigues))
        expected_rotation, _ = cv2.Rodrigues(rvec)
        relative, _ = cv2.Rodrigues(recovered_rotation.T @ expected_rotation)
        rotation_error_deg = float(np.rad2deg(np.linalg.norm(relative)))
        translation_error = float(np.linalg.norm(np.asarray(pose.translation_mm) - tvec.ravel()))
        self.assertLess(rotation_error_deg, 0.5)
        self.assertLess(translation_error, 0.01 * float(np.linalg.norm(tvec)))
        self.assertLess(pose.rms_reprojection_px, 1.0)

    def test_round_trip_tilted_view(self) -> None:
        rvec = np.deg2rad(np.array([[25.0], [10.0], [5.0]]))
        tvec = np.array([[-90.0], [-60.0], [600.0]])
        self.check_round_trip(rvec, tvec)

    def test_round_trip_orbit_views(self) -> None:
        for yaw in (-30.0, 0.0, 30.0):
            rvec = np.deg2rad(np.array([[20.0], [yaw], [0.0]]))
            tvec = np.array([[-80.0], [-50.0], [700.0]])
            with self.subTest(yaw=yaw):
                self.check_round_trip(rvec, tvec)

    def test_estimation_is_deterministic(self) -> None:
        rvec = np.deg2rad(np.array([[22.0], [-15.0], [0.0]]))
        tvec = np.array([[-70.0], [-40.0], [650.0]])
        view = synthetic_view(self.spec, rvec, tvec)
        first = estimate_frame_pose(view, self.spec, CAMERA)
        second = estimate_frame_pose(view, self.spec, CAMERA)
        self.assertEqual(first, second)

    def test_boardless_frame_fails_closed(self) -> None:
        blank = np.full((720, 1280), 255, dtype=np.uint8)
        self.assertIsNone(estimate_frame_pose(blank, self.spec, CAMERA))
        noise = (np.indices((720, 1280)).sum(axis=0) % 256).astype(np.uint8)
        self.assertIsNone(estimate_frame_pose(noise, self.spec, CAMERA))

    def test_estimate_poses_mixes_good_and_failed_frames(self) -> None:
        rvec = np.deg2rad(np.array([[25.0], [10.0], [0.0]]))
        tvec = np.array([[-90.0], [-60.0], [600.0]])
        good = synthetic_view(self.spec, rvec, tvec)
        blank = np.full((720, 1280), 255, dtype=np.uint8)
        poses = estimate_poses([good, blank], self.spec, CAMERA)
        self.assertIsNotNone(poses[0])
        self.assertIsNone(poses[1])

    def test_approximate_camera_matrix_shape(self) -> None:
        matrix = approximate_camera_matrix(1280, 720, 65.0)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertGreater(matrix[0, 0], 500.0)

    def test_board_spec_validation(self) -> None:
        with self.assertRaises(ValueError):
            BoardSpec(squares_x=2)
        with self.assertRaises(ValueError):
            BoardSpec(marker_mm=40.0)
        with self.assertRaises(ValueError):
            BoardSpec(dictionary="DICT_UNKNOWN")

    def test_render_board_image_is_printable(self) -> None:
        image = render_board_image(self.spec, pixels_per_square=60)
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(image.shape[0], 300)
        self.assertGreater((image == 255).mean(), 0.3)


class PoseCoverageSelectionTests(unittest.TestCase):
    def test_azimuth_round_trip(self) -> None:
        for azimuth in (-150.0, -30.0, 0.0, 45.0, 170.0):
            recovered, elevation = camera_azimuth_elevation(pose_at_azimuth(azimuth))
            self.assertAlmostEqual(recovered, azimuth, places=6)
            self.assertGreater(elevation, 0.0)

    def test_selection_spreads_azimuths_and_skips_failed_frames(self) -> None:
        poses: list[FramePose | None] = [pose_at_azimuth(step * 10.0) for step in range(36)]
        poses[7] = None
        poses[8] = None
        selected = select_by_pose_coverage(poses, count=6, min_azimuth_gap_degrees=30.0)
        self.assertEqual(len(selected), 6)
        self.assertNotIn(7, selected)
        self.assertNotIn(8, selected)
        azimuths = sorted(camera_azimuth_elevation(poses[index])[0] for index in selected)
        gaps = [b - a for a, b in zip(azimuths, azimuths[1:])]
        gaps.append(360.0 - (azimuths[-1] - azimuths[0]))
        self.assertGreaterEqual(min(gaps), 30.0)

    def test_selection_is_deterministic_and_prefers_supported_seed(self) -> None:
        poses: list[FramePose | None] = [
            pose_at_azimuth(0.0, corners=8, rms=1.5),
            pose_at_azimuth(120.0, corners=24, rms=0.3),
            pose_at_azimuth(240.0, corners=12, rms=0.8),
        ]
        first = select_by_pose_coverage(poses, count=3)
        second = select_by_pose_coverage(poses, count=3)
        self.assertEqual(first, second)
        self.assertEqual(first, [0, 1, 2])

    def test_selection_relaxes_gap_on_narrow_orbits(self) -> None:
        poses: list[FramePose | None] = [pose_at_azimuth(float(step)) for step in range(8)]
        selected = select_by_pose_coverage(poses, count=4, min_azimuth_gap_degrees=45.0)
        self.assertEqual(len(selected), 4)

    def test_selection_handles_all_failed(self) -> None:
        self.assertEqual(select_by_pose_coverage([None, None], count=4), [])
        with self.assertRaises(ValueError):
            select_by_pose_coverage([pose_at_azimuth(0.0)], count=0)


if __name__ == "__main__":
    unittest.main()
