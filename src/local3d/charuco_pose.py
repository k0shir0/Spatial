"""Deterministic ChArUco-board pose estimation for orbit captures.

A printed ChArUco board under or behind the object gives every frame an
object-relative 6-DoF camera pose with zero learned inference: detection and
PnP are classical OpenCV, run single-threaded for repeatability.  A board
(rather than one square marker) avoids the two-fold single-marker pose
ambiguity and keeps PnP well-conditioned when the object occludes part of
the pattern.

Frames whose detections fail the corner-count or reprojection gates return
``None`` so low-support poses fail closed for review instead of poisoning a
reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import cv2
import numpy as np

_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
}


@dataclass(frozen=True)
class BoardSpec:
    """Printable board geometry; lengths are millimeters."""

    squares_x: int = 7
    squares_y: int = 5
    square_mm: float = 30.0
    marker_mm: float = 22.0
    dictionary: str = "DICT_4X4_100"

    def __post_init__(self) -> None:
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("board needs at least 3x3 squares")
        if not 0 < self.marker_mm < self.square_mm:
            raise ValueError("marker_mm must be positive and smaller than square_mm")
        if self.dictionary not in _DICTIONARIES:
            raise ValueError(f"dictionary must be one of {sorted(_DICTIONARIES)}")


@dataclass(frozen=True)
class FramePose:
    """Object-relative camera pose for one frame; translation in millimeters."""

    rotation_rodrigues: tuple[float, float, float]
    translation_mm: tuple[float, float, float]
    corner_count: int
    rms_reprojection_px: float

    def as_json(self) -> dict[str, Any]:
        return {
            "rotation_rodrigues": list(self.rotation_rodrigues),
            "translation_mm": list(self.translation_mm),
            "corner_count": self.corner_count,
            "rms_reprojection_px": self.rms_reprojection_px,
        }


def make_board(spec: BoardSpec) -> "cv2.aruco.CharucoBoard":
    dictionary = cv2.aruco.getPredefinedDictionary(_DICTIONARIES[spec.dictionary])
    return cv2.aruco.CharucoBoard(
        (spec.squares_x, spec.squares_y),
        float(spec.square_mm),
        float(spec.marker_mm),
        dictionary,
    )


def render_board_image(spec: BoardSpec, pixels_per_square: int = 60, margin_squares: float = 0.5) -> np.ndarray:
    """Render the printable board pattern (white margin included)."""

    if pixels_per_square < 20:
        raise ValueError("pixels_per_square must be at least 20 for reliable printing")
    board = make_board(spec)
    width = int(round((spec.squares_x + 2 * margin_squares) * pixels_per_square))
    height = int(round((spec.squares_y + 2 * margin_squares) * pixels_per_square))
    margin = int(round(margin_squares * pixels_per_square))
    return board.generateImage((width, height), marginSize=margin)


def approximate_camera_matrix(width: int, height: int, horizontal_fov_degrees: float = 65.0) -> np.ndarray:
    """Rough intrinsics from an assumed field of view.

    Uncalibrated intrinsics bias absolute distance, so poses built with this
    helper are suitable for relative orbit geometry, not metric claims.
    """

    if not 20.0 <= horizontal_fov_degrees <= 120.0:
        raise ValueError("horizontal_fov_degrees must be within [20, 120]")
    focal = (width / 2.0) / np.tan(np.deg2rad(horizontal_fov_degrees) / 2.0)
    return np.array(
        [[focal, 0.0, (width - 1) / 2.0], [0.0, focal, (height - 1) / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def estimate_frame_pose(
    image: np.ndarray,
    spec: BoardSpec,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    min_corners: int = 6,
    max_rms_px: float = 3.0,
) -> FramePose | None:
    """Estimate one frame's board pose, or ``None`` when gates fail."""

    cv2.setNumThreads(1)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distortion = np.zeros((5, 1)) if dist_coeffs is None else np.asarray(dist_coeffs, dtype=np.float64)
    board = make_board(spec)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _marker_corners, _marker_ids = detector.detectBoard(image)
    if charuco_corners is None or charuco_ids is None or len(charuco_corners) < min_corners:
        return None
    object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
    if object_points is None or len(object_points) < min_corners:
        return None
    ok, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points, image_points, camera_matrix, distortion, rvec, tvec
    )
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    residuals = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    if rms > max_rms_px:
        return None
    return FramePose(
        rotation_rodrigues=tuple(float(value) for value in rvec.ravel()),
        translation_mm=tuple(float(value) for value in tvec.ravel()),
        corner_count=int(len(object_points)),
        rms_reprojection_px=rms,
    )


def estimate_poses(
    frames: Sequence[np.ndarray],
    spec: BoardSpec,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
    *,
    min_corners: int = 6,
    max_rms_px: float = 3.0,
) -> list[FramePose | None]:
    """Per-frame board poses; failed frames are ``None`` (fail closed)."""

    return [
        estimate_frame_pose(
            frame,
            spec,
            camera_matrix,
            dist_coeffs,
            min_corners=min_corners,
            max_rms_px=max_rms_px,
        )
        for frame in frames
    ]


def camera_position_board_frame(pose: FramePose) -> np.ndarray:
    """Camera center in board coordinates (millimeters)."""

    rotation, _ = cv2.Rodrigues(np.asarray(pose.rotation_rodrigues, dtype=np.float64))
    translation = np.asarray(pose.translation_mm, dtype=np.float64).reshape(3, 1)
    return (-rotation.T @ translation).ravel()


def camera_azimuth_elevation(pose: FramePose) -> tuple[float, float]:
    """Board-relative viewing direction in degrees.

    Azimuth is the angle around the board normal (Z); elevation is the angle
    above the board plane.  Together they describe orbit coverage without any
    image content analysis.
    """

    position = camera_position_board_frame(pose)
    azimuth = float(np.rad2deg(np.arctan2(position[1], position[0])))
    planar = float(np.hypot(position[0], position[1]))
    elevation = float(np.rad2deg(np.arctan2(position[2], max(planar, 1e-9))))
    return azimuth, elevation


def _azimuth_distance(left: float, right: float) -> float:
    delta = abs(left - right) % 360.0
    return min(delta, 360.0 - delta)


def select_by_pose_coverage(
    poses: Sequence[FramePose | None],
    *,
    count: int,
    min_azimuth_gap_degrees: float = 12.0,
) -> list[int]:
    """Pick frames that spread measured azimuths around the orbit.

    Selection is pure geometry: candidates are frames with a gated pose,
    seeded from the most corner-supported pose, then greedily maximizing the
    minimum azimuth distance to already-selected frames (preferring the
    configured gap, relaxing it only when the orbit has too little spread).
    Deterministic for identical inputs; returns chronological indices.
    """

    if count < 1:
        raise ValueError("count must be at least 1")
    candidates = [index for index, pose in enumerate(poses) if pose is not None]
    if not candidates:
        return []
    azimuths = {index: camera_azimuth_elevation(poses[index])[0] for index in candidates}

    def support(index: int) -> tuple[int, float, int]:
        pose = poses[index]
        assert pose is not None
        # More corners, then lower reprojection error, then earlier frame.
        return (-pose.corner_count, pose.rms_reprojection_px, index)

    selected = [min(candidates, key=support)]

    def best_remaining(enforce_gap: bool) -> int | None:
        best_index: int | None = None
        best_key: tuple[float, int, float, int] | None = None
        for index in candidates:
            if index in selected:
                continue
            spread = min(_azimuth_distance(azimuths[index], azimuths[chosen]) for chosen in selected)
            if enforce_gap and spread < min_azimuth_gap_degrees:
                continue
            pose = poses[index]
            assert pose is not None
            key = (-spread, -pose.corner_count, pose.rms_reprojection_px, index)
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        return best_index

    while len(selected) < min(count, len(candidates)):
        candidate = best_remaining(enforce_gap=True)
        if candidate is None:
            candidate = best_remaining(enforce_gap=False)
        if candidate is None:
            break
        selected.append(candidate)
    return sorted(selected)
