from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from local3d.ingest import FrameMetrics, FrameRecord, IngestConfig, VideoAnalysis, VideoMetadata
from local3d.source_guard import assess_source_video


def _metadata(
    root: Path,
    *,
    fps: float = 30.0,
    has_audio: bool = True,
    width: int = 1620,
    height: int = 1080,
    duration: float = 8.0,
) -> VideoMetadata:
    return VideoMetadata(
        source_path=str(root / "source.mov"),
        file_size_bytes=1,
        modified_time_ns=1,
        container="mov",
        duration_s=duration,
        start_time_s=0.0,
        width=width,
        height=height,
        display_width=width,
        display_height=height,
        rotation_degrees=0,
        fps=fps,
        frame_count=round(fps * duration),
        codec="h264",
        pixel_format="yuv420p",
        bit_rate_bps=None,
        color_space=None,
        color_transfer=None,
        color_primaries=None,
        creation_time=None,
        has_audio=has_audio,
    )


def _metrics() -> FrameMetrics:
    return FrameMetrics(
        brightness_mean=0.5,
        contrast_stddev=0.2,
        entropy_bits=5.0,
        sharpness_laplacian=100.0,
        edge_density=0.1,
        dark_fraction=0.0,
        clipped_fraction=0.0,
        subject_coverage=0.2,
        central_subject_coverage=0.3,
        subject_border_touch=0.0,
        coverage_source="temporal_change",
    )


def _analysis(root: Path, images: list[np.ndarray], metadata: VideoMetadata) -> VideoAnalysis:
    records: list[FrameRecord] = []
    for index, image in enumerate(images):
        path = root / f"frame_{index:04d}.jpg"
        assert cv2.imwrite(str(path), image)
        records.append(
            FrameRecord(
                candidate_index=index,
                source_frame_index=index * 30,
                timestamp_s=float(index),
                path=str(path),
                width=image.shape[1],
                height=image.shape[0],
                metrics=_metrics(),
            )
        )
    return VideoAnalysis(
        source=metadata.source_path,
        metadata=metadata,
        config=IngestConfig(),
        frames=records,
        keyframes=records,
    )


def _continuous_rotation_frames(count: int = 9) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for index in range(count):
        image = np.full((180, 280, 3), (96, 126, 116), dtype=np.uint8)
        center_x = 85 + index * 14
        color = (40 + index * 8, 150 - index * 4, 210 - index * 5)
        cv2.rectangle(image, (center_x - 36, 54), (center_x + 36, 134), color, -1)
        cv2.circle(image, (center_x, 94), 18, (15, 25, 35), 3)
        output.append(image)
    return output


def test_accepts_continuous_camera_capture(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path)
    assessment = assess_source_video(_analysis(tmp_path, _continuous_rotation_frames(), metadata))

    assert assessment["accepted"] is True
    assert assessment["hard_failures"] == []
    assert assessment["scene_analysis"]["suspected_cuts"] == []


def test_rejects_persistent_hard_cut_between_scenes(tmp_path: Path) -> None:
    first = np.full((180, 280, 3), (30, 160, 45), dtype=np.uint8)
    second = np.full((180, 280, 3), (210, 25, 190), dtype=np.uint8)
    cv2.rectangle(first, (70, 45), (210, 145), (20, 220, 60), -1)
    cv2.line(second, (0, 0), (279, 179), (255, 255, 255), 16)
    images = [first.copy() for _ in range(5)] + [second.copy() for _ in range(5)]

    assessment = assess_source_video(_analysis(tmp_path, images, _metadata(tmp_path, duration=10.0)))

    assert assessment["accepted"] is False
    assert len(assessment["scene_analysis"]["suspected_cuts"]) == 1
    assert "hard scene cut" in assessment["hard_failures"][0]


def test_rejects_obvious_screen_recording_metadata(tmp_path: Path) -> None:
    metadata = _metadata(
        tmp_path,
        fps=59.94,
        has_audio=False,
        width=2588,
        height=1600,
    )
    assessment = assess_source_video(_analysis(tmp_path, _continuous_rotation_frames(), metadata))

    assert assessment["accepted"] is False
    assert "screen recording" in assessment["hard_failures"][0]


def test_accepts_silent_high_rate_video_with_camera_aspect(tmp_path: Path) -> None:
    metadata = _metadata(
        tmp_path,
        fps=60.0,
        has_audio=False,
        width=1920,
        height=1080,
    )
    assessment = assess_source_video(_analysis(tmp_path, _continuous_rotation_frames(), metadata))

    assert assessment["accepted"] is True
    assert assessment["hard_failures"] == []
    assert assessment["warnings"]


def test_reports_extreme_opening_transition_excluded_by_boundary(tmp_path: Path) -> None:
    opening = np.full((180, 280, 3), (30, 160, 45), dtype=np.uint8)
    capture = np.full((180, 280, 3), (210, 25, 190), dtype=np.uint8)
    cv2.rectangle(opening, (70, 45), (210, 145), (20, 220, 60), -1)
    cv2.line(capture, (0, 0), (279, 179), (255, 255, 255), 16)
    images = [opening] + [capture.copy() for _ in range(9)]

    assessment = assess_source_video(
        _analysis(tmp_path, images, _metadata(tmp_path, duration=40.0))
    )

    boundary = assessment["scene_analysis"]["boundary_excluded_transitions"]
    assert assessment["accepted"] is True
    assert assessment["scene_analysis"]["suspected_cuts"] == []
    assert len(boundary) == 1
    assert boundary[0]["boundary"] == "opening"
    assert boundary[0]["excluded_from_hard_cut"] is True
    assert "first/last 5%" in boundary[0]["exclusion_reason"]
    assert any("accepted conservatively" in warning for warning in assessment["warnings"])


def test_reports_extreme_closing_transition_excluded_by_boundary(tmp_path: Path) -> None:
    capture = np.full((180, 280, 3), (30, 160, 45), dtype=np.uint8)
    closing = np.full((180, 280, 3), (210, 25, 190), dtype=np.uint8)
    cv2.rectangle(capture, (70, 45), (210, 145), (20, 220, 60), -1)
    cv2.line(closing, (0, 0), (279, 179), (255, 255, 255), 16)
    images = [capture.copy() for _ in range(9)] + [closing]

    assessment = assess_source_video(
        _analysis(tmp_path, images, _metadata(tmp_path, duration=9.0))
    )

    boundary = assessment["scene_analysis"]["boundary_excluded_transitions"]
    assert assessment["accepted"] is True
    assert assessment["scene_analysis"]["suspected_cuts"] == []
    assert len(boundary) == 1
    assert boundary[0]["boundary"] == "closing"
    assert any("first/last 5%" in warning for warning in assessment["warnings"])


def test_rejects_boundary_cut_when_three_frames_support_each_side(tmp_path: Path) -> None:
    first = np.full((180, 280, 3), (30, 160, 45), dtype=np.uint8)
    second = np.full((180, 280, 3), (210, 25, 190), dtype=np.uint8)
    cv2.rectangle(first, (70, 45), (210, 145), (20, 220, 60), -1)
    cv2.line(second, (0, 0), (279, 179), (255, 255, 255), 16)
    images = [first.copy() for _ in range(3)] + [second.copy() for _ in range(7)]

    assessment = assess_source_video(
        _analysis(tmp_path, images, _metadata(tmp_path, duration=100.0))
    )

    cuts = assessment["scene_analysis"]["suspected_cuts"]
    assert assessment["accepted"] is False
    assert len(cuts) == 1
    assert cuts[0]["boundary"] == "opening"
    assert cuts[0]["boundary_context_override"] is True
    assert assessment["scene_analysis"]["boundary_excluded_transitions"] == []
