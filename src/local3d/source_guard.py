"""Conservative source-video guards for reconstruction inputs.

The checks in this module run after sparse ingest and before segmentation or
reconstruction.  They intentionally reject only strong evidence of an invalid
source: an obvious screen recording, or a persistent hard cut between scenes.
Normal changes caused by rotating an object, moving a hand, or exposure drift
are retained for downstream quality analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np

from .ingest import VideoAnalysis


class SourceGuardError(RuntimeError):
    """Raised when sparse source frames cannot be evaluated."""


# These thresholds have deliberately wide margins.  A cut must be extreme in
# both global color distribution and registered pixel change, and the frames on
# either side must remain different.  This avoids classifying an object flip,
# brief occlusion, or single bad exposure as a second scene.
_CUT_HISTOGRAM_DISTANCE = 0.72
_CUT_COLOR_MAD = 0.25
_CUT_CONTEXT_DISTANCE = 0.58
_BOUNDARY_CUT_CONTEXT_FRAMES = 3


def _read_analysis_frame(path: str) -> np.ndarray:
    image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise SourceGuardError(f"could not read sampled source frame: {path}")
    return cv2.resize(image, (160, 96), interpolation=cv2.INTER_AREA)


def _color_histogram(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    return cv2.normalize(histogram, None).astype(np.float32)


def _histogram_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(cv2.compareHist(left, right, cv2.HISTCMP_BHATTACHARYYA))


def _context_histogram(histograms: list[np.ndarray], start: int, stop: int) -> np.ndarray:
    average = np.mean(np.stack(histograms[start:stop], axis=0), axis=0)
    return cv2.normalize(average, None).astype(np.float32)


def _camera_aspect_error(width: int, height: int) -> float:
    ratio = max(width, height) / max(1, min(width, height))
    common_ratios = (1.0, 4.0 / 3.0, 3.0 / 2.0, 16.0 / 9.0)
    return min(abs(ratio - candidate) / candidate for candidate in common_ratios)


def assess_source_video(analysis: VideoAnalysis) -> dict[str, Any]:
    """Return a serializable, conservative reconstruction-source assessment.

    The function reads only the sparse frames already produced by ingest.  It
    does not create intermediates and does not inspect filenames, so renamed
    phone captures and screen recordings receive identical treatment.
    """

    ordered_records = sorted(analysis.frames, key=lambda frame: frame.timestamp_s)
    failures: list[str] = []
    warnings: list[str] = []

    aspect_error = _camera_aspect_error(
        analysis.metadata.display_width,
        analysis.metadata.display_height,
    )
    metadata_signals = {
        "has_audio": analysis.metadata.has_audio,
        "fps": round(float(analysis.metadata.fps), 4),
        "display_width": analysis.metadata.display_width,
        "display_height": analysis.metadata.display_height,
        "common_camera_aspect_relative_error": round(aspect_error, 4),
        "high_frame_rate": analysis.metadata.fps >= 45.0,
        "atypical_camera_aspect": aspect_error > 0.04,
    }
    if (
        not analysis.metadata.has_audio
        and metadata_signals["high_frame_rate"]
        and metadata_signals["atypical_camera_aspect"]
    ):
        failures.append(
            "metadata strongly indicates a screen recording "
            "(silent high-frame-rate video with a display-shaped aspect ratio)"
        )
    elif not analysis.metadata.has_audio and analysis.metadata.fps >= 45.0:
        warnings.append(
            "silent high-frame-rate source; accepted because its aspect ratio is camera-standard"
        )

    transitions: list[dict[str, Any]] = []
    suspected_cuts: list[dict[str, Any]] = []
    boundary_excluded_transitions: list[dict[str, Any]] = []
    if len(ordered_records) < 6:
        warnings.append("too few sparse frames for reliable multi-scene detection")
    else:
        images = [_read_analysis_frame(frame.path) for frame in ordered_records]
        histograms = [_color_histogram(image) for image in images]
        duration = max(float(analysis.metadata.duration_s), 1e-6)
        for index in range(len(images) - 1):
            timestamp = float(ordered_records[index + 1].timestamp_s)
            histogram_distance = _histogram_distance(histograms[index], histograms[index + 1])
            color_mad = float(
                np.mean(
                    np.abs(images[index].astype(np.float32) - images[index + 1].astype(np.float32))
                )
                / 255.0
            )
            before_start = max(0, index - 2)
            after_stop = min(len(histograms), index + 4)
            before = _context_histogram(histograms, before_start, index + 1)
            after = _context_histogram(histograms, index + 1, after_stop)
            context_distance = _histogram_distance(before, after)
            transition = {
                "timestamp_s": round(timestamp, 4),
                "position_fraction": round(timestamp / duration, 4),
                "histogram_distance": round(histogram_distance, 4),
                "color_mad": round(color_mad, 4),
                "context_histogram_distance": round(context_distance, 4),
            }
            transitions.append(transition)

            position_fraction = timestamp / duration
            is_interior = 0.05 <= position_fraction <= 0.95
            is_extreme = (
                histogram_distance >= _CUT_HISTOGRAM_DISTANCE
                and color_mad >= _CUT_COLOR_MAD
                and context_distance >= _CUT_CONTEXT_DISTANCE
            )
            if is_extreme:
                context_before = index + 1
                context_after = len(images) - index - 1
                has_boundary_context = (
                    context_before >= _BOUNDARY_CUT_CONTEXT_FRAMES
                    and context_after >= _BOUNDARY_CUT_CONTEXT_FRAMES
                )
                if is_interior or has_boundary_context:
                    suspected = dict(transition)
                    if not is_interior:
                        suspected.update(
                            {
                                "boundary": (
                                    "opening" if position_fraction < 0.05 else "closing"
                                ),
                                "boundary_context_override": True,
                                "context_frames_before": context_before,
                                "context_frames_after": context_after,
                            }
                        )
                    suspected_cuts.append(suspected)
                else:
                    boundary_excluded = dict(transition)
                    boundary_excluded.update(
                        {
                            "boundary": (
                                "opening" if position_fraction < 0.05 else "closing"
                            ),
                            "excluded_from_hard_cut": True,
                            "exclusion_reason": (
                                "within first/last 5% with insufficient two-sided context"
                            ),
                            "context_frames_before": context_before,
                            "context_frames_after": context_after,
                        }
                    )
                    boundary_excluded_transitions.append(boundary_excluded)

        if boundary_excluded_transitions:
            warnings.append(
                f"{len(boundary_excluded_transitions)} extreme scene transition(s) fell "
                "within the first/last 5% boundary exclusion and lacked enough two-sided "
                "context to reject; accepted conservatively and listed for review"
            )

        if suspected_cuts:
            first = suspected_cuts[0]
            failures.append(
                "persistent hard scene cut detected near "
                f"{first['timestamp_s']:.2f}s; submit one continuous object capture"
            )

    transition_histograms = [item["histogram_distance"] for item in transitions]
    transition_color_mad = [item["color_mad"] for item in transitions]
    scene_analysis = {
        "frames_examined": len(ordered_records),
        "transitions_examined": len(transitions),
        "median_histogram_distance": (
            round(float(np.median(transition_histograms)), 4) if transition_histograms else None
        ),
        "max_histogram_distance": max(transition_histograms, default=None),
        "median_color_mad": round(float(np.median(transition_color_mad)), 4) if transition_color_mad else None,
        "max_color_mad": max(transition_color_mad, default=None),
        "suspected_cuts": suspected_cuts,
        "boundary_excluded_transitions": boundary_excluded_transitions,
    }
    return {
        "accepted": not failures,
        "hard_failures": failures,
        "warnings": warnings,
        "metadata_signals": metadata_signals,
        "scene_analysis": scene_analysis,
    }
