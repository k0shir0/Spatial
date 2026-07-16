"""Automatic held-soft-object salvage from pipeline-owned frames and masks.

This is deliberately a *salvage* backend, not photogrammetry.  It selects a
clean, detailed primary view and a temporally separated, appearance-dissimilar
secondary view, normalizes both silhouettes, and inflates the primary
silhouette into a smooth bilateral volume.  Local thickness follows the 2D
distance transform, while the maximum depth is an explicit shape prior.  The
two observed views are projected onto opposite hemispheres of the result.

The public integration point is :func:`fit_soft_volume`::

    report = fit_soft_volume(frames_dir, masks_dir, candidate_output_dir)

``frames_dir`` must contain pipeline-produced ``frame_*.jpg`` (PNG is also
accepted), and ``masks_dir`` must contain matching ``frame_*.png`` masks.  The
output directory must be new or empty.  The function owns every intermediate
it creates and returns a JSON-serializable provenance report.  A caller should
only promote the candidate when the function returns successfully; every mask,
border, topology, and triangle gate raises :class:`AutoSoftError` on failure.

No learned completion or generative texture is performed here.  In particular,
the inferred sidewall, occluded geometry, deformation between views, metric
scale, and correspondence between the two projections are not observations.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from importlib.metadata import version as package_version
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import cv2
import manifold3d
import numpy as np
import trimesh

from .parametric_assets import (
    Material,
    MeshPart,
    author_usda,
    export_glb,
    package_usdz_if_available,
    render_parts,
    sha256_file,
    topology_metrics,
)
from .visual_hull import occupancy_to_mesh, taubin_smooth


MIN_VALID_VIEWS = 6
MIN_COVERAGE = 0.025
MAX_COVERAGE = 0.72
MIN_COMPONENT_FRACTION = 0.88
MIN_SOLIDITY = 0.50
MIN_EXTENT = 0.38
MIN_SUPPORT_EXTENT = 0.30
MIN_MARGIN_FRACTION = 0.012
MAX_BORDER_FRACTION = 0.001
MIN_INTERNAL_SHARPNESS = 1.0
MIN_INTERIOR_ENTROPY_BITS = 1.7
MIN_PRIMARY_MULTISCALE_DETAIL = 2.5
MIN_VIEW_DISSIMILARITY = 0.10
MARKED_FAMILY_MINIMUM = 0.80
PLAIN_FAMILY_MAXIMUM = 0.55
MIN_FAMILY_SUPPORT = 3
MAX_FAMILY_TIMESTAMP_GAP_MS = 2_500
MAX_FAMILY_ASPECT_LOG_DEVIATION = 0.28
MIN_FAMILY_MARKING_SEPARATION = 0.30
MAX_FAMILY_MEDIAN_SILHOUETTE_DISTANCE = 0.50
DEFAULT_VOLUME_RESOLUTION = 96
DEFAULT_MAX_TRIANGLES = 18_000
DEFAULT_TEXTURE_TILE = 768
NOMINAL_HEIGHT_METERS = 0.25


class AutoSoftError(RuntimeError):
    """Raised when source evidence cannot safely support a soft-volume fit."""


@dataclass
class SoftViewCandidate:
    """One frame/mask pair that passed the fail-closed source gates."""

    frame: Path
    mask_path: Path
    sequence_index: int
    frame_bgr: np.ndarray
    mask: np.ndarray
    bbox_xywh: tuple[int, int, int, int]
    coverage: float
    component_fraction: float
    solidity: float
    extent: float
    margin_fraction: float
    border_fraction: float
    sharpness: float
    detail_density: float
    interior_entropy_bits: float
    possible_skin_fraction: float
    score: float
    feature: np.ndarray
    normalized_silhouette: np.ndarray
    capture_time_ms: int = 0
    crop_aspect: float = 1.0
    multiscale_detail: float = 0.0
    marking_fraction: float = 0.0
    marking_score: float = 0.0
    rotation_features: tuple[np.ndarray, ...] = ()


@dataclass(frozen=True)
class CandidateAssessment:
    """Accepted candidate or explicit rejection reasons for provenance."""

    frame: Path
    mask: Path
    candidate: SoftViewCandidate | None
    reasons: tuple[str, ...]
    measurements: dict[str, Any]
    support_candidate: SoftViewCandidate | None = None


@dataclass(frozen=True)
class ViewSelection:
    primary: SoftViewCandidate
    secondary: SoftViewCandidate
    appearance_distance: float
    silhouette_distance: float
    combined_dissimilarity: float
    sequence_separation: int
    secondary_rotation_quarters: int = 0
    primary_family_support: int = 0
    secondary_family_support: int = 0
    primary_family_median_silhouette_distance: float = 0.0
    secondary_family_median_silhouette_distance: float = 0.0


@dataclass(frozen=True)
class VolumeEvidence:
    occupancy: np.ndarray
    silhouette: np.ndarray
    relative_thickness: np.ndarray
    canvas_aspect: float
    inferred_depth_to_height: float
    depth_slices: int
    nominal_height_m: float = NOMINAL_HEIGHT_METERS


def _strict_skin_mask(image: np.ndarray) -> np.ndarray:
    """Return only a conservative skin-color hint, never a segmentation."""

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return (
        (hsv[..., 0] <= 18)
        & (hsv[..., 1] >= 48)
        & (hsv[..., 1] <= 205)
        & (hsv[..., 2] >= 60)
        & (ycrcb[..., 1] >= 142)
        & (ycrcb[..., 1] <= 176)
        & (ycrcb[..., 2] >= 84)
        & (ycrcb[..., 2] <= 126)
    )


def _largest_component_masks(
    binary: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the raw largest component, its hole-filled silhouette, and retention."""

    raw = np.asarray(binary, dtype=np.uint8) > 0
    raw_pixels = int(raw.sum())
    if raw.ndim != 2 or raw_pixels == 0:
        empty = np.zeros(raw.shape, dtype=np.uint8)
        return empty, empty.copy(), 0.0
    count, labels, statistics, _centroids = cv2.connectedComponentsWithStats(
        raw.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        empty = np.zeros(raw.shape, dtype=np.uint8)
        return empty, empty.copy(), 0.0
    component_index = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
    component_pixels = int(statistics[component_index, cv2.CC_STAT_AREA])
    component = (labels == component_index).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(component)
    if contours:
        cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    return component, filled, float(component_pixels / raw_pixels)


def _largest_filled_component(binary: np.ndarray) -> tuple[np.ndarray, float]:
    _component, filled, retained = _largest_component_masks(binary)
    return filled, retained


def _capture_time_ms(path: Path, fallback: int) -> int:
    match = re.search(r"_(\d+)ms$", path.stem)
    return int(match.group(1)) if match else int(fallback)


def _crop_bounds(
    bbox_xywh: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
    *,
    padding_fraction: float = 0.055,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox_xywh
    pad = max(3, int(round(max(width, height) * padding_fraction)))
    image_height, image_width = image_shape[:2]
    return (
        max(0, x - pad),
        max(0, y - pad),
        min(image_width, x + width + pad),
        min(image_height, y + height + pad),
    )


def _normalize_view(
    frame: np.ndarray,
    mask: np.ndarray,
    bbox_xywh: tuple[int, int, int, int],
    *,
    size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Crop, square-normalize, and replace non-object pixels deterministically."""

    left, top, right, bottom = _crop_bounds(bbox_xywh, frame.shape)
    if right - left < 16 or bottom - top < 16:
        raise AutoSoftError("selected silhouette crop is too small")
    crop = frame[top:bottom, left:right]
    crop_mask = (mask[top:bottom, left:right] > 0).astype(np.uint8)
    canvas_aspect = float((right - left) / max(bottom - top, 1))
    interpolation = cv2.INTER_AREA if max(crop.shape[:2]) > size else cv2.INTER_LANCZOS4
    normalized = cv2.resize(crop, (size, size), interpolation=interpolation)
    normalized_mask = cv2.resize(crop_mask, (size, size), interpolation=cv2.INTER_NEAREST) > 0
    if int(normalized_mask.sum()) < size * size * 0.02:
        raise AutoSoftError("normalized silhouette retained too few object pixels")

    # A robust interior color prevents the holder/background from leaking into
    # side-projection texels.  This is flat extrapolation, not generative fill.
    erode_size = max(3, (size // 96) | 1)
    interior = cv2.erode(
        normalized_mask.astype(np.uint8), np.ones((erode_size, erode_size), np.uint8)
    ) > 0
    pixels = normalized[interior] if interior.any() else normalized[normalized_mask]
    fill = np.median(pixels, axis=0).astype(np.uint8)
    cleaned = np.where(normalized_mask[..., None], normalized, fill)
    return cleaned.astype(np.uint8), normalized_mask.astype(np.uint8), canvas_aspect


def _appearance_feature(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build a deterministic masked descriptor for opposite-view selection."""

    binary = (mask > 0).astype(np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    if binary.any():
        median = np.median(lab[binary > 0], axis=0)
    else:
        median = np.asarray([128.0, 128.0, 128.0], dtype=np.float32)
    lab = np.where(binary[..., None] > 0, lab, median)
    spatial = cv2.resize(lab, (14, 14), interpolation=cv2.INTER_AREA)
    spatial -= spatial.mean(axis=(0, 1), keepdims=True)
    spatial /= spatial.std(axis=(0, 1), keepdims=True) + 10.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    histograms: list[np.ndarray] = []
    for channel, bins, maximum in ((0, 18, 180), (1, 10, 256), (2, 10, 256)):
        histogram = cv2.calcHist([hsv], [channel], binary, [bins], [0, maximum]).ravel()
        histogram /= max(float(histogram.sum()), 1.0)
        histograms.append(histogram.astype(np.float32) * 3.0)
    silhouette = cv2.resize(binary.astype(np.float32), (14, 14), interpolation=cv2.INTER_AREA)
    feature = np.concatenate((spatial.ravel(), silhouette.ravel() * 1.5, *histograms))
    length = float(np.linalg.norm(feature))
    if length <= 1e-8:
        raise AutoSoftError("appearance descriptor was degenerate")
    return (feature / length).astype(np.float32)


def _assess_pair(
    frame_path: Path,
    mask_path: Path,
    sequence_index: int,
) -> CandidateAssessment:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    reasons: list[str] = []
    measurements: dict[str, Any] = {}
    if frame is None:
        return CandidateAssessment(frame_path, mask_path, None, ("frame_decode_failed",), measurements)
    if mask_image is None:
        return CandidateAssessment(frame_path, mask_path, None, ("mask_decode_failed",), measurements)
    if frame.shape[:2] != mask_image.shape:
        return CandidateAssessment(frame_path, mask_path, None, ("mask_size_mismatch",), measurements)
    if frame.shape[0] * frame.shape[1] > 40_000_000:
        return CandidateAssessment(frame_path, mask_path, None, ("frame_pixel_limit",), measurements)

    raw_binary = mask_image > 127
    raw_component, component, component_fraction = _largest_component_masks(raw_binary)
    object_pixels = int(component.sum())
    if object_pixels == 0:
        return CandidateAssessment(frame_path, mask_path, None, ("empty_mask",), measurements)
    image_height, image_width = component.shape
    coverage = float(object_pixels / (image_height * image_width))
    ys, xs = np.nonzero(component)
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    bbox = (left, top, right - left + 1, bottom - top + 1)
    margin_pixels = min(left, top, image_width - 1 - right, image_height - 1 - bottom)
    margin_fraction = float(margin_pixels / min(image_height, image_width))
    border_width = max(2, int(round(min(image_height, image_width) * 0.012)))
    border = np.zeros_like(component, dtype=bool)
    border[:border_width] = True
    border[-border_width:] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    border_fraction = float(np.count_nonzero((component > 0) & border) / object_pixels)

    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = float(contour_area / hull_area) if hull_area > 0 else 0.0
    extent = float(contour_area / max(bbox[2] * bbox[3], 1))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    # Exclude the silhouette boundary from the detail score.  Otherwise a
    # razor-sharp hand/background edge can outrank a recognizable but gently
    # focused face.  The retained interior still includes eyes, text, seams,
    # and other appearance evidence useful for choosing the primary view.
    erosion_radius = max(2, int(round(min(bbox[2], bbox[3]) * 0.035)))
    erosion_size = erosion_radius * 2 + 1
    interior = cv2.erode(
        raw_component, np.ones((erosion_size, erosion_size), np.uint8)
    ) > 0
    if int(interior.sum()) < max(256, object_pixels // 5):
        interior = raw_component > 0
    interior_laplacian = laplacian[interior]
    sharpness = float(np.var(interior_laplacian))
    detail_density = float(np.mean(np.abs(interior_laplacian) >= 14.0))
    histogram = cv2.calcHist([gray], [0], interior.astype(np.uint8), [32], [0, 256]).ravel()
    probabilities = histogram / max(float(histogram.sum()), 1.0)
    nonzero = probabilities[probabilities > 0]
    interior_entropy_bits = float(-np.sum(nonzero * np.log2(nonzero)))
    possible_skin_fraction = float(np.mean(_strict_skin_mask(frame)[raw_component > 0]))
    measurements.update(
        {
            "coverage": round(coverage, 6),
            "component_fraction": round(component_fraction, 6),
            "solidity": round(solidity, 6),
            "extent": round(extent, 6),
            "margin_fraction": round(margin_fraction, 6),
            "border_fraction": round(border_fraction, 6),
            "bbox_xywh": list(bbox),
            "internal_sharpness": round(sharpness, 3),
            "internal_detail_density": round(detail_density, 6),
            "interior_entropy_bits": round(interior_entropy_bits, 6),
            "possible_skin_fraction": round(possible_skin_fraction, 6),
        }
    )
    support_reasons: list[str] = []
    if not MIN_COVERAGE <= coverage <= MAX_COVERAGE:
        reasons.append("coverage_out_of_range")
        support_reasons.append("coverage_out_of_range")
    if component_fraction < MIN_COMPONENT_FRACTION:
        reasons.append("fragmented_mask")
        support_reasons.append("fragmented_mask")
    if solidity < MIN_SOLIDITY:
        reasons.append("implausibly_concave_mask")
        support_reasons.append("implausibly_concave_mask")
    if extent < MIN_EXTENT:
        reasons.append("implausibly_sparse_mask")
    if extent < MIN_SUPPORT_EXTENT:
        support_reasons.append("implausibly_sparse_support_mask")
    if min(bbox[2], bbox[3]) < 64:
        reasons.append("silhouette_too_small")
        support_reasons.append("silhouette_too_small")
    if margin_fraction < MIN_MARGIN_FRACTION:
        reasons.append("silhouette_too_close_to_frame_border")
        support_reasons.append("silhouette_too_close_to_frame_border")
    if border_fraction > MAX_BORDER_FRACTION:
        reasons.append("mask_touches_frame_border")
        support_reasons.append("mask_touches_frame_border")
    # Plush fabric and consumer-video denoising can make a valid face very
    # smooth even when its eyes, nose, and mouth remain recognizable.  Reject
    # truly flat interiors using both local edge energy and tonal entropy,
    # rather than a sharpness threshold tuned for printed rigid surfaces.
    if sharpness < MIN_INTERNAL_SHARPNESS and interior_entropy_bits < MIN_INTERIOR_ENTROPY_BITS:
        reasons.append("insufficient_detail")
    if support_reasons:
        return CandidateAssessment(frame_path, mask_path, None, tuple(reasons), measurements)

    # Retain the raw component for appearance.  Filling segmentation holes here
    # would paint holder fingers or background into the projected texture.  A
    # separately filled silhouette is used only for shape consensus/inflation.
    cleaned, normalized_mask, crop_aspect = _normalize_view(
        frame, raw_component, bbox, size=192
    )
    normalized_silhouette, _retained = _largest_filled_component(normalized_mask)
    rotation_features = tuple(
        _appearance_feature(
            np.rot90(cleaned, quarters).copy(),
            np.rot90(normalized_mask, quarters).copy(),
        )
        for quarters in range(4)
    )
    feature = rotation_features[0]

    core = cv2.erode(normalized_mask, np.ones((25, 25), np.uint8)) > 0
    if int(core.sum()) >= 1_024:
        normalized_gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dog = cv2.GaussianBlur(normalized_gray, (0, 0), 1.0) - cv2.GaussianBlur(
            normalized_gray, (0, 0), 4.0
        )
        multiscale_detail = float(np.std(dog[core]))
        normalized_lab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB).astype(np.float32)
        median_lab = np.median(normalized_lab[core], axis=0)
        lab_distance = np.linalg.norm(normalized_lab - median_lab, axis=2)
        dark_marking = (
            (lab_distance >= 15.0)
            & (normalized_lab[..., 0] <= median_lab[0] - 4.0)
            & core
        )
        marking_fraction = float(dark_marking.sum() / max(int(core.sum()), 1))
    else:
        multiscale_detail = 0.0
        marking_fraction = 0.0
    marking_score = float(
        min(1.0, multiscale_detail / 3.0, marking_fraction / 0.08)
    )
    sharpness_term = float(np.clip(np.log1p(sharpness) / np.log(1200.0), 0.0, 1.0))
    density_term = float(np.clip(detail_density / 0.16, 0.0, 1.0))
    entropy_term = float(np.clip(interior_entropy_bits / 5.0, 0.0, 1.0))
    margin_term = float(np.clip(margin_fraction / 0.10, 0.0, 1.0))
    score = (
        0.44 * sharpness_term
        + 0.20 * density_term
        + 0.13 * entropy_term
        + 0.12 * margin_term
        + 0.04 * solidity
        + 0.02 * extent
        + 0.05 * np.sqrt(coverage)
    )
    candidate = SoftViewCandidate(
        frame=frame_path,
        mask_path=mask_path,
        sequence_index=sequence_index,
        frame_bgr=frame,
        mask=component,
        bbox_xywh=bbox,
        coverage=coverage,
        component_fraction=component_fraction,
        solidity=solidity,
        extent=extent,
        margin_fraction=margin_fraction,
        border_fraction=border_fraction,
        sharpness=sharpness,
        detail_density=detail_density,
        interior_entropy_bits=interior_entropy_bits,
        possible_skin_fraction=possible_skin_fraction,
        score=float(score),
        feature=feature,
        normalized_silhouette=normalized_silhouette,
        capture_time_ms=_capture_time_ms(frame_path, sequence_index),
        crop_aspect=crop_aspect,
        multiscale_detail=multiscale_detail,
        marking_fraction=marking_fraction,
        marking_score=marking_score,
        rotation_features=rotation_features,
    )
    measurements.update(
        {
            "selection_score": round(float(score), 6),
            "crop_aspect": round(crop_aspect, 6),
            "multiscale_detail": round(multiscale_detail, 6),
            "dark_marking_fraction": round(marking_fraction, 6),
            "marking_score": round(marking_score, 6),
            "support_eligible": True,
        }
    )
    strict_candidate = candidate if not reasons else None
    return CandidateAssessment(
        frame_path,
        mask_path,
        strict_candidate,
        tuple(reasons),
        measurements,
        candidate,
    )


def _silhouette_distance(first: np.ndarray, second: np.ndarray) -> float:
    shape = (96, 96)
    a = cv2.resize((first > 0).astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST) > 0
    b = cv2.resize((second > 0).astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST) > 0
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    return float(1.0 - np.count_nonzero(a & b) / union)


def _aligned_view_metrics(
    first: SoftViewCandidate,
    second: SoftViewCandidate,
) -> tuple[float, float, int]:
    """Return appearance/silhouette distance and the best quarter-turn alignment."""

    features = second.rotation_features or (second.feature,) * 4
    choices: list[tuple[float, float, float, int]] = []
    for quarters in range(4):
        appearance = float(
            np.clip(1.0 - np.dot(first.feature, features[quarters]), 0.0, 2.0)
        )
        silhouette = _silhouette_distance(
            first.normalized_silhouette,
            np.rot90(second.normalized_silhouette, quarters),
        )
        # Silhouette is the orientation cue; appearance only breaks ties.  An
        # opposite face is expected to look different and must not be rotated
        # merely to make its pixels resemble the primary face.
        choices.append((silhouette + 0.15 * appearance, silhouette, appearance, quarters))
    _alignment_cost, silhouette, appearance, quarters = min(choices)
    return appearance, silhouette, quarters


def _temporal_runs(
    candidates: Sequence[SoftViewCandidate],
) -> list[list[SoftViewCandidate]]:
    ordered = sorted(candidates, key=lambda item: (item.capture_time_ms, item.sequence_index))
    runs: list[list[SoftViewCandidate]] = []
    for candidate in ordered:
        if (
            not runs
            or candidate.capture_time_ms - runs[-1][-1].capture_time_ms
            > MAX_FAMILY_TIMESTAMP_GAP_MS
        ):
            runs.append([candidate])
        else:
            runs[-1].append(candidate)
    return runs


def _aspect_value(candidate: SoftViewCandidate) -> float:
    aspect = max(float(candidate.crop_aspect), 1e-6)
    return max(aspect, 1.0 / aspect)


def _trim_aspect_outliers(
    candidates: Sequence[SoftViewCandidate],
) -> tuple[list[SoftViewCandidate], float]:
    aspects = np.asarray([_aspect_value(item) for item in candidates], dtype=np.float64)
    median = float(np.median(aspects))
    retained = [
        item
        for item, aspect in zip(candidates, aspects, strict=True)
        if abs(math.log(float(aspect) / median)) <= MAX_FAMILY_ASPECT_LOG_DEVIATION
    ]
    return retained, median


def _stable_family_run(
    support: Sequence[SoftViewCandidate],
    strict_frames: set[Path],
    *,
    label: str,
) -> tuple[list[SoftViewCandidate], float]:
    choices: list[tuple[tuple[float, ...], list[SoftViewCandidate], float]] = []
    for run in _temporal_runs(support):
        retained, median_aspect = _trim_aspect_outliers(run)
        strict_count = sum(item.frame in strict_frames for item in retained)
        if len(retained) < MIN_FAMILY_SUPPORT or strict_count == 0:
            continue
        median_quality = float(np.median([item.score for item in retained]))
        duration = retained[-1].capture_time_ms - retained[0].capture_time_ms
        key = (
            float(len(retained)),
            float(strict_count),
            median_quality,
            float(duration),
            float(-retained[0].capture_time_ms),
        )
        choices.append((key, retained, median_aspect))
    if not choices:
        raise AutoSoftError(
            f"no {label} appearance family retained {MIN_FAMILY_SUPPORT} "
            "temporally coherent, aspect-consistent supporting views"
        )
    _key, retained, median_aspect = max(choices, key=lambda item: item[0])
    return retained, median_aspect


def _family_median_silhouette_distance(
    candidates: Sequence[SoftViewCandidate],
) -> float:
    distances: list[float] = []
    for first, second in combinations(candidates, 2):
        _appearance, silhouette, _quarters = _aligned_view_metrics(first, second)
        distances.append(silhouette)
    return float(np.median(distances)) if distances else 1.0


def _select_views(
    candidates: Sequence[SoftViewCandidate],
    support_candidates: Sequence[SoftViewCandidate] | None = None,
) -> ViewSelection:
    if len(candidates) < MIN_VALID_VIEWS:
        raise AutoSoftError(
            f"only {len(candidates)} frames passed the soft-object source gates; "
            f"need at least {MIN_VALID_VIEWS}"
        )
    support = list(support_candidates or candidates)
    marked = [item for item in support if item.marking_score >= MARKED_FAMILY_MINIMUM]
    plain = [item for item in support if item.marking_score <= PLAIN_FAMILY_MAXIMUM]
    if len(marked) < MIN_FAMILY_SUPPORT or len(plain) < MIN_FAMILY_SUPPORT:
        raise AutoSoftError(
            "soft views do not contain two supported marked/plain appearance families "
            f"(marked={len(marked)}, plain={len(plain)}, need {MIN_FAMILY_SUPPORT} each)"
        )
    marking_separation = float(
        np.median([item.marking_score for item in marked])
        - np.median([item.marking_score for item in plain])
    )
    if marking_separation < MIN_FAMILY_MARKING_SEPARATION:
        raise AutoSoftError(
            "soft appearance families are not separated enough to label opposite projections "
            f"({marking_separation:.3f} < {MIN_FAMILY_MARKING_SEPARATION:.3f})"
        )

    strict_frames = {item.frame for item in candidates}
    primary_family, primary_aspect = _stable_family_run(
        marked, strict_frames, label="marked primary"
    )
    secondary_family, secondary_aspect = _stable_family_run(
        plain, strict_frames, label="plain secondary"
    )
    primary_consensus = _family_median_silhouette_distance(primary_family)
    secondary_consensus = _family_median_silhouette_distance(secondary_family)
    if primary_consensus > MAX_FAMILY_MEDIAN_SILHOUETTE_DISTANCE:
        raise AutoSoftError(
            "marked primary family failed silhouette consensus "
            f"({primary_consensus:.3f} > {MAX_FAMILY_MEDIAN_SILHOUETTE_DISTANCE:.3f})"
        )
    if secondary_consensus > MAX_FAMILY_MEDIAN_SILHOUETTE_DISTANCE:
        raise AutoSoftError(
            "plain secondary family failed silhouette consensus "
            f"({secondary_consensus:.3f} > {MAX_FAMILY_MEDIAN_SILHOUETTE_DISTANCE:.3f})"
        )

    primary_choices: list[tuple[float, SoftViewCandidate]] = []
    for item in primary_family:
        if item.frame not in strict_frames or item.multiscale_detail < MIN_PRIMARY_MULTISCALE_DETAIL:
            continue
        aspect_penalty = float(
            np.clip(
                abs(math.log(_aspect_value(item) / primary_aspect))
                / MAX_FAMILY_ASPECT_LOG_DEVIATION,
                0.0,
                1.0,
            )
        )
        primary_choices.append((item.score - 0.05 * aspect_penalty, item))
    if not primary_choices:
        raise AutoSoftError(
            "marked family has no strictly accepted, structurally detailed primary view"
        )
    _primary_quality, primary = max(
        primary_choices,
        key=lambda item: (item[0], item[1].score, -item[1].sequence_index),
    )

    secondary_choices: list[tuple[float, SoftViewCandidate]] = []
    for item in secondary_family:
        if item.frame not in strict_frames:
            continue
        mask_area = float(np.mean(item.normalized_silhouette > 0))
        aspect_penalty = float(
            np.clip(
                abs(math.log(_aspect_value(item) / secondary_aspect))
                / MAX_FAMILY_ASPECT_LOG_DEVIATION,
                0.0,
                1.0,
            )
        )
        quality = (
            0.35 * item.solidity
            + 0.25 * item.extent
            + 0.20 * float(np.clip(item.margin_fraction / 0.10, 0.0, 1.0))
            + 0.10 * float(np.clip(mask_area / 0.55, 0.0, 1.0))
            + 0.10 * (1.0 - float(np.clip(item.multiscale_detail / 3.0, 0.0, 1.0)))
            - 0.05 * aspect_penalty
        )
        secondary_choices.append((quality, item))
    if not secondary_choices:
        raise AutoSoftError("plain family has no strictly accepted secondary view")
    _secondary_quality, secondary = max(
        secondary_choices,
        key=lambda item: (item[0], item[1].score, -item[1].sequence_index),
    )

    appearance, silhouette, secondary_rotation = _aligned_view_metrics(primary, secondary)
    dissimilarity = 0.74 * appearance + 0.26 * silhouette
    if dissimilarity < MIN_VIEW_DISSIMILARITY:
        raise AutoSoftError(
            "the supported secondary family is not visually distinct enough to label as an opposite projection "
            f"({dissimilarity:.3f} < {MIN_VIEW_DISSIMILARITY:.3f})"
        )
    separation = abs(secondary.sequence_index - primary.sequence_index)
    return ViewSelection(
        primary=primary,
        secondary=secondary,
        appearance_distance=appearance,
        silhouette_distance=silhouette,
        combined_dissimilarity=dissimilarity,
        sequence_separation=separation,
        secondary_rotation_quarters=secondary_rotation,
        primary_family_support=len(primary_family),
        secondary_family_support=len(secondary_family),
        primary_family_median_silhouette_distance=primary_consensus,
        secondary_family_median_silhouette_distance=secondary_consensus,
    )


def _build_volume_evidence(
    normalized_mask: np.ndarray,
    canvas_aspect: float,
    *,
    resolution: int,
) -> VolumeEvidence:
    if not 72 <= resolution <= 192:
        raise ValueError("volume resolution must be between 72 and 192")
    if not 0.25 <= canvas_aspect <= 4.0:
        raise AutoSoftError(f"silhouette crop aspect is implausible: {canvas_aspect:.3f}")
    silhouette = cv2.resize(
        (normalized_mask > 0).astype(np.uint8),
        (resolution, resolution),
        interpolation=cv2.INTER_NEAREST,
    )
    silhouette, fraction = _largest_filled_component(silhouette)
    if fraction < 0.98 or not silhouette.any():
        raise AutoSoftError("normalized primary silhouette is fragmented")
    silhouette = cv2.morphologyEx(
        silhouette, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(silhouette)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 1, -1)
    silhouette = filled
    distance = cv2.distanceTransform(silhouette, cv2.DIST_L2, 5)
    maximum_distance = float(distance.max())
    if maximum_distance < 2.5:
        raise AutoSoftError("primary silhouette is too thin to inflate robustly")

    # The medial radius is the only depth cue used.  The coefficient and clamps
    # are an explicit soft-object prior, recorded in the report as inferred.
    inferred_depth = float(
        np.clip(1.60 * maximum_distance / resolution, 0.24, 0.62)
    )
    depth_slices = max(32, int(round(resolution * inferred_depth)))
    normalized_distance = np.clip(distance / maximum_distance, 0.0, 1.0)
    relative_thickness = np.where(
        silhouette > 0,
        0.055 + 0.945 * np.power(normalized_distance, 0.62),
        0.0,
    ).astype(np.float32)
    z = np.linspace(-1.0, 1.0, depth_slices, dtype=np.float32)[:, None, None]
    occupancy = (silhouette[None, ...] > 0) & (np.abs(z) <= relative_thickness[None, ...])
    if int(occupancy.sum()) < 4_000:
        raise AutoSoftError("inferred soft volume contains too few occupied samples")
    return VolumeEvidence(
        occupancy=occupancy,
        silhouette=silhouette,
        relative_thickness=relative_thickness,
        canvas_aspect=canvas_aspect,
        inferred_depth_to_height=inferred_depth,
        depth_slices=depth_slices,
    )


def _simplify_manifold(mesh: trimesh.Trimesh, face_count: int) -> trimesh.Trimesh:
    """Tolerance-simplify without changing the source's closed genus-zero topology."""

    source = manifold3d.Manifold(
        manifold3d.Mesh(
            np.ascontiguousarray(mesh.vertices, dtype=np.float32),
            np.ascontiguousarray(mesh.faces, dtype=np.uint32),
        )
    )
    if source.status() != manifold3d.Error.NoError or source.is_empty():
        raise AutoSoftError(f"manifold conversion failed: {source.status()}")
    if len(source.decompose()) != 1 or source.genus() != 0:
        raise AutoSoftError("source soft mesh is not one genus-zero manifold")

    diagonal = max(float(np.linalg.norm(mesh.extents)), 1e-6)
    low, high = 0.0, diagonal * 1e-6
    best: manifold3d.Manifold | None = None
    for _ in range(32):
        trial = source.simplify(high)
        if trial.status() != manifold3d.Error.NoError:
            raise AutoSoftError(f"manifold simplification failed: {trial.status()}")
        if trial.num_tri() <= face_count:
            best = trial
            break
        low, high = high, high * 2.0
    if best is None:
        raise AutoSoftError(
            f"could not bracket a {face_count:,}-triangle topology-preserving simplification"
        )

    for _ in range(32):
        middle = (low + high) * 0.5
        trial = source.simplify(middle)
        if trial.status() != manifold3d.Error.NoError:
            raise AutoSoftError(f"manifold simplification failed: {trial.status()}")
        if trial.num_tri() > face_count:
            low = middle
        else:
            high = middle
            if trial.num_tri() > best.num_tri():
                best = trial
    if (
        best.num_tri() > face_count
        or len(best.decompose()) != 1
        or best.genus() != 0
    ):
        raise AutoSoftError("simplified soft mesh violated manifold topology or ceiling")

    packed = best.to_mesh()
    result = trimesh.Trimesh(
        vertices=np.asarray(packed.vert_properties, dtype=np.float32)[:, :3].copy(),
        faces=np.asarray(packed.tri_verts, dtype=np.int32).copy(),
        process=False,
    )
    result.remove_unreferenced_vertices()
    return result


def _mesh_from_volume(
    evidence: VolumeEvidence,
    *,
    max_triangles: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 2_000 <= max_triangles <= 100_000:
        raise ValueError("max_triangles must be between 2,000 and 100,000")
    vertices, faces = occupancy_to_mesh(evidence.occupancy)
    vertices[:, 0] *= evidence.canvas_aspect * evidence.nominal_height_m * 0.5
    vertices[:, 1] *= -evidence.nominal_height_m * 0.5
    vertices[:, 2] *= evidence.inferred_depth_to_height * evidence.nominal_height_m * 0.5
    vertices = taubin_smooth(vertices, faces, iterations=6, lamb=0.42, mu=-0.44)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    if float(mesh.volume) < 0.0:
        mesh.invert()
    if (
        not mesh.is_watertight
        or not mesh.is_winding_consistent
        or int(mesh.body_count) != 1
        or int(mesh.euler_number) != 2
    ):
        raise AutoSoftError("marching-cubes source failed closed genus-zero topology gates")
    source_volume = float(mesh.volume)
    if len(mesh.faces) > max_triangles:
        mesh = _simplify_manifold(mesh, max_triangles)
    if float(mesh.volume) < 0.0:
        mesh.invert()
    if len(mesh.faces) < 500:
        raise AutoSoftError("soft volume topology is too coarse")
    if len(mesh.faces) > max_triangles:
        raise AutoSoftError(
            f"soft volume exceeded the {max_triangles:,}-triangle hard ceiling"
        )
    if not mesh.is_watertight:
        raise AutoSoftError("soft volume failed the watertight topology gate")
    if not mesh.is_winding_consistent:
        raise AutoSoftError("soft volume failed the winding-consistency gate")
    if int(mesh.body_count) != 1 or int(mesh.euler_number) != 2:
        raise AutoSoftError(
            "soft volume must be one genus-zero body "
            f"(bodies={mesh.body_count}, euler={mesh.euler_number})"
        )
    if not np.isfinite(mesh.vertices).all() or float(mesh.volume) <= 1e-8:
        raise AutoSoftError("soft volume is non-finite or degenerate")
    relative_volume_drift = abs(float(mesh.volume) - source_volume) / max(source_volume, 1e-12)
    if relative_volume_drift > 0.05:
        raise AutoSoftError(
            "soft simplification exceeded the 5% volume-drift gate "
            f"({relative_volume_drift:.3%})"
        )
    return np.asarray(mesh.vertices, dtype=np.float32), np.asarray(mesh.faces, dtype=np.int32)


def _projected_parts(
    vertices: np.ndarray,
    faces: np.ndarray,
    evidence: VolumeEvidence,
    atlas: np.ndarray,
    material: Material,
) -> list[MeshPart]:
    face_centers_z = vertices[faces, 2].mean(axis=1)
    groups = (("SoftPrimaryProjection", face_centers_z >= 0.0, False),
              ("SoftSecondaryProjection", face_centers_z < 0.0, True))
    parts: list[MeshPart] = []
    atlas_height, atlas_width = atlas.shape[:2]
    inset_u = 0.75 / atlas_width
    inset_v = 0.75 / atlas_height
    width_m = evidence.canvas_aspect * evidence.nominal_height_m
    for name, selected, mirror in groups:
        selected_faces = faces[selected]
        if len(selected_faces) < 50:
            raise AutoSoftError(f"{name} received too few triangles")
        indices = np.unique(selected_faces)
        remap = np.full(len(vertices), -1, dtype=np.int32)
        remap[indices] = np.arange(len(indices), dtype=np.int32)
        part_vertices = vertices[indices]
        part_faces = remap[selected_faces]
        local_x = np.clip(part_vertices[:, 0] / width_m + 0.5, 0.0, 1.0)
        local_y = np.clip(
            part_vertices[:, 1] / evidence.nominal_height_m + 0.5, 0.0, 1.0
        )
        if mirror:
            local_x = 1.0 - local_x
            u = 0.5 + inset_u + local_x * (0.5 - 2.0 * inset_u)
        else:
            u = inset_u + local_x * (0.5 - 2.0 * inset_u)
        v = inset_v + local_y * (1.0 - 2.0 * inset_v)
        uv = np.column_stack((u, v)).astype(np.float32)
        if not np.isfinite(uv).all() or np.any(uv < 0.0) or np.any(uv > 1.0):
            raise AutoSoftError("projected soft-object UVs failed their finite atlas gate")
        parts.append(
            MeshPart(
                name=name,
                vertices=part_vertices,
                faces=part_faces,
                material=material,
                uv=uv,
                texture_key="soft_atlas",
                texture_bgr=atlas,
            )
        )
    return parts


def _validate_exported_soft_glb(
    path: Path,
    *,
    expected_triangles: int,
    expected_vertices_after_position_merge: int,
    expected_material: str,
) -> dict[str, Any]:
    """Reload a GLB and gate its positional topology across intentional UV seams.

    glTF indexes position, normal, and UV attributes together.  The primary and
    secondary projections therefore reload as separate open primitives at their
    intentional atlas seam even though their boundary positions describe one
    closed surface.  This gate bakes scene transforms, validates both textured
    projection primitives, then merges coincident positions while explicitly
    ignoring UV and normal differences before measuring delivered topology.
    """

    try:
        loaded = trimesh.load(path, force="scene", process=False)
    except Exception as error:
        raise AutoSoftError(f"could not reload exported soft GLB: {error}") from error
    if not isinstance(loaded, trimesh.Scene):
        raise AutoSoftError("exported soft GLB did not reload as a scene")

    expected_names = {"SoftPrimaryProjection", "SoftSecondaryProjection"}
    transformed: list[trimesh.Trimesh] = []
    primitive_records: list[dict[str, Any]] = []
    for node_name in sorted(loaded.graph.nodes_geometry):
        transform, geometry_name = loaded.graph[node_name]
        geometry = loaded.geometry.get(geometry_name)
        if not isinstance(geometry, trimesh.Trimesh):
            raise AutoSoftError(f"exported GLB node {node_name!r} is not a triangle mesh")
        mesh = geometry.copy()
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise AutoSoftError(f"exported GLB node {node_name!r} has a non-finite transform")
        mesh.apply_transform(matrix)
        if not np.isfinite(mesh.vertices).all():
            raise AutoSoftError(f"exported GLB node {node_name!r} has non-finite vertices")

        uv_raw = getattr(mesh.visual, "uv", None)
        if uv_raw is None:
            raise AutoSoftError(f"exported GLB node {node_name!r} lost projected UV coordinates")
        uv = np.asarray(uv_raw, dtype=np.float64)
        if uv.shape != (len(mesh.vertices), 2) or not np.isfinite(uv).all():
            raise AutoSoftError(f"exported GLB node {node_name!r} has invalid projected UVs")
        if np.any(uv < -1e-7) or np.any(uv > 1.0 + 1e-7):
            raise AutoSoftError(f"exported GLB node {node_name!r} has out-of-atlas UVs")

        material = getattr(mesh.visual, "material", None)
        material_name = getattr(material, "name", None)
        texture = getattr(material, "baseColorTexture", None)
        if material_name != expected_material or texture is None:
            raise AutoSoftError(
                f"exported GLB node {node_name!r} lost its {expected_material!r} texture material"
            )
        texture_size = getattr(texture, "size", None)
        if (
            not isinstance(texture_size, tuple)
            or len(texture_size) != 2
            or min(int(value) for value in texture_size) <= 0
        ):
            raise AutoSoftError(f"exported GLB node {node_name!r} has an invalid embedded texture")

        u_min, v_min = np.min(uv, axis=0)
        u_max, v_max = np.max(uv, axis=0)
        primitive_records.append(
            {
                "node": str(node_name),
                "geometry": str(geometry_name),
                "vertices": int(len(mesh.vertices)),
                "triangles": int(len(mesh.faces)),
                "watertight_individually": bool(mesh.is_watertight),
                "material": str(material_name),
                "embedded_texture": True,
                "texture_dimensions_px": [int(texture_size[0]), int(texture_size[1])],
                "uv_bounds": [
                    [round(float(u_min), 8), round(float(v_min), 8)],
                    [round(float(u_max), 8), round(float(v_max), 8)],
                ],
            }
        )
        transformed.append(mesh)

    names = {record["geometry"] for record in primitive_records}
    if names != expected_names or len(transformed) != 2:
        raise AutoSoftError(
            "exported GLB must retain exactly the primary and secondary projection primitives"
        )
    by_name = {record["geometry"]: record for record in primitive_records}
    if by_name["SoftPrimaryProjection"]["uv_bounds"][1][0] >= 0.5:
        raise AutoSoftError("primary projection UVs escaped the left atlas tile")
    if by_name["SoftSecondaryProjection"]["uv_bounds"][0][0] <= 0.5:
        raise AutoSoftError("secondary projection UVs escaped the right atlas tile")

    joined = trimesh.util.concatenate(transformed)
    triangles_before_merge = int(len(joined.faces))
    vertices_before_merge = int(len(joined.vertices))
    raw_watertight = bool(joined.is_watertight)
    raw_body_count = int(joined.body_count)
    joined.merge_vertices(
        merge_tex=True,
        merge_norm=True,
        digits_vertex=8,
    )
    joined.remove_unreferenced_vertices()
    if triangles_before_merge != expected_triangles or int(len(joined.faces)) != expected_triangles:
        raise AutoSoftError(
            "exported soft GLB triangle count changed across serialization "
            f"({len(joined.faces)} != {expected_triangles})"
        )
    if int(len(joined.vertices)) != expected_vertices_after_position_merge:
        raise AutoSoftError(
            "exported soft GLB position-welded vertex count changed across serialization "
            f"({len(joined.vertices)} != {expected_vertices_after_position_merge})"
        )
    if (
        not joined.is_watertight
        or not joined.is_winding_consistent
        or int(joined.body_count) != 1
        or int(joined.euler_number) != 2
    ):
        raise AutoSoftError(
            "exported soft GLB failed position-welded genus-zero topology gates "
            f"(watertight={joined.is_watertight}, winding={joined.is_winding_consistent}, "
            f"bodies={joined.body_count}, euler={joined.euler_number})"
        )
    if not np.isfinite(joined.vertices).all() or float(joined.volume) <= 1e-8:
        raise AutoSoftError("exported soft GLB is non-finite or degenerate after position welding")

    return {
        "quality_gate_passed": True,
        "topology_basis": (
            "GLB reloaded with scene transforms baked; projection primitives concatenated; "
            "coincident positions merged to 8 decimal digits while ignoring UV/normal seams"
        ),
        "attribute_indexed_before_position_merge": {
            "vertices": vertices_before_merge,
            "triangles": triangles_before_merge,
            "watertight": raw_watertight,
            "body_count": raw_body_count,
            "interpretation": (
                "open projection primitives are expected at the intentional atlas UV seam"
            ),
        },
        "position_welded": {
            "vertices": int(len(joined.vertices)),
            "triangles": int(len(joined.faces)),
            "watertight": bool(joined.is_watertight),
            "winding_consistent": bool(joined.is_winding_consistent),
            "body_count": int(joined.body_count),
            "euler_number": int(joined.euler_number),
            "volume_m3_signed": round(float(joined.volume), 12),
        },
        "projection_primitives": primitive_records,
    }


def _labeled_card(image: np.ndarray, label: str, *, size: int = 480) -> np.ndarray:
    canvas = np.full((size, size, 3), (242, 240, 235), dtype=np.uint8)
    available = size - 54
    scale = min(available / image.shape[1], available / image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    x = (size - width) // 2
    y = 44 + (available - height) // 2
    canvas[y : y + height, x : x + width] = resized
    cv2.rectangle(canvas, (0, 0), (size, 42), (242, 240, 235), -1)
    cv2.putText(
        canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
        (42, 45, 40), 2, cv2.LINE_AA,
    )
    return canvas


def _write_qa_contact(
    parts: Sequence[MeshPart],
    primary: np.ndarray,
    secondary: np.ndarray,
    evidence: VolumeEvidence,
    destination: Path,
) -> None:
    directions = (
        ((0.48, 0.22, 1.0), "primary three-quarter"),
        ((1.0, 0.12, 0.05), "inferred profile"),
        ((-0.48, 0.22, -1.0), "secondary three-quarter"),
    )
    model_cards = [
        _labeled_card(render_parts(parts, direction), label)
        for direction, label in directions
    ]
    thickness = np.clip(evidence.relative_thickness * 255.0, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(thickness, cv2.COLORMAP_VIRIDIS)
    heatmap[evidence.silhouette == 0] = (242, 240, 235)
    evidence_cards = [
        _labeled_card(primary, "observed primary projection"),
        _labeled_card(secondary, "observed secondary projection"),
        _labeled_card(heatmap, "inferred thickness prior"),
    ]
    contact = np.vstack((np.hstack(model_cards), np.hstack(evidence_cards)))
    if not cv2.imwrite(str(destination), contact, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise AutoSoftError(f"could not write QA contact sheet {destination}")


def _candidate_record(candidate: SoftViewCandidate) -> dict[str, Any]:
    return {
        "frame": str(candidate.frame),
        "mask": str(candidate.mask_path),
        "sequence_index": candidate.sequence_index,
        "coverage": round(candidate.coverage, 6),
        "component_fraction": round(candidate.component_fraction, 6),
        "solidity": round(candidate.solidity, 6),
        "extent": round(candidate.extent, 6),
        "margin_fraction": round(candidate.margin_fraction, 6),
        "border_fraction": round(candidate.border_fraction, 6),
        "internal_sharpness": round(candidate.sharpness, 3),
        "internal_detail_density": round(candidate.detail_density, 6),
        "interior_entropy_bits": round(candidate.interior_entropy_bits, 6),
        "possible_skin_fraction": round(candidate.possible_skin_fraction, 6),
        "crop_aspect": round(candidate.crop_aspect, 6),
        "multiscale_detail": round(candidate.multiscale_detail, 6),
        "dark_marking_fraction": round(candidate.marking_fraction, 6),
        "marking_score": round(candidate.marking_score, 6),
        "selection_score": round(candidate.score, 6),
        "frame_sha256": sha256_file(candidate.frame),
        "mask_sha256": sha256_file(candidate.mask_path),
    }


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _paired_inputs(frames_directory: Path, masks_directory: Path) -> list[tuple[Path, Path]]:
    supported = {".jpg", ".jpeg", ".png"}
    frames = sorted(
        path for path in frames_directory.glob("frame_*")
        if path.is_file() and path.suffix.lower() in supported
    )
    pairs = [(frame, masks_directory / f"{frame.stem}.png") for frame in frames]
    if not pairs:
        raise AutoSoftError(f"no pipeline frame_*-named images found in {frames_directory}")
    missing = [mask for _frame, mask in pairs if not mask.is_file()]
    if missing:
        raise AutoSoftError(
            f"{len(missing)} frame(s) have no matching pipeline mask; first missing: {missing[0]}"
        )
    return pairs


def fit_soft_volume(
    frames_directory: Path,
    masks_directory: Path,
    output_directory: Path,
    *,
    allow_usdz: bool = True,
    volume_resolution: int = DEFAULT_VOLUME_RESOLUTION,
    max_triangles: int = DEFAULT_MAX_TRIANGLES,
) -> dict[str, Any]:
    """Build and validate an automatic soft-volume salvage candidate.

    Inputs must be pipeline-owned frame/mask directories.  The function does
    not accept selected frames, quads, depth, dimensions, or an existing mesh,
    which keeps selection and every intermediate inside one front-to-end run.
    """

    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except AttributeError:
        pass
    frames_directory = frames_directory.expanduser().resolve()
    masks_directory = masks_directory.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    if not frames_directory.is_dir() or not masks_directory.is_dir():
        raise AutoSoftError("frames_directory and masks_directory must both exist")
    if output_directory in {frames_directory, masks_directory}:
        raise AutoSoftError("soft-object output must be separate from pipeline input directories")
    if output_directory.exists():
        if not output_directory.is_dir():
            raise AutoSoftError("soft-object candidate output path is not a directory")
        if any(output_directory.iterdir()):
            raise AutoSoftError("soft-object candidate output directory must be empty")

    pairs = _paired_inputs(frames_directory, masks_directory)
    assessments = [
        _assess_pair(frame, mask, index) for index, (frame, mask) in enumerate(pairs)
    ]
    candidates = [item.candidate for item in assessments if item.candidate is not None]
    support_candidates = [
        item.support_candidate
        for item in assessments
        if item.support_candidate is not None
    ]
    selection = _select_views(candidates, support_candidates)

    primary_clean, primary_mask, primary_aspect = _normalize_view(
        selection.primary.frame_bgr,
        selection.primary.mask,
        selection.primary.bbox_xywh,
        size=DEFAULT_TEXTURE_TILE,
    )
    secondary_clean, secondary_mask, _secondary_aspect = _normalize_view(
        selection.secondary.frame_bgr,
        selection.secondary.mask,
        selection.secondary.bbox_xywh,
        size=DEFAULT_TEXTURE_TILE,
    )
    if selection.secondary_rotation_quarters:
        secondary_clean = np.rot90(
            secondary_clean, selection.secondary_rotation_quarters
        ).copy()
        secondary_mask = np.rot90(
            secondary_mask, selection.secondary_rotation_quarters
        ).copy()
    evidence = _build_volume_evidence(
        primary_mask, primary_aspect, resolution=volume_resolution
    )
    vertices, faces = _mesh_from_volume(evidence, max_triangles=max_triangles)
    atlas = np.hstack((primary_clean, secondary_clean))
    observed_pixels = np.concatenate(
        (
            primary_clean[primary_mask > 0],
            secondary_clean[secondary_mask > 0],
        ),
        axis=0,
    )
    median_bgr = np.median(observed_pixels, axis=0).astype(np.uint8)
    material = Material(
        "soft_surface",
        tuple(int(channel) for channel in median_bgr[::-1]),
        metallic=0.0,
        roughness=0.88,
    )
    parts = _projected_parts(vertices, faces, evidence, atlas, material)
    topology = topology_metrics(parts)
    if not topology["watertight"] or not topology["winding_consistent"]:
        raise AutoSoftError(f"export parts failed topology revalidation: {topology}")
    if topology["body_count"] != 1 or topology["euler_number"] != 2:
        raise AutoSoftError(f"export parts are not one genus-zero body: {topology}")
    if topology["triangles"] > max_triangles:
        raise AutoSoftError("export parts exceeded the triangle hard ceiling")

    output_directory.mkdir(parents=True, exist_ok=True)
    primary_path = output_directory / "soft_primary_clean.png"
    secondary_path = output_directory / "soft_secondary_clean.png"
    atlas_path = output_directory / "soft_texture_atlas.png"
    for path, image in (
        (primary_path, primary_clean),
        (secondary_path, secondary_clean),
        (atlas_path, atlas),
    ):
        if not cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise AutoSoftError(f"could not write texture intermediate {path}")

    glb_path = output_directory / "soft_model.glb"
    usda_path = output_directory / "soft_model.usda"
    usdz_path = output_directory / "soft_model.usdz"
    texture_paths = {"soft_atlas": atlas_path}
    export_glb(parts, glb_path, texture_paths, "soft_model")
    glb_validation = _validate_exported_soft_glb(
        glb_path,
        expected_triangles=int(topology["triangles"]),
        expected_vertices_after_position_merge=int(topology["vertices_after_seam_merge"]),
        expected_material=material.key,
    )
    author_usda(parts, usda_path, texture_paths, "soft_model")
    usd_package = package_usdz_if_available(
        usda_path, usdz_path, enabled=allow_usdz
    )
    qa_path = output_directory / "qa_soft_contact.png"
    _write_qa_contact(parts, primary_clean, secondary_clean, evidence, qa_path)

    rejection_counts = Counter(
        reason for assessment in assessments for reason in assessment.reasons
    )
    artifacts = [primary_path, secondary_path, atlas_path, glb_path, usda_path, qa_path]
    if usd_package.get("created"):
        artifacts.append(usdz_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": None,
        "created_utc_policy": "omitted for deterministic provenance",
        "method": "automatic held-soft-object silhouette inflation salvage",
        "classification": "inferred bilateral 2.5D soft volume; not recovered photogrammetry",
        "source": {
            "frames_directory": str(frames_directory),
            "masks_directory": str(masks_directory),
            "input_pairs": len(pairs),
            "accepted_views": len(candidates),
            "support_views": len(support_candidates),
            "rejected_views": len(assessments) - len(candidates),
            "rejection_reason_counts": dict(sorted(rejection_counts.items())),
            "assessments": [
                {
                    "frame": str(item.frame),
                    "mask": str(item.mask),
                    "frame_sha256": sha256_file(item.frame),
                    "mask_sha256": sha256_file(item.mask),
                    "accepted": item.candidate is not None,
                    "reasons": list(item.reasons),
                    "measurements": item.measurements,
                }
                for item in assessments
            ],
        },
        "selection": {
            "primary": _candidate_record(selection.primary),
            "secondary": _candidate_record(selection.secondary),
            "appearance_distance": round(selection.appearance_distance, 6),
            "silhouette_distance": round(selection.silhouette_distance, 6),
            "combined_dissimilarity": round(selection.combined_dissimilarity, 6),
            "sequence_separation": selection.sequence_separation,
            "secondary_quarter_turns": selection.secondary_rotation_quarters,
            "primary_family_support": selection.primary_family_support,
            "secondary_family_support": selection.secondary_family_support,
            "primary_family_median_silhouette_distance": round(
                selection.primary_family_median_silhouette_distance, 6
            ),
            "secondary_family_median_silhouette_distance": round(
                selection.secondary_family_median_silhouette_distance, 6
            ),
            "front_back_semantics": (
                "primary/secondary are stable projection labels only; semantic front/back was not inferred"
            ),
        },
        "geometry": {
            "nominal_height_m": evidence.nominal_height_m,
            "absolute_scale": "ambiguous; standardized to a 0.25 m nominal height for viewing",
            "canvas_width_to_height": round(evidence.canvas_aspect, 6),
            "inferred_depth_to_height": round(evidence.inferred_depth_to_height, 6),
            "depth_inference": (
                "1.60 times primary-silhouette medial radius/height, clamped to [0.24, 0.62]"
            ),
            "local_thickness": (
                "distance-transform power prior: 0.055 + 0.945 * normalized_distance^0.62"
            ),
            "volume_resolution_xy": [
                int(evidence.silhouette.shape[1]), int(evidence.silhouette.shape[0])
            ],
            "volume_resolution_depth": evidence.depth_slices,
            "triangle_ceiling": max_triangles,
            "topology": topology | {
                "basis": (
                    "pre-export projection parts concatenated and welded by position; "
                    "UV/material attributes are excluded from this source topology measurement"
                )
            },
            "exported_glb_validation": glb_validation,
        },
        "appearance": {
            "atlas": str(atlas_path),
            "projection": (
                "primary pixels project to +Z triangles; the automatically quarter-turn-aligned, "
                "mirrored secondary pixels project to -Z triangles"
            ),
            "outside_mask_fill": (
                "flat median of observed interior pixels; no learned or generative completion"
            ),
            "unobserved_sides": (
                "nearest hemisphere projection only; side appearance is not directly observed or view-fused"
            ),
            "projection_seam": (
                "primary and secondary remain distinct textured GLB/USD projection primitives; "
                "each primitive is open at the intentional atlas UV seam, while their transformed, "
                "position-welded union is one closed surface"
            ),
        },
        "hard_gates": {
            "minimum_valid_views": MIN_VALID_VIEWS,
            "coverage_range": [MIN_COVERAGE, MAX_COVERAGE],
            "minimum_largest_component_fraction": MIN_COMPONENT_FRACTION,
            "minimum_solidity": MIN_SOLIDITY,
            "minimum_extent": MIN_EXTENT,
            "minimum_support_extent": MIN_SUPPORT_EXTENT,
            "minimum_source_border_margin_fraction": MIN_MARGIN_FRACTION,
            "maximum_mask_border_fraction": MAX_BORDER_FRACTION,
            "minimum_view_dissimilarity": MIN_VIEW_DISSIMILARITY,
            "minimum_internal_laplacian_variance": MIN_INTERNAL_SHARPNESS,
            "minimum_interior_entropy_bits": MIN_INTERIOR_ENTROPY_BITS,
            "minimum_primary_multiscale_detail": MIN_PRIMARY_MULTISCALE_DETAIL,
            "marked_family_minimum": MARKED_FAMILY_MINIMUM,
            "plain_family_maximum": PLAIN_FAMILY_MAXIMUM,
            "minimum_family_support": MIN_FAMILY_SUPPORT,
            "maximum_family_timestamp_gap_ms": MAX_FAMILY_TIMESTAMP_GAP_MS,
            "maximum_family_aspect_log_deviation": MAX_FAMILY_ASPECT_LOG_DEVIATION,
            "minimum_family_marking_separation": MIN_FAMILY_MARKING_SEPARATION,
            "maximum_family_median_silhouette_distance": (
                MAX_FAMILY_MEDIAN_SILHOUETTE_DISTANCE
            ),
            "detail_gate": (
                "reject only when both eroded-interior Laplacian variance and tonal entropy "
                "fall below their minima"
            ),
            "primary_detail_policy": (
                "the primary must pass a boundary-safe sigma-1 minus sigma-4 detail floor; "
                "outer holder/background edges are excluded"
            ),
            "topology": (
                "post-export GLB reload with scene transforms applied and coincident positions "
                "welded while ignoring intentional UV/normal seams: watertight, "
                "winding-consistent, one body, Euler number 2"
            ),
            "maximum_triangles": max_triangles,
        },
        "limitations": [
            "Depth is inferred from a distance-transform prior and is not measured from the video.",
            "A soft object can deform between frames, so the two selected silhouettes need not describe one rigid shape.",
            "The primary silhouette alone defines geometry; the secondary view contributes appearance, not geometric constraints.",
            "Occluded surfaces, self-occlusion, holder contact, and back-to-front pixel correspondence are not reconstructed.",
            "Side texture is a projected extrapolation from the two selected views, not a directly observed texture bake.",
            "Absolute metric scale is unavailable without calibrated depth or an external scale reference.",
        ],
        "execution": {
            "local_only": True,
            "network_access": False,
            "learned_completion": False,
            "generative_texture": False,
            "libraries": {
                "opencv": cv2.__version__,
                "manifold3d": package_version("manifold3d"),
                "numpy": np.__version__,
                "trimesh": trimesh.__version__,
            },
        },
        "usd_package": usd_package,
        "delivery_validation": {
            "glb": glb_validation,
            "usdz": {
                "created": bool(usd_package.get("created")),
                "basis": (
                    "authored from the same gated projection parts; package/ARKit validation is "
                    "reported by usd_package, but USD primitive topology is not independently welded"
                ),
                "arkit_checker_passed": (
                    usd_package.get("validation", {}).get("passed")
                    if isinstance(usd_package.get("validation"), dict)
                    else None
                ),
            },
        },
        "artifacts": {path.name: _artifact_record(path) for path in artifacts},
        "quality_gate_passed": True,
    }
    report_path = output_directory / "automatic_soft_report.json"
    temporary = output_directory / ".automatic_soft_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


__all__ = ["AutoSoftError", "fit_soft_volume"]
