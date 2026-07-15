"""Tiny, local shape-family router for masked multi-view object videos.

The router deliberately classifies geometry families rather than object names.
It consumes reviewed binary object masks, extracts scale/translation/rotation
invariant silhouette features, and applies a small NumPy softmax model trained
only on procedurally generated masks.  Runtime has no network, Torch, GPU, or
pretrained-model dependency.

This module is intentionally standalone for the MVP.  It writes a route report
but never dispatches a reconstruction builder; semantic subtype selection and
builder execution are separate safety decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


SCHEMA_VERSION = "1.0"
FEATURE_SCHEMA_VERSION = "silhouette-clip-v3"
CLASSIFIER_VERSION = "synthetic-softmax-v3"
FAMILIES = (
    "planar",
    "rounded_slab",
    "rectangular_prism",
    "cylinder",
    "bottle",
    "revolved",
    "free_form",
)
FRAME_FEATURE_NAMES = (
    "minor_major_ratio",
    "rectangularity",
    "hull_rectangularity",
    "solidity",
    "circularity",
    "convexity",
    "symmetry_lr",
    "symmetry_tb",
    "symmetry_180",
    "end_ratio",
    "end_asymmetry",
    "profile_cv",
    "radial_cv",
    "rounded_rect_iou",
    "rounded_radius_ratio",
)
QUANTILES = (0.10, 0.50, 0.90)
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "shape_router_v3.json"
ANALYSIS_LONG_SIDE = 512


class ShapeRouterError(RuntimeError):
    """Raised when router input or a saved model is invalid."""


@dataclass(frozen=True)
class MaskSample:
    path: Path
    candidate_index: int | None = None
    timestamp_s: float | None = None
    expected_width: int | None = None
    expected_height: int | None = None
    expected_sha256: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class FrameShapeFeatures:
    values: Mapping[str, float]
    quality: float
    area_fraction: float
    border_touch_fraction: float
    component_dominance: float
    hole_fraction: float


@dataclass(frozen=True)
class ClipShapeFeatures:
    names: tuple[str, ...]
    vector: np.ndarray
    values: Mapping[str, float]
    evidence: Mapping[str, Any]
    per_frame: tuple[Mapping[str, Any], ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShapeRouterError(f"could not read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ShapeRouterError(f"expected a JSON object in {path}")
    return value


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _resolve_manifest_path(value: str | Path, manifest: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def load_mask_sequence(
    analysis_path: str | Path | None = None,
    *,
    segmentation_manifest_path: str | Path | None = None,
    masks_dir: str | Path | None = None,
) -> list[MaskSample]:
    """Load canonical segmentation masks or adapt a legacy same-stem directory."""

    if segmentation_manifest_path is not None:
        manifest_path = Path(segmentation_manifest_path).expanduser().resolve()
        manifest = _read_json(manifest_path)
        if manifest.get("placeholder") is True:
            raise ShapeRouterError("placeholder full-frame segmentation cannot be classified")
        rows = manifest.get("frames")
        if not isinstance(rows, list):
            raise ShapeRouterError("segmentation manifest has no frames array")
        samples: list[MaskSample] = []
        for row in rows:
            if not isinstance(row, dict) or not row.get("object_mask_path"):
                continue
            confidence = row.get("confidence")
            if confidence is not None and float(confidence) <= 0.0:
                continue
            samples.append(
                MaskSample(
                    path=_resolve_manifest_path(row["object_mask_path"], manifest_path),
                    candidate_index=_optional_int(row.get("candidate_index")),
                    timestamp_s=_optional_float(row.get("timestamp_s")),
                    expected_width=_optional_int(row.get("width")),
                    expected_height=_optional_int(row.get("height")),
                    expected_sha256=str(row["object_mask_sha256"])
                    if row.get("object_mask_sha256")
                    else None,
                    confidence=_optional_float(confidence),
                )
            )
        if not samples:
            raise ShapeRouterError("segmentation manifest contains no usable object masks")
        return samples

    if masks_dir is None:
        raise ShapeRouterError("provide --segmentation-manifest or --masks-dir")
    directory = Path(masks_dir).expanduser().resolve()
    if not directory.is_dir():
        raise ShapeRouterError(f"mask directory does not exist: {directory}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".pbm", ".pgm", ".jpg", ".jpeg"}
    )
    if not paths:
        raise ShapeRouterError(f"no masks found in {directory}")

    analysis_rows: dict[str, Mapping[str, Any]] = {}
    if analysis_path is not None:
        analysis = _read_json(Path(analysis_path).expanduser().resolve())
        rows = analysis.get("frames") or analysis.get("keyframes") or []
        if not isinstance(rows, list):
            raise ShapeRouterError("analysis frames/keyframes must be an array")
        for row in rows:
            if isinstance(row, dict) and row.get("path"):
                analysis_rows[Path(str(row["path"])).stem] = row

    samples = []
    for index, path in enumerate(paths):
        row = analysis_rows.get(path.stem, {})
        samples.append(
            MaskSample(
                path=path,
                candidate_index=_optional_int(row.get("candidate_index", index)),
                timestamp_s=_optional_float(row.get("timestamp_s")),
                expected_width=_optional_int(row.get("width")),
                expected_height=_optional_int(row.get("height")),
            )
        )
    return samples


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise ShapeRouterError("mask has no foreground component")
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas)) + 1
    total = float(np.sum(areas))
    dominance = float(areas[largest_label - 1] / max(total, 1.0))
    component = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ShapeRouterError("mask has no usable contour")
    filled = np.zeros_like(component)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    visible_area = int(np.count_nonzero(component))
    filled_area = int(np.count_nonzero(filled))
    hole_fraction = 1.0 - visible_area / max(filled_area, 1)
    return filled, dominance, float(np.clip(hole_fraction, 0.0, 1.0))


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    a = first > 0
    b = second > 0
    union = int(np.count_nonzero(a | b))
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def _normalized_contour_mask(contour: np.ndarray, size: int = 128) -> np.ndarray:
    points = contour[:, 0, :].astype(np.float64)
    center = points.mean(axis=0)
    centered = points - center
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    minor = np.array([-major[1], major[0]])
    # Explicit two-term products avoid noisy Accelerate matmul warnings seen on
    # some Apple-silicon NumPy builds for these tiny vectors.
    transformed = np.column_stack(
        (
            centered[:, 0] * minor[0] + centered[:, 1] * minor[1],
            centered[:, 0] * major[0] + centered[:, 1] * major[1],
        )
    )
    lower = transformed.min(axis=0)
    upper = transformed.max(axis=0)
    span = np.maximum(upper - lower, 1e-6)
    normalized = (transformed - lower) / span
    pixels = 3.0 + normalized * (size - 7.0)
    output = np.zeros((size, size), np.uint8)
    cv2.fillPoly(output, [np.rint(pixels).astype(np.int32)], 255)
    return output


def _rounded_rect_template(size: int, radius_ratio: float) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    low, high = 3, size - 4
    radius = int(round(radius_ratio * (high - low + 1)))
    if radius <= 0:
        cv2.rectangle(mask, (low, low), (high, high), 255, cv2.FILLED)
        return mask
    radius = min(radius, (high - low + 1) // 2)
    cv2.rectangle(mask, (low + radius, low), (high - radius, high), 255, cv2.FILLED)
    cv2.rectangle(mask, (low, low + radius), (high, high - radius), 255, cv2.FILLED)
    for x in (low + radius, high - radius):
        for y in (low + radius, high - radius):
            cv2.circle(mask, (x, y), radius, 255, cv2.FILLED)
    return mask


def _width_profile(mask: np.ndarray) -> np.ndarray:
    widths = np.count_nonzero(mask > 0, axis=1).astype(np.float64)
    nonzero = widths[widths > 0]
    if len(nonzero) == 0:
        return np.zeros(1, np.float64)
    return widths[np.flatnonzero(widths > 0)[0] : np.flatnonzero(widths > 0)[-1] + 1]


def extract_frame_features(mask: np.ndarray) -> FrameShapeFeatures:
    """Extract invariant silhouette measurements from one binary mask."""

    if mask is None or mask.ndim not in (2, 3):
        raise ShapeRouterError("mask must be a decoded 2D image")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary = np.where(mask >= 128, 255, 0).astype(np.uint8)
    component, dominance, hole_fraction = _largest_component(binary)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 16.0:
        raise ShapeRouterError("foreground contour is too small")
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))
    rect_width, rect_height = cv2.minAreaRect(contour)[1]
    major = max(float(rect_width), float(rect_height))
    minor = min(float(rect_width), float(rect_height))
    if minor <= 1.0 or major <= 1.0:
        raise ShapeRouterError("foreground contour is degenerate")
    rectangle_area = major * minor
    normalized = _normalized_contour_mask(contour)

    radius_candidates = (0.0, 0.04, 0.08, 0.12, 0.18, 0.25, 0.38, 0.50)
    rounded_scores = [
        _iou(normalized, _rounded_rect_template(normalized.shape[0], radius))
        for radius in radius_candidates
    ]
    best_radius_index = int(np.argmax(rounded_scores))

    profile = _width_profile(normalized)
    band = max(1, int(round(len(profile) * 0.20)))
    middle_start = max(0, int(round(len(profile) * 0.40)))
    middle_end = max(middle_start + 1, int(round(len(profile) * 0.60)))
    top = float(np.mean(profile[:band]))
    bottom = float(np.mean(profile[-band:]))
    middle = float(np.mean(profile[middle_start:middle_end]))
    end_ratio = min(top, bottom) / max(middle, 1.0)
    end_asymmetry = abs(top - bottom) / max(top, bottom, 1.0)
    profile_cv = float(np.std(profile) / max(np.mean(profile), 1.0))

    moments = cv2.moments(contour)
    cx = moments["m10"] / max(moments["m00"], 1e-9)
    cy = moments["m01"] / max(moments["m00"], 1e-9)
    contour_points = contour[:, 0, :].astype(np.float64)
    radii = np.linalg.norm(contour_points - np.array([cx, cy]), axis=1)
    radial_cv = float(np.std(radii) / max(np.mean(radii), 1e-9))

    border_pixels = np.concatenate(
        (component[0, :], component[-1, :], component[:, 0], component[:, -1])
    )
    border_touch = float(np.count_nonzero(border_pixels) / max(len(border_pixels), 1))
    area_fraction = float(np.count_nonzero(component) / component.size)
    quality = float(
        np.clip(dominance, 0.0, 1.0)
        * np.clip(1.0 - 3.0 * border_touch, 0.0, 1.0)
        * np.clip(area_fraction / 0.01, 0.0, 1.0)
    )
    values = {
        "minor_major_ratio": minor / major,
        "rectangularity": area / max(rectangle_area, 1.0),
        "hull_rectangularity": hull_area / max(rectangle_area, 1.0),
        "solidity": area / max(hull_area, 1.0),
        "circularity": 4.0 * math.pi * area / max(perimeter * perimeter, 1.0),
        "convexity": hull_perimeter / max(perimeter, 1.0),
        "symmetry_lr": _iou(normalized, np.fliplr(normalized)),
        "symmetry_tb": _iou(normalized, np.flipud(normalized)),
        "symmetry_180": _iou(normalized, np.rot90(normalized, 2)),
        "end_ratio": float(np.clip(end_ratio, 0.0, 1.5)),
        "end_asymmetry": float(np.clip(end_asymmetry, 0.0, 1.0)),
        "profile_cv": float(np.clip(profile_cv, 0.0, 2.0)),
        "radial_cv": float(np.clip(radial_cv, 0.0, 2.0)),
        "rounded_rect_iou": rounded_scores[best_radius_index],
        "rounded_radius_ratio": radius_candidates[best_radius_index],
    }
    return FrameShapeFeatures(
        values, quality, area_fraction, border_touch, dominance, hole_fraction
    )


def _feature_names() -> tuple[str, ...]:
    names = []
    for feature in FRAME_FEATURE_NAMES:
        for quantile in QUANTILES:
            names.append(f"{feature}_q{int(round(quantile * 100)):02d}")
    names.extend(
        (
            "aspect_view_range",
            "face_fraction",
            "edge_fraction",
            "rounded_face_fraction",
            "edge_face_ratio",
            "median_quality",
        )
    )
    return tuple(names)


def _view_representatives(frames: Sequence[FrameShapeFeatures]) -> list[FrameShapeFeatures]:
    """Deduplicate dwell-heavy clips using aspect-ratio view bins."""

    bins: dict[int, FrameShapeFeatures] = {}
    for frame in frames:
        key = int(round(frame.values["minor_major_ratio"] / 0.05))
        current = bins.get(key)
        if current is None or frame.quality > current.quality:
            bins[key] = frame
    return [bins[key] for key in sorted(bins)]


def aggregate_frame_features(
    frames: Sequence[FrameShapeFeatures],
    *,
    per_frame_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> ClipShapeFeatures:
    valid = [frame for frame in frames if frame.quality >= 0.20]
    if not valid:
        raise ShapeRouterError("no valid masks remained after quality checks")
    representatives = _view_representatives(valid)
    values: dict[str, float] = {}
    for feature in FRAME_FEATURE_NAMES:
        observations = np.array([frame.values[feature] for frame in representatives])
        for quantile in QUANTILES:
            values[f"{feature}_q{int(round(quantile * 100)):02d}"] = float(
                np.quantile(observations, quantile)
            )
    aspect_q10 = values["minor_major_ratio_q10"]
    aspect_q90 = values["minor_major_ratio_q90"]
    representative_aspects = np.asarray(
        [frame.values["minor_major_ratio"] for frame in representatives], np.float64
    )
    # When the capture contains a real edge view, min/max aspect is the direct
    # silhouette estimate needed to separate paper-like sheets from thin books.
    # It is guarded later by view coverage, temporal, and mask-quality checks.
    relative_thickness = float(
        representative_aspects.min() / max(representative_aspects.max(), 1e-6)
    )
    face_threshold = max(0.12, aspect_q90 * 0.85)
    edge_threshold = min(0.60, max(0.04, aspect_q10 * 1.30))
    values.update(
        {
            "aspect_view_range": (aspect_q90 - aspect_q10) / max(aspect_q90, 1e-6),
            "face_fraction": float(
                np.mean(
                    [
                        frame.values["minor_major_ratio"] >= face_threshold
                        for frame in representatives
                    ]
                )
            ),
            "edge_fraction": float(
                np.mean(
                    [
                        frame.values["minor_major_ratio"] <= edge_threshold
                        for frame in representatives
                    ]
                )
            ),
            "rounded_face_fraction": float(
                np.mean(
                    [
                        frame.values["minor_major_ratio"] >= face_threshold
                        and frame.values["rounded_rect_iou"] >= 0.88
                        for frame in representatives
                    ]
                )
            ),
            "edge_face_ratio": relative_thickness,
            "median_quality": float(np.median([frame.quality for frame in valid])),
        }
    )
    names = _feature_names()
    vector = np.array([values[name] for name in names], np.float64)
    evidence = {
        "input_masks": len(frames),
        "valid_masks": len(valid),
        "view_clusters": len(representatives),
        "face_views": int(
            sum(f.values["minor_major_ratio"] >= face_threshold for f in representatives)
        ),
        "edge_views": int(
            sum(f.values["minor_major_ratio"] <= edge_threshold for f in representatives)
        ),
        "estimated_thickness_ratio": round(relative_thickness, 4),
        "aspect_view_range": round(values["aspect_view_range"], 4),
        "median_quality": round(values["median_quality"], 4),
        "clipped_fraction": round(
            float(np.mean([frame.border_touch_fraction > 0.01 for frame in frames])), 4
        ),
        "maximum_hole_fraction": round(max(frame.hole_fraction for frame in frames), 4),
        "minimum_component_dominance": round(
            min(frame.component_dominance for frame in frames), 4
        ),
    }
    if len(frames) >= 2:
        temporal_jumps = []
        for first, second in zip(frames, frames[1:]):
            jump = (
                abs(first.values["minor_major_ratio"] - second.values["minor_major_ratio"])
                + abs(first.values["solidity"] - second.values["solidity"])
                + abs(first.values["profile_cv"] - second.values["profile_cv"])
            )
            temporal_jumps.append(jump)
        evidence["temporal_jump_fraction"] = round(
            float(np.mean(np.asarray(temporal_jumps) > 0.55)), 4
        )
    else:
        evidence["temporal_jump_fraction"] = 0.0
    per_frame: list[Mapping[str, Any]] = []
    for index, frame in enumerate(frames):
        metadata = dict(per_frame_metadata[index]) if per_frame_metadata else {"index": index}
        metadata.update(
            {
                "quality": round(frame.quality, 5),
                "area_fraction": round(frame.area_fraction, 6),
                "border_touch_fraction": round(frame.border_touch_fraction, 6),
                "component_dominance": round(frame.component_dominance, 6),
                "hole_fraction": round(frame.hole_fraction, 6),
                "features": {key: round(float(value), 6) for key, value in frame.values.items()},
            }
        )
        per_frame.append(metadata)
    return ClipShapeFeatures(names, vector, values, evidence, tuple(per_frame))


def aggregate_clip_features(samples: Sequence[MaskSample]) -> ClipShapeFeatures:
    frames: list[FrameShapeFeatures] = []
    metadata: list[Mapping[str, Any]] = []
    errors: list[str] = []
    for sample in samples:
        if not sample.path.is_file():
            errors.append(f"missing mask: {sample.path}")
            continue
        if sample.expected_sha256 and _sha256(sample.path) != sample.expected_sha256:
            errors.append(f"checksum mismatch: {sample.path.name}")
            continue
        mask = cv2.imread(str(sample.path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            errors.append(f"could not decode: {sample.path.name}")
            continue
        if sample.expected_width and mask.shape[1] != sample.expected_width:
            errors.append(f"width mismatch: {sample.path.name}")
            continue
        if sample.expected_height and mask.shape[0] != sample.expected_height:
            errors.append(f"height mismatch: {sample.path.name}")
            continue
        intermediate_fraction = float(np.mean((mask > 0) & (mask < 255)))
        if intermediate_fraction > 0.01:
            errors.append(f"mask is not binary: {sample.path.name}")
            continue
        original_height, original_width = mask.shape
        if max(mask.shape) > ANALYSIS_LONG_SIDE:
            scale = ANALYSIS_LONG_SIDE / max(mask.shape)
            mask = cv2.resize(
                mask,
                (max(1, int(round(mask.shape[1] * scale))), max(1, int(round(mask.shape[0] * scale)))),
                interpolation=cv2.INTER_NEAREST,
            )
        try:
            frame = extract_frame_features(mask)
        except ShapeRouterError as exc:
            errors.append(f"{sample.path.name}: {exc}")
            continue
        if sample.confidence is not None:
            confidence = float(np.clip(sample.confidence, 0.0, 1.0))
            frame = FrameShapeFeatures(
                values=frame.values,
                quality=frame.quality * confidence,
                area_fraction=frame.area_fraction,
                border_touch_fraction=frame.border_touch_fraction,
                component_dominance=frame.component_dominance,
                hole_fraction=frame.hole_fraction,
            )
        frames.append(frame)
        metadata.append(
            {
                "mask": str(sample.path),
                "candidate_index": sample.candidate_index,
                "timestamp_s": sample.timestamp_s,
                "segmentation_confidence": sample.confidence,
                "source_dimensions": [original_width, original_height],
                "analysis_dimensions": [mask.shape[1], mask.shape[0]],
            }
        )
    if not frames:
        detail = "; ".join(errors[:3])
        raise ShapeRouterError(f"no masks could be analyzed{': ' + detail if detail else ''}")
    clip = aggregate_frame_features(frames, per_frame_metadata=metadata)
    evidence = dict(clip.evidence)
    evidence["skipped_masks"] = len(errors)
    evidence["skip_reasons"] = errors
    return ClipShapeFeatures(clip.names, clip.vector, clip.values, evidence, clip.per_frame)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    model_path = Path(path).expanduser().resolve()
    model = _read_json(model_path)
    if model.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ShapeRouterError("model feature schema is incompatible with this router")
    if tuple(model.get("families", [])) != FAMILIES:
        raise ShapeRouterError("model family ordering is incompatible with this router")
    if tuple(model.get("feature_names", [])) != _feature_names():
        raise ShapeRouterError("model feature ordering is incompatible with this router")
    feature_count = len(_feature_names())
    class_count = len(FAMILIES)
    try:
        mean = np.asarray(model["normalization"]["mean"], np.float64)
        scale = np.asarray(model["normalization"]["scale"], np.float64)
        weights = np.asarray(model["softmax"]["weights"], np.float64)
        bias = np.asarray(model["softmax"]["bias"], np.float64)
        temperature = float(model["softmax"]["temperature"])
        centroids = np.asarray(model["ood"]["class_centroids"], np.float64)
        p99_distances = np.asarray(model["ood"]["class_p99_distances"], np.float64)
        reject_distances = np.asarray(model["ood"]["class_reject_distances"], np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ShapeRouterError("model tensors are missing or malformed") from exc
    if mean.shape != (feature_count,) or scale.shape != (feature_count,):
        raise ShapeRouterError("model normalization tensors have invalid shapes")
    if weights.shape != (feature_count, class_count) or bias.shape != (class_count,):
        raise ShapeRouterError("model softmax tensors have invalid shapes")
    if centroids.shape != (class_count, feature_count):
        raise ShapeRouterError("model OOD centroids have invalid shapes")
    if p99_distances.shape != (class_count,) or reject_distances.shape != (class_count,):
        raise ShapeRouterError("model OOD thresholds have invalid shapes")
    tensors = (mean, scale, weights, bias, centroids, p99_distances, reject_distances)
    if not all(np.all(np.isfinite(tensor)) for tensor in tensors):
        raise ShapeRouterError("model tensors must contain only finite values")
    if np.any(scale <= 0.0):
        raise ShapeRouterError("model normalization scales must be positive")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ShapeRouterError("model temperature must be positive and finite")
    if np.any(p99_distances <= 0.0) or np.any(reject_distances < p99_distances):
        raise ShapeRouterError("model OOD thresholds must be positive and ordered")
    model["_path"] = str(model_path)
    model["_sha256"] = _sha256(model_path)
    return model


def _raw_probabilities(features: ClipShapeFeatures, model: Mapping[str, Any]) -> np.ndarray:
    mean = np.asarray(model["normalization"]["mean"], np.float64)
    scale = np.asarray(model["normalization"]["scale"], np.float64)
    weights = np.asarray(model["softmax"]["weights"], np.float64)
    bias = np.asarray(model["softmax"]["bias"], np.float64)
    temperature = float(model["softmax"].get("temperature", 1.0))
    normalized = (features.vector - mean) / scale
    logits = np.einsum("f,fc->c", normalized, weights) + bias
    probabilities = _softmax(logits / temperature)
    thickness = float(features.values.get("edge_face_ratio", 0.0))
    incompatible: list[str] = []
    if thickness > 0.06:
        incompatible.append("planar")
    if not 0.025 <= thickness <= 0.55:
        incompatible.append("rounded_slab")
    if thickness < 0.05:
        incompatible.append("rectangular_prism")
    for family in incompatible:
        probabilities[FAMILIES.index(family)] = 0.0
    total = float(probabilities.sum())
    if total > 0.0:
        probabilities /= total
    return probabilities


def _bootstrap_agreement(
    features: ClipShapeFeatures,
    model: Mapping[str, Any],
    predicted_index: int,
    *,
    trials: int = 16,
) -> float:
    frames: list[FrameShapeFeatures] = []
    for row in features.per_frame:
        raw_values = row.get("features")
        if not isinstance(raw_values, Mapping) or not all(
            name in raw_values for name in FRAME_FEATURE_NAMES
        ):
            continue
        frames.append(
            FrameShapeFeatures(
                values={name: float(raw_values[name]) for name in FRAME_FEATURE_NAMES},
                quality=float(row.get("quality", 1.0)),
                area_fraction=float(row.get("area_fraction", 0.1)),
                border_touch_fraction=float(row.get("border_touch_fraction", 0.0)),
                component_dominance=float(row.get("component_dominance", 1.0)),
                hole_fraction=float(row.get("hole_fraction", 0.0)),
            )
        )
    valid = [frame for frame in frames if frame.quality >= 0.20]
    representatives = _view_representatives(valid)
    if len(valid) < 8:
        return 0.0
    # Rotational families legitimately produce one or two silhouette-aspect
    # clusters across a full yaw sweep. In that case sample actual frames;
    # non-rotational coverage is still rejected separately below.
    population = representatives if len(representatives) >= 4 else valid
    sample_size = max(3, int(math.ceil(0.75 * len(population))))
    if len(population) >= 4:
        sample_size = min(sample_size, len(population) - 1)
    rng = np.random.default_rng(20260714)
    matches = 0
    for _ in range(trials):
        indices = rng.choice(len(population), size=sample_size, replace=False)
        subset = aggregate_frame_features([population[int(index)] for index in indices])
        if int(np.argmax(_raw_probabilities(subset, model))) == predicted_index:
            matches += 1
    return matches / trials


def classify_shape(
    features: ClipShapeFeatures,
    model: Mapping[str, Any],
    *,
    mask_provenance: str = "unknown",
    rotation_coverage_confirmed: bool = False,
) -> dict[str, Any]:
    probabilities = _raw_probabilities(features, model)
    order = np.argsort(probabilities)[::-1]
    top_index, second_index = int(order[0]), int(order[1])
    predicted = FAMILIES[top_index]
    confidence = float(probabilities[top_index])
    margin = confidence - float(probabilities[second_index])
    evidence = dict(features.evidence)
    thickness_for_constraints = float(features.values.get("edge_face_ratio", 0.0))
    evidence["thickness_incompatible_families"] = [
        family
        for family, incompatible in (
            ("planar", thickness_for_constraints > 0.06),
            ("rounded_slab", not 0.025 <= thickness_for_constraints <= 0.55),
            ("rectangular_prism", thickness_for_constraints < 0.05),
        )
        if incompatible
    ]
    agreement = _bootstrap_agreement(features, model, top_index)
    evidence["bootstrap_agreement"] = round(agreement, 4)
    mean = np.asarray(model["normalization"]["mean"], np.float64)
    scale = np.asarray(model["normalization"]["scale"], np.float64)
    normalized = (features.vector - mean) / scale
    centroid = np.asarray(model["ood"]["class_centroids"][top_index], np.float64)
    training_distance = float(np.mean((normalized - centroid) ** 2))
    p99_distance = float(model["ood"]["class_p99_distances"][top_index])
    reject_distance = float(model["ood"]["class_reject_distances"][top_index])
    evidence["training_distance"] = round(training_distance, 4)
    evidence["training_p99_distance"] = round(p99_distance, 4)
    evidence["training_reject_distance"] = round(reject_distance, 4)
    warnings: list[str] = []
    reject_reasons: list[str] = []

    if int(evidence["valid_masks"]) < 8:
        reject_reasons.append("fewer than 8 valid masks")
    rotational_family = predicted in {"cylinder", "bottle", "revolved"}
    if int(evidence["view_clusters"]) < 4 and not rotational_family:
        reject_reasons.append("fewer than 4 distinct silhouette view clusters")
    if (
        rotational_family
        and int(evidence["view_clusters"]) < 4
        and not rotation_coverage_confirmed
    ):
        reject_reasons.append("rotational silhouette needs externally confirmed orbit coverage")
    if float(evidence["clipped_fraction"]) > 0.35:
        reject_reasons.append("too many masks touch the image border")
    if confidence < 0.72:
        reject_reasons.append("top probability is below 0.72")
    if margin < 0.20:
        reject_reasons.append("top-two probability margin is below 0.20")
    if agreement < 0.75:
        reject_reasons.append("frame-subset agreement is below 0.75")
    if training_distance > reject_distance:
        reject_reasons.append("feature vector is outside the procedural training support")
    elif training_distance > p99_distance:
        warnings.append("Feature vector is beyond the predicted family's training p99 distance.")

    thickness_ratio = float(evidence["estimated_thickness_ratio"])
    prismatic_family = predicted in {"planar", "rounded_slab", "rectangular_prism"}
    if prismatic_family and (
        int(evidence["face_views"]) < 1
        or int(evidence["edge_views"]) < 1
        or float(evidence["aspect_view_range"]) < 0.20
    ):
        reject_reasons.append("prismatic route lacks both face-like and edge-like views")
    if predicted == "planar" and thickness_ratio > 0.06:
        reject_reasons.append("observed edge is too thick for a planar route")
    if predicted == "rounded_slab" and not 0.025 <= thickness_ratio <= 0.55:
        reject_reasons.append("estimated thickness lies outside rounded-slab support")
    if predicted == "rectangular_prism" and thickness_ratio < 0.05:
        reject_reasons.append("estimated thickness is too small for a rectangular prism")
    if float(evidence.get("maximum_hole_fraction", 0.0)) > 0.05:
        reject_reasons.append("significant silhouette holes require topology/part review")
    if float(evidence.get("minimum_component_dominance", 1.0)) < 0.75:
        reject_reasons.append("detached mask components make the silhouette unreliable")
    if float(evidence.get("temporal_jump_fraction", 0.0)) > 0.35:
        reject_reasons.append("temporal silhouette jumps suggest segmentation drift or deformation")

    if rotational_family and int(evidence["view_clusters"]) < 4:
        warnings.append(
            "A rotational family can keep the same silhouette through yaw; confirm orbit "
            "coverage from capture telemetry before reconstruction."
        )

    if mask_provenance in {"legacy_directory", "object_specific"}:
        warnings.append(
            "Masks came from an object-specific/legacy source; this is a smoke test, not "
            "independent classifier validation."
        )
    elif mask_provenance == "unknown":
        warnings.append("Mask provenance was not supplied; review segmentation before routing.")
    warnings.append(
        "No independent real-object calibration set exists yet; every v2 route requires review."
    )

    builders = {
        "planar": "planar_extrusion",
        "rounded_slab": "rounded_slab",
        "rectangular_prism": "box",
        "cylinder": "cylinder",
        "bottle": "lathed_profile",
        "revolved": "lathed_profile",
        "free_form": "free_form_reconstruction",
    }
    candidate_valid = not reject_reasons
    family = predicted if candidate_valid else "unknown"
    alternatives = [
        {"family": FAMILIES[int(index)], "probability": round(float(probabilities[index]), 6)}
        for index in order[:3]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "classifier": {
            "version": CLASSIFIER_VERSION,
            "model_path": model.get("_path"),
            "model_sha256": model.get("_sha256"),
            "training": model.get("training", {}),
        },
        "family": family,
        "predicted_family": predicted,
        "confidence": round(confidence, 6),
        "margin": round(margin, 6),
        "alternatives": alternatives,
        "decision": "review",
        "candidate_valid": candidate_valid,
        "auto_route_eligible": False,
        "candidate_builder": builders[predicted] if candidate_valid else None,
        "recommended_builder": None,
        "builder_supported": False,
        "semantic_subtype": None,
        "reject_reasons": reject_reasons,
        "warnings": warnings,
        "mask_provenance": mask_provenance,
        "rotation_coverage_confirmed": rotation_coverage_confirmed,
        "evidence": evidence,
        "aggregate_features": {
            key: round(float(value), 6) for key, value in features.values.items()
        },
        "per_frame": list(features.per_frame),
    }


def write_route_report(report: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    _atomic_json_write(destination, report)
    return destination


# ---------------------------------------------------------------------------
# Procedural training data.  These routines execute only for the train command.


def _draw_rounded_rectangle(
    width: float, height: float, radius_ratio: float, angle: float, size: int = 192
) -> np.ndarray:
    local = np.zeros((size, size), np.uint8)
    w = max(3, int(round(width)))
    h = max(3, int(round(height)))
    x0, x1 = (size - w) // 2, (size + w) // 2
    y0, y1 = (size - h) // 2, (size + h) // 2
    radius = min(int(round(radius_ratio * min(w, h))), w // 2, h // 2)
    if radius <= 0:
        cv2.rectangle(local, (x0, y0), (x1, y1), 255, cv2.FILLED)
    else:
        cv2.rectangle(local, (x0 + radius, y0), (x1 - radius, y1), 255, cv2.FILLED)
        cv2.rectangle(local, (x0, y0 + radius), (x1, y1 - radius), 255, cv2.FILLED)
        for x in (x0 + radius, x1 - radius):
            for y in (y0 + radius, y1 - radius):
                cv2.circle(local, (x, y), radius, 255, cv2.FILLED)
    matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
    return cv2.warpAffine(local, matrix, (size, size), flags=cv2.INTER_NEAREST)


def _draw_profile(
    half_widths: Sequence[float], levels: Sequence[float], angle: float, size: int = 192
) -> np.ndarray:
    center = size / 2.0
    left = [(center - width, level) for width, level in zip(half_widths, levels)]
    right = [(center + width, level) for width, level in reversed(list(zip(half_widths, levels)))]
    polygon = np.rint(left + right).astype(np.int32)
    mask = np.zeros((size, size), np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    matrix = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    return cv2.warpAffine(mask, matrix, (size, size), flags=cv2.INTER_NEAREST)


def _corrupt_mask(mask: np.ndarray, rng: np.random.Generator, amount: float) -> np.ndarray:
    output = mask.copy()
    if rng.random() < amount:
        kernel_size = int(rng.choice([1, 2, 3]))
        if kernel_size > 1:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            narrow = False
            if contours:
                _, _, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
                narrow = min(width, height) <= kernel_size + 2
            operation = cv2.dilate if narrow or rng.random() >= 0.5 else cv2.erode
            output = operation(output, kernel)
    if rng.random() < amount * 0.55:
        contours, _ = cv2.findContours(output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            if min(w, h) < 8:
                return output
            side = int(rng.integers(0, 4))
            centers = (
                (x, int(rng.uniform(y, y + h))),
                (x + w, int(rng.uniform(y, y + h))),
                (int(rng.uniform(x, x + w)), y),
                (int(rng.uniform(x, x + w)), y + h),
            )
            axes = (max(2, int(w * rng.uniform(0.04, 0.12))), max(2, int(h * rng.uniform(0.08, 0.20))))
            cv2.ellipse(output, centers[side], axes, float(rng.uniform(0, 180)), 0, 360, 0, -1)
    return output


def _synthetic_clip(
    family: str,
    rng: np.random.Generator,
    *,
    frames: int = 14,
    corruption: float = 0.22,
) -> list[np.ndarray]:
    result: list[np.ndarray] = []
    base_angle = float(rng.uniform(-25.0, 25.0))
    yaws = np.linspace(0.0, math.pi, frames, endpoint=False) + rng.uniform(-0.03, 0.03, frames)
    if family in {"planar", "rounded_slab", "rectangular_prism"}:
        height = float(rng.uniform(100.0, 145.0))
        face_width = height * float(rng.uniform(0.40, 1.45))
        if family == "planar":
            # Reserve planar for genuinely sheet-like objects.  A paperback is
            # thin, but its observed page block is still a real prism.
            depth = face_width * float(rng.uniform(0.001, 0.015))
            radius = float(rng.uniform(0.0, 0.07))
        elif family == "rounded_slab":
            # Covers slim phones/cases as well as deeper rounded tins.
            depth_ratio = (
                float(rng.uniform(0.035, 0.16))
                if rng.random() < 0.55
                else float(rng.uniform(0.16, 0.38))
            )
            depth = face_width * depth_ratio
            radius = float(rng.uniform(0.06, 0.20))
        else:
            # Sharp-edged prisms include paperback books, not just thick boxes.
            depth_ratio = (
                float(rng.uniform(0.04, 0.20))
                if rng.random() < 0.55
                else float(rng.uniform(0.20, 1.20))
            )
            depth = face_width * depth_ratio
            radius = float(rng.uniform(0.0, 0.045))
        normalizer = 145.0 / max(height, face_width + depth)
        height *= normalizer
        face_width *= normalizer
        depth *= normalizer
        for yaw in yaws:
            projected_width = abs(face_width * math.cos(yaw)) + abs(depth * math.sin(yaw))
            projected_radius = radius * min(height, projected_width)
            radius_ratio = projected_radius / max(min(height, projected_width), 1.0)
            mask = _draw_rounded_rectangle(
                projected_width,
                height * float(rng.uniform(0.98, 1.02)),
                radius_ratio,
                base_angle + float(rng.uniform(-3.0, 3.0)),
            )
            result.append(_corrupt_mask(mask, rng, corruption))
        return result

    if family == "cylinder":
        height = float(rng.uniform(105.0, 148.0))
        width = height * float(rng.uniform(0.28, 0.88))
        for _ in yaws:
            mask = _draw_rounded_rectangle(
                width * float(rng.uniform(0.96, 1.04)),
                height,
                float(rng.uniform(0.06, 0.14)),
                base_angle + float(rng.uniform(-2.0, 2.0)),
            )
            result.append(_corrupt_mask(mask, rng, corruption))
        return result

    if family == "bottle":
        body = float(rng.uniform(35.0, 58.0))
        neck = body * float(rng.uniform(0.28, 0.62))
        shoulder = body * float(rng.uniform(0.75, 0.96))
        levels = (24.0, 45.0, 61.0, 79.0, 145.0, 161.0)
        widths = (neck, neck, shoulder, body, body, body * 0.88)
        for _ in yaws:
            varied = [value * float(rng.uniform(0.98, 1.02)) for value in widths]
            mask = _draw_profile(varied, levels, base_angle + float(rng.uniform(-2.0, 2.0)))
            result.append(_corrupt_mask(mask, rng, corruption))
        return result

    if family == "revolved":
        base = float(rng.uniform(30.0, 58.0))
        mode = int(rng.integers(0, 3))
        levels = (24.0, 48.0, 78.0, 112.0, 145.0, 162.0)
        if mode == 0:  # sphere/ellipsoid-like
            widths = (base * 0.45, base * 0.82, base, base, base * 0.82, base * 0.42)
        elif mode == 1:  # vase/hourglass
            widths = (base * 0.55, base * 0.85, base * 0.52, base * 0.88, base, base * 0.72)
        else:  # bowl/bulged vessel
            widths = (base * 0.82, base, base * 0.92, base * 0.72, base * 0.58, base * 0.42)
        for _ in yaws:
            varied = [value * float(rng.uniform(0.97, 1.03)) for value in widths]
            mask = _draw_profile(varied, levels, base_angle + float(rng.uniform(-2.0, 2.0)))
            result.append(_corrupt_mask(mask, rng, corruption))
        return result

    if family == "free_form":
        count = int(rng.integers(9, 17))
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        base_radii = rng.uniform(35.0, 72.0, count)
        base_radii[:: int(rng.integers(2, 5))] *= rng.uniform(0.45, 0.72)
        center = np.array([96.0, 96.0])
        for yaw in yaws:
            modulation = 1.0 + 0.16 * np.sin(angles * 2.0 + yaw * 1.7)
            radii = base_radii * modulation
            points = center + np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
            mask = np.zeros((192, 192), np.uint8)
            cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
            matrix = cv2.getRotationMatrix2D((96.0, 96.0), base_angle, 1.0)
            mask = cv2.warpAffine(mask, matrix, (192, 192), flags=cv2.INTER_NEAREST)
            result.append(_corrupt_mask(mask, rng, corruption))
        return result
    raise ShapeRouterError(f"unknown synthetic family: {family}")


def _synthetic_features(family: str, rng: np.random.Generator, corruption: float) -> np.ndarray:
    masks = _synthetic_clip(family, rng, corruption=corruption)
    frames = [extract_frame_features(mask) for mask in masks]
    return aggregate_frame_features(frames).vector


def _fit_softmax(
    x: np.ndarray,
    y: np.ndarray,
    *,
    iterations: int = 900,
    learning_rate: float = 0.035,
    l2: float = 2e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-5] = 1.0
    normalized = (x - mean) / scale
    weights = np.zeros((x.shape[1], len(FAMILIES)), np.float64)
    bias = np.zeros(len(FAMILIES), np.float64)
    mw = np.zeros_like(weights)
    vw = np.zeros_like(weights)
    mb = np.zeros_like(bias)
    vb = np.zeros_like(bias)
    targets = np.eye(len(FAMILIES), dtype=np.float64)[y]
    for step in range(1, iterations + 1):
        logits = np.einsum("nf,fc->nc", normalized, weights) + bias
        probabilities = _softmax(logits)
        error = (probabilities - targets) / len(x)
        gradient_w = np.einsum("nf,nc->fc", normalized, error) + l2 * weights
        gradient_b = error.sum(axis=0)
        mw = 0.9 * mw + 0.1 * gradient_w
        vw = 0.999 * vw + 0.001 * gradient_w * gradient_w
        mb = 0.9 * mb + 0.1 * gradient_b
        vb = 0.999 * vb + 0.001 * gradient_b * gradient_b
        corrected_mw = mw / (1.0 - 0.9**step)
        corrected_vw = vw / (1.0 - 0.999**step)
        corrected_mb = mb / (1.0 - 0.9**step)
        corrected_vb = vb / (1.0 - 0.999**step)
        weights -= learning_rate * corrected_mw / (np.sqrt(corrected_vw) + 1e-8)
        bias -= learning_rate * corrected_mb / (np.sqrt(corrected_vb) + 1e-8)
    return mean, scale, weights, bias


def train_model(
    *,
    samples_per_family: int = 180,
    validation_per_family: int = 45,
    test_per_family: int = 45,
    seed: int = 7312026,
) -> dict[str, Any]:
    if samples_per_family < 20 or validation_per_family < 5 or test_per_family < 5:
        raise ShapeRouterError(
            "training requires at least 20 train, 5 validation, and 5 test clips per family"
        )
    rng = np.random.default_rng(seed)
    train_x: list[np.ndarray] = []
    train_y: list[int] = []
    validation_x: list[np.ndarray] = []
    validation_y: list[int] = []
    test_x: list[np.ndarray] = []
    test_y: list[int] = []
    for family_index, family in enumerate(FAMILIES):
        for _ in range(samples_per_family):
            train_x.append(_synthetic_features(family, rng, corruption=0.22))
            train_y.append(family_index)
        for _ in range(validation_per_family):
            validation_x.append(_synthetic_features(family, rng, corruption=0.28))
            validation_y.append(family_index)
        for _ in range(test_per_family):
            test_x.append(_synthetic_features(family, rng, corruption=0.30))
            test_y.append(family_index)
    x = np.vstack(train_x)
    y = np.asarray(train_y, np.int64)
    vx = np.vstack(validation_x)
    vy = np.asarray(validation_y, np.int64)
    tx = np.vstack(test_x)
    ty = np.asarray(test_y, np.int64)
    mean, scale, weights, bias = _fit_softmax(x, y)
    normalized_train = (x - mean) / scale
    class_centroids = np.vstack(
        [normalized_train[y == index].mean(axis=0) for index in range(len(FAMILIES))]
    )
    class_distances = [
        np.mean((normalized_train[y == index] - class_centroids[index]) ** 2, axis=1)
        for index in range(len(FAMILIES))
    ]
    class_p99_distances = np.array(
        [np.quantile(distances, 0.99) for distances in class_distances], np.float64
    )
    # Synthetic support is much narrower than reality. The outer threshold is
    # deliberately generous and only catches extreme vectors; crossing p99 is
    # separately surfaced as a warning and every current route remains review-only.
    class_reject_distances = np.maximum(
        class_p99_distances * 4.0, class_p99_distances + 5.0
    )
    normalized_validation = (vx - mean) / scale
    best_temperature = 1.0
    best_loss = float("inf")
    for temperature in np.linspace(0.65, 2.0, 28):
        logits = np.einsum("nf,fc->nc", normalized_validation, weights) + bias
        probabilities = _softmax(logits / temperature)
        loss = float(-np.mean(np.log(np.clip(probabilities[np.arange(len(vy)), vy], 1e-9, 1.0))))
        if loss < best_loss:
            best_loss = loss
            best_temperature = float(temperature)
    logits = np.einsum("nf,fc->nc", normalized_validation, weights) + bias
    probabilities = _softmax(logits / best_temperature)
    predictions = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predictions == vy))
    per_family = {
        family: float(np.mean(predictions[vy == index] == index))
        for index, family in enumerate(FAMILIES)
    }
    normalized_test = (tx - mean) / scale
    test_logits = np.einsum("nf,fc->nc", normalized_test, weights) + bias
    test_probabilities = _softmax(test_logits / best_temperature)
    test_predictions = np.argmax(test_probabilities, axis=1)
    test_accuracy = float(np.mean(test_predictions == ty))
    test_loss = float(
        -np.mean(
            np.log(np.clip(test_probabilities[np.arange(len(ty)), ty], 1e-9, 1.0))
        )
    )
    test_per_family_accuracy = {
        family: float(np.mean(test_predictions[ty == index] == index))
        for index, family in enumerate(FAMILIES)
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "classifier_version": CLASSIFIER_VERSION,
        "families": list(FAMILIES),
        "feature_names": list(_feature_names()),
        "normalization": {
            "mean": mean.round(9).tolist(),
            "scale": scale.round(9).tolist(),
        },
        "softmax": {
            "weights": weights.round(9).tolist(),
            "bias": bias.round(9).tolist(),
            "temperature": round(best_temperature, 6),
        },
        "ood": {
            "metric": "mean squared distance in standardized feature space",
            "class_centroids": class_centroids.round(9).tolist(),
            "class_p99_distances": class_p99_distances.round(9).tolist(),
            "class_reject_distances": class_reject_distances.round(9).tolist(),
        },
        "training": {
            "source": "procedural binary silhouettes only",
            "seed": seed,
            "train_clips_per_family": samples_per_family,
            "validation_clips_per_family": validation_per_family,
            "test_clips_per_family": test_per_family,
            "train_clip_count": len(x),
            "validation_clip_count": len(vx),
            "test_clip_count": len(tx),
            "validation_accuracy": round(accuracy, 6),
            "validation_accuracy_by_family": {
                key: round(value, 6) for key, value in per_family.items()
            },
            "validation_nll": round(best_loss, 6),
            "test_accuracy": round(test_accuracy, 6),
            "test_accuracy_by_family": {
                key: round(value, 6) for key, value in test_per_family_accuracy.items()
            },
            "test_nll": round(test_loss, 6),
            "limitations": (
                "Synthetic test results are software checks, not evidence of real-world accuracy. "
                "Calibrate on clips split by physical object before production routing."
            ),
        },
    }


def save_model(model: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    _atomic_json_write(destination, model)
    return destination


def _command_train(arguments: argparse.Namespace) -> int:
    model = train_model(
        samples_per_family=arguments.samples_per_family,
        validation_per_family=arguments.validation_per_family,
        test_per_family=arguments.test_per_family,
        seed=arguments.seed,
    )
    path = save_model(model, arguments.output)
    print(
        json.dumps(
            {"model": str(path), "bytes": path.stat().st_size, **model["training"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _command_classify(arguments: argparse.Namespace) -> int:
    samples = load_mask_sequence(
        arguments.analysis,
        segmentation_manifest_path=arguments.segmentation_manifest,
        masks_dir=arguments.masks_dir,
    )
    features = aggregate_clip_features(samples)
    model = load_model(arguments.model)
    report = classify_shape(
        features,
        model,
        mask_provenance=arguments.mask_provenance,
        rotation_coverage_confirmed=arguments.rotation_coverage_confirmed,
    )
    if arguments.output:
        output = write_route_report(report, arguments.output)
        report = dict(report)
        report["report_path"] = str(output)
    if not arguments.include_per_frame:
        report = dict(report)
        report.pop("per_frame", None)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["candidate_valid"] else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m local3d.shape_router",
        description=(
            "Classify a masked multi-view object clip into a geometry family. "
            "No network, Torch, GPU, or pretrained weights are used."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train the tiny model on procedural silhouettes")
    train.add_argument("--output", type=Path, default=DEFAULT_MODEL_PATH)
    train.add_argument("--samples-per-family", type=int, default=180)
    train.add_argument("--validation-per-family", type=int, default=45)
    train.add_argument("--test-per-family", type=int, default=45)
    train.add_argument("--seed", type=int, default=7312026)
    train.set_defaults(handler=_command_train)

    classify = subparsers.add_parser("classify", help="classify a directory/manifest of masks")
    source = classify.add_mutually_exclusive_group(required=True)
    source.add_argument("--segmentation-manifest", type=Path)
    source.add_argument("--masks-dir", type=Path)
    classify.add_argument("--analysis", type=Path)
    classify.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    classify.add_argument("--output", type=Path)
    classify.add_argument(
        "--mask-provenance",
        choices=("generic_reviewed", "legacy_directory", "object_specific", "unknown"),
        default="unknown",
    )
    classify.add_argument("--include-per-frame", action="store_true")
    classify.add_argument(
        "--rotation-coverage-confirmed",
        action="store_true",
        help=(
            "capture telemetry/user workflow confirms a real orbit; required to consider "
            "silhouette-invariant cylinders, bottles, and revolved objects"
        ),
    )
    classify.set_defaults(handler=_command_classify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (ShapeRouterError, ValueError, OSError) as exc:
        print(f"shape-router: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("shape-router: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
