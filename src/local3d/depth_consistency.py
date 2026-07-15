"""Evidence-first comparison of monocular depth backends.

The reconstruction pipeline can obtain more than one depth prediction for a
frame, but two plausible-looking depth images are not automatically useful
multi-view geometry.  This module evaluates them without depending on a
particular model:

* sparse SfM point IDs are split *globally* into calibration and evaluation
  sets, so the same 3-D track can never train one frame and score another;
* inverse-depth predictions receive a robust per-view affine calibration,
  while metric predictions may receive only one global positive scale;
* every calibration is judged on held-out tracks;
* deterministic, angularly distributed view pairs are reprojected in both
  directions with the COLMAP ``SIMPLE_RADIAL`` camera model; and
* points hidden behind a target surface are recorded as occlusions, while
  points asserted in front of that surface/background are counted as genuine
  free-space contradictions.

The public result objects deliberately keep NumPy depth arrays separate from
their strictly JSON-serializable reports.  Nothing in this module downloads a
model, runs inference, or mutates reconstruction state.

Pose convention: ``x_camera = R @ X_world + t``.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .recon_common import Intrinsics, bilinear_sample, project_points


@dataclass(frozen=True)
class DepthConsistencyThresholds:
    """Conservative evidence thresholds for a depth backend.

    They are intentionally exposed and copied into every report.  Production
    callers can tune them for resolution/capture length, while tests and old
    reports remain reproducible.
    """

    calibration_fraction: float = 0.67
    minimum_track_length: int = 2
    maximum_track_reprojection_error_px: float = 4.0
    minimum_calibration_points_per_view: int = 8
    minimum_evaluation_points_per_view: int = 4
    minimum_prediction_iqr: float = 1e-5
    # Legacy camera-z-relative fields remain accepted for report/API
    # compatibility, but evidence decisions use the object-scale-normalized
    # thresholds below.
    maximum_heldout_median_relative_error: float = 0.12
    maximum_heldout_p90_relative_error: float = 0.25
    minimum_sfm_object_scale_points: int = 12
    maximum_heldout_median_object_scale_error: float = 0.08
    maximum_heldout_p90_object_scale_error: float = 0.18
    minimum_sfm_depth_span_object_scale_fraction: float = 0.04
    minimum_prediction_to_sfm_depth_span_ratio: float = 0.35

    minimum_aligned_views: int = 4
    minimum_aligned_view_fraction: float = 0.80

    pair_angle_bin_edges_degrees: tuple[float, ...] = (
        0.0,
        15.0,
        35.0,
        65.0,
        100.0,
        140.0,
        180.000001,
    )
    minimum_pair_angle_degrees: float = 8.0
    maximum_selected_pairs: int = 24
    minimum_selected_pairs: int = 3
    minimum_occupied_pair_angle_bins: int = 2
    minimum_maximum_pair_angle_degrees: float = 40.0

    mask_boundary_margin_px: int = 2
    depth_gradient_relative_limit: float = 0.08
    depth_gradient_object_scale_fraction: float = 0.05
    relative_depth_match_tolerance: float = 0.06
    object_scale_depth_tolerance_fraction: float = 0.03
    maximum_pixels_per_direction: int = 5000
    minimum_comparable_pixels_per_pair: int = 40
    minimum_comparable_pixels_per_direction: int = 20
    minimum_shared_evaluation_tracks_per_pair: int = 3
    minimum_pair_consistency: float = 0.72
    minimum_median_pair_consistency: float = 0.82
    minimum_p10_pair_consistency: float = 0.65
    maximum_free_space_contradiction_rate: float = 0.18
    minimum_median_bidirectional_coverage: float = 0.12
    maximum_bad_view_fraction: float = 0.20

    material_quality_margin: float = 0.08
    consensus_median_relative_difference: float = 0.055
    minimum_consensus_overlap_pixels: int = 100


@dataclass
class DepthBackendEvaluation:
    """A serializable evaluation report plus its validated metric depths."""

    name: str
    report: dict[str, Any]
    aligned_depths: dict[str, np.ndarray] = field(repr=False)
    aligned_confidences: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


@dataclass
class DepthSelectionResult:
    """Backend choice (or consensus) plus the depths safe for downstream use."""

    report: dict[str, Any]
    aligned_depths: dict[str, np.ndarray] = field(repr=False)
    aligned_confidences: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return _finite_float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _strict_report(report: dict[str, Any]) -> dict[str, Any]:
    """Round-trip through strict JSON to catch arrays/NaNs at the boundary."""

    return json.loads(json.dumps(report, allow_nan=False, sort_keys=True))


def _point_id_token(point_id: Any) -> str:
    """Stable, type-aware token for integer/string NumPy point IDs."""

    if isinstance(point_id, np.generic):
        point_id = point_id.item()
    if isinstance(point_id, bool):
        return f"bool:{int(point_id)}"
    if isinstance(point_id, int):
        return f"int:{point_id}"
    if isinstance(point_id, str):
        return f"str:{point_id}"
    if isinstance(point_id, float) and math.isfinite(point_id) and point_id.is_integer():
        return f"int:{int(point_id)}"
    raise TypeError(f"point IDs must be finite integers or strings, got {point_id!r}")


def deterministic_point_id_split(
    point_ids: Iterable[Any],
    *,
    calibration_fraction: float = 0.67,
    seed: str = "local3d-depth-v1",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split global track IDs reproducibly, independent of input ordering.

    IDs are canonicalized to type-aware strings and ranked by SHA-256.  A
    fixed-count rank split (instead of ``hash < probability``) guarantees that
    both sets exist whenever at least two unique IDs exist.
    """

    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    unique = sorted({_point_id_token(value) for value in point_ids})
    return _deterministic_token_split(unique, calibration_fraction, seed)


def _deterministic_token_split(
    tokens: Iterable[str], calibration_fraction: float, seed: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Internal split for IDs that have already been canonicalized."""

    if not 0.0 < float(calibration_fraction) < 1.0:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    unique = sorted(set(tokens))
    if len(unique) < 2:
        return tuple(unique), tuple()
    ranked = sorted(
        unique,
        key=lambda token: (
            hashlib.sha256(f"{seed}\0{token}".encode("utf-8")).digest(),
            token,
        ),
    )
    calibration_count = int(round(len(ranked) * float(calibration_fraction)))
    calibration_count = min(max(calibration_count, 1), len(ranked) - 1)
    calibration = tuple(sorted(ranked[:calibration_count]))
    evaluation = tuple(sorted(ranked[calibration_count:]))
    return calibration, evaluation


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    cutoff = float(np.sum(ordered_weights)) * float(q)
    index = int(np.searchsorted(cumulative, cutoff, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def robust_sfm_object_scale(
    sfm_evidence: Mapping[str, Any],
    *,
    minimum_points: int = 12,
) -> tuple[float | None, dict[str, Any]]:
    """Estimate object scale from globally unique, robust SfM point extents.

    The 5th--95th percentile bounding-box diagonal is insensitive to the
    remaining sparse outlier shell but still measures a flat object's width
    and height.  Per-view observations are used only as a fallback and are
    deduplicated by global point ID.
    """

    points_raw = sfm_evidence.get("points_xyz")
    ids_raw = sfm_evidence.get("point3d_ids")
    points: np.ndarray
    source: str
    if points_raw is not None:
        points = np.asarray(points_raw, dtype=np.float64)
        source = "global_sfm_points_xyz"
    else:
        unique: dict[str, np.ndarray] = {}
        evidence_views = sfm_evidence.get("views", {})
        if isinstance(evidence_views, Mapping):
            for name in sorted(evidence_views):
                record = evidence_views[name]
                if not isinstance(record, Mapping) or "xyz_world" not in record:
                    continue
                ids = np.asarray(record.get("point3d_ids", []), dtype=object).reshape(-1)
                xyz = np.asarray(record.get("xyz_world", []), dtype=np.float64)
                if xyz.ndim != 2 or xyz.shape[1] != 3 or len(ids) != len(xyz):
                    continue
                for point_id, point in zip(ids, xyz):
                    try:
                        token = _point_id_token(point_id)
                    except TypeError:
                        continue
                    if token not in unique and np.isfinite(point).all():
                        unique[token] = point
        points = (
            np.asarray([unique[token] for token in sorted(unique)], dtype=np.float64)
            if unique
            else np.empty((0, 3), dtype=np.float64)
        )
        source = "deduplicated_per_view_xyz_world"
    if points.ndim != 2 or points.shape[1:] != (3,):
        points = np.empty((0, 3), dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    # If global IDs accompany the points, collapse duplicate IDs deterministically.
    if ids_raw is not None and len(points) == len(np.asarray(ids_raw).reshape(-1)):
        unique_rows: dict[str, np.ndarray] = {}
        for point_id, point in zip(np.asarray(ids_raw).reshape(-1), points):
            try:
                token = _point_id_token(point_id)
            except TypeError:
                continue
            unique_rows.setdefault(token, point)
        points = np.asarray(
            [unique_rows[token] for token in sorted(unique_rows)], dtype=np.float64
        ).reshape(-1, 3)
    if len(points) < int(minimum_points):
        return None, {
            "accepted": False,
            "source": source,
            "point_count": int(len(points)),
            "required_points": int(minimum_points),
            "robust_extent_5_95": None,
            "scale": None,
            "reason": "insufficient_sfm_points_for_object_scale",
        }
    lower = np.percentile(points, 5.0, axis=0)
    upper = np.percentile(points, 95.0, axis=0)
    extent = upper - lower
    scale = float(np.linalg.norm(extent))
    if not math.isfinite(scale) or scale <= 1e-9:
        return None, {
            "accepted": False,
            "source": source,
            "point_count": int(len(points)),
            "required_points": int(minimum_points),
            "robust_extent_5_95": [float(value) for value in extent],
            "scale": None,
            "reason": "degenerate_sfm_object_scale",
        }
    return scale, {
        "accepted": True,
        "source": source,
        "point_count": int(len(points)),
        "required_points": int(minimum_points),
        "robust_extent_5_95": [float(value) for value in extent],
        "scale": scale,
        "reason": None,
    }


def _robust_linear_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    intercept: bool,
    iterations: int = 12,
) -> tuple[float, float] | None:
    """Deterministic Huber IRLS, initialized by weighted least squares."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(base) & (base > 0.0)
    x, y, base = x[good], y[good], base[good]
    if len(x) < (2 if intercept else 1):
        return None
    design = np.column_stack((x, np.ones_like(x))) if intercept else x[:, None]
    current = base.copy()
    params: np.ndarray | None = None
    for _ in range(max(int(iterations), 1)):
        normal = design.T @ (design * current[:, None])
        rhs = design.T @ (current * y)
        try:
            params = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            return None
        residual = design @ params - y
        median = _weighted_quantile(residual, base, 0.5)
        absolute = np.abs(residual - median)
        mad = _weighted_quantile(absolute, base, 0.5)
        scale = max(1.4826 * mad, 1e-8 * max(float(np.median(np.abs(y))), 1.0))
        normalized = absolute / (1.345 * scale)
        huber = np.ones_like(normalized)
        outside = normalized > 1.0
        huber[outside] = 1.0 / normalized[outside]
        current = base * huber
    if params is None or not np.isfinite(params).all():
        return None
    slope = float(params[0])
    offset = float(params[1]) if intercept else 0.0
    return slope, offset


def _canonical_views(
    views: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if isinstance(views, Mapping):
        items = [(str(name), value) for name, value in views.items()]
    else:
        items = []
        for index, view in enumerate(views):
            name = str(view.get("name", f"view_{index:06d}"))
            items.append((name, view))
    result: dict[str, Mapping[str, Any]] = {}
    for name, view in sorted(items, key=lambda item: item[0]):
        if name in result:
            raise ValueError(f"duplicate view name: {name}")
        result[name] = view
    return result


def _view_mask(view: Mapping[str, Any], shape: tuple[int, int]) -> np.ndarray | None:
    mask = view.get("mask_tight")
    if mask is None:
        return None
    array = np.asarray(mask)
    if array.ndim != 2 or array.shape != shape:
        return None
    return array > 0


def _observation_arrays(
    record: Mapping[str, Any],
    prediction: np.ndarray,
    mask: np.ndarray | None,
    thresholds: DepthConsistencyThresholds,
) -> dict[str, np.ndarray]:
    """Canonicalize one SfM view record and bilinearly sample a prediction."""

    ids_raw = np.asarray(record.get("point3d_ids", []), dtype=object).reshape(-1)
    xy = np.asarray(record.get("xy", []), dtype=np.float64)
    z = np.asarray(record.get("z_camera", []), dtype=np.float64).reshape(-1)
    if xy.size == 0:
        xy = np.empty((0, 2), dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or not (len(ids_raw) == len(xy) == len(z)):
        return {
            "tokens": np.empty(0, dtype=object),
            "prediction": np.empty(0),
            "z": np.empty(0),
            "weights": np.empty(0),
        }
    track_lengths = np.asarray(
        record.get("track_lengths", np.full(len(z), thresholds.minimum_track_length)),
        dtype=np.float64,
    ).reshape(-1)
    errors = np.asarray(
        record.get("reprojection_errors_px", np.zeros(len(z))), dtype=np.float64
    ).reshape(-1)
    if len(track_lengths) != len(z) or len(errors) != len(z):
        return {
            "tokens": np.empty(0, dtype=object),
            "prediction": np.empty(0),
            "z": np.empty(0),
            "weights": np.empty(0),
        }

    rows: list[tuple[str, float, float, float, float, float]] = []
    height, width = prediction.shape
    for raw_id, point_xy, point_z, length, error in zip(
        ids_raw, xy, z, track_lengths, errors
    ):
        try:
            token = _point_id_token(raw_id)
        except TypeError:
            continue
        u, v = float(point_xy[0]), float(point_xy[1])
        if not all(math.isfinite(value) for value in (u, v, point_z, length, error)):
            continue
        if point_z <= 1e-9 or length < thresholds.minimum_track_length:
            continue
        if error > thresholds.maximum_track_reprojection_error_px:
            continue
        if not (0.0 <= u < width - 1 and 0.0 <= v < height - 1):
            continue
        if mask is not None and not bool(mask[int(round(v)), int(round(u))]):
            continue
        # Sorting quality before coordinates makes duplicate-ID resolution
        # independent of observation array order.
        rows.append((token, error, -length, u, v, point_z))
    rows.sort()
    unique: list[tuple[str, float, float, float, float, float]] = []
    seen: set[str] = set()
    for row in rows:
        if row[0] not in seen:
            unique.append(row)
            seen.add(row[0])
    if not unique:
        return {
            "tokens": np.empty(0, dtype=object),
            "prediction": np.empty(0),
            "z": np.empty(0),
            "weights": np.empty(0),
        }
    tokens = np.asarray([row[0] for row in unique], dtype=object)
    u = np.asarray([row[3] for row in unique], dtype=np.float64)
    v = np.asarray([row[4] for row in unique], dtype=np.float64)
    z_values = np.asarray([row[5] for row in unique], dtype=np.float64)
    lengths = -np.asarray([row[2] for row in unique], dtype=np.float64)
    errors = np.asarray([row[1] for row in unique], dtype=np.float64)
    sampled = np.asarray(bilinear_sample(prediction, u, v), dtype=np.float64)
    weights = np.minimum(lengths, 10.0) / 10.0
    weights *= 1.0 / (1.0 + (errors / 2.0) ** 2)
    good = np.isfinite(sampled) & np.isfinite(z_values) & (weights > 0.0)
    return {
        "tokens": tokens[good],
        "prediction": sampled[good],
        "z": z_values[good],
        "weights": weights[good],
    }


def _aligned_depth_values(
    prediction: np.ndarray,
    slope: float,
    offset: float,
    *,
    prediction_space: str,
) -> np.ndarray:
    if prediction_space == "inverse_depth":
        inverse = slope * prediction + offset
        return np.where(inverse > 1e-9, 1.0 / inverse, np.nan)
    return slope * prediction


def _error_summary(
    aligned: np.ndarray, target: np.ndarray, object_scale: float | None
) -> dict[str, float | int | str | None]:
    aligned = np.asarray(aligned, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    usable = np.isfinite(aligned) & np.isfinite(target) & (target > 1e-9)
    if object_scale is None or object_scale <= 1e-9 or not np.any(usable):
        return {
            "count": 0,
            "normalization": "robust_sfm_object_scale",
            "median": None,
            "p90": None,
            "rms": None,
            "absolute_median": None,
            "camera_depth_relative_median_diagnostic": None,
        }
    absolute = np.abs(aligned[usable] - target[usable])
    normalized = absolute / object_scale
    camera_relative = absolute / target[usable]
    return {
        "count": int(len(normalized)),
        "normalization": "robust_sfm_object_scale",
        "median": float(np.median(normalized)),
        "p90": float(np.percentile(normalized, 90.0)),
        "rms": float(np.sqrt(np.mean(normalized**2))),
        "absolute_median": float(np.median(absolute)),
        "camera_depth_relative_median_diagnostic": float(np.median(camera_relative)),
    }


def align_depth_predictions(
    predictions: Mapping[str, np.ndarray],
    views: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    sfm_evidence: Mapping[str, Any],
    *,
    prediction_space: str,
    thresholds: DepthConsistencyThresholds | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Align one backend to SfM tracks and evaluate every fit on held-out IDs.

    ``sfm_evidence`` accepts the pipeline's native structure::

        {"views": {name: {"point3d_ids": N, "xy": (N,2),
                           "z_camera": N, "track_lengths": N,
                           "reprojection_errors_px": N}}}

    In ``inverse_depth`` mode each view fits ``1/z = a*p + b`` and requires
    ``a > 0``.  In ``metric_depth`` mode every view shares one fit
    ``z = scale*p``; no per-frame offset or scale is allowed.

    Only views that pass held-out evaluation are returned in ``aligned``.
    """

    thresholds = thresholds or DepthConsistencyThresholds()
    if prediction_space not in {"inverse_depth", "metric_depth"}:
        raise ValueError("prediction_space must be 'inverse_depth' or 'metric_depth'")
    canonical_views = _canonical_views(views)
    evidence_views = sfm_evidence.get("views", {})
    if not isinstance(evidence_views, Mapping):
        raise ValueError("sfm_evidence['views'] must be a mapping")
    object_scale, object_scale_report = robust_sfm_object_scale(
        sfm_evidence, minimum_points=thresholds.minimum_sfm_object_scale_points
    )

    prepared: dict[str, dict[str, np.ndarray]] = {}
    all_ids: list[Any] = []
    preparation_reasons: dict[str, str] = {}
    for name in sorted(set(canonical_views) & set(map(str, predictions.keys()))):
        prediction = np.asarray(predictions[name], dtype=np.float64)
        if prediction.ndim != 2 or prediction.size == 0:
            preparation_reasons[name] = "invalid_prediction_shape"
            continue
        mask = _view_mask(canonical_views[name], prediction.shape)
        record = evidence_views.get(name)
        if not isinstance(record, Mapping):
            preparation_reasons[name] = "missing_sfm_observations"
            continue
        observations = _observation_arrays(record, prediction, mask, thresholds)
        prepared[name] = observations
        all_ids.extend(observations["tokens"].tolist())

    calibration_ids, evaluation_ids = _deterministic_token_split(
        all_ids, thresholds.calibration_fraction, "local3d-depth-v1"
    )
    calibration_set, evaluation_set = set(calibration_ids), set(evaluation_ids)
    split_digest = hashlib.sha256(
        ("C:" + ",".join(calibration_ids) + "|E:" + ",".join(evaluation_ids)).encode(
            "utf-8"
        )
    ).hexdigest()

    global_metric_fit: tuple[float, float] | None = None
    if prediction_space == "metric_depth":
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        ws: list[np.ndarray] = []
        for observations in prepared.values():
            select = np.asarray(
                [token in calibration_set for token in observations["tokens"]], dtype=bool
            )
            select &= observations["prediction"] > 1e-9
            if np.any(select):
                xs.append(observations["prediction"][select])
                ys.append(observations["z"][select])
                ws.append(observations["weights"][select])
        if xs:
            global_metric_fit = _robust_linear_fit(
                np.concatenate(xs), np.concatenate(ys), np.concatenate(ws), intercept=False
            )

    aligned: dict[str, np.ndarray] = {}
    view_reports: dict[str, dict[str, Any]] = {}
    for name in sorted(canonical_views):
        prediction_raw = predictions.get(name)
        if prediction_raw is None or name not in prepared:
            reason = preparation_reasons.get(name, "missing_prediction")
            view_reports[name] = {
                "accepted": False,
                "reasons": [reason],
                "calibration_points": 0,
                "evaluation_points": 0,
                "heldout_point_ids": [],
                "slope_or_scale": None,
                "offset": None,
                "calibration_error": _error_summary(
                    np.empty(0), np.empty(0), object_scale
                ),
                "heldout_error": _error_summary(
                    np.empty(0), np.empty(0), object_scale
                ),
                "depth_relief": None,
            }
            continue
        prediction = np.asarray(prediction_raw, dtype=np.float64)
        observations = prepared[name]
        tokens = observations["tokens"]
        calibration = np.asarray([token in calibration_set for token in tokens], dtype=bool)
        evaluation = np.asarray([token in evaluation_set for token in tokens], dtype=bool)
        x = observations["prediction"]
        z = observations["z"]
        weights = observations["weights"]
        reasons: list[str] = []
        if int(np.count_nonzero(calibration)) < thresholds.minimum_calibration_points_per_view:
            reasons.append("insufficient_calibration_tracks")
        if int(np.count_nonzero(evaluation)) < thresholds.minimum_evaluation_points_per_view:
            reasons.append("insufficient_heldout_tracks")
        if object_scale is None:
            reasons.append("missing_independent_object_scale")

        fit: tuple[float, float] | None
        if prediction_space == "inverse_depth":
            calibration_x = x[calibration]
            if len(calibration_x):
                q25, q75 = np.percentile(calibration_x, (25.0, 75.0))
                spread = float(q75 - q25)
            else:
                spread = 0.0
            if spread < thresholds.minimum_prediction_iqr:
                reasons.append("degenerate_prediction_range")
                fit = None
            else:
                fit = _robust_linear_fit(
                    x[calibration],
                    1.0 / np.maximum(z[calibration], 1e-9),
                    weights[calibration],
                    intercept=True,
                )
        else:
            fit = global_metric_fit
            if fit is None:
                reasons.append("global_metric_scale_fit_failed")

        slope: float | None = None
        offset: float | None = None
        calibration_error = _error_summary(np.empty(0), np.empty(0), object_scale)
        heldout_error = _error_summary(np.empty(0), np.empty(0), object_scale)
        depth_relief: dict[str, Any] | None = None
        if fit is None:
            if "degenerate_prediction_range" not in reasons and "global_metric_scale_fit_failed" not in reasons:
                reasons.append("alignment_fit_failed")
        else:
            slope, offset = fit
            if not math.isfinite(slope) or slope <= 1e-12:
                reasons.append(
                    "nonpositive_inverse_depth_slope"
                    if prediction_space == "inverse_depth"
                    else "nonpositive_global_metric_scale"
                )
            else:
                calibrated_depth = _aligned_depth_values(
                    x[calibration], slope, offset, prediction_space=prediction_space
                )
                evaluated_depth = _aligned_depth_values(
                    x[evaluation], slope, offset, prediction_space=prediction_space
                )
                calibration_error = _error_summary(
                    calibrated_depth, z[calibration], object_scale
                )
                heldout_error = _error_summary(
                    evaluated_depth, z[evaluation], object_scale
                )
                median = heldout_error["median"]
                p90 = heldout_error["p90"]
                if (
                    median is not None
                    and median > thresholds.maximum_heldout_median_object_scale_error
                ):
                    reasons.append("heldout_median_error_too_high")
                if (
                    p90 is not None
                    and p90 > thresholds.maximum_heldout_p90_object_scale_error
                ):
                    reasons.append("heldout_tail_error_too_high")

                if prediction_space == "inverse_depth":
                    inverse = slope * prediction + offset
                    valid_fraction = float(np.mean(np.isfinite(inverse) & (inverse > 1e-9)))
                    depth = np.zeros_like(inverse, dtype=np.float32)
                    valid = np.isfinite(inverse) & (inverse > 1e-9)
                    depth[valid] = (1.0 / inverse[valid]).astype(np.float32)
                else:
                    depth64 = slope * prediction
                    valid_fraction = float(
                        np.mean(np.isfinite(depth64) & (depth64 > 1e-9))
                    )
                    depth = np.where(
                        np.isfinite(depth64) & (depth64 > 1e-9), depth64, 0.0
                    ).astype(np.float32)
                if valid_fraction < 0.50:
                    reasons.append("too_little_positive_aligned_depth")
                mask = _view_mask(canonical_views[name], depth.shape)
                inside = (
                    np.isfinite(depth)
                    & (depth > 1e-9)
                    & (mask if mask is not None else np.ones(depth.shape, dtype=bool))
                )
                predicted_span = (
                    float(np.percentile(depth[inside], 90.0) - np.percentile(depth[inside], 10.0))
                    if int(np.count_nonzero(inside)) >= 20
                    else 0.0
                )
                sfm_span = (
                    float(np.percentile(z, 90.0) - np.percentile(z, 10.0))
                    if len(z) >= 4
                    else 0.0
                )
                relief_ratio = predicted_span / max(sfm_span, 1e-9)
                sfm_span_fraction = (
                    sfm_span / object_scale if object_scale is not None else None
                )
                depth_relief = {
                    "predicted_depth_p10_p90_span": predicted_span,
                    "sfm_track_depth_p10_p90_span": sfm_span,
                    "sfm_span_object_scale_fraction": sfm_span_fraction,
                    "prediction_to_sfm_span_ratio": relief_ratio,
                }
                if (
                    sfm_span_fraction is not None
                    and sfm_span_fraction
                    >= thresholds.minimum_sfm_depth_span_object_scale_fraction
                    and relief_ratio
                    < thresholds.minimum_prediction_to_sfm_depth_span_ratio
                ):
                    reasons.append("insufficient_depth_relief_for_sfm_tracks")
                if not reasons:
                    aligned[name] = depth

        view_reports[name] = {
            "accepted": not reasons,
            "reasons": reasons,
            "usable_track_observations": int(len(tokens)),
            "calibration_points": int(np.count_nonzero(calibration)),
            "evaluation_points": int(np.count_nonzero(evaluation)),
            "heldout_point_ids": sorted(tokens[evaluation].tolist()),
            "slope_or_scale": _finite_float(slope),
            "offset": _finite_float(offset),
            "calibration_error": calibration_error,
            "heldout_error": heldout_error,
            "depth_relief": depth_relief,
        }

    report = {
        "schema": "local3d.depth_alignment.v1",
        "prediction_space": prediction_space,
        "model": (
            "per_view_positive_affine_inverse_depth"
            if prediction_space == "inverse_depth"
            else "one_global_positive_metric_scale_no_offset"
        ),
        "point_id_split": {
            "method": "sha256_global_point_id_rank",
            "calibration_fraction": float(thresholds.calibration_fraction),
            "calibration_ids": list(calibration_ids),
            "evaluation_ids": list(evaluation_ids),
            "digest_sha256": split_digest,
        },
        "global_metric_scale": (
            _finite_float(global_metric_fit[0]) if global_metric_fit is not None else None
        ),
        "object_scale": object_scale_report,
        "aligned_view_count": len(aligned),
        "candidate_view_count": len(canonical_views),
        "views": view_reports,
    }
    return aligned, _strict_report(report)


def _camera_direction(view: Mapping[str, Any], object_center: np.ndarray) -> np.ndarray | None:
    try:
        rotation = np.asarray(view["rotation"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(view["translation"], dtype=np.float64).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None
    center = -rotation.T @ translation
    offset = center - object_center
    norm = float(np.linalg.norm(offset))
    if not np.isfinite(offset).all() or norm <= 1e-9:
        return None
    return offset / norm


def select_eligible_view_pairs(
    views: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    eligible_names: Iterable[str],
    *,
    object_center: np.ndarray | None = None,
    shared_evaluation_track_tokens_by_view: Mapping[str, Iterable[str]] | None = None,
    thresholds: DepthConsistencyThresholds | None = None,
) -> dict[str, Any]:
    """Choose deterministic view pairs while balancing occupied angle bins.

    When held-out track tokens are supplied, a pair is eligible only if both
    views observed enough of the *same* evaluation tracks.  This prevents an
    angularly attractive but geometrically disconnected pair from inflating
    coverage.
    """

    thresholds = thresholds or DepthConsistencyThresholds()
    canonical = _canonical_views(views)
    center = np.zeros(3, dtype=np.float64) if object_center is None else np.asarray(
        object_center, dtype=np.float64
    ).reshape(3)
    directions = {
        name: direction
        for name in sorted(set(map(str, eligible_names)) & set(canonical))
        if (direction := _camera_direction(canonical[name], center)) is not None
    }
    edges = np.asarray(thresholds.pair_angle_bin_edges_degrees, dtype=np.float64)
    if len(edges) < 2 or not np.all(np.diff(edges) > 0.0):
        raise ValueError("pair angle bin edges must be strictly increasing")
    groups: dict[int, list[dict[str, Any]]] = {index: [] for index in range(len(edges) - 1)}
    track_sets = (
        {
            str(name): set(map(str, tokens))
            for name, tokens in shared_evaluation_track_tokens_by_view.items()
        }
        if shared_evaluation_track_tokens_by_view is not None
        else None
    )
    geometric_candidate_count = 0
    insufficient_shared_track_pair_count = 0
    names = sorted(directions)
    for first_index, first in enumerate(names):
        for second in names[first_index + 1 :]:
            cosine = float(np.clip(np.dot(directions[first], directions[second]), -1.0, 1.0))
            angle = math.degrees(math.acos(cosine))
            if angle < thresholds.minimum_pair_angle_degrees:
                continue
            geometric_candidate_count += 1
            shared_count: int | None = None
            if track_sets is not None:
                shared_count = len(track_sets.get(first, set()) & track_sets.get(second, set()))
                if shared_count < thresholds.minimum_shared_evaluation_tracks_per_pair:
                    insufficient_shared_track_pair_count += 1
                    continue
            bin_index = int(np.searchsorted(edges, angle, side="right") - 1)
            if 0 <= bin_index < len(edges) - 1:
                groups[bin_index].append(
                    {
                        "view_a": first,
                        "view_b": second,
                        "angle_degrees": float(angle),
                        "angle_bin": bin_index,
                        "shared_evaluation_track_count": shared_count,
                    }
                )

    selected: list[dict[str, Any]] = []
    used_views: set[str] = set()
    active = [index for index, candidates in groups.items() if candidates]
    while active and len(selected) < thresholds.maximum_selected_pairs:
        next_active: list[int] = []
        for bin_index in active:
            candidates = groups[bin_index]
            if not candidates:
                continue
            midpoint = float((edges[bin_index] + edges[bin_index + 1]) * 0.5)
            candidates.sort(
                key=lambda item: (
                    -int(item["view_a"] not in used_views)
                    - int(item["view_b"] not in used_views),
                    abs(float(item["angle_degrees"]) - midpoint),
                    item["view_a"],
                    item["view_b"],
                )
            )
            chosen = candidates.pop(0)
            selected.append(chosen)
            used_views.update((chosen["view_a"], chosen["view_b"]))
            if candidates:
                next_active.append(bin_index)
            if len(selected) >= thresholds.maximum_selected_pairs:
                break
        active = next_active

    occupied = sorted({int(pair["angle_bin"]) for pair in selected})
    return _strict_report(
        {
            "schema": "local3d.depth_view_pairs.v1",
            "eligible_view_count": len(directions),
            "shared_track_filter_applied": track_sets is not None,
            "geometric_candidate_pair_count": geometric_candidate_count,
            "insufficient_shared_track_pair_count": insufficient_shared_track_pair_count,
            "candidate_pair_count": int(sum(len(items) for items in groups.values()) + len(selected)),
            "selected_pair_count": len(selected),
            "occupied_angle_bins": occupied,
            "occupied_angle_bin_count": len(occupied),
            "maximum_selected_angle_degrees": (
                max((pair["angle_degrees"] for pair in selected), default=None)
            ),
            "pairs": selected,
        }
    )


def _invert_simple_radial(
    u: np.ndarray, v: np.ndarray, intrinsics: Intrinsics
) -> tuple[np.ndarray, np.ndarray]:
    focal, center_x, center_y, radial = map(float, intrinsics)
    if not math.isfinite(focal) or focal <= 0.0:
        raise ValueError("SIMPLE_RADIAL focal length must be positive and finite")
    distorted_x = (np.asarray(u, dtype=np.float64) - center_x) / focal
    distorted_y = (np.asarray(v, dtype=np.float64) - center_y) / focal
    x, y = distorted_x.copy(), distorted_y.copy()
    for _ in range(12):
        factor = 1.0 + radial * (x * x + y * y)
        safe = np.where(np.abs(factor) > 1e-12, factor, 1.0)
        x = distorted_x / safe
        y = distorted_y / safe
    return x, y


def _stable_depth_support(
    depth: np.ndarray,
    mask: np.ndarray,
    object_scale: float,
    thresholds: DepthConsistencyThresholds,
) -> tuple[np.ndarray, np.ndarray]:
    margin = max(int(thresholds.mask_boundary_margin_px), 0)
    kernel_size = 2 * margin + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask_u8 = mask.astype(np.uint8)
    eroded = cv2.erode(mask_u8, kernel, iterations=1) > 0 if margin else mask.copy()
    dilated = cv2.dilate(mask_u8, kernel, iterations=1) > 0 if margin else mask.copy()
    valid = np.isfinite(depth) & (depth > 1e-9)
    safe_depth = np.where(valid, depth, 0.0).astype(np.float32)
    # Max forward/backward absolute jump, normalized by independent object
    # scale.  Camera-z-relative gradients become arbitrarily permissive when
    # the object is small or far away.
    neighbors = [
        np.roll(safe_depth, 1, axis=0),
        np.roll(safe_depth, -1, axis=0),
        np.roll(safe_depth, 1, axis=1),
        np.roll(safe_depth, -1, axis=1),
    ]
    gradient = np.zeros_like(safe_depth, dtype=np.float32)
    for neighbor in neighbors:
        normalized = np.abs(neighbor - safe_depth) / max(float(object_scale), 1e-9)
        gradient = np.maximum(gradient, normalized)
    gradient[[0, -1], :] = np.inf
    gradient[:, [0, -1]] = np.inf
    stable = valid & eroded & (
        gradient <= thresholds.depth_gradient_object_scale_fraction
    )
    return stable, dilated


def _sample_pixels(mask: np.ndarray, maximum: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.nonzero(mask)
    if len(x) > maximum:
        indices = np.linspace(0, len(x) - 1, maximum, dtype=np.int64)
        x, y = x[indices], y[indices]
    return x.astype(np.float64), y.astype(np.float64)


def _directional_reprojection(
    source_view: Mapping[str, Any],
    target_view: Mapping[str, Any],
    source_depth: np.ndarray,
    target_depth: np.ndarray,
    intrinsics: Intrinsics,
    object_scale: float,
    thresholds: DepthConsistencyThresholds,
) -> dict[str, Any]:
    source_mask = _view_mask(source_view, source_depth.shape)
    target_mask = _view_mask(target_view, target_depth.shape)
    empty = {
        "source_stable_pixels": 0,
        "projected_in_frame": 0,
        "matches": 0,
        "free_space_contradictions": 0,
        "occluded_by_target": 0,
        "boundary_or_depth_unavailable": 0,
        "outside_target_frame": 0,
        "comparable_pixels": 0,
        "consistency": None,
        "free_space_contradiction_rate": None,
        "projection_coverage": 0.0,
        "median_comparable_relative_error": None,
        "median_comparable_object_scale_error": None,
    }
    if source_mask is None or target_mask is None:
        return empty
    source_stable, _ = _stable_depth_support(
        source_depth, source_mask, object_scale, thresholds
    )
    target_stable, target_dilated = _stable_depth_support(
        target_depth, target_mask, object_scale, thresholds
    )
    u, v = _sample_pixels(source_stable, thresholds.maximum_pixels_per_direction)
    if not len(u):
        return empty
    z = source_depth[v.astype(np.int64), u.astype(np.int64)].astype(np.float64)
    x_norm, y_norm = _invert_simple_radial(u, v, intrinsics)
    camera = np.column_stack((x_norm * z, y_norm * z, z))
    source_rotation = np.asarray(source_view["rotation"], dtype=np.float64).reshape(3, 3)
    source_translation = np.asarray(source_view["translation"], dtype=np.float64).reshape(3)
    world = (camera - source_translation) @ source_rotation
    target_u, target_v, target_z = project_points(
        world,
        np.asarray(target_view["rotation"], dtype=np.float64),
        np.asarray(target_view["translation"], dtype=np.float64),
        intrinsics,
    )
    height, width = target_depth.shape
    in_frame = (
        np.isfinite(target_u)
        & np.isfinite(target_v)
        & np.isfinite(target_z)
        & (target_z > 1e-9)
        & (target_u >= 0.0)
        & (target_u < width - 1)
        & (target_v >= 0.0)
        & (target_v < height - 1)
    )
    outside_count = int(len(u) - np.count_nonzero(in_frame))
    if not np.any(in_frame):
        result = dict(empty)
        result.update(
            {
                "source_stable_pixels": int(len(u)),
                "outside_target_frame": outside_count,
            }
        )
        return result
    iu = np.rint(target_u[in_frame]).astype(np.int64)
    iv = np.rint(target_v[in_frame]).astype(np.int64)
    iu = np.clip(iu, 0, width - 1)
    iv = np.clip(iv, 0, height - 1)
    stable = target_stable[iv, iu]
    dilated = target_dilated[iv, iu]
    # Projecting clearly outside the dilated silhouette asserts matter in
    # observed background and is therefore a free-space contradiction.
    background_free = ~dilated
    unavailable = dilated & ~stable
    sampled_target = np.asarray(
        bilinear_sample(target_depth, target_u[in_frame], target_v[in_frame]),
        dtype=np.float64,
    )
    valid_target = stable & np.isfinite(sampled_target) & (sampled_target > 1e-9)
    tolerance = thresholds.object_scale_depth_tolerance_fraction * object_scale
    projected_z = target_z[in_frame]
    near_free = valid_target & (projected_z < sampled_target - tolerance)
    occluded = valid_target & (projected_z > sampled_target + tolerance)
    matches = valid_target & ~near_free & ~occluded
    free = background_free | near_free
    unavailable |= dilated & ~valid_target
    comparable = matches | free
    comparable_count = int(np.count_nonzero(comparable))
    match_count = int(np.count_nonzero(matches))
    free_count = int(np.count_nonzero(free))
    relative = np.abs(projected_z - sampled_target) / np.maximum(sampled_target, 1e-9)
    object_normalized = np.abs(projected_z - sampled_target) / object_scale
    error_values = relative[matches | near_free]
    object_error_values = object_normalized[matches | near_free]
    return {
        "source_stable_pixels": int(len(u)),
        "projected_in_frame": int(np.count_nonzero(in_frame)),
        "matches": match_count,
        "free_space_contradictions": free_count,
        "occluded_by_target": int(np.count_nonzero(occluded)),
        "boundary_or_depth_unavailable": int(np.count_nonzero(unavailable)),
        "outside_target_frame": outside_count,
        "comparable_pixels": comparable_count,
        "consistency": (float(match_count / comparable_count) if comparable_count else None),
        "free_space_contradiction_rate": (
            float(free_count / comparable_count) if comparable_count else None
        ),
        "projection_coverage": float(comparable_count / max(len(u), 1)),
        "median_comparable_relative_error": (
            float(np.median(error_values)) if len(error_values) else None
        ),
        "median_comparable_object_scale_error": (
            float(np.median(object_error_values)) if len(object_error_values) else None
        ),
    }


def score_symmetric_reprojection(
    view_a: Mapping[str, Any],
    view_b: Mapping[str, Any],
    depth_a: np.ndarray,
    depth_b: np.ndarray,
    intrinsics: Intrinsics,
    *,
    object_scale: float | None = None,
    thresholds: DepthConsistencyThresholds | None = None,
) -> dict[str, Any]:
    """Score A→B and B→A, separating occlusion from contradiction."""

    thresholds = thresholds or DepthConsistencyThresholds()
    depth_a = np.asarray(depth_a, dtype=np.float32)
    depth_b = np.asarray(depth_b, dtype=np.float32)
    if depth_a.ndim != 2 or depth_b.ndim != 2:
        raise ValueError("depth maps must be 2-D")
    scale_source = "explicit_independent_sfm"
    if object_scale is None:
        # Backwards-compatible low-level fallback for diagnostics/tests.  The
        # backend aggregate never uses this self-estimated scale: it passes
        # robust SfM scale explicitly and fails closed when that is missing.
        estimates: list[float] = []
        focal = float(intrinsics[0])
        for view, depth in ((view_a, depth_a), (view_b, depth_b)):
            mask = _view_mask(view, depth.shape)
            if mask is None or not np.any(mask):
                continue
            y, x = np.nonzero(mask)
            valid_depth = depth[mask & np.isfinite(depth) & (depth > 1e-9)]
            if not len(valid_depth) or focal <= 0.0:
                continue
            median_depth = float(np.median(valid_depth))
            width = float(x.max() - x.min() + 1) * median_depth / focal
            height = float(y.max() - y.min() + 1) * median_depth / focal
            estimates.append(math.hypot(width, height))
        object_scale = float(np.median(estimates)) if estimates else None
        scale_source = "diagnostic_mask_extent_fallback"
    if object_scale is None or not math.isfinite(object_scale) or object_scale <= 1e-9:
        raise ValueError("a positive object scale is required for depth reprojection")
    forward = _directional_reprojection(
        view_a, view_b, depth_a, depth_b, intrinsics, object_scale, thresholds
    )
    backward = _directional_reprojection(
        view_b, view_a, depth_b, depth_a, intrinsics, object_scale, thresholds
    )
    matches = int(forward["matches"] + backward["matches"])
    free = int(
        forward["free_space_contradictions"] + backward["free_space_contradictions"]
    )
    comparable = matches + free
    source_total = int(forward["source_stable_pixels"] + backward["source_stable_pixels"])
    consistency = float(matches / comparable) if comparable else None
    report = {
        "schema": "local3d.depth_reprojection_pair.v1",
        "object_scale": float(object_scale),
        "object_scale_source": scale_source,
        "absolute_depth_match_tolerance": float(
            object_scale * thresholds.object_scale_depth_tolerance_fraction
        ),
        "forward": forward,
        "backward": backward,
        "totals": {
            "matches": matches,
            "free_space_contradictions": free,
            "occluded_by_target": int(
                forward["occluded_by_target"] + backward["occluded_by_target"]
            ),
            "comparable_pixels": comparable,
            "consistency": consistency,
            "free_space_contradiction_rate": (
                float(free / comparable) if comparable else None
            ),
            "bidirectional_coverage": float(comparable / max(source_total, 1)),
        },
    }
    return _strict_report(report)


def evaluate_depth_backend(
    name: str,
    predictions: Mapping[str, np.ndarray],
    views: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    intrinsics: Intrinsics,
    sfm_evidence: Mapping[str, Any],
    *,
    prediction_space: str,
    object_center: np.ndarray | None = None,
    thresholds: DepthConsistencyThresholds | None = None,
) -> DepthBackendEvaluation:
    """Run held-out calibration and multi-view consistency for one backend."""

    thresholds = thresholds or DepthConsistencyThresholds()
    canonical = _canonical_views(views)
    aligned, alignment_report = align_depth_predictions(
        predictions,
        canonical,
        sfm_evidence,
        prediction_space=prediction_space,
        thresholds=thresholds,
    )
    object_scale_value = alignment_report["object_scale"]["scale"]
    heldout_tracks = {
        view_name: alignment_report["views"][view_name].get("heldout_point_ids", [])
        for view_name in aligned
    }
    pairs = select_eligible_view_pairs(
        canonical,
        aligned.keys(),
        object_center=object_center,
        shared_evaluation_track_tokens_by_view=heldout_tracks,
        thresholds=thresholds,
    )
    pair_reports: list[dict[str, Any]] = []
    for pair in pairs["pairs"]:
        first, second = pair["view_a"], pair["view_b"]
        score = score_symmetric_reprojection(
            canonical[first],
            canonical[second],
            aligned[first],
            aligned[second],
            intrinsics,
            object_scale=float(object_scale_value),
            thresholds=thresholds,
        )
        totals = score["totals"]
        directionally_evaluable = bool(
            score["forward"]["comparable_pixels"]
            >= thresholds.minimum_comparable_pixels_per_direction
            and score["backward"]["comparable_pixels"]
            >= thresholds.minimum_comparable_pixels_per_direction
            and totals["comparable_pixels"]
            >= thresholds.minimum_comparable_pixels_per_pair
        )
        pair_reports.append(
            {
                **pair,
                "directionally_evaluable": directionally_evaluable,
                "accepted": bool(
                    directionally_evaluable
                    and totals["consistency"] is not None
                    and totals["consistency"] >= thresholds.minimum_pair_consistency
                ),
                "reprojection": score,
            }
        )

    evaluable_pair_reports = [
        item for item in pair_reports if item["directionally_evaluable"]
    ]
    consistencies = [
        float(item["reprojection"]["totals"]["consistency"])
        for item in evaluable_pair_reports
        if item["reprojection"]["totals"]["consistency"] is not None
    ]
    coverages = [
        float(item["reprojection"]["totals"]["bidirectional_coverage"])
        for item in evaluable_pair_reports
    ]
    total_matches = sum(
        item["reprojection"]["totals"]["matches"] for item in evaluable_pair_reports
    )
    total_free = sum(
        item["reprojection"]["totals"]["free_space_contradictions"]
        for item in evaluable_pair_reports
    )
    free_rate = float(total_free / max(total_matches + total_free, 1))
    aligned_fraction = float(len(aligned) / max(len(canonical), 1))

    per_view_scores: dict[str, list[float]] = {view_name: [] for view_name in aligned}
    for pair in evaluable_pair_reports:
        score = pair["reprojection"]["totals"]["consistency"]
        if score is not None:
            per_view_scores[pair["view_a"]].append(float(score))
            per_view_scores[pair["view_b"]].append(float(score))
    view_consistency = {
        view_name: (float(np.median(scores)) if scores else None)
        for view_name, scores in sorted(per_view_scores.items())
    }
    bad_views = sorted(
        view_name
        for view_name, score in view_consistency.items()
        if score is None or score < thresholds.minimum_pair_consistency
    )
    bad_view_fraction = float(len(bad_views) / max(len(aligned), 1))

    reasons: list[str] = []
    if len(aligned) < thresholds.minimum_aligned_views:
        reasons.append("insufficient_aligned_views")
    if aligned_fraction < thresholds.minimum_aligned_view_fraction:
        reasons.append("too_many_heldout_alignment_failures")
    effective_bins = sorted({int(pair["angle_bin"]) for pair in evaluable_pair_reports})
    effective_maximum_angle = max(
        (float(pair["angle_degrees"]) for pair in evaluable_pair_reports), default=None
    )
    if len(evaluable_pair_reports) < thresholds.minimum_selected_pairs:
        reasons.append("insufficient_view_pairs")
    if len(effective_bins) < thresholds.minimum_occupied_pair_angle_bins:
        reasons.append("insufficient_angular_pair_coverage")
    if (
        effective_maximum_angle is None
        or effective_maximum_angle < thresholds.minimum_maximum_pair_angle_degrees
    ):
        if "insufficient_angular_pair_coverage" not in reasons:
            reasons.append("insufficient_angular_pair_coverage")
    median_consistency = _percentile(consistencies, 50.0)
    p10_consistency = _percentile(consistencies, 10.0)
    median_coverage = _percentile(coverages, 50.0)
    if median_consistency is None or median_consistency < thresholds.minimum_median_pair_consistency:
        reasons.append("weak_median_reprojection_consistency")
    if p10_consistency is None or p10_consistency < thresholds.minimum_p10_pair_consistency:
        reasons.append("weak_tail_reprojection_consistency")
    if free_rate > thresholds.maximum_free_space_contradiction_rate:
        reasons.append("excess_free_space_contradictions")
    if median_coverage is None or median_coverage < thresholds.minimum_median_bidirectional_coverage:
        reasons.append("insufficient_bidirectional_reprojection_coverage")
    if bad_view_fraction > thresholds.maximum_bad_view_fraction:
        reasons.append("too_many_inconsistent_depth_views")

    heldout_medians = [
        float(view_report["heldout_error"]["median"])
        for view_report in alignment_report["views"].values()
        if view_report["heldout_error"]["median"] is not None
    ]
    alignment_quality = 1.0 - min(_percentile(heldout_medians, 50.0) or 1.0, 1.0)
    reprojection_quality = median_consistency or 0.0
    coverage_quality = math.sqrt(max(median_coverage or 0.0, 0.0))
    quality_score = float(
        np.clip(
            alignment_quality * reprojection_quality * coverage_quality * aligned_fraction,
            0.0,
            1.0,
        )
    )
    report = _strict_report(
        {
            "schema": "local3d.depth_backend_evaluation.v1",
            "backend": str(name),
            "accepted": not reasons,
            "reasons": reasons,
            "prediction_space": prediction_space,
            "quality_score": quality_score,
            "alignment": alignment_report,
            "pair_selection": pairs,
            "pair_reports": pair_reports,
            "aggregate": {
                "aligned_view_fraction": aligned_fraction,
                "median_pair_consistency": median_consistency,
                "p10_pair_consistency": p10_consistency,
                "free_space_contradiction_rate": free_rate,
                "median_bidirectional_coverage": median_coverage,
                "per_view_median_consistency": view_consistency,
                "bad_views": bad_views,
                "bad_view_fraction": bad_view_fraction,
                "median_heldout_object_scale_error": _percentile(heldout_medians, 50.0),
                "effective_evaluable_pair_count": len(evaluable_pair_reports),
                "effective_occupied_angle_bins": effective_bins,
                "effective_maximum_pair_angle_degrees": effective_maximum_angle,
            },
            "thresholds": {
                field_name: (
                    list(value) if isinstance(value, tuple) else value
                )
                for field_name, value in thresholds.__dict__.items()
            },
        }
    )
    confidences: dict[str, np.ndarray] = {}
    for view_name, depth in aligned.items():
        error = alignment_report["views"][view_name]["heldout_error"]["median"]
        base = float(
            np.clip(
                1.0 - float(error or 0.0) / max(
                    thresholds.maximum_heldout_median_object_scale_error, 1e-9
                ),
                0.05,
                1.0,
            )
        )
        mask = _view_mask(canonical[view_name], depth.shape)
        valid = np.isfinite(depth) & (depth > 1e-9)
        if mask is not None:
            valid &= mask
        confidences[view_name] = np.where(valid, base, 0.0).astype(np.float32)
    return DepthBackendEvaluation(str(name), report, aligned, confidences)


def evaluate_depth_predictions(
    name: str,
    predictions: Mapping[str, Any],
    views: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    intrinsics: Intrinsics,
    sfm_evidence: Mapping[str, Any],
    *,
    object_center: np.ndarray | None = None,
    thresholds: DepthConsistencyThresholds | None = None,
) -> DepthBackendEvaluation:
    """Evaluate duck-typed ``DepthPrediction`` objects from depth backends.

    Each value must expose ``values`` (H×W), ``representation`` equal to
    ``relative_disparity`` or ``metric_depth_m``, and may expose ``source_id``,
    ``focal_length_px``, and JSON-like ``provenance``.  Mixing representations
    inside one backend run is rejected because it would silently mix two
    calibration models.
    """

    ordered = sorted((str(view_name), value) for view_name, value in predictions.items())
    representations = {str(getattr(value, "representation", "")) for _, value in ordered}
    allowed = {"relative_disparity", "metric_depth_m"}
    if len(representations) != 1 or not representations.issubset(allowed):
        raise ValueError(
            "one backend evaluation must contain exactly one representation: "
            "'relative_disparity' or 'metric_depth_m'"
        )
    representation = next(iter(representations))
    arrays = {
        view_name: np.asarray(getattr(value, "values"), dtype=np.float32)
        for view_name, value in ordered
    }
    result = evaluate_depth_backend(
        name,
        arrays,
        views,
        intrinsics,
        sfm_evidence,
        prediction_space=(
            "inverse_depth" if representation == "relative_disparity" else "metric_depth"
        ),
        object_center=object_center,
        thresholds=thresholds,
    )
    contracts: dict[str, dict[str, Any]] = {}
    for view_name, prediction in ordered:
        provenance = getattr(prediction, "provenance", {})
        if isinstance(provenance, Mapping):
            provenance = dict(provenance)
        try:
            provenance_json = json.loads(json.dumps(provenance, allow_nan=False, sort_keys=True))
        except (TypeError, ValueError):
            provenance_json = {"unserializable": True}
        contracts[view_name] = {
            "source_id": str(getattr(prediction, "source_id", name)),
            "representation": representation,
            "focal_length_px": _finite_float(getattr(prediction, "focal_length_px", None)),
            "provenance": provenance_json,
        }
    report = dict(result.report)
    report["prediction_contract"] = {
        "representation": representation,
        "metric_conversion": (
            "none_already_optical_axis_camera_z"
            if representation == "metric_depth_m"
            else "none"
        ),
        "views": contracts,
    }
    result.report = _strict_report(report)
    return result


def align_backend(
    predictions: Mapping[str, Any],
    views: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]],
    track_evidence: Mapping[str, Any],
    intrinsics: Intrinsics,
    *,
    name: str = "depth_backend",
    object_center: np.ndarray | None = None,
    thresholds: DepthConsistencyThresholds | None = None,
) -> DepthBackendEvaluation:
    """Compatibility entry point using the pipeline's backend terminology."""

    return evaluate_depth_predictions(
        name,
        predictions,
        views,
        intrinsics,
        track_evidence,
        object_center=object_center,
        thresholds=thresholds,
    )


def _backend_depth_difference(
    first: DepthBackendEvaluation,
    second: DepthBackendEvaluation,
    *,
    maximum_samples: int = 20000,
) -> tuple[int, float | None]:
    differences: list[np.ndarray] = []
    for name in sorted(set(first.aligned_depths) & set(second.aligned_depths)):
        a = np.asarray(first.aligned_depths[name], dtype=np.float64)
        b = np.asarray(second.aligned_depths[name], dtype=np.float64)
        if a.shape != b.shape:
            continue
        valid = np.isfinite(a) & np.isfinite(b) & (a > 1e-9) & (b > 1e-9)
        if np.any(valid):
            differences.append(2.0 * np.abs(a[valid] - b[valid]) / (a[valid] + b[valid]))
    if not differences:
        return 0, None
    values = np.concatenate(differences)
    if len(values) > maximum_samples:
        pick = np.linspace(0, len(values) - 1, maximum_samples, dtype=np.int64)
        values = values[pick]
    return int(len(values)), float(np.median(values))


def _consensus_depths(
    evaluations: Sequence[DepthBackendEvaluation], relative_tolerance: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    result: dict[str, np.ndarray] = {}
    confidences: dict[str, np.ndarray] = {}
    names = sorted(set.intersection(*(set(item.aligned_depths) for item in evaluations)))
    for name in names:
        arrays = [np.asarray(item.aligned_depths[name], dtype=np.float32) for item in evaluations]
        if len({array.shape for array in arrays}) != 1:
            continue
        stack = np.stack([np.where(array > 1e-9, array, np.nan) for array in arrays])
        with np.errstate(all="ignore"):
            median = np.nanmedian(stack, axis=0)
            minimum = np.nanmin(stack, axis=0)
            maximum = np.nanmax(stack, axis=0)
        finite_count = np.sum(np.isfinite(stack), axis=0)
        disagreement = (maximum - minimum) / np.maximum(median, 1e-9)
        agreed = (
            (finite_count == len(arrays))
            & np.isfinite(median)
            & (disagreement <= relative_tolerance)
        )
        result[name] = np.where(agreed, median, 0.0).astype(np.float32)
        input_confidences = []
        for item, array in zip(evaluations, arrays):
            confidence = item.aligned_confidences.get(name)
            if confidence is None or confidence.shape != array.shape:
                confidence = np.where(array > 1e-9, 1.0, 0.0).astype(np.float32)
            input_confidences.append(confidence)
        confidence_stack = np.stack(input_confidences)
        confidences[name] = np.where(
            agreed, np.min(confidence_stack, axis=0), 0.0
        ).astype(np.float32)
    return result, confidences


def select_depth_backend(
    evaluations: Mapping[str, DepthBackendEvaluation] | Sequence[DepthBackendEvaluation],
    *,
    thresholds: DepthConsistencyThresholds | None = None,
) -> DepthSelectionResult:
    """Select only a material winner; otherwise return consensus or reject.

    A sole passing backend wins.  With multiple passers, the best quality score
    must beat the runner-up by ``material_quality_margin``.  If it does not,
    all passers must agree in metric depth over sufficient pixels; their
    pixelwise median is then returned as a consensus.  Close scores with
    disagreeing geometry fail closed as ``ambiguous_disagreement``.
    """

    thresholds = thresholds or DepthConsistencyThresholds()
    if isinstance(evaluations, Mapping):
        items = [evaluations[name] for name in sorted(evaluations)]
    else:
        items = sorted(evaluations, key=lambda item: item.name)
    passing = [item for item in items if bool(item.report.get("accepted"))]
    ranking = sorted(
        passing,
        key=lambda item: (-float(item.report.get("quality_score", 0.0)), item.name),
    )
    comparisons: list[dict[str, Any]] = []
    report: dict[str, Any]
    depths: dict[str, np.ndarray]
    if not ranking:
        report = {
            "schema": "local3d.depth_backend_selection.v1",
            "decision": "reject",
            "reason": "no_backend_passed_independent_evidence",
            "selected_backend": None,
            "consensus_backends": [],
            "ranking": [],
            "backend_depth_comparisons": [],
        }
        depths = {}
        confidences: dict[str, np.ndarray] = {}
    elif len(ranking) == 1:
        winner = ranking[0]
        report = {
            "schema": "local3d.depth_backend_selection.v1",
            "decision": "selected",
            "reason": "only_backend_to_pass_independent_evidence",
            "selected_backend": winner.name,
            "consensus_backends": [],
            "ranking": [
                {"backend": winner.name, "quality_score": winner.report["quality_score"]}
            ],
            "backend_depth_comparisons": [],
        }
        depths = dict(winner.aligned_depths)
        confidences = dict(winner.aligned_confidences)
    else:
        gap = float(ranking[0].report["quality_score"] - ranking[1].report["quality_score"])
        if gap >= thresholds.material_quality_margin:
            winner = ranking[0]
            report = {
                "schema": "local3d.depth_backend_selection.v1",
                "decision": "selected",
                "reason": "material_independent_evidence_margin",
                "selected_backend": winner.name,
                "consensus_backends": [],
                "quality_margin": gap,
                "ranking": [
                    {"backend": item.name, "quality_score": item.report["quality_score"]}
                    for item in ranking
                ],
                "backend_depth_comparisons": [],
            }
            depths = dict(winner.aligned_depths)
            confidences = dict(winner.aligned_confidences)
        else:
            consensus = True
            for first_index, first in enumerate(ranking):
                for second in ranking[first_index + 1 :]:
                    overlap, difference = _backend_depth_difference(first, second)
                    agreed = bool(
                        overlap >= thresholds.minimum_consensus_overlap_pixels
                        and difference is not None
                        and difference <= thresholds.consensus_median_relative_difference
                    )
                    comparisons.append(
                        {
                            "backend_a": first.name,
                            "backend_b": second.name,
                            "overlap_pixels": overlap,
                            "median_symmetric_relative_difference": difference,
                            "agreed": agreed,
                        }
                    )
                    consensus &= agreed
            if consensus:
                report = {
                    "schema": "local3d.depth_backend_selection.v1",
                    "decision": "consensus",
                    "reason": "no_material_winner_but_metric_depths_agree",
                    "selected_backend": None,
                    "consensus_backends": [item.name for item in ranking],
                    "quality_margin": gap,
                    "ranking": [
                        {"backend": item.name, "quality_score": item.report["quality_score"]}
                        for item in ranking
                    ],
                    "backend_depth_comparisons": comparisons,
                }
                depths, confidences = _consensus_depths(
                    ranking, thresholds.consensus_median_relative_difference
                )
            else:
                report = {
                    "schema": "local3d.depth_backend_selection.v1",
                    "decision": "reject",
                    "reason": "ambiguous_backend_disagreement",
                    "selected_backend": None,
                    "consensus_backends": [],
                    "quality_margin": gap,
                    "ranking": [
                        {"backend": item.name, "quality_score": item.report["quality_score"]}
                        for item in ranking
                    ],
                    "backend_depth_comparisons": comparisons,
                }
                depths = {}
                confidences = {}
    strict = _strict_report(report)
    return DepthSelectionResult(strict, depths, confidences)


__all__ = [
    "DepthBackendEvaluation",
    "DepthConsistencyThresholds",
    "DepthSelectionResult",
    "align_depth_predictions",
    "align_backend",
    "deterministic_point_id_split",
    "evaluate_depth_backend",
    "evaluate_depth_predictions",
    "score_symmetric_reprojection",
    "select_depth_backend",
    "select_eligible_view_pairs",
]
