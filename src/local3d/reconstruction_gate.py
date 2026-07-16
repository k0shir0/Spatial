"""Independent evidence gate for automatically reconstructed meshes.

The reconstruction pipeline has several ways to produce a technically valid
mesh.  A watertight file, however, is not evidence that the camera motion was
recoverable or that the delivered surface agrees with the source capture.
This module deliberately sits outside those reconstruction backends and asks
four object-agnostic questions:

* Do the independently estimated cameras cover a useful sweep around the mesh?
* Does the delivered mesh reproject into the available object masks?
* How much surface can be traced to real source-image pixels, from how many views?
* Is depth supported by aligned depth maps or a non-degenerate SfM point cloud?

No colour, semantic class, primitive family, or object-specific dimension is
used.  The result contains only JSON-serializable values and stable reason
codes, so an orchestrator can fail closed without having to interpret NumPy
objects or backend-specific reports.

The gate is intentionally evidence-oriented rather than topology-oriented.
Callers should continue to run their existing manifold/file-format checks as a
separate delivery gate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .recon_common import (
    Intrinsics,
    bilinear_sample,
    face_normals,
    project_points,
    rasterize_zbuffer,
)


_INFERRED_POSE_SOURCES = {
    "completed",
    "inferred",
    "silhouette",
    "turntable",
    "visual_hull",
}


@dataclass(frozen=True)
class GateThresholds:
    """Conservative defaults for a deliverable general reconstruction.

    The thresholds describe evidence, not an object category.  They are kept
    in the returned report so a run remains auditable if callers override them.
    """

    minimum_mask_views: int = 8
    maximum_evaluation_views: int = 24
    render_scale: float = 0.35

    minimum_pose_coverage_degrees: float = 150.0
    minimum_pairwise_view_angle_degrees: float = 105.0
    direction_bin_separation_degrees: float = 18.0
    minimum_direction_bins: int = 6
    maximum_camera_radius_cv: float = 0.55
    minimum_median_look_at_cosine: float = 0.45
    minimum_independent_pose_views: int = 6
    minimum_sfm_registered_fraction: float = 0.20

    minimum_silhouette_iou: float = 0.58
    minimum_median_silhouette_iou: float = 0.70
    minimum_p10_silhouette_iou: float = 0.50
    minimum_good_silhouette_fraction: float = 0.70
    minimum_holdout_views_for_gate: int = 3
    minimum_holdout_median_iou: float = 0.66

    minimum_texture_views: int = 4
    minimum_observed_surface_fraction: float = 0.60
    minimum_multiview_surface_fraction: float = 0.32
    maximum_multiview_color_rms: float = 0.35

    minimum_depth_views: int = 3
    minimum_depth_pixel_coverage: float = 0.25
    minimum_depth_surface_fraction: float = 0.20
    maximum_median_depth_relative_error: float = 0.22

    minimum_sfm_points: int = 50
    minimum_sfm_point_second_axis_ratio: float = 0.01
    minimum_sfm_point_extent_ratio: float = 0.18
    minimum_sfm_point_mask_consistent_fraction: float = 0.50
    minimum_sfm_point_median_view_support: float = 2.0


def _finite_float(value: float | int | np.floating[Any]) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return _finite_float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    usable = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(usable):
        return None
    ordered = np.argsort(values[usable], kind="stable")
    sorted_values = values[usable][ordered]
    sorted_weights = weights[usable][ordered]
    cutoff = float(sorted_weights.sum()) * 0.5
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    areas = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    ) * 0.5
    return np.asarray(areas, dtype=np.float64)


def _view_center(view: Mapping[str, Any]) -> np.ndarray | None:
    try:
        rotation = np.asarray(view["rotation"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(view["translation"], dtype=np.float64).reshape(3)
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        return None
    center = -rotation.T @ translation
    return center if np.isfinite(center).all() else None


def _view_mask(view: Mapping[str, Any]) -> np.ndarray | None:
    value = view.get("mask_tight")
    if value is None:
        return None
    mask = np.asarray(value)
    if mask.ndim != 2 or mask.size == 0:
        return None
    binary = mask > 0
    return binary if np.any(binary) else None


def _view_image(view: Mapping[str, Any], width: int, height: int) -> np.ndarray | None:
    raw = view.get("image_bgr")
    image: np.ndarray | None
    if raw is not None:
        image = np.asarray(raw)
    else:
        path = view.get("image_path")
        image = cv2.imread(str(Path(path)), cv2.IMREAD_COLOR) if path else None
    if image is None or image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return None
    image = np.asarray(image[..., :3], dtype=np.uint8)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _view_depth(view: Mapping[str, Any], width: int, height: int) -> np.ndarray | None:
    raw = view.get("depth_map")
    if raw is None:
        return None
    depth = np.asarray(raw, dtype=np.float32)
    if depth.ndim != 2 or depth.size == 0:
        return None
    if depth.shape != (height, width):
        depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST)
    return depth


def _direction_bin_count(directions: np.ndarray, separation_degrees: float) -> int:
    """Greedily count reproducible, angularly separated camera directions."""

    if len(directions) == 0:
        return 0
    cosine_limit = math.cos(math.radians(separation_degrees))
    chosen: list[np.ndarray] = []
    for direction in directions:
        if not chosen or all(float(np.dot(direction, other)) < cosine_limit for other in chosen):
            chosen.append(direction)
    return len(chosen)


def _pose_metrics(
    views: Sequence[Mapping[str, Any]],
    object_center: np.ndarray,
    *,
    pose_mode: str | None,
    total_frame_count: int | None,
    sfm_registered_fraction: float | None,
    thresholds: GateThresholds,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    valid: list[tuple[Mapping[str, Any], np.ndarray, np.ndarray, float]] = []
    source_counts: dict[str, int] = {}
    independent_count = 0
    mode = (pose_mode or "").strip().lower()
    for view in views:
        center = _view_center(view)
        if center is None:
            continue
        offset = center - object_center
        radius = float(np.linalg.norm(offset))
        if not math.isfinite(radius) or radius <= 1e-9:
            continue
        direction = offset / radius
        source_raw = view.get("pose_source")
        source = str(source_raw).strip().lower() if source_raw is not None else "unspecified"
        source_counts[source] = source_counts.get(source, 0) + 1
        inferred = source in _INFERRED_POSE_SOURCES
        if source == "unspecified" and mode in {"silhouette", "turntable"}:
            inferred = True
        if not inferred:
            independent_count += 1
        valid.append((view, center, direction, radius))

    reasons: list[str] = []
    details: list[dict[str, Any]] = []

    def reject(code: str, message: str, observed: Any, required: Any) -> None:
        if code not in reasons:
            reasons.append(code)
            details.append(
                {"code": code, "message": message, "observed": observed, "required": required}
            )

    if not valid:
        metrics = {
            "valid_pose_views": 0,
            "invalid_pose_views": len(views),
            "pose_source_counts": source_counts,
            "independent_pose_views": 0,
            "direction_coverage_degrees": 0.0,
            "maximum_pairwise_view_angle_degrees": 0.0,
            "direction_bins": 0,
            "camera_radius_cv": None,
            "median_look_at_cosine": None,
            "sfm_registered_fraction": None,
        }
        reject(
            "insufficient_pose_views",
            "no finite camera poses were available",
            0,
            thresholds.minimum_mask_views,
        )
        reject(
            "insufficient_independent_pose_evidence",
            "too few poses came from an independent geometric estimator",
            0,
            thresholds.minimum_independent_pose_views,
        )
        return metrics, reasons, details

    directions = np.stack([item[2] for item in valid])
    radii = np.asarray([item[3] for item in valid], dtype=np.float64)
    gram = np.clip(directions @ directions.T, -1.0, 1.0)
    maximum_angle = float(np.degrees(np.arccos(gram)).max())

    # Project camera directions into their strongest two-dimensional subspace.
    # This measures either a normal orbit or a more general broad sweep without
    # assuming which world axis is vertical.
    second_moment = directions.T @ directions / max(len(directions), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(second_moment)
    basis = eigenvectors[:, np.argsort(eigenvalues)[-2:]]
    planar = directions @ basis
    lengths = np.linalg.norm(planar, axis=1)
    usable = lengths > 1e-5
    if int(np.count_nonzero(usable)) >= 3:
        angles = np.mod(np.arctan2(planar[usable, 1], planar[usable, 0]), 2.0 * np.pi)
        angles.sort()
        gaps = np.diff(np.concatenate((angles, angles[:1] + 2.0 * np.pi)))
        coverage = float(np.degrees(2.0 * np.pi - float(gaps.max())))
    else:
        coverage = 0.0

    look_cosines: list[float] = []
    for view, center, _direction, _radius in valid:
        rotation = np.asarray(view["rotation"], dtype=np.float64).reshape(3, 3)
        optical_forward = rotation[2]
        optical_forward /= max(float(np.linalg.norm(optical_forward)), 1e-12)
        toward_object = object_center - center
        toward_object /= max(float(np.linalg.norm(toward_object)), 1e-12)
        look_cosines.append(float(np.dot(optical_forward, toward_object)))

    radius_mean = float(np.mean(radii))
    radius_cv = float(np.std(radii) / max(radius_mean, 1e-12))
    direction_bins = _direction_bin_count(directions, thresholds.direction_bin_separation_degrees)
    registered_fraction: float | None
    if sfm_registered_fraction is not None:
        registered_fraction = float(sfm_registered_fraction)
    elif total_frame_count is not None and total_frame_count > 0:
        registered_fraction = float(independent_count / total_frame_count)
    else:
        registered_fraction = None

    metrics = {
        "valid_pose_views": len(valid),
        "invalid_pose_views": len(views) - len(valid),
        "pose_source_counts": dict(sorted(source_counts.items())),
        "independent_pose_views": independent_count,
        "direction_coverage_degrees": round(coverage, 6),
        "maximum_pairwise_view_angle_degrees": round(maximum_angle, 6),
        "direction_bins": int(direction_bins),
        "direction_bin_separation_degrees": thresholds.direction_bin_separation_degrees,
        "camera_radius_median": _finite_float(np.median(radii)),
        "camera_radius_cv": round(radius_cv, 6),
        "median_look_at_cosine": _finite_float(np.median(look_cosines)),
        "sfm_registered_fraction": (
            _finite_float(registered_fraction) if registered_fraction is not None else None
        ),
    }

    if len(valid) < thresholds.minimum_mask_views:
        reject(
            "insufficient_pose_views",
            "too few finite poses were available for a general reconstruction",
            len(valid),
            thresholds.minimum_mask_views,
        )
    if coverage < thresholds.minimum_pose_coverage_degrees:
        reject(
            "degenerate_camera_sweep",
            "camera directions occupy too little of their dominant orbit plane",
            round(coverage, 6),
            thresholds.minimum_pose_coverage_degrees,
        )
    if maximum_angle < thresholds.minimum_pairwise_view_angle_degrees:
        reject(
            "insufficient_opposing_views",
            "no camera pair observes sufficiently different sides of the mesh",
            round(maximum_angle, 6),
            thresholds.minimum_pairwise_view_angle_degrees,
        )
    if direction_bins < thresholds.minimum_direction_bins:
        reject(
            "insufficient_distinct_view_directions",
            "too few angularly distinct camera directions support the reconstruction",
            direction_bins,
            thresholds.minimum_direction_bins,
        )
    if radius_cv > thresholds.maximum_camera_radius_cv:
        reject(
            "unstable_camera_radius",
            "camera distances vary implausibly for one coherent object frame",
            round(radius_cv, 6),
            thresholds.maximum_camera_radius_cv,
        )
    median_look = float(np.median(look_cosines))
    if median_look < thresholds.minimum_median_look_at_cosine:
        reject(
            "cameras_do_not_converge_on_mesh",
            "camera optical axes do not consistently point toward the delivered mesh",
            round(median_look, 6),
            thresholds.minimum_median_look_at_cosine,
        )
    if independent_count < thresholds.minimum_independent_pose_views:
        reject(
            "insufficient_independent_pose_evidence",
            "too few poses came from SfM, calibration, or another independent geometric estimator",
            independent_count,
            thresholds.minimum_independent_pose_views,
        )
    if (
        registered_fraction is not None
        and registered_fraction < thresholds.minimum_sfm_registered_fraction
    ):
        reject(
            "weak_sfm_registration_fraction",
            "too little of the source capture has independently registered camera geometry",
            round(registered_fraction, 6),
            thresholds.minimum_sfm_registered_fraction,
        )
    return metrics, reasons, details


def _evenly_sample_views(
    views: Sequence[Mapping[str, Any]], maximum: int
) -> list[Mapping[str, Any]]:
    if len(views) <= maximum:
        return list(views)
    indices = np.linspace(0, len(views) - 1, maximum).round().astype(np.int64)
    return [views[int(index)] for index in sorted(set(indices.tolist()))]


def _surface_and_agreement_metrics(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: Sequence[Mapping[str, Any]],
    intrinsics: Intrinsics,
    thresholds: GateThresholds,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[str],
    list[dict[str, Any]],
    list[str],
]:
    areas = _triangle_areas(vertices, faces)
    total_area = float(areas.sum())
    centroids = vertices[faces].mean(axis=1)
    normals = face_normals(vertices, faces)
    mesh_diagonal = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-9)
    texture_support = np.zeros(len(faces), dtype=np.int32)
    depth_support = np.zeros(len(faces), dtype=np.int32)
    color_sum = np.zeros((len(faces), 3), dtype=np.float64)
    color_square_sum = np.zeros((len(faces), 3), dtype=np.float64)
    color_count = np.zeros(len(faces), dtype=np.int32)

    records: list[dict[str, Any]] = []
    texture_view_count = 0
    depth_view_records: list[dict[str, float]] = []
    holdout_ious: list[float] = []

    eligible = [
        view
        for view in views
        if _view_mask(view) is not None and _view_center(view) is not None
    ]
    evaluated = _evenly_sample_views(eligible, thresholds.maximum_evaluation_views)
    for index, view in enumerate(evaluated):
        mask = _view_mask(view)
        center = _view_center(view)
        assert mask is not None and center is not None
        height, width = mask.shape
        rotation = np.asarray(view["rotation"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(view["translation"], dtype=np.float64).reshape(3)
        zbuffer, _face_index = rasterize_zbuffer(
            vertices,
            faces,
            rotation,
            translation,
            intrinsics,
            width,
            height,
            scale=thresholds.render_scale,
        )
        predicted = np.isfinite(zbuffer)
        render_height, render_width = predicted.shape
        observed_mask = cv2.resize(
            mask.astype(np.uint8),
            (render_width, render_height),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        intersection = int(np.count_nonzero(predicted & observed_mask))
        union = int(np.count_nonzero(predicted | observed_mask))
        predicted_count = int(np.count_nonzero(predicted))
        mask_count = int(np.count_nonzero(observed_mask))
        iou = intersection / max(union, 1)
        precision = intersection / max(predicted_count, 1)
        recall = intersection / max(mask_count, 1)

        role = str(view.get("evaluation_role", "")).strip().lower()
        is_holdout = (
            role in {"holdout", "validation", "test"}
            or view.get("used_for_geometry") is False
        )
        if is_holdout:
            holdout_ious.append(float(iou))

        u, v, centroid_depth = project_points(centroids, rotation, translation, intrinsics)
        ui_render = np.clip(
            np.rint(u * render_width / width), 0, render_width - 1
        ).astype(np.int64)
        vi_render = np.clip(
            np.rint(v * render_height / height), 0, render_height - 1
        ).astype(np.int64)
        in_image = (
            (centroid_depth > 1e-9)
            & (u >= 0.0)
            & (u < width)
            & (v >= 0.0)
            & (v < height)
        )
        sampled_z = zbuffer[vi_render, ui_render]
        depth_visible = (
            in_image
            & np.isfinite(sampled_z)
            & (centroid_depth <= sampled_z + 0.015 * mesh_diagonal)
        )
        toward_camera = center[None, :] - centroids
        toward_camera /= np.maximum(np.linalg.norm(toward_camera, axis=1, keepdims=True), 1e-12)
        frontal = np.einsum("ij,ij->i", normals, toward_camera) >= 0.08
        ui_full = np.clip(np.rint(u), 0, width - 1).astype(np.int64)
        vi_full = np.clip(np.rint(v), 0, height - 1).astype(np.int64)
        centroid_in_mask = np.zeros(len(faces), dtype=bool)
        centroid_in_mask[in_image] = mask[vi_full[in_image], ui_full[in_image]]
        supported = depth_visible & frontal & centroid_in_mask

        image = _view_image(view, width, height)
        image_supported_area = 0.0
        if image is not None and np.any(supported):
            texture_view_count += 1
            texture_support[supported] += 1
            image_supported_area = float(areas[supported].sum() / max(total_area, 1e-12))
            face_indices = np.flatnonzero(supported)
            samples = bilinear_sample(image, u[face_indices], v[face_indices]) / 255.0
            color_sum[face_indices] += samples
            color_square_sum[face_indices] += samples**2
            color_count[face_indices] += 1

        depth_map = _view_depth(view, width, height)
        depth_record: dict[str, float] | None = None
        if depth_map is not None:
            small_depth = cv2.resize(
                depth_map, (render_width, render_height), interpolation=cv2.INTER_NEAREST
            )
            valid_pixels = (
                predicted
                & observed_mask
                & np.isfinite(small_depth)
                & (small_depth > 1e-9)
            )
            pixel_coverage = float(np.count_nonzero(valid_pixels) / max(intersection, 1))
            if np.any(valid_pixels):
                relative = np.abs(small_depth[valid_pixels] - zbuffer[valid_pixels]) / np.maximum(
                    np.maximum(np.abs(small_depth[valid_pixels]), np.abs(zbuffer[valid_pixels])),
                    1e-9,
                )
                median_relative_error = float(np.median(relative))
            else:
                median_relative_error = 1.0
            sampled_depth = np.zeros(len(faces), dtype=np.float64)
            sampled_depth[in_image] = depth_map[vi_full[in_image], ui_full[in_image]]
            face_relative = np.full(len(faces), np.inf, dtype=np.float64)
            face_valid = supported & np.isfinite(sampled_depth) & (sampled_depth > 1e-9)
            face_relative[face_valid] = np.abs(
                sampled_depth[face_valid] - centroid_depth[face_valid]
            ) / np.maximum(
                np.maximum(
                    np.abs(sampled_depth[face_valid]),
                    np.abs(centroid_depth[face_valid]),
                ),
                1e-9,
            )
            credible_faces = face_valid & (
                face_relative <= thresholds.maximum_median_depth_relative_error * 2.0
            )
            depth_support[credible_faces] += 1
            depth_record = {
                "pixel_coverage": round(pixel_coverage, 6),
                "median_relative_error": round(median_relative_error, 6),
            }
            depth_view_records.append(depth_record)

        records.append(
            {
                "name": str(view.get("name", f"view_{index:04d}")),
                "silhouette_iou": round(float(iou), 6),
                "silhouette_precision": round(float(precision), 6),
                "silhouette_recall": round(float(recall), 6),
                "holdout": bool(is_holdout),
                "source_image_available": image is not None,
                "source_supported_surface_fraction": round(image_supported_area, 6),
                "depth": depth_record,
            }
        )

    ious = [float(record["silhouette_iou"]) for record in records]
    good_fraction = (
        float(np.mean(np.asarray(ious) >= thresholds.minimum_silhouette_iou)) if ious else 0.0
    )
    agreement = {
        "eligible_mask_views": len(eligible),
        "evaluated_views": len(records),
        "median_iou": _percentile(ious, 50.0),
        "p10_iou": _percentile(ious, 10.0),
        "minimum_iou": min(ious) if ious else None,
        "good_view_fraction": round(good_fraction, 6),
        "good_view_iou_threshold": thresholds.minimum_silhouette_iou,
        "holdout_views": len(holdout_ious),
        "holdout_median_iou": _percentile(holdout_ious, 50.0),
        "per_view": records,
    }

    observed_fraction = float(areas[texture_support >= 1].sum() / max(total_area, 1e-12))
    multiview_fraction = float(areas[texture_support >= 2].sum() / max(total_area, 1e-12))
    color_eligible = color_count >= 2
    color_variance = np.zeros(len(faces), dtype=np.float64)
    if np.any(color_eligible):
        mean = color_sum[color_eligible] / color_count[color_eligible, None]
        mean_square = color_square_sum[color_eligible] / color_count[color_eligible, None]
        variance = np.maximum(mean_square - mean**2, 0.0)
        color_variance[color_eligible] = np.sqrt(np.mean(variance, axis=1))
    color_rms = _weighted_median(color_variance[color_eligible], areas[color_eligible])
    texture = {
        "source_image_views": texture_view_count,
        "observed_surface_fraction": round(observed_fraction, 6),
        "multiview_surface_fraction": round(multiview_fraction, 6),
        "median_multiview_color_rms_0_to_1": (
            _finite_float(color_rms) if color_rms is not None else None
        ),
        "maximum_face_view_support": int(texture_support.max(initial=0)),
        "basis": (
            "area-weighted mesh-face centroids that are visible, front-facing, "
            "and inside a source mask"
        ),
    }

    depth_errors = [item["median_relative_error"] for item in depth_view_records]
    depth_coverages = [item["pixel_coverage"] for item in depth_view_records]
    depth_surface_fraction = float(areas[depth_support >= 1].sum() / max(total_area, 1e-12))
    depth = {
        "depth_map_views": len(depth_view_records),
        "median_pixel_coverage": _percentile(depth_coverages, 50.0),
        "median_relative_error": _percentile(depth_errors, 50.0),
        "supported_surface_fraction": round(depth_surface_fraction, 6),
        "basis": (
            "delivered z-buffer compared directly with positive aligned source depth "
            "inside the object mask"
        ),
    }

    reasons: list[str] = []
    details: list[dict[str, Any]] = []
    warnings: list[str] = []

    def reject(code: str, message: str, observed: Any, required: Any) -> None:
        if code not in reasons:
            reasons.append(code)
            details.append(
                {"code": code, "message": message, "observed": observed, "required": required}
            )

    median_iou = agreement["median_iou"] or 0.0
    p10_iou = agreement["p10_iou"] or 0.0
    if len(records) < thresholds.minimum_mask_views:
        reject(
            "insufficient_mask_agreement_views",
            "too few posed object masks were available for mesh agreement",
            len(records),
            thresholds.minimum_mask_views,
        )
    if median_iou < thresholds.minimum_median_silhouette_iou:
        reject(
            "weak_median_silhouette_agreement",
            "the delivered mesh does not reproduce the typical source silhouette",
            median_iou,
            thresholds.minimum_median_silhouette_iou,
        )
    if p10_iou < thresholds.minimum_p10_silhouette_iou:
        reject(
            "weak_tail_silhouette_agreement",
            "the mesh strongly disagrees with at least one part of the camera sweep",
            p10_iou,
            thresholds.minimum_p10_silhouette_iou,
        )
    if good_fraction < thresholds.minimum_good_silhouette_fraction:
        reject(
            "too_few_silhouette_consistent_views",
            "too few source views agree with the delivered mesh",
            round(good_fraction, 6),
            thresholds.minimum_good_silhouette_fraction,
        )
    if len(holdout_ious) >= thresholds.minimum_holdout_views_for_gate:
        holdout_median = float(np.median(holdout_ious))
        if holdout_median < thresholds.minimum_holdout_median_iou:
            reject(
                "weak_holdout_silhouette_agreement",
                "masks not used for geometry disagree with the delivered mesh",
                round(holdout_median, 6),
                thresholds.minimum_holdout_median_iou,
            )
    else:
        warnings.append(
            "fewer than three views were marked as held-out; mask agreement may "
            "include reconstruction inputs"
        )

    if texture_view_count < thresholds.minimum_texture_views:
        reject(
            "insufficient_source_texture_views",
            "too few decodable source images can support delivered appearance",
            texture_view_count,
            thresholds.minimum_texture_views,
        )
    if observed_fraction < thresholds.minimum_observed_surface_fraction:
        reject(
            "insufficient_observed_texture_surface",
            "too much mesh area lacks a direct source-image observation",
            round(observed_fraction, 6),
            thresholds.minimum_observed_surface_fraction,
        )
    if multiview_fraction < thresholds.minimum_multiview_surface_fraction:
        reject(
            "insufficient_multiview_texture_support",
            "too little mesh area is corroborated by more than one source image",
            round(multiview_fraction, 6),
            thresholds.minimum_multiview_surface_fraction,
        )
    if color_rms is not None and color_rms > thresholds.maximum_multiview_color_rms:
        reject(
            "inconsistent_multiview_appearance",
            "the same supported surface locations have incompatible source colours",
            round(color_rms, 6),
            thresholds.maximum_multiview_color_rms,
        )

    return agreement, texture, depth, reasons, details, warnings


def _sfm_point_metrics(
    points: np.ndarray | None,
    views: Sequence[Mapping[str, Any]],
    intrinsics: Intrinsics,
    vertices: np.ndarray,
    thresholds: GateThresholds,
) -> tuple[dict[str, Any], bool]:
    if points is None:
        return {
            "available": False,
            "point_count": 0,
            "second_axis_ratio": None,
            "extent_to_mesh_diagonal": None,
            "mask_consistent_fraction": None,
            "median_view_support": None,
            "passed": False,
        }, False
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("sfm_points must have shape (N, 3)")
    array = array[np.isfinite(array).all(axis=1)]
    if len(array) == 0:
        return {
            "available": True,
            "point_count": 0,
            "second_axis_ratio": None,
            "extent_to_mesh_diagonal": 0.0,
            "mask_consistent_fraction": 0.0,
            "median_view_support": 0.0,
            "passed": False,
        }, False

    centered = array - np.median(array, axis=0)
    covariance = centered.T @ centered / max(len(array), 1)
    eigenvalues = np.sort(np.maximum(np.linalg.eigvalsh(covariance), 0.0))[::-1]
    second_ratio = float(eigenvalues[1] / max(eigenvalues[0], 1e-12))
    point_extent = float(np.linalg.norm(np.ptp(array, axis=0)))
    mesh_diagonal = max(float(np.linalg.norm(np.ptp(vertices, axis=0))), 1e-12)
    extent_ratio = point_extent / mesh_diagonal

    support = np.zeros(len(array), dtype=np.int32)
    for view in views:
        mask = _view_mask(view)
        if mask is None or _view_center(view) is None:
            continue
        height, width = mask.shape
        u, v, depth = project_points(
            array,
            np.asarray(view["rotation"], dtype=np.float64),
            np.asarray(view["translation"], dtype=np.float64),
            intrinsics,
        )
        inside = (depth > 1e-9) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        ui = np.clip(np.rint(u), 0, width - 1).astype(np.int64)
        vi = np.clip(np.rint(v), 0, height - 1).astype(np.int64)
        on_mask = np.zeros(len(array), dtype=bool)
        on_mask[inside] = mask[vi[inside], ui[inside]]
        support += on_mask.astype(np.int32)

    consistent_fraction = float(np.mean(support >= 2))
    median_support = float(np.median(support))
    passed = (
        len(array) >= thresholds.minimum_sfm_points
        and second_ratio >= thresholds.minimum_sfm_point_second_axis_ratio
        and extent_ratio >= thresholds.minimum_sfm_point_extent_ratio
        and consistent_fraction >= thresholds.minimum_sfm_point_mask_consistent_fraction
        and median_support >= thresholds.minimum_sfm_point_median_view_support
    )
    return {
        "available": True,
        "point_count": int(len(array)),
        "second_axis_ratio": round(second_ratio, 6),
        "extent_to_mesh_diagonal": round(extent_ratio, 6),
        "mask_consistent_fraction": round(consistent_fraction, 6),
        "median_view_support": round(median_support, 6),
        "passed": bool(passed),
    }, bool(passed)


def assess_reconstruction(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: Sequence[Mapping[str, Any]],
    intrinsics: Intrinsics,
    *,
    sfm_points: np.ndarray | None = None,
    pose_mode: str | None = None,
    total_frame_count: int | None = None,
    sfm_registered_fraction: float | None = None,
    thresholds: GateThresholds | None = None,
) -> dict[str, Any]:
    """Assess whether a mesh has enough independent source evidence to deliver.

    ``views`` use the plain view dictionaries from :mod:`local3d.recon_common`.
    A view contributes mask agreement when it has ``mask_tight``; appearance
    evidence additionally needs ``image_bgr`` or a decodable ``image_path``;
    direct depth evidence additionally needs a positive ``depth_map`` in camera
    z units.  ``pose_source`` may be ``sfm``, ``charuco``, ``silhouette``, etc.
    Missing pose-source labels are treated as independent unless
    ``pose_mode`` explicitly says ``turntable``/``silhouette``.

    The function never mutates inputs or writes artifacts.  Invalid array
    contracts raise :class:`ValueError`; insufficient reconstruction evidence
    returns ``accepted=False`` with stable reason codes.
    """

    config = thresholds or GateThresholds()
    vertices_array = np.asarray(vertices, dtype=np.float64)
    faces_array = np.asarray(faces, dtype=np.int64)
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3 or len(vertices_array) < 4:
        raise ValueError("vertices must have shape (N, 3) with at least four vertices")
    if faces_array.ndim != 2 or faces_array.shape[1] != 3 or len(faces_array) < 4:
        raise ValueError("faces must have shape (M, 3) with at least four triangles")
    if not np.isfinite(vertices_array).all():
        raise ValueError("vertices must be finite")
    if np.any(faces_array < 0) or np.any(faces_array >= len(vertices_array)):
        raise ValueError("faces reference vertices outside the vertex array")
    intrinsics_tuple = tuple(float(value) for value in intrinsics)
    if len(intrinsics_tuple) != 4 or not all(math.isfinite(value) for value in intrinsics_tuple):
        raise ValueError("intrinsics must contain four finite SIMPLE_RADIAL values")
    if intrinsics_tuple[0] <= 0.0:
        raise ValueError("intrinsics focal length must be positive")
    if total_frame_count is not None and total_frame_count < 1:
        raise ValueError("total_frame_count must be positive when provided")
    if sfm_registered_fraction is not None and (
        not math.isfinite(float(sfm_registered_fraction))
        or not 0.0 <= float(sfm_registered_fraction) <= 1.0
    ):
        raise ValueError("sfm_registered_fraction must be finite and in [0, 1]")
    if not 0.05 <= config.render_scale <= 1.0:
        raise ValueError("thresholds.render_scale must be in [0.05, 1.0]")
    if config.minimum_mask_views < 1 or config.maximum_evaluation_views < 1:
        raise ValueError("view-count thresholds must be positive")

    areas = _triangle_areas(vertices_array, faces_array)
    if not np.isfinite(areas).all() or float(areas.sum()) <= 1e-12:
        raise ValueError("mesh triangles have no finite surface area")

    # A textured mesh commonly duplicates vertices at UV seams.  The bounding
    # box midpoint is invariant to those duplicates, unlike the arithmetic
    # vertex mean.
    object_center = (vertices_array.min(axis=0) + vertices_array.max(axis=0)) * 0.5
    pose_metrics, pose_reasons, pose_details = _pose_metrics(
        views,
        object_center,
        pose_mode=pose_mode,
        total_frame_count=total_frame_count,
        sfm_registered_fraction=sfm_registered_fraction,
        thresholds=config,
    )
    agreement, texture, depth, surface_reasons, surface_details, warnings = (
        _surface_and_agreement_metrics(
            vertices_array,
            faces_array,
            views,
            intrinsics_tuple,  # type: ignore[arg-type]
            config,
        )
    )
    sfm, sfm_passed = _sfm_point_metrics(
        sfm_points,
        views,
        intrinsics_tuple,  # type: ignore[arg-type]
        vertices_array,
        config,
    )

    depth_map_passed = (
        depth["depth_map_views"] >= config.minimum_depth_views
        and (depth["median_pixel_coverage"] or 0.0) >= config.minimum_depth_pixel_coverage
        and depth["supported_surface_fraction"] >= config.minimum_depth_surface_fraction
        and (depth["median_relative_error"] if depth["median_relative_error"] is not None else 1.0)
        <= config.maximum_median_depth_relative_error
    )

    reasons = list(pose_reasons) + [
        reason for reason in surface_reasons if reason not in pose_reasons
    ]
    pose_detail_codes = {item["code"] for item in pose_details}
    details = list(pose_details) + [
        detail for detail in surface_details if detail["code"] not in pose_detail_codes
    ]
    if not depth_map_passed and not sfm_passed:
        reasons.append("insufficient_geometric_depth_support")
        details.append(
            {
                "code": "insufficient_geometric_depth_support",
                "message": (
                    "neither aligned depth maps nor a spatially supported SfM point cloud "
                    "provides independent depth evidence"
                ),
                "observed": {
                    "depth_map_views": depth["depth_map_views"],
                    "depth_surface_fraction": depth["supported_surface_fraction"],
                    "sfm_points": sfm["point_count"],
                },
                "required": "aligned depth-map evidence or a passing SfM point cloud",
            }
        )

    geometry_mode = (
        "depth_maps_and_sfm_points"
        if depth_map_passed and sfm_passed
        else "aligned_depth_maps"
        if depth_map_passed
        else "sfm_points"
        if sfm_passed
        else "unsupported"
    )
    metrics = {
        "mesh": {
            "vertices": int(len(vertices_array)),
            "triangles": int(len(faces_array)),
            "surface_area": _finite_float(areas.sum()),
            "bounds": [
                [float(value) for value in vertices_array.min(axis=0)],
                [float(value) for value in vertices_array.max(axis=0)],
            ],
        },
        "pose": pose_metrics,
        "mask_agreement": agreement,
        "texture_support": texture,
        "depth_support": depth,
        "sfm_points": sfm,
        "geometric_evidence_mode": geometry_mode,
    }
    # Stable ordering keeps reports diff-friendly and avoids duplicated reason
    # codes if two sub-gates identify the same evidence failure.
    unique_reasons = list(dict.fromkeys(reasons))
    detail_by_code = {detail["code"]: detail for detail in details}
    unique_details = [detail_by_code[code] for code in unique_reasons]
    return {
        "schema_version": 1,
        "accepted": not unique_reasons,
        "status": "accepted" if not unique_reasons else "rejected",
        "reasons": unique_reasons,
        "reason_details": unique_details,
        "warnings": warnings,
        "metrics": metrics,
        "thresholds": asdict(config),
        "method": (
            "object-agnostic camera diversity + z-buffer/mask agreement + "
            "area-weighted observed appearance + independent depth evidence"
        ),
    }


__all__ = ["GateThresholds", "assess_reconstruction"]
