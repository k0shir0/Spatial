"""Synthetic tests for held-out alignment and multi-view depth evidence."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.depth_consistency import (  # noqa: E402
    DepthBackendEvaluation,
    DepthConsistencyThresholds,
    align_depth_predictions,
    deterministic_point_id_split,
    evaluate_depth_backend,
    evaluate_depth_predictions,
    score_symmetric_reprojection,
    select_depth_backend,
    select_eligible_view_pairs,
)


SIZE = 96
INTRINSICS = (80.0, SIZE / 2.0, SIZE / 2.0, 0.0)


def _look_at(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=np.float64)
    forward = -center / np.linalg.norm(center)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    if np.linalg.norm(right) < 1e-8:
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    return rotation, -rotation @ center


def _view(name: str, center: np.ndarray, mask: np.ndarray | None = None) -> dict:
    rotation, translation = _look_at(center)
    return {
        "name": name,
        "rotation": rotation,
        "translation": translation,
        "mask_tight": np.ones((SIZE, SIZE), bool) if mask is None else mask,
    }


def _track_fixture(
    view_names: list[str], *, count: int = 36, z_by_view: dict[str, float] | None = None
) -> tuple[dict, dict[str, np.ndarray], dict[str, dict]]:
    ids = np.arange(1000, 1000 + count, dtype=np.int64)
    # Non-integer positions exercise bilinear sampling but remain separated.
    x = 8.25 + (np.arange(count) % 9) * 8.0
    y = 8.25 + (np.arange(count) // 9) * 15.0
    xy = np.column_stack((x, y))
    world_points = np.column_stack(
        (
            np.linspace(-0.9, 0.9, count),
            0.55 * np.sin(np.linspace(0.0, 3.0 * np.pi, count)),
            0.35 * np.cos(np.linspace(0.0, 2.0 * np.pi, count)),
        )
    )
    evidence = {
        "point3d_ids": ids.copy(),
        "points_xyz": world_points,
        "views": {},
    }
    predictions: dict[str, np.ndarray] = {}
    views: dict[str, dict] = {}
    for index, name in enumerate(view_names):
        base_z = (z_by_view or {}).get(name, 2.2 + 0.05 * index)
        z = base_z + np.linspace(-0.35, 0.35, count)
        inverse = 1.0 / z
        grid_y, grid_x = np.indices((SIZE, SIZE), dtype=np.float64)
        inferred_index = ((grid_y - 8.25) / 15.0) * 9.0 + (grid_x - 8.25) / 8.0
        dense_z = base_z - 0.35 + (0.70 / max(count - 1, 1)) * inferred_index
        dense_z = np.maximum(dense_z, 0.5)
        prediction = (1.0 / dense_z).astype(np.float32)
        # A 2x2 constant patch makes bilinear samples exact.
        for (u, v), value in zip(xy, inverse):
            ui, vi = int(np.floor(u)), int(np.floor(v))
            prediction[vi : vi + 2, ui : ui + 2] = value
        predictions[name] = prediction
        evidence["views"][name] = {
            "point3d_ids": ids.copy(),
            "xy": xy.copy(),
            "xyz_world": world_points.copy(),
            "z_camera": z,
            "track_lengths": np.full(count, 5, np.int32),
            "reprojection_errors_px": np.full(count, 0.25),
        }
        angle = index * (360.0 / max(len(view_names), 1))
        radians = np.radians(angle)
        views[name] = _view(
            name, np.array([3.0 * np.sin(radians), 0.35, 3.0 * np.cos(radians)])
        )
    return evidence, predictions, views


def _plane_view(name: str, center_x: float) -> tuple[dict, np.ndarray]:
    # Parallel cameras observe a finite plane at world z=3.  The mask is the
    # exact projection of x∈[-1,1], y∈[-0.7,0.7].
    rotation = np.eye(3)
    translation = np.array([-center_x, 0.0, 0.0])
    yy, xx = np.indices((SIZE, SIZE), dtype=np.float64)
    world_x = center_x + (xx - INTRINSICS[1]) / INTRINSICS[0] * 3.0
    world_y = (yy - INTRINSICS[2]) / INTRINSICS[0] * 3.0
    mask = (np.abs(world_x) <= 1.0) & (np.abs(world_y) <= 0.7)
    depth = np.where(mask, 3.0, 0.0).astype(np.float32)
    view = {
        "name": name,
        "rotation": rotation,
        "translation": translation,
        "mask_tight": mask,
    }
    return view, depth


def test_global_point_split_and_alignment_are_order_stable() -> None:
    ids = list(range(50))
    first = deterministic_point_id_split(ids)
    second = deterministic_point_id_split(reversed(ids))
    assert first == second
    assert set(first[0]).isdisjoint(first[1])

    evidence, predictions, views = _track_fixture(["a", "b", "c", "d"])
    thresholds = replace(
        DepthConsistencyThresholds(),
        minimum_calibration_points_per_view=6,
        minimum_evaluation_points_per_view=4,
    )
    aligned_a, report_a = align_depth_predictions(
        predictions, views, evidence, prediction_space="inverse_depth", thresholds=thresholds
    )
    shuffled_evidence = {
        "point3d_ids": evidence["point3d_ids"][::-1],
        "points_xyz": evidence["points_xyz"][::-1],
        "views": {},
    }
    for name in reversed(list(evidence["views"])):
        record = evidence["views"][name]
        order = np.arange(len(record["point3d_ids"]))[::-1]
        shuffled_evidence["views"][name] = {
            key: np.asarray(value)[order] for key, value in record.items()
        }
    aligned_b, report_b = align_depth_predictions(
        dict(reversed(list(predictions.items()))),
        dict(reversed(list(views.items()))),
        shuffled_evidence,
        prediction_space="inverse_depth",
        thresholds=thresholds,
    )
    assert report_a == report_b
    assert aligned_a.keys() == aligned_b.keys()
    for name in aligned_a:
        np.testing.assert_array_equal(aligned_a[name], aligned_b[name])


def test_heldout_tracks_reject_overfit_negative_and_degenerate_inverse_predictions() -> None:
    evidence, predictions, views = _track_fixture(["only"], count=45)
    record = evidence["views"]["only"]
    calibration, evaluation = deterministic_point_id_split(record["point3d_ids"])
    calibration = set(calibration)
    evaluation = set(evaluation)
    xy = record["xy"]
    z = record["z_camera"]

    overfit = predictions["only"].copy()
    for point_id, (u, v), value in zip(record["point3d_ids"], xy, 1.0 / z):
        token = f"int:{int(point_id)}"
        corrupted = value if token in calibration else value * (1.8 if token in evaluation else 1.0)
        ui, vi = int(np.floor(u)), int(np.floor(v))
        overfit[vi : vi + 2, ui : ui + 2] = corrupted
    _, overfit_report = align_depth_predictions(
        {"only": overfit}, views, evidence, prediction_space="inverse_depth"
    )
    assert not overfit_report["views"]["only"]["accepted"]
    assert "heldout_median_error_too_high" in overfit_report["views"]["only"]["reasons"]
    assert overfit_report["views"]["only"]["calibration_error"]["median"] < 1e-5

    negative = -predictions["only"]
    _, negative_report = align_depth_predictions(
        {"only": negative}, views, evidence, prediction_space="inverse_depth"
    )
    assert "nonpositive_inverse_depth_slope" in negative_report["views"]["only"]["reasons"]

    constant = np.ones_like(predictions["only"])
    _, constant_report = align_depth_predictions(
        {"only": constant}, views, evidence, prediction_space="inverse_depth"
    )
    assert "degenerate_prediction_range" in constant_report["views"]["only"]["reasons"]


def test_metric_mode_uses_one_scale_and_rejects_a_drifting_frame() -> None:
    names = ["a", "b", "c", "drift"]
    evidence, inverse_predictions, views = _track_fixture(names, count=48)
    metric: dict[str, np.ndarray] = {}
    for name in names:
        metric[name] = np.where(
            inverse_predictions[name] > 0.0, 1.0 / inverse_predictions[name], 0.0
        ).astype(np.float32)
    metric["drift"] *= 1.65
    aligned, report = align_depth_predictions(
        metric, views, evidence, prediction_space="metric_depth"
    )
    scales = {
        view_report["slope_or_scale"]
        for view_report in report["views"].values()
        if view_report["slope_or_scale"] is not None
    }
    assert len(scales) == 1
    assert "drift" not in aligned
    assert any(
        reason.startswith("heldout_") for reason in report["views"]["drift"]["reasons"]
    )


def test_track_patched_near_flat_depth_cannot_pass_nonplanar_sfm_object() -> None:
    evidence, dense_predictions, views = _track_fixture(["only"], count=45)
    record = evidence["views"]["only"]
    # Adversarially write the right value only around every sparse SfM sample,
    # leaving >95% of the object as a generic fronto-parallel plane.  Both the
    # calibration and held-out points fit, so only dense-relief evidence can
    # expose the under-modelled geometry.
    flat = np.full_like(dense_predictions["only"], np.median(dense_predictions["only"]))
    for (u, v), z in zip(record["xy"], record["z_camera"]):
        ui, vi = int(np.floor(u)), int(np.floor(v))
        flat[vi : vi + 2, ui : ui + 2] = 1.0 / z
    aligned, report = align_depth_predictions(
        {"only": flat}, views, evidence, prediction_space="inverse_depth"
    )
    view_report = report["views"]["only"]
    assert view_report["calibration_error"]["median"] < 1e-5
    assert view_report["heldout_error"]["median"] < 1e-5
    assert "insufficient_depth_relief_for_sfm_tracks" in view_report["reasons"]
    assert view_report["depth_relief"]["prediction_to_sfm_span_ratio"] < 0.1
    assert aligned == {}


def test_perfect_synthetic_depth_reprojects_symmetrically() -> None:
    view_a, depth_a = _plane_view("a", -0.20)
    view_b, depth_b = _plane_view("b", 0.25)
    thresholds = replace(
        DepthConsistencyThresholds(),
        minimum_comparable_pixels_per_pair=20,
    )
    report = score_symmetric_reprojection(
        view_a, view_b, depth_a, depth_b, INTRINSICS, thresholds=thresholds
    )
    assert report["totals"]["comparable_pixels"] > 500
    assert report["totals"]["consistency"] == 1.0
    assert report["totals"]["free_space_contradictions"] == 0
    json.dumps(report, allow_nan=False, sort_keys=True)


def test_reprojection_distinguishes_occlusion_from_free_space() -> None:
    view, depth = _plane_view("same", 0.0)
    nearer = depth.copy()
    nearer[35:61, 35:61] = 2.0
    report = score_symmetric_reprojection(view, view, depth, nearer, INTRINSICS)
    # Far source points are legitimately hidden by the nearer target patch.
    assert report["forward"]["occluded_by_target"] > 100
    # Reversing the direction asserts the nearer patch in the other view's
    # observed free space, which is a contradiction rather than an occlusion.
    assert report["backward"]["free_space_contradictions"] > 100


def test_pair_selection_spans_bins_and_is_deterministic() -> None:
    names = [f"v{index}" for index in range(8)]
    views = {}
    for index, name in enumerate(names):
        angle = np.radians(index * 45.0)
        views[name] = _view(name, np.array([3.0 * np.sin(angle), 0.3, 3.0 * np.cos(angle)]))
    thresholds = replace(DepthConsistencyThresholds(), maximum_selected_pairs=10)
    first = select_eligible_view_pairs(views, names, thresholds=thresholds)
    second = select_eligible_view_pairs(
        dict(reversed(list(views.items()))), reversed(names), thresholds=thresholds
    )
    assert first == second
    assert first["occupied_angle_bin_count"] >= 4
    assert first["maximum_selected_angle_degrees"] > 160.0


def test_aggregate_rejects_missing_camera_coverage() -> None:
    names = ["a", "b", "c", "d"]
    evidence, predictions, _ = _track_fixture(names, count=48)
    # Identical poses remove every eligible angular pair, even though each
    # frame's held-out track calibration is perfect.
    views = {name: _view(name, np.array([0.0, 0.2, 3.0])) for name in names}
    report = evaluate_depth_backend(
        "relative",
        predictions,
        views,
        INTRINSICS,
        evidence,
        prediction_space="inverse_depth",
    ).report
    assert not report["accepted"]
    assert "insufficient_angular_pair_coverage" in report["reasons"]
    assert report["alignment"]["aligned_view_count"] == len(names)


@dataclass
class _Prediction:
    values: np.ndarray
    representation: str
    source_id: str = "synthetic"
    focal_length_px: float = INTRINSICS[0]
    provenance: dict | None = None


def test_depth_prediction_adapter_and_backend_selection_are_deterministic() -> None:
    names = ["a", "b", "c", "d"]
    evidence, arrays, views = _track_fixture(names, count=48)
    predictions = {
        name: _Prediction(values, "relative_disparity", provenance={"offline": True})
        for name, values in arrays.items()
    }
    evaluation = evaluate_depth_predictions(
        "adapter", predictions, views, INTRINSICS, evidence
    )
    assert evaluation.report["prediction_contract"]["representation"] == "relative_disparity"
    assert evaluation.aligned_confidences.keys() == evaluation.aligned_depths.keys()

    depth = np.full((20, 20), 2.0, np.float32)
    confidence = np.ones_like(depth)
    first = DepthBackendEvaluation(
        "first", {"accepted": True, "quality_score": 0.81}, {"v": depth}, {"v": confidence}
    )
    second = DepthBackendEvaluation(
        "second",
        {"accepted": True, "quality_score": 0.80},
        {"v": depth * 1.01},
        {"v": confidence},
    )
    consensus_a = select_depth_backend([first, second])
    consensus_b = select_depth_backend([second, first])
    assert consensus_a.report == consensus_b.report
    assert consensus_a.report["decision"] == "consensus"
    assert np.all(consensus_a.aligned_depths["v"] > 0.0)

    disagreeing = DepthBackendEvaluation(
        "second",
        {"accepted": True, "quality_score": 0.80},
        {"v": depth * 1.30},
        {"v": confidence},
    )
    rejected = select_depth_backend([first, disagreeing])
    assert rejected.report["decision"] == "reject"
    assert rejected.aligned_depths == {}

    materially_better = DepthBackendEvaluation(
        "best", {"accepted": True, "quality_score": 0.95}, {"v": depth}, {"v": confidence}
    )
    selected = select_depth_backend([second, materially_better])
    assert selected.report["decision"] == "selected"
    assert selected.report["selected_backend"] == "best"


def test_metric_prediction_values_remain_optical_axis_camera_z() -> None:
    names = ["a", "b", "c", "d"]
    evidence, _arrays, views = _track_fixture(names, count=48)
    for record in evidence["views"].values():
        record["z_camera"] = np.full(48, 3.0, dtype=np.float64)
    predictions = {
        name: _Prediction(
            np.full((SIZE, SIZE), 3.0, dtype=np.float32),
            "metric_depth_m",
            source_id="depth_pro_semantics_regression",
        )
        for name in names
    }
    result = evaluate_depth_predictions(
        "metric", predictions, views, INTRINSICS, evidence
    )
    assert (
        result.report["prediction_contract"]["metric_conversion"]
        == "none_already_optical_axis_camera_z"
    )
    assert result.aligned_depths.keys() == predictions.keys()
    for depth in result.aligned_depths.values():
        np.testing.assert_allclose(depth, 3.0, rtol=0.0, atol=1e-6)


def test_consensus_zeros_pixels_where_backends_disagree() -> None:
    first_depth = np.full((20, 20), 2.0, np.float32)
    second_depth = first_depth.copy()
    second_depth[5:15, 5:15] = 3.0
    report = {"accepted": True, "quality_score": 0.8}
    first = DepthBackendEvaluation("a", report, {"v": first_depth})
    second = DepthBackendEvaluation("b", report, {"v": second_depth})
    thresholds = replace(
        DepthConsistencyThresholds(), consensus_median_relative_difference=0.30
    )
    # The global median difference is 0 (75% agreement), so this is a valid
    # consensus, but its locally disagreeing quarter must not reach fusion.
    selected = select_depth_backend([first, second], thresholds=thresholds)
    assert selected.report["decision"] == "consensus"
    assert np.all(selected.aligned_depths["v"][5:15, 5:15] == 0.0)
    assert np.all(selected.aligned_confidences["v"][5:15, 5:15] == 0.0)
