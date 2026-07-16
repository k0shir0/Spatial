from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from local3d.depth_backends import DepthPrediction

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_depth_backends", ROOT / "scripts" / "compare_depth_backends.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _snapshot(tmp_path: Path, *, adequate: bool = True):
    names = ("frame_a.png", "frame_b.png", "frame_c.png", "frame_d.png")
    views = {}
    evidence_views = {}
    for index, name in enumerate(names):
        image_path = tmp_path / name
        Image.new("RGB", (4, 3), (20 * index, 40, 80)).save(image_path)
        angle = 2.0 * np.pi * index / len(names)
        rotation = np.eye(3)
        translation = np.array([-3.0 * np.cos(angle), 0.0, -3.0 * np.sin(angle)])
        views[name] = {
            "name": name,
            "image_path": image_path,
            "rotation": rotation,
            "translation": translation,
            "mask_tight": np.ones((3, 4), dtype=bool),
        }
        evidence_views[name] = {
            "point3d_ids": np.arange(12),
            "xy": np.tile([[1.0, 1.0]], (12, 1)),
            "z_camera": np.linspace(2.0, 3.0, 12),
            "track_lengths": np.full(12, 4),
            "reprojection_errors_px": np.full(12, 0.5),
        }
    coverage = {
        "azimuth_span_degrees": 240.0 if adequate else 20.0,
        "occupied_30deg_bins": 8 if adequate else 1,
        "elevation_span_degrees": 10.0,
        "maximum_opposing_angle_degrees": 180.0 if adequate else 15.0,
    }
    return MODULE.FrozenSfm(
        views=views,
        intrinsics=(100.0, 2.0, 1.5, 0.0),
        track_evidence={"views": evidence_views},
        points_xyz=np.zeros((12, 3)),
        object_center=np.zeros(3),
        selected_names=names,
        registered_view_count=20 if adequate else 3,
        total_frame_count=30,
        coverage=coverage,
        selected_coverage=coverage,
        provenance={
            "schema": "test.freeze.v1",
            "selected_names": list(names),
            "selected_inputs": {},
        },
    )


def _config(tmp_path: Path, **overrides):
    values = {
        "sfm_model_dir": tmp_path / "sfm",
        "frames_dir": tmp_path,
        "masks_dir": tmp_path,
        "output_dir": tmp_path / "experiment",
        "cache_dir": tmp_path / "cache",
        "depth_anything_model": tmp_path / "model.onnx",
        "maximum_frames": 4,
        "minimum_registered_views": 4,
        "minimum_registered_fraction": 0.2,
    }
    values.update(overrides)
    return MODULE.ExperimentConfig(**values)


def _predictions(expected, representation):
    return {
        frame["id"]: DepthPrediction(
            values=np.full((frame["height"], frame["width"]), 2.0, dtype=np.float32),
            representation=representation,
            source_id=frame["id"],
            focal_length_px=(frame["focal_length_px"] if representation == "metric_depth_m" else None),
            provenance={"fake": True},
        )
        for frame in expected
    }


def test_preflight_failure_publishes_rejection_without_running_models(tmp_path: Path):
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("model must not run")

    report = MODULE.run_experiment(
        _config(tmp_path),
        snapshot=_snapshot(tmp_path, adequate=False),
        depth_anything_provider=forbidden,
        depth_pro_provider=forbidden,
    )

    assert not calls
    assert report["final_selection"]["decision"] == "reject"
    assert report["final_selection"]["reason"] == "sfm_preflight_failed_before_depth_inference"
    persisted = json.loads((tmp_path / "experiment" / "report.json").read_text())
    assert persisted["sfm_preflight"]["accepted"] is False
    assert (tmp_path / "experiment" / "frozen_evidence.npz").is_file()


def test_backends_receive_identical_frozen_frames_and_outputs_are_persisted(tmp_path: Path):
    seen = {}

    def da_provider(config, expected):
        seen["da"] = tuple(frame["id"] for frame in expected)
        return _predictions(expected, "relative_disparity"), {"cache_hit": False}

    def dp_provider(config, expected):
        seen["dp"] = tuple(frame["id"] for frame in expected)
        return _predictions(expected, "metric_depth_m"), {"cache_hit": True}

    def evaluator(name, predictions, views, intrinsics, evidence, **kwargs):
        assert tuple(predictions) == tuple(views)
        depths = {key: np.full((3, 4), 2.5, dtype=np.float32) for key in predictions}
        return SimpleNamespace(
            name=name,
            report={"accepted": True, "quality_score": 0.8, "backend": name},
            aligned_depths=depths,
            aligned_confidences={key: np.ones((3, 4), dtype=np.float32) for key in predictions},
        )

    def selector(evaluations):
        winner = evaluations["apple_depth_pro"]
        return SimpleNamespace(
            report={
                "decision": "selected",
                "reason": "test",
                "selected_backend": "apple_depth_pro",
            },
            aligned_depths=winner.aligned_depths,
            aligned_confidences=winner.aligned_confidences,
        )

    report = MODULE.run_experiment(
        _config(tmp_path, emit_geometry_inputs=True),
        snapshot=_snapshot(tmp_path),
        depth_anything_provider=da_provider,
        depth_pro_provider=dp_provider,
        evaluator=evaluator,
        selector=selector,
    )

    assert seen["da"] == seen["dp"] == _snapshot(tmp_path).selected_names
    assert report["final_selection"]["selected_backend"] == "apple_depth_pro"
    assert (tmp_path / "experiment" / "aligned" / "apple_depth_pro" / "frame_a.png.npz").is_file()
    geometry = tmp_path / "experiment" / report["geometry_inputs"]["manifest"]
    assert geometry.is_file()


def test_diagnostic_scoring_can_never_promote_inadequate_sfm(tmp_path: Path):
    def da_provider(config, expected):
        return _predictions(expected, "relative_disparity"), {"cache_hit": True}

    def no_dp(config, expected):
        return None, {"available": False}

    def evaluator(name, predictions, views, intrinsics, evidence, **kwargs):
        return SimpleNamespace(
            name=name,
            report={"accepted": True, "quality_score": 1.0},
            aligned_depths={key: np.ones((3, 4), dtype=np.float32) for key in predictions},
            aligned_confidences={},
        )

    def selector(evaluations):
        evaluation = next(iter(evaluations.values()))
        return SimpleNamespace(
            report={"decision": "selected", "selected_backend": evaluation.name},
            aligned_depths=evaluation.aligned_depths,
            aligned_confidences={},
        )

    report = MODULE.run_experiment(
        _config(
            tmp_path,
            diagnostic_on_inadequate_sfm=True,
            emit_geometry_inputs=True,
        ),
        snapshot=_snapshot(tmp_path, adequate=False),
        depth_anything_provider=da_provider,
        depth_pro_provider=no_dp,
        evaluator=evaluator,
        selector=selector,
    )

    assert report["raw_depth_selection"]["decision"] == "selected"
    assert report["final_selection"]["decision"] == "reject"
    assert report["geometry_inputs"] is None


def test_cache_loader_rejects_tampered_npz(tmp_path: Path):
    expected = [
        {
            "id": "frame.png",
            "image_path": str(tmp_path / "frame.png"),
            "input_sha256": "a" * 64,
            "width": 4,
            "height": 3,
            "focal_length_px": 100.0,
        }
    ]
    cache = tmp_path / "cache"
    (cache / "predictions").mkdir(parents=True)
    prediction = cache / "predictions" / "frame.png.npz"
    with prediction.open("wb") as handle:
        np.savez_compressed(handle, values=np.ones((3, 4), dtype=np.float32))
    MODULE._atomic_json(
        cache / "input_manifest.json",
        {"backend": "fake", "frames": expected},
    )
    MODULE._atomic_json(
        cache / "provenance.json",
        {
            "backend": "fake",
            "frames": [
                {
                    "id": "frame.png",
                    "input_sha256": "a" * 64,
                    "npz_path": "predictions/frame.png.npz",
                    "output_sha256": "0" * 64,
                }
            ]
        },
    )

    try:
        MODULE._prediction_cache_records(
            cache,
            expected,
            backend="fake",
            representation="relative_disparity",
        )
    except ValueError as error:
        assert "output hash mismatch" in str(error)
    else:
        raise AssertionError("tampered cache was accepted")


def test_default_subsampling_preserves_camera_direction_coverage():
    views = {}
    # Lexicographic/temporal order deliberately places several near-duplicate
    # directions first; an index-prefix sampler would see only a narrow arc.
    angles = (0, 3, 6, 9, 60, 120, 180, 240, 300)
    for index, degrees in enumerate(angles):
        radians = np.radians(degrees)
        views[f"frame_{index:02d}.png"] = {
            "rotation": np.eye(3),
            "translation": np.array(
                [-3.0 * np.cos(radians), 0.0, -3.0 * np.sin(radians)]
            ),
        }

    names = MODULE._farthest_direction_names(views, np.zeros(3), 4)
    coverage = MODULE._camera_coverage([views[name] for name in names], np.zeros(3))

    assert len(names) == 4
    assert coverage["maximum_opposing_angle_degrees"] >= 170.0
    assert names == MODULE._farthest_direction_names(dict(reversed(list(views.items()))), np.zeros(3), 4)


def test_valid_depth_pro_cache_reuses_exact_frozen_frame(tmp_path: Path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (4, 3), "purple").save(image)
    input_hash = MODULE._sha256_file(image)
    expected = [
        {
            "id": "frame.png",
            "image_path": str(image),
            "input_sha256": input_hash,
            "width": 4,
            "height": 3,
            "focal_length_px": 100.0,
        }
    ]
    cache = tmp_path / "depth_pro"
    predictions = cache / "predictions"
    predictions.mkdir(parents=True)
    prediction_path = predictions / "frame.png.npz"
    with prediction_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            depth_m=np.full((3, 4), 1.75, dtype=np.float32),
            focal_length_px=np.float32(100.0),
        )
    identity = {
        "backend": "apple_depth_pro",
        "model_commit": "a" * 40,
        "checkpoint_sha256": "b" * 64,
    }
    MODULE._atomic_json(
        cache / "input_manifest.json", {**identity, "frames": expected}
    )
    MODULE._atomic_json(
        cache / "provenance.json",
        {
            **identity,
            "frames": [
                {
                    "id": "frame.png",
                    "input_sha256": input_hash,
                    "npz_path": "predictions/frame.png.npz",
                    "output_sha256": MODULE._sha256_file(prediction_path),
                }
            ],
        },
    )

    loaded = MODULE._prediction_cache_records(
        cache,
        expected,
        backend="apple_depth_pro",
        representation="metric_depth_m",
    )

    assert list(loaded) == ["frame.png"]
    assert loaded["frame.png"].provenance["cache_hit"] is True
    np.testing.assert_array_equal(loaded["frame.png"].values, np.full((3, 4), 1.75))
