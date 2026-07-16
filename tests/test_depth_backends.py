from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

import local3d.depth_backends as depth_backends
from local3d.depth_backends import (
    DepthAnythingV2Adapter,
    DepthFrameInput,
    DepthPrediction,
    DepthProSubprocessBackend,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeEstimator:
    def disparity(self, image: np.ndarray) -> np.ndarray:
        return np.mean(image.astype(np.float32), axis=2)


def test_depth_anything_adapter_preserves_relative_depth_semantics(tmp_path: Path):
    model = tmp_path / "depth_anything.onnx"
    model.write_bytes(b"pinned onnx model")
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    backend = DepthAnythingV2Adapter(model, estimator=_FakeEstimator())

    prediction = backend.predict(image, source_id="frame-1")

    assert prediction.representation == "relative_disparity"
    assert prediction.focal_length_px is None
    assert prediction.values.shape == (4, 5)
    assert prediction.provenance["model_sha256"] == _sha256(model)
    assert prediction.provenance["device"] == "cpu"
    assert not prediction.values.flags.writeable

    batch = backend.predict_batch_by_id({"left": image, "right": image[:, ::-1]})
    assert list(batch) == ["left", "right"]
    assert all(item.representation == "relative_disparity" for item in batch.values())


def test_prediction_rejects_nonpositive_metric_depth():
    with pytest.raises(ValueError, match="strictly positive"):
        DepthPrediction(
            values=np.array([[1.0, 0.0]], dtype=np.float32),
            representation="metric_depth_m",
            source_id="frame",
            focal_length_px=100.0,
            provenance={},
        )


def _make_backend(tmp_path: Path, **kwargs) -> DepthProSubprocessBackend:
    python = tmp_path / "depth-pro-python"
    python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python.chmod(0o755)
    checkpoint = tmp_path / "depth_pro.pt"
    checkpoint.write_bytes(b"checkpoint bytes")
    return DepthProSubprocessBackend(
        python_executable=python,
        checkpoint_path=checkpoint,
        model_commit="a" * 40,
        **kwargs,
    )


def _fake_successful_runner(command, *, cwd, env, timeout):
    def argument(name: str) -> Path:
        return Path(command[command.index(name) + 1])

    manifest = json.loads(argument("--manifest").read_text(encoding="utf-8"))
    output_dir = argument("--output-dir")
    provenance_path = argument("--provenance")
    output_dir.mkdir()
    records = []
    for frame in manifest["frames"]:
        path = output_dir / f"{frame['id']}.npz"
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                depth_m=np.full(
                    (frame["height"], frame["width"]), 2.5, dtype=np.float32
                ),
                focal_length_px=np.float32(frame["focal_length_px"]),
            )
        records.append(
            {
                "id": frame["id"],
                "input_sha256": frame["input_sha256"],
                "npz_path": f"predictions/{frame['id']}.npz",
                "output_sha256": _sha256(path),
            }
        )
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": "apple_depth_pro",
                "model_commit": manifest["model_commit"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "device": command[command.index("--device") + 1],
                "precision": "float16",
                "frames": records,
            }
        ),
        encoding="utf-8",
    )
    assert command[1] == "-I"
    assert command[2].endswith("scripts/depth_pro_batch.py")
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"
    assert "PYTHONPATH" not in env
    return subprocess.CompletedProcess(command, 0, "", "")


def test_depth_pro_subprocess_validates_and_atomically_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    backend = _make_backend(tmp_path)
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (7, 5), (20, 40, 60)).save(frame_path)
    monkeypatch.setattr(depth_backends, "_run_subprocess", _fake_successful_runner)
    output = tmp_path / "depth-output"

    predictions = backend.predict_batch(
        [DepthFrameInput("frame_0001", frame_path, 720.0)], output_dir=output
    )

    assert output.is_dir()
    assert (output / "input_manifest.json").is_file()
    assert (output / "provenance.json").is_file()
    assert (output / "predictions" / "frame_0001.npz").is_file()
    assert len(predictions) == 1
    assert predictions[0].representation == "metric_depth_m"
    assert predictions[0].focal_length_px == pytest.approx(720.0)
    np.testing.assert_array_equal(predictions[0].values, np.full((5, 7), 2.5))
    assert predictions[0].provenance["npz_path"] == str(
        output / "predictions" / "frame_0001.npz"
    )


def test_depth_pro_batch_by_id_preserves_frame_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    backend = _make_backend(tmp_path)
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (3, 2), "green").save(first)
    Image.new("RGB", (3, 2), "yellow").save(second)
    monkeypatch.setattr(depth_backends, "_run_subprocess", _fake_successful_runner)

    predictions = backend.predict_batch_by_id(
        [
            DepthFrameInput("first", first, 400.0),
            DepthFrameInput("second", second, 400.0),
        ],
        output_dir=tmp_path / "named-output",
    )

    assert list(predictions) == ["first", "second"]
    assert predictions["second"].source_id == "second"


def test_depth_pro_rejects_tampered_output_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    backend = _make_backend(tmp_path)
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3), "red").save(frame_path)

    def tampered_runner(command, **kwargs):
        result = _fake_successful_runner(command, **kwargs)
        provenance_path = Path(command[command.index("--provenance") + 1])
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        payload["frames"][0]["output_sha256"] = "0" * 64
        provenance_path.write_text(json.dumps(payload), encoding="utf-8")
        return result

    monkeypatch.setattr(depth_backends, "_run_subprocess", tampered_runner)
    output = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="prediction hash mismatch"):
        backend.predict_batch(
            [DepthFrameInput("frame", frame_path, 500.0)], output_dir=output
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".must-not-exist.depth-pro-*"))


def test_depth_pro_requires_pinned_inputs_and_explicit_non_mps(tmp_path: Path):
    python = tmp_path / "python"
    python.write_text("#!/bin/sh\n", encoding="utf-8")
    python.chmod(0o755)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")
    with pytest.raises(ValueError, match="40-character"):
        DepthProSubprocessBackend(
            python_executable=python,
            checkpoint_path=checkpoint,
            model_commit="main",
        )
    with pytest.raises(ValueError, match="allow_non_mps"):
        DepthProSubprocessBackend(
            python_executable=python,
            checkpoint_path=checkpoint,
            model_commit="b" * 40,
            device="cpu",
        )


def test_depth_pro_preserves_virtualenv_python_symlink(tmp_path: Path):
    actual_python = tmp_path / "base-python"
    actual_python.write_text("#!/bin/sh\n", encoding="utf-8")
    actual_python.chmod(0o755)
    environment = tmp_path / "depth-pro-env" / "bin"
    environment.mkdir(parents=True)
    venv_python = environment / "python"
    venv_python.symlink_to(actual_python)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model")

    backend = DepthProSubprocessBackend(
        python_executable=venv_python,
        checkpoint_path=checkpoint,
        model_commit="c" * 40,
    )

    assert backend.python_executable == venv_python


def test_depth_pro_rejects_unsafe_or_duplicate_frame_ids(tmp_path: Path):
    backend = _make_backend(tmp_path)
    frame_path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3), "blue").save(frame_path)
    with pytest.raises(ValueError, match="unsafe frame_id"):
        backend.predict_batch(
            [DepthFrameInput("../escape", frame_path, 500.0)],
            output_dir=tmp_path / "out-a",
        )
    with pytest.raises(ValueError, match="duplicate frame_id"):
        backend.predict_batch(
            [
                DepthFrameInput("same", frame_path, 500.0),
                DepthFrameInput("same", frame_path, 500.0),
            ],
            output_dir=tmp_path / "out-b",
        )
