"""Explicit, offline depth-prediction backends.

This module deliberately stops at prediction I/O.  It does not decide whether
a prediction is geometrically consistent enough to fuse into a reconstruction.
In particular, a metric label from a monocular model is not treated as measured
depth; callers still need the independent multi-view evidence gate.

``DepthProSubprocessBackend`` keeps Apple's external reference implementation
in its own pinned Python environment.  The host process supplies an exact interpreter, an exact
checkpoint, and a source commit.  The only executable script is the runner
shipped with this repository, arguments are never passed through a shell, and
the child process is forced into offline mode.  Batch results are validated and
then promoted with a single directory rename.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps

from local3d.monodepth import DepthEstimator

DepthRepresentation = Literal["metric_depth_m", "relative_disparity"]

_FRAME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    """Hash an array without losing its dtype or shape identity."""

    value = np.ascontiguousarray(array)
    header = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256(header)
    digest.update(b"\0")
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DepthPrediction:
    """One validated depth-like image plus sufficient provenance to audit it.

    ``relative_disparity`` values are unitless and larger means nearer.
    ``metric_depth_m`` values are positive optical-axis camera-Z metres.  This
    matches Apple's Depth Pro output and the COLMAP/SfM depth convention used
    downstream; it is not Euclidean range along the viewing ray.  Values are
    copied, converted to float32, and made read-only so the provenance does not
    silently become detached from mutated prediction pixels.
    """

    values: np.ndarray
    representation: DepthRepresentation
    source_id: str
    focal_length_px: float | None
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float32, order="C", copy=True)
        if values.ndim != 2 or values.size == 0:
            raise ValueError("depth prediction must be a non-empty H x W array")
        if not np.isfinite(values).all():
            raise ValueError("depth prediction contains non-finite values")
        if self.representation == "metric_depth_m" and not np.all(values > 0.0):
            raise ValueError("metric depth values must be strictly positive")
        if self.representation not in ("metric_depth_m", "relative_disparity"):
            raise ValueError(f"unsupported depth representation: {self.representation!r}")
        if not self.source_id:
            raise ValueError("source_id must not be empty")
        focal = self.focal_length_px
        if focal is not None and (not math.isfinite(float(focal)) or float(focal) <= 0.0):
            raise ValueError("focal_length_px must be finite and positive")
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


@dataclass(frozen=True)
class DepthFrameInput:
    """A full-resolution RGB frame and its already-calibrated focal length."""

    frame_id: str
    image_path: Path
    focal_length_px: float


class DepthAnythingV2Adapter:
    """Adapt the existing local ONNX estimator to :class:`DepthPrediction`.

    Depth Anything emits relative disparity, not metric depth.  This adapter
    intentionally does not run the SfM alignment from :mod:`local3d.monodepth`;
    that remains a later, separately gated operation.
    """

    def __init__(
        self,
        model_path: Path,
        *,
        threads: int = 4,
        estimator: DepthEstimator | None = None,
    ) -> None:
        model = Path(model_path).expanduser().resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Depth Anything model not found: {model}")
        self.model_path = model
        self.model_sha256 = _sha256_file(model)
        self.estimator = estimator or DepthEstimator(model, threads=threads)

    def predict(self, image_bgr: np.ndarray, *, source_id: str) -> DepthPrediction:
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_bgr must be an H x W x 3 array")
        disparity = self.estimator.disparity(image)
        return DepthPrediction(
            values=disparity,
            representation="relative_disparity",
            source_id=source_id,
            focal_length_px=None,
            provenance={
                "schema_version": 1,
                "backend": "depth_anything_v2_onnx",
                "model_path": str(self.model_path),
                "model_sha256": self.model_sha256,
                "device": "cpu",
                "precision": "float32",
                "input_sha256": _sha256_array(image),
            },
        )

    def predict_batch_by_id(
        self, images_bgr: Mapping[str, np.ndarray]
    ) -> dict[str, DepthPrediction]:
        """Predict named in-memory frames, preserving mapping iteration order."""

        if not images_bgr:
            raise ValueError("at least one Depth Anything frame is required")
        return {
            source_id: self.predict(image, source_id=source_id)
            for source_id, image in images_bgr.items()
        }


def _default_depth_pro_runner() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "depth_pro_batch.py"


def _offline_environment() -> dict[str, str]:
    """Sanitise Python injection hooks and force common model clients offline."""

    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONINSPECT"):
        env.pop(name, None)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        }
    )
    return env


def _run_subprocess(
    command: list[str], *, cwd: Path, env: Mapping[str, str], timeout: float
) -> subprocess.CompletedProcess[str]:
    """Small seam for model-free tests; production always uses ``shell=False``."""

    return subprocess.run(
        command,
        cwd=str(cwd),
        env=dict(env),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        shell=False,
        check=False,
        timeout=timeout,
    )


class DepthProSubprocessBackend:
    """Run a pinned Apple Depth Pro environment through the fixed batch runner."""

    def __init__(
        self,
        *,
        python_executable: Path,
        checkpoint_path: Path,
        model_commit: str,
        device: Literal["mps", "cpu", "cuda"] = "mps",
        allow_non_mps: bool = False,
        timeout_seconds: float = 3600.0,
    ) -> None:
        python_raw = Path(python_executable).expanduser()
        checkpoint_raw = Path(checkpoint_path).expanduser()
        if not python_raw.is_absolute() or not checkpoint_raw.is_absolute():
            raise ValueError("python_executable and checkpoint_path must be absolute paths")
        # Do not resolve this symlink: ``venv/bin/python`` commonly points at a
        # base interpreter, and Python finds ``pyvenv.cfg`` from the invoked
        # symlink path.  Resolving it here would silently leave the pinned Depth
        # Pro environment behind.
        python = Path(os.path.abspath(python_raw))
        checkpoint = checkpoint_raw.resolve()
        runner = _default_depth_pro_runner().resolve()
        if not python.is_file() or not os.access(python, os.X_OK):
            raise FileNotFoundError(f"Depth Pro Python is not executable: {python}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Depth Pro checkpoint not found: {checkpoint}")
        if not runner.is_file():
            raise FileNotFoundError(f"fixed Depth Pro runner not found: {runner}")
        if not _COMMIT_RE.fullmatch(model_commit):
            raise ValueError("model_commit must be a full 40-character hexadecimal Git SHA")
        if device != "mps" and not allow_non_mps:
            raise ValueError("non-MPS execution requires allow_non_mps=True")
        if not math.isfinite(timeout_seconds) or not 1.0 <= timeout_seconds <= 86400.0:
            raise ValueError("timeout_seconds must be between 1 and 86400")

        self.python_executable = python
        self.checkpoint_path = checkpoint
        self.checkpoint_sha256 = _sha256_file(checkpoint)
        self.model_commit = model_commit.lower()
        self.runner_path = runner
        self.device = device
        self.allow_non_mps = bool(allow_non_mps)
        self.timeout_seconds = float(timeout_seconds)

    @staticmethod
    def _normalise_frames(frames: Sequence[DepthFrameInput]) -> list[dict[str, Any]]:
        if not frames:
            raise ValueError("at least one Depth Pro frame is required")
        seen: set[str] = set()
        normalised: list[dict[str, Any]] = []
        for frame in frames:
            if not _FRAME_ID_RE.fullmatch(frame.frame_id):
                raise ValueError(f"unsafe frame_id: {frame.frame_id!r}")
            if frame.frame_id in seen:
                raise ValueError(f"duplicate frame_id: {frame.frame_id}")
            seen.add(frame.frame_id)
            image_path = Path(frame.image_path).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(f"input frame not found: {image_path}")
            focal = float(frame.focal_length_px)
            if not math.isfinite(focal) or focal <= 0.0:
                raise ValueError(f"invalid focal_length_px for {frame.frame_id}")
            with Image.open(image_path) as image:
                width, height = ImageOps.exif_transpose(image).size
            if width <= 0 or height <= 0:
                raise ValueError(f"empty input frame: {image_path}")
            normalised.append(
                {
                    "id": frame.frame_id,
                    "image_path": str(image_path),
                    "input_sha256": _sha256_file(image_path),
                    "width": int(width),
                    "height": int(height),
                    "focal_length_px": focal,
                }
            )
        return normalised

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")

    def _validate_outputs(
        self, staging: Path, frames: list[dict[str, Any]]
    ) -> list[DepthPrediction]:
        provenance_path = staging / "provenance.json"
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("Depth Pro runner did not write valid provenance.json") from error
        if provenance.get("schema_version") != 1:
            raise RuntimeError("unsupported Depth Pro provenance schema")
        if provenance.get("backend") != "apple_depth_pro":
            raise RuntimeError("Depth Pro provenance backend mismatch")
        if provenance.get("model_commit") != self.model_commit:
            raise RuntimeError("Depth Pro source commit mismatch")
        if provenance.get("checkpoint_sha256") != self.checkpoint_sha256:
            raise RuntimeError("Depth Pro checkpoint hash mismatch")
        if provenance.get("device") != self.device or provenance.get("precision") != "float16":
            raise RuntimeError("Depth Pro execution provenance mismatch")
        records = provenance.get("frames")
        if not isinstance(records, list) or len(records) != len(frames):
            raise RuntimeError("Depth Pro frame provenance is incomplete")

        predictions: list[DepthPrediction] = []
        for expected, record in zip(frames, records, strict=True):
            if not isinstance(record, dict) or record.get("id") != expected["id"]:
                raise RuntimeError("Depth Pro frame order/id mismatch")
            if record.get("input_sha256") != expected["input_sha256"]:
                raise RuntimeError(f"input hash mismatch for {expected['id']}")
            relative = f"predictions/{expected['id']}.npz"
            if record.get("npz_path") != relative:
                raise RuntimeError(f"unexpected prediction path for {expected['id']}")
            prediction_path = staging / relative
            if not prediction_path.is_file():
                raise RuntimeError(f"missing prediction for {expected['id']}")
            output_hash = _sha256_file(prediction_path)
            if not _SHA256_RE.fullmatch(str(record.get("output_sha256", ""))):
                raise RuntimeError(f"invalid output hash for {expected['id']}")
            if output_hash != record["output_sha256"]:
                raise RuntimeError(f"prediction hash mismatch for {expected['id']}")
            try:
                with np.load(prediction_path, allow_pickle=False) as archive:
                    if set(archive.files) != {"depth_m", "focal_length_px"}:
                        raise RuntimeError(f"unexpected NPZ fields for {expected['id']}")
                    depth = np.asarray(archive["depth_m"], dtype=np.float32)
                    focal = float(np.asarray(archive["focal_length_px"]).reshape(()))
            except (OSError, ValueError) as error:
                raise RuntimeError(f"invalid prediction NPZ for {expected['id']}") from error
            if depth.shape != (expected["height"], expected["width"]):
                raise RuntimeError(f"prediction dimensions mismatch for {expected['id']}")
            if not math.isclose(focal, expected["focal_length_px"], rel_tol=1e-5):
                raise RuntimeError(f"prediction focal length mismatch for {expected['id']}")
            frame_provenance = dict(record)
            frame_provenance.update(
                {
                    "schema_version": 1,
                    "backend": "apple_depth_pro",
                    "model_commit": self.model_commit,
                    "checkpoint_sha256": self.checkpoint_sha256,
                    "device": self.device,
                    "precision": "float16",
                }
            )
            predictions.append(
                DepthPrediction(
                    values=depth,
                    representation="metric_depth_m",
                    source_id=expected["id"],
                    focal_length_px=focal,
                    provenance=frame_provenance,
                )
            )
        return predictions

    def predict_batch(
        self, frames: Sequence[DepthFrameInput], *, output_dir: Path
    ) -> list[DepthPrediction]:
        """Predict a batch and atomically publish its validated files.

        ``output_dir`` must not already exist.  On any child-process or output
        validation failure, its private sibling staging directory is removed
        and nothing is published at the requested path.
        """

        normalised = self._normalise_frames(frames)
        target = Path(output_dir).expanduser().resolve()
        if target.exists():
            raise FileExistsError(f"Depth Pro output directory already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.depth-pro-", dir=target.parent))
        try:
            manifest = {
                "schema_version": 1,
                "backend": "apple_depth_pro",
                "model_commit": self.model_commit,
                "checkpoint_sha256": self.checkpoint_sha256,
                "frames": normalised,
            }
            manifest_path = staging / "input_manifest.json"
            self._write_json(manifest_path, manifest)
            command = [
                str(self.python_executable),
                "-I",
                str(self.runner_path),
                "--manifest",
                str(manifest_path),
                "--checkpoint",
                str(self.checkpoint_path),
                "--expected-checkpoint-sha256",
                self.checkpoint_sha256,
                "--model-commit",
                self.model_commit,
                "--output-dir",
                str(staging / "predictions"),
                "--provenance",
                str(staging / "provenance.json"),
                "--device",
                self.device,
            ]
            if self.allow_non_mps:
                command.append("--allow-non-mps")
            result = _run_subprocess(
                command,
                cwd=self.runner_path.parent,
                env=_offline_environment(),
                timeout=self.timeout_seconds,
            )
            if result.returncode != 0:
                stderr = (result.stderr or "").strip()[-4000:]
                raise RuntimeError(
                    f"Depth Pro runner failed with exit code {result.returncode}: {stderr}"
                )
            predictions = self._validate_outputs(staging, normalised)
            os.replace(staging, target)
            for prediction in predictions:
                record = dict(prediction.provenance)
                record["npz_path"] = str(target / record["npz_path"])
                object.__setattr__(prediction, "provenance", MappingProxyType(record))
            return predictions
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    def predict_batch_by_id(
        self, frames: Sequence[DepthFrameInput], *, output_dir: Path
    ) -> dict[str, DepthPrediction]:
        """Return :meth:`predict_batch` as an insertion-ordered name mapping."""

        predictions = self.predict_batch(frames, output_dir=output_dir)
        return {prediction.source_id: prediction for prediction in predictions}
