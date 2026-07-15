#!/usr/bin/env python3
"""Compare monocular depth backends on one frozen existing-video SfM result.

The experiment boundary is intentionally strict:

* COLMAP poses, intrinsics, sparse track observations, masks, and frame names
  are loaded once and frozen before any depth model runs;
* every backend predicts exactly those same full-resolution RGB files;
* cached predictions are accepted only when all input/output hashes match;
* :mod:`local3d.depth_consistency` performs held-out track alignment and dense
  symmetric multi-view scoring; and
* weak SfM registration or angular coverage always forces the final decision
  to ``reject``, even when ``--diagnostic-on-inadequate-sfm`` asks to compute
  backend metrics for investigation.

No model is downloaded.  Depth Anything requires an explicit local ONNX file.
Depth Pro is optional and requires either a valid cache or an explicit pinned
Python/checkpoint/commit configuration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence

import cv2
import numpy as np

# Make direct checkout execution use this checkout, rather than an unrelated
# editable installation from another worktree.  Installed-package execution is
# unchanged because the same module path resolves normally.
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from local3d.depth_backends import (
    DepthAnythingV2Adapter,
    DepthFrameInput,
    DepthPrediction,
    DepthProSubprocessBackend,
)
from local3d.depth_consistency import (
    DepthBackendEvaluation,
    DepthSelectionResult,
    evaluate_depth_predictions,
    select_depth_backend,
)
from local3d.recon_common import Intrinsics

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ExperimentConfig:
    sfm_model_dir: Path
    frames_dir: Path
    masks_dir: Path
    output_dir: Path
    cache_dir: Path
    depth_anything_model: Path
    maximum_frames: int = 12
    minimum_registered_views: int = 8
    minimum_registered_fraction: float = 0.20
    minimum_azimuth_span_degrees: float = 120.0
    minimum_occupied_30deg_bins: int = 5
    minimum_opposing_angle_degrees: float = 90.0
    diagnostic_on_inadequate_sfm: bool = False
    emit_geometry_inputs: bool = False
    depth_pro_python: Path | None = None
    depth_pro_checkpoint: Path | None = None
    depth_pro_commit: str | None = None
    depth_pro_device: str = "mps"
    allow_non_mps: bool = False
    disable_depth_pro: bool = False


@dataclass
class FrozenSfm:
    views: dict[str, dict[str, Any]]
    intrinsics: Intrinsics
    track_evidence: dict[str, Any]
    points_xyz: np.ndarray
    object_center: np.ndarray
    selected_names: tuple[str, ...]
    registered_view_count: int
    total_frame_count: int
    coverage: dict[str, Any]
    selected_coverage: dict[str, Any]
    provenance: dict[str, Any]


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_arrays(items: Sequence[tuple[str, np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, array in items:
        value = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def _strict_json(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray | float]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _mask_for_frame(masks_dir: Path, name: str) -> tuple[Path, np.ndarray] | None:
    stem = Path(name).stem
    candidates = (
        masks_dir / f"{stem}_mask.png",
        masks_dir / f"{name}.png",
        masks_dir / name,
    )
    for path in candidates:
        if not path.is_file():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.ndim == 2 and np.any(mask > 127):
            return path.resolve(), mask > 127
    return None


def _camera_coverage(views: Sequence[Mapping[str, Any]], center: np.ndarray) -> dict[str, Any]:
    directions: list[np.ndarray] = []
    for view in views:
        rotation = np.asarray(view["rotation"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(view["translation"], dtype=np.float64).reshape(3)
        camera_center = -rotation.T @ translation
        direction = camera_center - center
        norm = float(np.linalg.norm(direction))
        if np.isfinite(direction).all() and norm > 1e-9:
            directions.append(direction / norm)
    if len(directions) < 2:
        return {
            "azimuth_span_degrees": 0.0,
            "occupied_30deg_bins": 0,
            "elevation_span_degrees": 0.0,
            "maximum_opposing_angle_degrees": 0.0,
        }
    array = np.asarray(directions)
    centered = array - array.mean(axis=0)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u, axis_v = vh[0], vh[1]
    normal = np.cross(axis_u, axis_v)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    azimuth = np.mod(np.degrees(np.arctan2(array @ axis_v, array @ axis_u)), 360.0)
    ordered = np.sort(azimuth)
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 360.0)))
    span = 360.0 - float(gaps.max())
    bins = len({int(value // 30.0) % 12 for value in azimuth})
    elevation = np.degrees(np.arcsin(np.clip(array @ normal, -1.0, 1.0)))
    dots = np.clip(array @ array.T, -1.0, 1.0)
    maximum_angle = float(np.degrees(np.arccos(np.min(dots))))
    return {
        "azimuth_span_degrees": round(span, 6),
        "occupied_30deg_bins": int(bins),
        "elevation_span_degrees": round(float(np.ptp(elevation)), 6),
        "maximum_opposing_angle_degrees": round(maximum_angle, 6),
    }


def _farthest_direction_names(
    views: Mapping[str, Mapping[str, Any]], center: np.ndarray, maximum: int
) -> tuple[str, ...]:
    """Deterministic angular farthest-point sampling, returned name-sorted."""

    names = sorted(views)
    if len(names) <= maximum:
        return tuple(names)
    directions: dict[str, np.ndarray] = {}
    for name in names:
        view = views[name]
        rotation = np.asarray(view["rotation"], dtype=np.float64).reshape(3, 3)
        translation = np.asarray(view["translation"], dtype=np.float64).reshape(3)
        offset = -rotation.T @ translation - center
        norm = float(np.linalg.norm(offset))
        if norm > 1e-9 and np.isfinite(offset).all():
            directions[name] = offset / norm
    if len(directions) <= maximum:
        return tuple(sorted(directions))
    selected = [sorted(directions)[0]]
    while len(selected) < maximum:
        remaining = sorted(set(directions) - set(selected))
        choice = max(
            remaining,
            key=lambda name: (
                min(
                    math.acos(
                        float(np.clip(np.dot(directions[name], directions[other]), -1.0, 1.0))
                    )
                    for other in selected
                ),
                # ``max`` would otherwise prefer the lexicographically largest tie.
                tuple(-ord(character) for character in name),
            ),
        )
        selected.append(choice)
    return tuple(sorted(selected))


def _cached_frame_names(cache_dir: Path) -> tuple[str, ...] | None:
    manifest_path = cache_dir / "depth_pro" / "input_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        frames = manifest["frames"]
        names = tuple(str(item["id"]) for item in frames)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid cached Depth Pro input manifest: {manifest_path}") from error
    if not names or len(names) != len(set(names)) or any(
        not _SAFE_NAME_RE.fullmatch(name) for name in names
    ):
        raise ValueError("cached Depth Pro manifest has unsafe/duplicate frame IDs")
    return names


def load_frozen_sfm(config: ExperimentConfig) -> FrozenSfm:
    """Load one existing COLMAP model and attach its already-generated masks."""

    import pycolmap
    from local3d.sfm_video import (
        _build_views,
        _intrinsics,
        _points_xyz,
        _prune_views_by_coherence,
        _track_evidence,
    )

    model_dir = config.sfm_model_dir.expanduser().resolve()
    frames_dir = config.frames_dir.expanduser().resolve()
    masks_dir = config.masks_dir.expanduser().resolve()
    if not model_dir.is_dir() or not frames_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError("SfM model, frames, and masks directories must all exist")
    reconstruction = pycolmap.Reconstruction(str(model_dir))
    raw_views = _build_views(reconstruction, frames_dir)
    eligible: dict[str, dict[str, Any]] = {}
    mask_paths: dict[str, Path] = {}
    missing_masks: list[str] = []
    for view in raw_views:
        name = str(view["name"])
        image_path = frames_dir / name
        mask_result = _mask_for_frame(masks_dir, name)
        if not image_path.is_file() or mask_result is None:
            missing_masks.append(name)
            continue
        mask_path, mask = mask_result
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or mask.shape != image.shape[:2]:
            missing_masks.append(name)
            continue
        record = dict(view)
        record["image_path"] = image_path.resolve()
        record["mask_tight"] = mask
        record["pose_source"] = "sfm"
        eligible[name] = record
        mask_paths[name] = mask_path
    points = np.asarray(_points_xyz(reconstruction, min_track_length=3), dtype=np.float64)
    intrinsics = tuple(float(value) for value in _intrinsics(reconstruction))
    coherent_views, coherent_ious, pruning_report = _prune_views_by_coherence(
        list(eligible.values()), intrinsics, points
    )
    coherent_names = {str(view["name"]) for view in coherent_views}
    pruned_names = sorted(set(eligible) - coherent_names)
    eligible = {name: eligible[name] for name in sorted(coherent_names)}
    center = np.median(points, axis=0) if len(points) else np.zeros(3, dtype=np.float64)
    coverage = _camera_coverage(list(eligible.values()), center)
    cached_names = (
        None
        if config.disable_depth_pro
        else _cached_frame_names(config.cache_dir.expanduser().resolve())
    )
    if cached_names is not None:
        unknown = sorted(set(cached_names) - set(eligible))
        if unknown:
            raise ValueError(
                "Depth Pro cache does not describe this frozen SfM/mask set: "
                + ", ".join(unknown[:8])
            )
        if len(cached_names) > config.maximum_frames:
            raise ValueError(
                f"cached Depth Pro batch has {len(cached_names)} frames, above --maximum-frames"
            )
        selected_names = tuple(cached_names)
        selection_method = "existing_depth_pro_cache_manifest"
    else:
        selected_names = _farthest_direction_names(eligible, center, config.maximum_frames)
        selection_method = "deterministic_angular_farthest_point"
    selected_views = {name: eligible[name] for name in selected_names}
    selected_coverage = _camera_coverage(list(selected_views.values()), center)
    evidence = _track_evidence(
        reconstruction, list(selected_views.values()), min_track_length=3
    )
    total_frames = len(
        [
            path
            for path in frames_dir.iterdir()
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ]
    )
    model_files = sorted(path for path in model_dir.iterdir() if path.is_file())
    provenance = {
        "schema": "local3d.frozen_depth_experiment_sfm.v1",
        "model_dir": str(model_dir),
        "model_files": {path.name: _sha256_file(path) for path in model_files},
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "eligible_registered_names": sorted(eligible),
        "missing_or_invalid_masks": sorted(missing_masks),
        "coherence_pruning": {
            "report": pruning_report,
            "pruned_names": pruned_names,
            "retained_view_ious": (
                {
                    str(view["name"]): float(iou)
                    for view, iou in zip(coherent_views, coherent_ious, strict=True)
                }
                if len(coherent_views) == len(coherent_ious)
                else {}
            ),
        },
        "selected_names": list(selected_names),
        "selection_method": selection_method,
        "selected_inputs": {
            name: {
                "image_path": str(selected_views[name]["image_path"]),
                "image_sha256": _sha256_file(Path(selected_views[name]["image_path"])),
                "mask_path": str(mask_paths[name]),
                "mask_sha256": _sha256_file(mask_paths[name]),
            }
            for name in selected_names
        },
    }
    return FrozenSfm(
        views=selected_views,
        intrinsics=intrinsics,
        track_evidence=evidence,
        points_xyz=points,
        object_center=np.asarray(center, dtype=np.float64),
        selected_names=selected_names,
        registered_view_count=len(eligible),
        total_frame_count=total_frames,
        coverage=coverage,
        selected_coverage=selected_coverage,
        provenance=provenance,
    )


def sfm_preflight(snapshot: FrozenSfm, config: ExperimentConfig) -> dict[str, Any]:
    fraction = float(snapshot.registered_view_count / max(snapshot.total_frame_count, 1))
    reasons: list[str] = []
    if snapshot.registered_view_count < config.minimum_registered_views:
        reasons.append("insufficient_registered_views")
    if fraction < config.minimum_registered_fraction:
        reasons.append("insufficient_registered_fraction")
    if snapshot.coverage.get("azimuth_span_degrees", 0.0) < config.minimum_azimuth_span_degrees:
        reasons.append("insufficient_azimuth_coverage")
    if snapshot.coverage.get("occupied_30deg_bins", 0) < config.minimum_occupied_30deg_bins:
        reasons.append("insufficient_direction_bins")
    if (
        snapshot.coverage.get("maximum_opposing_angle_degrees", 0.0)
        < config.minimum_opposing_angle_degrees
    ):
        reasons.append("insufficient_opposing_views")
    if (
        snapshot.selected_coverage.get("azimuth_span_degrees", 0.0)
        < config.minimum_azimuth_span_degrees
    ):
        reasons.append("selected_frames_have_insufficient_azimuth_coverage")
    if (
        snapshot.selected_coverage.get("occupied_30deg_bins", 0)
        < config.minimum_occupied_30deg_bins
    ):
        reasons.append("selected_frames_have_insufficient_direction_bins")
    if (
        snapshot.selected_coverage.get("maximum_opposing_angle_degrees", 0.0)
        < config.minimum_opposing_angle_degrees
    ):
        reasons.append("selected_frames_have_insufficient_opposing_views")
    return {
        "accepted": not reasons,
        "reasons": reasons,
        "registered_view_count": snapshot.registered_view_count,
        "total_frame_count": snapshot.total_frame_count,
        "registered_fraction": fraction,
        "coverage": {
            "all_eligible_registered_views": snapshot.coverage,
            "selected_backend_frames": snapshot.selected_coverage,
        },
        "thresholds": {
            "minimum_registered_views": config.minimum_registered_views,
            "minimum_registered_fraction": config.minimum_registered_fraction,
            "minimum_azimuth_span_degrees": config.minimum_azimuth_span_degrees,
            "minimum_occupied_30deg_bins": config.minimum_occupied_30deg_bins,
            "minimum_opposing_angle_degrees": config.minimum_opposing_angle_degrees,
        },
    }


def _expected_frames(snapshot: FrozenSfm) -> list[dict[str, Any]]:
    focal = float(snapshot.intrinsics[0])
    result: list[dict[str, Any]] = []
    for name in snapshot.selected_names:
        path = Path(snapshot.views[name]["image_path"]).resolve()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode frozen frame: {path}")
        height, width = image.shape[:2]
        result.append(
            {
                "id": name,
                "image_path": str(path),
                "input_sha256": _sha256_file(path),
                "width": width,
                "height": height,
                "focal_length_px": focal,
            }
        )
    return result


def _prediction_cache_records(
    cache_dir: Path,
    expected_frames: Sequence[Mapping[str, Any]],
    *,
    backend: str,
    representation: str,
) -> dict[str, DepthPrediction]:
    manifest_path = cache_dir / "input_manifest.json"
    provenance_path = cache_dir / "provenance.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {backend} prediction cache: {cache_dir}") from error
    manifest_frames = manifest.get("frames")
    provenance_frames = provenance.get("frames")
    if manifest.get("backend") != backend or provenance.get("backend") != backend:
        raise ValueError(f"{backend} cache backend identity mismatch")
    if manifest.get("representation", representation) != representation or provenance.get(
        "representation", representation
    ) != representation:
        raise ValueError(f"{backend} cache representation mismatch")
    for identity_key in ("model_commit", "checkpoint_sha256", "model_sha256"):
        if (
            identity_key in manifest
            and identity_key in provenance
            and manifest[identity_key] != provenance[identity_key]
        ):
            raise ValueError(f"{backend} cache {identity_key} mismatch")
    if not isinstance(manifest_frames, list) or not isinstance(provenance_frames, list):
        raise ValueError(f"incomplete {backend} cache manifests")
    if [item.get("id") for item in manifest_frames] != [item["id"] for item in expected_frames]:
        raise ValueError(f"{backend} cache frame set/order differs from frozen experiment")
    if [item.get("id") for item in provenance_frames] != [item["id"] for item in expected_frames]:
        raise ValueError(f"{backend} cache provenance frame order differs")
    records: dict[str, DepthPrediction] = {}
    for expected, source, record in zip(
        expected_frames, manifest_frames, provenance_frames, strict=True
    ):
        for key in ("input_sha256", "width", "height"):
            if source.get(key) != expected[key]:
                raise ValueError(f"{backend} cache input {key} mismatch for {expected['id']}")
        if record.get("input_sha256") != expected["input_sha256"]:
            raise ValueError(f"{backend} cache provenance input hash mismatch")
        source_focal = float(source.get("focal_length_px", 0.0))
        if not math.isclose(
            source_focal, float(expected["focal_length_px"]), rel_tol=1e-9, abs_tol=1e-6
        ):
            raise ValueError(f"{backend} cache focal length mismatch for {expected['id']}")
        relative = record.get("npz_path")
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"unsafe {backend} cache output path")
        path = cache_dir / relative
        output_hash = record.get("output_sha256")
        if (
            not path.is_file()
            or not isinstance(output_hash, str)
            or not _SHA256_RE.fullmatch(output_hash)
            or _sha256_file(path) != output_hash
        ):
            raise ValueError(f"{backend} cache output hash mismatch for {expected['id']}")
        with np.load(path, allow_pickle=False) as archive:
            value_key = "depth_m" if representation == "metric_depth_m" else "values"
            if value_key not in archive.files:
                raise ValueError(f"{backend} cache is missing {value_key}")
            values = np.asarray(archive[value_key], dtype=np.float32)
            if representation == "metric_depth_m":
                focal = float(np.asarray(archive["focal_length_px"]).reshape(()))
            else:
                focal = None
        if values.shape != (expected["height"], expected["width"]):
            raise ValueError(f"{backend} cache dimensions mismatch for {expected['id']}")
        if focal is not None and not math.isclose(
            focal, float(expected["focal_length_px"]), rel_tol=1e-5
        ):
            raise ValueError(f"{backend} cached output focal length mismatch")
        records[str(expected["id"])] = DepthPrediction(
            values=values,
            representation=representation,
            source_id=str(expected["id"]),
            focal_length_px=focal,
            provenance={
                "cache_dir": str(cache_dir),
                "cache_hit": True,
                "input_manifest": {
                    key: value for key, value in manifest.items() if key != "frames"
                },
                "batch_provenance": {
                    key: value for key, value in provenance.items() if key != "frames"
                },
                "frame": record,
            },
        )
    return records


def _write_depth_anything_cache(
    cache_dir: Path,
    expected_frames: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, DepthPrediction],
) -> None:
    if cache_dir.exists():
        raise FileExistsError(f"cache already exists: {cache_dir}")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.", dir=cache_dir.parent))
    try:
        predictions_dir = staging / "predictions"
        predictions_dir.mkdir()
        provenance_records: list[dict[str, Any]] = []
        for expected in expected_frames:
            name = str(expected["id"])
            prediction = predictions[name]
            path = predictions_dir / f"{name}.npz"
            _atomic_npz(path, {"values": prediction.values})
            provenance_records.append(
                {
                    "id": name,
                    "input_sha256": expected["input_sha256"],
                    "npz_path": f"predictions/{name}.npz",
                    "output_sha256": _sha256_file(path),
                    "prediction": _strict_json(dict(prediction.provenance)),
                }
            )
        first = predictions[str(expected_frames[0]["id"])]
        _atomic_json(
            staging / "input_manifest.json",
            {
                "schema_version": 1,
                "backend": "depth_anything_v2_onnx",
                "representation": "relative_disparity",
                "model_sha256": first.provenance.get("model_sha256"),
                "frames": list(expected_frames),
            },
        )
        _atomic_json(
            staging / "provenance.json",
            {
                "schema_version": 1,
                "backend": "depth_anything_v2_onnx",
                "representation": "relative_disparity",
                "model_sha256": first.provenance.get("model_sha256"),
                "device": first.provenance.get("device"),
                "precision": first.provenance.get("precision"),
                "frames": provenance_records,
            },
        )
        os.replace(staging, cache_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def obtain_depth_anything(
    config: ExperimentConfig,
    expected_frames: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, DepthPrediction], dict[str, Any]]:
    cache_dir = config.cache_dir.expanduser().resolve() / "depth_anything"
    if cache_dir.is_dir():
        predictions = _prediction_cache_records(
            cache_dir,
            expected_frames,
            backend="depth_anything_v2_onnx",
            representation="relative_disparity",
        )
        return predictions, {"cache_hit": True, "cache_dir": str(cache_dir)}
    backend = DepthAnythingV2Adapter(config.depth_anything_model.expanduser().resolve())
    predictions: dict[str, DepthPrediction] = {}
    for frame in expected_frames:
        image = cv2.imread(str(frame["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode frame: {frame['image_path']}")
        predictions[str(frame["id"])] = backend.predict(image, source_id=str(frame["id"]))
    _write_depth_anything_cache(cache_dir, expected_frames, predictions)
    loaded = _prediction_cache_records(
        cache_dir,
        expected_frames,
        backend="depth_anything_v2_onnx",
        representation="relative_disparity",
    )
    return loaded, {"cache_hit": False, "cache_dir": str(cache_dir)}


def obtain_depth_pro(
    config: ExperimentConfig,
    expected_frames: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, DepthPrediction] | None, dict[str, Any]]:
    if config.disable_depth_pro:
        return None, {"available": False, "reason": "disabled"}
    cache_dir = config.cache_dir.expanduser().resolve() / "depth_pro"
    if cache_dir.is_dir():
        predictions = _prediction_cache_records(
            cache_dir,
            expected_frames,
            backend="apple_depth_pro",
            representation="metric_depth_m",
        )
        return predictions, {"available": True, "cache_hit": True, "cache_dir": str(cache_dir)}
    supplied = (
        config.depth_pro_python,
        config.depth_pro_checkpoint,
        config.depth_pro_commit,
    )
    if all(value is None for value in supplied):
        return None, {
            "available": False,
            "reason": "no_cache_or_explicit_depth_pro_configuration",
        }
    if any(value is None for value in supplied):
        raise ValueError(
            "Depth Pro generation requires --depth-pro-python, --depth-pro-checkpoint, "
            "and --depth-pro-commit together"
        )
    backend = DepthProSubprocessBackend(
        python_executable=Path(config.depth_pro_python),
        checkpoint_path=Path(config.depth_pro_checkpoint),
        model_commit=str(config.depth_pro_commit),
        device=config.depth_pro_device,
        allow_non_mps=config.allow_non_mps,
    )
    frames = [
        DepthFrameInput(
            str(frame["id"]),
            Path(str(frame["image_path"])),
            float(frame["focal_length_px"]),
        )
        for frame in expected_frames
    ]
    backend.predict_batch_by_id(frames, output_dir=cache_dir)
    loaded = _prediction_cache_records(
        cache_dir,
        expected_frames,
        backend="apple_depth_pro",
        representation="metric_depth_m",
    )
    return loaded, {"available": True, "cache_hit": False, "cache_dir": str(cache_dir)}


def _freeze_snapshot(staging: Path, snapshot: FrozenSfm) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "points_xyz": np.asarray(snapshot.points_xyz, dtype=np.float64),
        "object_center": np.asarray(snapshot.object_center, dtype=np.float64),
        "intrinsics": np.asarray(snapshot.intrinsics, dtype=np.float64),
    }
    index: dict[str, Any] = {}
    hash_items: list[tuple[str, np.ndarray]] = sorted(arrays.items())
    evidence_views = snapshot.track_evidence.get("views", {})
    for view_index, name in enumerate(snapshot.selected_names):
        prefix = f"view_{view_index:04d}"
        view = snapshot.views[name]
        evidence = evidence_views.get(name, {})
        record = {
            "name": name,
            "image_path": str(view["image_path"]),
            "array_prefix": prefix,
        }
        index[prefix] = record
        values = {
            f"{prefix}_rotation": np.asarray(view["rotation"], dtype=np.float64),
            f"{prefix}_translation": np.asarray(view["translation"], dtype=np.float64),
            f"{prefix}_mask_tight": np.asarray(view["mask_tight"], dtype=np.uint8),
            f"{prefix}_point3d_ids": np.asarray(evidence.get("point3d_ids", []), dtype=np.int64),
            f"{prefix}_xy": np.asarray(evidence.get("xy", []), dtype=np.float64).reshape(-1, 2),
            f"{prefix}_z_camera": np.asarray(evidence.get("z_camera", []), dtype=np.float64),
            f"{prefix}_track_lengths": np.asarray(evidence.get("track_lengths", []), dtype=np.int32),
            f"{prefix}_reprojection_errors_px": np.asarray(
                evidence.get("reprojection_errors_px", []), dtype=np.float64
            ),
        }
        arrays.update(values)
        hash_items.extend(sorted(values.items()))
    arrays_path = staging / "frozen_evidence.npz"
    _atomic_npz(arrays_path, arrays)
    content_digest = _hash_arrays(hash_items)
    manifest = {
        **snapshot.provenance,
        "registered_view_count": snapshot.registered_view_count,
        "total_frame_count": snapshot.total_frame_count,
        "intrinsics": list(snapshot.intrinsics),
        "object_center": snapshot.object_center.tolist(),
        "coverage": snapshot.coverage,
        "selected_coverage": snapshot.selected_coverage,
        "views": index,
        "array_content_sha256": content_digest,
        "npz_sha256": _sha256_file(arrays_path),
        "npz_path": "frozen_evidence.npz",
    }
    _atomic_json(staging / "frozen_evidence.json", _strict_json(manifest))
    return manifest


def _persist_evaluation(
    staging: Path, evaluation: DepthBackendEvaluation
) -> dict[str, Any]:
    directory = staging / "aligned" / evaluation.name
    directory.mkdir(parents=True, exist_ok=False)
    records: dict[str, Any] = {}
    for name in sorted(evaluation.aligned_depths):
        path = directory / f"{name}.npz"
        depth = np.asarray(evaluation.aligned_depths[name], dtype=np.float32)
        confidence = np.asarray(
            evaluation.aligned_confidences.get(
                name, np.where(depth > 0.0, 1.0, 0.0).astype(np.float32)
            ),
            dtype=np.float32,
        )
        _atomic_npz(path, {"depth_z": depth, "confidence": confidence})
        records[name] = {
            "path": str(path.relative_to(staging)),
            "sha256": _sha256_file(path),
            "valid_fraction": float(np.mean(depth > 0.0)),
        }
    report_path = directory / "evaluation.json"
    _atomic_json(report_path, evaluation.report)
    return {
        "report_path": str(report_path.relative_to(staging)),
        "report_sha256": _sha256_file(report_path),
        "aligned_depths": records,
    }


def _persist_geometry_inputs(
    staging: Path,
    snapshot: FrozenSfm,
    selection: DepthSelectionResult,
) -> dict[str, Any]:
    directory = staging / "geometry_inputs"
    directory.mkdir()
    records: dict[str, Any] = {}
    for name in sorted(selection.aligned_depths):
        depth = np.asarray(selection.aligned_depths[name], dtype=np.float32)
        confidence = np.asarray(
            selection.aligned_confidences.get(
                name, np.where(depth > 0.0, 1.0, 0.0).astype(np.float32)
            ),
            dtype=np.float32,
        )
        mask = np.asarray(snapshot.views[name]["mask_tight"], dtype=np.uint8)
        path = directory / f"{name}.npz"
        _atomic_npz(
            path,
            {"depth_z": depth, "confidence": confidence, "mask_tight": mask},
        )
        records[name] = {
            "path": str(path.relative_to(staging)),
            "sha256": _sha256_file(path),
            "image_path": str(snapshot.views[name]["image_path"]),
            "rotation": np.asarray(snapshot.views[name]["rotation"]).tolist(),
            "translation": np.asarray(snapshot.views[name]["translation"]).tolist(),
        }
    manifest = {
        "schema": "local3d.precomputed_depth_geometry_inputs.v1",
        "classification": "aligned monocular predictions; not measured depth",
        "intrinsics": list(snapshot.intrinsics),
        "selection": selection.report,
        "views": records,
    }
    path = directory / "manifest.json"
    _atomic_json(path, _strict_json(manifest))
    return {"manifest": str(path.relative_to(staging)), "sha256": _sha256_file(path)}


def run_experiment(
    config: ExperimentConfig,
    *,
    snapshot: FrozenSfm | None = None,
    depth_anything_provider: Callable[
        [ExperimentConfig, Sequence[Mapping[str, Any]]],
        tuple[dict[str, DepthPrediction], dict[str, Any]],
    ] = obtain_depth_anything,
    depth_pro_provider: Callable[
        [ExperimentConfig, Sequence[Mapping[str, Any]]],
        tuple[dict[str, DepthPrediction] | None, dict[str, Any]],
    ] = obtain_depth_pro,
    evaluator: Callable[..., DepthBackendEvaluation] = evaluate_depth_predictions,
    selector: Callable[..., DepthSelectionResult] = select_depth_backend,
) -> dict[str, Any]:
    """Run, persist, and atomically publish one reproducible comparison."""

    if config.maximum_frames < 4:
        raise ValueError("maximum_frames must be at least 4")
    target = config.output_dir.expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"experiment output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    frozen = snapshot or load_frozen_sfm(config)
    preflight = sfm_preflight(frozen, config)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.depth-experiment-", dir=target.parent))
    try:
        freeze_manifest = _freeze_snapshot(staging, frozen)
        report: dict[str, Any] = {
            "schema": "local3d.depth_backend_experiment.v1",
            "classification": (
                "comparison of monocular predictions against frozen SfM evidence; "
                "not a claim of measured depth"
            ),
            "frozen_evidence": {
                "manifest": "frozen_evidence.json",
                "arrays": "frozen_evidence.npz",
                "array_content_sha256": freeze_manifest["array_content_sha256"],
            },
            "selected_frames": list(frozen.selected_names),
            "sfm_preflight": preflight,
            "diagnostic_only": bool(
                not preflight["accepted"] and config.diagnostic_on_inadequate_sfm
            ),
            "backends": {},
            "raw_depth_selection": None,
            "final_selection": None,
            "geometry_inputs": None,
            "output_dir": str(target),
        }
        if not preflight["accepted"] and not config.diagnostic_on_inadequate_sfm:
            report["final_selection"] = {
                "decision": "reject",
                "reason": "sfm_preflight_failed_before_depth_inference",
                "selected_backend": None,
            }
        else:
            expected = _expected_frames(frozen)
            depth_anything, da_cache = depth_anything_provider(config, expected)
            if tuple(depth_anything) != frozen.selected_names:
                raise RuntimeError("Depth Anything did not return the exact frozen frame order")
            prediction_sets: dict[str, dict[str, DepthPrediction]] = {
                "depth_anything_v2": depth_anything
            }
            report["backends"]["depth_anything_v2"] = {"cache": da_cache}
            depth_pro, dp_cache = depth_pro_provider(config, expected)
            report["backends"]["apple_depth_pro"] = {"cache": dp_cache}
            if depth_pro is not None:
                if tuple(depth_pro) != frozen.selected_names:
                    raise RuntimeError("Depth Pro did not return the exact frozen frame order")
                prediction_sets["apple_depth_pro"] = depth_pro

            evaluations: dict[str, DepthBackendEvaluation] = {}
            for backend_name, predictions in prediction_sets.items():
                evaluation = evaluator(
                    backend_name,
                    predictions,
                    frozen.views,
                    frozen.intrinsics,
                    frozen.track_evidence,
                    object_center=frozen.object_center,
                )
                evaluations[backend_name] = evaluation
                report["backends"][backend_name]["evaluation"] = _persist_evaluation(
                    staging, evaluation
                )
                report["backends"][backend_name]["accepted"] = bool(
                    evaluation.report.get("accepted")
                )

            selection = selector(evaluations)
            report["raw_depth_selection"] = selection.report
            if preflight["accepted"]:
                report["final_selection"] = selection.report
                if (
                    config.emit_geometry_inputs
                    and selection.report.get("decision") in {"selected", "consensus"}
                ):
                    report["geometry_inputs"] = _persist_geometry_inputs(
                        staging, frozen, selection
                    )
            else:
                report["final_selection"] = {
                    "schema": "local3d.depth_backend_selection.v1",
                    "decision": "reject",
                    "reason": "sfm_preflight_failed_diagnostic_results_not_promotable",
                    "selected_backend": None,
                    "diagnostic_raw_decision": selection.report.get("decision"),
                }

        report_path = staging / "report.json"
        _atomic_json(report_path, _strict_json(report))
        os.replace(staging, target)
        return report
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sfm-model", required=True, type=Path)
    parser.add_argument("--frames-dir", required=True, type=Path)
    parser.add_argument("--masks-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--depth-anything-model", required=True, type=Path)
    parser.add_argument("--maximum-frames", type=int, default=12)
    parser.add_argument("--minimum-registered-views", type=int, default=8)
    parser.add_argument("--minimum-registered-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-azimuth-span", type=float, default=120.0)
    parser.add_argument("--minimum-direction-bins", type=int, default=5)
    parser.add_argument("--minimum-opposing-angle", type=float, default=90.0)
    parser.add_argument("--diagnostic-on-inadequate-sfm", action="store_true")
    parser.add_argument("--emit-geometry-inputs", action="store_true")
    parser.add_argument("--disable-depth-pro", action="store_true")
    parser.add_argument("--depth-pro-python", type=Path)
    parser.add_argument("--depth-pro-checkpoint", type=Path)
    parser.add_argument("--depth-pro-commit")
    parser.add_argument("--depth-pro-device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--allow-non-mps", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.maximum_frames < 4:
        raise SystemExit("--maximum-frames must be at least 4")
    config = ExperimentConfig(
        sfm_model_dir=args.sfm_model,
        frames_dir=args.frames_dir,
        masks_dir=args.masks_dir,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        depth_anything_model=args.depth_anything_model,
        maximum_frames=args.maximum_frames,
        minimum_registered_views=args.minimum_registered_views,
        minimum_registered_fraction=args.minimum_registered_fraction,
        minimum_azimuth_span_degrees=args.minimum_azimuth_span,
        minimum_occupied_30deg_bins=args.minimum_direction_bins,
        minimum_opposing_angle_degrees=args.minimum_opposing_angle,
        diagnostic_on_inadequate_sfm=args.diagnostic_on_inadequate_sfm,
        emit_geometry_inputs=args.emit_geometry_inputs,
        depth_pro_python=args.depth_pro_python,
        depth_pro_checkpoint=args.depth_pro_checkpoint,
        depth_pro_commit=args.depth_pro_commit,
        depth_pro_device=args.depth_pro_device,
        allow_non_mps=args.allow_non_mps,
        disable_depth_pro=args.disable_depth_pro,
    )
    report = run_experiment(config)
    final = report["final_selection"]
    print(
        json.dumps(
            {
                "decision": final.get("decision"),
                "reason": final.get("reason"),
                "selected_backend": final.get("selected_backend"),
                "output_dir": report["output_dir"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
