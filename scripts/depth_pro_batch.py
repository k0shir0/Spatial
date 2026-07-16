#!/usr/bin/env python3
"""Offline batch runner for Apple's external ``depth_pro`` package.

This script is intentionally dependency-isolated from the main Spatial
environment.  It never downloads code or weights: the caller supplies a local
checkpoint, an exact expected SHA-256, and the source commit of the installed
Apple repository.  MPS and float16 are the defaults/normal production mode.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageOps

_FRAME_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _force_offline() -> None:
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        }
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, *, depth_m: np.ndarray, focal_length_px: float) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite output: {path}")
    with temporary.open("xb") as handle:
        np.savez_compressed(
            handle,
            depth_m=np.asarray(depth_m, dtype=np.float32),
            focal_length_px=np.asarray(focal_length_px, dtype=np.float32),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_manifest(
    path: Path, *, model_commit: str, checkpoint_sha256: str
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid input manifest: {path}") from error
    if payload.get("schema_version") != 1 or payload.get("backend") != "apple_depth_pro":
        raise ValueError("unsupported input manifest schema/backend")
    if payload.get("model_commit") != model_commit:
        raise ValueError("manifest source commit does not match command line")
    if payload.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("manifest checkpoint hash does not match command line")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("manifest must contain at least one frame")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for raw in frames:
        if not isinstance(raw, dict):
            raise ValueError("manifest frame must be an object")
        frame_id = raw.get("id")
        if not isinstance(frame_id, str) or not _FRAME_ID_RE.fullmatch(frame_id):
            raise ValueError(f"unsafe frame id: {frame_id!r}")
        if frame_id in seen:
            raise ValueError(f"duplicate frame id: {frame_id}")
        seen.add(frame_id)
        image_path = Path(str(raw.get("image_path", ""))).expanduser()
        if not image_path.is_absolute() or not image_path.is_file():
            raise ValueError(f"frame path must be an existing absolute file: {image_path}")
        input_hash = raw.get("input_sha256")
        if not isinstance(input_hash, str) or not _SHA256_RE.fullmatch(input_hash):
            raise ValueError(f"invalid input hash for {frame_id}")
        focal = float(raw.get("focal_length_px", 0.0))
        width = int(raw.get("width", 0))
        height = int(raw.get("height", 0))
        if not math.isfinite(focal) or focal <= 0.0 or width <= 0 or height <= 0:
            raise ValueError(f"invalid dimensions/focal length for {frame_id}")
        validated.append(
            {
                "id": frame_id,
                "image_path": image_path.resolve(),
                "input_sha256": input_hash,
                "width": width,
                "height": height,
                "focal_length_px": focal,
            }
        )
    return validated


def _package_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    for distribution in ("depth-pro", "depth_pro"):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return "not-declared"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--model-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument(
        "--allow-non-mps",
        action="store_true",
        help="Explicitly permit CPU/CUDA for diagnostics; production defaults to MPS.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _force_offline()

    checkpoint = args.checkpoint.expanduser()
    if not checkpoint.is_absolute() or not checkpoint.is_file():
        raise SystemExit("checkpoint must be an existing absolute file")
    checkpoint = checkpoint.resolve()
    expected_hash = str(args.expected_checkpoint_sha256).lower()
    if not _SHA256_RE.fullmatch(expected_hash):
        raise SystemExit("expected checkpoint SHA-256 must be 64 lowercase hex characters")
    checkpoint_hash = _sha256_file(checkpoint)
    if checkpoint_hash != expected_hash:
        raise SystemExit("checkpoint SHA-256 mismatch")
    model_commit = str(args.model_commit).lower()
    if not _COMMIT_RE.fullmatch(model_commit):
        raise SystemExit("model commit must be a full 40-character Git SHA")
    if args.device != "mps" and not args.allow_non_mps:
        raise SystemExit("non-MPS execution requires --allow-non-mps")

    manifest = args.manifest.expanduser().resolve()
    frames = _load_manifest(
        manifest, model_commit=model_commit, checkpoint_sha256=checkpoint_hash
    )
    output_dir = args.output_dir.expanduser().resolve()
    provenance_path = args.provenance.expanduser().resolve()
    if output_dir.exists():
        raise SystemExit(f"output directory already exists: {output_dir}")
    if provenance_path.exists():
        raise SystemExit(f"provenance path already exists: {provenance_path}")
    output_dir.mkdir(parents=False, exist_ok=False)

    # Heavy/version-specific imports happen only after every path/hash check and
    # after offline mode is forced.  The supplied checkpoint is wired directly
    # into Apple's config; ``use_pretrained=False`` in Depth Pro's factory means
    # no backbone fetch occurs.
    try:
        import torch
        import depth_pro
        from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
    except ImportError as error:
        raise SystemExit(
            "the explicit Python environment must contain Apple's depth_pro package and torch"
        ) from error

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; no prediction was run")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; no prediction was run")
    device = torch.device(args.device)
    precision = torch.float16
    config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint))
    model, transform = depth_pro.create_model_and_transforms(
        config=config, device=device, precision=precision
    )
    model.eval()

    started_at = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    for frame in frames:
        # Decode the exact bytes that were hashed; this avoids EXIF focal-length
        # inference and guarantees the supplied calibration is the one used.
        image_bytes = frame["image_path"].read_bytes()
        actual_input_hash = _sha256_bytes(image_bytes)
        if actual_input_hash != frame["input_sha256"]:
            raise RuntimeError(f"input changed before inference: {frame['id']}")
        with Image.open(BytesIO(image_bytes)) as opened:
            rgb_image = ImageOps.exif_transpose(opened).convert("RGB")
            rgb = np.asarray(rgb_image, dtype=np.uint8)
        height, width = rgb.shape[:2]
        if (width, height) != (frame["width"], frame["height"]):
            raise RuntimeError(
                f"decoded dimensions do not match manifest for {frame['id']}"
            )
        image_tensor = transform(rgb)
        supplied_focal = torch.tensor(
            frame["focal_length_px"], device=device, dtype=torch.float32
        )
        with torch.inference_mode():
            prediction = model.infer(image_tensor, f_px=supplied_focal)
        depth = prediction["depth"].detach().to(device="cpu", dtype=torch.float32).numpy()
        output_focal = float(
            prediction["focallength_px"].detach().to(device="cpu", dtype=torch.float32)
        )
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape != (height, width):
            raise RuntimeError(f"Depth Pro returned the wrong shape for {frame['id']}")
        if not np.isfinite(depth).all() or not np.all(depth > 0.0):
            raise RuntimeError(f"Depth Pro returned invalid metric depth for {frame['id']}")
        if not math.isclose(output_focal, frame["focal_length_px"], rel_tol=1e-5):
            raise RuntimeError(f"Depth Pro changed supplied focal length for {frame['id']}")

        relative_path = f"predictions/{frame['id']}.npz"
        prediction_path = output_dir / f"{frame['id']}.npz"
        _atomic_npz(
            prediction_path,
            depth_m=depth,
            focal_length_px=frame["focal_length_px"],
        )
        records.append(
            {
                "id": frame["id"],
                "input_path": str(frame["image_path"]),
                "input_sha256": actual_input_hash,
                "width": width,
                "height": height,
                "supplied_focal_length_px": frame["focal_length_px"],
                "returned_focal_length_px": output_focal,
                "focal_length_source": "supplied",
                "npz_path": relative_path,
                "output_sha256": _sha256_file(prediction_path),
                "minimum_depth_m": float(depth.min()),
                "maximum_depth_m": float(depth.max()),
                "valid_fraction": 1.0,
            }
        )

    provenance = {
        "schema_version": 1,
        "backend": "apple_depth_pro",
        "implementation": "apple/ml-depth-pro",
        "model_commit": model_commit,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "depth_pro_package_version": _package_version(),
        "torch_version": str(torch.__version__),
        "device": args.device,
        "precision": "float16",
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "frames": records,
    }
    _atomic_json(provenance_path, provenance)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
