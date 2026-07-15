"""Masked sequential structure-from-motion for hand-rotated-object video (CPU).

The public entry point :func:`run_masked_sfm` replaces the old exhaustive
matcher: it runs COLMAP (via pycolmap 4.1) with a single shared camera prior,
object-only masks, sequential + loop-closure matching, and the incremental
mapper, then returns the best model as ``recon_common`` view dicts so the
texture/QA stages can reload it. Everything is single-threaded-friendly,
seed-pinned, and CPU-only. No vocab tree is used (no data file ships); loop
closure is fed an explicit, deterministic pair list instead.

INTROSPECTION REPORT (pycolmap 4.1.0, confirmed in the target venv)
--------------------------------------------------------------------
- ``ImageReaderOptions``: ``camera_model`` (str), ``camera_params`` (str),
  ``mask_path`` (str). CameraMode.SINGLE passed to ``extract_features(...,
  camera_mode=...)``.
- ``extract_features(database_path, image_path, image_names=[], camera_mode,
  reader_options, extraction_options, device)``.
- ``FeatureExtractionOptions``: ``num_threads``, ``use_gpu``, ``sift``
  (SiftExtractionOptions).
- ``SiftExtractionOptions``: ``max_num_features``, ``peak_threshold``,
  ``edge_threshold``, ``first_octave``, ``estimate_affine_shape``,
  ``domain_size_pooling``.
- ``match_sequential(database_path, matching_options, pairing_options,
  verification_options, device)``.
- ``SequentialPairingOptions``: ``overlap``, ``quadratic_overlap``,
  ``loop_detection``.
- ``FeatureMatchingOptions``: ``num_threads``, ``use_gpu``,
  ``guided_matching`` (confirmed here, NOT on SiftMatchingOptions).
- ``match_image_pairs(database_path, matching_options, pairing_options,
  verification_options, device)`` with ``ImportedPairingOptions.match_list_path``
  pointing at a whitespace-separated ``name1 name2`` file — no name->id mapping
  needed.
- ``incremental_mapping(database_path, image_path, output_path, options)`` ->
  ``dict[int, Reconstruction]``.
- ``IncrementalPipelineOptions``: ``num_threads``, ``random_seed``,
  ``min_num_matches``, ``multiple_models``, ``max_num_models``,
  ``ba_refine_principal_point``, and nested ``mapper``
  (IncrementalMapperOptions) with ``init_min_num_inliers``,
  ``init_min_tri_angle``, ``abs_pose_min_num_inliers``,
  ``abs_pose_min_inlier_ratio``, ``filter_max_reproj_error``.
- ``Image.cam_from_world()`` -> ``Rigid3d``; ``.rotation.matrix()`` is the
  (3,3) world->camera rotation, ``.translation`` the (3,) translation.
- ``Reconstruction``: ``reg_image_ids()``, ``images``, ``points3D``,
  ``cameras``, ``compute_mean_reprojection_error()``, ``num_reg_images``,
  ``write(dir)`` (binary format).
- ``Point3D.track.length()`` gives the track length used for filtering.

Determinism: fixed frame ordering (sorted by name), ``pycolmap.set_random_seed``
plus ``random_seed`` in the mapper options, no wall-clock in any output value
(the extraction timing probe only influences a discrete first_octave choice,
which is recorded, not the geometry). Cross-machine bit-identity is not
promised; COLMAP's RANSAC is seed-stable on one platform.

Honest limits: registration success depends on real texture on the object;
featureless / specular objects still fail. Loop closure here is geometric
(explicit pairs), not appearance-based, so it helps a turntable-style capture
but will not rescue arbitrary re-visits.
"""

from __future__ import annotations

import math
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from local3d.recon_common import Intrinsics, make_view

_FRAME_SUFFIXES = (".png", ".jpg", ".jpeg")


def sequential_loop_pairs(
    names: list[str], *, overlap: int, loop_stride: int
) -> list[tuple[str, str]]:
    """Deterministic extra loop-closure pairs.

    Considers every ``loop_stride``-th frame in temporal (sorted-name) order and
    emits all pairs ``(i, j)`` among them whose temporal index distance
    ``|i - j|`` exceeds ``overlap`` (those within ``overlap`` are already
    covered by sequential matching). Returns ``(name_i, name_j)`` tuples sorted
    by ``(i, j)`` with ``i < j``, so identical inputs yield identical output.
    """

    if loop_stride < 1:
        raise ValueError("loop_stride must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    ordered = sorted(names)
    anchors = list(range(0, len(ordered), loop_stride))
    pairs: list[tuple[str, str]] = []
    for a in range(len(anchors)):
        for b in range(a + 1, len(anchors)):
            i, j = anchors[a], anchors[b]
            if j - i > overlap:
                pairs.append((ordered[i], ordered[j]))
    return pairs


def _frame_paths(frames_dir: Path) -> list[Path]:
    """Frame image paths in deterministic (sorted-name) temporal order."""

    return sorted(
        p
        for p in Path(frames_dir).iterdir()
        if p.suffix.lower() in _FRAME_SUFFIXES
    )


def focal_from_fov(width: int, fov_deg: float) -> float:
    """Pinhole focal length in pixels for a horizontal field of view."""

    return (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def stage_masks(
    frame_paths: list[Path], eroded_masks_dir: Path, colmap_masks_dir: Path
) -> dict[str, Any]:
    """Stage eroded object masks into COLMAP's ``<image_name>.png`` layout.

    The eroded mask directory holds ``<stem>_mask.png`` (255 = object). COLMAP
    wants, for image ``frame.jpg``, a mask named ``frame.jpg.png`` (nonzero =
    keep) inside the directory handed to ``ImageReaderOptions.mask_path``.
    Returns a small report listing which frames got a mask and which were
    missing one (missing masks mean COLMAP keeps the whole frame).
    """

    colmap_masks_dir = Path(colmap_masks_dir)
    colmap_masks_dir.mkdir(parents=True, exist_ok=True)
    eroded_masks_dir = Path(eroded_masks_dir)
    staged: list[str] = []
    missing: list[str] = []
    for frame in frame_paths:
        source = eroded_masks_dir / f"{frame.stem}_mask.png"
        if source.is_file():
            shutil.copyfile(source, colmap_masks_dir / f"{frame.name}.png")
            staged.append(frame.name)
        else:
            missing.append(frame.name)
    return {
        "mask_dir": str(colmap_masks_dir),
        "staged": staged,
        "missing_masks": missing,
        "staged_count": len(staged),
    }


def _model_sanity(rec: Any) -> dict[str, Any]:
    """Geometric sanity of a sub-model: the object must sit inside the orbit.

    A healthy hand-rotation model has camera centers orbiting a compact point
    cloud.  False loop closures (repetitive label text) instead produce point
    sheets larger than the orbit radius; those models must never win selection.
    """

    points = _points_xyz(rec, min_track_length=3)
    report: dict[str, Any] = {"sane": False, "reason": None}
    if len(points) < 50:
        report["reason"] = "too few tracked points"
        return report
    centers = np.array([
        np.asarray(rec.image(image_id).projection_center())
        for image_id in rec.reg_image_ids()
    ])
    centroid = points.mean(axis=0)
    cam_dist = float(np.median(np.linalg.norm(centers - centroid, axis=1)))
    point_radius = np.linalg.norm(points - centroid, axis=1)
    inside = float(np.mean(point_radius < 0.6 * cam_dist))
    rms_radius = float(np.sqrt(np.mean(point_radius**2)))
    report.update(
        {
            "median_camera_distance": cam_dist,
            "point_rms_radius": rms_radius,
            "points_inside_orbit_fraction": inside,
        }
    )
    if inside < 0.85:
        report["reason"] = "points spill outside the camera orbit"
        return report
    if cam_dist < 1.5 * rms_radius:
        report["reason"] = "cameras are inside the point cloud"
        return report
    report["sane"] = True
    return report


def _rec_mean_error(rec: Any) -> float:
    try:
        value = float(rec.compute_mean_reprojection_error())
    except Exception:
        return float("inf")
    if not math.isfinite(value):
        return float("inf")
    return value


def _build_views(rec: Any, frames_dir: Path) -> list[dict[str, Any]]:
    frames_dir = Path(frames_dir)
    views: list[tuple[str, dict[str, Any]]] = []
    for image_id in rec.reg_image_ids():
        image = rec.image(image_id)
        pose = image.cam_from_world()
        rotation = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        translation = np.asarray(pose.translation, dtype=np.float64).reshape(3)
        views.append(
            (
                image.name,
                make_view(
                    name=image.name,
                    image_path=frames_dir / image.name,
                    rotation=rotation,
                    translation=translation,
                    extras={"image_id": int(image_id)},
                ),
            )
        )
    views.sort(key=lambda item: item[0])
    return [view for _, view in views]


def _intrinsics(rec: Any) -> Intrinsics:
    camera = next(iter(rec.cameras.values()))
    params = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    focal, cx, cy, k1 = (float(params[0]), float(params[1]), float(params[2]), float(params[3]))
    return (focal, cx, cy, k1)


def _points_xyz(rec: Any, *, min_track_length: int = 3) -> np.ndarray:
    points: list[np.ndarray] = []
    for point_id in sorted(rec.points3D.keys()):
        point = rec.points3D[point_id]
        if point.track.length() >= min_track_length:
            points.append(np.asarray(point.xyz, dtype=np.float64).reshape(3))
    if not points:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray(points, dtype=np.float64)


def _write_pairs_file(pairs: list[tuple[str, str]], path: Path) -> None:
    path.write_text(
        "".join(f"{a} {b}\n" for a, b in pairs), encoding="utf-8"
    )


def _probe_first_octave(
    pc: Any,
    frames_dir: Path,
    frame_names: list[str],
    reader_options: Any,
    extraction_options: Any,
    work_dir: Path,
    *,
    max_seconds_per_frame: float = 4.0,
) -> tuple[int, dict[str, Any]]:
    """Time first_octave=-1 extraction on up to 3 frames; drop to 0 if slow.

    ``first_octave=-1`` doubles image resolution (more features, ~2-4x cost).
    Returns the chosen ``first_octave`` and a report entry. The measured time
    only selects a discrete option; it never enters any geometry output.
    """

    probe_names = frame_names[: min(3, len(frame_names))]
    if not probe_names:
        return 0, {"skipped": "no frames"}
    probe_db = Path(work_dir) / "probe.db"
    if probe_db.exists():
        probe_db.unlink()
    probe_extraction = pc.FeatureExtractionOptions()
    probe_extraction.mergedict(extraction_options.todict())
    probe_extraction.sift.first_octave = -1
    start = time.perf_counter()
    pc.extract_features(
        database_path=str(probe_db),
        image_path=str(frames_dir),
        image_names=list(probe_names),
        camera_mode=pc.CameraMode.SINGLE,
        reader_options=reader_options,
        extraction_options=probe_extraction,
    )
    elapsed = time.perf_counter() - start
    per_frame = elapsed / len(probe_names)
    if probe_db.exists():
        probe_db.unlink()
    chosen = -1 if per_frame <= max_seconds_per_frame else 0
    return chosen, {
        "probe_frames": len(probe_names),
        "seconds_per_frame_bucket": "<=4s" if per_frame <= max_seconds_per_frame else ">4s",
        "chosen_first_octave": chosen,
    }


def run_masked_sfm(
    frames_dir: Path,
    eroded_masks_dir: Path,
    work_dir: Path,
    *,
    seed: int = 0,
    fov_deg: float = 65.0,
    overlap: int = 20,
    loop_stride: int = 6,
    max_features: int = 12288,
    match_threads: int = 1,
    mapper_threads: int = 1,
) -> dict[str, Any]:
    """Register a masked object video into one COLMAP model on CPU.

    See the module docstring for the pipeline stages. Returns the contract dict;
    any pycolmap error is caught and surfaced as ``ok=False`` with the message
    in ``report['error']`` instead of raising.
    """

    frames_dir = Path(frames_dir)
    eroded_masks_dir = Path(eroded_masks_dir)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "options": {
            "seed": seed,
            "fov_deg": fov_deg,
            "overlap": overlap,
            "loop_stride": loop_stride,
            "max_features": max_features,
            "match_threads": match_threads,
            "mapper_threads": mapper_threads,
        },
        "models": [],
    }
    failure = {
        "ok": False,
        "reconstruction": None,
        "model_dir": None,
        "views": [],
        "intrinsics": None,
        "points_xyz": np.zeros((0, 3), dtype=np.float64),
        "registered_fraction": 0.0,
        "report": report,
    }

    try:
        import pycolmap as pc
    except Exception as exc:  # pragma: no cover - pycolmap always present in venv
        report["error"] = f"pycolmap import failed: {exc}"
        return failure

    try:
        cv2.setNumThreads(1)
        pc.set_random_seed(seed)

        frame_paths = _frame_paths(frames_dir)
        if not frame_paths:
            report["error"] = f"no frames found in {frames_dir}"
            return failure
        frame_names = [p.name for p in frame_paths]
        report["frame_count"] = len(frame_names)

        # Camera prior from the first frame's dimensions.
        first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
        if first is None:
            report["error"] = f"could not decode {frame_paths[0]}"
            return failure
        height, width = first.shape[:2]
        focal = focal_from_fov(width, fov_deg)
        cx, cy = width / 2.0, height / 2.0
        report["image_size"] = [int(width), int(height)]
        report["camera_prior"] = f"SIMPLE_RADIAL {focal:.3f},{cx:.1f},{cy:.1f},0"

        # Stage masks into COLMAP naming.
        colmap_masks_dir = work_dir / "colmap_masks"
        report["masks"] = stage_masks(frame_paths, eroded_masks_dir, colmap_masks_dir)

        reader_options = pc.ImageReaderOptions()
        reader_options.camera_model = "SIMPLE_RADIAL"
        reader_options.camera_params = f"{focal},{cx},{cy},0"
        reader_options.mask_path = str(colmap_masks_dir)

        extraction_options = pc.FeatureExtractionOptions()
        extraction_options.num_threads = max(1, mapper_threads)
        extraction_options.use_gpu = False
        extraction_options.sift.max_num_features = max_features
        extraction_options.sift.peak_threshold = 0.0033
        extraction_options.sift.edge_threshold = 12.0
        extraction_options.sift.estimate_affine_shape = True
        extraction_options.sift.domain_size_pooling = True

        # Decide first_octave via a short timing probe (records the choice).
        first_octave, probe_report = _probe_first_octave(
            pc,
            frames_dir,
            frame_names,
            reader_options,
            extraction_options,
            work_dir,
        )
        extraction_options.sift.first_octave = first_octave
        report["extraction_probe"] = probe_report
        report["extraction"] = {
            "num_threads": extraction_options.num_threads,
            "max_num_features": max_features,
            "peak_threshold": 0.0033,
            "edge_threshold": 12.0,
            "first_octave": first_octave,
            "estimate_affine_shape": True,
            "domain_size_pooling": True,
        }

        database_path = work_dir / "database.db"
        if database_path.exists():
            database_path.unlink()

        pc.extract_features(
            database_path=str(database_path),
            image_path=str(frames_dir),
            image_names=frame_names,
            camera_mode=pc.CameraMode.SINGLE,
            reader_options=reader_options,
            extraction_options=extraction_options,
        )

        matching_options = pc.FeatureMatchingOptions()
        matching_options.num_threads = match_threads
        matching_options.use_gpu = False
        matching_options.guided_matching = True

        sequential_options = pc.SequentialPairingOptions()
        sequential_options.overlap = overlap
        sequential_options.quadratic_overlap = True
        sequential_options.loop_detection = False
        pc.match_sequential(
            database_path=str(database_path),
            matching_options=matching_options,
            pairing_options=sequential_options,
        )

        # Explicit geometric loop-closure pairs (no vocab tree).
        loop_pairs = sequential_loop_pairs(
            frame_names, overlap=overlap, loop_stride=loop_stride
        )
        report["loop_pair_count"] = len(loop_pairs)
        if loop_pairs:
            pairs_path = work_dir / "loop_pairs.txt"
            _write_pairs_file(loop_pairs, pairs_path)
            imported_options = pc.ImportedPairingOptions()
            imported_options.match_list_path = str(pairs_path)
            pc.match_image_pairs(
                database_path=str(database_path),
                matching_options=matching_options,
                pairing_options=imported_options,
            )

        mapper_options = pc.IncrementalPipelineOptions()
        mapper_options.num_threads = mapper_threads
        mapper_options.random_seed = seed
        mapper_options.min_num_matches = 15
        mapper_options.multiple_models = True
        mapper_options.max_num_models = 10
        mapper_options.ba_refine_principal_point = False
        mapper_options.mapper.init_min_num_inliers = 50
        mapper_options.mapper.init_min_tri_angle = 8.0
        mapper_options.mapper.abs_pose_min_num_inliers = 20
        mapper_options.mapper.abs_pose_min_inlier_ratio = 0.20
        mapper_options.mapper.filter_max_reproj_error = 4.0
        report["mapper"] = {
            "num_threads": mapper_threads,
            "random_seed": seed,
            "min_num_matches": 15,
            "multiple_models": True,
            "max_num_models": 10,
            "ba_refine_principal_point": False,
            "init_min_num_inliers": 50,
            "init_min_tri_angle": 8.0,
            "abs_pose_min_num_inliers": 20,
            "abs_pose_min_inlier_ratio": 0.20,
            "filter_max_reproj_error": 4.0,
        }

        sparse_dir = work_dir / "sparse"
        sparse_dir.mkdir(parents=True, exist_ok=True)
        reconstructions = pc.incremental_mapping(
            database_path=str(database_path),
            image_path=str(frames_dir),
            output_path=str(sparse_dir),
            options=mapper_options,
        )

        if not reconstructions:
            report["error"] = "mapper produced no reconstruction"
            return failure

        # Model selection: most registered images, tie-break lower mean error.
        best_key = None
        best_rank: tuple[int, float] | None = None
        for key in sorted(reconstructions.keys()):
            rec = reconstructions[key]
            num_reg = int(rec.num_reg_images())
            mean_err = _rec_mean_error(rec)
            sanity = _model_sanity(rec)
            report["models"].append(
                {
                    "model_id": int(key),
                    "num_reg_images": num_reg,
                    "mean_reprojection_error": mean_err if math.isfinite(mean_err) else None,
                    "num_points3D": int(rec.num_points3D()),
                    "sanity": sanity,
                }
            )
            if not sanity["sane"]:
                continue
            rank = (num_reg, -mean_err)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_key = key

        if best_key is None:
            report["error"] = "no sub-model passed the geometric sanity gate"
            return failure

        best = reconstructions[best_key]
        report["chosen_model"] = int(best_key)

        model_dir = work_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        best.write(str(model_dir))

        registered_fraction = float(best.num_reg_images()) / float(len(frame_names))
        return {
            "ok": True,
            "reconstruction": best,
            "model_dir": model_dir,
            "views": _build_views(best, frames_dir),
            "intrinsics": _intrinsics(best),
            "points_xyz": _points_xyz(best, min_track_length=3),
            "registered_fraction": registered_fraction,
            "report": report,
        }
    except Exception as exc:  # noqa: BLE001 - fail closed, surface in report
        report["error"] = f"{type(exc).__name__}: {exc}"
        return failure
