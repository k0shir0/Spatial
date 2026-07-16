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

from local3d.recon_common import Intrinsics, make_view, project_points

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


def _masked_views_from_rec(rec: Any, masks_dir: Path) -> list[dict[str, Any]]:
    views = []
    for image_id in rec.reg_image_ids():
        image = rec.image(image_id)
        mask = cv2.imread(
            str(Path(masks_dir) / f"{Path(image.name).stem}_mask.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        if mask is None:
            continue
        pose = image.cam_from_world()
        views.append(
            make_view(
                name=image.name,
                image_path=Path(image.name),
                rotation=np.asarray(pose.rotation.matrix()),
                translation=np.asarray(pose.translation),
                mask_tight=mask > 127,
                extras={"image_id": int(image_id)},
            )
        )
    return views


def _hull_view_ious(
    views: list[dict[str, Any]],
    intrinsics: Intrinsics,
    points: np.ndarray,
    *,
    grid: int = 96,
) -> np.ndarray:
    """Per-view silhouette-reprojection IoU of the carved (point-bounded) hull.

    Carves a coarse hull from ALL given views' masks, intersected with the
    inflated point convex hull, then scores every view by how well the hull
    surface reprojects onto that view's mask.
    """

    from local3d.fusion import carve_hull, point_hull_occupancy
    from scipy import ndimage

    lower = np.percentile(points, 1, axis=0)
    upper = np.percentile(points, 99, axis=0)
    pad = 0.15 * (upper - lower)
    bounds = (lower - pad, upper + pad)
    occupancy, _ = carve_hull(views, intrinsics, bounds, resolution=grid)
    try:
        occupancy &= point_hull_occupancy(points, bounds, grid)
    except Exception:
        pass
    if not occupancy.any():
        return np.zeros(len(views))

    surface = occupancy & ~ndimage.binary_erosion(occupancy)
    zz, yy, xx = np.nonzero(surface)
    axes = [np.linspace(bounds[0][i], bounds[1][i], grid) for i in range(3)]
    world = np.column_stack([axes[0][xx], axes[1][yy], axes[2][zz]])

    ious = np.zeros(len(views))
    for index, view in enumerate(views):
        mask = view["mask_tight"]
        height, width = mask.shape
        u, v, depth = project_points(
            world, view["rotation"], view["translation"], intrinsics
        )
        ok = (depth > 0) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        quarter_h, quarter_w = height // 4, width // 4
        silhouette = np.zeros((quarter_h, quarter_w), np.uint8)
        vi = np.clip((v[ok] / 4).astype(np.int64), 0, quarter_h - 1)
        ui = np.clip((u[ok] / 4).astype(np.int64), 0, quarter_w - 1)
        silhouette[vi, ui] = 1
        silhouette = cv2.morphologyEx(
            cv2.dilate(silhouette, np.ones((5, 5), np.uint8)),
            cv2.MORPH_CLOSE,
            np.ones((9, 9), np.uint8),
        )
        small = cv2.resize(
            mask.astype(np.uint8), (quarter_w, quarter_h),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        union = float(np.logical_or(silhouette > 0, small).sum())
        ious[index] = float(np.logical_and(silhouette > 0, small).sum()) / max(union, 1.0)
    return ious


def _prune_views_by_coherence(
    views: list[dict[str, Any]],
    intrinsics: Intrinsics,
    points: np.ndarray,
    *,
    min_iou: float = 0.45,
    rounds: int = 3,
    min_views: int = 10,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Iteratively drop views whose silhouettes disagree with the common hull.

    A large sub-model often mixes an accurate majority with a sloppy minority
    (arcs glued by weak matches).  Dropping incoherent views and re-carving
    salvages the majority instead of rejecting the whole model.
    """

    kept = list(views)
    report: dict[str, Any] = {"rounds": []}
    ious = np.zeros(len(kept))
    for _ in range(rounds):
        if len(kept) < min_views:
            break
        ious = _hull_view_ious(kept, intrinsics, points)
        bad = ious < min_iou
        report["rounds"].append(
            {
                "views": len(kept),
                "median_iou": float(np.median(ious)),
                "dropped": int(bad.sum()),
            }
        )
        if not bad.any():
            break
        if len(kept) - int(bad.sum()) < min_views:
            kept = [v for v, b in zip(kept, bad) if not b]
            break
        kept = [v for v, b in zip(kept, bad) if not b]
    if len(ious) != len(kept) and len(kept) >= min_views:
        ious = _hull_view_ious(kept, intrinsics, points)
    return kept, ious, report


def _rec_mean_error(rec: Any) -> float:
    try:
        value = float(rec.compute_mean_reprojection_error())
    except Exception:
        return float("inf")
    if not math.isfinite(value):
        return float("inf")
    return value


def camera_coverage(views: list[dict[str, Any]], points_xyz: np.ndarray) -> dict[str, Any]:
    """Measure object-centric angular support of a coherent SfM sub-model."""

    if len(views) < 2 or len(points_xyz) < 2:
        return {"azimuth_span_deg": 0.0, "occupied_30deg_bins": 0, "elevation_span_deg": 0.0}
    object_center = np.median(np.asarray(points_xyz, dtype=np.float64), axis=0)
    centers = np.stack([
        -np.asarray(view["rotation"], dtype=np.float64).T
        @ np.asarray(view["translation"], dtype=np.float64)
        for view in views
    ])
    directions = centers - object_center
    norms = np.linalg.norm(directions, axis=1)
    directions = directions[norms > 1e-9]
    norms = norms[norms > 1e-9]
    if len(directions) < 2:
        return {"azimuth_span_deg": 0.0, "occupied_30deg_bins": 0, "elevation_span_deg": 0.0}
    directions = directions / norms[:, None]
    _u, _s, vh = np.linalg.svd(
        directions - directions.mean(axis=0), full_matrices=False
    )
    axis_u, axis_v = vh[0], vh[1]
    normal = np.cross(axis_u, axis_v)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    azimuth = np.mod(
        np.degrees(np.arctan2(directions @ axis_v, directions @ axis_u)), 360.0
    )
    ordered = np.sort(azimuth)
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 360.0)))
    span = 360.0 - float(gaps.max())
    bins = len(set(int(value // 30.0) % 12 for value in azimuth))
    elevation = np.degrees(np.arcsin(np.clip(directions @ normal, -1.0, 1.0)))
    return {
        "azimuth_span_deg": round(span, 4),
        "occupied_30deg_bins": int(bins),
        "elevation_span_deg": round(float(np.ptp(elevation)), 4),
    }


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


def _filtered_point_records(
    rec: Any, *, min_track_length: int = 3
) -> list[tuple[int, Any, np.ndarray]]:
    """Return deterministic object-point records after the positional filter.

    Keeping the COLMAP point id and ``Point3D`` object beside ``xyz`` is
    essential for honest depth validation: a depth prediction for one image
    may only be calibrated against tracks that COLMAP actually observed in
    that image.  The previous ``_points_xyz`` helper discarded this identity
    and downstream code projected every global point into every camera,
    including occluded/rear points.
    """

    records: list[tuple[int, Any, np.ndarray]] = []
    for point_id in sorted(rec.points3D.keys()):
        point = rec.points3D[point_id]
        if point.track.length() >= min_track_length:
            records.append(
                (
                    int(point_id),
                    point,
                    np.asarray(point.xyz, dtype=np.float64).reshape(3),
                )
            )
    if not records:
        return []
    array = np.asarray([record[2] for record in records], dtype=np.float64)
    # Drop the positional outlier shell (background/hand leakage): keep points
    # within 3x the 75th-percentile radius of the median center, so downstream
    # bounds are sized by the object, not by stray tracks.
    center = np.median(array, axis=0)
    radius = np.linalg.norm(array - center, axis=1)
    keep = radius <= 3.0 * max(float(np.percentile(radius, 75)), 1e-9)
    return [record for record, retained in zip(records, keep) if bool(retained)]


def _points_xyz(rec: Any, *, min_track_length: int = 3) -> np.ndarray:
    records = _filtered_point_records(rec, min_track_length=min_track_length)
    if not records:
        return np.zeros((0, 3), dtype=np.float64)
    return np.asarray([record[2] for record in records], dtype=np.float64)


def _track_evidence(
    rec: Any,
    views: list[dict[str, Any]],
    *,
    min_track_length: int = 3,
) -> dict[str, Any]:
    """Extract actual per-image sparse observations for depth calibration.

    Arrays are ordered by ``point3d_id`` and contain only observations from
    the coherent delivery views.  This structure stays out of JSON reports;
    callers persist only aggregate metrics and hashes.
    """

    records = _filtered_point_records(rec, min_track_length=min_track_length)
    view_by_name = {str(view["name"]): view for view in views}
    observations: dict[str, list[tuple[int, np.ndarray, np.ndarray, float, int, float]]] = {
        name: [] for name in sorted(view_by_name)
    }
    point_ids: list[int] = []
    points: list[np.ndarray] = []
    track_lengths: list[int] = []
    errors: list[float] = []

    for point_id, point, xyz in records:
        length = int(point.track.length())
        error = float(point.error) if bool(getattr(point, "has_error", False)) else float("nan")
        point_ids.append(point_id)
        points.append(xyz)
        track_lengths.append(length)
        errors.append(error)
        for element in point.track.elements:
            image = rec.image(int(element.image_id))
            name = str(image.name)
            view = view_by_name.get(name)
            if view is None:
                continue
            point2d = image.points2D[int(element.point2D_idx)]
            xy = np.asarray(point2d.xy, dtype=np.float64).reshape(2)
            camera_xyz = (
                np.asarray(view["rotation"], dtype=np.float64) @ xyz
                + np.asarray(view["translation"], dtype=np.float64)
            )
            z_camera = float(camera_xyz[2])
            if not np.isfinite(z_camera) or z_camera <= 1e-9:
                continue
            observations[name].append((point_id, xy, xyz, z_camera, length, error))

    per_view: dict[str, dict[str, np.ndarray]] = {}
    for name, items in observations.items():
        items.sort(key=lambda item: item[0])
        per_view[name] = {
            "point3d_ids": np.asarray([item[0] for item in items], dtype=np.int64),
            "xy": np.asarray([item[1] for item in items], dtype=np.float64).reshape(-1, 2),
            "xyz_world": np.asarray([item[2] for item in items], dtype=np.float64).reshape(-1, 3),
            "z_camera": np.asarray([item[3] for item in items], dtype=np.float64),
            "track_lengths": np.asarray([item[4] for item in items], dtype=np.int32),
            "reprojection_errors_px": np.asarray([item[5] for item in items], dtype=np.float64),
        }

    return {
        "point3d_ids": np.asarray(point_ids, dtype=np.int64),
        "points_xyz": np.asarray(points, dtype=np.float64).reshape(-1, 3),
        "track_lengths": np.asarray(track_lengths, dtype=np.int32),
        "reprojection_errors_px": np.asarray(errors, dtype=np.float64),
        "views": per_view,
    }


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
    match_threads: int = 1,  # keep 1: threaded matching makes the solve vary run to run
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
        "track_evidence": {
            "point3d_ids": np.zeros(0, dtype=np.int64),
            "points_xyz": np.zeros((0, 3), dtype=np.float64),
            "track_lengths": np.zeros(0, dtype=np.int32),
            "reprojection_errors_px": np.zeros(0, dtype=np.float64),
            "views": {},
        },
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

        # Model selection: broad coherent angular support first. A large narrow
        # arc of hand-anchored views must not beat a smaller object orbit.
        best_key = None
        best_kept: list[dict[str, Any]] | None = None
        best_rank: tuple[int, float, float, int, float] | None = None
        for key in sorted(reconstructions.keys()):
            rec = reconstructions[key]
            num_reg = int(rec.num_reg_images())
            mean_err = _rec_mean_error(rec)
            entry: dict[str, Any] = {
                "model_id": int(key),
                "num_reg_images": num_reg,
                "mean_reprojection_error": mean_err if math.isfinite(mean_err) else None,
                "num_points3D": int(rec.num_points3D()),
            }
            report["models"].append(entry)
            if num_reg < 10:
                entry["skipped"] = "too few registered images"
                continue
            points = _points_xyz(rec, min_track_length=3)
            if len(points) < 50:
                entry["skipped"] = "too few tracked points"
                continue
            model_views = _masked_views_from_rec(rec, eroded_masks_dir)
            kept, ious, prune_report = _prune_views_by_coherence(
                model_views, _intrinsics(rec), points
            )
            entry["coherence"] = prune_report
            entry["kept_views"] = len(kept)
            if len(kept) < 10 or len(ious) != len(kept):
                continue
            median_iou = float(np.median(ious))
            entry["kept_median_iou"] = median_iou
            if median_iou < 0.45:
                continue
            coverage = camera_coverage(kept, points)
            entry["camera_coverage"] = coverage
            rank = (
                int(coverage["occupied_30deg_bins"]),
                float(coverage["azimuth_span_deg"]),
                median_iou,
                len(kept),
                -mean_err,
            )
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_key = key
                best_kept = kept

        if best_key is None or best_kept is None:
            report["error"] = "no sub-model passed the silhouette coherence gate"
            return failure

        best = reconstructions[best_key]
        report["chosen_model"] = int(best_key)
        report["kept_views"] = len(best_kept)
        report["camera_coverage"] = camera_coverage(
            best_kept, _points_xyz(best, min_track_length=3)
        )

        model_dir = work_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        best.write(str(model_dir))

        # Rebuild delivery views (with image paths) restricted to the coherent set.
        kept_names = {view["name"] for view in best_kept}
        views = [
            view for view in _build_views(best, frames_dir)
            if view["name"] in kept_names
        ]
        registered_fraction = float(len(views)) / float(len(frame_names))
        return {
            "ok": True,
            "reconstruction": best,
            "model_dir": model_dir,
            "views": views,
            "intrinsics": _intrinsics(best),
            "points_xyz": _points_xyz(best, min_track_length=3),
            "track_evidence": _track_evidence(best, views, min_track_length=3),
            "registered_fraction": registered_fraction,
            "report": report,
        }
    except Exception as exc:  # noqa: BLE001 - fail closed, surface in report
        report["error"] = f"{type(exc).__name__}: {exc}"
        return failure
