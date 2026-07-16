"""Silhouette-only turntable pose fallback for when SfM cannot register frames.

Textureless or deformable objects (a plush toy turned by hand) starve classical
SfM of stable features, so :mod:`masked_sfm_hull` returns no poses.  When the
capture approximates a single-axis turntable — the operator rotates the object
roughly about one axis in front of a static camera — the *silhouettes* alone
still constrain the orbit.  This module recovers an approximate circular-orbit
camera set from the masks only, following the classical turntable geometry of
Fitzgibbon/Cross/Zisserman and refining it by silhouette coherence in the
spirit of Hernandez et al.: score candidate orbits by carving a visual hull and
reprojecting it back into every mask, then maximise the mean silhouette IoU.

Honest limits.  The output is *scale-ambiguous* (object assumed centred at the
origin with characteristic radius 1.0, exactly like the SfM path) and rests on a
constant-angular-velocity assumption (azimuth grows linearly with frame index).
Concavities are never recovered — a visual hull is a silhouette bound — and the
axis/elevation are only weakly observed for near-symmetric objects.  The result
is a *fallback* pose set for hull carving, not a metric scan; callers gate on the
returned ``ok`` flag and ``score``.

Everything here is deterministic and CPU-only: no unseeded randomness, fixed
frame iteration order, gradient-free Powell refinement, and no wall-clock in any
output.  Geometry conventions match :mod:`local3d.recon_common`
(``x_cam = R @ X_world + t``; SIMPLE_RADIAL intrinsics ``(focal, cx, cy, k1)``;
boolean masks).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from scipy.ndimage import binary_erosion, uniform_filter1d
from scipy.optimize import minimize

from local3d.recon_common import Intrinsics, make_view, project_points

# Voxel carving box is [-BOX_HALF, BOX_HALF]^3 in object units (radius ~1.0).
BOX_HALF = 1.2
# Self-similarity normalisation canvas and its zoom (pixels per normalised unit).
_CANVAS = 64
_ZOOM = 24.0
# A tail frame this self-similar to frame 0 is treated as a completed revolution.
_CLOSURE_IOU = 0.86
# Views past this fraction of the clip are eligible to close the loop.
_CLOSURE_START_FRACTION = 0.60
# Refinement budget and scoring subsample (each eval carves a hull from N masks).
_MAXFEV = 120
_MAXITER = 120
_SCORE_SUBSAMPLE = 36
# Silhouette carving tolerates this fraction of disagreeing views (hand occlusion).
_MAX_VIEW_VIOLATION_FRAC = 0.05
# Acceptance gates.
_ACCEPT_SCORE = 0.66
_MIN_MASKS = 12
_MAX_AREA_RELATIVE_MAD = 0.50

# Powell parameter order: distance R, elevation, tilt_x, tilt_z, sweep (radians).
_PARAM_BOUNDS = (
    (1.8, 6.0),
    (-1.2, 1.2),
    (-0.5, 0.5),
    (-0.5, 0.5),
    (np.deg2rad(90.0), np.deg2rad(400.0)),
)


def _tilted_axis(tilt_x: float, tilt_z: float) -> np.ndarray:
    """World +Y rotated by ``tilt_x`` about X then ``tilt_z`` about Z (unit)."""

    cx, sx = np.cos(tilt_x), np.sin(tilt_x)
    cz, sz = np.cos(tilt_z), np.sin(tilt_z)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    axis = rot_z @ rot_x @ np.array([0.0, 1.0, 0.0])
    return axis / np.linalg.norm(axis)


def _equatorial_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to ``axis``."""

    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(axis @ reference)) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(axis, reference)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    return e1, e2


def camera_pose(
    distance: float,
    elevation: float,
    azimuth: float,
    *,
    tilt_x: float = 0.0,
    tilt_z: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """World->camera ``(rotation, translation)`` for one orbit position.

    The camera sits at ``distance`` from the origin at the given ``elevation``
    (radians above the orbit plane) and ``azimuth`` around the rotation axis,
    which is world +Y tilted by ``(tilt_x, tilt_z)``.  It looks at the origin
    with the tilted axis as up, following ``x_cam = R @ X + t`` (COLMAP, y-down).
    """

    axis = _tilted_axis(tilt_x, tilt_z)
    e1, e2 = _equatorial_basis(axis)
    in_plane = np.cos(azimuth) * e1 + np.sin(azimuth) * e2
    center = distance * (np.cos(elevation) * in_plane + np.sin(elevation) * axis)

    forward = -center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, axis)
    norm_right = np.linalg.norm(right)
    if norm_right < 1e-8:  # pragma: no cover - looking straight down the axis
        right = e1.copy()
    else:
        right /= norm_right
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ center
    return rotation, translation


def _voxel_grid(grid: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Centred ``(grid**3, 3)`` voxel coordinates plus the (z, y, x) shape."""

    axis = np.linspace(-BOX_HALF, BOX_HALF, grid)
    gz, gy, gx = np.meshgrid(axis, axis, axis, indexing="ij")
    voxels = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
    return voxels, (grid, grid, grid)


def _carve(
    masks: Sequence[np.ndarray],
    view_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    intrinsics: Intrinsics,
    voxels: np.ndarray,
    shape: tuple[int, int, int],
    max_view_violation_frac: float,
) -> np.ndarray:
    """Boolean occupancy: voxels inside every silhouette (bar a few outliers).

    A voxel that projects outside the image or onto background counts as one
    disagreement; it is carved once it disagrees with more than
    ``floor(max_view_violation_frac * len(masks))`` views, so transient hand
    occlusion cannot bite the whole object away.
    """

    occupied = np.ones(len(voxels), dtype=bool)
    violations = np.zeros(len(voxels), dtype=np.int32)
    max_violations = int(np.floor(max_view_violation_frac * len(masks)))
    for mask, (rotation, translation) in zip(masks, view_pairs):
        active = np.flatnonzero(occupied)
        if active.size == 0:
            break
        height, width = mask.shape
        u, v, depth = project_points(voxels[active], rotation, translation, intrinsics)
        inside = (depth > 1e-9) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        col = np.clip(np.rint(u).astype(np.int64), 0, width - 1)
        row = np.clip(np.rint(v).astype(np.int64), 0, height - 1)
        survives = np.zeros(active.size, dtype=bool)
        survives[inside] = mask[row[inside], col[inside]]
        failed = active[~survives]
        violations[failed] += 1
        occupied[failed[violations[failed] > max_violations]] = False
    return occupied.reshape(shape)


def _silhouette_iou(
    occupancy: np.ndarray,
    voxels: np.ndarray,
    masks: Sequence[np.ndarray],
    view_pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    intrinsics: Intrinsics,
) -> float:
    """Mean IoU of the reprojected hull surface against each input mask.

    Only surface voxels are reprojected; the sparse splat is solidified with a
    2px dilation and a 5px close (mirroring ``masked_sfm_hull.evaluate_and_carve``)
    before comparison.
    """

    surface = occupancy & ~binary_erosion(occupancy)
    points = voxels[surface.ravel()]
    if points.shape[0] == 0:
        return 0.0
    dilate_kernel = np.ones((3, 3), np.uint8)
    close_kernel = np.ones((5, 5), np.uint8)
    ious: list[float] = []
    for mask, (rotation, translation) in zip(masks, view_pairs):
        height, width = mask.shape
        u, v, depth = project_points(points, rotation, translation, intrinsics)
        valid = (depth > 1e-9) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        silhouette = np.zeros((height, width), dtype=np.uint8)
        if valid.any():
            col = np.clip(np.rint(u[valid]).astype(np.int64), 0, width - 1)
            row = np.clip(np.rint(v[valid]).astype(np.int64), 0, height - 1)
            silhouette[row, col] = 1
            silhouette = cv2.dilate(silhouette, dilate_kernel, iterations=2)
            silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, close_kernel)
        solid = silhouette > 0
        intersection = float(np.logical_and(solid, mask).sum())
        union = float(np.logical_or(solid, mask).sum())
        ious.append(intersection / max(union, 1.0))
    return float(np.mean(ious))


def carve_hull(
    masks: Sequence[np.ndarray],
    views: Sequence[dict[str, Any]],
    intrinsics: Intrinsics,
    *,
    grid: int = 96,
    max_view_violation_frac: float = _MAX_VIEW_VIOLATION_FRAC,
) -> np.ndarray:
    """Carve a boolean occupancy grid from posed silhouettes (public helper).

    ``views`` are :func:`recon_common.make_view` dicts; only their ``rotation``
    and ``translation`` are used.  The grid spans ``[-BOX_HALF, BOX_HALF]^3``.
    """

    cv2.setNumThreads(1)
    intr = tuple(float(value) for value in intrinsics)
    binary = [np.asarray(mask).astype(bool) for mask in masks]
    voxels, shape = _voxel_grid(grid)
    view_pairs = [(view["rotation"], view["translation"]) for view in views]
    return _carve(binary, view_pairs, intr, voxels, shape, max_view_violation_frac)


def hull_volume(occupancy: np.ndarray, *, box_half: float = BOX_HALF) -> float:
    """Object-unit volume of an occupancy grid (voxel count times cell volume)."""

    grid = occupancy.shape[0]
    spacing = (2.0 * box_half) / (grid - 1)
    return float(occupancy.sum()) * spacing**3


def _normalized_silhouette(mask: np.ndarray) -> np.ndarray:
    """Render one silhouette centred and scaled by ``sqrt(area)`` onto a canvas.

    Removing translation and scale leaves only shape, so IoU between two
    normalised silhouettes measures how similar the object's orientation is —
    the basis for detecting a completed revolution.
    """

    rows, cols = np.nonzero(mask)
    if cols.size == 0:
        return np.zeros((_CANVAS, _CANVAS), dtype=bool)
    center_x = float(cols.mean())
    center_y = float(rows.mean())
    scale = np.sqrt(float(cols.size))
    factor = _ZOOM / scale
    affine = np.array(
        [
            [factor, 0.0, _CANVAS / 2.0 - factor * center_x],
            [0.0, factor, _CANVAS / 2.0 - factor * center_y],
        ],
        dtype=np.float64,
    )
    warped = cv2.warpAffine(
        mask.astype(np.uint8),
        affine,
        (_CANVAS, _CANVAS),
        flags=cv2.INTER_NEAREST,
    )
    return warped > 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return intersection / max(union, 1.0)


def _estimate_sweep(masks: Sequence[np.ndarray]) -> dict[str, Any]:
    """Initial sweep from the frame-0 self-similarity curve (Fitzgibbon idea).

    If a frame past ``_CLOSURE_START_FRACTION`` of the clip returns to frame 0's
    normalised silhouette (IoU over ``_CLOSURE_IOU``) the capture is treated as a
    full revolution ending there; otherwise a partial 240 degree sweep is assumed.
    """

    n = len(masks)
    normalized = [_normalized_silhouette(mask) for mask in masks]
    similarity = np.array([_iou(normalized[0], normalized[i]) for i in range(n)])
    smoothed = uniform_filter1d(similarity, size=5, mode="nearest")

    start = int(round(_CLOSURE_START_FRACTION * (n - 1)))
    start = min(max(start, 1), n - 1)
    tail_peak_index = start + int(np.argmax(smoothed[start:]))
    peak = float(smoothed[tail_peak_index])

    if peak > _CLOSURE_IOU:
        sweep_deg = 360.0 * (n - 1) / tail_peak_index
        closure = True
    else:
        sweep_deg = 240.0
        closure = False
    sweep_deg = float(np.clip(sweep_deg, 90.0, 400.0))
    return {
        "similarity": similarity,
        "smoothed": smoothed,
        "closure_frame": int(tail_peak_index),
        "closure_peak_iou": peak,
        "closure_detected": closure,
        "sweep_init_deg": sweep_deg,
    }


def _clip_params(x: Sequence[float]) -> np.ndarray:
    return np.array(
        [float(np.clip(x[i], low, high)) for i, (low, high) in enumerate(_PARAM_BOUNDS)]
    )


def _even_subsample(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, k).round().astype(int).tolist()))


def _view_pairs(params: Sequence[float], indices: np.ndarray, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    distance, elevation, tilt_x, tilt_z, sweep = params
    azimuths = sweep * indices / (n - 1)
    return [
        camera_pose(distance, elevation, float(a), tilt_x=tilt_x, tilt_z=tilt_z)
        for a in azimuths
    ]


def _characteristic_radius(occupancy: np.ndarray, voxels: np.ndarray) -> float:
    """Largest object-unit half-extent of an occupancy grid (0 when empty)."""

    points = voxels[occupancy.ravel()]
    if points.shape[0] == 0:
        return 0.0
    return float(np.abs(points).max())


def _pin_and_score(
    params: np.ndarray,
    masks: Sequence[np.ndarray],
    indices: np.ndarray,
    n: int,
    intrinsics: Intrinsics,
    voxels: np.ndarray,
    shape: tuple[int, int, int],
    *,
    pin_iters: int,
) -> tuple[float, np.ndarray, list[tuple[np.ndarray, np.ndarray]], float]:
    """Resolve the R/scale ambiguity, then score the carved hull.

    Silhouette IoU is invariant to scaling camera distance and object size
    together, so Powell leaves R undetermined and it drifts to a bound, leaving
    the object badly resolved on the fixed voxel box.  Here the camera distance
    is chosen so the hull has characteristic radius 1.0 (the module's scale
    convention and a box-filling, well-resolved object): hull radius grows in
    proportion to distance, so a few exact linear updates settle it.  Returns the
    mean reprojected-silhouette IoU, the pinned occupancy/poses and the distance.
    """

    distance = float(params[0])
    others = params[1:]
    pairs = _view_pairs(np.concatenate(([distance], others)), indices, n)
    occupancy = _carve(masks, pairs, intrinsics, voxels, shape, _MAX_VIEW_VIOLATION_FRAC)
    for _ in range(pin_iters):
        radius = _characteristic_radius(occupancy, voxels)
        if radius <= 1e-6 or abs(radius - 1.0) < 0.02:
            break
        distance = float(np.clip(distance / radius, 0.1, 50.0))
        pairs = _view_pairs(np.concatenate(([distance], others)), indices, n)
        occupancy = _carve(masks, pairs, intrinsics, voxels, shape, _MAX_VIEW_VIOLATION_FRAC)
    if not occupancy.any():
        return 0.0, occupancy, pairs, distance
    score = _silhouette_iou(occupancy, voxels, masks, pairs, intrinsics)
    return score, occupancy, pairs, distance


def fit_turntable_poses(
    masks: list[np.ndarray],
    intrinsics: Intrinsics,
    *,
    grid: int = 96,
    refine_grid: int = 128,
    seed: int = 0,
) -> dict[str, Any]:
    """Recover approximate turntable camera poses from object silhouettes alone.

    Returns a dict with ``ok`` (accepted), ``views`` (one
    :func:`recon_common.make_view` per input frame, ``image_path`` a placeholder
    the caller fills in later), ``score`` (mean reprojected-silhouette IoU),
    ``sweep_deg`` (total recovered azimuth sweep) and a JSON-able ``report``.
    Poses are scale-ambiguous: the object is assumed centred at the origin with
    characteristic radius ~1.0.
    """

    cv2.setNumThreads(1)
    intr = tuple(float(value) for value in intrinsics)
    binary = [np.asarray(mask).astype(bool) for mask in masks]
    n = len(binary)

    reasons: list[str] = []
    if n < _MIN_MASKS:
        reasons.append(f"too_few_masks (<{_MIN_MASKS})")

    if n:
        areas = np.array([float(mask.sum()) for mask in binary])
        median_area = float(np.median(areas))
        if median_area <= 0.0:
            relative_mad = float("inf")
            reasons.append("empty_masks")
        else:
            relative_mad = float(np.median(np.abs(areas - median_area)) / median_area)
            if relative_mad > _MAX_AREA_RELATIVE_MAD:
                reasons.append("mask_area_varies (>50% relative MAD)")
    else:
        relative_mad = float("inf")
        reasons.append("no_masks")

    # Too small to carve meaningfully: return default poses, fail closed.
    if n < 4:
        sweep_init_deg = 240.0
        best = _clip_params([3.5, 0.15, 0.0, 0.0, np.deg2rad(sweep_init_deg)])
        pairs = _view_pairs(best, np.arange(n), max(n, 2))
        views = _package_views(best, np.arange(n), max(n, 2), pairs, binary)
        report = _make_report(
            n=n, n_scored=0, params=best, sweep_init_deg=sweep_init_deg,
            closure_frame=-1, closure_peak_iou=0.0, closure_detected=False,
            score=0.0, relative_mad=relative_mad, reasons=reasons, ok=False,
            optimizer={"method": "Powell", "maxfev": _MAXFEV, "evals": 0, "success": False},
            grid=grid, refine_grid=refine_grid, seed=seed,
        )
        return {"ok": False, "views": views, "score": 0.0, "sweep_deg": float(np.rad2deg(best[4])), "report": report}

    sweep = _estimate_sweep(binary)

    score_indices = np.asarray(_even_subsample(n, _SCORE_SUBSAMPLE), dtype=float)
    score_masks = [binary[int(i)] for i in score_indices]
    voxels_obj, shape_obj = _voxel_grid(grid)

    def objective(x: np.ndarray) -> float:
        params = _clip_params(x)
        score, _occ, _pairs, _dist = _pin_and_score(
            params, score_masks, score_indices, n, intr, voxels_obj, shape_obj, pin_iters=1
        )
        return -score

    x0 = np.array([3.5, 0.15, 0.0, 0.0, np.deg2rad(sweep["sweep_init_deg"])])
    result = minimize(
        objective,
        x0,
        method="Powell",
        options={"maxfev": _MAXFEV, "maxiter": _MAXITER, "xtol": 1e-3, "ftol": 1e-3},
    )
    # Powell should not return worse than x0, but guard deterministically.
    best = _clip_params(result.x) if result.fun <= objective(x0) else _clip_params(x0)

    indices_all = np.arange(n, dtype=float)
    voxels_ref, shape_ref = _voxel_grid(refine_grid)
    # Pin the scale ambiguity to characteristic radius 1.0, then score all frames.
    score, _occ, pairs_all, distance = _pin_and_score(
        best, binary, indices_all, n, intr, voxels_ref, shape_ref, pin_iters=3
    )
    best[0] = distance

    ok = (score >= _ACCEPT_SCORE) and not reasons
    if score < _ACCEPT_SCORE:
        reasons = reasons + [f"score_below_{_ACCEPT_SCORE}"]

    views = _package_views(best, indices_all, n, pairs_all, binary)
    report = _make_report(
        n=n, n_scored=len(score_indices), params=best, sweep_init_deg=sweep["sweep_init_deg"],
        closure_frame=sweep["closure_frame"], closure_peak_iou=sweep["closure_peak_iou"],
        closure_detected=sweep["closure_detected"], score=score, relative_mad=relative_mad,
        reasons=reasons, ok=ok,
        optimizer={
            "method": "Powell",
            "maxfev": _MAXFEV,
            "evals": int(result.nfev),
            "success": bool(result.success),
        },
        grid=grid, refine_grid=refine_grid, seed=seed,
    )
    return {
        "ok": bool(ok),
        "views": views,
        "score": float(score),
        "sweep_deg": float(np.rad2deg(best[4])),
        "report": report,
    }


def _package_views(
    params: Sequence[float],
    indices: np.ndarray,
    n: int,
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    masks: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    distance, elevation, tilt_x, tilt_z, sweep = params
    views: list[dict[str, Any]] = []
    for position, index in enumerate(indices):
        rotation, translation = pairs[position]
        azimuth_deg = float(np.rad2deg(sweep * index / (n - 1))) if n > 1 else 0.0
        views.append(
            make_view(
                name="frame_%04d" % int(index),
                image_path=Path(str(int(index))),
                rotation=rotation,
                translation=translation,
                mask_tight=masks[int(index)] if int(index) < len(masks) else None,
                extras={"azimuth_deg": azimuth_deg},
            )
        )
    return views


def _make_report(
    *,
    n: int,
    n_scored: int,
    params: Sequence[float],
    sweep_init_deg: float,
    closure_frame: int,
    closure_peak_iou: float,
    closure_detected: bool,
    score: float,
    relative_mad: float,
    reasons: list[str],
    ok: bool,
    optimizer: dict[str, Any],
    grid: int,
    refine_grid: int,
    seed: int,
) -> dict[str, Any]:
    distance, elevation, tilt_x, tilt_z, sweep = params
    return {
        "method": (
            "silhouette turntable (Fitzgibbon/Cross/Zisserman) refined by "
            "silhouette coherence (Hernandez); CPU, deterministic"
        ),
        "n_frames": int(n),
        "n_scored": int(n_scored),
        "distance": float(distance),
        "elevation_rad": float(elevation),
        "tilt_x_rad": float(tilt_x),
        "tilt_z_rad": float(tilt_z),
        "sweep_deg": float(np.rad2deg(sweep)),
        "sweep_init_deg": float(sweep_init_deg),
        "closure_detected": bool(closure_detected),
        "closure_frame": int(closure_frame),
        "closure_peak_iou": float(closure_peak_iou),
        "score": float(score),
        "accept_score_threshold": float(_ACCEPT_SCORE),
        "mask_area_relative_mad": float(relative_mad),
        "ok": bool(ok),
        "fail_reasons": list(reasons),
        "optimizer": optimizer,
        "grid": int(grid),
        "refine_grid": int(refine_grid),
        "seed": int(seed),
        "assumptions": [
            "single rigid rotation axis (turntable / hand rotation)",
            "constant angular velocity: azimuth_i = sweep * i/(N-1)",
            "object centred at origin, characteristic radius ~1.0 (scale ambiguous)",
            "given intrinsics trusted; camera distance R absorbs focal/scale error",
            "visual-hull bound: concavities are not recovered",
        ],
    }
