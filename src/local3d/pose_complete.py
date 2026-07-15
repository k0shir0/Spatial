"""Fill SfM pose gaps from silhouettes so hull carving sees the full orbit.

COLMAP often registers only part of a hand-rotated capture: the feature-poor or
motion-blurred arcs never register even though the object *mask* on those frames
is perfectly good.  The registered cameras still span most of the orbit (e.g.
331 deg with a 48-99 deg hole), so the missing poses lie on the same smooth
turntable-like path.  This module recovers those missing poses from the masks
alone:

1. Carve a coarse visual hull from the registered views (``local3d.fusion``) and
   take the object centroid and hull surface voxels from it.
2. Fit the registered camera centres to an orbit: a PCA plane (normal + two
   in-plane axes) about the centroid, giving each registered frame an
   ``(azimuth, elevation, radius)``.  Azimuths are unwrapped in *temporal* order
   so interpolation crosses the +-180 deg seam correctly.
3. Interpolate ``(azimuth, elevation, radius)`` for every unregistered frame
   between its temporally nearest registered anchors (ends extrapolate with the
   local angular velocity, capped).
4. Build a look-at pose from that orbit sample (up = plane normal), matching the
   ``x_cam = R @ X + t`` / y-down convention of :mod:`local3d.recon_common`.
5. Refine each pose with a gradient-free Powell search that maximises the IoU
   between the frame's mask and the reprojected hull silhouette.
6. Accept a frame when its refined IoU clears a threshold calibrated from the
   registered views' own hull-reprojection IoU.  With ``iterations > 1`` the
   hull is re-carved including the accepted silhouette views and the still
   rejected frames are retried against the improved hull.

Everything is deterministic and CPU-only: views are processed in ``name`` order,
the optimiser starts from a fixed ``x0`` with no randomness, and no wall-clock or
unseeded state enters any output.  Geometry conventions match
:mod:`local3d.recon_common` (``x_cam = R @ X_world + t``; SIMPLE_RADIAL
intrinsics ``(focal, cx, cy, k1)``; boolean masks).

Honest limits: the orbit model assumes the missing cameras lie on the same
smooth path as the registered ones and that the object up-axis is the plane
normal (no per-frame roll).  Completed poses are only good enough to *carve*
silhouettes; the caller must not feed them to metric depth fusion or texturing.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.optimize import minimize

from local3d import fusion
from local3d.recon_common import Intrinsics, project_points

# Scoring canvas: masks and rendered silhouettes are compared at roughly this
# width (never upscaled) to keep the ~thousands of IoU evaluations cheap.
_TARGET_WIDTH = 360
# Silhouette solidification (identical recipe to turntable_pose / masked_sfm_hull).
_DILATE_KERNEL = np.ones((3, 3), np.uint8)
_CLOSE_KERNEL = np.ones((5, 5), np.uint8)
# Powell refinement budget and per-parameter clip ranges.
_MAXFEV = 50
_D_AZ_DEG = 35.0
_D_EL_DEG = 15.0
_LOG_R = 0.15
# Acceptance floor and fraction of the registered-view IoU it must clear.
_ACCEPT_FLOOR = 0.50
_ACCEPT_FACTOR = 0.70
# End-extrapolation cap on azimuth change (deg) beyond the outermost anchor.
_EXTRAP_AZ_CAP_DEG = 45.0
# Upper bound on reprojected surface voxels (deterministic stride if exceeded).
_SURFACE_CAP = 120_000


# --------------------------------------------------------------------------- #
# Small geometry / rendering helpers
# --------------------------------------------------------------------------- #
def _axes(bounds: tuple[np.ndarray, np.ndarray], resolution: int) -> list[np.ndarray]:
    lower, upper = bounds
    return [
        np.linspace(float(lower[i]), float(upper[i]), resolution) for i in range(3)
    ]


def _occupied_world(
    occupancy: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """World xyz of every occupied voxel (matches the fusion ``[z, y, x]`` layout)."""

    resolution = int(occupancy.shape[0])
    axes = _axes(bounds, resolution)
    iz, iy, ix = np.nonzero(occupancy)
    return np.stack((axes[0][ix], axes[1][iy], axes[2][iz]), axis=1).astype(np.float64)


def _tighten_bounds(
    occupancy: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    views: list[dict],
    intrinsics: Intrinsics,
    *,
    pad: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Shrink a loose carve box to the object's frustum-intersection region.

    ``fusion.bounds_from_silhouettes`` only *brackets* the object (its own test
    tolerates a box up to 20 units wide around a unit sphere): for a narrow-FOV,
    near-equatorial orbit the generous cube's exterior voxels project off-image
    in every view, so they never carve and the box stays huge.  The object,
    however, lives where every camera can see it, so the axis-aligned box of the
    occupied voxels visible on-image in *all* views (relaxing the fraction only
    if that leaves too few) is a tight, deterministic bracket.  Falls back to the
    input ``bounds`` if the intersection is degenerate.
    """

    world = _occupied_world(occupancy, bounds)
    if len(world) == 0:
        return bounds
    seen = np.zeros(len(world), dtype=np.int64)
    for view in views:
        u, v, depth = project_points(
            world, view["rotation"], view["translation"], intrinsics
        )
        height, width = np.asarray(view["mask_tight"]).shape[:2]
        seen += (
            (depth > 1e-9) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        ).astype(np.int64)
    total = len(views)
    keep = world[seen >= total]
    for fraction in (0.9, 0.75):
        if len(keep) >= 8:
            break
        keep = world[seen >= int(np.ceil(fraction * total))]
    if len(keep) < 8:
        return bounds
    lower = keep.min(axis=0)
    upper = keep.max(axis=0)
    margin = pad * np.maximum(upper - lower, 1e-6)
    return lower - margin, upper + margin


def _surface_world(
    occupancy: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """World xyz of hull *surface* voxels (occ & ~erode(occ)), capped by stride."""

    occ = np.asarray(occupancy, dtype=bool)
    surface = occ & ~binary_erosion(occ)
    resolution = int(occ.shape[0])
    axes = _axes(bounds, resolution)
    iz, iy, ix = np.nonzero(surface)
    points = np.stack(
        (axes[0][ix], axes[1][iy], axes[2][iz]), axis=1
    ).astype(np.float64)
    if len(points) > _SURFACE_CAP:
        stride = int(np.ceil(len(points) / _SURFACE_CAP))
        points = points[::stride]
    return points


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return intersection / max(union, 1.0)


def _downsample_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    """Boolean mask resized to ``(width, height)`` for IoU scoring (area, >half)."""

    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if (binary.shape[1], binary.shape[0]) != (width, height):
        binary = cv2.resize(
            binary, (width, height), interpolation=cv2.INTER_AREA
        )
    return binary > 127


def _render_silhouette(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: Intrinsics,
    width: int,
    height: int,
) -> np.ndarray:
    """Splat hull surface voxels into a small canvas and solidify to a silhouette."""

    u, v, depth = project_points(points, rotation, translation, intrinsics)
    valid = (depth > 1e-9) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    silhouette = np.zeros((height, width), dtype=np.uint8)
    if valid.any():
        col = np.clip(np.rint(u[valid]).astype(np.int64), 0, width - 1)
        row = np.clip(np.rint(v[valid]).astype(np.int64), 0, height - 1)
        silhouette[row, col] = 1
        silhouette = cv2.dilate(silhouette, _DILATE_KERNEL, iterations=2)
        silhouette = cv2.morphologyEx(silhouette, cv2.MORPH_CLOSE, _CLOSE_KERNEL)
    return silhouette > 0


# --------------------------------------------------------------------------- #
# Orbit frame
# --------------------------------------------------------------------------- #
def _plane_normal(centers: np.ndarray) -> np.ndarray:
    """Least-variance (orbit-plane) normal of the camera centres, +Y-oriented.

    The sign is fixed deterministically so the normal points into the positive
    Y hemisphere (with lexicographic tie-breaks on Z then X for the degenerate
    ``ny == 0`` case).
    """

    centered = centers - centers.mean(axis=0)
    # eigh returns ascending eigenvalues; the first eigenvector spans the
    # thinnest direction of the (near-planar) camera cloud.
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    normal = np.asarray(vectors[:, 0], dtype=np.float64)
    flip = normal[1] < 0 or (
        normal[1] == 0
        and (normal[2] < 0 or (normal[2] == 0 and normal[0] < 0))
    )
    if flip:
        normal = -normal
    return normal / max(float(np.linalg.norm(normal)), 1e-12)


def _in_plane_basis(
    reference: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal in-plane axes ``(e1, e2)`` with ``e1`` along ``reference``.

    ``e1 x e2 = normal`` (right-handed), so azimuth measured as
    ``atan2(rel.e2, rel.e1)`` matches the look-at pose built in :func:`_pose`.
    """

    projected = reference - float(reference @ normal) * normal
    length = float(np.linalg.norm(projected))
    if length < 1e-9:  # camera sits on the axis; pick a deterministic in-plane axis
        fallback = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array(
            [0.0, 0.0, 1.0]
        )
        projected = fallback - float(fallback @ normal) * normal
        length = float(np.linalg.norm(projected))
    e1 = projected / length
    e2 = np.cross(normal, e1)
    return e1, e2


def _orbit_coordinates(
    rel: np.ndarray, e1: np.ndarray, e2: np.ndarray, normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row ``(azimuth, elevation, radius)`` of centroid-relative centres ``rel``."""

    x = rel @ e1
    y = rel @ e2
    z = rel @ normal
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.hypot(x, y))
    radius = np.linalg.norm(rel, axis=1)
    return azimuth, elevation, radius


def _pose(
    azimuth: float,
    elevation: float,
    radius: float,
    e1: np.ndarray,
    e2: np.ndarray,
    normal: np.ndarray,
    centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """World->camera ``(R, t)`` looking at ``centroid`` from an orbit sample.

    Mirrors :func:`turntable_pose.camera_pose` (rows ``[right, down, forward]``,
    up = orbit normal, ``t = -R @ camera_centre``) so completed poses share the
    x_cam = R @ X + t / y-down convention with the registered SfM poses.
    """

    in_plane = np.cos(azimuth) * e1 + np.sin(azimuth) * e2
    rel = radius * (np.cos(elevation) * in_plane + np.sin(elevation) * normal)
    center = centroid + rel
    forward = -rel
    forward = forward / max(float(np.linalg.norm(forward)), 1e-12)
    right = np.cross(forward, normal)
    norm_right = float(np.linalg.norm(right))
    if norm_right < 1e-8:  # pragma: no cover - looking straight down the axis
        right = e1.copy()
    else:
        right = right / norm_right
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ center
    return rotation, translation


def _interpolate(
    coordinate: float,
    anchor_t: np.ndarray,
    azimuth: np.ndarray,
    elevation: np.ndarray,
    radius: np.ndarray,
    cap_rad: float,
) -> tuple[float, float, float]:
    """Interpolate/extrapolate ``(az, el, r)`` at temporal ``coordinate``.

    Interior frames blend the two bracketing anchors linearly.  Frames past
    either end hold the outer anchor's elevation/radius and extrapolate azimuth
    with the local angular velocity, capping the azimuth change at ``cap_rad``.
    """

    count = len(anchor_t)
    if coordinate <= anchor_t[0]:
        velocity = (
            (azimuth[1] - azimuth[0]) / (anchor_t[1] - anchor_t[0])
            if count >= 2
            else 0.0
        )
        delta = float(np.clip(velocity * (coordinate - anchor_t[0]), -cap_rad, cap_rad))
        return float(azimuth[0] + delta), float(elevation[0]), float(radius[0])
    if coordinate >= anchor_t[-1]:
        velocity = (
            (azimuth[-1] - azimuth[-2]) / (anchor_t[-1] - anchor_t[-2])
            if count >= 2
            else 0.0
        )
        delta = float(
            np.clip(velocity * (coordinate - anchor_t[-1]), -cap_rad, cap_rad)
        )
        return float(azimuth[-1] + delta), float(elevation[-1]), float(radius[-1])
    upper = int(np.searchsorted(anchor_t, coordinate))
    lower = upper - 1
    span = anchor_t[upper] - anchor_t[lower]
    frac = float((coordinate - anchor_t[lower]) / span) if span > 0 else 0.0
    az = float(azimuth[lower] + frac * (azimuth[upper] - azimuth[lower]))
    el = float(elevation[lower] + frac * (elevation[upper] - elevation[lower]))
    rad = float(radius[lower] + frac * (radius[upper] - radius[lower]))
    return az, el, rad


# --------------------------------------------------------------------------- #
# Per-frame refinement
# --------------------------------------------------------------------------- #
def _refine_pose(
    az0: float,
    el0: float,
    r0: float,
    mask_ds: np.ndarray,
    surface: np.ndarray,
    intrinsics: Intrinsics,
    width: int,
    height: int,
    e1: np.ndarray,
    e2: np.ndarray,
    normal: np.ndarray,
    centroid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    """Powell-refine an orbit sample to maximise mask/hull silhouette IoU.

    Search variables are ``(d_azimuth_deg, d_elevation_deg, log_radius)``, each
    clipped inside the objective (no ``bounds`` argument).  Returns the refined
    ``(R, t, iou, azimuth, elevation, radius)``, never worse than the ``x0`` init.
    """

    def objective(x: np.ndarray) -> float:
        d_az = float(np.clip(x[0], -_D_AZ_DEG, _D_AZ_DEG))
        d_el = float(np.clip(x[1], -_D_EL_DEG, _D_EL_DEG))
        log_r = float(np.clip(x[2], -_LOG_R, _LOG_R))
        rotation, translation = _pose(
            az0 + np.deg2rad(d_az),
            el0 + np.deg2rad(d_el),
            r0 * float(np.exp(log_r)),
            e1, e2, normal, centroid,
        )
        silhouette = _render_silhouette(
            surface, rotation, translation, intrinsics, width, height
        )
        return -_iou(silhouette, mask_ds)

    x0 = np.zeros(3, dtype=np.float64)
    iou0 = -objective(x0)
    result = minimize(
        objective,
        x0,
        method="Powell",
        options={"maxfev": _MAXFEV, "maxiter": _MAXFEV, "xtol": 1e-2, "ftol": 1e-2},
    )
    clipped = np.array(
        [
            float(np.clip(result.x[0], -_D_AZ_DEG, _D_AZ_DEG)),
            float(np.clip(result.x[1], -_D_EL_DEG, _D_EL_DEG)),
            float(np.clip(result.x[2], -_LOG_R, _LOG_R)),
        ]
    )
    az_r = az0 + np.deg2rad(clipped[0])
    el_r = el0 + np.deg2rad(clipped[1])
    r_r = r0 * float(np.exp(clipped[2]))
    rotation, translation = _pose(az_r, el_r, r_r, e1, e2, normal, centroid)
    iou_r = _iou(
        _render_silhouette(surface, rotation, translation, intrinsics, width, height),
        mask_ds,
    )
    if iou_r >= iou0:
        return rotation, translation, iou_r, az_r, el_r, r_r
    rotation0, translation0 = _pose(az0, el0, r0, e1, e2, normal, centroid)
    return rotation0, translation0, iou0, az0, el0, r0


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def complete_poses(
    views: list[dict],
    masks_by_name: dict[str, np.ndarray],
    intrinsics: Intrinsics,
    *,
    points_xyz: np.ndarray | None = None,
    grid: int = 112,
    iterations: int = 2,
    seed: int = 0,
) -> dict:
    """Recover approximate poses for unregistered frames from their silhouettes.

    Parameters
    ----------
    views:
        Registered SfM view dicts (``recon_common.make_view``); each must carry
        ``name``, ``rotation``, ``translation``, ``center`` and ``mask_tight``.
        Names sort in temporal order (``frame_%04d_...ms.jpg``).
    masks_by_name:
        Tight boolean masks for *all* frames (a superset of ``views`` names).
    intrinsics:
        SIMPLE_RADIAL ``(focal, cx, cy, k1)`` shared by every frame.
    grid:
        Hull carve resolution used for the orbit centroid and reprojected surface.
    iterations:
        ``> 1`` re-carves the hull with the accepted silhouette views and retries
        the still-rejected frames against the improved hull.
    seed:
        Recorded for provenance; the routine is fully deterministic regardless.

    Returns
    -------
    dict with keys ``views_all`` (registered views tagged ``pose_source='sfm'``
    plus accepted silhouette views tagged ``pose_source='silhouette'`` with
    ``silhouette_iou`` and ``image_path=None``, all sorted by ``name``),
    ``accepted``, ``rejected`` and a JSON-able ``report``.
    """

    cv2.setNumThreads(1)
    intr = tuple(float(value) for value in intrinsics)

    def _bound_by_points(occ: np.ndarray, bnds) -> np.ndarray:
        """Intersect with the inflated sparse-point hull when points exist.

        Silhouette carving from a partial arc leaves the hull elongated along
        unobserved directions; the point hull restores a realistic depth bound
        so silhouette registration scores against a credible template.
        """

        if points_xyz is None or len(np.asarray(points_xyz)) < 8:
            return occ
        try:
            bound = fusion.point_hull_occupancy(points_xyz, bnds, occ.shape[0])
        except Exception:
            return occ
        merged = occ & bound
        return merged if merged.any() else occ

    registered = sorted(views, key=lambda item: item["name"])
    if len(registered) < 2:
        raise ValueError("complete_poses needs at least two registered views")
    registered_names = {view["name"] for view in registered}
    all_names = sorted(masks_by_name.keys())
    missing_names = [name for name in all_names if name not in registered_names]
    rank = {name: index for index, name in enumerate(all_names)}

    # --- Step 1: coarse hull, centroid, surface voxels ---------------------- #
    # bounds_from_silhouettes only loosely brackets the object; tighten to the
    # frustum intersection so the fine carve yields the object, not a box blob.
    loose_bounds = fusion.bounds_from_silhouettes(registered, intr)
    loose_occ, _ = fusion.carve_hull(
        registered, intr, loose_bounds, resolution=grid
    )
    if not loose_occ.any():
        raise ValueError("registered-view hull carve left an empty grid")
    bounds = _tighten_bounds(loose_occ, loose_bounds, registered, intr)
    occupancy, carve_report = fusion.carve_hull(
        registered, intr, bounds, resolution=grid
    )
    if not occupancy.any():  # tightening was degenerate; keep the loose carve
        bounds, occupancy = loose_bounds, loose_occ
        _, carve_report = fusion.carve_hull(
            registered, intr, bounds, resolution=grid
        )
    occupancy = _bound_by_points(occupancy, bounds)
    centroid = _occupied_world(occupancy, bounds).mean(axis=0)
    surface = _surface_world(occupancy, bounds)

    # --- Step 2: orbit fit from registered camera centres ------------------- #
    centers = np.array(
        [np.asarray(view["center"], dtype=np.float64) for view in registered]
    )
    rel = centers - centroid
    normal = _plane_normal(centers)
    e1, e2 = _in_plane_basis(rel[0], normal)
    azimuth_raw, elevation, radius = _orbit_coordinates(rel, e1, e2, normal)
    # Unwrap over temporal (name) order so interpolation crosses the seam.
    azimuth = np.unwrap(azimuth_raw)
    anchor_t = np.array([float(rank[view["name"]]) for view in registered])

    # --- Scoring canvas + registered-view calibration ----------------------- #
    sample_mask = np.asarray(masks_by_name[all_names[0]])
    full_h, full_w = sample_mask.shape[:2]
    scale = min(1.0, _TARGET_WIDTH / float(full_w))
    score_w = max(int(round(full_w * scale)), 8)
    score_h = max(int(round(full_h * scale)), 8)
    intr_s = (intr[0] * scale, intr[1] * scale, intr[2] * scale, intr[3])

    registered_iou: list[float] = []
    self_iou: dict[str, float] = {}
    for index, view in enumerate(registered):
        mask_ds = _downsample_mask(masks_by_name[view["name"]], score_w, score_h)
        direct = _render_silhouette(
            surface, view["rotation"], view["translation"], intr_s, score_w, score_h
        )
        registered_iou.append(_iou(direct, mask_ds))
        rot_r, trans_r = _pose(
            float(azimuth[index]), float(elevation[index]), float(radius[index]),
            e1, e2, normal, centroid,
        )
        rebuilt = _render_silhouette(surface, rot_r, trans_r, intr_s, score_w, score_h)
        self_iou[view["name"]] = _iou(rebuilt, mask_ds)

    median_registered = float(np.median(registered_iou)) if registered_iou else 0.0
    threshold = max(_ACCEPT_FLOOR, _ACCEPT_FACTOR * median_registered)
    cap_rad = float(np.deg2rad(_EXTRAP_AZ_CAP_DEG))

    # Downsample every missing frame's mask once (reused across iterations).
    missing_mask_ds = {
        name: _downsample_mask(masks_by_name[name], score_w, score_h)
        for name in missing_names
    }

    # --- Steps 3-7: init, refine, accept, iterate --------------------------- #
    results: dict[str, dict[str, Any]] = {}
    current_surface = surface
    passes = max(1, int(iterations))
    for iteration in range(passes):
        if iteration == 0:
            todo = missing_names
        else:
            accepted_now = [
                name for name in missing_names if results[name]["iou"] >= threshold
            ]
            rejected_now = [
                name for name in missing_names if results[name]["iou"] < threshold
            ]
            if not accepted_now or not rejected_now:
                break
            carve_views = list(registered) + [
                {
                    "name": name,
                    "rotation": results[name]["rotation"],
                    "translation": results[name]["translation"],
                    "mask_tight": masks_by_name[name],
                }
                for name in accepted_now
            ]
            occ_iter, _ = fusion.carve_hull(
                carve_views, intr, bounds, resolution=grid
            )
            occ_iter = _bound_by_points(occ_iter, bounds)
            if occ_iter.any():
                current_surface = _surface_world(occ_iter, bounds)
            todo = rejected_now

        for name in todo:
            az0, el0, r0 = _interpolate(
                float(rank[name]), anchor_t, azimuth, elevation, radius, cap_rad
            )
            rotation, translation, iou, az, el, rad = _refine_pose(
                az0, el0, r0, missing_mask_ds[name], current_surface,
                intr_s, score_w, score_h, e1, e2, normal, centroid,
            )
            previous = results.get(name)
            if previous is None or iou > previous["iou"]:
                results[name] = {
                    "rotation": rotation,
                    "translation": translation,
                    "iou": float(iou),
                    "azimuth": float(az),
                    "elevation": float(el),
                    "radius": float(rad),
                }

    accepted_names = sorted(
        name for name in missing_names if results[name]["iou"] >= threshold
    )
    rejected_count = len(missing_names) - len(accepted_names)

    # --- Assemble views_all ------------------------------------------------- #
    views_all: list[dict[str, Any]] = []
    for view in registered:
        tagged = dict(view)
        tagged["pose_source"] = "sfm"
        tagged["self_iou"] = float(self_iou[view["name"]])
        views_all.append(tagged)
    for name in accepted_names:
        entry = results[name]
        rotation = np.asarray(entry["rotation"], dtype=np.float64)
        translation = np.asarray(entry["translation"], dtype=np.float64)
        views_all.append(
            {
                "name": name,
                "image_path": None,
                "rotation": rotation,
                "translation": translation,
                "center": -rotation.T @ translation,
                "mask_tight": masks_by_name[name],
                "mask_eroded": None,
                "pose_source": "silhouette",
                "silhouette_iou": float(entry["iou"]),
            }
        )
    views_all.sort(key=lambda item: item["name"])

    report = {
        "method": (
            "silhouette pose completion: PCA orbit fit of registered cameras, "
            "temporal (az, el, r) interpolation, Powell silhouette-IoU refine; "
            "CPU, deterministic"
        ),
        "grid": int(grid),
        "iterations": int(passes),
        "seed": int(seed),
        "n_registered": len(registered),
        "n_missing": len(missing_names),
        "accepted": len(accepted_names),
        "rejected": int(rejected_count),
        "accept_threshold": round(float(threshold), 6),
        "median_registered_iou": round(median_registered, 6),
        "registered_iou_min": round(float(min(registered_iou)), 6),
        "self_consistency_iou_min": round(float(min(self_iou.values())), 6),
        "self_consistency_iou_median": round(float(np.median(list(self_iou.values()))), 6),
        "score_resolution": [int(score_w), int(score_h)],
        "surface_voxels": int(len(surface)),
        "centroid": [round(float(c), 6) for c in centroid],
        "plane_normal": [round(float(c), 6) for c in normal],
        "azimuth_span_deg": round(float(np.rad2deg(azimuth.max() - azimuth.min())), 4),
        "elevation_deg_mean": round(float(np.rad2deg(np.mean(elevation))), 4),
        "radius_mean": round(float(np.mean(radius)), 6),
        "bounds": [
            [round(float(v), 6) for v in bounds[0]],
            [round(float(v), 6) for v in bounds[1]],
        ],
        "carve": carve_report,
        "per_frame": {
            name: {
                "iou": round(float(results[name]["iou"]), 5),
                "accepted": bool(results[name]["iou"] >= threshold),
                "azimuth_deg": round(float(np.rad2deg(results[name]["azimuth"])), 4),
                "elevation_deg": round(float(np.rad2deg(results[name]["elevation"])), 4),
                "radius": round(float(results[name]["radius"]), 6),
                "source": "silhouette",
            }
            for name in missing_names
        },
    }

    return {
        "views_all": views_all,
        "accepted": len(accepted_names),
        "rejected": int(rejected_count),
        "report": report,
    }
