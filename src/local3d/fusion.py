"""Dense non-convex geometry: silhouette hull as an outer bound + TSDF fusion.

This replaces the convex-hull-only reconstruction stage.  It recovers *inside*
detail (concavities, dents) that a silhouette hull cannot see, by fusing
per-frame monocular depth (see :mod:`local3d.monodepth`) with a Curless-Levoy
truncated signed distance field, and falls back gracefully to the pure hull
where no depth was observed.

Pipeline
--------
1. ``bounds`` — a world-space box, from the sparse SfM points
   (:func:`bounds_from_points`) or, with no points, from a coarse silhouette
   carve (:func:`bounds_from_silhouettes`).
2. ``carve_hull`` — visual-hull occupancy (the outer bound; silhouettes never
   under-estimate the object, so this is a safe envelope).
3. ``fuse_tsdf`` — average the aligned metric depth maps into a TSDF on the
   thin band around the hull surface.
4. ``extract_mesh`` — blend the measured TSDF (where enough views observed a
   voxel) with the hull SDF (everywhere else) and march the zero level set.

Voxel layout (REPLICATED from ``scripts/masked_sfm_hull.py`` so meshes come out
in the same world frame)
-------------------------------------------------------------------------------
For ``bounds = (lower, upper)`` and resolution ``R`` we use
``axes[i] = linspace(lower[i], upper[i], R)`` for ``i in {x, y, z}``.  The
occupancy / field arrays are indexed ``[z, y, x]``: element ``[iz, iy, ix]``
sits at world ``(axes[0][ix], axes[1][iy], axes[2][iz])`` and its flat C-order
index is ``iz * R*R + iy * R + ix``.  :func:`_marching_to_world` reproduces
``masked_sfm_hull.occupancy_to_world_mesh`` exactly (pad by 1, undo, reverse to
xyz, normalise by ``shape - 1``, scale into ``bounds``).

Sign convention (SHARED by both distance fields — verified by the tests)
------------------------------------------------------------------------
Both :func:`hull_sdf` and the TSDF are **positive OUTSIDE the object and
negative INSIDE** (a voxel in free space in front of the measured surface is
positive; a voxel behind the surface is negative).  ``marching_cubes(field,
level=0)`` on either field therefore places the surface at the same crossing,
so the blend in :func:`extract_mesh` is sign-consistent.

Determinism: fixed voxel iteration order, views processed in ``name`` order, no
randomness, memory bounded by chunked per-view projection (never an
all-views x all-voxels array).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.measure import marching_cubes

from local3d.recon_common import Intrinsics, project_points

# Cap on voxels projected at once, to bound peak memory at high resolution.
_CHUNK = 1_000_000


def bounds_from_points(
    points_xyz: np.ndarray, *, pad: float = 0.10
) -> tuple[np.ndarray, np.ndarray]:
    """1st/99th-percentile axis-aligned box around the points, padded by ``pad``."""

    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if len(points) < 2:
        raise ValueError("need at least two points to form a bounds box")
    lower = np.percentile(points, 1, axis=0)
    upper = np.percentile(points, 99, axis=0)
    span = np.maximum(upper - lower, 1e-6)
    margin = pad * span
    return lower - margin, upper + margin


def _axes(bounds: tuple[np.ndarray, np.ndarray], resolution: int) -> list[np.ndarray]:
    lower, upper = bounds
    return [
        np.linspace(float(lower[i]), float(upper[i]), resolution) for i in range(3)
    ]


def _coords_from_flat(
    flat_index: np.ndarray, axes: list[np.ndarray], resolution: int
) -> np.ndarray:
    """World xyz for flat ``[z, y, x]`` C-order indices (matches the voxel layout)."""

    plane = resolution * resolution
    iz = flat_index // plane
    remainder = flat_index - iz * plane
    iy = remainder // resolution
    ix = remainder - iy * resolution
    return np.stack(
        (axes[0][ix], axes[1][iy], axes[2][iz]), axis=1
    ).astype(np.float64)


def point_hull_occupancy(
    points_xyz: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    resolution: int,
    *,
    inflate: float = 1.12,
    chunk: int = 1_000_000,
) -> np.ndarray:
    """Occupancy of the inflated convex hull of the triangulated points.

    Sparse SfM points hug the observed surface, so their (slightly inflated)
    convex hull is a tight outer depth bound that silhouettes alone cannot
    provide when the pose coverage has angular gaps.  Layout matches
    :func:`carve_hull` (``[z, y, x]`` C order).
    """

    from scipy.spatial import Delaunay

    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if len(points) < 8:
        raise ValueError("need at least eight points for a hull bound")
    centroid = points.mean(axis=0)
    inflated = centroid + (points - centroid) * float(inflate)
    hull_vertices = inflated[np.unique(Delaunay(inflated).convex_hull.ravel())]
    triangulation = Delaunay(hull_vertices)

    axes = _axes(bounds, resolution)
    total = resolution**3
    occupancy = np.zeros(total, dtype=bool)
    for start in range(0, total, chunk):
        flat = np.arange(start, min(start + chunk, total))
        world = _coords_from_flat(flat, axes, resolution)
        occupancy[flat] = triangulation.find_simplex(world) >= 0
    return occupancy.reshape((resolution, resolution, resolution))


def _marching_to_world(
    volume: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    level: float,
    pad_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes + the masked_sfm_hull voxel->world mapping.

    ``volume`` is padded with ``pad_value`` (use the exterior value so the
    surface stays closed at the box boundary) before extraction.
    """

    lower, upper = np.asarray(bounds[0], dtype=np.float64), np.asarray(
        bounds[1], dtype=np.float64
    )
    padded = np.pad(
        volume.astype(np.float32), 1, mode="constant", constant_values=float(pad_value)
    )
    vertices_zyx, faces, _normals, _values = marching_cubes(padded, level=level)
    vertices_zyx -= 1.0
    shape = np.asarray(volume.shape, dtype=np.float64)
    normalized = vertices_zyx[:, ::-1] / (shape[::-1] - 1.0)
    world = lower + normalized * (upper - lower)
    return world.astype(np.float32), faces.astype(np.int32)


def carve_hull(
    views: list[dict],
    intrinsics: Intrinsics,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    resolution: int = 256,
    max_view_violations: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Silhouette-carve a ``[z, y, x]`` boolean occupancy grid (the outer bound).

    Each view's ``mask_tight`` votes: a voxel projecting *inside the image but
    outside the mask* accrues one violation; it is carved once its violations
    exceed ``max_view_violations`` (default ``round(0.1 * n_views)``, min 1),
    which tolerates transient occlusion.  Voxels projecting off-image are left
    untouched (a view that cannot see a voxel does not carve it).

    Memory is bounded by chunking the occupied-voxel projection per view; the
    full ``resolution^3`` coordinate array is never materialised.
    """

    ordered = sorted(views, key=lambda item: item["name"])
    if not ordered:
        raise ValueError("carve_hull needs at least one view")
    if max_view_violations is None:
        max_view_violations = max(1, round(0.1 * len(ordered)))

    total = resolution**3
    occupancy_flat = np.ones(total, dtype=bool)
    violations = np.zeros(total, dtype=np.uint16)
    axes = _axes(bounds, resolution)

    for view in ordered:
        mask = np.asarray(view["mask_tight"])
        if mask is None or mask.ndim != 2:
            raise ValueError(f"view {view['name']!r} has no 2D mask_tight")
        binary = mask > 0
        height, width = binary.shape
        rotation = view["rotation"]
        translation = view["translation"]
        occupied_index = np.flatnonzero(occupancy_flat)
        for start in range(0, len(occupied_index), _CHUNK):
            subset = occupied_index[start : start + _CHUNK]
            coords = _coords_from_flat(subset, axes, resolution)
            u, v, depth = project_points(coords, rotation, translation, intrinsics)
            inside = (
                (depth > 1e-9)
                & (u >= 0)
                & (u < width)
                & (v >= 0)
                & (v < height)
            )
            fails = np.zeros(len(subset), dtype=bool)
            ui = np.rint(u[inside]).astype(np.int64)
            vi = np.rint(v[inside]).astype(np.int64)
            ui = np.clip(ui, 0, width - 1)
            vi = np.clip(vi, 0, height - 1)
            fails[inside] = ~binary[vi, ui]
            failed = subset[fails]
            violations[failed] += 1
            occupancy_flat[failed[violations[failed] > max_view_violations]] = False

    occupancy = occupancy_flat.reshape((resolution, resolution, resolution))
    report = {
        "resolution": int(resolution),
        "views": len(ordered),
        "max_view_violations": int(max_view_violations),
        "occupied_voxels": int(occupancy.sum()),
    }
    return occupancy, report


def bounds_from_silhouettes(
    views: list[dict],
    intrinsics: Intrinsics,
    *,
    initial_halfwidth_scale: float = 3.0,
    coarse: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Bounds box for the no-sparse-points path via a coarse silhouette carve.

    A generous cube is centred on the least-squares intersection of the camera
    optical axes (the ``look-at`` point); its half-width is
    ``initial_halfwidth_scale * mean_camera_distance * (diag / 2) / focal``
    (the frame's world half-diagonal at object distance, scaled up).  The cube
    is carved at ``coarse`` resolution and a tight, 10%-padded box is returned
    around the surviving voxels — falling back to the generous cube if the
    carve leaves nothing.
    """

    ordered = sorted(views, key=lambda item: item["name"])
    if not ordered:
        raise ValueError("bounds_from_silhouettes needs at least one view")
    focal = float(intrinsics[0])

    centers = np.array([np.asarray(v["center"], dtype=np.float64) for v in ordered])
    forwards = np.array(
        [np.asarray(v["rotation"], dtype=np.float64)[2, :] for v in ordered]
    )
    forwards /= np.maximum(np.linalg.norm(forwards, axis=1, keepdims=True), 1e-12)

    # Least-squares closest point to every camera axis (ray p_i + s * d_i).
    accum_a = np.zeros((3, 3), dtype=np.float64)
    accum_b = np.zeros(3, dtype=np.float64)
    identity = np.eye(3)
    for center, forward in zip(centers, forwards):
        projector = identity - np.outer(forward, forward)
        accum_a += projector
        accum_b += projector @ center
    try:
        look_at = np.linalg.solve(accum_a, accum_b)
    except np.linalg.LinAlgError:
        look_at = centers.mean(axis=0)

    mean_distance = float(np.mean(np.linalg.norm(centers - look_at, axis=1)))
    mask0 = np.asarray(ordered[0]["mask_tight"])
    height, width = mask0.shape[:2]
    diag = float(np.hypot(width, height))
    halfwidth = initial_halfwidth_scale * mean_distance * (0.5 * diag) / max(focal, 1e-9)
    halfwidth = max(halfwidth, 1e-6)

    generous = (look_at - halfwidth, look_at + halfwidth)
    occupancy, _ = carve_hull(ordered, intrinsics, generous, resolution=coarse)
    if not occupancy.any():
        return generous

    axes = _axes(generous, coarse)
    occupied_index = np.flatnonzero(occupancy.ravel())
    coords = _coords_from_flat(occupied_index, axes, coarse)
    lower = coords.min(axis=0)
    upper = coords.max(axis=0)
    span = np.maximum(upper - lower, 1e-6)
    margin = 0.10 * span
    return lower - margin, upper + margin


def hull_sdf(occupancy: np.ndarray, *, trunc_voxels: float = 4.0) -> np.ndarray:
    """Signed distance to the hull surface in voxels: OUTSIDE +, INSIDE -.

    Computed from two Euclidean distance transforms and truncated to
    ``[-1, 1]`` after dividing by ``trunc_voxels`` (so ``+1`` is
    ``trunc_voxels`` voxels outside, ``-1`` the same distance inside).  The
    sign convention matches the TSDF from :func:`fuse_tsdf`.
    """

    occ = np.asarray(occupancy, dtype=bool)
    distance_outside = distance_transform_edt(~occ)
    distance_inside = distance_transform_edt(occ)
    signed = distance_outside - distance_inside
    scaled = signed / max(float(trunc_voxels), 1e-9)
    return np.clip(scaled, -1.0, 1.0).astype(np.float32)


def _edge_mask(depth: np.ndarray, extent: float, thresh: float) -> np.ndarray:
    """Depth-discontinuity mask: |grad(depth)| / object_extent > ``thresh``."""

    import cv2

    grad_x = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y) / max(extent, 1e-9)
    return magnitude > thresh


def fuse_tsdf(
    views: list[dict],
    intrinsics: Intrinsics,
    depths: list[np.ndarray],
    occupancy: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    trunc_voxels: float = 4.0,
    min_weight: float = 1.5,
    edge_grad_thresh: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Curless-Levoy TSDF over the hull band from aligned metric depth maps.

    Only the band ``binary_dilation(occupancy, iters=2)`` is integrated.  For
    each band voxel and view, ``sdf = (measured_depth - voxel_depth) / mu`` with
    ``mu = trunc_voxels * voxel_size``; samples strictly behind the surface
    (``sdf < -1``, i.e. occluded) are skipped and the rest clipped to
    ``[-1, 1]``.  Sign: ``+`` in front of the surface (outside), ``-`` behind
    (inside), matching :func:`hull_sdf`.  Samples landing within
    ``edge_grad_thresh`` of a depth discontinuity are discarded.  Voxel weight
    is ``frame_conf / voxel_depth`` (``frame_conf`` from ``view['depth_conf']``,
    default 1.0).

    Returns ``(tsdf, weight)`` as ``[z, y, x]`` float32 grids (weight 0 and
    tsdf 0 where nothing was observed).
    """

    ordered = sorted(range(len(views)), key=lambda i: views[i]["name"])
    resolution = int(occupancy.shape[0])
    total = resolution**3
    lower, upper = np.asarray(bounds[0], dtype=np.float64), np.asarray(
        bounds[1], dtype=np.float64
    )
    voxel_size = float(np.mean((upper - lower) / max(resolution - 1, 1)))
    mu = max(float(trunc_voxels) * voxel_size, 1e-9)
    extent = float(np.linalg.norm(upper - lower))

    band = binary_dilation(np.asarray(occupancy, dtype=bool), iterations=2)
    band_index = np.flatnonzero(band.ravel())
    axes = _axes(bounds, resolution)
    band_coords = _coords_from_flat(band_index, axes, resolution)
    n_band = len(band_index)

    tsdf_sum = np.zeros(n_band, dtype=np.float64)
    weight_sum = np.zeros(n_band, dtype=np.float64)

    for i in ordered:
        view = views[i]
        depth_map = np.asarray(depths[i], dtype=np.float32)
        height, width = depth_map.shape[:2]
        conf = float(view.get("depth_conf", 1.0))
        confidence_map = view.get("depth_confidence_map")
        if confidence_map is not None:
            confidence_map = np.asarray(confidence_map, dtype=np.float32)
            if confidence_map.shape != depth_map.shape:
                raise ValueError(
                    f"depth confidence shape {confidence_map.shape} does not match "
                    f"depth shape {depth_map.shape} for {view['name']}"
                )
        edges = _edge_mask(depth_map, extent, edge_grad_thresh)

        u, v, voxel_depth = project_points(
            band_coords, view["rotation"], view["translation"], intrinsics
        )
        inside = (
            (voxel_depth > 1e-9)
            & (u >= 0)
            & (u < width)
            & (v >= 0)
            & (v < height)
        )
        ui = np.clip(np.rint(u), 0, width - 1).astype(np.int64)
        vi = np.clip(np.rint(v), 0, height - 1).astype(np.int64)
        measured = np.zeros(n_band, dtype=np.float64)
        on_edge = np.zeros(n_band, dtype=bool)
        measured[inside] = depth_map[vi[inside], ui[inside]]
        on_edge[inside] = edges[vi[inside], ui[inside]]

        valid = inside & (measured > 1e-6) & ~on_edge
        sdf = np.zeros(n_band, dtype=np.float64)
        sdf[valid] = (measured[valid] - voxel_depth[valid]) / mu
        keep = valid & (sdf >= -1.0)
        if not keep.any():
            continue
        sdf_clipped = np.clip(sdf, -1.0, 1.0)
        if confidence_map is None:
            sample_confidence = np.full(n_band, conf, dtype=np.float64)
        else:
            sample_confidence = np.zeros(n_band, dtype=np.float64)
            sample_confidence[inside] = confidence_map[vi[inside], ui[inside]]
            sample_confidence *= conf
        weight = sample_confidence / np.maximum(voxel_depth, 1e-6)
        tsdf_sum[keep] += weight[keep] * sdf_clipped[keep]
        weight_sum[keep] += weight[keep]

    tsdf_band = np.zeros(n_band, dtype=np.float32)
    observed = weight_sum > 0
    tsdf_band[observed] = (tsdf_sum[observed] / weight_sum[observed]).astype(np.float32)

    tsdf = np.zeros(total, dtype=np.float32)
    weight = np.zeros(total, dtype=np.float32)
    tsdf[band_index] = tsdf_band
    weight[band_index] = weight_sum.astype(np.float32)
    return (
        tsdf.reshape((resolution, resolution, resolution)),
        weight.reshape((resolution, resolution, resolution)),
    )


def extract_mesh(
    tsdf: np.ndarray,
    weight: np.ndarray,
    occupancy: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray],
    *,
    min_weight: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Blend the measured TSDF with the hull SDF and march the zero level set.

    ``wnorm = clip(weight / min_weight, 0, 1)`` mixes the observed TSDF (fully
    trusted at ``weight >= min_weight``) with :func:`hull_sdf` (the outer bound,
    used where nothing was observed): ``field = wnorm * tsdf + (1 - wnorm) *
    hull_sdf``.  Both terms share the OUTSIDE-positive / INSIDE-negative
    convention, so ``marching_cubes(field, level=0)`` is well posed.
    """

    from scipy import ndimage

    occ = np.asarray(occupancy, dtype=bool)
    # Close small carving tunnels before the SDF so the blended field cannot
    # produce spongy topology at the observed/unobserved transition.
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3, 3), bool), iterations=1)
    tsdf = np.asarray(tsdf, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    hull = hull_sdf(occ)
    wnorm = np.clip(weight / max(float(min_weight), 1e-9), 0.0, 1.0)
    field = wnorm * tsdf + (1.0 - wnorm) * hull
    field = ndimage.gaussian_filter(field, sigma=1.0).astype(np.float32)

    vertices, faces = _marching_to_world(field, bounds, level=0.0, pad_value=1.0)

    observed = weight >= float(min_weight)
    occupied_count = int(occ.sum())
    supported_fraction = float(
        np.count_nonzero(observed & occ) / max(occupied_count, 1)
    )
    report = {
        # Legacy key retained for report compatibility. This support comes from
        # a monocular prediction aligned to sparse SfM, not measured depth.
        "observed_voxel_fraction": supported_fraction,
        "predicted_depth_supported_voxel_fraction": supported_fraction,
        "support_semantics": "monocular predicted depth aligned to sparse SfM; not sensor-measured geometry",
        "observed_band_voxels": int(np.count_nonzero(observed)),
        "occupied_voxels": occupied_count,
        "vertices": int(len(vertices)),
        "triangles": int(len(faces)),
    }
    return vertices, faces, report


def reconstruct_geometry(
    views: list[dict],
    intrinsics: Intrinsics,
    points_xyz: np.ndarray | None,
    depth_model_path: Any | None,
    *,
    carve_views: list[dict] | None = None,
    resolution: int = 256,
    target_triangles: int = 20000,
    depth_threads: int = 4,
    intersect_point_hull: bool = True,
    use_precomputed_depths: bool = False,
) -> dict:
    """Orchestrate hull carving + optional monocular-depth TSDF fusion.

    With ``depth_model_path`` and ``points_xyz`` both provided, a
    :class:`~local3d.monodepth.DepthEstimator` predicts per-frame disparity,
    aligns it to the projected sparse points (inside the eroded mask, depth > 0)
    and fuses the frames that pass the alignment gate; others are excluded and
    counted in ``report['depth_frames_rejected']``.  Otherwise the hull
    occupancy is meshed directly.

    Cleanup: largest connected component -> fill holes -> Taubin smoothing ->
    optional decimation (kept undecimated if the mesh extra is unavailable).
    Returns ``{'vertices', 'faces', 'report'}``.
    """

    import cv2
    import trimesh

    from local3d.visual_hull import taubin_smooth

    if points_xyz is not None and len(np.asarray(points_xyz)) >= 2:
        bounds = bounds_from_points(points_xyz)
        bounds_source = "points"
    else:
        bounds = bounds_from_silhouettes(views, intrinsics)
        bounds_source = "silhouettes"

    silhouette_views = carve_views if carve_views else views
    occupancy, carve_report = carve_hull(
        silhouette_views, intrinsics, bounds, resolution=resolution
    )
    if intersect_point_hull and points_xyz is not None and len(np.asarray(points_xyz)) >= 8:
        try:
            point_bound = point_hull_occupancy(points_xyz, bounds, resolution)
            before = int(occupancy.sum())
            occupancy &= point_bound
            carve_report["point_hull_intersection"] = {
                "before_voxels": before,
                "after_voxels": int(occupancy.sum()),
            }
        except Exception as exc:  # hull degenerate -> keep silhouette-only
            carve_report["point_hull_intersection"] = {"skipped": str(exc)}
    elif points_xyz is not None and len(np.asarray(points_xyz)) >= 8:
        carve_report["point_hull_intersection"] = {
            "skipped": "disabled: sparse points constrain bounds but are not a hard surface hull",
        }

    report: dict[str, Any] = {
        "bounds_source": bounds_source,
        "carve": carve_report,
        "depth_frames_used": 0,
        "depth_frames_rejected": 0,
    }

    fused = False
    precomputed_views = (
        [
            view
            for view in views
            if view.get("depth_map") is not None
            and view.get("evaluation_role", "reconstruction") != "holdout"
        ]
        if use_precomputed_depths
        else []
    )
    can_fuse = bool(precomputed_views) or (
        depth_model_path is not None and points_xyz is not None
    )
    if can_fuse and occupancy.any():
        from local3d.monodepth import (
            DepthEstimator,
            align_disparity_to_points,
            disparity_to_depth,
        )

        fusion_views: list[dict] = []
        fusion_depths: list[np.ndarray] = []
        alignments: list[dict] = []
        if precomputed_views:
            for view in sorted(precomputed_views, key=lambda item: item["name"]):
                metric = np.asarray(view["depth_map"], dtype=np.float32)
                if metric.ndim != 2 or not np.any(np.isfinite(metric) & (metric > 0)):
                    report["depth_frames_rejected"] += 1
                    continue
                masked = np.where(np.isfinite(metric) & (metric > 0), metric, 0.0)
                if view.get("mask_tight") is not None:
                    masked = masked * (np.asarray(view["mask_tight"]) > 0)
                fused_view = dict(view)
                fused_view.setdefault("depth_conf", 1.0)
                fusion_views.append(fused_view)
                fusion_depths.append(masked.astype(np.float32))
            report["depth_source"] = "precomputed evidence-gated aligned maps"
            report["depth_backend"] = sorted(
                {
                    str(view.get("depth_backend", "unknown"))
                    for view in fusion_views
                }
            )
        else:
            estimator = DepthEstimator(depth_model_path, threads=depth_threads)
            points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
            for view in sorted(views, key=lambda item: item["name"]):
                image = cv2.imread(str(view["image_path"]), cv2.IMREAD_COLOR)
                if image is None:
                    report["depth_frames_rejected"] += 1
                    continue
                observations = view.get("sfm_observations")
                if observations is not None:
                    u = np.asarray(observations.get("xy", []), dtype=np.float64).reshape(-1, 2)[:, 0]
                    v = np.asarray(observations.get("xy", []), dtype=np.float64).reshape(-1, 2)[:, 1]
                    depth = np.asarray(observations.get("z_camera", []), dtype=np.float64)
                else:
                    # Compatibility only. New SfM results always carry actual
                    # per-image track observations; global projection is not
                    # independent visibility evidence.
                    u, v, depth = project_points(
                        points, view["rotation"], view["translation"], intrinsics
                    )
                height, width = image.shape[:2]
                eroded = view.get("mask_eroded")
                keep = (depth > 1e-6) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
                if eroded is not None:
                    eroded = np.asarray(eroded) > 0
                    ui = np.clip(np.rint(u), 0, width - 1).astype(np.int64)
                    vi = np.clip(np.rint(v), 0, height - 1).astype(np.int64)
                    on_object = np.zeros(len(depth), dtype=bool)
                    on_object[keep] = eroded[vi[keep], ui[keep]]
                    keep = keep & on_object
                disp = estimator.disparity(image)
                alignment = align_disparity_to_points(disp, u[keep], v[keep], depth[keep])
                if not alignment["ok"]:
                    report["depth_frames_rejected"] += 1
                    continue
                metric = disparity_to_depth(disp, alignment["a"], alignment["b"])
                if view.get("mask_tight") is not None:
                    metric = metric * (np.asarray(view["mask_tight"]) > 0)
                fused_view = dict(view)
                fused_view.setdefault("depth_conf", 1.0)
                # Preserve the aligned map for the independent delivery gate.
                view["depth_map"] = metric.astype(np.float32)
                view["depth_backend"] = "depth_anything_legacy"
                fusion_views.append(fused_view)
                fusion_depths.append(metric.astype(np.float32))
                alignments.append(alignment)
            report["depth_source"] = (
                "legacy per-frame Depth Anything alignment; actual image-track "
                "observations used when available"
            )

        report["depth_frames_used"] = len(fusion_views)
        if fusion_views:
            tsdf, weight = fuse_tsdf(
                fusion_views, intrinsics, fusion_depths, occupancy, bounds
            )
            vertices, faces, extract_report = extract_mesh(
                tsdf, weight, occupancy, bounds
            )
            report["extract"] = extract_report
            report["depth_rms_rel"] = [round(a["rms_rel"], 5) for a in alignments]
            fused = True

    if not fused:
        if not occupancy.any():
            raise ValueError("hull carve left an empty occupancy grid")
        vertices, faces = _marching_to_world(
            occupancy.astype(np.uint8), bounds, level=0.5, pad_value=0.0
        )

    report["mode"] = "tsdf_fused" if fused else "hull_only"

    vertices, faces, cleanup = _cleanup_mesh(
        vertices, faces, target_triangles=target_triangles, trimesh=trimesh,
        taubin_smooth=taubin_smooth,
    )
    report["cleanup"] = cleanup
    report["triangles"] = int(len(faces))
    return {"vertices": vertices, "faces": faces, "report": report}


def _cleanup_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_triangles: int,
    trimesh: Any,
    taubin_smooth: Any,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cleanup: dict[str, Any] = {"steps": []}
    mesh = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda part: (len(part.faces), len(part.vertices)))
        cleanup["steps"].append("largest_component")
    trimesh.repair.fill_holes(mesh)
    cleanup["steps"].append("fill_holes")

    smoothed = taubin_smooth(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        iterations=6,
    )
    out_vertices = smoothed.astype(np.float32)
    out_faces = np.asarray(mesh.faces, dtype=np.int32)
    cleanup["steps"].append("taubin_smooth")

    if target_triangles:
        try:
            from local3d.mesh_post import postprocess

            out_vertices, out_faces, post = postprocess(
                out_vertices, out_faces, target_triangles=target_triangles
            )
            cleanup["postprocess"] = post
        except (ImportError, RuntimeError, ValueError) as exc:  # keep undecimated
            cleanup["postprocess_error"] = str(exc)

    # Decimation + welding can leave detached slivers; deliver one body.
    final = trimesh.Trimesh(
        vertices=np.asarray(out_vertices, dtype=np.float64),
        faces=np.asarray(out_faces, dtype=np.int64),
        process=False,
    )
    parts = final.split(only_watertight=False)
    if len(parts) > 1:
        final = max(parts, key=lambda part: (len(part.faces), len(part.vertices)))
        out_vertices = np.asarray(final.vertices, dtype=np.float32)
        out_faces = np.asarray(final.faces, dtype=np.int32)
        cleanup["steps"].append(f"final_largest_component_of_{len(parts)}")

    return out_vertices, out_faces, cleanup
