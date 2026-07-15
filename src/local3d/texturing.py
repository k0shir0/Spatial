"""Ghost-free texture atlas by per-face single-best-view assignment.

Instead of blending every source view into each texel (the median-projection
approach of ``scripts/bake_texture.py``, which smears fingers, highlights and
mis-registration into ghosts), this module assigns **one** source view to each
triangle — a small labelling problem in the Waechter/Lempitsky "Let There Be
Color!" / ``mvs-texturing`` tradition — then copies contiguous chart patches
straight from that one view.  Every face's texels therefore come from a single
camera: no cross-view averaging, no ghosts.

Pipeline (all deterministic, CPU-only, no ``xatlas``):

1. Per (face, view) hard gates: z-buffer visibility, frontality, all three
   vertices projecting inside the image and inside ``mask_tight``, and a
   minimum distance from the tight-mask boundary (via ``cv2.distanceTransform``)
   so silhouette bleed never enters the atlas.
2. A data quality ``q(f, v) = area_px * (mean_gradient + 1) * boundary_term``,
   keeping the best ``K = 8`` views per face.  The ``+1`` gradient floor keeps
   ``q`` informative on flat, texture-free surfaces (there frontal, close,
   well-inside-the-mask views win) instead of collapsing to zero — a small,
   documented departure from a strict product.
3. Label init by argmax ``q``, refined by Iterated Conditional Modes with a
   Potts smoothness ``lambda * (# differently-labelled edge-neighbours)``.
4. Faces no view could observe get label ``-1`` (unobserved).
5. Charts = connected components of equal-label adjacent faces.  Each observed
   chart is copied — masked to exactly its own faces via a full-resolution
   face-index render, so neighbouring charts never leak in — into a
   shelf-packed atlas rectangle with a dilation-filled gutter.
6. Unobserved charts get a small rectangle filled, after seam levelling, with
   the mean colour of the geodesically nearest observed faces.
7. Seam levelling (when ``seam_level``): a per-chart additive colour offset per
   channel solved by least squares over cross-chart seam-edge colour
   differences (``scipy.sparse.linalg.lsqr``), then a 3 px feather.  This is the
   contract's explicitly-permitted simplification of per-vertex correction; it
   is reported in ``report['seam_leveling']``.

Honest limits: within-face self-occlusion is not modelled (faces are gated as
whole triangles); the per-chart offset cannot fix intra-chart shading
gradients; UV packing is a plain shelf packer, not an optimal bin-packer; and
cross-architecture bit-identity is not promised (single-platform reproducibility
is).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

from .recon_common import (
    Intrinsics,
    bilinear_sample,
    face_normals,
    project_points,
    rasterize_zbuffer,
    visible_faces,
)

_TOP_K = 8
_GUTTER = 4
_UNOBSERVED_INNER = 16
_SEAM_GAMMA = 0.1
_KERNEL3 = np.ones((3, 3), np.uint8)


def _load_bgr(view: dict[str, Any], width: int, height: int) -> np.ndarray:
    """Load one source frame as BGR uint8, resized to the mask geometry."""

    image = cv2.imread(str(Path(view["image_path"])), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read source frame: {view['image_path']}")
    if image.shape[0] != height or image.shape[1] != width:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return image


def _build_topology(
    faces: np.ndarray,
) -> tuple[dict[tuple[int, int], list[int]], list[np.ndarray]]:
    """Edge -> incident faces, and per-face edge-neighbour arrays."""

    face_count = len(faces)
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for f in range(face_count):
        a, b, c = int(faces[f, 0]), int(faces[f, 1]), int(faces[f, 2])
        for i, j in ((a, b), (b, c), (c, a)):
            edge_faces[(i, j) if i < j else (j, i)].append(f)
    raw: list[set[int]] = [set() for _ in range(face_count)]
    for incident in edge_faces.values():
        if len(incident) >= 2:
            for i in range(len(incident)):
                for j in range(i + 1, len(incident)):
                    raw[incident[i]].add(incident[j])
                    raw[incident[j]].add(incident[i])
    neighbours = [np.array(sorted(group), dtype=np.int64) for group in raw]
    return edge_faces, neighbours


def _score_views(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    centroids: np.ndarray,
    views: list[dict[str, Any]],
    intrinsics: Intrinsics,
    view_hw: list[tuple[int, int]],
    *,
    min_frontality: float,
    boundary_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-face top-``K`` candidate view indices and their quality.

    One pass over the views; each image is loaded exactly once (for its
    half-scale Scharr magnitude) and released before the next.
    """

    face_count = len(faces)
    cand_view = np.full((face_count, _TOP_K), -1, dtype=np.int32)
    cand_q = np.full((face_count, _TOP_K), -np.inf, dtype=np.float64)
    gate_stats = {
        "pairs": 0,
        "visible": 0,
        "frontality": 0,
        "inside_image": 0,
        "inside_mask": 0,
        "boundary": 0,
        "passed": 0,
    }

    for view_index, view in enumerate(views):
        height, width = view_hw[view_index]
        mask = view.get("mask_tight")
        if mask is None:
            mask_bool = np.ones((height, width), dtype=bool)
        else:
            mask_bool = np.asarray(mask).astype(bool)
        rotation = view["rotation"]
        translation = view["translation"]
        center = view.get("center")
        if center is None:
            center = -np.asarray(rotation).T @ np.asarray(translation)
        center = np.asarray(center, dtype=np.float64)

        distance = cv2.distanceTransform(mask_bool.astype(np.uint8), cv2.DIST_L2, 5)

        visible = visible_faces(
            vertices, faces, view, intrinsics, width, height, scale=0.5
        )

        image = _load_bgr(view, width, height)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        half = cv2.resize(
            gray,
            (max(1, round(width * 0.5)), max(1, round(height * 0.5))),
            interpolation=cv2.INTER_AREA,
        )
        grad = cv2.magnitude(
            cv2.Scharr(half, cv2.CV_32F, 1, 0), cv2.Scharr(half, cv2.CV_32F, 0, 1)
        )
        sx = grad.shape[1] / float(width)
        sy = grad.shape[0] / float(height)

        u, v, depth = project_points(vertices, rotation, translation, intrinsics)
        fu = u[faces]
        fv = v[faces]
        fdepth = depth[faces]
        cu, cvv, _ = project_points(centroids, rotation, translation, intrinsics)

        inside = (
            (fdepth > 1e-9)
            & (fu >= 0)
            & (fu <= width - 1)
            & (fv >= 0)
            & (fv <= height - 1)
        )
        ui = np.clip(np.rint(fu).astype(np.int64), 0, width - 1)
        vi = np.clip(np.rint(fv).astype(np.int64), 0, height - 1)
        in_mask = mask_bool[vi, ui]
        boundary = distance[vi, ui]
        min_boundary = boundary.min(axis=1)

        direction = center - centroids
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        frontality = np.einsum("ij,ij->i", normals, direction)

        gate = (
            visible
            & (frontality > min_frontality)
            & inside.all(axis=1)
            & in_mask.all(axis=1)
            & (min_boundary > boundary_px)
        )
        gate_stats["pairs"] += face_count
        gate_stats["visible"] += int(visible.sum())
        gate_stats["frontality"] += int((frontality > min_frontality).sum())
        gate_stats["inside_image"] += int(inside.all(axis=1).sum())
        gate_stats["inside_mask"] += int(in_mask.all(axis=1).sum())
        gate_stats["boundary"] += int((min_boundary > boundary_px).sum())
        gate_stats["passed"] += int(gate.sum())

        area = 0.5 * np.abs(
            (fu[:, 1] - fu[:, 0]) * (fv[:, 2] - fv[:, 0])
            - (fu[:, 2] - fu[:, 0]) * (fv[:, 1] - fv[:, 0])
        )
        sample_u = np.stack([cu, fu[:, 0], fu[:, 1], fu[:, 2]], axis=1).ravel() * sx
        sample_v = np.stack([cvv, fv[:, 0], fv[:, 1], fv[:, 2]], axis=1).ravel() * sy
        mean_grad = bilinear_sample(grad, sample_u, sample_v).reshape(face_count, 4)
        mean_grad = mean_grad.mean(axis=1)
        boundary_term = np.minimum(1.0, min_boundary / (3.0 * boundary_px))
        quality = area * (mean_grad + 1.0) * boundary_term

        rows = np.flatnonzero(gate)
        if not len(rows):
            continue
        slot = np.argmin(cand_q[rows], axis=1)
        better = quality[rows] > cand_q[rows, slot]
        winners = rows[better]
        slots = slot[better]
        cand_q[winners, slots] = quality[winners]
        cand_view[winners, slots] = view_index

    return cand_view, cand_q, gate_stats


def _label_faces(
    cand_view: np.ndarray,
    cand_q: np.ndarray,
    neighbours: list[np.ndarray],
    *,
    smooth_lambda: float | None,
    icm_sweeps: int,
) -> tuple[np.ndarray, float]:
    """Init labels by argmax quality, then refine with Potts-smoothed ICM."""

    face_count = len(cand_view)
    face_rows = np.arange(face_count)
    has_cand = np.isfinite(cand_q).any(axis=1)
    best_slot = np.argmax(cand_q, axis=1)
    label = np.where(has_cand, cand_view[face_rows, best_slot], -1).astype(np.int64)

    if smooth_lambda is not None:
        lam = float(smooth_lambda)
    elif has_cand.any():
        chosen_q = cand_q[face_rows, best_slot][has_cand]
        lam = 0.15 * float(np.median(chosen_q))
    else:
        lam = 0.0

    if lam > 0.0:
        for _ in range(icm_sweeps):
            for f in range(face_count):
                if not has_cand[f]:
                    continue
                slots = np.flatnonzero(np.isfinite(cand_q[f]))
                order = slots[np.argsort(cand_view[f, slots], kind="stable")]
                nbr = neighbours[f]
                nbr_labels = label[nbr] if len(nbr) else label[:0]
                best_cost = np.inf
                best_label = label[f]
                for slot in order:
                    view_index = int(cand_view[f, slot])
                    data = -float(cand_q[f, slot])
                    smooth = lam * float(np.count_nonzero(nbr_labels != view_index))
                    cost = data + smooth
                    if cost < best_cost:
                        best_cost = cost
                        best_label = view_index
                label[f] = best_label

    return label, lam


def _connected_charts(
    label: np.ndarray, neighbours: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Group faces into charts of equal-label edge-connected components."""

    face_count = len(label)
    comp = np.full(face_count, -1, dtype=np.int64)
    chart_labels: list[int] = []
    chart_id = 0
    for seed in range(face_count):
        if comp[seed] >= 0:
            continue
        this_label = label[seed]
        comp[seed] = chart_id
        stack = [seed]
        while stack:
            f = stack.pop()
            for nbr in neighbours[f]:
                if comp[nbr] < 0 and label[nbr] == this_label:
                    comp[nbr] = chart_id
                    stack.append(int(nbr))
        chart_labels.append(int(this_label))
        chart_id += 1
    return comp, np.array(chart_labels, dtype=np.int64)


def _pca_layout(
    points: np.ndarray, x0: int, y0: int, inner_w: int, inner_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Spread 3D points across a rectangle interior via their two main axes."""

    count = len(points)
    if count == 0:
        return np.zeros(0), np.zeros(0)
    centered = points - points.mean(axis=0)
    if count >= 3 and np.any(np.abs(centered) > 1e-9):
        _, _, basis = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ basis[:2].T
    else:
        coords = centered[:, :2]
    out = np.zeros((count, 2))
    for axis in range(2):
        column = coords[:, axis]
        span = column.max() - column.min()
        out[:, axis] = 0.5 if span < 1e-9 else (column - column.min()) / span
    col = x0 + _GUTTER + out[:, 0] * max(inner_w - 1, 0)
    row = y0 + _GUTTER + out[:, 1] * max(inner_h - 1, 0)
    return col, row


def _rect_dims(chart: dict[str, Any], scale: float) -> tuple[int, int, int, int]:
    """Return (rect_w, rect_h, inner_w, inner_h) for a chart at a global scale."""

    if chart["label"] >= 0:
        inner_w = max(1, round(chart["src_w"] * chart["s_chart"] * scale))
        inner_h = max(1, round(chart["src_h"] * chart["s_chart"] * scale))
    else:
        inner_w = inner_h = max(4, round(_UNOBSERVED_INNER * scale))
    return inner_w + 2 * _GUTTER, inner_h + 2 * _GUTTER, inner_w, inner_h


def _shelf_pack(
    charts: list[dict[str, Any]],
    min_face: dict[int, int],
    scale: float,
    atlas_size: int,
) -> dict[int, tuple[int, int, int, int, int, int]] | None:
    """Deterministic descending-height shelf packing; None on overflow."""

    dims = {c["id"]: _rect_dims(c, scale) for c in charts}
    order = sorted(
        (c["id"] for c in charts),
        key=lambda cid: (-dims[cid][1], -dims[cid][0], min_face[cid]),
    )
    placement: dict[int, tuple[int, int, int, int, int, int]] = {}
    x = 0
    y = 0
    shelf_h = 0
    for cid in order:
        rect_w, rect_h, inner_w, inner_h = dims[cid]
        if rect_w > atlas_size or rect_h > atlas_size:
            return None
        if x + rect_w > atlas_size:
            y += shelf_h
            x = 0
            shelf_h = 0
        if y + rect_h > atlas_size:
            return None
        placement[cid] = (x, y, rect_w, rect_h, inner_w, inner_h)
        x += rect_w
        shelf_h = max(shelf_h, rect_h)
    return placement


def bake_texture_atlas(
    vertices: np.ndarray,
    faces: np.ndarray,
    views: list[dict[str, Any]],
    intrinsics: Intrinsics,
    *,
    atlas_size: int = 2048,
    smooth_lambda: float | None = None,
    icm_sweeps: int = 6,
    min_frontality: float = 0.25,
    boundary_px: float = 12.0,
    max_chart_px: int = 1024,
    seam_level: bool = True,
) -> dict[str, Any]:
    """Bake a ghost-free single-best-view texture atlas.

    See the module docstring for the algorithm.  Returns a dict with
    ``vertices`` (float32, duplicated per chart), ``faces`` (int32, reindexed,
    in the original face order), ``uvs`` (float32 in ``[0, 1]``, GL v-up
    convention), ``texture_bgr`` (uint8 atlas) and a ``report`` dict.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    face_count = len(faces)
    normals = face_normals(vertices, faces)
    centroids = vertices[faces].mean(axis=1)

    view_hw: list[tuple[int, int]] = []
    for view in views:
        mask = view.get("mask_tight")
        if mask is not None:
            view_hw.append((int(mask.shape[0]), int(mask.shape[1])))
        else:
            probe = _load_bgr(view, 1, 1)
            view_hw.append((probe.shape[0], probe.shape[1]))

    cand_view, cand_q, gate_stats = _score_views(
        vertices,
        faces,
        normals,
        centroids,
        views,
        intrinsics,
        view_hw,
        min_frontality=min_frontality,
        boundary_px=boundary_px,
    )

    edge_faces, neighbours = _build_topology(faces)
    label, lam = _label_faces(
        cand_view,
        cand_q,
        neighbours,
        smooth_lambda=smooth_lambda,
        icm_sweeps=icm_sweeps,
    )
    comp, chart_label = _connected_charts(label, neighbours)
    n_charts = len(chart_label)

    faces_of_chart = [np.flatnonzero(comp == c) for c in range(n_charts)]
    min_face = {c: int(faces_of_chart[c].min()) for c in range(n_charts)}

    # Per-chart geometry (source bounding box for observed charts).
    charts: list[dict[str, Any]] = []
    for c in range(n_charts):
        members = faces_of_chart[c]
        vids = np.unique(faces[members].ravel())
        chart: dict[str, Any] = {
            "id": c,
            "label": int(chart_label[c]),
            "faces": members,
            "vids": vids,
        }
        if chart_label[c] >= 0:
            view_index = int(chart_label[c])
            height, width = view_hw[view_index]
            view = views[view_index]
            u, v, _ = project_points(
                vertices[vids], view["rotation"], view["translation"], intrinsics
            )
            x0 = int(np.clip(np.floor(u.min()), 0, width - 1))
            x1 = int(np.clip(np.ceil(u.max()), 0, width - 1))
            y0 = int(np.clip(np.floor(v.min()), 0, height - 1))
            y1 = int(np.clip(np.ceil(v.max()), 0, height - 1))
            src_w = x1 - x0 + 1
            src_h = y1 - y0 + 1
            chart.update(
                x0=x0,
                y0=y0,
                src_w=src_w,
                src_h=src_h,
                s_chart=min(1.0, max_chart_px / float(max(src_w, src_h))),
            )
        charts.append(chart)

    # Shelf packing with global sqrt-style rescale on overflow.
    scale = 1.0
    repack_count = 0
    placement = _shelf_pack(charts, min_face, scale, atlas_size)
    while placement is None:
        if scale < 0.05:
            raise ValueError("charts do not fit in the atlas even after rescaling")
        scale *= 0.8
        repack_count += 1
        placement = _shelf_pack(charts, min_face, scale, atlas_size)

    for chart in charts:
        px, py, rect_w, rect_h, inner_w, inner_h = placement[chart["id"]]
        chart.update(
            px=px, py=py, rect_w=rect_w, rect_h=rect_h, dst_w=inner_w, dst_h=inner_h
        )

    atlas = np.zeros((atlas_size, atlas_size, 3), dtype=np.uint8)

    # Copy observed chart pixels, grouped by view so each frame loads once.
    by_view: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for chart in charts:
        if chart["label"] >= 0:
            by_view[chart["label"]].append(chart)

    for view_index in sorted(by_view):
        view = views[view_index]
        height, width = view_hw[view_index]
        image = _load_bgr(view, width, height)
        _, face_index = rasterize_zbuffer(
            vertices,
            faces,
            view["rotation"],
            view["translation"],
            intrinsics,
            width,
            height,
            scale=1.0,
        )
        for chart in by_view[view_index]:
            chart_mask = np.isin(face_index, chart["faces"])
            x0, y0 = chart["x0"], chart["y0"]
            src_w, src_h = chart["src_w"], chart["src_h"]
            dst_w, dst_h = chart["dst_w"], chart["dst_h"]
            sub_img = image[y0 : y0 + src_h, x0 : x0 + src_w]
            sub_mask = chart_mask[y0 : y0 + src_h, x0 : x0 + src_w]
            interp = (
                cv2.INTER_AREA
                if (dst_w < src_w or dst_h < src_h)
                else cv2.INTER_LINEAR
            )
            resized_img = cv2.resize(sub_img, (dst_w, dst_h), interpolation=interp)
            resized_mask = (
                cv2.resize(
                    sub_mask.astype(np.uint8),
                    (dst_w, dst_h),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
            rect_w, rect_h = chart["rect_w"], chart["rect_h"]
            canvas = np.zeros((rect_h, rect_w, 3), dtype=np.uint8)
            cov = np.zeros((rect_h, rect_w), dtype=bool)
            inner = canvas[_GUTTER : _GUTTER + dst_h, _GUTTER : _GUTTER + dst_w]
            inner[resized_mask] = resized_img[resized_mask]
            cov[_GUTTER : _GUTTER + dst_h, _GUTTER : _GUTTER + dst_w] = resized_mask
            core = cov.copy()
            fill = cov.copy()
            for _ in range(_GUTTER + 1):
                grown = cv2.dilate(canvas, _KERNEL3)
                grown_cov = cv2.dilate(fill.astype(np.uint8), _KERNEL3) > 0
                new_pixels = grown_cov & (~fill)
                canvas[new_pixels] = grown[new_pixels]
                fill = grown_cov
            px, py = chart["px"], chart["py"]
            atlas[py : py + rect_h, px : px + rect_w] = canvas
            chart["core"] = core
            chart["fill"] = fill

    # Seam levelling: per-chart additive colour offset (permitted simplification).
    seam_records: list[tuple[int, int, int, int, np.ndarray]] = []
    if seam_level:
        for (a, b), incident in edge_faces.items():
            if len(incident) != 2:
                continue
            f0, f1 = incident
            c0, c1 = int(comp[f0]), int(comp[f1])
            if c0 == c1:
                continue
            l0, l1 = int(chart_label[c0]), int(chart_label[c1])
            if l0 < 0 or l1 < 0:
                continue
            midpoint = (vertices[a] + vertices[b]) * 0.5
            points = np.stack([vertices[a], midpoint, vertices[b]])
            seam_records.append((c0, c1, l0, l1, points))

    observed_ids = [c for c in range(n_charts) if chart_label[c] >= 0]
    index_of = {c: i for i, c in enumerate(observed_ids)}
    offsets = np.zeros((len(observed_ids), 3))
    if seam_level and seam_records and observed_ids:
        colour_a: list[np.ndarray | None] = [None] * len(seam_records)
        colour_b: list[np.ndarray | None] = [None] * len(seam_records)
        needed: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for i, (_, _, l0, l1, _) in enumerate(seam_records):
            needed[l0].append((i, 0))
            needed[l1].append((i, 1))
        for view_index in sorted(needed):
            view = views[view_index]
            height, width = view_hw[view_index]
            image = _load_bgr(view, width, height).astype(np.float64)
            for i, side in needed[view_index]:
                points = seam_records[i][4]
                u, v, _ = project_points(
                    points, view["rotation"], view["translation"], intrinsics
                )
                colour = bilinear_sample(image, u, v).mean(axis=0)
                if side == 0:
                    colour_a[i] = colour
                else:
                    colour_b[i] = colour

        n_obs = len(observed_ids)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs = np.zeros((len(seam_records) + n_obs, 3))
        for i, (c0, c1, _, _, _) in enumerate(seam_records):
            rows += [i, i]
            cols += [index_of[c0], index_of[c1]]
            data += [1.0, -1.0]
            rhs[i] = np.asarray(colour_b[i]) - np.asarray(colour_a[i])
        for j in range(n_obs):
            rows.append(len(seam_records) + j)
            cols.append(j)
            data.append(_SEAM_GAMMA)
        system = coo_matrix(
            (data, (rows, cols)), shape=(len(seam_records) + n_obs, n_obs)
        ).tocsr()
        for channel in range(3):
            offsets[:, channel] = lsqr(
                system, rhs[:, channel], atol=1e-8, btol=1e-8, iter_lim=2000
            )[0]

        for chart in charts:
            if chart["label"] < 0:
                continue
            offset = offsets[index_of[chart["id"]]]
            px, py = chart["px"], chart["py"]
            rect_w, rect_h = chart["rect_w"], chart["rect_h"]
            region = atlas[py : py + rect_h, px : px + rect_w].astype(np.float64)
            fill = chart["fill"]
            region[fill] = np.clip(region[fill] + offset, 0.0, 255.0)
            atlas[py : py + rect_h, px : px + rect_w] = region.astype(np.uint8)

        # 3 px feather within each chart + its gutter (single-source only).
        for chart in charts:
            if chart["label"] < 0:
                continue
            px, py = chart["px"], chart["py"]
            rect_w, rect_h = chart["rect_w"], chart["rect_h"]
            region = atlas[py : py + rect_h, px : px + rect_w].astype(np.float32)
            blurred = cv2.GaussianBlur(region, (5, 5), 0)
            core = chart["core"]
            eroded = cv2.erode(core.astype(np.uint8), _KERNEL3, iterations=3) > 0
            band = core & (~eroded)
            region[band] = blurred[band]
            atlas[py : py + rect_h, px : px + rect_w] = region.astype(np.uint8)

    # Fill unobserved charts with the nearest observed faces' mean colour.
    chart_mean: dict[int, np.ndarray] = {}
    for chart in charts:
        if chart["label"] < 0:
            continue
        px, py = chart["px"], chart["py"]
        rect_w, rect_h = chart["rect_w"], chart["rect_h"]
        region = atlas[py : py + rect_h, px : px + rect_w]
        core = chart["core"]
        chart_mean[chart["id"]] = (
            region[core].mean(axis=0)
            if core.any()
            else np.array([0.0, 0.0, 0.0])
        )

    for chart in charts:
        if chart["label"] >= 0:
            continue
        nearest = _nearest_labelled(chart["faces"], neighbours, label)
        colours = [chart_mean[int(comp[f])] for f in nearest if int(comp[f]) in chart_mean]
        fill_colour = (
            np.mean(colours, axis=0) if colours else np.array([128.0, 128.0, 128.0])
        )
        px, py = chart["px"], chart["py"]
        rect_w, rect_h = chart["rect_w"], chart["rect_h"]
        atlas[py : py + rect_h, px : px + rect_w] = np.round(fill_colour).astype(np.uint8)

    # Build duplicated-per-chart vertices, reindexed faces (original order), UVs.
    out_vertices: list[np.ndarray] = []
    out_uvs: list[list[float]] = []
    vmap: dict[tuple[int, int], int] = {}
    denom = float(atlas_size - 1)
    for chart in charts:
        cid = chart["id"]
        vids = chart["vids"]
        if chart["label"] >= 0:
            view = views[chart["label"]]
            u, v, _ = project_points(
                vertices[vids], view["rotation"], view["translation"], intrinsics
            )
            scale_x = chart["dst_w"] / float(chart["src_w"])
            scale_y = chart["dst_h"] / float(chart["src_h"])
            col = chart["px"] + _GUTTER + (u - chart["x0"]) * scale_x
            row = chart["py"] + _GUTTER + (v - chart["y0"]) * scale_y
        else:
            col, row = _pca_layout(
                vertices[vids], chart["px"], chart["py"], chart["dst_w"], chart["dst_h"]
            )
        col = np.clip(col, 0.0, denom)
        row = np.clip(row, 0.0, denom)
        for k, vid in enumerate(vids):
            vmap[(cid, int(vid))] = len(out_vertices)
            out_vertices.append(vertices[int(vid)])
            out_uvs.append([float(col[k] / denom), float(1.0 - row[k] / denom)])

    out_faces = np.empty((face_count, 3), dtype=np.int32)
    source_view = np.empty(face_count, dtype=np.int32)
    for f in range(face_count):
        cid = int(comp[f])
        out_faces[f] = [vmap[(cid, int(faces[f, k]))] for k in range(3)]
        source_view[f] = int(chart_label[cid])

    out_vertices_arr = np.asarray(out_vertices, dtype=np.float32).reshape(-1, 3)
    out_uvs_arr = np.clip(np.asarray(out_uvs, dtype=np.float32), 0.0, 1.0)

    unobserved = int(np.count_nonzero(label < 0))
    report = {
        "method": "per-face single-best-view (Waechter-style), no cross-view blending",
        "views": len(views),
        "faces": int(face_count),
        "chart_count": int(n_charts),
        "observed_charts": int(len(observed_ids)),
        "unobserved_charts": int(n_charts - len(observed_ids)),
        "unobserved_face_fraction": float(unobserved / face_count) if face_count else 0.0,
        "gate_stats": gate_stats,
        "smooth_lambda": float(lam),
        "icm_sweeps": int(icm_sweeps),
        "atlas_size": int(atlas_size),
        "atlas_rescaled": bool(repack_count > 0),
        "pack_scale": float(scale),
        "repack_count": int(repack_count),
        "seam_leveling": (
            "per-chart additive offset (permitted simplification of per-vertex)"
            if seam_level
            else "disabled"
        ),
        "seam_edge_count": int(len(seam_records)),
        "source_view": source_view,
    }

    return {
        "vertices": out_vertices_arr,
        "faces": out_faces,
        "uvs": out_uvs_arr,
        "texture_bgr": atlas,
        "report": report,
    }


def _nearest_labelled(
    seed_faces: np.ndarray, neighbours: list[np.ndarray], label: np.ndarray
) -> list[int]:
    """Multi-source BFS from ``seed_faces`` to the nearest labelled faces."""

    visited = set(int(f) for f in seed_faces)
    frontier = list(visited)
    while frontier:
        found: list[int] = []
        nxt: list[int] = []
        for f in frontier:
            for nbr in neighbours[f]:
                n = int(nbr)
                if n in visited:
                    continue
                visited.add(n)
                if label[n] >= 0:
                    found.append(n)
                else:
                    nxt.append(n)
        if found:
            return sorted(found)
        frontier = nxt
    return []
