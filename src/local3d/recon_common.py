"""Shared geometry core for the automatic video reconstruction pipeline.

Conventions (identical to COLMAP / the rest of this repo):

- World-to-camera: ``x_cam = R @ X_world + t`` with ``R`` a (3, 3) rotation and
  ``t`` a (3,) translation.
- Camera model is COLMAP ``SIMPLE_RADIAL``: ``params = (focal, cx, cy, k1)``.
- Images are OpenCV BGR uint8 arrays; masks are boolean arrays of the same
  height/width.
- A *view* is a plain dict — see :func:`make_view` — so stages stay decoupled
  and JSON-reportable.

Everything here is deterministic, CPU-only, and dependency-light (numpy +
OpenCV only).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

Intrinsics = tuple[float, float, float, float]


def make_view(
    *,
    name: str,
    image_path: Path,
    rotation: np.ndarray,
    translation: np.ndarray,
    mask_tight: np.ndarray | None = None,
    mask_eroded: np.ndarray | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bundle one posed source frame.

    ``rotation``/``translation`` follow ``x_cam = R @ X + t``.  The camera
    center in world coordinates is ``-R.T @ t``.
    """

    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    view: dict[str, Any] = {
        "name": name,
        "image_path": Path(image_path),
        "rotation": rotation,
        "translation": translation,
        "center": -rotation.T @ translation,
        "mask_tight": mask_tight,
        "mask_eroded": mask_eroded,
    }
    if extras:
        view.update(extras)
    return view


def project_points(
    points: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: Intrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points with the SIMPLE_RADIAL model.

    Returns float pixel coordinates ``(u, v)`` and ``depth`` (camera-frame z).
    Points at or behind the camera get depth <= 0; callers must gate on it.
    No rounding — callers choose nearest/bilinear sampling.
    """

    focal, center_x, center_y, radial = intrinsics
    camera_points = points @ np.asarray(rotation).T + np.asarray(translation)
    depth = camera_points[:, 2]
    safe = np.where(depth > 1e-9, depth, 1.0)
    x_normalized = camera_points[:, 0] / safe
    y_normalized = camera_points[:, 1] / safe
    distortion = 1.0 + radial * (x_normalized**2 + y_normalized**2)
    u = focal * x_normalized * distortion + center_x
    v = focal * y_normalized * distortion + center_y
    return u, v, depth


def bilinear_sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``image`` (H, W) or (H, W, C) at float pixel coords.

    Coordinates are clamped to the valid rectangle; callers gate validity
    separately.  Returns float64 samples with shape (N,) or (N, C).
    """

    height, width = image.shape[:2]
    u = np.clip(np.asarray(u, dtype=np.float64), 0.0, width - 1.000001)
    v = np.clip(np.asarray(v, dtype=np.float64), 0.0, height - 1.000001)
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = np.minimum(u0 + 1, width - 1)
    v1 = np.minimum(v0 + 1, height - 1)
    fu = (u - u0)[..., None] if image.ndim == 3 else (u - u0)
    fv = (v - v0)[..., None] if image.ndim == 3 else (v - v0)
    data = image.astype(np.float64)
    top = data[v0, u0] * (1.0 - fu) + data[v0, u1] * fu
    bottom = data[v1, u0] * (1.0 - fu) + data[v1, u1] * fu
    return top * (1.0 - fv) + bottom * fv


def rasterize_zbuffer(
    vertices: np.ndarray,
    faces: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: Intrinsics,
    width: int,
    height: int,
    *,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """CPU z-buffer of a triangle mesh in one camera.

    Renders at ``round(width * scale) x round(height * scale)`` (use scale < 1
    to keep visibility tests cheap).  Returns ``(zbuffer, face_index)`` at that
    resolution: ``zbuffer`` is float32 camera depth (np.inf where empty) and
    ``face_index`` is int32 (-1 where empty).

    Deterministic: faces are painted in a fixed back-to-front order and ties
    resolve to the nearer (already painted, then overwritten only if strictly
    nearer) triangle via per-pixel depth comparison.
    """

    render_w = max(int(round(width * scale)), 8)
    render_h = max(int(round(height * scale)), 8)
    sx = render_w / float(width)
    sy = render_h / float(height)

    u, v, depth = project_points(vertices, rotation, translation, intrinsics)
    u = u * sx
    v = v * sy

    zbuffer = np.full((render_h, render_w), np.inf, dtype=np.float32)
    face_index = np.full((render_h, render_w), -1, dtype=np.int32)

    tri_u = u[faces]
    tri_v = v[faces]
    tri_z = depth[faces]
    # Cull triangles with any vertex behind the camera or fully off-screen.
    ok = (
        (tri_z > 1e-9).all(axis=1)
        & (tri_u.max(axis=1) >= 0)
        & (tri_u.min(axis=1) < render_w)
        & (tri_v.max(axis=1) >= 0)
        & (tri_v.min(axis=1) < render_h)
    )
    candidates = np.flatnonzero(ok)
    if not len(candidates):
        return zbuffer, face_index

    order = candidates[np.argsort(-tri_z[candidates].mean(axis=1), kind="stable")]
    for fi in order:
        us, vs, zs = tri_u[fi], tri_v[fi], tri_z[fi]
        min_x = max(int(np.floor(us.min())), 0)
        max_x = min(int(np.ceil(us.max())), render_w - 1)
        min_y = max(int(np.floor(vs.min())), 0)
        max_y = min(int(np.ceil(vs.max())), render_h - 1)
        if min_x > max_x or min_y > max_y:
            continue
        gx, gy = np.meshgrid(
            np.arange(min_x, max_x + 1), np.arange(min_y, max_y + 1)
        )
        px = gx.ravel().astype(np.float64) + 0.0
        py = gy.ravel().astype(np.float64) + 0.0
        edge_b = np.array([us[1] - us[0], vs[1] - vs[0]])
        edge_c = np.array([us[2] - us[0], vs[2] - vs[0]])
        denom = edge_b[0] * edge_c[1] - edge_b[1] * edge_c[0]
        if abs(denom) < 1e-12:
            continue
        rel_x = px - us[0]
        rel_y = py - vs[0]
        w_b = (rel_x * edge_c[1] - rel_y * edge_c[0]) / denom
        w_c = (rel_y * edge_b[0] - rel_x * edge_b[1]) / denom
        w_a = 1.0 - w_b - w_c
        inside = (w_a >= -1e-6) & (w_b >= -1e-6) & (w_c >= -1e-6)
        if not inside.any():
            continue
        # Interpolate inverse depth for perspective correctness.
        inv_z = w_a[inside] / zs[0] + w_b[inside] / zs[1] + w_c[inside] / zs[2]
        z_pix = (1.0 / np.maximum(inv_z, 1e-12)).astype(np.float32)
        yy = gy.ravel()[inside]
        xx = gx.ravel()[inside]
        nearer = z_pix < zbuffer[yy, xx]
        zbuffer[yy[nearer], xx[nearer]] = z_pix[nearer]
        face_index[yy[nearer], xx[nearer]] = fi
    return zbuffer, face_index


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit face normals, outward-oriented against the mesh centroid.

    Suitable for closed, roughly star-shaped delivery meshes; faces whose
    normal points toward the centroid are flipped.
    """

    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    outward = np.einsum(
        "ij,ij->i", normals, tri.mean(axis=1) - vertices.mean(axis=0)
    )
    normals[outward < 0] *= -1.0
    return normals


def visible_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    view: dict[str, Any],
    intrinsics: Intrinsics,
    width: int,
    height: int,
    *,
    scale: float = 0.5,
    depth_tolerance: float = 0.01,
) -> np.ndarray:
    """Boolean visibility per face in one view via the z-buffer.

    A face is visible when its centroid projects inside the image and its
    centroid depth matches the z-buffer within ``depth_tolerance`` (relative
    to the mesh diagonal).
    """

    zbuffer, _ = rasterize_zbuffer(
        vertices, faces, view["rotation"], view["translation"], intrinsics,
        width, height, scale=scale,
    )
    centroids = vertices[faces].mean(axis=1)
    u, v, depth = project_points(
        centroids, view["rotation"], view["translation"], intrinsics
    )
    render_h, render_w = zbuffer.shape
    ui = np.clip(np.rint(u * render_w / width), 0, render_w - 1).astype(np.int64)
    vi = np.clip(np.rint(v * render_h / height), 0, render_h - 1).astype(np.int64)
    in_image = (depth > 1e-9) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    diag = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    tolerance = max(depth_tolerance * diag, 1e-9)
    zface = zbuffer[vi, ui]
    return in_image & np.isfinite(zface) & (depth <= zface + tolerance)
