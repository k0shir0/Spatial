"""Pipeline-owned QA artifacts: turntable contact sheets and a geometry gate.

The reconstruction pipeline produces mesh + texture files that a reviewer must
be able to judge *from files alone*, without a GPU viewer.  This module renders
a small orbit of the final mesh into a single contact-sheet image (textured, or
flat-shaded to reveal raw geometry) and provides a pass/fail geometry gate that
codifies the minimum topology a deliverable must satisfy.

Honest limits: the renderer is a deterministic CPU rasteriser meant for review,
not photorealism.  Shading is a single fixed head-light with no shadows, no
ambient occlusion, and no perspective-correct UV interpolation (affine within
each triangle, which is adequate for the small triangles of a delivery mesh).
It reuses the shared geometry core (``recon_common``) for projection, z-buffer
visibility, bilinear texture sampling, and outward face normals so its camera
conventions match the rest of the pipeline exactly.

Everything is deterministic: no randomness, fixed iteration order, no
wall-clock, numpy + OpenCV + trimesh only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from local3d.recon_common import (
    Intrinsics,
    bilinear_sample,
    face_normals,
    project_points,
    rasterize_zbuffer,
)

# Fixed head-light in CAMERA space (points up-left-toward viewer).
_LIGHT = np.array([-0.4, 0.6, 1.0], dtype=np.float64)
_LIGHT /= np.linalg.norm(_LIGHT)

# Contact sheets are laid out three tiles wide (2 rows x 3 cols by default).
_COLUMNS = 3

# Flat base colour used by the geometry-only render (shade is painted onto it).
_GEOMETRY_GRAY = 190.0

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _orbit_view(
    centroid: np.ndarray,
    radius: float,
    azimuth_deg: float,
    elevation_deg: float,
    *,
    distance_scale: float = 2.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a look-at camera orbiting ``centroid`` at ``distance_scale*radius``.

    Returns ``(rotation, translation)`` following ``x_cam = R @ X_world + t``,
    with the camera's +z axis pointing at the centroid and +y pointing down the
    image (COLMAP/OpenCV convention).
    """

    azimuth = np.radians(azimuth_deg)
    elevation = np.radians(elevation_deg)
    direction = np.array(
        [
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
            np.cos(elevation) * np.cos(azimuth),
        ],
        dtype=np.float64,
    )
    distance = distance_scale * max(radius, 1e-9)
    camera_center = centroid + distance * direction

    forward = centroid - camera_center
    forward /= max(np.linalg.norm(forward), 1e-12)
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(float(forward @ world_up)) > 0.999:
        world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, forward)
    right /= max(np.linalg.norm(right), 1e-12)
    down = np.cross(forward, right)

    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ camera_center
    return rotation, translation


def _barycentric(
    tri_u: np.ndarray,
    tri_v: np.ndarray,
    face_ids: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
) -> np.ndarray:
    """Per-pixel barycentric weights within each pixel's covering triangle.

    ``tri_u``/``tri_v`` are projected vertex coords indexed as ``[face, corner]``.
    Returns an ``(N, 3)`` array of weights ordered ``(a, b, c)``.
    """

    ax, ay = tri_u[face_ids, 0], tri_v[face_ids, 0]
    bx, by = tri_u[face_ids, 1], tri_v[face_ids, 1]
    cx, cy = tri_u[face_ids, 2], tri_v[face_ids, 2]
    edge_bx, edge_by = bx - ax, by - ay
    edge_cx, edge_cy = cx - ax, cy - ay
    denom = edge_bx * edge_cy - edge_by * edge_cx
    safe = np.where(np.abs(denom) < 1e-12, 1.0, denom)
    rel_x = px - ax
    rel_y = py - ay
    w_b = (rel_x * edge_cy - rel_y * edge_cx) / safe
    w_c = (rel_y * edge_bx - rel_x * edge_by) / safe
    w_a = 1.0 - w_b - w_c
    return np.stack([w_a, w_b, w_c], axis=1)


def _render_view(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsics: Intrinsics,
    size: int,
    background: int,
    *,
    uvs: np.ndarray | None = None,
    texture_bgr: np.ndarray | None = None,
) -> np.ndarray:
    """Rasterise one camera into a ``size x size`` BGR tile.

    When ``uvs``/``texture_bgr`` are supplied the base colour is the sampled
    texel; otherwise a flat gray is used (geometry-only render).  The final
    colour is ``base * (0.35 + 0.65*shade)`` with a fixed camera-space head-light.
    """

    tile = np.full((size, size, 3), int(background), dtype=np.uint8)
    _, face_index = rasterize_zbuffer(
        vertices, faces, rotation, translation, intrinsics, size, size, scale=1.0
    )
    covered = face_index >= 0
    if not covered.any():
        return tile

    ys, xs = np.nonzero(covered)
    face_ids = face_index[ys, xs]
    px = xs.astype(np.float64)
    py = ys.astype(np.float64)

    u, v, _ = project_points(vertices, rotation, translation, intrinsics)
    tri_u = u[faces]
    tri_v = v[faces]
    weights = _barycentric(tri_u, tri_v, face_ids, px, py)

    # Shade: |dot(face normal, light)| in camera space, ambient-lifted.
    normals_camera = normals[face_ids] @ rotation.T
    shade = 0.25 + 0.75 * np.abs(normals_camera @ _LIGHT)
    factor = (0.35 + 0.65 * shade)[:, None]

    if texture_bgr is not None and uvs is not None:
        corner_uv = uvs[faces][face_ids]  # (N, 3, 2)
        uv = np.einsum("nk,nkc->nc", weights, corner_uv)
        tex_h, tex_w = texture_bgr.shape[:2]
        tex_u = uv[:, 0] * (tex_w - 1)
        tex_v = (1.0 - uv[:, 1]) * (tex_h - 1)
        base = bilinear_sample(texture_bgr, tex_u, tex_v)
        if base.ndim == 1:
            base = np.repeat(base[:, None], 3, axis=1)
    else:
        base = np.full((len(face_ids), 3), _GEOMETRY_GRAY, dtype=np.float64)

    color = np.clip(np.rint(base * factor), 0, 255).astype(np.uint8)
    tile[ys, xs] = color
    return tile


def _label(tile: np.ndarray, angle_deg: float) -> None:
    """Draw the orbit angle in the tile's top-left corner (outline + fill)."""

    text = f"{int(round(angle_deg))} deg"
    cv2.putText(tile, text, (12, 34), _FONT, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(tile, text, (12, 34), _FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)


def _contact_sheet(tiles: list[np.ndarray], size: int, background: int) -> np.ndarray:
    """Grid the tiles row-major, ``_COLUMNS`` wide, padding with blanks."""

    tiles = list(tiles)
    blank = np.full((size, size, 3), int(background), dtype=np.uint8)
    while len(tiles) % _COLUMNS:
        tiles.append(blank)
    rows = [
        np.hstack(tiles[start : start + _COLUMNS])
        for start in range(0, len(tiles), _COLUMNS)
    ]
    return np.vstack(rows)


def render_textured_views(
    vertices: np.ndarray,
    faces: np.ndarray,
    uvs: np.ndarray,
    texture_bgr: np.ndarray,
    *,
    angles_deg: tuple = (0, 60, 120, 180, 240, 300),
    elevation_deg: float = 18.0,
    size: int = 512,
    background: int = 34,
) -> np.ndarray:
    """Textured turntable contact sheet of ``vertices``/``faces``.

    Orbits the mesh centroid at ``2.6 * bounding-sphere radius`` with a
    perspective focal of ``1.2 * size``, renders each angle as a ``size x size``
    tile (labelled with its angle), and grids them ``_COLUMNS`` wide.  Returns a
    BGR uint8 image; the default six angles give a ``(2*size, 3*size, 3)`` sheet.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float64)
    texture_bgr = np.asarray(texture_bgr)

    normals = face_normals(vertices, faces)
    centroid = vertices.mean(axis=0)
    radius = float(np.linalg.norm(vertices - centroid, axis=1).max())
    intrinsics: Intrinsics = (1.2 * size, size / 2.0, size / 2.0, 0.0)

    tiles = []
    for angle in angles_deg:
        rotation, translation = _orbit_view(centroid, radius, float(angle), elevation_deg)
        tile = _render_view(
            vertices, faces, normals, rotation, translation, intrinsics,
            size, background, uvs=uvs, texture_bgr=texture_bgr,
        )
        _label(tile, float(angle))
        tiles.append(tile)
    return _contact_sheet(tiles, size, background)


def render_geometry_views(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    angles_deg: tuple = (0, 60, 120, 180, 240, 300),
    elevation_deg: float = 18.0,
    size: int = 512,
    background: int = 34,
) -> np.ndarray:
    """Flat-shaded (untextured) turntable contact sheet.

    Same camera path as :func:`render_textured_views` but paints shade onto a
    flat gray, so raw geometry — dents, spikes, holes — is visible without
    texture masking it.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    normals = face_normals(vertices, faces)
    centroid = vertices.mean(axis=0)
    radius = float(np.linalg.norm(vertices - centroid, axis=1).max())
    intrinsics: Intrinsics = (1.2 * size, size / 2.0, size / 2.0, 0.0)

    tiles = []
    for angle in angles_deg:
        rotation, translation = _orbit_view(centroid, radius, float(angle), elevation_deg)
        tile = _render_view(
            vertices, faces, normals, rotation, translation, intrinsics,
            size, background,
        )
        _label(tile, float(angle))
        tiles.append(tile)
    return _contact_sheet(tiles, size, background)


def save_contact_sheet(sheet: np.ndarray, path: Path) -> None:
    """Write a contact sheet to ``path`` (creating parent directories)."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise OSError(f"failed to write contact sheet to {path}")


def geometry_gate(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_triangles: int = 500,
) -> dict[str, Any]:
    """Pass/fail topology gate for a delivery mesh.

    Checks a single connected component, watertightness, a triangle-count
    floor, finite positive bounding-box extents, and a non-degenerate aspect
    ratio (max/min extent < 25).  Returns ``{'pass', 'reasons', 'metrics'}``;
    ``reasons`` lists every failed check so the caller can report them.
    """

    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    triangles = int(len(faces))
    components = int(mesh.body_count)
    watertight = bool(mesh.is_watertight)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    finite_positive = bool(np.all(np.isfinite(extents)) and np.all(extents > 0.0))
    aspect = float(extents.max() / extents.min()) if finite_positive else float("inf")

    reasons: list[str] = []
    if components != 1:
        reasons.append(f"not a single connected component (found {components})")
    if not watertight:
        reasons.append("mesh is not watertight")
    if triangles < min_triangles:
        reasons.append(f"too few triangles ({triangles} < {min_triangles})")
    if not finite_positive:
        reasons.append("bounding-box extents are not finite and positive")
    elif aspect >= 25.0:
        reasons.append(f"degenerate aspect ratio ({aspect:.2f} >= 25)")

    metrics = {
        "triangles": triangles,
        "components": components,
        "watertight": watertight,
        "extents": [float(value) for value in extents],
        "aspect": aspect,
    }
    return {"pass": len(reasons) == 0, "reasons": reasons, "metrics": metrics}
