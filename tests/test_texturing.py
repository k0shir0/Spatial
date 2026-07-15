"""Synthetic ground-truth tests for the ghost-free texture atlas baker.

A subdivided cube with six flat, strongly-distinct side colours is rendered
from a ring of four cameras (looking slightly down, so the bottom face is
never seen).  Each source frame paints every pixel with its true face colour
plus a small per-view brightness offset, so any cross-view averaging would
show up as a wrong colour family, and any unlevelled seam as a brightness jump.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.recon_common import make_view, rasterize_zbuffer  # noqa: E402
from local3d.texturing import bake_texture_atlas  # noqa: E402


# Six strongly separated BGR colours, one per cube side, keyed by dominant
# normal axis and sign so every triangle inherits its side's colour family.
_SIDE_COLORS = {
    (0, 1): (40, 40, 220),   # +X  red
    (0, -1): (220, 220, 40),  # -X  cyan
    (1, 1): (40, 220, 40),   # +Y  green
    (1, -1): (220, 40, 220),  # -Y  magenta
    (2, 1): (220, 40, 40),   # +Z  blue
    (2, -1): (40, 220, 220),  # -Z  yellow (bottom, never seen)
}
_INTRINSICS = (250.0, 100.0, 100.0, 0.0)
_IMG = 200


def _cube() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh = box.subdivide().subdivide()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    axis = np.argmax(np.abs(normals), axis=1)
    sign = np.sign(normals[np.arange(len(faces)), axis]).astype(int)
    colors = np.array(
        [_SIDE_COLORS[(int(axis[f]), int(sign[f]) if sign[f] != 0 else 1)] for f in range(len(faces))],
        dtype=np.uint8,
    )
    return vertices, faces, colors


def _look_at(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = -center / np.linalg.norm(center)  # camera looks toward origin
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])  # rows: camera axes in world
    translation = -rotation @ center
    return rotation, translation


def _render_views(tmp_path: Path) -> list[dict]:
    vertices, faces, colors = _cube()
    offsets = [12, -12, 9, -9]
    views: list[dict] = []
    for i, theta in enumerate((0.0, 90.0, 180.0, 270.0)):
        rad = np.radians(theta)
        center = np.array([4.0 * np.cos(rad), 4.0 * np.sin(rad), 3.0])
        rotation, translation = _look_at(center)
        _, face_index = rasterize_zbuffer(
            vertices, faces, rotation, translation, _INTRINSICS, _IMG, _IMG, scale=1.0
        )
        painted = face_index >= 0
        image = np.zeros((_IMG, _IMG, 3), dtype=np.uint8)
        fill = np.clip(
            colors[np.where(painted, face_index, 0)].astype(np.int16) + offsets[i], 0, 255
        ).astype(np.uint8)
        image[painted] = fill[painted]
        # Dilate the mask so silhouette vertices clear the boundary gate; the
        # copy step still uses the true face-index render, not the dilation.
        mask = cv2.dilate(painted.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=18) > 0
        path = tmp_path / f"view_{i}.png"
        cv2.imwrite(str(path), image)
        views.append(
            make_view(
                name=f"view_{i}",
                image_path=path,
                rotation=rotation,
                translation=translation,
                mask_tight=mask,
            )
        )
    return views, vertices, faces, colors


def _nearest_color_family(bgr: np.ndarray) -> tuple[int, int, int]:
    palette = np.array(list(_SIDE_COLORS.values()), dtype=np.float64)
    keys = list(_SIDE_COLORS.values())
    distances = np.linalg.norm(palette - np.asarray(bgr, dtype=np.float64), axis=1)
    return keys[int(np.argmin(distances))]


def _atlas_pixel(uv: np.ndarray, atlas_size: int) -> tuple[int, int]:
    col = int(round(float(uv[0]) * (atlas_size - 1)))
    row = int(round((1.0 - float(uv[1])) * (atlas_size - 1)))
    return row, col


def _face_texels(result: dict, face: int) -> np.ndarray:
    atlas = result["texture_bgr"]
    size = atlas.shape[0]
    uv = result["uvs"][result["faces"][face]]
    pixels = np.array([_atlas_pixel(uv[k], size) for k in range(3)], dtype=np.float64)
    r0 = int(np.floor(pixels[:, 0].min()))
    r1 = int(np.ceil(pixels[:, 0].max()))
    c0 = int(np.floor(pixels[:, 1].min()))
    c1 = int(np.ceil(pixels[:, 1].max()))
    a = pixels[0]
    edge_b = pixels[1] - a
    edge_c = pixels[2] - a
    denom = edge_b[0] * edge_c[1] - edge_b[1] * edge_c[0]
    texels = []
    if abs(denom) < 1e-9:
        return np.zeros((0, 3))
    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            rel = np.array([row - a[0], col - a[1]])
            wb = (rel[0] * edge_c[1] - rel[1] * edge_c[0]) / denom
            wc = (rel[1] * edge_b[0] - rel[0] * edge_b[1]) / denom
            wa = 1.0 - wb - wc
            if wa >= -1e-6 and wb >= -1e-6 and wc >= -1e-6:
                texels.append(atlas[row, col])
    return np.asarray(texels, dtype=np.float64)


def _seam_brightness_diff(result: dict, faces: np.ndarray, vertices: np.ndarray) -> float:
    source_view = result["report"]["source_view"]
    atlas = result["texture_bgr"].astype(np.float64)
    size = atlas.shape[0]
    out_faces = result["faces"]
    out_verts = result["vertices"]
    uvs = result["uvs"]

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for f in range(len(faces)):
        a, b, c = (int(x) for x in faces[f])
        for i, j in ((a, b), (b, c), (c, a)):
            edge_faces.setdefault((i, j) if i < j else (j, i), []).append(f)

    def uv_of(out_face: int, position: np.ndarray) -> np.ndarray:
        corners = out_verts[out_faces[out_face]]
        idx = int(np.argmin(np.linalg.norm(corners - position, axis=1)))
        return uvs[out_faces[out_face][idx]]

    def sample(uv: np.ndarray) -> np.ndarray:
        row, col = _atlas_pixel(uv, size)
        return atlas[np.clip(row, 0, size - 1), np.clip(col, 0, size - 1)]

    diffs = []
    for (a, b), incident in edge_faces.items():
        if len(incident) != 2:
            continue
        f0, f1 = incident
        if source_view[f0] < 0 or source_view[f1] < 0 or source_view[f0] == source_view[f1]:
            continue
        mid = (vertices[a] + vertices[b]) * 0.5
        p0 = vertices[a]
        for pos in (p0, mid):
            c_left = sample(uv_of(f0, pos))
            c_right = sample(uv_of(f1, pos))
            diffs.append(abs(c_left.mean() - c_right.mean()))
    return float(np.mean(diffs)) if diffs else 0.0


@pytest.fixture()
def scene(tmp_path: Path):
    views, vertices, faces, colors = _render_views(tmp_path)
    return views, vertices, faces, colors


def test_no_cross_view_ghosting_every_face_keeps_its_color_family(scene) -> None:
    views, vertices, faces, colors = scene
    result = bake_texture_atlas(vertices, faces, views, _INTRINSICS, atlas_size=1024)
    source_view = result["report"]["source_view"]

    checked = 0
    for f in range(len(faces)):
        if source_view[f] < 0:
            continue
        texels = _face_texels(result, f)
        if len(texels) < 6:  # skip sub-pixel triangles with too few texels to judge
            continue
        true_family = tuple(int(x) for x in colors[f])
        families = [_nearest_color_family(t) for t in texels]
        dominant = sum(1 for fam in families if fam == true_family) / len(families)
        assert dominant > 0.90, f"face {f}: only {dominant:.2f} texels match its color family"
        checked += 1
    assert checked > 50


def test_multiview_faces_form_few_charts(scene) -> None:
    views, vertices, faces, colors = scene
    result = bake_texture_atlas(vertices, faces, views, _INTRINSICS, atlas_size=1024)
    assert result["report"]["chart_count"] < len(faces) / 2


def test_seam_leveling_reduces_brightness_jumps(scene) -> None:
    views, vertices, faces, colors = scene
    leveled = bake_texture_atlas(vertices, faces, views, _INTRINSICS, atlas_size=1024, seam_level=True)
    raw = bake_texture_atlas(vertices, faces, views, _INTRINSICS, atlas_size=1024, seam_level=False)
    diff_leveled = _seam_brightness_diff(leveled, faces, vertices)
    diff_raw = _seam_brightness_diff(raw, faces, vertices)
    assert diff_raw > 1.0  # the +/- brightness offsets do create real seams
    assert diff_leveled < diff_raw


def test_deterministic_texture_bytes(scene) -> None:
    views, vertices, faces, colors = scene
    first = bake_texture_atlas(vertices, faces, views, _INTRINSICS, atlas_size=1024)
    second = bake_texture_atlas(vertices, faces, views, _INTRINSICS, atlas_size=1024)
    assert np.array_equal(first["texture_bgr"], second["texture_bgr"])
    assert np.array_equal(first["uvs"], second["uvs"])
    assert np.array_equal(first["faces"], second["faces"])
    assert np.array_equal(first["report"]["source_view"], second["report"]["source_view"])


def test_unobserved_bottom_is_flagged_filled_and_outputs_are_valid(scene) -> None:
    views, vertices, faces, colors = scene
    result = bake_texture_atlas(vertices, faces, views, _INTRINSICS, atlas_size=1024)
    report = result["report"]
    source_view = report["source_view"]

    # The downward-pointing bottom face (-Z) is never seen by the camera ring.
    tri = vertices[faces]
    face_normals_z = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2]
    bottom = face_normals_z < -1e-6
    assert report["unobserved_face_fraction"] > 0.0
    assert np.all(source_view[bottom] < 0)

    # A representative unobserved face was filled with a real (non-black) colour.
    unobserved = np.flatnonzero(source_view < 0)
    filled = _face_texels(result, int(unobserved[0]))
    assert len(filled) > 0
    assert filled.mean() > 5.0

    # UVs in range and faces reindex validly against the duplicated vertices.
    uvs = result["uvs"]
    assert uvs.min() >= 0.0 and uvs.max() <= 1.0
    assert result["faces"].min() >= 0
    assert result["faces"].max() < len(result["vertices"])
    assert result["vertices"].dtype == np.float32
    assert result["faces"].dtype == np.int32
    assert result["uvs"].dtype == np.float32
    assert result["texture_bgr"].dtype == np.uint8
