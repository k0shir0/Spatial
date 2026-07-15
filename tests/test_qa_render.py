"""Tests for the QA render module (turntable contact sheets + geometry gate)."""

from __future__ import annotations

import numpy as np
import trimesh

from local3d.qa_render import (
    geometry_gate,
    render_geometry_views,
    render_textured_views,
)

SIZE = 512
BACKGROUND = 34

# Cube corners for each of the six faces, wound consistently; each face gets its
# own four vertices so it can carry a full [0,1]x[0,1] UV tile.
_FACE_QUADS = [
    [(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5)],
    [(-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, -0.5)],
    [(-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5)],
    [(-0.5, -0.5, 0.5), (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5)],
    [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5)],
    [(0.5, -0.5, -0.5), (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5)],
]


def _checker_cube():
    """Cube with duplicated per-face verts and full-tile UVs (24 verts, 12 tris)."""

    vertices, faces, uvs = [], [], []
    for quad in _FACE_QUADS:
        base = len(vertices)
        vertices.extend(quad)
        uvs.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])
        faces.append((base, base + 1, base + 2))
        faces.append((base, base + 2, base + 3))
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        np.asarray(uvs, dtype=np.float64),
    )


def _checker_texture(size: int = 64, tile: int = 8) -> np.ndarray:
    """64x64 black/white checkerboard as a BGR uint8 image."""

    idx = (np.arange(size)[:, None] // tile) + (np.arange(size)[None, :] // tile)
    checker = np.where(idx % 2 == 0, 255, 0).astype(np.uint8)
    return np.stack([checker, checker, checker], axis=-1)


def _tiles(sheet: np.ndarray, size: int = SIZE):
    """Split a 2x3 contact sheet into its six tiles in row-major order."""

    out = []
    for row in range(2):
        for col in range(3):
            out.append(sheet[row * size : (row + 1) * size, col * size : (col + 1) * size])
    return out


def test_textured_contact_sheet_shape_and_dtype():
    vertices, faces, uvs = _checker_cube()
    sheet = render_textured_views(vertices, faces, uvs, _checker_texture())
    assert sheet.shape == (2 * SIZE, 3 * SIZE, 3)
    assert sheet.dtype == np.uint8


def test_tiles_are_non_empty():
    vertices, faces, uvs = _checker_cube()
    sheet = render_textured_views(vertices, faces, uvs, _checker_texture())
    for tile in _tiles(sheet):
        non_background = np.any(tile != BACKGROUND, axis=-1)
        assert non_background.mean() > 0.02


def test_different_angles_differ():
    vertices, faces, uvs = _checker_cube()
    sheet = render_textured_views(vertices, faces, uvs, _checker_texture())
    tiles = _tiles(sheet)
    assert np.any(tiles[0] != tiles[3])


def test_render_is_byte_identical_across_runs():
    vertices, faces, uvs = _checker_cube()
    texture = _checker_texture()
    first = render_textured_views(vertices, faces, uvs, texture)
    second = render_textured_views(vertices, faces, uvs, texture)
    assert first.tobytes() == second.tobytes()


def test_geometry_views_shape_and_non_empty():
    vertices, faces, _ = _checker_cube()
    sheet = render_geometry_views(vertices, faces)
    assert sheet.shape == (2 * SIZE, 3 * SIZE, 3)
    for tile in _tiles(sheet):
        assert np.any(tile != BACKGROUND)


def test_geometry_gate_passes_on_cube():
    box = trimesh.creation.box().subdivide().subdivide().subdivide()
    result = geometry_gate(np.asarray(box.vertices), np.asarray(box.faces))
    assert result["pass"] is True
    assert result["reasons"] == []
    assert result["metrics"]["components"] == 1
    assert result["metrics"]["watertight"] is True


def test_geometry_gate_fails_on_two_components():
    box_a = trimesh.creation.box().subdivide().subdivide().subdivide()
    box_b = box_a.copy()
    box_b.apply_translation([5.0, 0.0, 0.0])
    vertices = np.vstack([box_a.vertices, box_b.vertices])
    faces = np.vstack([box_a.faces, box_b.faces + len(box_a.vertices)])
    result = geometry_gate(vertices, faces)
    assert result["pass"] is False
    assert result["reasons"]
    assert result["metrics"]["components"] == 2


def test_geometry_gate_fails_on_small_sheet():
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]], dtype=np.int64)
    result = geometry_gate(vertices, faces)
    assert result["pass"] is False
    assert result["reasons"]
