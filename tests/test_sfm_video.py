"""Pure-logic tests for masked sequential SfM (no pycolmap run).

The end-to-end COLMAP pipeline is too slow for unit tests and is exercised by
the pipeline driver; those paths are marked skipped here. What we can pin
cheaply is the deterministic pair generation, mask staging naming, and the
focal-from-fov math.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from local3d.sfm_video import (
    focal_from_fov,
    sequential_loop_pairs,
    stage_masks,
)


def _names(count: int) -> list[str]:
    return [f"frame_{i:04d}.jpg" for i in range(count)]


def test_loop_pairs_deterministic() -> None:
    names = _names(40)
    first = sequential_loop_pairs(names, overlap=15, loop_stride=6)
    second = sequential_loop_pairs(list(reversed(names)), overlap=15, loop_stride=6)
    assert first == second


def test_loop_pairs_respect_overlap_gap() -> None:
    names = _names(40)
    index = {name: i for i, name in enumerate(names)}
    pairs = sequential_loop_pairs(names, overlap=15, loop_stride=6)
    assert pairs, "expected some loop pairs for this many frames"
    for a, b in pairs:
        assert abs(index[a] - index[b]) > 15


def test_loop_pairs_only_use_stride_anchors() -> None:
    names = _names(40)
    index = {name: i for i, name in enumerate(names)}
    stride = 6
    pairs = sequential_loop_pairs(names, overlap=15, loop_stride=stride)
    for a, b in pairs:
        assert index[a] % stride == 0
        assert index[b] % stride == 0


def test_loop_pairs_sorted_and_ordered() -> None:
    names = _names(40)
    index = {name: i for i, name in enumerate(names)}
    pairs = sequential_loop_pairs(names, overlap=15, loop_stride=6)
    ranks = [(index[a], index[b]) for a, b in pairs]
    assert ranks == sorted(ranks)
    for a, b in pairs:
        assert index[a] < index[b]


def test_loop_pairs_cover_every_anchor() -> None:
    names = _names(48)
    stride = 6
    overlap = 15
    pairs = sequential_loop_pairs(names, overlap=overlap, loop_stride=stride)
    anchors = list(range(0, len(names), stride))
    # Every anchor that has at least one partner beyond `overlap` must appear.
    expected = {
        names[i]
        for i in anchors
        if any(abs(i - j) > overlap for j in anchors if j != i)
    }
    seen = {name for pair in pairs for name in pair}
    assert expected == seen


def test_loop_pairs_no_pairs_when_all_within_overlap() -> None:
    names = _names(10)
    # Anchors at 0 and 6; gap 6 <= overlap 15 -> no loop pairs.
    assert sequential_loop_pairs(names, overlap=15, loop_stride=6) == []


def test_loop_pairs_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        sequential_loop_pairs(_names(5), overlap=15, loop_stride=0)
    with pytest.raises(ValueError):
        sequential_loop_pairs(_names(5), overlap=-1, loop_stride=6)


def test_focal_from_fov_matches_formula() -> None:
    width = 1920
    fov = 65.0
    expected = (width / 2.0) / math.tan(math.radians(fov) / 2.0)
    assert focal_from_fov(width, fov) == pytest.approx(expected)
    # 90-degree fov -> focal == half width.
    assert focal_from_fov(1000, 90.0) == pytest.approx(500.0)


def _write_png(path: Path) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.full((8, 8), 255, dtype=np.uint8))


def test_stage_masks_uses_colmap_naming(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    eroded_dir = tmp_path / "eroded"
    colmap_dir = tmp_path / "colmap_masks"
    frame_paths = [frames_dir / "frame_0000.jpg", frames_dir / "frame_0001.jpg"]
    for fp in frame_paths:
        _write_png(fp)
    # Eroded masks are named "<stem>_mask.png".
    _write_png(eroded_dir / "frame_0000_mask.png")
    _write_png(eroded_dir / "frame_0001_mask.png")

    report = stage_masks(frame_paths, eroded_dir, colmap_dir)

    # COLMAP wants "<image_name>.png" (i.e. keep the original extension).
    assert (colmap_dir / "frame_0000.jpg.png").is_file()
    assert (colmap_dir / "frame_0001.jpg.png").is_file()
    assert report["staged_count"] == 2
    assert report["missing_masks"] == []


def test_stage_masks_reports_missing(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    eroded_dir = tmp_path / "eroded"
    colmap_dir = tmp_path / "colmap_masks"
    frame_paths = [frames_dir / "a.png", frames_dir / "b.png"]
    for fp in frame_paths:
        _write_png(fp)
    _write_png(eroded_dir / "a_mask.png")  # only one mask present

    report = stage_masks(frame_paths, eroded_dir, colmap_dir)

    assert (colmap_dir / "a.png.png").is_file()
    assert not (colmap_dir / "b.png.png").exists()
    assert report["staged"] == ["a.png"]
    assert report["missing_masks"] == ["b.png"]


@pytest.mark.skip(reason="integration, exercised by driver")
def test_run_masked_sfm_end_to_end() -> None:  # pragma: no cover
    # Requires a real masked video's frames + a full pycolmap run; too slow for
    # unit tests. The driver validates >=80% registration on real captures.
    ...
