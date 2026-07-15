from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.monodepth import (
    DepthEstimator,
    align_disparity_to_points,
    disparity_to_depth,
)

MODEL_PATH = ROOT / "runs" / "models" / "depth_anything_v2_small_int8.onnx"


def _disparity_field(u, v, values, height, width):
    """Sparse disparity image with ``values`` planted at integer pixel coords."""
    disp = np.zeros((height, width), dtype=np.float32)
    disp[v.astype(np.int64), u.astype(np.int64)] = values.astype(np.float32)
    return disp


def test_align_recovers_scale_and_offset_under_gross_outliers():
    rng = np.random.default_rng(7)
    height, width = 220, 320
    a_true, b_true = 0.8, 0.15

    # Distinct integer pixels so bilinear sampling returns the planted value.
    coords = set()
    while len(coords) < 300:
        coords.add((int(rng.integers(2, width - 2)), int(rng.integers(2, height - 2))))
    coords = sorted(coords)
    u = np.array([c[0] for c in coords], dtype=np.float64)
    v = np.array([c[1] for c in coords], dtype=np.float64)
    n = len(u)

    z = rng.uniform(2.0, 4.0, size=n)
    inverse_depth = 1.0 / z
    disp_values = (inverse_depth - b_true) / a_true
    disp_values += rng.normal(0.0, 1e-4, size=n)  # mild inlier noise

    # 20% gross outliers: corrupt the sampled disparity so the affine fit breaks.
    outliers = rng.choice(n, size=int(0.2 * n), replace=False)
    disp_values[outliers] += rng.uniform(1.0, 3.0, size=len(outliers)) * rng.choice(
        [-1.0, 1.0], size=len(outliers)
    )

    disp = _disparity_field(u, v, disp_values, height, width)
    result = align_disparity_to_points(disp, u, v, z)

    assert result["ok"] is True
    assert abs(result["a"] - a_true) < 0.02 * a_true
    assert abs(result["b"] - b_true) < 0.02 * abs(b_true)
    assert result["rms_rel"] < 0.08
    # Roughly the clean 80% should be retained as inliers.
    assert result["inliers"] >= int(0.7 * n)


def test_align_rejects_frame_with_too_few_points():
    rng = np.random.default_rng(1)
    height, width = 64, 64
    u = rng.integers(2, width - 2, size=10).astype(np.float64)
    v = rng.integers(2, height - 2, size=10).astype(np.float64)
    z = rng.uniform(2.0, 4.0, size=10)
    disp = _disparity_field(u, v, (1.0 / z - 0.1) / 0.5, height, width)
    result = align_disparity_to_points(disp, u, v, z)
    assert result["ok"] is False
    assert result["inliers"] < 40


def test_disparity_to_depth_inverts_affine_and_zeros_invalid():
    disp = np.array([[0.5, 0.25], [0.0, -0.5]], dtype=np.float32)
    a, b = 0.8, 0.15
    depth = disparity_to_depth(disp, a, b)
    assert depth.dtype == np.float32
    # Valid entries invert a*d+b.
    np.testing.assert_allclose(depth[0, 0], 1.0 / (0.8 * 0.5 + 0.15), rtol=1e-5)
    np.testing.assert_allclose(depth[0, 1], 1.0 / (0.8 * 0.25 + 0.15), rtol=1e-5)
    # a*d+b = -0.25 <= 1e-6 -> invalid -> 0.
    assert depth[1, 1] == 0.0


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="depth model not present")
def test_depth_estimator_runs_and_returns_matching_size():
    estimator = DepthEstimator(MODEL_PATH, threads=2)
    height, width = 96, 128
    # Synthetic image with a bright central blob (a plausible near object).
    image = np.zeros((height, width, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    blob = ((xx - width / 2) ** 2 + (yy - height / 2) ** 2) < (0.25 * width) ** 2
    image[blob] = (200, 180, 160)

    disp = estimator.disparity(image)
    assert disp.shape == (height, width)
    assert disp.dtype == np.float32
    assert np.isfinite(disp).all()
    # Reusing the cached session must give a bit-identical result.
    disp_again = estimator.disparity(image)
    np.testing.assert_array_equal(disp, disp_again)
