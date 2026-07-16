"""Monocular relative depth (Depth Anything V2 Small ONNX) aligned to SfM points.

A small local monocular depth model gives per-pixel *relative* disparity — it
has no absolute scale and no per-frame consistency.  This module turns that
into a metric depth map for one posed frame by robustly fitting an affine
map ``a * disparity + b ~= 1 / z`` against the sparse SfM points that project
into the frame (their camera-frame ``z`` is known).  Inverse depth is linear in
disparity for a pinhole camera up to an affine ambiguity, so a two-parameter
fit is the right model; the fit is done with IRLS + a Tukey biweight so that
gross monocular-depth errors (a common failure on thin or reflective parts) do
not drag the scale.

Honest limits: the recovered depth is only as good as the monocular prediction
between the sparse anchors; regions with no nearby SfM points inherit whatever
the network guessed and are down-weighted (not trusted) by the fusion stage.
The affine model cannot fix a network that gets the *ordering* wrong — the
``ok`` flag and ``rms_rel`` in the returned dict exist so callers can reject
such frames instead of fusing them.

Conventions (shared with :mod:`local3d.recon_common`):

- ``disparity`` is float32, same H x W as the input image, larger = nearer
  (it is inverse-depth-like, exactly what Depth Anything V2 emits).
- Projected points arrive as pixel coordinates ``u, v`` (floats) plus their
  camera-frame depth ``z`` (> 0); disparity is read back with
  :func:`recon_common.bilinear_sample`.

Everything is deterministic and CPU-only (onnxruntime CPUExecutionProvider,
single fixed input size, no randomness).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from local3d.recon_common import bilinear_sample

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Depth Anything V2 uses a patch size of 14; the input side must be a multiple.
_PATCH = 14
_DEFAULT_SIDE = 518  # 14 * 37, the model's training resolution


class DepthEstimator:
    """Cached onnxruntime session for Depth Anything V2 Small (int8, CPU).

    The session is built once and reused across frames — building it per frame
    (as a naive port of :func:`scripts.depth_relief.predict_depth` would) is the
    dominant cost, so callers should keep one estimator for a whole clip.
    """

    def __init__(self, model_path: Path, *, threads: int = 4) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        # Deterministic single-run graph; no parallel-execution reordering.
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_height, self.input_width = self._resolve_input_size(model_input.shape)

    @staticmethod
    def _resolve_input_size(shape: object) -> tuple[int, int]:
        """Introspect the fixed spatial size, defaulting to 518 for dynamic dims."""

        height = _DEFAULT_SIDE
        width = _DEFAULT_SIDE
        if isinstance(shape, (list, tuple)) and len(shape) == 4:
            if isinstance(shape[2], int) and shape[2] > 0:
                height = shape[2]
            if isinstance(shape[3], int) and shape[3] > 0:
                width = shape[3]
        # Clamp to a valid multiple of the patch size.
        height = max(_PATCH, height - height % _PATCH)
        width = max(_PATCH, width - width % _PATCH)
        return height, width

    def disparity(self, image_bgr: np.ndarray) -> np.ndarray:
        """Relative disparity for ``image_bgr`` (BGR uint8), float32 H x W.

        Larger values are nearer.  The result is resized back to the input
        resolution with cubic interpolation, matching ``predict_depth`` in
        ``scripts/depth_relief.py`` but with the session reused.
        """

        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image_bgr must be an H x W x 3 BGR array")
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(
            rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC
        )
        tensor = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
        prediction = self.session.run(
            None, {self.input_name: tensor.astype(np.float32)}
        )[0]
        prediction = np.asarray(prediction)
        prediction = prediction.reshape(prediction.shape[-2], prediction.shape[-1])
        resized = cv2.resize(
            prediction.astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC
        )
        return resized.astype(np.float32)


def _robust_scale(residual: np.ndarray, target: np.ndarray) -> float:
    """``1.4826 * MAD`` of the residuals, floored so a near-perfect fit is stable."""

    mad = float(np.median(np.abs(residual - np.median(residual))))
    scale = 1.4826 * mad
    floor = 1e-6 * max(float(np.median(np.abs(target))), 1e-9)
    return max(scale, floor)


def _theil_sen(d: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Theil-Sen line fit: median pairwise slope + median intercept.

    Robust to high-leverage outliers (extreme ``d`` values) that break an
    ordinary least-squares initialisation, which matters because monocular
    disparity does produce gross per-region errors.  Points are subsampled
    deterministically to bound the pairwise cost.
    """

    n = len(d)
    if n > 400:
        pick = np.unique(np.linspace(0, n - 1, 400).astype(np.int64))
        ds, ys = d[pick], target[pick]
    else:
        ds, ys = d, target
    i, j = np.triu_indices(len(ds), k=1)
    delta = ds[j] - ds[i]
    good = np.abs(delta) > 1e-12
    if not good.any():
        return 0.0, float(np.median(target))
    slopes = (ys[j][good] - ys[i][good]) / delta[good]
    slope = float(np.median(slopes))
    intercept = float(np.median(target - slope * d))
    return slope, intercept


def align_disparity_to_points(
    disp: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    z: np.ndarray,
    *,
    iters: int = 12,
) -> dict:
    """Robustly fit ``a * disparity + b ~= 1 / z`` at the projected SfM points.

    A Theil-Sen fit initialises the affine map (robust to high-leverage
    monocular-depth outliers), then IRLS with a Tukey biweight (``c = 4.685``,
    robust scale ``1.4826 * MAD`` of the residuals) refines it.  ``disp`` is
    sampled bilinearly at ``(u, v)``; ``z`` is the known camera-frame depth.

    Returns ``{'a', 'b', 'inliers', 'rms_rel', 'ok'}`` where ``inliers`` is the
    number of points inside the final biweight support, ``rms_rel`` is the RMS
    relative depth error over those inliers, and ``ok`` is ``False`` when there
    are fewer than 40 inliers or ``rms_rel > 0.08`` (i.e. the frame should not
    be fused).
    """

    d = np.asarray(bilinear_sample(disp, u, v), dtype=np.float64).reshape(-1)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    failure = {"a": 0.0, "b": 0.0, "inliers": 0, "rms_rel": float("inf"), "ok": False}
    valid = np.isfinite(d) & np.isfinite(z) & (z > 1e-9)
    d = d[valid]
    z = z[valid]
    if d.size < 2:
        return failure

    target = 1.0 / z
    design = np.stack([d, np.ones_like(d)], axis=1)
    c = 4.685

    slope, intercept = _theil_sen(d, target)
    params = np.array([slope, intercept], dtype=np.float64)

    for _ in range(max(int(iters), 1)):
        residual = design @ params - target
        scale = _robust_scale(residual, target)
        normalized = residual / (c * scale)
        weights = np.where(np.abs(normalized) < 1.0, (1.0 - normalized**2) ** 2, 0.0)
        normal = design.T @ (design * weights[:, None])
        rhs = design.T @ (weights * target)
        try:
            params = np.linalg.solve(normal, rhs)
        except np.linalg.LinAlgError:
            return failure

    a, b = float(params[0]), float(params[1])
    residual = design @ params - target
    scale = _robust_scale(residual, target)
    inlier_mask = np.abs(residual) < c * scale
    inliers = int(np.count_nonzero(inlier_mask))
    if inliers < 2:
        return failure

    predicted_inv = a * d[inlier_mask] + b
    safe = np.maximum(predicted_inv, 1e-6)
    predicted_depth = 1.0 / safe
    rel = (predicted_depth - z[inlier_mask]) / z[inlier_mask]
    rms_rel = float(np.sqrt(np.mean(rel**2)))
    ok = bool(inliers >= 40 and rms_rel <= 0.08)
    return {"a": a, "b": b, "inliers": inliers, "rms_rel": rms_rel, "ok": ok}


def disparity_to_depth(disp: np.ndarray, a: float, b: float) -> np.ndarray:
    """Convert relative disparity to metric depth via ``1 / (a * disp + b)``.

    Depth is 0 wherever ``a * disp + b <= 1e-6`` (points at or behind the
    affine horizon, which have no meaningful metric depth).
    """

    disp = np.asarray(disp, dtype=np.float32)
    inverse_depth = a * disp.astype(np.float64) + b
    depth = np.zeros_like(inverse_depth)
    valid = inverse_depth > 1e-6
    depth[valid] = 1.0 / inverse_depth[valid]
    return depth.astype(np.float32)
