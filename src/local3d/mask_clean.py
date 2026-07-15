"""Deterministic clean-up of raw object masks from hand-held captures.

The matting model (u2netp via rembg) segments *foreground*, which for a
hand-held object routinely includes the holding hand/forearm entering from a
frame border and sometimes wrapping fingers or face skin.  This module turns
one raw mask into an object-only mask by:

1. binarising and filling internal holes;
2. pruning arm/wrist necks that connect the object to a frame border, keeping
   the central "core" component and reconstructing the object back into it via
   constrained dilation into the original mask;
3. optionally suppressing skin-coloured pixels that wrap the object boundary
   (skipped entirely when the object itself is skin-coloured, e.g. a pink
   plush, to avoid deleting the subject).

Honest limits: this is morphology + a fixed YCrCb skin gate, not a learned
matte.  It cannot recover an object fused edge-to-edge with a same-width arm
(no thin neck to sever), and the skin gate is a coarse colour heuristic that
fails on skin-toned objects — for those the suppression step self-disables and
sets a report flag rather than eating the subject.  Everything is CPU-only,
single-threaded, and free of unseeded randomness or wall-clock, so identical
input arrays reproduce identical output bytes.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy import ndimage

# Morphology / suppression tunables (fractions of the mask bbox diagonal).
OPEN_FRAC = 0.03            # kernel to sever thin wrist/arm necks (~3%).
SKIN_BOUNDARY_FRAC = 0.06   # "near boundary" band for skin removal (~6%).
SKIN_SKIP_FRACTION = 0.55   # skip skin suppression above this skin-in-mask ratio.
ARM_PRUNE_FLAG_FRACTION = 0.03  # flag 'arm_pruned' when pruning removes >3% of raw.
AREA_OUTLIER_FRACTION = 0.30    # sequence area-deviation threshold.
AREA_MEDIAN_WINDOW = 9          # rolling-median window for the sequence pass.

# YCrCb skin gate (inclusive), matching the module contract.
_CR_LOW, _CR_HIGH = 133, 173
_CB_LOW, _CB_HIGH = 77, 127


def _binarize(mask: np.ndarray) -> np.ndarray:
    """Boolean foreground from bool / 0-1 / 0-255 masks."""

    array = np.asarray(mask)
    if array.dtype == bool:
        return array.copy()
    if array.ndim == 3:
        array = array[..., 0]
    peak = float(array.max()) if array.size else 0.0
    if peak <= 1.0:
        return array > 0
    return array >= 128


def _bbox_diag(binary: np.ndarray) -> float:
    """Diagonal (px) of the tight bounding box of the True pixels; 0 if empty."""

    rows = np.flatnonzero(binary.any(axis=1))
    cols = np.flatnonzero(binary.any(axis=0))
    if not len(rows) or not len(cols):
        return 0.0
    height = float(rows[-1] - rows[0] + 1)
    width = float(cols[-1] - cols[0] + 1)
    return float(np.hypot(height, width))


def _odd_kernel_size(value: float, minimum: int = 3) -> int:
    size = max(int(round(value)), minimum)
    return size if size % 2 == 1 else size + 1


def _ellipse_kernel(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    if not binary.any():
        return binary
    return ndimage.binary_fill_holes(binary)


def _largest_component(binary: np.ndarray) -> np.ndarray:
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(binary, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # ignore background
    return labels == int(np.argmax(sizes))


def _component_count(binary: np.ndarray) -> int:
    if not binary.any():
        return 0
    count, _ = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    return int(count - 1)


def _border_touch(binary: np.ndarray) -> float:
    if binary.size == 0:
        return 0.0
    border = np.concatenate(
        (binary[0], binary[-1], binary[:, 0], binary[:, -1])
    )
    return float(border.mean())


def _prune_arm(filled: np.ndarray, diag: float) -> np.ndarray:
    """Sever thin border-connected necks, keep the core, reconstruct the object.

    Opening with an elliptical kernel (~OPEN_FRAC of the bbox diagonal) cuts
    thin wrist/arm necks; among the resulting islands the *core* is the one
    maximising ``area * (centroid distance-to-border)`` — i.e. big and central,
    not a border-hugging limb.  The core is dilated by the same kernel and
    intersected with the original filled mask (constrained dilation) so the
    full object regrows without re-absorbing the pruned arm, which lies beyond
    the kernel reach of the core.
    """

    if not filled.any() or diag <= 0.0:
        return filled

    size = _odd_kernel_size(OPEN_FRAC * diag)
    kernel = _ellipse_kernel(size)
    opened = cv2.morphologyEx(
        filled.astype(np.uint8), cv2.MORPH_OPEN, kernel
    ).astype(bool)
    if not opened.any():
        # Object thinner than the kernel; fall back to the largest island.
        return _largest_component(filled)

    height, width = filled.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        opened.astype(np.uint8), connectivity=8
    )
    best_label = 0
    best_score = -1.0
    for label in range(1, count):
        area = float(stats[label, cv2.CC_STAT_AREA])
        cx, cy = centroids[label]
        border_dist = min(cx, width - 1 - cx, cy, height - 1 - cy)
        score = area * (border_dist + 1.0)
        if score > best_score:
            best_score = score
            best_label = label
    core = labels == best_label

    grown = cv2.dilate(core.astype(np.uint8), kernel).astype(bool) & filled
    grown = _largest_component(grown)
    return _fill_holes(grown)


def _skin_gate(image_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]
    return (
        (cr >= _CR_LOW) & (cr <= _CR_HIGH) & (cb >= _CB_LOW) & (cb <= _CB_HIGH)
    )


def clean_mask(
    mask: np.ndarray, image_bgr: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clean one frame's raw mask into a tight object-only mask + report.

    Returns ``(tight_bool_mask, report)`` where the report carries per-frame
    ``coverage``, ``border_touch``, ``removed_fraction``, ``components`` and a
    ``flags`` list (``'arm_pruned'`` when pruning removed more than 3% of raw
    pixels, ``'skin_suppression_skipped'`` when the object reads as skin).
    """

    binary = _binarize(mask)
    height, width = binary.shape
    raw_count = int(binary.sum())
    flags: list[str] = []

    if raw_count == 0:
        report = {
            "coverage": 0.0,
            "border_touch": 0.0,
            "removed_fraction": 0.0,
            "components": 0,
            "flags": flags,
        }
        return np.zeros((height, width), dtype=bool), report

    filled = _fill_holes(binary)
    diag = _bbox_diag(filled)

    pruned = _prune_arm(filled, diag)
    pruned_removed = int(filled.sum()) - int(pruned.sum())
    if raw_count and pruned_removed / raw_count > ARM_PRUNE_FLAG_FRACTION:
        flags.append("arm_pruned")

    tight = pruned
    if image_bgr is not None and tight.any():
        if image_bgr.shape[:2] != (height, width):
            image_bgr = cv2.resize(
                image_bgr, (width, height), interpolation=cv2.INTER_NEAREST
            )
        skin = _skin_gate(image_bgr)
        mask_area = int(tight.sum())
        skin_fraction = float((skin & tight).sum()) / max(mask_area, 1)
        if skin_fraction > SKIN_SKIP_FRACTION:
            flags.append("skin_suppression_skipped")
        else:
            dist = cv2.distanceTransform(
                tight.astype(np.uint8), cv2.DIST_L2, 5
            )
            near_boundary = dist < (SKIN_BOUNDARY_FRAC * diag)
            remove = skin & tight & near_boundary
            if remove.any():
                carved = tight & ~remove
                carved = _largest_component(carved)
                tight = _fill_holes(carved)

    final_count = int(tight.sum())
    report = {
        "coverage": round(float(tight.mean()), 4),
        "border_touch": round(_border_touch(tight), 4),
        "removed_fraction": round((raw_count - final_count) / max(raw_count, 1), 4),
        "components": _component_count(tight),
        "flags": flags,
    }
    return tight, report


def erode_for_sfm(
    mask: np.ndarray, *, min_px: int = 12, frac_of_diag: float = 0.015
) -> np.ndarray:
    """Erode a tight mask inward for conservative SfM/silhouette use.

    Kernel size is ``max(min_px, frac_of_diag * bbox_diagonal)`` (elliptical).
    Guarantees a strict inward shrink for any non-empty finite mask.
    """

    binary = _binarize(mask)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    diag = _bbox_diag(binary)
    size = _odd_kernel_size(max(float(min_px), frac_of_diag * diag), minimum=3)
    kernel = _ellipse_kernel(size)
    eroded = cv2.erode(binary.astype(np.uint8), kernel).astype(bool)
    return eroded


def clean_mask_sequence(
    masks: list[np.ndarray], images: list[np.ndarray] | None = None
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Clean a whole capture: per-frame clean-up plus a sequence outlier pass.

    Returns ``(tight_masks, eroded_masks, report)``.  The report holds a
    per-frame entry list plus sequence-level fields: frames whose tight-mask
    area deviates more than 30% from the rolling median (window 9) get an
    ``'area_outlier'`` flag (never dropped — the caller decides), and the
    sequence-level ``regrip_outlier`` flag/flag-list is set when any exist.
    """

    if images is not None and len(images) != len(masks):
        raise ValueError("images and masks must have equal length")

    tight_masks: list[np.ndarray] = []
    frame_reports: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        image = images[index] if images is not None else None
        tight, report = clean_mask(mask, image)
        tight_masks.append(tight)
        frame_reports.append(report)

    eroded_masks = [erode_for_sfm(tight) for tight in tight_masks]

    areas = np.array([int(tight.sum()) for tight in tight_masks], dtype=np.float64)
    outlier_frames: list[int] = []
    half = AREA_MEDIAN_WINDOW // 2
    for index in range(len(areas)):
        lo = max(0, index - half)
        hi = min(len(areas), index + half + 1)
        median = float(np.median(areas[lo:hi]))
        if median <= 0.0:
            continue
        if abs(areas[index] - median) / median > AREA_OUTLIER_FRACTION:
            outlier_frames.append(index)
            frame_reports[index]["flags"].append("area_outlier")

    sequence_flags: list[str] = []
    if outlier_frames:
        sequence_flags.append("regrip_outlier")

    report: dict[str, Any] = {
        "frames": frame_reports,
        "frame_count": len(frame_reports),
        "area_outlier_frames": outlier_frames,
        "regrip_outlier": bool(outlier_frames),
        "flags": sequence_flags,
    }
    return tight_masks, eroded_masks, report
