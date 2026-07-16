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
   plush, to avoid deleting the subject);
4. optionally *depth pruning* with a monocular relative-disparity map: the held
   object is the nearest thing in view (a compact high-disparity blob around
   the mask core) while the holding forearm slopes away toward the frame
   border, so pixels whose disparity departs from the interior anchor's
   disparity by more than a fraction of the object-to-background disparity gap
   are severed.  This is the only step that can remove a same-toned, same-width
   arm that morphology + the skin gate cannot (the pink-plush-in-a-bare-hand
   case, verified on real data).

Honest limits: steps 1-3 are morphology + a fixed YCrCb skin gate, not a
learned matte.  They cannot recover an object fused edge-to-edge with a
same-width arm (no thin neck to sever), and the skin gate is a coarse colour
heuristic that fails on skin-toned objects.  Depth pruning fixes those cases
when a disparity map is supplied, but it is only as good as the monocular
prediction: it self-disables (flag ``'depth_prune_skipped'``) whenever the
disparity is too flat to separate object from arm, or the surviving component
would drop below a third of the mask, rather than eating the subject.  It also
*cannot* remove a hand that is genuinely coplanar with the object surface
(fingers wrapped over the front at the object's own depth) — depth carries no
signal there.  Everything is CPU-only, single-threaded, and free of unseeded
randomness or wall-clock, so identical input arrays reproduce identical output
bytes.

Depth-pruning band scale — DEVIATION FROM SPEC, driven by the data.  The task
spec proposed ``depth_ok = |disparity - d0| < ALPHA * (p95 - p5)`` with the
band scaled by the in-mask disparity spread.  On the real plush frames
(runs/upgrade_v1/plush) that spread is dominated by *how much arm is in view*:
when little arm is present (frame 0000) the spread is ~1.4 and ALPHA=0.35 shaves
the body/ears; when a lot of far arm is present (frame 0120) the spread is ~3.9
and the same ALPHA leaves the arm inside the band.  No single ALPHA on that
spread could both keep the ears and sever the arm, nor keep the compact tin
(a distance-transform "core" spread failed too — the forearm is as thick as the
object).  The fix that generalises is to scale the band by the object-to-
background disparity gap ``d0 - d_bg`` (``d_bg`` = median disparity *outside*
the mask), which is independent of arm extent and affine-robust (offset+scale
covariant).  Constants chosen empirically on the five plush + three tin frames
(overlays reviewed by eye):

- ``DEPTH_ALPHA = 0.30`` — band half-width as a fraction of ``d0 - d_bg``.  At
  0.30 the forearm/hand wedge and all face pixels are removed on 4 of the 5
  plush frames, both ears survive, and the tin keeps 98.8-100% of its mask.
  0.20-0.25 begins clipping the floppy-ear tips; >0.35 starts to re-admit the
  near end of the forearm.
- ``DEPTH_ANCHOR_DISK_FRAC = 0.35`` — anchor-disk radius as a fraction of the
  max distance-transform value; a disk this size sits wholly inside the object
  body so its median disparity is a clean body-depth reference ``d0`` (per spec).
- ``DEPTH_BLUR_FRAC = 0.012`` — Gaussian sigma (fraction of bbox diag) applied
  to the disparity *only* for the depth-ok test; it smooths monocular noise on
  thin parts (ears) so their locally-consistent depth is not shattered into
  sub-band speckle, without moving the body/arm decision.
- ``DEPTH_CLOSE_FRAC = 0.03`` — kernel (fraction of bbox diag) for a
  morphological close of the in-band region before the component step: it
  bridges the hairline below-band crease where a floppy ear roots into the head
  (which otherwise splits the ear into its own component and drops it), while
  being far too small to bridge the wide depth gap the far arm makes.
- ``DEPTH_DILATE_FRAC = 0.015`` x ``DEPTH_DILATE_ITERS = 2`` — geodesic
  (mask-constrained) dilation to recover the soft matte edge the band test
  nibbles, without flowing back down the severed arm.
- ``DEPTH_KEEP_FRACTION = 0.35`` / ``DEPTH_PRUNE_FLAG_FRACTION = 0.03`` —
  fail-safe floor and reporting threshold (per spec).  On the tin the band
  keeps ~100% of the mask, so pruning is a no-op (no harm), exactly as wanted.
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

# Depth-pruning tunables (see module docstring for the empirical rationale).
DEPTH_ALPHA = 0.30              # band half-width as a fraction of (d0 - d_bg).
DEPTH_ANCHOR_DISK_FRAC = 0.35   # anchor-disk radius as a fraction of max dist.
DEPTH_BLUR_FRAC = 0.012         # Gaussian sigma (frac of diag) for the band test.
DEPTH_CLOSE_FRAC = 0.03         # close kernel (frac of diag) to bridge thin creases.
DEPTH_DILATE_FRAC = 0.015       # constrained-dilation kernel (frac of diag).
DEPTH_DILATE_ITERS = 2          # geodesic dilation iterations to recover edges.
DEPTH_KEEP_FRACTION = 0.35      # fail-safe: keep un-pruned below this survival.
DEPTH_PRUNE_FLAG_FRACTION = 0.03  # flag 'depth_pruned' when >3% removed.

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


def _component_containing(binary: np.ndarray, row: int, col: int) -> np.ndarray:
    """Largest-context component of ``binary`` at/nearest to ``(row, col)``.

    Returns the connected component whose label sits at ``(row, col)``; if that
    pixel is background, the component containing the nearest True pixel (by L2
    distance transform of the background) is returned instead.  Empty result
    when ``binary`` has no True pixel.
    """

    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(binary, dtype=bool)
    label = int(labels[row, col])
    if label == 0:
        # Anchor fell in a gap: snap to the nearest foreground pixel (rare, so a
        # vectorised argmin over the True pixels is fine and fully deterministic).
        ys, xs = np.nonzero(binary)
        nearest = int(np.argmin((ys - row) ** 2 + (xs - col) ** 2))
        label = int(labels[ys[nearest], xs[nearest]])
        if label == 0:
            return np.zeros_like(binary, dtype=bool)
    return labels == label


def _depth_prune(
    mask: np.ndarray, disparity: np.ndarray, diag: float
) -> tuple[np.ndarray, float, str | None]:
    """Sever the holding arm using monocular relative disparity.

    ``mask`` is the current tight (bool) mask, ``disparity`` a float32 map
    (same H x W, larger = nearer).  Returns ``(pruned_mask, removed_fraction,
    flag)`` where ``flag`` is ``'depth_pruned'`` (>3% removed), ``None`` (kept,
    negligible change) or ``'depth_prune_skipped'`` (fail-safe fired, mask
    returned unchanged).  See the module docstring for the algorithm and the
    empirically chosen constants.
    """

    mask_area = int(mask.sum())
    if mask_area == 0 or diag <= 0.0:
        return mask, 0.0, "depth_prune_skipped"

    disparity = np.asarray(disparity, dtype=np.float32)
    if disparity.shape != mask.shape:
        disparity = cv2.resize(
            disparity, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    # 1. Anchor: the deepest-interior point of the mask.
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    max_dist = float(dist.max())
    anchor_row, anchor_col = np.unravel_index(int(np.argmax(dist)), dist.shape)

    # 2. Body-depth reference: median disparity in a disk around the anchor.
    radius = max(int(round(DEPTH_ANCHOR_DISK_FRAC * max_dist)), 1)
    disk = np.zeros(mask.shape, dtype=np.uint8)
    cv2.circle(disk, (int(anchor_col), int(anchor_row)), radius, 1, -1)
    disk_region = (disk > 0) & mask
    if not disk_region.any():
        disk_region = mask
    d0 = float(np.median(disparity[disk_region]))

    # 3. Depth band: keep pixels within DEPTH_ALPHA * (d0 - d_bg) of the anchor
    #    depth.  The scale is the object-to-background disparity gap, NOT the
    #    in-mask (p95-p5) spread the naive spec uses: p95-p5 is dominated by how
    #    far/large the arm is, which couples the band width to the arm and makes
    #    a single ALPHA unable to both keep a compact object (tin) and sever a
    #    far arm (verified on runs/upgrade_v1 — see the module docstring).  The
    #    object-to-background gap is arm-independent and affine-robust (offset +
    #    scale covariant), so one ALPHA generalises across frames.
    inside = disparity[mask]
    p5, p95 = np.percentile(inside, [5.0, 95.0])
    spread = float(p95 - p5)
    if spread <= 1e-6:
        # Disparity perfectly flat: no depth signal to separate object from arm.
        return mask, 0.0, "depth_prune_skipped"

    outside = disparity[~mask]
    if outside.size >= 64:
        d_bg = float(np.median(outside))
    else:
        d_bg = float(p5)  # mask fills the frame: fall back to the far in-mask tail.
    scale = DEPTH_ALPHA * (d0 - d_bg)
    if scale <= 1e-6:
        # Object not resolvably nearer than the background — do not guess.
        return mask, 0.0, "depth_prune_skipped"

    sigma = max(DEPTH_BLUR_FRAC * diag, 0.5)
    disp_smooth = cv2.GaussianBlur(disparity, (0, 0), sigmaX=sigma, sigmaY=sigma)
    depth_ok = np.abs(disp_smooth - d0) < scale

    # Bridge thin below-band creases (e.g. the recessed fold where a floppy ear
    # roots into the head) with a morphological close so the protrusion stays in
    # the anchor's component.  The kernel is small, so it cannot bridge the WIDE
    # depth gap the far arm makes — only hairline severances of otherwise
    # depth-consistent object parts.
    close_kernel = _ellipse_kernel(_odd_kernel_size(DEPTH_CLOSE_FRAC * diag))
    depth_ok = (
        cv2.morphologyEx(
            (depth_ok & mask).astype(np.uint8), cv2.MORPH_CLOSE, close_kernel
        ).astype(bool)
        & mask
    )

    # 4. Candidate: the in-band component holding the anchor, re-grown into mask.
    candidate = _component_containing(depth_ok, int(anchor_row), int(anchor_col))
    if not candidate.any():
        return mask, 0.0, "depth_prune_skipped"

    kernel = _ellipse_kernel(_odd_kernel_size(DEPTH_DILATE_FRAC * diag))
    grown = candidate.astype(np.uint8)
    for _ in range(DEPTH_DILATE_ITERS):
        grown = cv2.dilate(grown, kernel) & mask.astype(np.uint8)
    candidate = _fill_holes(grown > 0)

    # 5. Fail-safe: reject an implausibly small or anchor-losing survivor.
    candidate_area = int(candidate.sum())
    if candidate_area < DEPTH_KEEP_FRACTION * mask_area or not candidate[anchor_row, anchor_col]:
        return mask, 0.0, "depth_prune_skipped"

    removed_fraction = (mask_area - candidate_area) / mask_area
    flag = "depth_pruned" if removed_fraction > DEPTH_PRUNE_FLAG_FRACTION else None
    return candidate, float(removed_fraction), flag


def clean_mask(
    mask: np.ndarray,
    image_bgr: np.ndarray | None = None,
    disparity: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Clean one frame's raw mask into a tight object-only mask + report.

    Returns ``(tight_bool_mask, report)`` where the report carries per-frame
    ``coverage``, ``border_touch``, ``removed_fraction``, ``components``,
    ``depth_pruned_fraction`` and a ``flags`` list (``'arm_pruned'`` when
    morphological pruning removed more than 3% of raw pixels,
    ``'skin_suppression_skipped'`` when the object reads as skin,
    ``'depth_pruned'`` when depth pruning removed more than 3% of the mask, and
    ``'depth_prune_skipped'`` when the depth fail-safe fired).

    When ``disparity`` (float32 H x W, larger = nearer, same size as ``mask``)
    is supplied, a monocular depth-pruning pass runs *after* the morphology and
    skin steps to sever a same-toned, same-width holding arm that those steps
    cannot — see :func:`_depth_prune` and the module docstring.  Without it the
    behaviour is exactly as before (backwards compatible).
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
            "depth_pruned_fraction": 0.0,
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

    depth_pruned_fraction = 0.0
    if disparity is not None and tight.any():
        # Depth pruning is measured against — and re-grows into — the current
        # (post-skin) mask, so it never resurrects skin-suppressed pixels.
        diag_tight = _bbox_diag(tight)
        tight, depth_pruned_fraction, depth_flag = _depth_prune(
            tight, disparity, diag_tight
        )
        if depth_flag is not None:
            flags.append(depth_flag)

    final_count = int(tight.sum())
    report = {
        "coverage": round(float(tight.mean()), 4),
        "border_touch": round(_border_touch(tight), 4),
        "removed_fraction": round((raw_count - final_count) / max(raw_count, 1), 4),
        "components": _component_count(tight),
        "depth_pruned_fraction": round(float(depth_pruned_fraction), 4),
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
    masks: list[np.ndarray],
    images: list[np.ndarray] | None = None,
    disparities: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    """Clean a whole capture: per-frame clean-up plus a sequence outlier pass.

    Returns ``(tight_masks, eroded_masks, report)``.  The report holds a
    per-frame entry list plus sequence-level fields: frames whose tight-mask
    area deviates more than 30% from the rolling median (window 9) get an
    ``'area_outlier'`` flag (never dropped — the caller decides), and the
    sequence-level ``regrip_outlier`` flag/flag-list is set when any exist.

    ``disparities`` (when given, one float32 map per frame) enables the
    monocular depth-pruning pass in :func:`clean_mask`; pass ``None`` entries
    to skip individual frames.
    """

    if images is not None and len(images) != len(masks):
        raise ValueError("images and masks must have equal length")
    if disparities is not None and len(disparities) != len(masks):
        raise ValueError("disparities and masks must have equal length")

    tight_masks: list[np.ndarray] = []
    frame_reports: list[dict[str, Any]] = []
    for index, mask in enumerate(masks):
        image = images[index] if images is not None else None
        disparity = disparities[index] if disparities is not None else None
        tight, report = clean_mask(mask, image, disparity)
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
