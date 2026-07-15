"""Automatic object masks for difficult held-object captures.

The primary pipeline uses Apple Vision foreground instances.  When Vision
merges a held object with the person, this module can recover a centrally held
rigid object without a prompt or reviewed seed.  A small local U2Net matte
supplies the outer foreground boundary; chroma evidence or a conservative
non-skin appearance seed identifies the object and prevents the person's larger
silhouette from winning.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np
from PIL import Image


class MaskingError(RuntimeError):
    """Raised when the automatic mask backend is unavailable."""


def _u2netp_model_path(model_name: str) -> Path:
    """Return rembg's cache path without importing or invoking rembg.

    rembg resolves its model directory as ``U2NET_HOME`` when set, otherwise
    ``$XDG_DATA_HOME/.u2net`` (with ``~`` as the XDG fallback).  Resolving the
    same path here lets the local-only pipeline fail before rembg's session
    factory has an opportunity to call its download helper.
    """

    if model_name != "u2netp":
        raise MaskingError(
            "the local color-anchored fallback supports only the cached u2netp model"
        )
    cache_value = os.environ.get("U2NET_HOME")
    if cache_value is None:
        cache_value = os.path.join(os.environ.get("XDG_DATA_HOME", "~"), ".u2net")
    return Path(os.path.expanduser(cache_value)).resolve() / "u2netp.onnx"


def _u2netp_cache_path_source() -> str:
    if "U2NET_HOME" in os.environ:
        return "U2NET_HOME"
    if "XDG_DATA_HOME" in os.environ:
        return "XDG_DATA_HOME"
    return "rembg_default"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_stream:
        for block in iter(lambda: model_stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_cached_u2netp_model(model_name: str) -> tuple[Path, str]:
    model_path = _u2netp_model_path(model_name)
    if not model_path.is_file() or model_path.stat().st_size <= 0:
        raise MaskingError(
            "local U2Net fallback requires a pre-cached, non-empty u2netp model at "
            f"{model_path}; automatic model downloads are disabled"
        )
    return model_path, _sha256_file(model_path)


def _load_rembg_backend() -> tuple[
    Callable[..., Any], Callable[..., Any], type[Any], str
]:
    """Load the optional runtime and return its installed package version."""

    try:
        from importlib.metadata import version

        from rembg import new_session, remove
        from rembg.sessions.u2netp import U2netpSession

        rembg_version = version("rembg")
    except Exception as exc:  # pragma: no cover - depends on optional installation
        raise MaskingError(
            "color-anchored fallback requires rembg/onnxruntime; "
            "install the segmentation extra"
        ) from exc
    return new_session, remove, U2netpSession, rembg_version


def _new_cached_u2netp_session(
    new_session: Callable[..., Any],
    session_class: type[Any],
    model_path: Path,
) -> Any:
    """Create a rembg session while making network retrieval unreachable.

    ``BaseSession.__init__`` always calls the selected session class's
    ``download_models`` method, even when the model is already cached.  For the
    duration of this single-threaded CLI initialization, redirect that method
    to the preflighted file and restore the original descriptor immediately.
    """

    original_download = vars(session_class).get("download_models")
    if original_download is None:
        raise MaskingError("installed rembg u2netp session has no model resolver")

    def cached_model(_class: type[Any], *_args: Any, **_kwargs: Any) -> str:
        if not model_path.is_file() or model_path.stat().st_size <= 0:
            raise MaskingError(
                f"cached u2netp model disappeared before session load: {model_path}"
            )
        return str(model_path)

    setattr(session_class, "download_models", classmethod(cached_model))
    try:
        return new_session("u2netp")
    finally:
        setattr(session_class, "download_models", original_download)


def _frame_paths(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def _largest_component(binary: np.ndarray) -> np.ndarray:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return np.zeros_like(binary, dtype=np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas)) + 1
    return (labels == label).astype(np.uint8)


def _fill_holes(binary: np.ndarray) -> np.ndarray:
    height, width = binary.shape
    flood = (binary * 255).copy()
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return np.where((binary > 0) | (holes > 0), 1, 0).astype(np.uint8)


def _green_anchor(frame: np.ndarray, coarse: np.ndarray) -> np.ndarray | None:
    """Return a conservative full-object mask from strong green evidence."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(frame)
    excess = green.astype(np.int16) - np.maximum(red, blue).astype(np.int16)
    chroma = (
        (hsv[..., 0] >= 28)
        & (hsv[..., 0] <= 100)
        & (hsv[..., 1] >= 42)
        & (hsv[..., 2] >= 28)
        & (excess >= 8)
        & (coarse > 0)
    ).astype(np.uint8)

    height, width = chroma.shape
    # Holder clothing and room decor are irrelevant; the capture contract puts
    # the object near the image center.  This is an automatic prior, not a
    # per-video coordinate supplied by an operator.
    central = np.zeros_like(chroma)
    central[int(0.08 * height):int(0.94 * height), int(0.08 * width):int(0.92 * width)] = 1
    chroma &= central
    chroma = cv2.morphologyEx(chroma, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(chroma, 8)
    accepted = np.zeros_like(chroma)
    image_area = float(height * width)
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(40, int(image_area * 0.00004)):
            continue
        center_x, center_y = centroids[label]
        normalized_distance = np.hypot(
            (center_x - width * 0.5) / (width * 0.5),
            (center_y - height * 0.5) / (height * 0.5),
        )
        if normalized_distance <= 0.78:
            accepted[labels == label] = 1

    points_yx = np.argwhere(accepted > 0)
    if len(points_yx) < max(250, int(image_area * 0.0004)):
        return None
    points_xy = points_yx[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(points_xy)
    hull_mask = np.zeros_like(chroma)
    cv2.fillConvexPoly(hull_mask, hull, 1)

    # A very small expansion includes anti-aliased object edges while remaining
    # much tighter than the merged person foreground instance.
    radius = max(2, int(round(min(height, width) * 0.006)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    hull_mask = cv2.dilate(hull_mask, kernel)
    refined = hull_mask & coarse
    refined = cv2.morphologyEx(
        refined, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    refined = _largest_component(refined)
    refined = _fill_holes(refined)
    coverage = float(refined.mean())
    if coverage < 0.008 or coverage > 0.46:
        return None
    border = np.concatenate((refined[0], refined[-1], refined[:, 0], refined[:, -1]))
    if float(border.mean()) > 0.04:
        return None
    return refined


def _has_green_evidence(frame: np.ndarray) -> bool:
    """Whether a frame contains enough central product-like green chroma."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(frame)
    excess = green.astype(np.int16) - np.maximum(red, blue).astype(np.int16)
    evidence = (
        (hsv[..., 0] >= 28)
        & (hsv[..., 0] <= 100)
        & (hsv[..., 1] >= 42)
        & (hsv[..., 2] >= 28)
        & (excess >= 8)
    ).astype(np.uint8)
    height, width = evidence.shape
    evidence[:int(0.08 * height)] = 0
    evidence[int(0.94 * height):] = 0
    evidence[:, :int(0.08 * width)] = 0
    evidence[:, int(0.92 * width):] = 0
    evidence = cv2.morphologyEx(evidence, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    minimum = max(250, int(height * width * 0.0004))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(evidence, 8)
    return any(int(stats[label, cv2.CC_STAT_AREA]) >= minimum for label in range(1, count))


def _center_patch_lab(frame: np.ndarray) -> np.ndarray:
    """Measure the automatic capture-center seed in OpenCV Lab coordinates."""

    height, width = frame.shape[:2]
    center_x = int(round(width * 0.50))
    center_y = int(round(height * 0.44))
    radius = max(5, int(round(min(height, width) * 0.018)))
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    patch = lab[
        max(0, center_y - radius):min(height, center_y + radius + 1),
        max(0, center_x - radius):min(width, center_x + radius + 1),
    ]
    return np.median(patch.reshape(-1, 3), axis=0).astype(np.float32)


def _has_soft_pink_evidence(frame: np.ndarray) -> bool:
    """Whether the automatic center seed looks like the intended pink plush."""

    lightness, green_red, blue_yellow = _center_patch_lab(frame)
    return bool(
        80 <= lightness <= 190
        and 136 <= green_red <= 175
        and 130 <= blue_yellow <= 154
        and green_red - blue_yellow >= 1
    )


def _soft_color_anchor(
    frame: np.ndarray,
    _coarse: np.ndarray,
    diagnostics: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Recover a clip-consistent pink soft object without convex hulling.

    The color component is eroded enough to disconnect holder limbs, selected
    from the automatic center, then dilated only back through the original
    color evidence.  This bounded opening preserves ears and
    silhouette concavities while preventing a connected arm from growing to the
    frame border.
    """

    height, width = frame.shape[:2]
    image_area = float(height * width)
    target_x, target_y = width * 0.50, height * 0.44

    def centered_component(binary: np.ndarray) -> np.ndarray | None:
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
        choices: list[tuple[float, int, int]] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(80, int(image_area * 0.00008)):
                continue
            center_x, center_y = centroids[label]
            distance_to_target = (
                ((center_x - target_x) / (width * 0.5)) ** 2
                + ((center_y - target_y) / (height * 0.5)) ** 2
            )
            choices.append((float(distance_to_target), -area, label))
        if not choices or min(choices)[0] > 0.18:
            return None
        return (labels == min(choices)[2]).astype(np.uint8)

    def validated(
        mask: np.ndarray | None,
        *,
        minimum_solidity: float = 0.50,
        result_diagnostics: dict[str, Any] | None = None,
    ) -> np.ndarray | None:
        if mask is None or not np.any(mask):
            return None
        refined = _fill_holes(_largest_component(mask))
        coverage = float(refined.mean())
        if coverage < 0.025 or coverage > 0.58:
            return None
        ys, xs = np.nonzero(refined)
        if len(xs) == 0:
            return None
        margin = min(
            int(xs.min()), int(ys.min()),
            width - 1 - int(xs.max()), height - 1 - int(ys.max()),
        )
        if margin < int(round(min(height, width) * 0.012)):
            return None
        center_distance = np.hypot(
            (float(xs.mean()) - target_x) / (width * 0.50),
            (float(ys.mean()) - target_y) / (height * 0.50),
        )
        if center_distance > 0.30:
            return None
        contours, _hierarchy = cv2.findContours(
            refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = float(cv2.contourArea(contour)) / max(hull_area, 1.0)
        if result_diagnostics is not None:
            result_diagnostics["finalSolidity"] = round(solidity, 6)
            result_diagnostics["finalCoverage"] = round(coverage, 6)
        return refined if solidity >= minimum_solidity else None

    seed_lab = _center_patch_lab(frame)
    if not (
        70 <= seed_lab[0] <= 200
        and 134 <= seed_lab[1] <= 180
        and 128 <= seed_lab[2] <= 158
        and seed_lab[1] - seed_lab[2] >= -1
    ):
        return None
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    distance = np.linalg.norm(lab - seed_lab, axis=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    # Tight evidence is used for appendages, but never returned on its own.  A
    # plush ear can be disconnected from the center component by a small color
    # or illumination change; accepting the center component early silently
    # amputates it.  The broad, strongly opened component below establishes a
    # holder-resistant torso before any tight component is allowed to join.
    tight = (distance <= 18.0).astype(np.uint8)
    tight = cv2.morphologyEx(tight, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    tight = cv2.morphologyEx(tight, cv2.MORPH_CLOSE, close_kernel, iterations=2)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(tight, 8)
    minimum_margin = int(round(min(height, width) * 0.012))

    def with_safe_appendages(
        torso_input: np.ndarray | None,
        torso_source: str,
    ) -> np.ndarray | None:
        if torso_input is None:
            return None
        torso = _largest_component(torso_input)
        torso_area = int(torso.sum())
        if torso_area == 0:
            return None
        torso_yx = np.argwhere(torso > 0)
        torso_points = torso_yx[:, ::-1].astype(np.float32)
        torso_rect = cv2.minAreaRect(torso_points)
        torso_major = max(float(torso_rect[1][0]), float(torso_rect[1][1]), 1.0)
        torso_centroid = torso_points.mean(axis=0)
        torso_distance = cv2.distanceTransform(
            (torso == 0).astype(np.uint8), cv2.DIST_L2, 5
        )

        refined = torso.copy()
        appended = 0
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(40, int(image_area * 0.00004)):
                continue
            component = (labels == label).astype(np.uint8)
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            component_width = int(stats[label, cv2.CC_STAT_WIDTH])
            component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
            right = left + component_width - 1
            bottom = top + component_height - 1
            margin = min(left, top, width - 1 - right, height - 1 - bottom)
            overlap_pixels = int(np.count_nonzero((component > 0) & (torso > 0)))
            overlap_fraction = overlap_pixels / max(area, 1)
            trusted_overlap = (
                margin >= minimum_margin
                and overlap_fraction >= 0.10
                and area <= 1.50 * torso_area
            )
            if trusted_overlap:
                refined |= component
                continue

            # A one-pixel contact is not enough to bypass the satellite gates:
            # warm arms can graze the opened torso.  Every untrusted component
            # must independently look like a nearby elongated appendage.
            area_ratio = area / max(torso_area, 1)
            if not 0.025 <= area_ratio <= 0.35 or margin < minimum_margin:
                continue
            contours, _hierarchy = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = max(contours, key=cv2.contourArea)
            contour_area = float(cv2.contourArea(contour))
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = contour_area / max(hull_area, 1.0)
            extent = contour_area / max(component_width * component_height, 1)
            component_yx = np.argwhere(component > 0)
            component_rect = cv2.minAreaRect(
                component_yx[:, ::-1].astype(np.float32)
            )
            long_side, short_side = sorted(component_rect[1], reverse=True)
            elongation = float(long_side) / max(float(short_side), 1.0)
            separation = float(torso_distance[component > 0].min())
            centroid_distance = float(
                np.linalg.norm(np.asarray(centroids[label]) - torso_centroid)
            )
            if (
                separation > 0.08 * torso_major
                or centroid_distance > 0.80 * torso_major
                or solidity < 0.65
                or extent < 0.30
                or elongation < 1.35
            ):
                continue
            # Preserve the accepted appendage through the final largest-
            # component gate with a short, narrowly bounded bridge.
            satellite_yx = np.argwhere(component > 0)
            satellite_distances = torso_distance[component > 0]
            satellite_point = satellite_yx[int(np.argmin(satellite_distances))]
            squared = np.sum((torso_yx - satellite_point) ** 2, axis=1)
            torso_point = torso_yx[int(np.argmin(squared))]
            bridge_width = max(3, int(round(min(height, width) * 0.004)))
            cv2.line(
                refined,
                (int(torso_point[1]), int(torso_point[0])),
                (int(satellite_point[1]), int(satellite_point[0])),
                1,
                bridge_width,
                cv2.LINE_8,
            )
            appended += 1
            refined |= component

        refined = cv2.morphologyEx(
            refined, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
        )
        growth = float(refined.sum() / max(torso_area, 1))
        attempt = {
            "torsoPixels": torso_area,
            "torsoSource": torso_source,
            "safeAppendageCount": appended,
            "appendageGrowthRatio": round(growth, 6),
        }
        if growth > 1.55:
            return None
        result = validated(
            refined,
            minimum_solidity=0.55,
            result_diagnostics=attempt,
        )
        if result is not None and diagnostics is not None:
            diagnostics.update(attempt)
        return result

    # The broad component is opened before regrowth so holder limbs disconnect;
    # the strict components then restore only well-supported ears/appendages.
    candidate = (distance <= 35.0).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    radius = max(10, int(round(min(height, width) * 0.042)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    core = centered_component(cv2.erode(candidate, kernel))
    if core is None:
        return None
    broad_torso = cv2.dilate(core, kernel) & candidate
    broad_torso = cv2.morphologyEx(
        broad_torso, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    )
    return with_safe_appendages(broad_torso, "opened_broad_component")


def _strict_skin_mask(frame: np.ndarray) -> np.ndarray:
    """Return high-confidence skin pixels, intentionally favoring precision.

    This is used only to find an object seed, never as the final cutout.  The
    intersection of YCrCb and HSV/red-dominance tests avoids labelling most tan
    paper, wood and neutral product finishes as skin while still removing the
    holder's hands in the intended indoor phone captures.
    """

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(frame)
    luminance, cr, cb = cv2.split(ycrcb)
    hue, saturation, value = cv2.split(hsv)
    warm_hue = (hue <= 24) | (hue >= 174)
    red_dominant = (
        (red.astype(np.int16) >= green.astype(np.int16) + 7)
        & (red.astype(np.int16) >= blue.astype(np.int16) + 16)
    )
    return (
        (luminance >= 38)
        & (cr >= 132)
        & (cr <= 181)
        & (cb >= 74)
        & (cb <= 137)
        & warm_hue
        & (saturation >= 28)
        & (value >= 45)
        & red_dominant
    ).astype(np.uint8)


def _component_score(
    component: np.ndarray,
    *,
    image_shape: tuple[int, int],
) -> float:
    """Rank compact, centered appearance components over hair and clothing."""

    height, width = image_shape
    points_yx = np.argwhere(component > 0)
    if len(points_yx) == 0:
        return 0.0
    points_xy = points_yx[:, ::-1].astype(np.float32)
    center_x, center_y = points_xy.mean(axis=0)
    distance = np.hypot(
        (center_x - width * 0.5) / (width * 0.5),
        (center_y - height * 0.47) / (height * 0.5),
    )
    rect = cv2.minAreaRect(points_xy)
    rect_area = max(float(rect[1][0] * rect[1][1]), 1.0)
    rectangularity = min(1.0, float(len(points_xy)) / rect_area)
    # Sublinear area prevents a large hair or shirt region from overwhelming a
    # smaller edge-on object.  Rigid-product surfaces tend to remain compact and
    # rectilinear even as their apparent area changes during the orbit.
    return float(len(points_xy) ** 0.38) * (0.35 + rectangularity) * np.exp(-1.7 * distance)


def _complete_centered_rigid_hull(
    refined: np.ndarray,
    coarse: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Complete a clean half-face when its opposite half is skin-colored.

    A book cover can contain tan pixels indistinguishable from the holder's
    hands.  When the accepted rigid region ends close to the automatic capture
    center, mirror its oriented support about that center and bound the result by
    U2Net's matte.  Expansion is tightly capped and never runs for a remote or
    already-nearly-symmetric component.
    """

    height, width = refined.shape
    ys, xs = np.nonzero(refined)
    if len(xs) == 0:
        return refined, False
    points = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(points)
    theta = np.deg2rad(float(rect[2]))
    axis_u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    axis_v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    projection_u = points @ axis_u
    projection_v = points @ axis_v
    target = np.array([width * 0.5, height * 0.47], dtype=np.float32)
    target_u = float(target @ axis_u)
    target_v = float(target @ axis_v)
    bounds: list[tuple[float, float]] = []
    expansion = 1.0
    for projection, target_projection in (
        (projection_u, target_u),
        (projection_v, target_v),
    ):
        low = float(projection.min())
        high = float(projection.max())
        span = max(high - low, 1.0)
        if target_projection < low - span * 0.18 or target_projection > high + span * 0.18:
            return refined, False
        half_span = max(abs(target_projection - low), abs(high - target_projection))
        symmetric_low = target_projection - half_span
        symmetric_high = target_projection + half_span
        expansion *= (symmetric_high - symmetric_low) / span
        bounds.append((symmetric_low, symmetric_high))
    if expansion < 1.35 or expansion > 3.10:
        return refined, False

    (low_u, high_u), (low_v, high_v) = bounds
    corners = np.array([
        axis_u * low_u + axis_v * low_v,
        axis_u * high_u + axis_v * low_v,
        axis_u * high_u + axis_v * high_v,
        axis_u * low_u + axis_v * high_v,
    ])
    completed_hull = np.zeros_like(refined, dtype=np.uint8)
    cv2.fillConvexPoly(completed_hull, np.rint(corners).astype(np.int32), 1)
    completed = completed_hull & (coarse > 0).astype(np.uint8)
    completed = cv2.morphologyEx(
        completed,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    completed = _fill_holes(_largest_component(completed))
    completed_y, completed_x = np.nonzero(completed)
    if len(completed_x) == 0:
        return refined, False
    completed_points = np.column_stack((completed_x, completed_y)).astype(np.float32)
    completed_rect = cv2.minAreaRect(completed_points)
    completed_rect_area = max(float(completed_rect[1][0] * completed_rect[1][1]), 1.0)
    completed_contours, _hierarchy = cv2.findContours(
        completed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    completed_contour = max(completed_contours, key=cv2.contourArea)
    completed_perimeter = cv2.arcLength(completed_contour, True)
    completed_vertices = cv2.approxPolyDP(
        completed_contour, 0.025 * completed_perimeter, True
    )
    if float(completed.sum()) / completed_rect_area < 0.93 or len(completed_vertices) > 5:
        return refined, False
    return completed, bool(completed.sum() > refined.sum() * 1.18)


def _appearance_anchor(frame: np.ndarray, coarse: np.ndarray) -> np.ndarray | None:
    """Recover a centered rigid object from non-skin appearance evidence.

    This path deliberately fails closed: it requires a sizeable, compact seed
    with a centered convex support and still intersects the result with the
    learned foreground matte.  It is meant for colored/dark phones, books and
    similar held products, not arbitrary people or low-contrast skin-colored
    objects.
    """

    height, width = coarse.shape
    image_area = float(height * width)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin = _strict_skin_mask(frame)
    skin_radius = max(1, int(round(min(height, width) * 0.0025)))
    skin = cv2.dilate(
        skin,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (skin_radius * 2 + 1, skin_radius * 2 + 1)
        ),
    )

    blue, green, red = cv2.split(frame)
    color_range = (
        np.maximum(np.maximum(blue, green), red).astype(np.int16)
        - np.minimum(np.minimum(blue, green), red).astype(np.int16)
    )
    # Saturated or dark pixels carry useful product appearance.  Bright neutral
    # room surfaces and the holder's white shirt are excluded even when U2Net's
    # coarse matte connects them to the subject.
    discriminative = (
        (hsv[..., 1] >= 32)
        | (hsv[..., 2] <= 112)
        | (color_range >= 24)
    )
    central = np.zeros_like(coarse, dtype=np.uint8)
    central[
        int(0.07 * height):int(0.93 * height),
        int(0.07 * width):int(0.93 * width),
    ] = 1
    seed = (
        (coarse > 0)
        & (skin == 0)
        & discriminative
        & (central > 0)
    ).astype(np.uint8)
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(seed, 8)
    minimum_area = max(90, int(image_area * 0.00008))
    components: list[tuple[float, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        center_x = left + component_width * 0.5
        center_y = top + component_height * 0.5
        normalized_distance = np.hypot(
            (center_x - width * 0.5) / (width * 0.5),
            (center_y - height * 0.47) / (height * 0.5),
        )
        if normalized_distance > 0.72:
            continue
        component = (labels == label).astype(np.uint8)
        components.append((
            _component_score(component, image_shape=(height, width)),
            label,
        ))
    if not components:
        return None

    components.sort(reverse=True)
    best_score, best_label = components[0]
    if best_score <= 0:
        return None
    accepted = (labels == best_label).astype(np.uint8)
    best_yx = np.argwhere(accepted > 0)
    best_left = int(best_yx[:, 1].min())
    best_right = int(best_yx[:, 1].max())
    best_top = int(best_yx[:, 0].min())
    best_bottom = int(best_yx[:, 0].max())
    join_margin = max(5, int(round(min(height, width) * 0.018)))
    # Include split appearance islands (white phone markings, book title blocks,
    # camera lenses) only when they overlap the winning object's support.  This
    # improves the hull without allowing remote hair/clothing regions to join.
    for _score, label in components[1:]:
        left = int(stats[label, cv2.CC_STAT_LEFT])
        top = int(stats[label, cv2.CC_STAT_TOP])
        right = left + int(stats[label, cv2.CC_STAT_WIDTH]) - 1
        bottom = top + int(stats[label, cv2.CC_STAT_HEIGHT]) - 1
        overlaps_x = right >= best_left - join_margin and left <= best_right + join_margin
        overlaps_y = bottom >= best_top - join_margin and top <= best_bottom + join_margin
        if overlaps_x and overlaps_y:
            accepted[labels == label] = 1

    points_yx = np.argwhere(accepted > 0)
    if len(points_yx) < max(300, int(image_area * 0.00035)):
        return None
    hull = cv2.convexHull(points_yx[:, ::-1].astype(np.int32))
    hull_mask = np.zeros_like(coarse, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    radius = max(2, int(round(min(height, width) * 0.007)))
    hull_mask = cv2.dilate(
        hull_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)),
    )
    refined = hull_mask & (coarse > 0).astype(np.uint8)
    refined = cv2.morphologyEx(
        refined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    refined = _fill_holes(_largest_component(refined))

    coverage = float(refined.mean())
    if coverage < 0.004 or coverage > 0.46:
        return None
    border = np.concatenate((refined[0], refined[-1], refined[:, 0], refined[:, -1]))
    if float(border.mean()) > 0.04:
        return None
    ys, xs = np.nonzero(refined)
    if len(xs) == 0:
        return None
    center_distance = np.hypot(
        (float(xs.mean()) - width * 0.5) / (width * 0.5),
        (float(ys.mean()) - height * 0.47) / (height * 0.5),
    )
    if center_distance > 0.68:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(points)
    long_side, short_side = sorted(rect[1], reverse=True)
    rect_area = max(float(long_side * short_side), 1.0)
    rectangularity = float(len(points)) / rect_area
    contours, _hierarchy = cv2.findContours(
        refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    vertices = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
    skin_fraction = float(((_strict_skin_mask(frame) > 0) & (refined > 0)).sum()) / float(
        max(int(refined.sum()), 1)
    )
    # These gates intentionally discard ambiguous side views.  A smaller set of
    # clean rigid-product views is safer than silently accepting a face or hair
    # blob when the object is edge-on.
    if long_side < min(height, width) * 0.20:
        return None
    aspect_ratio = float(long_side) / max(float(short_side), 1.0)
    if aspect_ratio < 1.12:
        return None
    if rectangularity < 0.78 or len(vertices) > 6:
        return None
    if skin_fraction > 0.27:
        return None
    if float(ys.mean()) < height * 0.23 or float(ys.mean()) > height * 0.82:
        return None
    completed, _was_completed = _complete_centered_rigid_hull(refined, coarse)
    completed_coverage = float(completed.mean())
    if completed_coverage > 0.68:
        return refined
    completed_border = np.concatenate(
        (completed[0], completed[-1], completed[:, 0], completed[:, -1])
    )
    if float(completed_border.mean()) > 0.04:
        return refined
    return completed


def _overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    dimmed = (frame.astype(np.float32) * 0.25).astype(np.uint8)
    result = np.where(mask[..., None] > 0, frame, dimmed)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (40, 230, 255), 3, cv2.LINE_AA)
    return result


def load_cached_u2netp(model_name: str = "u2netp") -> tuple[Any, Callable[..., Any], dict[str, Any]]:
    """Load the pre-cached local matte model without permitting downloads.

    Returns ``(session, remove, provenance)`` so higher-level pipelines can run
    the coarse matte once and reuse it for multiple mask candidates.
    """

    model_file, model_sha256 = _require_cached_u2netp_model(model_name)
    new_session, remove, session_class, rembg_version = _load_rembg_backend()
    try:
        session = _new_cached_u2netp_session(new_session, session_class, model_file)
    except MaskingError:
        raise
    except Exception as exc:
        raise MaskingError(
            f"could not initialize cached local U2Net session from {model_file}"
        ) from exc
    provenance = {
        "backend": "rembg/u2netp coarse foreground matte",
        "runtime": {"package": "rembg", "version": rembg_version},
        "model": {
            "name": model_name,
            "path": str(model_file),
            "bytes": model_file.stat().st_size,
            "sha256": model_sha256,
            "provenance": {
                "kind": "preexisting_local_cache",
                "cachePathSource": _u2netp_cache_path_source(),
                "networkAttempted": False,
            },
        },
    }
    return session, remove, provenance


def refine_held_object_sequence(
    frames: Sequence[np.ndarray],
    coarse_masks: Sequence[np.ndarray],
) -> tuple[list[np.ndarray | None], dict[str, Any]]:
    """Produce an object-centric alternative to each coarse foreground matte.

    The clip-level color routes are candidates, not semantic classification.
    A caller should compare their coverage and continuity with its generic mask
    path and may retain only the frames that pass these conservative gates.
    """

    if len(frames) != len(coarse_masks):
        raise ValueError("frames and coarse_masks must have equal length")
    if not frames:
        return [], {
            "greenEvidenceFrames": 0,
            "softPinkEvidenceFrames": 0,
            "persistentGreen": False,
            "persistentSoftPink": False,
            "acceptedFrames": 0,
            "routes": {},
        }

    green_evidence = sum(_has_green_evidence(frame) for frame in frames)
    pink_evidence = sum(_has_soft_pink_evidence(frame) for frame in frames)
    persistent_green = green_evidence >= max(3, int(np.ceil(len(frames) * 0.45)))
    persistent_pink = (
        not persistent_green
        and pink_evidence >= max(4, int(np.ceil(len(frames) * 0.50)))
    )

    refined: list[np.ndarray | None] = []
    routes: dict[str, int] = {}
    for frame, coarse in zip(frames, coarse_masks):
        coarse_binary = (np.asarray(coarse) > 0).astype(np.uint8)
        candidate = _green_anchor(frame, coarse_binary) if persistent_green else None
        route = "persistent_green_chroma"
        if candidate is None and persistent_pink:
            candidate = _soft_color_anchor(frame, coarse_binary)
            route = "persistent_soft_pink_component"
        if candidate is None and not persistent_pink:
            candidate = _appearance_anchor(frame, coarse_binary)
            route = "central_non_skin_appearance"
        if candidate is None:
            refined.append(None)
            routes["rejected"] = routes.get("rejected", 0) + 1
        else:
            refined.append((candidate > 0).astype(np.uint8))
            routes[route] = routes.get(route, 0) + 1

    return refined, {
        "greenEvidenceFrames": int(green_evidence),
        "softPinkEvidenceFrames": int(pink_evidence),
        "persistentGreen": bool(persistent_green),
        "persistentSoftPink": bool(persistent_pink),
        "acceptedFrames": int(sum(item is not None for item in refined)),
        "requestedFrames": len(frames),
        "routes": dict(sorted(routes.items())),
    }


def generate_color_anchored_masks(
    frames_directory: Path,
    output_directory: Path,
    review_directory: Path,
    *,
    model_name: str = "u2netp",
) -> dict[str, Any]:
    """Generate prompt-free held-object masks and a Vision-compatible report."""

    paths = _frame_paths(frames_directory)
    if not paths:
        raise MaskingError(f"no input frames found in {frames_directory}")
    model_file, model_sha256 = _require_cached_u2netp_model(model_name)
    new_session, remove, session_class, rembg_version = _load_rembg_backend()
    output_directory.mkdir(parents=True, exist_ok=True)
    review_directory.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cv2.setNumThreads(1)
    try:
        session = _new_cached_u2netp_session(new_session, session_class, model_file)
    except MaskingError:
        raise
    except Exception as exc:
        raise MaskingError(
            f"could not initialize cached local U2Net session from {model_file}"
        ) from exc

    # A true green product remains green through most of the orbit.  Requiring
    # clip-level persistence prevents a single green phone wallpaper or room
    # reflection from hijacking one frame while retaining the established green
    # tin path unchanged.
    green_evidence_frames = 0
    soft_pink_evidence_frames = 0
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is not None:
            if _has_green_evidence(frame):
                green_evidence_frames += 1
            if _has_soft_pink_evidence(frame):
                soft_pink_evidence_frames += 1
    persistent_green = green_evidence_frames >= max(3, int(np.ceil(len(paths) * 0.45)))
    persistent_soft_pink = (
        not persistent_green
        and soft_pink_evidence_frames >= max(4, int(np.ceil(len(paths) * 0.50)))
    )

    frames: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            failures.append({"source": str(path), "reason": "could not decode frame"})
            continue
        try:
            with Image.open(path) as source_image:
                alpha = remove(
                    source_image.convert("RGB"),
                    session=session,
                    only_mask=True,
                    post_process_mask=False,
                )
        except Exception as exc:
            raise MaskingError(f"local U2Net inference failed for {path}") from exc
        coarse = np.asarray(alpha, dtype=np.uint8)
        if coarse.shape != frame.shape[:2]:
            coarse = cv2.resize(
                coarse, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        coarse_binary = (coarse >= 128).astype(np.uint8)
        refined = _green_anchor(frame, coarse_binary) if persistent_green else None
        anchor = "persistent_green_chroma"
        anchor_diagnostics: dict[str, Any] = {}
        if refined is None and persistent_soft_pink:
            refined = _soft_color_anchor(
                frame, coarse_binary, diagnostics=anchor_diagnostics
            )
            anchor = "persistent_soft_pink_component"
        if refined is None and not persistent_soft_pink:
            refined = _appearance_anchor(frame, coarse_binary)
            anchor = "central_non_skin_appearance"
        if refined is None:
            failures.append({
                "source": str(path),
                "reason": "no stable central held-object anchor passed coverage gates",
            })
            continue
        mask_path = output_directory / f"{path.stem}.png"
        cv2.imwrite(str(mask_path), refined * 255)
        review_path = review_directory / f"{path.stem}_overlay.jpg"
        cv2.imwrite(str(review_path), _overlay(frame, refined), [cv2.IMWRITE_JPEG_QUALITY, 92])
        coarse_count = max(int(coarse_binary.sum()), 1)
        frame_record: dict[str, Any] = {
            "source": str(path),
            "mask": str(mask_path),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
            "foregroundFraction": round(float(refined.mean()), 6),
            "removedFraction": round(
                max(0.0, 1.0 - float((refined & coarse_binary).sum()) / coarse_count), 6
            ),
            "anchor": anchor,
            "debugOverlay": str(review_path),
        }
        if anchor_diagnostics:
            frame_record["anchorDiagnostics"] = anchor_diagnostics
        frames.append(frame_record)

    model = {
        "name": model_name,
        "path": str(model_file),
        "bytes": model_file.stat().st_size,
        "sha256": model_sha256,
        "provenance": {
            "kind": "preexisting_local_cache",
            "cachePathSource": _u2netp_cache_path_source(),
            "networkAttempted": False,
        },
    }
    report = {
        "backend": (
            "local U2Net coarse matte + clip-persistent green rigid, non-skin rigid, "
            "or soft pink silhouette anchor"
        ),
        "runtime": {"package": "rembg", "version": rembg_version},
        "requestedFrames": len(paths),
        "clipAnalysis": {
            "greenEvidenceFrames": green_evidence_frames,
            "persistentGreen": persistent_green,
            "softPinkEvidenceFrames": soft_pink_evidence_frames,
            "persistentSoftPink": persistent_soft_pink,
        },
        "model": model,
        "frames": frames,
        "failures": failures,
    }
    (output_directory / "mask_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report
