"""Automatic rounded-slab fitting from a complete video pipeline run.

Rigid packages such as phones, books, and tins are poorly served by a sparse
point convex hull when the capture contains hands and long low-texture sides.
This module detects face-on views from the automatically generated masks,
clusters the two observed faces by appearance, rectifies their source pixels,
infers depth from profile silhouettes, and delegates deterministic mesh
authoring to :mod:`local3d.parametric_assets`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .parametric_assets import build_asset


class AutoFitError(RuntimeError):
    """Raised when automatic evidence cannot support a rounded-slab fit."""


@dataclass
class FaceCandidate:
    frame: Path
    mask: Path
    quad: np.ndarray
    rectified: np.ndarray
    face_ratio: float
    coverage: float
    sharpness: float
    skin_fraction: float
    score: float
    feature: np.ndarray
    rotate_quarter_turns: int = 0
    rectangularity: float = 1.0
    solidity: float = 1.0
    removed_fraction: float = 0.0


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    total = points.sum(axis=1)
    difference = points[:, 0] - points[:, 1]
    ordered = np.empty((4, 2), np.float32)
    ordered[0] = points[np.argmin(total)]  # top-left
    ordered[2] = points[np.argmax(total)]  # bottom-right
    ordered[1] = points[np.argmax(difference)]  # top-right
    ordered[3] = points[np.argmin(difference)]  # bottom-left
    return ordered


def _rectify(frame: np.ndarray, quad: np.ndarray, size: int = 320) -> np.ndarray:
    target = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), target)
    return cv2.warpPerspective(
        frame, matrix, (size, size), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _strict_skin_fraction(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin = (
        (hsv[..., 0] <= 18)
        & (hsv[..., 1] >= 45)
        & (hsv[..., 1] <= 205)
        & (hsv[..., 2] >= 55)
        & (ycrcb[..., 1] >= 140)
        & (ycrcb[..., 1] <= 178)
        & (ycrcb[..., 2] >= 82)
        & (ycrcb[..., 2] <= 128)
    )
    return float(skin.mean())


def _appearance_feature(rectified: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rectified, cv2.COLOR_BGR2LAB)
    spatial = cv2.resize(lab, (16, 16), interpolation=cv2.INTER_AREA).astype(np.float32)
    # Normalize illumination per channel, retaining layout/text while reducing
    # exposure differences between the beginning and end of the orbit.
    spatial -= spatial.mean(axis=(0, 1), keepdims=True)
    spatial /= spatial.std(axis=(0, 1), keepdims=True) + 8.0
    histograms = []
    hsv = cv2.cvtColor(rectified, cv2.COLOR_BGR2HSV)
    for channel, bins, maximum in ((0, 18, 180), (1, 8, 256), (2, 8, 256)):
        histogram = cv2.calcHist([hsv], [channel], None, [bins], [0, maximum]).ravel()
        histogram /= max(float(histogram.sum()), 1.0)
        histograms.append(histogram.astype(np.float32))
    return np.concatenate((spatial.ravel(), *histograms)).astype(np.float32)


def _candidate(
    frame_path: Path,
    mask_path: Path,
    *,
    removed_fraction: float = 0.0,
) -> FaceCandidate | None:
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if frame is None or mask is None or mask.shape != frame.shape[:2]:
        return None
    binary = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    rectangle = cv2.minAreaRect(contour)
    rect_width, rect_height = rectangle[1]
    if min(rect_width, rect_height) < 8:
        return None
    face_ratio = float(min(rect_width, rect_height) / max(rect_width, rect_height))
    rectangularity = contour_area / max(float(rect_width * rect_height), 1e-6)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    solidity = contour_area / max(hull_area, 1e-6)
    coverage = float(binary.mean())
    quad = _order_quad(cv2.boxPoints(rectangle))
    rectified = _rectify(frame, quad)
    top_edge = float(np.linalg.norm(quad[1] - quad[0]))
    side_edge = float(np.linalg.norm(quad[3] - quad[0]))
    # Normalize clearly portrait observations to the same intrinsic landscape
    # orientation before appearance clustering.  Without this, in-plane camera
    # roll can be mistaken for a different physical face.
    rotate_quarter_turns = 1 if side_edge > top_edge * 1.12 else 0
    if rotate_quarter_turns:
        rectified = np.ascontiguousarray(np.rot90(rectified, rotate_quarter_turns))
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    skin = _strict_skin_fraction(rectified)
    score = face_ratio**3 * np.sqrt(max(coverage, 1e-6)) * np.log1p(max(sharpness, 0.0))
    score *= max(0.15, 1.0 - 2.5 * skin)
    return FaceCandidate(
        frame=frame_path,
        mask=mask_path,
        quad=quad,
        rectified=rectified,
        face_ratio=face_ratio,
        coverage=coverage,
        sharpness=sharpness,
        skin_fraction=skin,
        score=float(score),
        feature=_appearance_feature(rectified),
        rotate_quarter_turns=rotate_quarter_turns,
        rectangularity=float(rectangularity),
        solidity=float(solidity),
        removed_fraction=float(np.clip(removed_fraction, 0.0, 1.0)),
    )


def _face_on_band(candidates: list[FaceCandidate]) -> tuple[float, float, float]:
    """Infer the dense intrinsic face-ratio band of a slab capture.

    Square tins have more profile frames than face frames, while phone captures
    can contain a few overly square masks caused by a hand/hair protrusion.  A
    dense high-ratio window avoids both failure modes: it chooses the highest
    0.10-wide cluster supported by at least a quarter of the clip, rather than
    blindly treating the maximum ratio as the physical face.
    """

    if not candidates:
        return 1.0, 1.0, 1.0
    ordered = sorted(candidates, key=lambda candidate: candidate.face_ratio)
    ratios = np.asarray([candidate.face_ratio for candidate in ordered], dtype=np.float64)
    minimum_support = max(4, int(np.ceil(len(ordered) * 0.30)))
    supported: list[list[FaceCandidate]] = []
    for start, candidate in enumerate(ordered):
        window = [
            value for value in ordered[start:]
            if value.face_ratio <= candidate.face_ratio + 0.10
        ]
        if len(window) >= minimum_support:
            supported.append(window)
    if supported:
        chosen = max(
            supported,
            key=lambda values: (
                sum(
                    value.coverage * max(0.0, 1.0 - value.removed_fraction)
                    for value in values
                ),
                float(np.median([value.face_ratio for value in values])),
            ),
        )
        center = float(np.median([candidate.face_ratio for candidate in chosen]))
    else:
        center = float(np.percentile(ratios, 70))
    center = max(0.30, center)
    tolerance = max(0.06, center * 0.15)
    lower = max(0.30, center - tolerance)
    upper = max(center, min(1.0, center + tolerance))
    return lower, upper, center


def _face_on_threshold(candidates: list[FaceCandidate]) -> float:
    """Backward-compatible lower edge of the inferred face-on band."""

    return _face_on_band(candidates)[0]


def _face_evidence_candidates(candidates: list[FaceCandidate]) -> list[FaceCandidate]:
    """Exclude masks whose segmentation discarded too much source evidence."""

    return [candidate for candidate in candidates if candidate.removed_fraction <= 0.35]


def _infer_depth_ratio(
    candidates: list[FaceCandidate],
    face_center: float,
    face_aspect: float = 1.0,
) -> tuple[float, bool, float]:
    """Estimate slab depth only when the clip contains convincing profiles.

    The smallest silhouettes in an incomplete phone orbit can still be almost
    face-on.  Treating those as edge profiles makes the inferred body much too
    thick.  We therefore require a substantial separation from the intrinsic
    face band; otherwise a conservative, explicitly reported slab prior is
    safer than claiming unsupported measured depth.
    """

    profile_ratios = sorted(candidate.face_ratio for candidate in candidates)
    profile_count = max(2, len(profile_ratios) // 5)
    apparent_profile_ratio = float(np.median(profile_ratios[:profile_count]))
    separated_profiles = [ratio for ratio in profile_ratios if ratio <= face_center * 0.60]
    minimum_profile_support = max(3, int(np.ceil(len(profile_ratios) * 0.15)))
    profile_views_observed = len(separated_profiles) >= minimum_profile_support
    if profile_views_observed:
        apparent_profile_ratio = float(np.median(separated_profiles))
        # A profile ratio is measured against the long silhouette axis, while
        # mesh depth is expressed against the short face axis.  Normalize that
        # basis before the fixed perspective correction.
        depth_ratio = float(
            np.clip(apparent_profile_ratio / max(face_aspect, 0.30) * 0.68, 0.08, 0.30)
        )
    else:
        depth_ratio = 0.16
    return apparent_profile_ratio, profile_views_observed, depth_ratio


def _cluster_faces(candidates: list[FaceCandidate]) -> tuple[FaceCandidate, FaceCandidate, list[int]]:
    threshold, upper_bound, _center = _face_on_band(candidates)
    face_candidates = [
        candidate for candidate in candidates
        if threshold <= candidate.face_ratio <= upper_bound
    ]
    if len(face_candidates) < 4:
        raise AutoFitError(
            f"only {len(face_candidates)} face-on frames were found; need at least four"
        )
    features = np.stack([candidate.feature for candidate in face_candidates]).astype(np.float32)
    # Balance the high-dimensional spatial descriptor before deterministic
    # two-face clustering.
    features -= features.mean(axis=0, keepdims=True)
    deviation = features.std(axis=0, keepdims=True)
    features /= np.where(deviation > 1e-5, deviation, 1.0)
    cv2.setRNGSeed(0)
    _compactness, labels, centers = cv2.kmeans(
        features,
        2,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        10,
        cv2.KMEANS_PP_CENTERS,
    )
    assignments = labels.ravel().astype(int).tolist()
    group_indices = [
        [index for index, label in enumerate(assignments) if label == group]
        for group in (0, 1)
    ]
    if min(map(len, group_indices)) < 2:
        raise AutoFitError(
            "appearance clustering did not find at least two supporting frames for each face"
        )
    representatives: list[FaceCandidate] = []
    normalizer = max(float(np.sqrt(features.shape[1])), 1.0)
    for group, indices in enumerate(group_indices):
        group_scores = np.asarray([face_candidates[index].score for index in indices])
        score_span = float(np.ptp(group_scores))
        quality = (
            (group_scores - float(group_scores.min())) / score_span
            if score_span > 1e-9 else np.ones_like(group_scores)
        )
        distance = np.asarray(
            [np.linalg.norm(features[index] - centers[group]) / normalizer for index in indices]
        )
        selected_local = int(np.argmin(distance - quality * 0.12))
        representatives.append(face_candidates[indices[selected_local]])

    # Preserve chronological face naming.  Geometry is symmetric, so this is
    # only a stable convention; it does not assert semantic front/back labels.
    representatives.sort(key=lambda candidate: candidate.frame.name)
    return representatives[0], representatives[1], assignments


def _body_color(candidates: list[FaceCandidate]) -> list[int]:
    colors: list[np.ndarray] = []
    for candidate in candidates:
        image = candidate.rectified
        height, width = image.shape[:2]
        y, x = np.ogrid[:height, :width]
        outer = (
            (x < width * 0.20)
            | (x >= width * 0.80)
            | (y < height * 0.20)
            | (y >= height * 0.80)
        )
        inner = (
            (x >= width * 0.035)
            & (x < width * 0.965)
            & (y >= height * 0.035)
            & (y < height * 0.965)
        )
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        skin = (
            (hsv[..., 0] <= 18)
            & (hsv[..., 1] >= 45)
            & (hsv[..., 1] <= 205)
            & (hsv[..., 2] >= 55)
            & (ycrcb[..., 1] >= 140)
            & (ycrcb[..., 1] <= 178)
            & (ycrcb[..., 2] >= 82)
            & (ycrcb[..., 2] <= 128)
        )
        usable = outer & inner & ~skin & (hsv[..., 2] >= 18) & (hsv[..., 2] <= 245)
        if int(usable.sum()) >= max(12, int(round(height * width * 0.08))):
            colors.append(image[usable])
    if not colors:
        return [112, 116, 120]
    # Border pixels carry the persistent body/case/cover color even when the
    # two broad faces contain unrelated labels or a dark phone screen.  Favor
    # the lit population slightly so profile shadows do not muddy sidewalls.
    bgr = np.percentile(np.concatenate(colors, axis=0), 62, axis=0)
    return [int(round(value)) for value in bgr[::-1]]


def _write_clean_face(
    candidate: FaceCandidate,
    destination: Path,
    body_rgb: list[int],
    *,
    size: int = 1200,
) -> None:
    """Rectify a face and replace only non-object/holder border pixels."""

    frame = cv2.imread(str(candidate.frame), cv2.IMREAD_COLOR)
    source_mask = cv2.imread(str(candidate.mask), cv2.IMREAD_GRAYSCALE)
    if frame is None or source_mask is None:
        raise AutoFitError(f"could not reload selected face {candidate.frame}")
    target = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(candidate.quad.astype(np.float32), target)
    image = cv2.warpPerspective(
        frame, matrix, (size, size), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    mask = cv2.warpPerspective(
        source_mask, matrix, (size, size), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 127
    if candidate.rotate_quarter_turns:
        image = np.ascontiguousarray(np.rot90(image, candidate.rotate_quarter_turns))
        mask = np.ascontiguousarray(np.rot90(mask, candidate.rotate_quarter_turns))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)) > 0

    # Fingers appear only around the outside of a selected planar face.  Keep
    # internal label/window pixels untouched and reject strict skin pixels only
    # inside a narrow border band.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    skin = (
        (hsv[..., 0] <= 18)
        & (hsv[..., 1] >= 45)
        & (hsv[..., 1] <= 205)
        & (hsv[..., 2] >= 55)
        & (ycrcb[..., 1] >= 140)
        & (ycrcb[..., 1] <= 178)
        & (ycrcb[..., 2] >= 82)
        & (ycrcb[..., 2] <= 128)
    )
    border = np.zeros((size, size), dtype=bool)
    margin = int(round(size * 0.14))
    border[:margin] = True
    border[-margin:] = True
    border[:, :margin] = True
    border[:, -margin:] = True
    # If the strict skin classifier covers a large part of the rectified face,
    # it is more likely warm artwork/material (for example a tan book cover)
    # than a sparse finger fringe.  In that case trust the already gated object
    # mask instead of punching false holes in the source texture.
    proposed_skin = skin & border & mask
    proposed_fraction = float(proposed_skin.sum() / max(int(mask.sum()), 1))
    remove_skin = proposed_skin if proposed_fraction <= 0.02 else np.zeros_like(skin)
    keep = mask & ~remove_skin
    fill_bgr = np.asarray(body_rgb[::-1], dtype=np.uint8)
    cleaned = np.where(keep[..., None], image, fill_bgr)
    if not cv2.imwrite(str(destination), cleaned, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
        raise AutoFitError(f"could not write cleaned face texture {destination}")


def _normalized_dimension_report(dimensions: dict[str, Any]) -> dict[str, Any]:
    """Describe normalized builder dimensions without implying recovered scale."""

    width = float(dimensions["width"])
    return {
        "normalized_builder_dimensions_mm": dimensions,
        "physical_scale_inferred": False,
        "dimension_normalization": f"face width = {width:g} builder mm",
        "absolute_scale": (
            "unknown; builder dimensions are normalized and not physical measurements"
        ),
    }


def fit_rounded_slab(
    frames_directory: Path,
    masks_directory: Path,
    output_directory: Path,
    *,
    mask_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    removed_by_stem: dict[str, float] = {}
    if isinstance(mask_report, dict):
        for frame_record in mask_report.get("frames", []):
            if not isinstance(frame_record, dict):
                continue
            source = frame_record.get("source")
            removed = frame_record.get("removedFraction")
            if isinstance(source, str) and isinstance(removed, (int, float)):
                removed_by_stem[Path(source).stem] = float(removed)

    candidates: list[FaceCandidate] = []
    for frame in sorted(frames_directory.glob("frame_*.jpg")):
        mask = masks_directory / f"{frame.stem}.png"
        if mask.is_file():
            candidate = _candidate(
                frame,
                mask,
                removed_fraction=removed_by_stem.get(frame.stem, 0.0),
            )
            if candidate is not None:
                candidates.append(candidate)
    if len(candidates) < 8:
        raise AutoFitError(f"only {len(candidates)} valid mask/frame measurements were available")

    evidence_candidates = _face_evidence_candidates(candidates)
    if len(evidence_candidates) < 4:
        raise AutoFitError(
            "only "
            f"{len(evidence_candidates)} masks retained enough source evidence for face fitting"
        )
    threshold, upper_bound, face_center = _face_on_band(evidence_candidates)
    regular_views = [
        candidate for candidate in evidence_candidates
        if threshold <= candidate.face_ratio <= upper_bound
    ]
    if len(regular_views) < 4:
        raise AutoFitError(
            f"only {len(regular_views)} face-on frames were found; need at least four"
        )
    median_rectangularity = float(np.median([candidate.rectangularity for candidate in regular_views]))
    median_solidity = float(np.median([candidate.solidity for candidate in regular_views]))
    if median_rectangularity < 0.72 or median_solidity < 0.88:
        raise AutoFitError(
            "silhouettes are not regular enough for a rigid rounded slab "
            f"(rectangularity={median_rectangularity:.3f}, solidity={median_solidity:.3f})"
        )

    first_face, second_face, assignments = _cluster_faces(evidence_candidates)
    # Face candidates are close to square for tins but the same calculation
    # supports portrait/landscape slabs.  Use the ordered quad edge ratio from
    # both selected faces and normalize the longer face dimension to 100 mm.
    ratios = []
    for candidate in (first_face, second_face):
        top = np.linalg.norm(candidate.quad[1] - candidate.quad[0])
        side = np.linalg.norm(candidate.quad[3] - candidate.quad[0])
        ratios.append(float(min(top, side) / max(top, side, 1e-6)))
    height_ratio = float(np.clip(np.median(ratios), 0.30, 1.0))
    apparent_profile_ratio, profile_views_observed, depth_ratio = _infer_depth_ratio(
        candidates, face_center, height_ratio
    )
    width_mm = 100.0
    height_mm = width_mm * height_ratio
    short_face = min(width_mm, height_mm)
    depth_mm = short_face * depth_ratio
    body = _body_color(evidence_candidates)

    output_directory.mkdir(parents=True, exist_ok=True)
    first_clean = output_directory / "automatic_front_source.png"
    second_clean = output_directory / "automatic_back_source.png"
    _write_clean_face(first_face, first_clean, body)
    _write_clean_face(second_face, second_clean, body)
    clean_quad = [[0, 0], [1199, 0], [1199, 1199], [0, 1199]]
    config = {
        "schema_version": 1,
        "asset_name": "Automatically fitted rounded slab",
        "kind": "phone",
        "authoring_mode": "automatic",
        "asset_kind": "rounded_slab",
        "output_name": "parametric_model",
        "dimension_basis": "relative dimensions inferred automatically from multi-view silhouettes; absolute scale is ambiguous",
        "notes": [
            "Two face textures were selected by deterministic appearance clustering of face-on video frames.",
            (
                "Profile thickness was inferred from separated edge-on silhouette views."
                if profile_views_observed
                else "No separated edge-on view was observed; thickness uses an automatic generic slab prior."
            ),
            "All quads, dimensions, and intermediate source images were generated by this automatic pipeline.",
        ],
        "front": {
            "image": str(first_clean.resolve()),
            "quad_px": clean_quad,
            "rotate_quarter_turns": 0,
            "texture_mode": "source",
        },
        "back": {
            "image": str(second_clean.resolve()),
            "quad_px": clean_quad,
            "rotate_quarter_turns": 0,
            "texture_mode": "source",
        },
        "dimensions_mm": {
            "width": width_mm,
            "height": height_mm,
            "depth": depth_mm,
            "corner_radius": short_face * 0.095,
            "bevel": min(depth_mm * 0.13, short_face * 0.018),
        },
        "texture_size": 1024,
        "corner_segments": 16,
        "output_rotation_deg": [0, 0, 0],
        "materials": {
            "front": {"color_rgb": body, "metallic": 0.04, "roughness": 0.42},
            "back": {"color_rgb": body, "metallic": 0.04, "roughness": 0.42},
            "body": {"color_rgb": body, "metallic": 0.08, "roughness": 0.38},
            "port": {
                "color_rgb": [max(20, int(body[0] * 0.58)), max(24, int(body[1] * 0.58)), max(12, int(body[2] * 0.58))],
                "metallic": 0.10,
                "roughness": 0.46,
            },
        },
        "phone": {
            "decorations": [],
            "body_seam": {"width_mm": max(0.9, depth_mm * 0.065), "offset_mm": 0.0},
        },
    }
    config_path = output_directory / "automatic_fit_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    manifest = build_asset(config_path, output_directory, allow_usdz=True)

    fit_report = {
        "method": "automatic multiview rounded-slab fit",
        "candidate_frames": len(candidates),
        "retained_evidence_frames": len(evidence_candidates),
        "maximum_face_removed_fraction": 0.35,
        "face_on_frames": sum(
            threshold <= candidate.face_ratio <= upper_bound
            for candidate in evidence_candidates
        ),
        "face_on_threshold": round(threshold, 6),
        "face_on_upper_bound": round(upper_bound, 6),
        "face_on_center": round(face_center, 6),
        "selected_faces": [
            {
                "frame": str(candidate.frame),
                "score": round(candidate.score, 6),
                "face_ratio": round(candidate.face_ratio, 6),
                "sharpness": round(candidate.sharpness, 3),
                "skin_fraction": round(candidate.skin_fraction, 6),
                "rotate_quarter_turns": candidate.rotate_quarter_turns,
                "rectangularity": round(candidate.rectangularity, 6),
                "solidity": round(candidate.solidity, 6),
                "removed_fraction": round(candidate.removed_fraction, 6),
            }
            for candidate in (first_face, second_face)
        ],
        **_normalized_dimension_report(config["dimensions_mm"]),
        "apparent_profile_ratio": round(apparent_profile_ratio, 6),
        "profile_views_observed": profile_views_observed,
        "depth_inference": "profile silhouettes" if profile_views_observed else "generic slab prior",
        "perspective_corrected_depth_ratio": round(depth_ratio, 6),
        "body_color_rgb": body,
        "regularity_gate": {
            "median_rectangularity": round(median_rectangularity, 6),
            "median_solidity": round(median_solidity, 6),
            "passed": True,
        },
        "quality_gate_passed": True,
        "cluster_assignments": assignments,
        "artifacts": {
            "glb": str(output_directory / "parametric_model.glb"),
            "usdz": str(output_directory / "parametric_model.usdz"),
            "qa": str(output_directory / "qa_model_contact.png"),
        },
        "builder_manifest": manifest,
    }
    (output_directory / "automatic_fit_report.json").write_text(
        json.dumps(fit_report, indent=2) + "\n", encoding="utf-8"
    )
    return fit_report
