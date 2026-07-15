"""Local-only video probing, frame extraction, and capture-quality analysis.

The ingest stage deliberately does not try to segment the object.  When no object
mask is available, ``subject_coverage`` is estimated from temporal change in the
central image region and is therefore only a preflight heuristic.  Downstream
segmentation should replace it with actual object-mask coverage.

The module shells out only to the local ``ffprobe`` executable for container
metadata.  Frames and metrics are produced with local OpenCV/numpy; nothing is
uploaded and no network service is contacted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # Keep metadata probing importable even in an ffmpeg-only environment.
    import cv2  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only in minimal installs.
    cv2 = None
    np = None


SCHEMA_VERSION = "1.0"


class IngestError(RuntimeError):
    """Raised when local video ingestion cannot be completed."""


@dataclass(frozen=True)
class IngestConfig:
    """Controls candidate extraction and keyframe selection.

    ``sample_fps`` is an upper bound.  Long clips are sampled more sparsely when
    needed to respect ``max_candidates``.  Analysis is performed at a bounded
    resolution while extracted images retain their decoded source resolution.
    """

    sample_fps: float = 3.0
    max_candidates: int = 240
    keyframe_count: int = 24
    min_keyframe_gap_s: float = 0.30
    analysis_long_side: int = 512
    image_format: str = "jpg"
    jpeg_quality: int = 95
    central_roi_fraction: float = 0.70

    def validate(self) -> None:
        if not math.isfinite(self.sample_fps) or self.sample_fps <= 0:
            raise ValueError("sample_fps must be finite and greater than zero")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be at least 1")
        if self.keyframe_count < 1:
            raise ValueError("keyframe_count must be at least 1")
        if self.min_keyframe_gap_s < 0:
            raise ValueError("min_keyframe_gap_s cannot be negative")
        if self.analysis_long_side < 64:
            raise ValueError("analysis_long_side must be at least 64 pixels")
        if self.image_format.lower() not in {"jpg", "jpeg", "png"}:
            raise ValueError("image_format must be jpg, jpeg, or png")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if not 0.25 <= self.central_roi_fraction <= 1.0:
            raise ValueError("central_roi_fraction must be between 0.25 and 1.0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoMetadata:
    source_path: str
    file_size_bytes: int
    modified_time_ns: int
    container: str
    duration_s: float
    start_time_s: float
    width: int
    height: int
    display_width: int
    display_height: int
    rotation_degrees: int
    fps: float
    frame_count: int | None
    codec: str
    pixel_format: str | None
    bit_rate_bps: int | None
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    creation_time: str | None
    has_audio: bool

    @property
    def aspect_ratio(self) -> float:
        return self.display_width / self.display_height

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["aspect_ratio"] = round(self.aspect_ratio, 8)
        return result


@dataclass
class FrameMetrics:
    """Image diagnostics measured at a normalized analysis resolution."""

    brightness_mean: float
    contrast_stddev: float
    entropy_bits: float
    sharpness_laplacian: float
    edge_density: float
    dark_fraction: float
    clipped_fraction: float
    subject_coverage: float
    central_subject_coverage: float
    subject_border_touch: float
    coverage_source: str
    motion_mad: float = 0.0
    view_change: float = 0.0
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, float):
                result[key] = round(value, 8)
        return result


@dataclass
class FrameRecord:
    candidate_index: int
    source_frame_index: int
    timestamp_s: float
    path: str
    width: int
    height: int
    metrics: FrameMetrics
    selection_rank: int | None = None

    @property
    def is_keyframe(self) -> bool:
        return self.selection_rank is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_index": self.candidate_index,
            "source_frame_index": self.source_frame_index,
            "timestamp_s": round(self.timestamp_s, 8),
            "path": self.path,
            "width": self.width,
            "height": self.height,
            "is_keyframe": self.is_keyframe,
            "selection_rank": self.selection_rank,
            "metrics": self.metrics.to_dict(),
        }


@dataclass
class VideoAnalysis:
    source: str
    metadata: VideoMetadata
    config: IngestConfig
    frames: list[FrameRecord]
    keyframes: list[FrameRecord]
    warnings: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source,
            "metadata": self.metadata.to_dict(),
            "config": self.config.to_dict(),
            "frames": [frame.to_dict() for frame in self.frames],
            "keyframes": [frame.to_dict() for frame in self.keyframes],
            "warnings": list(self.warnings),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"


def _parse_rational(value: Any) -> float:
    if value in (None, "", "N/A", "0/0"):
        return 0.0
    text = str(value)
    try:
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _optional_int(value: Any) -> int | None:
    try:
        if value in (None, "", "N/A"):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _rotation_from_stream(stream: Mapping[str, Any]) -> int:
    rotation: Any = stream.get("tags", {}).get("rotate")
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = side_data["rotation"]
            break
    try:
        # ffprobe can report -90; normalizing makes display dimensions predictable.
        return int(round(float(rotation))) % 360 if rotation is not None else 0
    except (TypeError, ValueError):
        return 0


def probe_video(video_path: os.PathLike[str] | str, *, ffprobe: str = "ffprobe") -> VideoMetadata:
    """Read deterministic container/stream metadata with the local ffprobe.

    The path is passed as a subprocess argument rather than through a shell, so
    spaces and Unicode punctuation in phone-generated filenames are safe.
    """

    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise IngestError(f"video does not exist or is not a file: {path}")
    binary = shutil.which(ffprobe)
    if binary is None:
        raise IngestError("ffprobe was not found on PATH; install a local ffmpeg build")

    command = [
        binary,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise IngestError(f"ffprobe failed for {path}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or "unknown ffprobe error"
        raise IngestError(f"ffprobe could not read {path}: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise IngestError(f"ffprobe returned invalid JSON for {path}") from exc

    streams = payload.get("streams", [])
    video_stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    if video_stream is None:
        raise IngestError(f"file has no video stream: {path}")
    format_data = payload.get("format", {})
    duration = _parse_rational(video_stream.get("duration")) or _parse_rational(format_data.get("duration"))
    if duration <= 0:
        raise IngestError(f"video duration is unavailable or zero: {path}")
    fps = _parse_rational(video_stream.get("avg_frame_rate")) or _parse_rational(video_stream.get("r_frame_rate"))
    if fps <= 0:
        raise IngestError(f"video frame rate is unavailable or zero: {path}")

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise IngestError(f"video dimensions are invalid: {path}")
    rotation = _rotation_from_stream(video_stream)
    display_width, display_height = (height, width) if rotation in {90, 270} else (width, height)
    stat = path.stat()
    stream_tags = video_stream.get("tags", {})
    format_tags = format_data.get("tags", {})

    return VideoMetadata(
        source_path=str(path),
        file_size_bytes=stat.st_size,
        modified_time_ns=stat.st_mtime_ns,
        container=str(format_data.get("format_name") or path.suffix.lstrip(".") or "unknown"),
        duration_s=duration,
        start_time_s=_parse_rational(format_data.get("start_time")),
        width=width,
        height=height,
        display_width=display_width,
        display_height=display_height,
        rotation_degrees=rotation,
        fps=fps,
        frame_count=_optional_int(video_stream.get("nb_frames")),
        codec=str(video_stream.get("codec_name") or "unknown"),
        pixel_format=video_stream.get("pix_fmt"),
        bit_rate_bps=_optional_int(video_stream.get("bit_rate") or format_data.get("bit_rate")),
        color_space=video_stream.get("color_space"),
        color_transfer=video_stream.get("color_transfer"),
        color_primaries=video_stream.get("color_primaries"),
        creation_time=stream_tags.get("creation_time") or format_tags.get("creation_time"),
        has_audio=any(item.get("codec_type") == "audio" for item in streams),
    )


def _require_image_stack() -> None:
    if cv2 is None or np is None:
        raise IngestError("frame analysis requires local packages opencv-python and numpy")


def _analysis_image(frame: Any, long_side: int) -> Any:
    height, width = frame.shape[:2]
    scale = min(1.0, long_side / max(width, height))
    if scale >= 1.0:
        return frame.copy()
    return cv2.resize(
        frame,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _normalized_mask(mask: Any, shape: tuple[int, int]) -> Any:
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = np.max(mask_array, axis=2)
    mask_array = (mask_array > 0).astype(np.uint8)
    if mask_array.shape != shape:
        mask_array = cv2.resize(mask_array, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_array


def _mask_coverage(mask: Any, central_roi_fraction: float) -> tuple[float, float, float]:
    height, width = mask.shape
    coverage = float(np.mean(mask > 0))
    margin_x = round(width * (1.0 - central_roi_fraction) / 2.0)
    margin_y = round(height * (1.0 - central_roi_fraction) / 2.0)
    central = mask[margin_y : height - margin_y or None, margin_x : width - margin_x or None]
    central_coverage = float(np.mean(central > 0)) if central.size else coverage
    border_width = max(1, round(min(width, height) * 0.02))
    border = np.concatenate(
        (
            mask[:border_width, :].ravel(),
            mask[-border_width:, :].ravel(),
            mask[:, :border_width].ravel(),
            mask[:, -border_width:].ravel(),
        )
    )
    border_touch = float(np.mean(border > 0))
    return coverage, central_coverage, border_touch


def compute_frame_metrics(
    frame_bgr: Any,
    *,
    subject_mask: Any | None = None,
    temporal_mask: Any | None = None,
    previous_frame_bgr: Any | None = None,
    central_roi_fraction: float = 0.70,
) -> FrameMetrics:
    """Compute image-quality metrics, optionally using a true object mask.

    Supplying ``subject_mask`` is strongly preferred.  ``temporal_mask`` is the
    ingest preflight fallback and represents pixels that differ from the temporal
    median, not a semantic object segmentation.
    """

    _require_image_stack()
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        raise ValueError("frame_bgr must be a non-empty image")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] not in {3, 4}:
        raise ValueError("frame_bgr must have three or four channels")
    if frame_bgr.shape[2] == 4:
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    probability = histogram / max(1.0, float(histogram.sum()))
    nonzero_probability = probability[probability > 0]
    entropy = float(-np.sum(nonzero_probability * np.log2(nonzero_probability)))
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    median_intensity = float(np.median(gray))
    lower = int(max(0, 0.66 * median_intensity))
    upper = int(min(255, 1.33 * median_intensity + 20))
    if upper <= lower:
        lower, upper = 40, 120
    edges = cv2.Canny(gray, lower, upper)

    coverage_mask = subject_mask if subject_mask is not None else temporal_mask
    if coverage_mask is not None:
        normalized = _normalized_mask(coverage_mask, gray.shape)
        coverage, central_coverage, border_touch = _mask_coverage(normalized, central_roi_fraction)
        coverage_source = "object_mask" if subject_mask is not None else "temporal_change"
    else:
        coverage = 0.0
        central_coverage = 0.0
        border_touch = 0.0
        coverage_source = "unavailable"

    motion = 0.0
    if previous_frame_bgr is not None:
        previous = previous_frame_bgr
        if previous.shape[:2] != frame_bgr.shape[:2]:
            previous = cv2.resize(previous, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_AREA)
        previous_gray = cv2.cvtColor(previous[:, :, :3], cv2.COLOR_BGR2GRAY)
        motion = float(np.mean(cv2.absdiff(gray, previous_gray))) / 255.0

    return FrameMetrics(
        brightness_mean=float(np.mean(gray)) / 255.0,
        contrast_stddev=float(np.std(gray)) / 255.0,
        entropy_bits=entropy,
        sharpness_laplacian=laplacian_variance,
        edge_density=float(np.mean(edges > 0)),
        dark_fraction=float(np.mean(gray <= 12)),
        clipped_fraction=float(np.mean(gray >= 245)),
        subject_coverage=coverage,
        central_subject_coverage=central_coverage,
        subject_border_touch=border_touch,
        coverage_source=coverage_source,
        motion_mad=motion,
    )


def _temporal_change_masks(images: Sequence[Any], central_roi_fraction: float) -> list[Any]:
    if not images:
        return []
    # Bound peak memory for long clips: 64 evenly distributed samples are enough
    # to estimate a temporal median while avoiding a multi-hundred-MB stack.
    reference_step = max(1, math.ceil(len(images) / 64))
    reference_images = images[::reference_step][:64]
    reference = np.median(np.stack(reference_images, axis=0), axis=0).astype(np.uint8)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float32)
    output: list[Any] = []
    kernel_open = np.ones((3, 3), dtype=np.uint8)
    kernel_close = np.ones((7, 7), dtype=np.uint8)
    for image in images:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta = lab - reference_lab
        # Half-weight luminance: illumination changes should not dominate coverage.
        distance = np.sqrt(0.5 * delta[:, :, 0] ** 2 + delta[:, :, 1] ** 2 + delta[:, :, 2] ** 2)
        median = float(np.median(distance))
        mad = float(np.median(np.abs(distance - median)))
        threshold = max(10.0, median + 2.25 * max(mad, 1.0))
        raw = (distance >= threshold).astype(np.uint8)
        raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel_open)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel_close)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
        cleaned = np.zeros_like(raw)
        height, width = raw.shape
        minimum_area = max(12, round(height * width * 0.001))
        roi_margin_x = width * (1.0 - central_roi_fraction) / 2.0
        roi_margin_y = height * (1.0 - central_roi_fraction) / 2.0
        for label_index in range(1, count):
            x, y, component_width, component_height, area = stats[label_index]
            if area < minimum_area:
                continue
            intersects_central_roi = (
                x + component_width >= roi_margin_x
                and x <= width - roi_margin_x
                and y + component_height >= roi_margin_y
                and y <= height - roi_margin_y
            )
            if intersects_central_roi:
                cleaned[labels == label_index] = 1
        output.append(cleaned)
    return output


def _frame_descriptor(frame_bgr: Any) -> Any:
    """Compact descriptor used only for within-clip view diversity."""

    small = cv2.resize(frame_bgr, (32, 24), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    descriptor = lab.reshape(-1) / 255.0
    return descriptor


def _descriptor_distance(left: Any, right: Any) -> float:
    return float(np.mean(np.abs(left - right)))


def _robust_unit(values: Iterable[float], *, neutral: float = 0.5) -> list[float]:
    values_array = np.asarray(list(values), dtype=np.float64)
    if not values_array.size:
        return []
    low, high = np.percentile(values_array, [10, 90])
    if high - low <= 1e-12:
        return [neutral for _ in values_array]
    return [float(np.clip((value - low) / (high - low), 0.0, 1.0)) for value in values_array]


def _populate_population_scores(records: Sequence[FrameRecord], descriptors: Sequence[Any]) -> None:
    sharpness = _robust_unit(math.log1p(record.metrics.sharpness_laplacian) for record in records)
    contrast = _robust_unit(record.metrics.contrast_stddev for record in records)
    motion = _robust_unit(record.metrics.motion_mad for record in records)

    prior_descriptor: Any | None = None
    changes: list[float] = []
    for descriptor in descriptors:
        changes.append(_descriptor_distance(descriptor, prior_descriptor) if prior_descriptor is not None else 0.0)
        prior_descriptor = descriptor
    normalized_changes = _robust_unit(changes, neutral=0.0)

    for index, record in enumerate(records):
        metrics = record.metrics
        metrics.view_change = normalized_changes[index]
        exposure = max(0.0, 1.0 - abs(metrics.brightness_mean - 0.50) / 0.48)
        entropy = float(np.clip(metrics.entropy_bits / 7.5, 0.0, 1.0))
        clipping_penalty = float(np.clip(metrics.dark_fraction + metrics.clipped_fraction, 0.0, 0.5)) * 1.4
        if metrics.coverage_source == "unavailable":
            coverage = 0.5
        else:
            # Enough central changing/subject area is valuable; full-frame motion is not.
            coverage = min(1.0, metrics.central_subject_coverage / 0.22)
            if metrics.subject_border_touch > 0.70:
                coverage *= 0.75
        score = (
            0.32 * sharpness[index]
            + 0.18 * exposure
            + 0.12 * contrast[index]
            + 0.10 * entropy
            + 0.17 * coverage
            + 0.06 * motion[index]
            + 0.05 * normalized_changes[index]
            - 0.15 * clipping_penalty
        )
        metrics.quality_score = float(np.clip(score, 0.0, 1.0))


def _select_keyframe_indices(
    records: Sequence[FrameRecord],
    descriptors: Sequence[Any],
    count: int,
    min_gap_s: float,
) -> list[int]:
    if not records:
        return []
    target = min(count, len(records))
    qualities = np.asarray([record.metrics.quality_score for record in records], dtype=np.float64)
    seed = int(np.argmax(qualities))
    selected = [seed]

    pairwise_values: list[float] = []
    for left_index in range(len(descriptors)):
        for right_index in range(left_index + 1, len(descriptors)):
            pairwise_values.append(_descriptor_distance(descriptors[left_index], descriptors[right_index]))
    diversity_scale = float(np.percentile(pairwise_values, 90)) if pairwise_values else 1.0
    diversity_scale = max(diversity_scale, 1e-6)
    duration = max(record.timestamp_s for record in records) - min(record.timestamp_s for record in records)
    time_scale = max(duration / max(1, target - 1), min_gap_s, 1e-6)

    def best_remaining(enforce_gap: bool) -> int | None:
        best_index: int | None = None
        best_utility = -math.inf
        for index, record in enumerate(records):
            if index in selected:
                continue
            temporal_distance = min(abs(record.timestamp_s - records[item].timestamp_s) for item in selected)
            if enforce_gap and temporal_distance + 1e-9 < min_gap_s:
                continue
            visual_distance = min(_descriptor_distance(descriptors[index], descriptors[item]) for item in selected)
            visual_novelty = min(1.0, visual_distance / diversity_scale)
            temporal_spread = min(1.0, temporal_distance / time_scale)
            # Quality remains dominant, but max-min spread prevents near-duplicate views.
            utility = 0.58 * record.metrics.quality_score + 0.27 * visual_novelty + 0.15 * temporal_spread
            if (
                record.metrics.coverage_source == "temporal_change"
                and record.metrics.central_subject_coverage < 0.02
            ):
                # A common capture tail contains only the operator/background.
                # This is intentionally a soft penalty: real object masks remain
                # the authoritative way to decide whether the object is present.
                utility -= 0.35
            if utility > best_utility:
                best_index = index
                best_utility = utility
        return best_index

    while len(selected) < target:
        candidate = best_remaining(enforce_gap=True)
        if candidate is None:
            candidate = best_remaining(enforce_gap=False)
        if candidate is None:
            break
        selected.append(candidate)

    # selection_rank records greedy preference; keyframe list itself is chronological.
    for rank, index in enumerate(selected):
        records[index].selection_rank = rank
    return sorted(selected, key=lambda index: records[index].timestamp_s)


def _encode_frame(path: Path, frame_bgr: Any, config: IngestConfig) -> None:
    suffix = config.image_format.lower()
    if suffix == "jpeg":
        suffix = "jpg"
    extension = f".{suffix}"
    parameters: list[int] = []
    if suffix == "jpg":
        parameters = [int(cv2.IMWRITE_JPEG_QUALITY), config.jpeg_quality]
    success, encoded = cv2.imencode(extension, frame_bgr, parameters)
    if not success:
        raise IngestError(f"OpenCV could not encode extracted frame: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded.tobytes())
    os.replace(temporary, path)


def extract_frames(
    video_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    *,
    config: IngestConfig | None = None,
    metadata: VideoMetadata | None = None,
) -> tuple[list[FrameRecord], list[Any], list[Any]]:
    """Decode and persist candidate frames.

    Returns frame records plus analysis-resolution images and descriptors.  Most
    callers should use :func:`analyze_video`, which completes scoring/selection.
    """

    _require_image_stack()
    chosen_config = config or IngestConfig()
    chosen_config.validate()
    video_metadata = metadata or probe_video(video_path)
    path = Path(video_metadata.source_path)
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise IngestError(f"OpenCV could not open video: {path}")
    try:
        # OpenCV honors QuickTime orientation metadata by default on current builds.
        if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
            capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        effective_sample_fps = min(
            chosen_config.sample_fps,
            chosen_config.max_candidates / max(video_metadata.duration_s, 1e-6),
        )
        interval_s = 1.0 / effective_sample_fps
        next_timestamp_s = 0.0
        source_frame_index = 0
        records: list[FrameRecord] = []
        analysis_images: list[Any] = []
        descriptors: list[Any] = []

        while len(records) < chosen_config.max_candidates:
            ok, frame = capture.read()
            if not ok:
                break
            # Index/fps is stable for CFR clips; POS_MSEC handles variable-rate input
            # when the backend exposes it.  Reject obviously stale backend timestamps.
            index_timestamp = source_frame_index / video_metadata.fps
            backend_timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            timestamp_s = backend_timestamp if backend_timestamp >= index_timestamp * 0.5 else index_timestamp
            if timestamp_s + 0.5 / video_metadata.fps >= next_timestamp_s:
                candidate_index = len(records)
                extension = "jpg" if chosen_config.image_format.lower() == "jpeg" else chosen_config.image_format.lower()
                filename = f"frame_{candidate_index:04d}_{round(timestamp_s * 1000):09d}ms.{extension}"
                frame_path = destination / filename
                _encode_frame(frame_path, frame, chosen_config)
                analysis_frame = _analysis_image(frame, chosen_config.analysis_long_side)
                analysis_images.append(analysis_frame)
                descriptors.append(_frame_descriptor(analysis_frame))
                records.append(
                    FrameRecord(
                        candidate_index=candidate_index,
                        source_frame_index=source_frame_index,
                        timestamp_s=timestamp_s,
                        path=str(frame_path),
                        width=int(frame.shape[1]),
                        height=int(frame.shape[0]),
                        metrics=FrameMetrics(
                            brightness_mean=0.0,
                            contrast_stddev=0.0,
                            entropy_bits=0.0,
                            sharpness_laplacian=0.0,
                            edge_density=0.0,
                            dark_fraction=0.0,
                            clipped_fraction=0.0,
                            subject_coverage=0.0,
                            central_subject_coverage=0.0,
                            subject_border_touch=0.0,
                            coverage_source="unavailable",
                        ),
                    )
                )
                next_timestamp_s += interval_s
                while next_timestamp_s <= timestamp_s:
                    next_timestamp_s += interval_s
            source_frame_index += 1
    finally:
        capture.release()

    if not records:
        raise IngestError(f"no frames could be decoded from video: {path}")
    return records, analysis_images, descriptors


def _analysis_warnings(metadata: VideoMetadata, records: Sequence[FrameRecord], keyframes: Sequence[FrameRecord]) -> list[str]:
    warnings = [
        "Object masks are not available during ingest; subject coverage is a temporal-change heuristic, not semantic object coverage."
    ]
    if metadata.duration_s < 8.0:
        warnings.append("Capture is shorter than 8 seconds; viewpoint coverage may be insufficient.")
    if metadata.display_width < 1280 or metadata.display_height < 720:
        warnings.append("Capture resolution is below 720p; thin geometry and texture may be lost.")
    if metadata.fps < 20.0:
        warnings.append("Capture frame rate is below 20 fps; motion blur and tracking gaps are more likely.")
    if len(keyframes) < 8:
        warnings.append("Fewer than 8 keyframes were selected; reconstruction may be underconstrained.")
    sharp_values = np.asarray([record.metrics.sharpness_laplacian for record in records], dtype=np.float64)
    if sharp_values.size and float(np.median(sharp_values)) < 35.0:
        warnings.append("Median frame sharpness is low; recapture more slowly with brighter lighting.")
    clipped = np.asarray([record.metrics.dark_fraction + record.metrics.clipped_fraction for record in records])
    if clipped.size and float(np.median(clipped)) > 0.12:
        warnings.append("A substantial image area is crushed or clipped; lock exposure and improve lighting.")
    border_touch = np.asarray([record.metrics.subject_border_touch for record in records])
    if border_touch.size and float(np.percentile(border_touch, 75)) > 0.80:
        warnings.append("The moving subject frequently touches the frame boundary; leave more capture margin.")
    coverage = np.asarray([record.metrics.central_subject_coverage for record in records])
    if coverage.size and float(np.percentile(coverage, 75)) < 0.05:
        warnings.append("Little central motion/subject coverage was detected; verify that the object is visible and rotating.")
    return warnings


def analyze_video(
    video_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    *,
    config: IngestConfig | None = None,
) -> VideoAnalysis:
    """Run the complete local ingest preflight and keyframe selection."""

    chosen_config = config or IngestConfig()
    chosen_config.validate()
    metadata = probe_video(video_path)
    records, images, descriptors = extract_frames(
        video_path,
        output_dir,
        config=chosen_config,
        metadata=metadata,
    )
    temporal_masks = _temporal_change_masks(images, chosen_config.central_roi_fraction)
    previous: Any | None = None
    for record, image, temporal_mask in zip(records, images, temporal_masks, strict=True):
        record.metrics = compute_frame_metrics(
            image,
            temporal_mask=temporal_mask,
            previous_frame_bgr=previous,
            central_roi_fraction=chosen_config.central_roi_fraction,
        )
        previous = image
    _populate_population_scores(records, descriptors)
    chosen_indices = _select_keyframe_indices(
        records,
        descriptors,
        chosen_config.keyframe_count,
        chosen_config.min_keyframe_gap_s,
    )
    keyframes = [records[index] for index in chosen_indices]
    warnings = _analysis_warnings(metadata, records, keyframes)
    return VideoAnalysis(
        source=metadata.source_path,
        metadata=metadata,
        config=chosen_config,
        frames=records,
        keyframes=keyframes,
        warnings=warnings,
    )


def write_analysis_json(analysis: VideoAnalysis, output_path: os.PathLike[str] | str) -> Path:
    """Atomically write the versioned analysis artifact."""

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(analysis.to_json(), encoding="utf-8")
    os.replace(temporary, path)
    return path


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-only video ingest and keyframe analysis")
    parser.add_argument("video", help="input video path")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--max-candidates", type=int, default=240)
    parser.add_argument("--keyframes", type=int, default=24)
    parser.add_argument("--min-gap", type=float, default=0.30, help="minimum keyframe gap in seconds")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    config = IngestConfig(
        sample_fps=args.sample_fps,
        max_candidates=args.max_candidates,
        keyframe_count=args.keyframes,
        min_keyframe_gap_s=args.min_gap,
    )
    output_dir = Path(args.output).expanduser().resolve()
    try:
        analysis = analyze_video(args.video, output_dir / "frames", config=config)
        artifact = write_analysis_json(analysis, output_dir / "analysis.json")
    except (IngestError, ValueError) as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 2
    print(artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
