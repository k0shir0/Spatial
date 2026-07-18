"""Deterministic source-video texture baking for fitted soft-part models.

The baker is intentionally conservative.  It uses only configured source
frames, reviewed validity masks, and explicit exclusion polygons.  Pixels that
cannot be traced to clean source evidence are filled with a measured fabric
fallback and reported as inferred.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .gltf_texture import TexturedMeshPart, write_textured_glb
from .soft_parts import ellipsoid_mesh, rotation_matrix, superellipsoid_mesh, tube_mesh


CANONICAL_DIRECTIONS = {"front", "rear", "left", "right", "top", "bottom"}
ARRAY_EPSILON = 1e-8


@dataclass
class BaseMeshPart:
    index: int
    name: str
    spec: dict[str, Any]
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    material_class: str
    fallback_rgb: tuple[int, int, int] | None
    face_offset: int = 0


@dataclass
class PreparedView:
    index: int
    name: str
    direction_name: str
    frame_index: int
    image_bgr: np.ndarray
    safe_mask: np.ndarray
    boundary_distance: np.ndarray
    confidence: np.ndarray
    direction: np.ndarray
    right: np.ndarray
    up: np.ndarray
    projection_rect: tuple[float, float, float, float]
    sharpness: float
    decoded_sha256: str
    exposure_gain_bgr: np.ndarray
    depth_map: np.ndarray | None = None


@dataclass
class BakeArtifacts:
    parts: list[TexturedMeshPart]
    base_color_bgr: np.ndarray
    normal_bgr: np.ndarray
    metallic_roughness_bgr: np.ndarray
    observed_mask: np.ndarray
    selected_view: np.ndarray
    surface_mask: np.ndarray
    part_map: np.ndarray
    report: dict[str, Any]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _resolve_path(base: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _normalized_polygon(raw: Any, label: str) -> list[tuple[float, float]]:
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError(f"{label} must contain at least three points")
    polygon: list[tuple[float, float]] = []
    for index, point in enumerate(raw):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{label}[{index}] must be [x, y]")
        x = _finite_float(point[0], f"{label}[{index}].x")
        y = _finite_float(point[1], f"{label}[{index}].y")
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError(f"{label} coordinates must be normalized")
        polygon.append((x, y))
    return polygon


def load_texture_config(path: str | Path) -> dict[str, Any]:
    """Load, resolve, and fail-closed validate a texture bake config."""

    config_path = Path(path).resolve()
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read texture config {config_path}: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("texture config root must be an object")
    base = config_path.parent
    source_video = _resolve_path(base, raw.get("source_video"), "source_video")
    mesh_config = _resolve_path(base, raw.get("mesh_config"), "mesh_config")
    expected_hash = raw.get("source_video_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("source_video_sha256 must be a 64-character SHA-256")
    actual_hash = sha256_file(source_video)
    if actual_hash.lower() != expected_hash.lower():
        raise ValueError(
            f"source video hash mismatch: configured {expected_hash}, observed {actual_hash}"
        )

    atlas = raw.get("atlas")
    if not isinstance(atlas, dict):
        raise ValueError("atlas must be an object")
    size = atlas.get("size")
    if isinstance(size, bool) or not isinstance(size, int) or not 128 <= size <= 4096:
        raise ValueError("atlas.size must be an integer from 128 to 4096")
    if size & (size - 1):
        raise ValueError("atlas.size must be a power of two")
    padding = atlas.get("padding", 4)
    if isinstance(padding, bool) or not isinstance(padding, int) or not 2 <= padding <= 64:
        raise ValueError("atlas.padding must be an integer from 2 to 64")
    quality = atlas.get("base_color_jpeg_quality", 90)
    if isinstance(quality, bool) or not isinstance(quality, int) or not 70 <= quality <= 100:
        raise ValueError("atlas.base_color_jpeg_quality must be from 70 to 100")
    detail_size = atlas.get("detail_map_size", size)
    if (
        isinstance(detail_size, bool)
        or not isinstance(detail_size, int)
        or detail_size < 128
        or detail_size > size
        or detail_size & (detail_size - 1)
    ):
        raise ValueError("atlas.detail_map_size must be a power of two no larger than atlas.size")

    views = raw.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError("views must be a non-empty list")
    seen_names: set[str] = set()
    seen_frames: set[int] = set()
    normalized_views: list[dict[str, Any]] = []
    for index, item in enumerate(views):
        if not isinstance(item, dict):
            raise ValueError(f"views[{index}] must be an object")
        name = item.get("name")
        direction_name = item.get("direction")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ValueError(f"views[{index}].name must be unique and non-empty")
        if direction_name not in CANONICAL_DIRECTIONS:
            raise ValueError(f"views[{index}].direction must be canonical")
        frame_index = item.get("frame_index")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError(f"views[{index}].frame_index must be non-negative")
        if frame_index in seen_frames:
            raise ValueError("configured source frame indices must be unique")
        projection = item.get("projection_rect_normalized")
        if not isinstance(projection, list) or len(projection) != 4:
            raise ValueError(f"views[{index}] needs projection_rect_normalized")
        projection_values = [
            _finite_float(value, f"views[{index}].projection_rect_normalized")
            for value in projection
        ]
        if projection_values[2] <= projection_values[0] or projection_values[3] <= projection_values[1]:
            raise ValueError(f"views[{index}] has an invalid projection rectangle")
        mask = _resolve_path(base, item.get("mask"), f"views[{index}].mask")
        exclusions_raw = item.get("exclude_polygons_normalized")
        if not isinstance(exclusions_raw, list):
            raise ValueError(
                f"views[{index}] must explicitly provide exclude_polygons_normalized"
            )
        exclusions = [
            _normalized_polygon(poly, f"views[{index}].exclude_polygons_normalized[{poly_index}]")
            for poly_index, poly in enumerate(exclusions_raw)
        ]
        trusted = [
            _normalized_polygon(poly, f"views[{index}].trusted_polygons_normalized[{poly_index}]")
            for poly_index, poly in enumerate(item.get("trusted_polygons_normalized", []))
        ]
        valid = [
            _normalized_polygon(poly, f"views[{index}].valid_polygons_normalized[{poly_index}]")
            for poly_index, poly in enumerate(item.get("valid_polygons_normalized", []))
        ]
        seed = item.get("component_seed_normalized")
        if seed is not None:
            if not isinstance(seed, list) or len(seed) != 2:
                raise ValueError(f"views[{index}].component_seed_normalized must be [x, y]")
            seed = [
                _finite_float(seed[0], f"views[{index}].component_seed_normalized.x"),
                _finite_float(seed[1], f"views[{index}].component_seed_normalized.y"),
            ]
            if not all(0.0 <= value <= 1.0 for value in seed):
                raise ValueError("component seed must be normalized")
        normalized = dict(item)
        normalized.update(
            {
                "name": name,
                "direction": direction_name,
                "frame_index": frame_index,
                "mask": str(mask),
                "projection_rect_normalized": projection_values,
                "exclude_polygons_normalized": exclusions,
                "trusted_polygons_normalized": trusted,
                "valid_polygons_normalized": valid,
                "component_seed_normalized": seed,
                "yaw_degrees": _finite_float(item.get("yaw_degrees", 0.0), "yaw"),
                "elevation_degrees": _finite_float(item.get("elevation_degrees", 0.0), "elevation"),
                "roll_degrees": _finite_float(item.get("roll_degrees", 0.0), "roll"),
                "erode_pixels": int(item.get("erode_pixels", 4)),
                "max_hole_area_fraction": _finite_float(
                    item.get("max_hole_area_fraction", 0.008), "max_hole_area_fraction"
                ),
            }
        )
        if normalized["erode_pixels"] < 0 or normalized["erode_pixels"] > 64:
            raise ValueError("erode_pixels must be from 0 to 64")
        if not 0.0 <= normalized["max_hole_area_fraction"] <= 0.05:
            raise ValueError("max_hole_area_fraction must be from 0 to 0.05")
        normalized_views.append(normalized)
        seen_names.add(name)
        seen_frames.add(frame_index)

    required = set(raw.get("required_directions", []))
    if required and required != CANONICAL_DIRECTIONS:
        raise ValueError("required_directions must list all six canonical directions")
    if required and {item["direction"] for item in normalized_views} != CANONICAL_DIRECTIONS:
        raise ValueError("the configured views do not cover all six canonical directions")

    result = dict(raw)
    result.update(
        {
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "source_video": str(source_video),
            "source_video_sha256": actual_hash,
            "mesh_config": str(mesh_config),
            "mesh_config_sha256": sha256_file(mesh_config),
            "views": normalized_views,
            "atlas": dict(atlas),
        }
    )
    return result


def _load_mesh_parts(config: Mapping[str, Any]) -> tuple[list[BaseMeshPart], dict[str, Any]]:
    mesh_path = Path(str(config["mesh_config"]))
    mesh_raw = json.loads(mesh_path.read_text(encoding="utf-8"))
    if not isinstance(mesh_raw, dict) or not isinstance(mesh_raw.get("parts"), list):
        raise ValueError("mesh config must contain a parts list")
    material_classes = config.get("part_material_classes")
    if not isinstance(material_classes, list) or len(material_classes) != len(mesh_raw["parts"]):
        raise ValueError("part_material_classes must contain one entry per mesh part")
    materials = config.get("materials")
    if not isinstance(materials, dict) or "fabric" not in materials:
        raise ValueError("materials must define at least the fabric class")
    part_names = config.get("part_names")
    if part_names is not None and (
        not isinstance(part_names, list) or len(part_names) != len(mesh_raw["parts"])
    ):
        raise ValueError("part_names must contain one entry per mesh part")

    parts: list[BaseMeshPart] = []
    face_offset = 0
    for index, item in enumerate(mesh_raw["parts"]):
        if not isinstance(item, dict):
            raise ValueError(f"mesh part {index} must be an object")
        kind = item.get("type")
        if kind == "ellipsoid":
            vertices, faces, normals = ellipsoid_mesh(
                item["center"],
                item["radii"],
                euler_degrees=item.get("rotation", [0, 0, 0]),
                rings=int(item.get("rings", 20)),
                segments=int(item.get("segments", 32)),
            )
        elif kind == "superellipsoid":
            vertices, faces, normals = superellipsoid_mesh(
                item["center"],
                item["radii"],
                vertical_exponent=float(item.get("vertical_exponent", 0.65)),
                horizontal_exponent=float(item.get("horizontal_exponent", 0.75)),
                euler_degrees=item.get("rotation", [0, 0, 0]),
                rings=int(item.get("rings", 24)),
                segments=int(item.get("segments", 40)),
            )
        elif kind == "tube":
            vertices, faces, normals = tube_mesh(
                item["points"],
                float(item["radius"]),
                segments=int(item.get("segments", 12)),
            )
        else:
            raise ValueError(f"unsupported soft part type: {kind!r}")
        material_class = material_classes[index]
        if not isinstance(material_class, str) or material_class not in materials:
            raise ValueError(f"part {index} references an unknown material class")
        material = materials[material_class]
        fallback_rgb: tuple[int, int, int] | None = None
        if isinstance(material, dict) and "fallback_rgb" in material:
            raw_color = material["fallback_rgb"]
            if (
                not isinstance(raw_color, list)
                or len(raw_color) != 3
                or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_color)
                or any(value < 0 or value > 255 for value in raw_color)
            ):
                raise ValueError(f"materials.{material_class}.fallback_rgb must be byte RGB")
            fallback_rgb = tuple(raw_color)
        parts.append(
            BaseMeshPart(
                index=index,
                name=str(
                    item.get(
                        "name",
                        part_names[index] if part_names is not None else f"part_{index:02d}",
                    )
                ),
                spec=dict(item),
                vertices=vertices,
                faces=faces,
                normals=normals,
                material_class=material_class,
                fallback_rgb=fallback_rgb,
                face_offset=face_offset,
            )
        )
        face_offset += len(faces)
    return parts, mesh_raw


def _validate_tiles(config: Mapping[str, Any], part_count: int) -> list[tuple[int, int, int, int]]:
    atlas = config["atlas"]
    size = int(atlas["size"])
    padding = int(atlas.get("padding", 4))
    tiles = atlas.get("part_tiles_px")
    if not isinstance(tiles, list) or len(tiles) != part_count:
        raise ValueError("atlas.part_tiles_px must contain one rectangle per mesh part")
    result: list[tuple[int, int, int, int]] = []
    occupancy = np.zeros((size, size), dtype=np.uint8)
    for index, raw in enumerate(tiles):
        if (
            not isinstance(raw, list)
            or len(raw) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in raw)
        ):
            raise ValueError(f"atlas tile {index} must be integer [x, y, width, height]")
        x, y, width, height = raw
        if x < 0 or y < 0 or width <= 2 * padding + 4 or height <= 2 * padding + 4:
            raise ValueError(f"atlas tile {index} is too small or outside the atlas")
        if x + width > size or y + height > size:
            raise ValueError(f"atlas tile {index} exceeds atlas bounds")
        if np.any(occupancy[y : y + height, x : x + width]):
            raise ValueError(f"atlas tile {index} overlaps an earlier tile")
        occupancy[y : y + height, x : x + width] = 1
        result.append((x, y, width, height))
    return result


def _ellipsoid_vertex_uv(part: BaseMeshPart) -> tuple[np.ndarray, np.ndarray]:
    spec = part.spec
    center = np.asarray(spec["center"], dtype=np.float64)
    radii = np.asarray(spec["radii"], dtype=np.float64)
    rotation = rotation_matrix(spec.get("rotation", [0, 0, 0]))
    local = (np.asarray(part.vertices, dtype=np.float64) - center) @ rotation
    unit = local / radii
    lengths = np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), ARRAY_EPSILON)
    unit /= lengths
    longitude = np.mod(np.arctan2(unit[:, 2], unit[:, 0]) / (2.0 * math.pi), 1.0)
    # Inverse-transform float noise can turn the canonical zero meridian into
    # 0.9999999.  Snap it back before deciding which triangles cross the seam.
    longitude[longitude > 1.0 - 1e-5] = 0.0
    latitude = np.arccos(np.clip(unit[:, 1], -1.0, 1.0)) / math.pi
    return np.column_stack((longitude, latitude)), unit


def _tube_face_uvs(part: BaseMeshPart) -> np.ndarray:
    spec = part.spec
    segments = int(spec.get("segments", 12))
    ring_count = len(spec["points"])
    side_face_count = 2 * (ring_count - 1) * segments
    result = np.zeros((len(part.faces), 3, 2), dtype=np.float64)
    for face_index, face in enumerate(part.faces):
        if face_index < side_face_count:
            local: list[list[float]] = []
            for vertex_index in face:
                ring = int(vertex_index) // segments
                segment = int(vertex_index) % segments
                local.append([segment / segments, 0.25 + 0.5 * ring / (ring_count - 1)])
            uv = np.asarray(local, dtype=np.float64)
            if np.ptp(uv[:, 0]) > 0.5:
                uv[uv[:, 0] < 0.5, 0] += 1.0
            result[face_index] = uv
            continue
        cap_sequence = face_index - side_face_count
        is_start = cap_sequence % 2 == 0
        center = np.array([0.25, 0.11]) if is_start else np.array([0.75, 0.89])
        radius = 0.10
        uv = []
        for vertex_index in face:
            if int(vertex_index) >= ring_count * segments:
                uv.append(center.copy())
            else:
                segment = int(vertex_index) % segments
                angle = 2.0 * math.pi * segment / segments
                uv.append(center + radius * np.array([math.cos(angle), math.sin(angle)]))
        result[face_index] = np.asarray(uv)
    return result


def _part_face_uvs(part: BaseMeshPart) -> np.ndarray:
    if part.spec["type"] == "tube":
        return _tube_face_uvs(part)
    vertex_uv, unit = _ellipsoid_vertex_uv(part)
    result = vertex_uv[part.faces].astype(np.float64, copy=True)
    for face_index, face in enumerate(part.faces):
        uv = result[face_index]
        pole = np.abs(unit[face, 1]) > 0.999999
        if np.any(pole):
            non_pole = ~pole
            if np.any(non_pole) and np.ptp(uv[non_pole, 0]) > 0.5:
                adjust = non_pole & (uv[:, 0] < 0.5)
                uv[adjust, 0] += 1.0
            replacement = float(np.mean(uv[non_pole, 0])) if np.any(non_pole) else 0.5
            uv[pole, 0] = replacement
        elif np.ptp(uv[:, 0]) > 0.5:
            uv[uv[:, 0] < 0.5, 0] += 1.0
        result[face_index] = uv
    return result


def _map_face_uvs_to_atlas(
    local: np.ndarray,
    tile: tuple[int, int, int, int],
    atlas_size: int,
    padding: int,
) -> np.ndarray:
    x, y, width, height = tile
    usable_width = width - 2 * padding - 1
    usable_height = height - 2 * padding - 1
    image_x = x + padding + np.clip(local[..., 0], 0.0, 1.0) * usable_width
    image_y = y + padding + np.clip(local[..., 1], 0.0, 1.0) * usable_height
    u = image_x / (atlas_size - 1)
    # glTF defines (0, 0) at the upper-left image pixel. Keep atlas Y in
    # that convention instead of applying the traditional OpenGL V flip.
    v = image_y / (atlas_size - 1)
    return np.stack((u, v), axis=-1)


def _triangle_tangents(
    vertices: np.ndarray,
    normals: np.ndarray,
    uvs: np.ndarray,
) -> np.ndarray:
    edge1 = vertices[1] - vertices[0]
    edge2 = vertices[2] - vertices[0]
    delta1 = uvs[1] - uvs[0]
    delta2 = uvs[2] - uvs[0]
    determinant = float(delta1[0] * delta2[1] - delta1[1] * delta2[0])
    if abs(determinant) < 1e-12:
        raise ValueError("texture parameterization produced a degenerate UV triangle")
    reciprocal = 1.0 / determinant
    tangent_raw = reciprocal * (delta2[1] * edge1 - delta1[1] * edge2)
    bitangent_raw = reciprocal * (-delta2[0] * edge1 + delta1[0] * edge2)
    result = np.empty((3, 4), dtype=np.float32)
    for corner in range(3):
        normal = normals[corner].astype(np.float64)
        normal_length = float(np.linalg.norm(normal))
        if normal_length < 1e-10:
            raise ValueError("texture tangent construction received a zero-length normal")
        normal /= normal_length
        tangent = tangent_raw - normal * float(np.dot(normal, tangent_raw))
        length = float(np.linalg.norm(tangent))
        if length < 1e-10:
            reference = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(reference, normal))) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            tangent = np.cross(reference, normal)
            length = float(np.linalg.norm(tangent))
        tangent /= length
        handedness = -1.0 if float(np.dot(np.cross(normal, tangent), bitangent_raw)) < 0.0 else 1.0
        result[corner, :3] = tangent
        result[corner, 3] = handedness
    return result


def build_textured_parts(
    parts: Sequence[BaseMeshPart],
    tiles: Sequence[tuple[int, int, int, int]],
    atlas_size: int,
    padding: int,
) -> list[TexturedMeshPart]:
    """Expand only UV seams (per corner here) and author tangent frames."""

    result: list[TexturedMeshPart] = []
    for part, tile in zip(parts, tiles):
        local_face_uv = _part_face_uvs(part)
        atlas_face_uv = _map_face_uvs_to_atlas(local_face_uv, tile, atlas_size, padding)
        expanded_vertices = part.vertices[part.faces].reshape(-1, 3).astype(np.float32)
        expanded_normals = part.normals[part.faces].reshape(-1, 3).astype(np.float32)
        expanded_uvs = atlas_face_uv.reshape(-1, 2).astype(np.float32)
        expanded_faces = np.arange(len(expanded_vertices), dtype=np.int32).reshape(-1, 3)
        tangents = np.empty((len(expanded_vertices), 4), dtype=np.float32)
        for face_index in range(len(part.faces)):
            offset = face_index * 3
            tangents[offset : offset + 3] = _triangle_tangents(
                expanded_vertices[offset : offset + 3],
                expanded_normals[offset : offset + 3],
                expanded_uvs[offset : offset + 3],
            )
        result.append(
            TexturedMeshPart(
                name=part.name,
                vertices=expanded_vertices,
                faces=expanded_faces,
                normals=expanded_normals,
                uvs=expanded_uvs,
                tangents=tangents,
                material_class=part.material_class,
            )
        )
    return result


def decode_video_frames(
    path: str | Path,
    frame_indices: Iterable[int],
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Decode exact, zero-based source-frame indices in one sequential pass."""

    requested = sorted(set(int(index) for index in frame_indices))
    if not requested or requested[0] < 0:
        raise ValueError("at least one non-negative source frame index is required")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    capture = cv2.VideoCapture(str(Path(path)))
    if not capture.isOpened():
        raise ValueError(f"OpenCV could not open video: {path}")
    reported_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: dict[int, np.ndarray] = {}
    target_set = set(requested)
    index = 0
    try:
        while index <= requested[-1]:
            ok, frame = capture.read()
            if not ok:
                break
            if index in target_set:
                frames[index] = np.ascontiguousarray(frame)
            index += 1
    finally:
        capture.release()
    missing = [item for item in requested if item not in frames]
    if missing:
        raise ValueError(f"source video ended before configured frames: {missing}")
    metadata = {
        "width": width,
        "height": height,
        "fps": round(fps, 9),
        "reported_frame_count": reported_frames,
        "decoded_through_frame": index - 1,
    }
    return frames, metadata


def _polygon_mask(
    shape: tuple[int, int],
    polygons: Sequence[Sequence[tuple[float, float]]],
) -> np.ndarray:
    height, width = shape
    result = np.zeros((height, width), dtype=np.uint8)
    for polygon in polygons:
        points = np.asarray(
            [
                [
                    int(round(x * (width - 1))),
                    int(round(y * (height - 1))),
                ]
                for x, y in polygon
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(result, [points], 255, lineType=cv2.LINE_8)
    return result


def fill_small_mask_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Restore enclosed embroidery holes without filling large hand cutouts."""

    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    inverse = cv2.bitwise_not(binary)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    height, width = binary.shape
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_border = (
            x == 0
            or y == 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if not touches_border and int(area) <= max_area:
            binary[labels == label] = 255
    return binary


def _component_at_seed(mask: np.ndarray, seed: Sequence[float] | None) -> np.ndarray:
    if seed is None:
        return mask
    count, labels = cv2.connectedComponents(np.where(mask > 0, 1, 0).astype(np.uint8), 8)
    if count <= 1:
        raise ValueError("view validity mask has no foreground component")
    height, width = mask.shape
    x = min(width - 1, max(0, int(round(float(seed[0]) * (width - 1)))))
    y = min(height - 1, max(0, int(round(float(seed[1]) * (height - 1)))))
    label = int(labels[y, x])
    if label == 0:
        # Use the closest foreground pixel rather than silently selecting every spill.
        foreground = np.column_stack(np.nonzero(labels > 0))
        if not len(foreground):
            raise ValueError("view validity mask has no foreground pixels")
        distances = np.square(foreground[:, 0] - y) + np.square(foreground[:, 1] - x)
        closest = foreground[int(np.argmin(distances))]
        label = int(labels[int(closest[0]), int(closest[1])])
    return np.where(labels == label, 255, 0).astype(np.uint8)


def prepare_safe_mask(mask: np.ndarray, view: Mapping[str, Any]) -> np.ndarray:
    """Build the only pixel set that source projection is allowed to sample."""

    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    height, width = mask.shape
    max_hole_area = int(
        round(float(view.get("max_hole_area_fraction", 0.008)) * height * width)
    )
    safe = fill_small_mask_holes(mask, max_hole_area)
    safe = _component_at_seed(safe, view.get("component_seed_normalized"))

    valid_polygons = view.get("valid_polygons_normalized", [])
    if valid_polygons:
        safe = cv2.bitwise_and(safe, _polygon_mask((height, width), valid_polygons))
    trusted_polygons = view.get("trusted_polygons_normalized", [])
    if trusted_polygons:
        safe = cv2.bitwise_or(safe, _polygon_mask((height, width), trusted_polygons))
    exclusions = view.get("exclude_polygons_normalized", [])
    if exclusions:
        exclusion_mask = _polygon_mask((height, width), exclusions)
        safe[exclusion_mask > 0] = 0
    erode_pixels = int(view.get("erode_pixels", 4))
    if erode_pixels:
        kernel_size = 2 * erode_pixels + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        safe = cv2.erode(safe, kernel, iterations=1)
    if int(np.count_nonzero(safe)) < 64:
        raise ValueError(f"view {view.get('name')} has too few reviewed valid pixels")
    return safe


def camera_basis(
    yaw_degrees: float,
    elevation_degrees: float,
    roll_degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return object-to-camera direction, screen-right, and screen-up vectors.

    The fitted plush front is +Z and +Y is up.  Positive yaw moves the camera
    toward +X; positive elevation moves it toward +Y.
    """

    yaw = math.radians(float(yaw_degrees))
    elevation = math.radians(float(elevation_degrees))
    roll = math.radians(float(roll_degrees))
    direction = np.array(
        [
            math.sin(yaw) * math.cos(elevation),
            math.sin(elevation),
            math.cos(yaw) * math.cos(elevation),
        ],
        dtype=np.float64,
    )
    direction /= np.linalg.norm(direction)
    right = np.array([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float64)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    up /= np.linalg.norm(up)
    if roll:
        rolled_right = math.cos(roll) * right - math.sin(roll) * up
        rolled_up = math.sin(roll) * right + math.cos(roll) * up
        right, up = rolled_right, rolled_up
    return direction.astype(np.float32), right.astype(np.float32), up.astype(np.float32)


def _project_points(
    points: np.ndarray,
    view: PreparedView,
    projection_bounds: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_right, max_right, min_up, max_up = projection_bounds
    rect_left, rect_top, rect_right, rect_bottom = view.projection_rect
    horizontal = np.asarray(points) @ view.right
    vertical = np.asarray(points) @ view.up
    depth = np.asarray(points) @ view.direction
    horizontal_fraction = (horizontal - min_right) / max(max_right - min_right, ARRAY_EPSILON)
    vertical_fraction = (max_up - vertical) / max(max_up - min_up, ARRAY_EPSILON)
    image_width = view.image_bgr.shape[1]
    image_height = view.image_bgr.shape[0]
    x = (rect_left + horizontal_fraction * (rect_right - rect_left)) * (image_width - 1)
    y = (rect_top + vertical_fraction * (rect_bottom - rect_top)) * (image_height - 1)
    return x.astype(np.float32), y.astype(np.float32), depth.astype(np.float32)


def _projection_bounds(points: np.ndarray, right: np.ndarray, up: np.ndarray) -> tuple[float, float, float, float]:
    horizontal = points @ right
    vertical = points @ up
    return (
        float(horizontal.min()),
        float(horizontal.max()),
        float(vertical.min()),
        float(vertical.max()),
    )


def _view_confidence(image_bgr: np.ndarray) -> np.ndarray:
    image = image_bgr.astype(np.float32)
    luminance = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    maximum = image.max(axis=2)
    minimum = image.min(axis=2)
    confidence = np.ones(image_bgr.shape[:2], dtype=np.float32)
    confidence -= 0.85 * (maximum >= 250.0)
    confidence -= 0.85 * (minimum <= 4.0)
    confidence -= 0.75 * (luminance <= 24.0)
    return np.clip(confidence, 0.0, 1.0)


def _patch_pixels(
    image: np.ndarray,
    rect: Sequence[float],
) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(0, min(width - 1, int(round(float(rect[0]) * (width - 1)))))
    y0 = max(0, min(height - 1, int(round(float(rect[1]) * (height - 1)))))
    x1 = max(x0 + 1, min(width, int(round(float(rect[2]) * (width - 1))) + 1))
    y1 = max(y0 + 1, min(height, int(round(float(rect[3]) * (height - 1))) + 1))
    return image[y0:y1, x0:x1]


def _fabric_patch_statistics(
    frames_by_name: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    patches = config.get("fabric_patches")
    if not isinstance(patches, list) or not patches:
        raise ValueError("fabric_patches must contain reviewed clean source patches")
    medians: list[np.ndarray] = []
    detail_tiles: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    fabric = config.get("fabric", {})
    tile_size = int(fabric.get("detail_tile_size", 128))
    sigma = float(fabric.get("source_highpass_sigma", 4.0))
    if tile_size < 32 or tile_size > 512 or sigma <= 0.0:
        raise ValueError("invalid fabric detail tile size or high-pass sigma")
    for index, item in enumerate(patches):
        if not isinstance(item, dict) or item.get("view") not in frames_by_name:
            raise ValueError(f"fabric_patches[{index}] references an unknown view")
        rect = item.get("rect_normalized")
        if not isinstance(rect, list) or len(rect) != 4:
            raise ValueError(f"fabric_patches[{index}] needs rect_normalized")
        crop = _patch_pixels(frames_by_name[str(item["view"])], rect)
        if crop.size < 64:
            raise ValueError(f"fabric_patches[{index}] is empty")
        median = np.median(crop.reshape(-1, 3), axis=0).astype(np.float32)
        medians.append(median)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        resized = cv2.resize(gray, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        exposure = max(float(np.median(resized)), 1.0)
        normalized = resized * (128.0 / exposure)
        highpass = normalized - cv2.GaussianBlur(normalized, (0, 0), sigma)
        standard_deviation = max(float(np.std(highpass)), 1e-5)
        detail_tiles.append(np.clip(highpass / standard_deviation, -3.0, 3.0))
        records.append(
            {
                "view": item["view"],
                "rect_normalized": [float(value) for value in rect],
                "median_rgb": [int(round(value)) for value in median[::-1]],
            }
        )
    target_bgr = np.median(np.stack(medians), axis=0).astype(np.float32)
    return target_bgr, detail_tiles, records


def _prepare_views(
    config: Mapping[str, Any],
    decoded: Mapping[int, np.ndarray],
    target_fabric_bgr: np.ndarray,
) -> list[PreparedView]:
    prepared: list[PreparedView] = []
    sharpness_values: list[float] = []
    temporary: list[tuple[dict[str, Any], np.ndarray, np.ndarray, float]] = []
    for item in config["views"]:
        frame = decoded[int(item["frame_index"])]
        mask = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"could not decode validity mask: {item['mask']}")
        if mask.shape != frame.shape[:2]:
            raise ValueError(
                f"view {item['name']} mask dimensions {mask.shape[::-1]} do not match "
                f"frame dimensions {frame.shape[1]}x{frame.shape[0]}"
            )
        safe = prepare_safe_mask(mask, item)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        sharpness = float(np.var(laplacian[safe > 0]))
        sharpness_values.append(sharpness)
        temporary.append((item, frame, safe, sharpness))
    low = min(sharpness_values)
    high = max(sharpness_values)
    for index, (item, frame, safe, sharpness) in enumerate(temporary):
        safe_pixels = frame[safe > 0]
        observed_median = np.median(safe_pixels, axis=0).astype(np.float32)
        gain = np.clip(target_fabric_bgr / np.maximum(observed_median, 1.0), 0.72, 1.35)
        direction, right, up = camera_basis(
            item["yaw_degrees"], item["elevation_degrees"], item["roll_degrees"]
        )
        normalized_sharpness = 1.0 if high <= low else (sharpness - low) / (high - low)
        prepared.append(
            PreparedView(
                index=index,
                name=item["name"],
                direction_name=item["direction"],
                frame_index=int(item["frame_index"]),
                image_bgr=frame,
                safe_mask=safe,
                boundary_distance=cv2.distanceTransform(safe, cv2.DIST_L2, 5),
                confidence=_view_confidence(frame),
                direction=direction,
                right=right,
                up=up,
                projection_rect=tuple(float(value) for value in item["projection_rect_normalized"]),
                sharpness=float(normalized_sharpness),
                decoded_sha256=sha256_bytes(frame.tobytes(order="C")),
                exposure_gain_bgr=gain,
            )
        )
    return prepared


def _flatten_mesh(
    parts: Sequence[BaseMeshPart],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[set[int]]]]:
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    face_normals: list[np.ndarray] = []
    part_indices: list[np.ndarray] = []
    adjacency: list[list[set[int]]] = []
    vertex_offset = 0
    for part in parts:
        vertices.append(part.vertices)
        triangles.append(part.faces + vertex_offset)
        normals = part.normals[part.faces].mean(axis=1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), ARRAY_EPSILON)
        face_normals.append(normals.astype(np.float32))
        part_indices.append(np.full(len(part.faces), part.index, dtype=np.int16))
        local_adjacency = [set() for _ in range(len(part.faces))]
        edge_owner: dict[tuple[int, int], int] = {}
        for face_index, face in enumerate(part.faces):
            for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                edge = tuple(sorted((int(first), int(second))))
                previous = edge_owner.get(edge)
                if previous is None:
                    edge_owner[edge] = face_index
                else:
                    local_adjacency[face_index].add(previous)
                    local_adjacency[previous].add(face_index)
        adjacency.append(local_adjacency)
        vertex_offset += len(part.vertices)
    return (
        np.vstack(vertices).astype(np.float32),
        np.vstack(triangles).astype(np.int32),
        np.vstack(face_normals).astype(np.float32),
        np.concatenate(part_indices),
        adjacency,
    )


def _build_depth_map(
    projected_x: np.ndarray,
    projected_y: np.ndarray,
    depth: np.ndarray,
    triangles: np.ndarray,
    image_shape: tuple[int, int],
    resolution: int,
) -> np.ndarray:
    image_height, image_width = image_shape
    width = int(resolution)
    height = max(16, int(round(resolution * image_height / image_width)))
    scale_x = (width - 1) / max(image_width - 1, 1)
    scale_y = (height - 1) / max(image_height - 1, 1)
    centers = depth[triangles].mean(axis=1)
    order = np.argsort(centers, kind="mergesort")
    result = np.full((height, width), -np.inf, dtype=np.float32)
    for face_index in order:
        face = triangles[int(face_index)]
        polygon = np.column_stack(
            (projected_x[face] * scale_x, projected_y[face] * scale_y)
        )
        if not np.isfinite(polygon).all():
            continue
        points = np.rint(polygon).astype(np.int32)
        cv2.fillConvexPoly(result, points, float(centers[int(face_index)]), lineType=cv2.LINE_8)
    return result


def score_face_views(
    parts: Sequence[BaseMeshPart],
    views: Sequence[PreparedView],
    scoring: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, float, float]]]:
    """Score configured views for every face and smooth labels over adjacency."""

    vertices, triangles, face_normals, _, adjacency = _flatten_mesh(parts)
    face_centers = vertices[triangles].mean(axis=1)
    weights = {
        "facing": float(scoring.get("facing_weight", 0.38)),
        "resolution": float(scoring.get("resolution_weight", 0.16)),
        "boundary": float(scoring.get("boundary_weight", 0.18)),
        "sharpness": float(scoring.get("sharpness_weight", 0.14)),
        "confidence": float(scoring.get("confidence_weight", 0.14)),
    }
    min_facing = float(scoring.get("minimum_facing", 0.12))
    boundary_scale = float(scoring.get("boundary_distance_pixels", 32.0))
    depth_tolerance = float(scoring.get("depth_tolerance", 0.035))
    depth_resolution = int(scoring.get("zbuffer_resolution", 320))
    scores = np.full((len(triangles), len(views)), -np.inf, dtype=np.float32)
    bounds: list[tuple[float, float, float, float]] = []

    for view_index, view in enumerate(views):
        projection_bounds = _projection_bounds(vertices, view.right, view.up)
        bounds.append(projection_bounds)
        x, y, depth = _project_points(vertices, view, projection_bounds)
        view.depth_map = _build_depth_map(
            x,
            y,
            depth,
            triangles,
            view.image_bgr.shape[:2],
            depth_resolution,
        )
        center_x, center_y, center_depth = _project_points(face_centers, view, projection_bounds)
        ix = np.rint(center_x).astype(np.int32)
        iy = np.rint(center_y).astype(np.int32)
        inside = (
            (ix >= 0)
            & (iy >= 0)
            & (ix < view.image_bgr.shape[1])
            & (iy < view.image_bgr.shape[0])
        )
        safe_ix = np.clip(ix, 0, view.image_bgr.shape[1] - 1)
        safe_iy = np.clip(iy, 0, view.image_bgr.shape[0] - 1)
        mask_valid = view.safe_mask[safe_iy, safe_ix] > 0
        facing = face_normals @ view.direction
        projected = np.stack((x[triangles], y[triangles]), axis=2)
        first = projected[:, 1] - projected[:, 0]
        second = projected[:, 2] - projected[:, 0]
        area = np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) * 0.5
        resolution_score = np.clip(np.sqrt(area) / 32.0, 0.0, 1.0)
        boundary_score = np.clip(
            view.boundary_distance[safe_iy, safe_ix] / max(boundary_scale, 1.0), 0.0, 1.0
        )
        confidence_score = view.confidence[safe_iy, safe_ix]
        depth_height, depth_width = view.depth_map.shape
        depth_x = np.clip(
            np.rint(center_x * (depth_width - 1) / max(view.image_bgr.shape[1] - 1, 1)).astype(np.int32),
            0,
            depth_width - 1,
        )
        depth_y = np.clip(
            np.rint(center_y * (depth_height - 1) / max(view.image_bgr.shape[0] - 1, 1)).astype(np.int32),
            0,
            depth_height - 1,
        )
        visible_depth = view.depth_map[depth_y, depth_x]
        visible = center_depth >= visible_depth - depth_tolerance
        valid = inside & mask_valid & visible & (facing >= min_facing)
        combined = (
            weights["facing"] * np.clip(facing, 0.0, 1.0)
            + weights["resolution"] * resolution_score
            + weights["boundary"] * boundary_score
            + weights["sharpness"] * view.sharpness
            + weights["confidence"] * confidence_score
        )
        scores[valid, view_index] = combined[valid]

    labels = np.argmax(scores, axis=1).astype(np.int16)
    labels[~np.isfinite(scores).any(axis=1)] = -1
    continuity_weight = float(scoring.get("continuity_weight", 0.12))
    smoothing_iterations = int(scoring.get("smoothing_iterations", 2))
    face_offset = 0
    for part, local_adjacency in zip(parts, adjacency):
        local_scores = scores[face_offset : face_offset + len(part.faces)]
        local_labels = labels[face_offset : face_offset + len(part.faces)].copy()
        for _ in range(max(0, smoothing_iterations)):
            next_labels = local_labels.copy()
            for face_index, neighbors in enumerate(local_adjacency):
                finite = np.isfinite(local_scores[face_index])
                if not np.any(finite) or not neighbors:
                    continue
                candidate = local_scores[face_index].astype(np.float64, copy=True)
                for view_index in np.flatnonzero(finite):
                    matches = sum(local_labels[neighbor] == view_index for neighbor in neighbors)
                    candidate[view_index] += continuity_weight * matches / len(neighbors)
                next_labels[face_index] = int(np.argmax(candidate))
            local_labels = next_labels
        labels[face_offset : face_offset + len(part.faces)] = local_labels
        face_offset += len(part.faces)
    return scores, labels, bounds


def rasterize_atlas_geometry(
    parts: Sequence[TexturedMeshPart],
    base_parts: Sequence[BaseMeshPart],
    atlas_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize model position/normal/face provenance into UV atlas space."""

    positions = np.zeros((atlas_size, atlas_size, 3), dtype=np.float32)
    normals = np.zeros_like(positions)
    face_map = np.full((atlas_size, atlas_size), -1, dtype=np.int32)
    part_map = np.full((atlas_size, atlas_size), -1, dtype=np.int16)
    surface = np.zeros((atlas_size, atlas_size), dtype=np.uint8)
    for part, base in zip(parts, base_parts):
        for local_face_index, face in enumerate(part.faces):
            vertices = part.vertices[face].astype(np.float64)
            vertex_normals = part.normals[face].astype(np.float64)
            uv = part.uvs[face].astype(np.float64)
            triangle = np.column_stack(
                (uv[:, 0] * (atlas_size - 1), uv[:, 1] * (atlas_size - 1))
            )
            min_x = max(0, int(math.floor(float(triangle[:, 0].min()))) - 1)
            max_x = min(atlas_size - 1, int(math.ceil(float(triangle[:, 0].max()))) + 1)
            min_y = max(0, int(math.floor(float(triangle[:, 1].min()))) - 1)
            max_y = min(atlas_size - 1, int(math.ceil(float(triangle[:, 1].max()))) + 1)
            if min_x > max_x or min_y > max_y:
                continue
            x0, y0 = triangle[0]
            x1, y1 = triangle[1]
            x2, y2 = triangle[2]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(float(denominator)) < 1e-10:
                raise ValueError("atlas rasterization encountered a degenerate UV triangle")
            yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
            sample_x = xx.astype(np.float64) + 0.5
            sample_y = yy.astype(np.float64) + 0.5
            weight0 = ((y1 - y2) * (sample_x - x2) + (x2 - x1) * (sample_y - y2)) / denominator
            weight1 = ((y2 - y0) * (sample_x - x2) + (x0 - x2) * (sample_y - y2)) / denominator
            weight2 = 1.0 - weight0 - weight1
            inside = (weight0 >= -1e-5) & (weight1 >= -1e-5) & (weight2 >= -1e-5)
            if not np.any(inside):
                continue
            interpolated_position = (
                weight0[..., None] * vertices[0]
                + weight1[..., None] * vertices[1]
                + weight2[..., None] * vertices[2]
            )
            interpolated_normal = (
                weight0[..., None] * vertex_normals[0]
                + weight1[..., None] * vertex_normals[1]
                + weight2[..., None] * vertex_normals[2]
            )
            normal_length = np.maximum(
                np.linalg.norm(interpolated_normal, axis=2, keepdims=True), ARRAY_EPSILON
            )
            interpolated_normal /= normal_length
            region_positions = positions[min_y : max_y + 1, min_x : max_x + 1]
            region_normals = normals[min_y : max_y + 1, min_x : max_x + 1]
            region_faces = face_map[min_y : max_y + 1, min_x : max_x + 1]
            region_parts = part_map[min_y : max_y + 1, min_x : max_x + 1]
            region_surface = surface[min_y : max_y + 1, min_x : max_x + 1]
            region_positions[inside] = interpolated_position[inside]
            region_normals[inside] = interpolated_normal[inside]
            region_faces[inside] = base.face_offset + local_face_index
            region_parts[inside] = base.index
            region_surface[inside] = 255
    return positions, normals, face_map, part_map, surface


def _fallback_atlas(
    base_parts: Sequence[BaseMeshPart],
    tiles: Sequence[tuple[int, int, int, int]],
    part_map: np.ndarray,
    surface_mask: np.ndarray,
    measured_fabric_bgr: np.ndarray,
) -> np.ndarray:
    height, width = part_map.shape
    fallback = np.empty((height, width, 3), dtype=np.float32)
    fallback[:] = measured_fabric_bgr
    for part, tile in zip(base_parts, tiles):
        if part.fallback_rgb is None:
            color_bgr = measured_fabric_bgr
        else:
            color_bgr = np.asarray(part.fallback_rgb[::-1], dtype=np.float32)
        x, y, tile_width, tile_height = tile
        # Fill the complete chart rectangle, including its padding, with the
        # owning part's fallback. This prevents filtering from pulling body
        # pink into light fabric or embroidery charts.
        fallback[y : y + tile_height, x : x + tile_width] = color_bgr
        fallback[part_map == part.index] = color_bgr
    return fallback


def _remap_view_values(
    source: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    nearest: bool = False,
) -> np.ndarray:
    return cv2.remap(
        source,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def project_source_atlas(
    position_map: np.ndarray,
    normal_map: np.ndarray,
    face_map: np.ndarray,
    surface_mask: np.ndarray,
    fallback_bgr: np.ndarray,
    views: Sequence[PreparedView],
    face_labels: np.ndarray,
    projection_bounds: Sequence[tuple[float, float, float, float]],
    scoring: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Project reviewed source pixels and select the strongest valid evidence."""

    height, width = surface_mask.shape
    best_score = np.full((height, width), -np.inf, dtype=np.float32)
    best_color = fallback_bgr.astype(np.float32, copy=True)
    best_view = np.full((height, width), -1, dtype=np.int16)
    surface = surface_mask > 0
    preferred = np.full_like(best_view, -1)
    face_pixels = face_map >= 0
    preferred[face_pixels] = face_labels[face_map[face_pixels]]
    boundary_scale = float(scoring.get("boundary_distance_pixels", 32.0))
    min_facing = float(scoring.get("minimum_facing", 0.12))
    depth_tolerance = float(scoring.get("depth_tolerance", 0.035))
    preference_weight = float(scoring.get("face_preference_weight", 0.22))
    selection_counts: dict[str, int] = {}

    flat_positions = position_map.reshape(-1, 3)
    for view_index, view in enumerate(views):
        map_x_flat, map_y_flat, depth_flat = _project_points(
            flat_positions, view, projection_bounds[view_index]
        )
        map_x = map_x_flat.reshape(height, width)
        map_y = map_y_flat.reshape(height, width)
        depth = depth_flat.reshape(height, width)
        safe = _remap_view_values(view.safe_mask, map_x, map_y, nearest=True) > 0
        boundary = _remap_view_values(view.boundary_distance, map_x, map_y)
        confidence = _remap_view_values(view.confidence, map_x, map_y)
        sampled_color = _remap_view_values(view.image_bgr, map_x, map_y).astype(np.float32)
        sampled_color *= view.exposure_gain_bgr.reshape(1, 1, 3)
        np.clip(sampled_color, 0.0, 255.0, out=sampled_color)
        facing = np.sum(normal_map * view.direction.reshape(1, 1, 3), axis=2)

        if view.depth_map is None:
            raise RuntimeError("view depth map was not prepared")
        depth_height, depth_width = view.depth_map.shape
        depth_map_x = map_x * (depth_width - 1) / max(view.image_bgr.shape[1] - 1, 1)
        depth_map_y = map_y * (depth_height - 1) / max(view.image_bgr.shape[0] - 1, 1)
        visible_depth = _remap_view_values(view.depth_map, depth_map_x, depth_map_y)
        visible = depth >= visible_depth - depth_tolerance
        inside = (
            (map_x >= 0.0)
            & (map_y >= 0.0)
            & (map_x <= view.image_bgr.shape[1] - 1)
            & (map_y <= view.image_bgr.shape[0] - 1)
        )
        valid = surface & safe & inside & visible & (facing >= min_facing)
        score = (
            0.45 * np.clip(facing, 0.0, 1.0)
            + 0.22 * np.clip(boundary / max(boundary_scale, 1.0), 0.0, 1.0)
            + 0.16 * confidence
            + 0.17 * view.sharpness
            + preference_weight * (preferred == view_index)
        )
        update = valid & (score > best_score)
        best_score[update] = score[update]
        best_color[update] = sampled_color[update]
        best_view[update] = view_index
        selection_counts[view.name] = int(np.count_nonzero(update))
    observed = (best_view >= 0) & surface
    return best_color, observed.astype(np.uint8) * 255, best_view, selection_counts


def harmonize_projected_atlas(
    projected_bgr: np.ndarray,
    fallback_bgr: np.ndarray,
    observed_mask: np.ndarray,
    selected_view: np.ndarray,
    surface_mask: np.ndarray,
    part_map: np.ndarray,
    parts: Sequence[BaseMeshPart],
    materials: Mapping[str, Any],
    *,
    feather_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reject implausible colors and feather source/view boundaries.

    This is deliberately a conservative operation: pixels far from the
    reviewed material's measured fallback are marked unsupported instead of
    allowing a hand, wall, or deep cast shadow into the final atlas.
    """

    result = projected_bgr.astype(np.float32, copy=True)
    observed = observed_mask > 0
    surface = surface_mask > 0
    rejected = np.zeros(observed.shape, dtype=bool)
    strength_map = np.ones(observed.shape, dtype=np.float32)
    for part in parts:
        pixels = (part_map == part.index) & observed
        if not np.any(pixels):
            continue
        material = materials[part.material_class]
        maximum_distance = float(material.get("source_color_max_distance", 88.0))
        strength = float(material.get("source_color_strength", 0.72))
        if maximum_distance <= 0.0 or not 0.0 <= strength <= 1.0:
            raise ValueError(f"invalid source color policy for {part.material_class}")
        difference = np.linalg.norm(result[pixels] - fallback_bgr[pixels], axis=1)
        local_rejected = difference > maximum_distance
        pixel_indices = np.flatnonzero(pixels)
        rejected.flat[pixel_indices[local_rejected]] = True
        strength_map[pixels] = strength
    observed[rejected] = False
    selected = selected_view.copy()
    selected[rejected] = -1
    result[rejected] = fallback_bgr[rejected]

    alpha = np.zeros(observed.shape, dtype=np.float32)
    for view_index in np.unique(selected[selected >= 0]):
        view_pixels = np.where((selected == view_index) & observed, 255, 0).astype(np.uint8)
        distance = cv2.distanceTransform(view_pixels, cv2.DIST_L2, 5)
        view_alpha = np.clip(distance / max(float(feather_pixels), 1.0), 0.0, 1.0)
        alpha[selected == view_index] = view_alpha[selected == view_index]
    alpha *= strength_map
    result = fallback_bgr + alpha[:, :, None] * (result - fallback_bgr)
    result[~surface] = fallback_bgr[~surface]
    metrics = {
        "source_pixels_rejected_as_color_outliers": int(np.count_nonzero(rejected)),
        "source_pixels_after_rejection": int(np.count_nonzero(observed)),
        "view_boundary_feather_pixels": int(feather_pixels),
    }
    return result, observed.astype(np.uint8) * 255, selected, metrics


def _detail_pattern(
    shape: tuple[int, int],
    detail_tiles: Sequence[np.ndarray],
    fabric: Mapping[str, Any],
) -> np.ndarray:
    if not detail_tiles:
        raise ValueError("at least one source detail tile is required")
    combined = np.mean(np.stack(detail_tiles).astype(np.float32), axis=0)
    height, width = shape
    repeats_y = math.ceil(height / combined.shape[0])
    repeats_x = math.ceil(width / combined.shape[1])
    source_pattern = np.tile(combined, (repeats_y, repeats_x))[:height, :width]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    periods = fabric.get("fiber_periods_px", [[7.0, 31.0], [19.0, -11.0]])
    procedural = np.zeros((height, width), dtype=np.float32)
    for index, pair in enumerate(periods):
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("fabric.fiber_periods_px entries must be [x_period, y_period]")
        x_period = float(pair[0])
        y_period = float(pair[1])
        if abs(x_period) < 1.0 or abs(y_period) < 1.0:
            raise ValueError("fabric fiber periods must have magnitude at least one pixel")
        phase = index * 0.731
        procedural += np.sin(2.0 * math.pi * (xx / x_period + yy / y_period) + phase)
    procedural /= max(len(periods), 1)
    source_weight = float(fabric.get("source_detail_weight", 0.72))
    procedural_weight = float(fabric.get("fixed_pattern_weight", 0.28))
    pattern = source_weight * source_pattern + procedural_weight * procedural
    standard_deviation = max(float(np.std(pattern)), 1e-5)
    return np.clip(pattern / standard_deviation, -3.0, 3.0)


def build_material_maps(
    base_color_bgr: np.ndarray,
    observed_mask: np.ndarray,
    selected_view: np.ndarray,
    surface_mask: np.ndarray,
    part_map: np.ndarray,
    base_parts: Sequence[BaseMeshPart],
    views: Sequence[PreparedView],
    detail_tiles: Sequence[np.ndarray],
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Add deterministic fabric relief and pack glTF normal/MR maps."""

    fabric = config.get("fabric", {})
    materials = config["materials"]
    surface = surface_mask > 0
    observed = observed_mask > 0
    pattern = _detail_pattern(surface_mask.shape, detail_tiles, fabric)

    final_color = base_color_bgr.astype(np.float32, copy=True)
    inferred_fabric = np.zeros(surface_mask.shape, dtype=bool)
    for part in base_parts:
        material = materials[part.material_class]
        if bool(material.get("fabric_detail", False)):
            inferred_fabric |= (part_map == part.index) & ~observed
    color_variation = float(fabric.get("fallback_color_variation", 0.014))
    final_color[inferred_fabric] *= (
        1.0 + color_variation * pattern[inferred_fabric, None]
    )
    np.clip(final_color, 0.0, 255.0, out=final_color)

    gray = cv2.cvtColor(final_color.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    highpass_sigma = float(fabric.get("atlas_highpass_sigma", 3.2))
    highpass = gray - cv2.GaussianBlur(gray, (0, 0), highpass_sigma)
    highpass = np.clip(highpass / 24.0, -1.0, 1.0)
    hsv = cv2.cvtColor(final_color.astype(np.uint8), cv2.COLOR_BGR2HSV)
    luminance = gray
    source_thread = observed & (
        (luminance < float(fabric.get("dark_thread_luma", 105.0)))
        | ((hsv[:, :, 1] > int(fabric.get("colored_thread_saturation", 72))) & (hsv[:, :, 0] < 178))
    )
    front_indices = {
        view.index for view in views if view.direction_name == "front"
    }
    front_source = np.isin(selected_view, list(front_indices))
    source_thread &= front_source

    embroidery_parts = np.zeros(surface_mask.shape, dtype=bool)
    fabric_parts = np.zeros(surface_mask.shape, dtype=bool)
    for part in base_parts:
        material = materials[part.material_class]
        if material.get("kind") == "embroidery":
            embroidery_parts |= part_map == part.index
        if bool(material.get("fabric_detail", False)):
            fabric_parts |= part_map == part.index
    embroidery = (embroidery_parts | source_thread) & surface

    fabric_amplitude = float(fabric.get("fabric_height", 0.032))
    source_amplitude = float(fabric.get("source_height", 0.045))
    embroidery_amplitude = float(fabric.get("embroidery_height", 0.075))
    height_field = np.zeros(surface_mask.shape, dtype=np.float32)
    height_field[fabric_parts] += fabric_amplitude * pattern[fabric_parts]
    height_field[observed & surface] += source_amplitude * highpass[observed & surface]
    height_field[embroidery] += embroidery_amplitude * np.clip(
        np.abs(highpass[embroidery]) + 0.25, 0.0, 1.25
    )
    # Prevent atlas tile boundaries from becoming false material ridges.
    interior = cv2.erode(
        surface_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    ) > 0
    height_field[~interior] = 0.0
    gradient_x = cv2.Sobel(height_field, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gradient_y = cv2.Sobel(height_field, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    normal_scale = float(fabric.get("normal_scale", 5.0))
    tangent_x = -gradient_x * normal_scale
    tangent_y = -gradient_y * normal_scale
    tangent_z = np.ones_like(tangent_x)
    normal_length = np.sqrt(tangent_x * tangent_x + tangent_y * tangent_y + tangent_z * tangent_z)
    normal_rgb = np.stack(
        (
            tangent_x / normal_length,
            tangent_y / normal_length,
            tangent_z / normal_length,
        ),
        axis=2,
    )
    normal_rgb = np.rint((normal_rgb * 0.5 + 0.5) * 255.0).astype(np.uint8)
    normal_rgb[~surface] = (128, 128, 255)
    normal_bgr = normal_rgb[:, :, ::-1].copy()

    roughness = np.full(surface_mask.shape, 0.94, dtype=np.float32)
    for part in base_parts:
        material = materials[part.material_class]
        base_roughness = float(material.get("roughness", 0.94))
        minimum = float(material.get("roughness_min", base_roughness))
        maximum = float(material.get("roughness_max", base_roughness))
        if not 0.0 <= minimum <= base_roughness <= maximum <= 1.0:
            raise ValueError(f"invalid roughness range for {part.material_class}")
        part_pixels = part_map == part.index
        variation = float(fabric.get("roughness_source_modulation", 0.025)) * highpass
        roughness[part_pixels] = np.clip(
            base_roughness + variation[part_pixels], minimum, maximum
        )
    roughness[~surface] = 0.94
    roughness_byte = np.rint(np.clip(roughness, 0.0, 1.0) * 255.0).astype(np.uint8)
    # OpenCV is BGR.  Desired glTF RGB is (unused=255, roughness=G, metallic=0).
    metallic_roughness_bgr = np.empty((*surface_mask.shape, 3), dtype=np.uint8)
    metallic_roughness_bgr[:, :, 0] = 0
    metallic_roughness_bgr[:, :, 1] = roughness_byte
    metallic_roughness_bgr[:, :, 2] = 255

    xy = normal_rgb[:, :, :2].astype(np.float32) / 255.0 * 2.0 - 1.0
    metrics = {
        "embroidery_pixels": int(np.count_nonzero(embroidery)),
        "fabric_detail_pixels": int(np.count_nonzero(fabric_parts)),
        "normal_xy_max": round(float(np.linalg.norm(xy[surface], axis=1).max()), 6),
        "roughness_min": round(float(roughness[surface].min()), 6),
        "roughness_max": round(float(roughness[surface].max()), 6),
        "metallic_max": 0.0,
    }
    return final_color.astype(np.uint8), normal_bgr, metallic_roughness_bgr, metrics


def _encode_image(extension: str, image: np.ndarray, parameters: list[int]) -> bytes:
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok:
        raise RuntimeError(f"OpenCV could not encode {extension} texture")
    return encoded.tobytes()


def _write_atlas_qa(
    path: Path,
    base_bgr: np.ndarray,
    normal_bgr: np.ndarray,
    mr_bgr: np.ndarray,
    observed_mask: np.ndarray,
) -> None:
    card_size = 480

    def card(image: np.ndarray, label: str) -> np.ndarray:
        display = cv2.resize(image, (card_size, card_size), interpolation=cv2.INTER_AREA)
        result = np.full((card_size + 44, card_size, 3), 24, dtype=np.uint8)
        result[44:] = display
        cv2.putText(
            result,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        return result

    observed_color = cv2.cvtColor(observed_mask, cv2.COLOR_GRAY2BGR)
    roughness_display = cv2.cvtColor(mr_bgr[:, :, 1], cv2.COLOR_GRAY2BGR)
    top = np.hstack((card(base_bgr, "base-color atlas"), card(normal_bgr, "tangent normal")))
    bottom = np.hstack(
        (card(roughness_display, "roughness (G); metallic B=0"), card(observed_color, "white=observed / black=inferred"))
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.vstack((top, bottom))):
        raise RuntimeError(f"could not write atlas QA image: {path}")


def render_textured_preview(
    parts: Sequence[TexturedMeshPart],
    atlas_bgr: np.ndarray,
    *,
    yaw_degrees: float,
    elevation_degrees: float,
    size: int = 512,
) -> np.ndarray:
    """Render a deterministic CPU orthographic preview with per-pixel UVs."""

    direction, right, up = camera_basis(yaw_degrees, elevation_degrees, 0.0)
    all_vertices = np.vstack([part.vertices for part in parts]).astype(np.float64)
    horizontal = all_vertices @ right
    vertical = all_vertices @ up
    min_horizontal, max_horizontal = float(horizontal.min()), float(horizontal.max())
    min_vertical, max_vertical = float(vertical.min()), float(vertical.max())
    span = max(max_horizontal - min_horizontal, max_vertical - min_vertical, ARRAY_EPSILON)
    center_horizontal = 0.5 * (min_horizontal + max_horizontal)
    center_vertical = 0.5 * (min_vertical + max_vertical)
    scale = (size - 32) / span
    depth_buffer = np.full((size, size), -np.inf, dtype=np.float32)
    uv_buffer = np.zeros((size, size, 2), dtype=np.float32)
    normal_buffer = np.zeros((size, size, 3), dtype=np.float32)
    covered = np.zeros((size, size), dtype=bool)

    for part in parts:
        x = (part.vertices @ right - center_horizontal) * scale + 0.5 * (size - 1)
        y = (center_vertical - part.vertices @ up) * scale + 0.5 * (size - 1)
        depth = part.vertices @ direction
        for face in part.faces:
            triangle = np.column_stack((x[face], y[face])).astype(np.float64)
            min_x = max(0, int(math.floor(float(triangle[:, 0].min()))))
            max_x = min(size - 1, int(math.ceil(float(triangle[:, 0].max()))))
            min_y = max(0, int(math.floor(float(triangle[:, 1].min()))))
            max_y = min(size - 1, int(math.ceil(float(triangle[:, 1].max()))))
            if min_x > max_x or min_y > max_y:
                continue
            x0, y0 = triangle[0]
            x1, y1 = triangle[1]
            x2, y2 = triangle[2]
            denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
            if abs(float(denominator)) < 1e-10:
                continue
            yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
            sample_x = xx.astype(np.float64) + 0.5
            sample_y = yy.astype(np.float64) + 0.5
            weight0 = ((y1 - y2) * (sample_x - x2) + (x2 - x1) * (sample_y - y2)) / denominator
            weight1 = ((y2 - y0) * (sample_x - x2) + (x0 - x2) * (sample_y - y2)) / denominator
            weight2 = 1.0 - weight0 - weight1
            inside = (weight0 >= -1e-5) & (weight1 >= -1e-5) & (weight2 >= -1e-5)
            interpolated_depth = weight0 * depth[face[0]] + weight1 * depth[face[1]] + weight2 * depth[face[2]]
            region_depth = depth_buffer[min_y : max_y + 1, min_x : max_x + 1]
            update = inside & (interpolated_depth > region_depth)
            if not np.any(update):
                continue
            interpolated_uv = (
                weight0[..., None] * part.uvs[face[0]]
                + weight1[..., None] * part.uvs[face[1]]
                + weight2[..., None] * part.uvs[face[2]]
            )
            interpolated_normal = (
                weight0[..., None] * part.normals[face[0]]
                + weight1[..., None] * part.normals[face[1]]
                + weight2[..., None] * part.normals[face[2]]
            )
            interpolated_normal /= np.maximum(
                np.linalg.norm(interpolated_normal, axis=2, keepdims=True), ARRAY_EPSILON
            )
            region_depth[update] = interpolated_depth[update]
            uv_buffer[min_y : max_y + 1, min_x : max_x + 1][update] = interpolated_uv[update]
            normal_buffer[min_y : max_y + 1, min_x : max_x + 1][update] = interpolated_normal[update]
            covered[min_y : max_y + 1, min_x : max_x + 1][update] = True

    map_x = uv_buffer[:, :, 0] * (atlas_bgr.shape[1] - 1)
    map_y = uv_buffer[:, :, 1] * (atlas_bgr.shape[0] - 1)
    sampled = cv2.remap(
        atlas_bgr,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(236, 238, 240),
    ).astype(np.float32)
    light = np.array([0.35, 0.55, 0.76], dtype=np.float32)
    light /= np.linalg.norm(light)
    intensity = 0.72 + 0.28 * np.clip(np.sum(normal_buffer * light, axis=2), 0.0, 1.0)
    sampled *= intensity[:, :, None]
    background = np.full((size, size, 3), (240, 237, 232), dtype=np.uint8)
    background[covered] = np.clip(sampled[covered], 0.0, 255.0).astype(np.uint8)
    return background


def _write_model_qa(
    path: Path,
    parts: Sequence[TexturedMeshPart],
    atlas_bgr: np.ndarray,
) -> None:
    views = [
        ("front 0", 0.0, 8.0),
        ("front-right 45", 45.0, 8.0),
        ("right 90", 90.0, 8.0),
        ("rear-right 135", 135.0, 8.0),
        ("rear 180", 180.0, 8.0),
        ("rear-left 225", 225.0, 8.0),
        ("left 270", 270.0, 8.0),
        ("front-left 315", 315.0, 8.0),
    ]
    cards: list[np.ndarray] = []
    for label, yaw, elevation in views:
        image = render_textured_preview(
            parts,
            atlas_bgr,
            yaw_degrees=yaw,
            elevation_degrees=elevation,
            size=320,
        )
        card = np.full((356, 320, 3), 24, dtype=np.uint8)
        card[36:] = image
        cv2.putText(
            card,
            label,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        cards.append(card)
    rows = [np.hstack(cards[:4]), np.hstack(cards[4:])]
    if not cv2.imwrite(str(path), np.vstack(rows)):
        raise RuntimeError(f"could not write model QA image: {path}")


def _write_source_qa(path: Path, views: Sequence[PreparedView]) -> None:
    card_width, card_height = 420, 280
    cards: list[np.ndarray] = []
    for view in views:
        overlay = view.image_bgr.copy()
        green = np.zeros_like(overlay)
        green[:, :, 1] = 255
        alpha = (view.safe_mask.astype(np.float32) / 255.0 * 0.36)[:, :, None]
        overlay = np.clip(overlay * (1.0 - alpha) + green * alpha, 0, 255).astype(np.uint8)
        resized = cv2.resize(overlay, (card_width, card_height), interpolation=cv2.INTER_AREA)
        card = np.full((card_height + 42, card_width, 3), 20, dtype=np.uint8)
        card[42:] = resized
        cv2.putText(
            card,
            f"{view.direction_name}: frame {view.frame_index} (green=reviewed safe)",
            (8, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
        cards.append(card)
    columns = 3
    while len(cards) % columns:
        cards.append(np.full_like(cards[0], 20))
    rows = [np.hstack(cards[offset : offset + columns]) for offset in range(0, len(cards), columns)]
    if not cv2.imwrite(str(path), np.vstack(rows)):
        raise RuntimeError(f"could not write source QA image: {path}")


def _format_vec_array(values: np.ndarray, dimensions: int) -> str:
    array = np.asarray(values).reshape(-1, dimensions)
    return ",\n            ".join(
        "(" + ", ".join(f"{float(value):.9g}" for value in row) + ")"
        for row in array
    )


def _format_int_array(values: np.ndarray) -> str:
    return ", ".join(str(int(value)) for value in np.asarray(values).reshape(-1))


def _gltf_uvs_to_usd(uvs: np.ndarray) -> np.ndarray:
    """Convert glTF's upper-left texture origin to USD's lower-left origin."""

    result = np.asarray(uvs, dtype=np.float32).copy()
    result[:, 1] = 1.0 - result[:, 1]
    return result


def author_textured_usda(
    path: Path,
    parts: Sequence[TexturedMeshPart],
    *,
    base_color_name: str,
    normal_name: str,
    metallic_roughness_name: str,
) -> None:
    """Author a RealityKit-oriented USD Preview Surface equivalent."""

    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = [
        '#usda 1.0\n(\n    defaultPrim = "Object"\n    metersPerUnit = 1\n    upAxis = "Y"\n)\n\n',
        'def Xform "Object" (\n    kind = "component"\n)\n{\n',
        '    def Scope "Looks"\n    {\n',
    ]
    for index, part in enumerate(parts):
        material_name = f"PartMaterial_{index}"
        chunks.append(
            f'''        def Material "{material_name}"
        {{
            token outputs:surface.connect = </Object/Looks/{material_name}/Preview.outputs:surface>

            def Shader "Preview"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Object/Looks/{material_name}/BaseColor.outputs:rgb>
                float inputs:metallic = 0
                normal3f inputs:normal.connect = </Object/Looks/{material_name}/Normal.outputs:rgb>
                float inputs:roughness.connect = </Object/Looks/{material_name}/MetallicRoughness.outputs:g>
                token outputs:surface
            }}
            def Shader "PrimvarReader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
            def Shader "BaseColor"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{base_color_name}@
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = </Object/Looks/{material_name}/PrimvarReader.outputs:result>
                float3 outputs:rgb
            }}
            def Shader "Normal"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{normal_name}@
                float4 inputs:bias = (-1, -1, -1, 0)
                float4 inputs:scale = (2, 2, 2, 1)
                token inputs:sourceColorSpace = "raw"
                float2 inputs:st.connect = </Object/Looks/{material_name}/PrimvarReader.outputs:result>
                float3 outputs:rgb
            }}
            def Shader "MetallicRoughness"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{metallic_roughness_name}@
                token inputs:sourceColorSpace = "raw"
                float2 inputs:st.connect = </Object/Looks/{material_name}/PrimvarReader.outputs:result>
                float outputs:g
            }}
        }}
'''
        )
    chunks.append("    }\n\n")
    for index, part in enumerate(parts):
        counts = np.full(len(part.faces), 3, dtype=np.int32)
        usd_uvs = _gltf_uvs_to_usd(part.uvs)
        chunks.append(
            f'''    def Mesh "Part_{index:02d}" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        int[] faceVertexCounts = [{_format_int_array(counts)}]
        int[] faceVertexIndices = [{_format_int_array(part.faces)}]
        normal3f[] normals = [
            {_format_vec_array(part.normals, 3)}
        ] (
            interpolation = "vertex"
        )
        point3f[] points = [
            {_format_vec_array(part.vertices, 3)}
        ]
        texCoord2f[] primvars:st = [
            {_format_vec_array(usd_uvs, 2)}
        ] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
        rel material:binding = </Object/Looks/PartMaterial_{index}>
    }}

'''
        )
    chunks.append("}\n")
    path.write_text("".join(chunks), encoding="utf-8", newline="\n")


def _package_and_validate_usdz(usda_path: Path, texture_paths: Sequence[Path]) -> dict[str, Any]:
    usdz_path = usda_path.with_suffix(".usdz")
    usdzip = shutil.which("usdzip")
    usdchecker = shutil.which("usdchecker")
    result: dict[str, Any] = {
        "usda": str(usda_path),
        "usdz": None,
        "usdchecker": "not_available",
    }
    if not usdzip:
        return result
    if usdz_path.exists():
        usdz_path.unlink()
    command = [usdzip, "--arkitAsset", usda_path.name, usdz_path.name]
    package = subprocess.run(
        command,
        cwd=usda_path.parent,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    result["usdzip_output"] = package.stdout.strip()
    if package.returncode != 0 or not usdz_path.is_file():
        result["usdzip_returncode"] = package.returncode
        return result
    result["usdz"] = str(usdz_path)
    result["usdz_sha256"] = sha256_file(usdz_path)
    if usdchecker:
        check = subprocess.run(
            [usdchecker, "--arkit", str(usdz_path)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        result["usdchecker"] = "passed" if check.returncode == 0 else "failed"
        result["usdchecker_returncode"] = check.returncode
        result["usdchecker_output"] = check.stdout.strip()
    return result


def bake_textured_soft_parts(
    config_path: str | Path,
    output: str | Path,
    *,
    write_usdz: bool = True,
) -> dict[str, Any]:
    """Run the complete deterministic fitted-plush texture build."""

    config = load_texture_config(config_path)
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_parts, mesh_config = _load_mesh_parts(config)
    tiles = _validate_tiles(config, len(base_parts))
    atlas_size = int(config["atlas"]["size"])
    padding = int(config["atlas"].get("padding", 4))
    textured_parts = build_textured_parts(base_parts, tiles, atlas_size, padding)

    frame_indices = [int(item["frame_index"]) for item in config["views"]]
    decoded, video_metadata = decode_video_frames(config["source_video"], frame_indices)
    frames_by_name = {
        item["name"]: decoded[int(item["frame_index"])] for item in config["views"]
    }
    target_fabric_bgr, detail_tiles, fabric_patch_records = _fabric_patch_statistics(
        frames_by_name, config
    )
    views = _prepare_views(config, decoded, target_fabric_bgr)
    face_scores, face_labels, projection_bounds = score_face_views(
        base_parts, views, config.get("view_scoring", {})
    )
    position_map, geometry_normal_map, face_map, part_map, surface_mask = rasterize_atlas_geometry(
        textured_parts, base_parts, atlas_size
    )
    fallback = _fallback_atlas(base_parts, tiles, part_map, surface_mask, target_fabric_bgr)
    appearance_safe_mode = bool(config.get("appearance_safe_mode", False))
    if appearance_safe_mode:
        projected_color = fallback.copy()
        observed_mask = np.zeros_like(surface_mask, dtype=np.uint8)
        selected_view = np.full(surface_mask.shape, -1, dtype=np.int16)
        harmonization_metrics = {
            "appearance_safe_mode": True,
            "raw_source_projection_suppressed": True,
            "source_pixels_rejected_as_color_outliers": 0,
            "source_pixels_after_rejection": 0,
            "view_boundary_feather_pixels": 0,
        }
    else:
        projected_color, observed_mask, selected_view, _ = project_source_atlas(
            position_map,
            geometry_normal_map,
            face_map,
            surface_mask,
            fallback,
            views,
            face_labels,
            projection_bounds,
            config.get("view_scoring", {}),
        )
        projected_color, observed_mask, selected_view, harmonization_metrics = harmonize_projected_atlas(
            projected_color,
            fallback,
            observed_mask,
            selected_view,
            surface_mask,
            part_map,
            base_parts,
            config["materials"],
            feather_pixels=int(config.get("fabric", {}).get("view_boundary_feather_pixels", 14)),
        )
    base_color_bgr, normal_bgr, mr_bgr, material_metrics = build_material_maps(
        projected_color,
        observed_mask,
        selected_view,
        surface_mask,
        part_map,
        base_parts,
        views,
        detail_tiles,
        config,
    )
    detail_map_size = int(config["atlas"].get("detail_map_size", atlas_size))
    encoded_normal_bgr = normal_bgr
    encoded_mr_bgr = mr_bgr
    if detail_map_size != atlas_size:
        encoded_normal_bgr = cv2.resize(
            normal_bgr,
            (detail_map_size, detail_map_size),
            interpolation=cv2.INTER_AREA,
        )
        # Renormalize averaged tangent vectors before encoding.
        normal_rgb_float = encoded_normal_bgr[:, :, ::-1].astype(np.float32) / 255.0 * 2.0 - 1.0
        normal_rgb_float /= np.maximum(
            np.linalg.norm(normal_rgb_float, axis=2, keepdims=True), ARRAY_EPSILON
        )
        encoded_normal_bgr = np.rint(
            (normal_rgb_float[:, :, ::-1] * 0.5 + 0.5) * 255.0
        ).astype(np.uint8)
        encoded_mr_bgr = cv2.resize(
            mr_bgr,
            (detail_map_size, detail_map_size),
            interpolation=cv2.INTER_AREA,
        )
        encoded_mr_bgr[:, :, 0] = 0
        encoded_mr_bgr[:, :, 2] = 255

    jpeg_quality = int(config["atlas"].get("base_color_jpeg_quality", 90))
    base_payload = _encode_image(
        ".jpg",
        base_color_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            jpeg_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE,
            1,
            cv2.IMWRITE_JPEG_PROGRESSIVE,
            0,
        ],
    )
    png_parameters = [cv2.IMWRITE_PNG_COMPRESSION, int(config["atlas"].get("png_compression", 9))]
    normal_payload = _encode_image(".png", encoded_normal_bgr, png_parameters)
    mr_payload = _encode_image(".png", encoded_mr_bgr, png_parameters)
    surface_pixels = int(np.count_nonzero(surface_mask))
    observed_pixels = int(np.count_nonzero(observed_mask))
    coverage = observed_pixels / max(surface_pixels, 1)
    final_view_counts = {
        view.name: int(np.count_nonzero(selected_view == view.index)) for view in views
    }
    face_view_counts = {
        view.name: int(np.count_nonzero(face_labels == view.index)) for view in views
    }
    build_extras = {
        "method": "deterministic_reviewed_source_projection",
        "sourceVideoSha256": config["source_video_sha256"],
        "meshConfigSha256": config["mesh_config_sha256"],
        "textureConfigSha256": config["config_sha256"],
        "selectedFrameIndices": frame_indices,
        "atlasSize": atlas_size,
        "observedCoverage": round(coverage, 8),
        "unsupportedPolicy": "measured_fabric_fallback",
        "geometry": "inferred_from_observed_front_side_proportions",
        "baseColorSha256": sha256_bytes(base_payload),
        "normalSha256": sha256_bytes(normal_payload),
        "metallicRoughnessSha256": sha256_bytes(mr_payload),
    }
    write_textured_glb(
        output_path,
        textured_parts,
        base_color=base_payload,
        base_color_mime="image/jpeg",
        normal_map=normal_payload,
        metallic_roughness=mr_payload,
        generator="local3d-soft-parts-texture-bake",
        extras=build_extras,
    )

    stem = output_path.stem
    base_path = output_path.parent / f"{stem}-basecolor.jpg"
    normal_path = output_path.parent / f"{stem}-normal.png"
    usd_normal_path = output_path.parent / f"{stem}-normal-usd.png"
    mr_path = output_path.parent / f"{stem}-metallic-roughness.png"
    base_path.write_bytes(base_payload)
    normal_path.write_bytes(normal_payload)
    mr_path.write_bytes(mr_payload)
    atlas_qa = output_path.parent / f"{stem}-atlas-qa.png"
    model_qa = output_path.parent / f"{stem}-model-qa.png"
    source_qa = output_path.parent / f"{stem}-source-qa.png"
    _write_atlas_qa(atlas_qa, base_color_bgr, normal_bgr, mr_bgr, observed_mask)
    _write_model_qa(model_qa, textured_parts, base_color_bgr)
    _write_source_qa(source_qa, views)

    usd_result: dict[str, Any] = {"status": "not_requested"}
    if write_usdz:
        # USD uses lower-left ST while glTF uses upper-left UV. The geometry V
        # conversion changes bitangent orientation, so invert the normal map's
        # green channel and retain the validator-standard decode scale/bias.
        usd_normal_bgr = encoded_normal_bgr.copy()
        usd_normal_bgr[:, :, 1] = 255 - usd_normal_bgr[:, :, 1]
        usd_normal_payload = _encode_image(".png", usd_normal_bgr, png_parameters)
        usd_normal_path.write_bytes(usd_normal_payload)
        usda_path = output_path.with_suffix(".usda")
        author_textured_usda(
            usda_path,
            textured_parts,
            base_color_name=base_path.name,
            normal_name=usd_normal_path.name,
            metallic_roughness_name=mr_path.name,
        )
        usd_result = _package_and_validate_usdz(
            usda_path, [base_path, usd_normal_path, mr_path]
        )

    per_part_coverage = []
    for part in base_parts:
        pixels = part_map == part.index
        count = int(np.count_nonzero(pixels))
        observed_count = int(np.count_nonzero(pixels & (observed_mask > 0)))
        per_part_coverage.append(
            {
                "part": part.index,
                "name": part.name,
                "material_class": part.material_class,
                "observed_fraction": round(observed_count / max(count, 1), 6),
            }
        )
    report = {
        "schema_version": "1.0",
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "output_bytes": output_path.stat().st_size,
        "source_video": config["source_video"],
        "source_video_sha256": config["source_video_sha256"],
        "mesh_config": config["mesh_config"],
        "mesh_config_sha256": config["mesh_config_sha256"],
        "texture_config": config["config_path"],
        "texture_config_sha256": config["config_sha256"],
        "video": video_metadata,
        "selected_views": [
            {
                "name": view.name,
                "direction": view.direction_name,
                "frame_index": view.frame_index,
                "decoded_frame_sha256": view.decoded_sha256,
                "safe_pixels": int(np.count_nonzero(view.safe_mask)),
                "selected_atlas_pixels": final_view_counts[view.name],
                "selected_faces": face_view_counts[view.name],
                "exposure_gain_bgr": [round(float(value), 6) for value in view.exposure_gain_bgr],
            }
            for view in views
        ],
        "fabric_patches": fabric_patch_records,
        "fallback_fabric_median_rgb": [int(round(value)) for value in target_fabric_bgr[::-1]],
        "atlas": {
            "size": [atlas_size, atlas_size],
            "detail_map_size": [detail_map_size, detail_map_size],
            "surface_pixels": surface_pixels,
            "observed_pixels": observed_pixels,
            "observed_fraction": round(coverage, 8),
            "inferred_fraction": round(1.0 - coverage, 8),
            "base_color_sha256": sha256_bytes(base_payload),
            "normal_sha256": sha256_bytes(normal_payload),
            "metallic_roughness_sha256": sha256_bytes(mr_payload),
            "base_color_bytes": len(base_payload),
            "normal_bytes": len(normal_payload),
            "metallic_roughness_bytes": len(mr_payload),
        },
        "material_metrics": material_metrics,
        "harmonization": harmonization_metrics,
        "mesh": {
            "parts": len(textured_parts),
            "triangles": int(sum(len(part.faces) for part in textured_parts)),
            "vertices_after_uv_seams": int(sum(len(part.vertices) for part in textured_parts)),
            "attributes": ["POSITION", "NORMAL", "TEXCOORD_0", "TANGENT"],
        },
        "per_part_coverage": per_part_coverage,
        "provenance": {
            "automatic_frame_selection": False,
            "unseeded_randomness": False,
            "generative_inpainting": False,
            "hands_or_background_allowed": False,
            "unsupported_surfaces": "measured median fabric or explicit part fallback",
        },
        "qa": {
            "atlas": str(atlas_qa),
            "model": str(model_qa),
            "source_masks": str(source_qa),
        },
        "usd": usd_result,
        "face_score_finite_fraction": round(float(np.isfinite(face_scores).any(axis=1).mean()), 8),
    }
    _json_dump(output_path.with_suffix(".json"), report)
    return report


def audit_video_directory(
    directory: str | Path,
    output: str | Path,
    *,
    texture_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Exercise deterministic exact-frame decoding for every video fixture."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError(f"video directory does not exist: {root}")
    records: list[dict[str, Any]] = []
    for video_path in sorted(root.glob("*.mov"), key=lambda item: item.name):
        source_hash = sha256_file(video_path)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            records.append(
                {"file": str(video_path), "sha256": source_hash, "decode_status": "open_failed"}
            )
            continue
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
        # Container frame counts are advisory. Some QuickTime files report one
        # more frame than their decoder can return, so determine the actual
        # sequentially decodable extent before choosing the first/middle/last
        # exact-frame probes.
        extent_capture = cv2.VideoCapture(str(video_path))
        decoded_frame_count = 0
        if extent_capture.isOpened():
            try:
                while True:
                    ok, _ = extent_capture.read()
                    if not ok:
                        break
                    decoded_frame_count += 1
            finally:
                extent_capture.release()
        sample_indices = sorted(
            {
                0,
                max(0, decoded_frame_count // 2),
                max(0, decoded_frame_count - 1),
            }
        )
        try:
            if decoded_frame_count == 0:
                raise ValueError("video opened but no frames could be decoded")
            frames, _ = decode_video_frames(video_path, sample_indices)
            samples = [
                {
                    "frame_index": index,
                    "decoded_sha256": sha256_bytes(frames[index].tobytes(order="C")),
                }
                for index in sample_indices
            ]
            status = "passed"
            error = None
        except ValueError as exc:
            samples = []
            status = "failed"
            error = str(exc)
        record: dict[str, Any] = {
            "file": str(video_path),
            "sha256": source_hash,
            "texture_source_match": source_hash == texture_source_sha256,
            "decode_status": status,
            "width": width,
            "height": height,
            "fps": round(fps, 9),
            "reported_frame_count": frame_count,
            "decoded_frame_count": decoded_frame_count,
            "sample_frames": samples,
        }
        if error:
            record["error"] = error
        records.append(record)
    report = {
        "schema_version": "1.0",
        "directory": str(root),
        "videos": records,
        "passed": sum(item["decode_status"] == "passed" for item in records),
        "failed": sum(item["decode_status"] != "passed" for item in records),
        "note": (
            "Exact-frame decoding is exercised for every fixture. Only the source-hash "
            "match is eligible for the fitted plush texture bake."
        ),
    }
    _json_dump(Path(output).resolve(), report)
    return report
