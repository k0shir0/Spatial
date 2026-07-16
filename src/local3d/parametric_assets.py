"""Safe, deterministic parametric assets from manually reviewed video frames.

The builder is deliberately small and boring: it rectifies two user-reviewed
image quads, wraps them onto either a phone-shaped rounded slab or a book-shaped
cuboid, and exports GLB plus USDA (and USDZ when Apple's command-line tools are
present).  It performs no segmentation, learned inference, network access,
Torch import, or GPU work.

Configuration schema (paths may be absolute or relative to the JSON file)::

    {
      "schema_version": 1,
      "asset_name": "Example phone",
      "kind": "phone",                         # phone | book
      "authoring_mode": "reviewed",            # reviewed | automatic
      "asset_kind": "phone",                   # semantic kind; defaults to kind
      "output_name": "example_phone",
      "front": {
        "image": "front.jpg",
        "quad_px": [[10, 10], [500, 12], [498, 900], [12, 898]],
        "rotate_quarter_turns": 0
      },
      "back": {"image": "back.jpg", "quad_px": [...]},
      "dimensions_mm": {
        "width": 71.5, "height": 146.7, "depth": 7.8,
        "corner_radius": 8.0, "bevel": 0.8
      },
      "texture_size": 1024,                    # longest edge; max 2048
      "output_rotation_deg": [0, 0, 0],        # XYZ Euler degrees
      "materials": {
        "body": {"color_rgb": [45, 47, 51], "metallic": 0.45,
                 "roughness": 0.28}
      },
      "phone": {
        "camera_bump": {
          "side": "back", "center_mm": [-20, 50],
          "size_mm": [30, 32], "protrusion_mm": 2.0,
          "corner_radius_mm": 5.0,
          "lenses": [
            {"center_mm": [-7, 7], "relative_to": "bump",
             "radius_mm": 5.5, "protrusion_mm": 1.5}
          ]
        }
      }
    }

Quad points are used exactly in top-left, top-right, bottom-right, bottom-left
order.  The program never guesses or adjusts them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import trimesh
from PIL import Image

from .usdz_pack import package_usdz


MAX_TEXTURE_SIZE = 2048
MIN_TEXTURE_SIZE = 128
MAX_SOURCE_PIXELS = 24_000_000
MAX_SOURCE_BYTES = 100 * 1024 * 1024
MAX_CORNER_SEGMENTS = 24
QA_SIZE = 480


@dataclass(frozen=True)
class Material:
    """A compact metallic/roughness material description."""

    key: str
    color_rgb: tuple[int, int, int]
    metallic: float
    roughness: float


@dataclass
class MeshPart:
    """One mesh/material section in the exported scene."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    material: Material
    uv: np.ndarray | None = None
    texture_key: str | None = None
    texture_bgr: np.ndarray | None = None


DEFAULT_MATERIALS: dict[str, Material] = {
    "front": Material("front", (220, 220, 220), 0.05, 0.38),
    "back": Material("back", (90, 92, 96), 0.12, 0.38),
    "body": Material("body", (55, 58, 63), 0.45, 0.28),
    "camera_bump": Material("camera_bump", (45, 47, 52), 0.35, 0.24),
    "lens": Material("lens", (18, 23, 28), 0.12, 0.10),
    "lens_plate": Material("lens_plate", (39, 56, 70), 0.02, 0.42),
    "lens_ring": Material("lens_ring", (154, 171, 182), 0.08, 0.30),
    "flash": Material("flash", (224, 217, 187), 0.00, 0.54),
    "magsafe": Material("magsafe", (218, 212, 199), 0.00, 0.62),
    "button": Material("button", (132, 148, 159), 0.04, 0.36),
    "port": Material("port", (15, 23, 29), 0.00, 0.48),
    "cover": Material("cover", (115, 45, 38), 0.03, 0.52),
    "pages": Material("pages", (228, 218, 191), 0.00, 0.78),
    "spine": Material("spine", (105, 39, 34), 0.02, 0.56),
}


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _positive(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _rgb(value: Any, label: str) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{label} must contain three RGB integers")
    result: list[int] = []
    for index, channel in enumerate(value):
        if isinstance(channel, bool) or not isinstance(channel, (int, float)):
            raise ValueError(f"{label}[{index}] must be an integer from 0 to 255")
        integer = int(channel)
        if float(channel) != integer or not 0 <= integer <= 255:
            raise ValueError(f"{label}[{index}] must be an integer from 0 to 255")
        result.append(integer)
    return tuple(result)  # type: ignore[return-value]


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_-")
    if not slug:
        raise ValueError("asset_name/output_name must contain a letter or number")
    return slug[:80]


def _resolve_path(raw: Any, base: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError(f"{label} exceeds the {MAX_SOURCE_BYTES // (1024 * 1024)} MB safety limit")
    return path


def _quad(raw: Any, label: str) -> np.ndarray:
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != (4, 2) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be four finite [x, y] points")
    contour = array.astype(np.float32).reshape(-1, 1, 2)
    if not cv2.isContourConvex(contour):
        raise ValueError(
            f"{label} must be a convex quad ordered top-left, top-right, bottom-right, bottom-left"
        )
    if abs(float(cv2.contourArea(contour))) < 64.0:
        raise ValueError(f"{label} is too small")
    return array.astype(np.float32)


def _material_map(config: Mapping[str, Any]) -> dict[str, Material]:
    materials = dict(DEFAULT_MATERIALS)
    colors = config.get("colors", {})
    if colors is not None:
        if not isinstance(colors, Mapping):
            raise ValueError("colors must be an object")
        for key, value in colors.items():
            if key not in materials:
                raise ValueError(f"unknown color/material key: {key}")
            materials[key] = replace(materials[key], color_rgb=_rgb(value, f"colors.{key}"))

    raw_materials = config.get("materials", {})
    if not isinstance(raw_materials, Mapping):
        raise ValueError("materials must be an object")
    for key, raw in raw_materials.items():
        if key not in materials:
            raise ValueError(f"unknown material key: {key}")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, Mapping)):
            materials[key] = replace(materials[key], color_rgb=_rgb(raw, f"materials.{key}"))
            continue
        if not isinstance(raw, Mapping):
            raise ValueError(f"materials.{key} must be an RGB array or object")
        base = materials[key]
        color = _rgb(raw.get("color_rgb", base.color_rgb), f"materials.{key}.color_rgb")
        metallic = _finite_number(raw.get("metallic", base.metallic), f"materials.{key}.metallic")
        roughness = _finite_number(raw.get("roughness", base.roughness), f"materials.{key}.roughness")
        if not 0.0 <= metallic <= 1.0 or not 0.0 <= roughness <= 1.0:
            raise ValueError(f"materials.{key} metallic/roughness must be from 0 to 1")
        materials[key] = Material(key, color, metallic, roughness)
    return materials


def load_config(path: Path) -> dict[str, Any]:
    """Load and fully validate a JSON build configuration.

    Returned image paths are absolute and quads are plain lists, making the
    normalized dictionary JSON-serializable for the provenance record.
    """

    config_path = path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON config: {error}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("config root must be an object")
    if raw.get("schema_version", 1) != 1:
        raise ValueError("only schema_version 1 is supported")

    kind = raw.get("kind")
    if kind not in {"phone", "book"}:
        raise ValueError("kind must be 'phone' or 'book'")
    authoring_mode = raw.get("authoring_mode", "reviewed")
    if authoring_mode not in {"reviewed", "automatic"}:
        raise ValueError("authoring_mode must be 'reviewed' or 'automatic'")
    asset_kind = raw.get("asset_kind", kind)
    if asset_kind not in {"phone", "book", "rounded_slab"}:
        raise ValueError("asset_kind must be 'phone', 'book', or 'rounded_slab'")
    compatible_asset_kinds = {
        "phone": {"phone", "rounded_slab"},
        "book": {"book"},
    }
    if asset_kind not in compatible_asset_kinds[kind]:
        raise ValueError(f"asset_kind {asset_kind!r} is incompatible with {kind!r} geometry")
    asset_name = raw.get("asset_name")
    if not isinstance(asset_name, str) or not asset_name.strip():
        raise ValueError("asset_name must be a non-empty string")
    output_name = _slug(str(raw.get("output_name", asset_name)))
    dimension_basis = raw.get("dimension_basis", "configured; not independently measured")
    if not isinstance(dimension_basis, str) or not dimension_basis.strip():
        raise ValueError("dimension_basis must be a non-empty string")
    notes_raw = raw.get("notes", [])
    if not isinstance(notes_raw, list) or len(notes_raw) > 20 or not all(
        isinstance(note, str) and note.strip() for note in notes_raw
    ):
        raise ValueError("notes must be a list of at most 20 non-empty strings")

    base = config_path.parent
    views: dict[str, dict[str, Any]] = {}
    for label in ("front", "back"):
        view = raw.get(label)
        if not isinstance(view, Mapping):
            raise ValueError(f"{label} must be an object")
        image = _resolve_path(view.get("image"), base, f"{label} image")
        quad = _quad(view.get("quad_px"), f"{label}.quad_px")
        turns = view.get("rotate_quarter_turns", 0)
        if isinstance(turns, bool) or not isinstance(turns, int) or turns not in {0, 1, 2, 3}:
            raise ValueError(f"{label}.rotate_quarter_turns must be 0, 1, 2, or 3")
        texture_mode = view.get("texture_mode", "source")
        if texture_mode not in {"source", "material"}:
            raise ValueError(f"{label}.texture_mode must be source or material")
        views[label] = {
            "image": str(image),
            "quad_px": quad.tolist(),
            "rotate_quarter_turns": turns,
            "texture_mode": texture_mode,
        }

    dimensions = raw.get("dimensions_mm")
    if not isinstance(dimensions, Mapping):
        raise ValueError("dimensions_mm must be an object")
    width = _positive(dimensions.get("width"), "dimensions_mm.width")
    height = _positive(dimensions.get("height"), "dimensions_mm.height")
    depth = _positive(dimensions.get("depth"), "dimensions_mm.depth")
    if max(width, height, depth) > 5000:
        raise ValueError("dimensions may not exceed 5000 mm")
    if kind == "phone":
        radius = _positive(
            dimensions.get("corner_radius", min(width, height) * 0.08),
            "dimensions_mm.corner_radius",
        )
        bevel = _positive(
            dimensions.get("bevel", min(depth * 0.12, radius * 0.18)),
            "dimensions_mm.bevel",
        )
        if radius >= min(width, height) / 2:
            raise ValueError("phone corner_radius must be less than half its face size")
        if bevel >= min(depth / 2, radius, width / 2, height / 2):
            raise ValueError("phone bevel must be smaller than half-depth and corner radius")
    else:
        radius = 0.0
        bevel = 0.0

    texture_size_raw = raw.get("texture_size", 1024)
    if isinstance(texture_size_raw, bool) or not isinstance(texture_size_raw, int):
        raise ValueError("texture_size must be an integer")
    if not MIN_TEXTURE_SIZE <= texture_size_raw <= MAX_TEXTURE_SIZE:
        raise ValueError(
            f"texture_size must be between {MIN_TEXTURE_SIZE} and {MAX_TEXTURE_SIZE}"
        )
    rotation_raw = raw.get("output_rotation_deg", [0.0, 0.0, 0.0])
    if not isinstance(rotation_raw, Sequence) or isinstance(rotation_raw, (str, bytes)) or len(rotation_raw) != 3:
        raise ValueError("output_rotation_deg must contain XYZ Euler angles")
    rotation = [_finite_number(value, f"output_rotation_deg[{index}]") for index, value in enumerate(rotation_raw)]

    corner_segments_raw = raw.get("corner_segments", 12)
    if isinstance(corner_segments_raw, bool) or not isinstance(corner_segments_raw, int):
        raise ValueError("corner_segments must be an integer")
    if not 3 <= corner_segments_raw <= MAX_CORNER_SEGMENTS:
        raise ValueError(f"corner_segments must be from 3 to {MAX_CORNER_SEGMENTS}")

    materials = _material_map(raw)
    normalized: dict[str, Any] = {
        "schema_version": 1,
        "asset_name": asset_name.strip(),
        "kind": kind,
        "authoring_mode": authoring_mode,
        "asset_kind": asset_kind,
        "output_name": output_name,
        "dimension_basis": dimension_basis.strip(),
        "notes": [note.strip() for note in notes_raw],
        "config_path": str(config_path),
        "front": views["front"],
        "back": views["back"],
        "dimensions_mm": {
            "width": width,
            "height": height,
            "depth": depth,
            "corner_radius": radius,
            "bevel": bevel,
        },
        "texture_size": texture_size_raw,
        "output_rotation_deg": rotation,
        "corner_segments": corner_segments_raw,
        "materials": {
            key: {
                "color_rgb": list(material.color_rgb),
                "metallic": material.metallic,
                "roughness": material.roughness,
            }
            for key, material in materials.items()
        },
    }
    if kind == "phone":
        normalized["phone"] = _normalize_phone(raw.get("phone", {}), width, height, depth)
    else:
        normalized["book"] = _normalize_book(raw.get("book", {}))
    return normalized


def _normalize_phone(raw: Any, width: float, height: float, depth: float) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("phone must be an object")

    decorations_raw = raw.get("decorations", [])
    if not isinstance(decorations_raw, list) or len(decorations_raw) > 64:
        raise ValueError("phone.decorations must be a list of at most 64 entries")
    decorations: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, decoration in enumerate(decorations_raw):
        label = f"phone.decorations[{index}]"
        if not isinstance(decoration, Mapping):
            raise ValueError(f"{label} must be an object")
        raw_name = decoration.get("name", f"Decoration{index + 1}")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(f"{label}.name must be a non-empty string")
        name = _slug(raw_name)
        if name in names:
            raise ValueError(f"{label}.name must be unique")
        names.add(name)
        kind = decoration.get("type")
        if kind not in {"annulus", "cylinder", "rounded_prism", "box"}:
            raise ValueError(
                f"{label}.type must be annulus, cylinder, rounded_prism, or box"
            )
        material = decoration.get("material")
        if not isinstance(material, str) or material not in DEFAULT_MATERIALS:
            raise ValueError(f"{label}.material must name a supported material")

        if kind == "box":
            center_raw = decoration.get("center_mm")
            size_raw = decoration.get("size_mm")
            if (
                not isinstance(center_raw, Sequence)
                or isinstance(center_raw, (str, bytes))
                or len(center_raw) != 3
            ):
                raise ValueError(f"{label}.center_mm must contain [x, y, z]")
            if (
                not isinstance(size_raw, Sequence)
                or isinstance(size_raw, (str, bytes))
                or len(size_raw) != 3
            ):
                raise ValueError(f"{label}.size_mm must contain [width, height, depth]")
            center = [
                _finite_number(value, f"{label}.center_mm[{axis}]")
                for axis, value in enumerate(center_raw)
            ]
            size = [
                _positive(value, f"{label}.size_mm[{axis}]")
                for axis, value in enumerate(size_raw)
            ]
            limits = (width / 2 + 10.0, height / 2 + 10.0, depth / 2 + 10.0)
            if any(
                abs(center[axis]) + size[axis] / 2 > limits[axis]
                for axis in range(3)
            ):
                raise ValueError(f"{label} lies implausibly far outside the phone")
            decorations.append(
                {
                    "name": name,
                    "type": kind,
                    "material": material,
                    "center_mm": center,
                    "size_mm": size,
                }
            )
            continue

        side = decoration.get("side", "back")
        if side not in {"front", "back"}:
            raise ValueError(f"{label}.side must be front or back")
        center_raw = decoration.get("center_mm")
        if (
            not isinstance(center_raw, Sequence)
            or isinstance(center_raw, (str, bytes))
            or len(center_raw) != 2
        ):
            raise ValueError(f"{label}.center_mm must contain [x, y]")
        center = [
            _finite_number(value, f"{label}.center_mm[{axis}]")
            for axis, value in enumerate(center_raw)
        ]
        offset = _finite_number(
            decoration.get("offset_mm", 0.0), f"{label}.offset_mm"
        )
        protrusion = _positive(
            decoration.get("protrusion_mm", 0.4), f"{label}.protrusion_mm"
        )
        if not 0.0 <= offset <= 20.0 or protrusion > 20.0:
            raise ValueError(f"{label} offset/protrusion is implausibly large")
        segments = decoration.get("segments", 24)
        if (
            isinstance(segments, bool)
            or not isinstance(segments, int)
            or not 8 <= segments <= 48
        ):
            raise ValueError(f"{label}.segments must be an integer from 8 to 48")
        normalized_decoration: dict[str, Any] = {
            "name": name,
            "type": kind,
            "material": material,
            "side": side,
            "center_mm": center,
            "offset_mm": offset,
            "protrusion_mm": protrusion,
            "segments": segments,
        }
        if kind in {"annulus", "cylinder"}:
            radius = _positive(decoration.get("radius_mm"), f"{label}.radius_mm")
            if (
                abs(center[0]) + radius > width / 2
                or abs(center[1]) + radius > height / 2
            ):
                raise ValueError(f"{label} lies outside the phone face")
            normalized_decoration["radius_mm"] = radius
            if kind == "annulus":
                inner = _positive(
                    decoration.get("inner_radius_mm"),
                    f"{label}.inner_radius_mm",
                )
                if inner >= radius:
                    raise ValueError(
                        f"{label}.inner_radius_mm must be smaller than radius_mm"
                    )
                normalized_decoration["inner_radius_mm"] = inner
        else:
            size_raw = decoration.get("size_mm")
            if (
                not isinstance(size_raw, Sequence)
                or isinstance(size_raw, (str, bytes))
                or len(size_raw) != 2
            ):
                raise ValueError(f"{label}.size_mm must contain [width, height]")
            size = [
                _positive(value, f"{label}.size_mm[{axis}]")
                for axis, value in enumerate(size_raw)
            ]
            radius = _positive(
                decoration.get("corner_radius_mm", min(size) * 0.2),
                f"{label}.corner_radius_mm",
            )
            if radius >= min(size) / 2:
                raise ValueError(
                    f"{label}.corner_radius_mm must be less than half its size"
                )
            if (
                abs(center[0]) + size[0] / 2 > width / 2
                or abs(center[1]) + size[1] / 2 > height / 2
            ):
                raise ValueError(f"{label} lies outside the phone face")
            normalized_decoration.update(
                {"size_mm": size, "corner_radius_mm": radius}
            )
        decorations.append(normalized_decoration)

    seam_raw = raw.get("body_seam")
    body_seam: dict[str, float] | None = None
    if seam_raw is not None:
        if not isinstance(seam_raw, Mapping):
            raise ValueError("phone.body_seam must be an object")
        seam_width = _positive(seam_raw.get("width_mm"), "phone.body_seam.width_mm")
        seam_offset = _finite_number(seam_raw.get("offset_mm", 0.0), "phone.body_seam.offset_mm")
        if seam_width >= depth * 0.4:
            raise ValueError("phone.body_seam.width_mm must be smaller than 40% of depth")
        if abs(seam_offset) + seam_width / 2 >= depth / 2:
            raise ValueError("phone.body_seam must lie inside the sidewall")
        body_seam = {"width_mm": seam_width, "offset_mm": seam_offset}

    bump = raw.get("camera_bump")
    if bump is None:
        return {"camera_bump": None, "decorations": decorations, "body_seam": body_seam}
    if not isinstance(bump, Mapping):
        raise ValueError("phone.camera_bump must be an object")
    side = bump.get("side", "back")
    if side not in {"front", "back"}:
        raise ValueError("phone.camera_bump.side must be front or back")
    center_raw = bump.get("center_mm")
    size_raw = bump.get("size_mm")
    if (
        not isinstance(center_raw, Sequence)
        or isinstance(center_raw, (str, bytes))
        or len(center_raw) != 2
    ):
        raise ValueError("phone.camera_bump.center_mm must contain [x, y]")
    if (
        not isinstance(size_raw, Sequence)
        or isinstance(size_raw, (str, bytes))
        or len(size_raw) != 2
    ):
        raise ValueError("phone.camera_bump.size_mm must contain [width, height]")
    center = [
        _finite_number(value, f"phone.camera_bump.center_mm[{index}]")
        for index, value in enumerate(center_raw)
    ]
    size = [
        _positive(value, f"phone.camera_bump.size_mm[{index}]")
        for index, value in enumerate(size_raw)
    ]
    protrusion = _positive(
        bump.get("protrusion_mm"), "phone.camera_bump.protrusion_mm"
    )
    radius = _positive(
        bump.get("corner_radius_mm", min(size) * 0.16),
        "phone.camera_bump.corner_radius_mm",
    )
    if radius >= min(size) / 2:
        raise ValueError("camera bump corner radius must be less than half its size")
    if (
        abs(center[0]) + size[0] / 2 > width / 2 + 1e-6
        or abs(center[1]) + size[1] / 2 > height / 2 + 1e-6
    ):
        raise ValueError("camera bump must lie inside the phone face")
    if protrusion > max(20.0, depth * 2.0):
        raise ValueError("camera bump protrusion is implausibly large")

    lenses_raw = bump.get("lenses", [])
    if not isinstance(lenses_raw, list) or len(lenses_raw) > 8:
        raise ValueError("camera bump lenses must be a list of at most eight entries")
    lenses: list[dict[str, Any]] = []
    for index, lens in enumerate(lenses_raw):
        label = f"phone.camera_bump.lenses[{index}]"
        if not isinstance(lens, Mapping):
            raise ValueError(f"{label} must be an object")
        lens_center_raw = lens.get("center_mm")
        if (
            not isinstance(lens_center_raw, Sequence)
            or isinstance(lens_center_raw, (str, bytes))
            or len(lens_center_raw) != 2
        ):
            raise ValueError(f"{label}.center_mm must contain [x, y]")
        lens_center = [
            _finite_number(value, f"{label}.center_mm[{axis}]")
            for axis, value in enumerate(lens_center_raw)
        ]
        relative_to = lens.get("relative_to", "bump")
        if relative_to not in {"bump", "object"}:
            raise ValueError(f"{label}.relative_to must be bump or object")
        lens_radius = _positive(lens.get("radius_mm"), f"{label}.radius_mm")
        lens_protrusion = _positive(
            lens.get("protrusion_mm", 1.0), f"{label}.protrusion_mm"
        )
        segments = lens.get("segments", 24)
        if (
            isinstance(segments, bool)
            or not isinstance(segments, int)
            or not 8 <= segments <= 48
        ):
            raise ValueError(f"{label}.segments must be an integer from 8 to 48")
        if relative_to == "bump":
            object_x = center[0] + lens_center[0]
            object_y = center[1] + lens_center[1]
        else:
            object_x, object_y = lens_center
        if (
            abs(object_x) + lens_radius > width / 2
            or abs(object_y) + lens_radius > height / 2
        ):
            raise ValueError(f"{label} lies outside the phone face")
        lens_color = lens.get("color_rgb")
        lenses.append(
            {
                "center_mm": lens_center,
                "relative_to": relative_to,
                "radius_mm": lens_radius,
                "protrusion_mm": lens_protrusion,
                "segments": segments,
                **(
                    {"color_rgb": list(_rgb(lens_color, f"{label}.color_rgb"))}
                    if lens_color is not None
                    else {}
                ),
            }
        )
    return {
        "camera_bump": {
            "side": side,
            "center_mm": center,
            "size_mm": size,
            "protrusion_mm": protrusion,
            "corner_radius_mm": radius,
            "lenses": lenses,
        },
        "decorations": decorations,
        "body_seam": body_seam,
    }


def _normalize_book(raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("book must be an object")
    side = raw.get("spine_side", "left")
    if side not in {"left", "right"}:
        raise ValueError("book.spine_side must be left or right")
    return {"spine_side": side}


def _source_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as error:
        raise ValueError(f"could not inspect image: {path}") from error
    if width <= 0 or height <= 0 or width * height > MAX_SOURCE_PIXELS:
        raise ValueError(
            f"source image {path.name} exceeds the {MAX_SOURCE_PIXELS:,}-pixel safety limit"
        )
    return int(width), int(height)


def _texture_dimensions(width_mm: float, height_mm: float, longest_edge: int) -> tuple[int, int]:
    if width_mm >= height_mm:
        width = longest_edge
        height = max(64, int(round(longest_edge * height_mm / width_mm)))
    else:
        height = longest_edge
        width = max(64, int(round(longest_edge * width_mm / height_mm)))
    return min(width, MAX_TEXTURE_SIZE), min(height, MAX_TEXTURE_SIZE)


def rectify_face(
    image_path: Path,
    quad_px: Sequence[Sequence[float]],
    output_size: tuple[int, int],
    border_rgb: tuple[int, int, int],
    rotate_quarter_turns: int = 0,
    *,
    authoring_mode: str = "reviewed",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rectify a provided quad and record whether it came from review or automation."""

    if authoring_mode not in {"reviewed", "automatic"}:
        raise ValueError("authoring_mode must be 'reviewed' or 'automatic'")

    source_width, source_height = _source_dimensions(image_path)
    source = _quad(quad_px, "quad_px")
    tolerance = 0.5
    if (
        float(source[:, 0].min()) < -tolerance
        or float(source[:, 1].min()) < -tolerance
        or float(source[:, 0].max()) > source_width - 1 + tolerance
        or float(source[:, 1].max()) > source_height - 1 + tolerance
    ):
        raise ValueError(f"reviewed quad is outside source image bounds: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    if image.shape[1] != source_width or image.shape[0] != source_height:
        raise ValueError(f"image dimensions changed while reading: {image_path}")
    if rotate_quarter_turns not in {0, 1, 2, 3}:
        raise ValueError("rotate_quarter_turns must be 0, 1, 2, or 3")
    width, height = output_size
    warp_width, warp_height = (
        (height, width) if rotate_quarter_turns % 2 else (width, height)
    )
    destination = np.asarray(
        [
            [0, 0],
            [warp_width - 1, 0],
            [warp_width - 1, warp_height - 1],
            [0, warp_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    border_bgr = tuple(int(value) for value in border_rgb[::-1])
    rectified = cv2.warpPerspective(
        image,
        transform,
        (warp_width, warp_height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_bgr,
    )
    if rotate_quarter_turns == 1:
        rectified = cv2.rotate(rectified, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_quarter_turns == 2:
        rectified = cv2.rotate(rectified, cv2.ROTATE_180)
    elif rotate_quarter_turns == 3:
        rectified = cv2.rotate(rectified, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rectified.shape[:2] != (height, width):
        raise RuntimeError("rectified texture orientation produced unexpected dimensions")
    del image
    quad_key = "reviewed_quad_px" if authoring_mode == "reviewed" else "automatic_quad_px"
    metrics = {
        "source_dimensions_px": [source_width, source_height],
        quad_key: [[round(float(x), 4), round(float(y), 4)] for x, y in source],
        "quad_area_px": round(abs(float(cv2.contourArea(source.reshape(-1, 1, 2)))), 3),
        "texture_dimensions_px": [width, height],
        "rotate_quarter_turns_clockwise": rotate_quarter_turns,
        "rectification": (
            "manual reviewed quad -> perspective transform"
            if authoring_mode == "reviewed"
            else "automatic upstream quad -> perspective transform"
        ),
        # Legacy field retained for schema compatibility.  This function never
        # performs detection; automatic callers supply a quad selected upstream.
        "automatic_detection": False,
        "quad_source": "automatic_upstream" if authoring_mode == "automatic" else "reviewed",
        "rectifier_performed_detection": False,
    }
    return rectified, metrics


def write_quad_review(
    image_path: Path,
    quad_px: Sequence[Sequence[float]],
    output: Path,
    *,
    max_long_side: int = 1280,
) -> None:
    """Write a bounded source-frame overlay for human rectification review."""

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {image_path}")
    scale = min(1.0, max_long_side / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    points = np.rint(np.asarray(quad_px, np.float32) * scale).astype(np.int32)
    cv2.polylines(image, [points], True, (38, 232, 255), 4, cv2.LINE_AA)
    for index, point in enumerate(points):
        cv2.circle(image, tuple(point), 8, (40, 40, 255), cv2.FILLED, cv2.LINE_AA)
        cv2.putText(
            image,
            ("TL", "TR", "BR", "BL")[index],
            (int(point[0]) + 10, int(point[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (40, 40, 255),
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"could not write {output}")


def rounded_rectangle(width: float, height: float, radius: float, segments: int) -> np.ndarray:
    half_width, half_height = width / 2.0, height / 2.0
    radius = min(radius, half_width, half_height)
    corners = (
        ((half_width - radius, half_height - radius), (0.0, 90.0)),
        ((-half_width + radius, half_height - radius), (90.0, 180.0)),
        ((-half_width + radius, -half_height + radius), (180.0, 270.0)),
        ((half_width - radius, -half_height + radius), (270.0, 360.0)),
    )
    points: list[tuple[float, float]] = []
    for (center_x, center_y), (start, stop) in corners:
        for angle in np.linspace(start, stop, segments + 1):
            radians = math.radians(float(angle))
            points.append(
                (center_x + radius * math.cos(radians), center_y + radius * math.sin(radians))
            )
    return np.asarray(points, dtype=np.float64)


def _cap_part(
    name: str,
    boundary: np.ndarray,
    z: float,
    outward_positive_z: bool,
    material: Material,
    width: float,
    height: float,
    texture_key: str | None = None,
    texture_bgr: np.ndarray | None = None,
    mirror_u: bool = False,
) -> MeshPart:
    center_xy = boundary.mean(axis=0)
    vertices = np.vstack(
        (
            np.asarray([[center_xy[0], center_xy[1], z]]),
            np.column_stack((boundary, np.full(len(boundary), z))),
        )
    )
    faces: list[tuple[int, int, int]] = []
    for index in range(len(boundary)):
        current = 1 + index
        following = 1 + ((index + 1) % len(boundary))
        faces.append(
            (0, current, following) if outward_positive_z else (0, following, current)
        )
    uv: np.ndarray | None = None
    if texture_key:
        u = vertices[:, 0] / width + 0.5
        if mirror_u:
            u = 0.5 - vertices[:, 0] / width
        v = vertices[:, 1] / height + 0.5
        uv = np.column_stack((u, v)).clip(0.0, 1.0)
    return MeshPart(
        name,
        vertices,
        np.asarray(faces, dtype=np.int64),
        material,
        uv=uv,
        texture_key=texture_key,
        texture_bgr=texture_bgr,
    )


def _strip_part(
    name: str,
    lower: np.ndarray,
    upper: np.ndarray,
    lower_z: float,
    upper_z: float,
    material: Material,
) -> MeshPart:
    count = len(lower)
    if count != len(upper):
        raise ValueError("ring vertex counts differ")
    vertices = np.vstack(
        (
            np.column_stack((lower, np.full(count, lower_z))),
            np.column_stack((upper, np.full(count, upper_z))),
        )
    )
    faces: list[tuple[int, int, int]] = []
    for index in range(count):
        following = (index + 1) % count
        faces.extend(
            ((index, following, count + following), (index, count + following, count + index))
        )
    return MeshPart(name, vertices, np.asarray(faces, dtype=np.int64), material)


def _closed_rounded_prism(
    name: str,
    center_xy: tuple[float, float],
    width: float,
    height: float,
    radius: float,
    low_z: float,
    high_z: float,
    material: Material,
    segments: int,
) -> list[MeshPart]:
    boundary = rounded_rectangle(width, height, radius, segments)
    boundary += np.asarray(center_xy, dtype=np.float64)
    return [
        _cap_part(f"{name}Low", boundary, low_z, False, material, width, height),
        _strip_part(f"{name}Side", boundary, boundary, low_z, high_z, material),
        _cap_part(f"{name}High", boundary, high_z, True, material, width, height),
    ]


def _closed_cylinder(
    name: str,
    center_xy: tuple[float, float],
    radius: float,
    low_z: float,
    high_z: float,
    material: Material,
    segments: int,
) -> list[MeshPart]:
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    boundary = np.column_stack((np.cos(angles), np.sin(angles))) * radius
    boundary += np.asarray(center_xy, dtype=np.float64)
    return [
        _cap_part(f"{name}Low", boundary, low_z, False, material, radius * 2, radius * 2),
        _strip_part(f"{name}Side", boundary, boundary, low_z, high_z, material),
        _cap_part(f"{name}High", boundary, high_z, True, material, radius * 2, radius * 2),
    ]


def _closed_annulus(
    name: str,
    center_xy: tuple[float, float],
    outer_radius: float,
    inner_radius: float,
    low_z: float,
    high_z: float,
    material: Material,
    segments: int,
) -> MeshPart:
    """Return one closed, flat ring mesh with an actual center opening."""

    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    unit = np.column_stack((np.cos(angles), np.sin(angles)))
    center = np.asarray(center_xy, dtype=np.float64)
    outer = unit * outer_radius + center
    inner = unit * inner_radius + center
    vertices = np.vstack(
        (
            np.column_stack((outer, np.full(segments, low_z))),
            np.column_stack((outer, np.full(segments, high_z))),
            np.column_stack((inner, np.full(segments, low_z))),
            np.column_stack((inner, np.full(segments, high_z))),
        )
    )
    outer_low = 0
    outer_high = segments
    inner_low = segments * 2
    inner_high = segments * 3
    faces: list[tuple[int, int, int]] = []
    for index in range(segments):
        following = (index + 1) % segments
        ol_i, ol_j = outer_low + index, outer_low + following
        oh_i, oh_j = outer_high + index, outer_high + following
        il_i, il_j = inner_low + index, inner_low + following
        ih_i, ih_j = inner_high + index, inner_high + following
        faces.extend(
            (
                (ol_i, ol_j, oh_j),
                (ol_i, oh_j, oh_i),
                (il_i, ih_i, ih_j),
                (il_i, ih_j, il_j),
                (oh_i, oh_j, ih_j),
                (oh_i, ih_j, ih_i),
                (ol_i, il_j, ol_j),
                (ol_i, il_i, il_j),
            )
        )
    return MeshPart(name, vertices, np.asarray(faces, dtype=np.int64), material)


def _closed_box(
    name: str,
    center_xyz: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    material: Material,
) -> MeshPart:
    half = np.asarray(size_xyz, dtype=np.float64) / 2.0
    center = np.asarray(center_xyz, dtype=np.float64)
    x0, y0, z0 = center - half
    x1, y1, z1 = center + half
    vertices = np.asarray(
        (
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ),
        dtype=np.float64,
    )
    faces = np.asarray(
        (
            (0, 2, 1), (0, 3, 2),
            (4, 5, 6), (4, 6, 7),
            (0, 1, 5), (0, 5, 4),
            (1, 2, 6), (1, 6, 5),
            (2, 3, 7), (2, 7, 6),
            (3, 0, 4), (3, 4, 7),
        ),
        dtype=np.int64,
    )
    return MeshPart(name, vertices, faces, material)


def _materials_from_normalized(config: Mapping[str, Any]) -> dict[str, Material]:
    result: dict[str, Material] = {}
    for key, raw in config["materials"].items():
        result[key] = Material(
            key,
            tuple(int(value) for value in raw["color_rgb"]),
            float(raw["metallic"]),
            float(raw["roughness"]),
        )
    return result


def build_phone_parts(
    config: Mapping[str, Any],
    front_texture: np.ndarray,
    back_texture: np.ndarray,
) -> list[MeshPart]:
    dimensions = config["dimensions_mm"]
    width = float(dimensions["width"]) * 0.001
    height = float(dimensions["height"]) * 0.001
    depth = float(dimensions["depth"]) * 0.001
    radius = float(dimensions["corner_radius"]) * 0.001
    bevel = float(dimensions["bevel"]) * 0.001
    segments = int(config["corner_segments"])
    materials = _materials_from_normalized(config)
    half_depth = depth / 2.0
    outer = rounded_rectangle(width, height, radius, segments)
    inset_width = width - 2.0 * bevel
    inset_height = height - 2.0 * bevel
    inset_radius = max(radius - bevel, radius * 0.25)
    inset = rounded_rectangle(inset_width, inset_height, inset_radius, segments)
    front_is_source = config["front"]["texture_mode"] == "source"
    back_is_source = config["back"]["texture_mode"] == "source"
    parts = [
        _cap_part(
            "ObservedFront",
            inset,
            half_depth,
            True,
            materials["front"],
            inset_width,
            inset_height,
            "front" if front_is_source else None,
            front_texture if front_is_source else None,
        ),
        _cap_part(
            "ObservedBack",
            inset,
            -half_depth,
            False,
            materials["back"],
            inset_width,
            inset_height,
            "back" if back_is_source else None,
            back_texture if back_is_source else None,
            mirror_u=True,
        ),
        _strip_part("BackBevel", inset, outer, -half_depth, -half_depth + bevel, materials["body"]),
    ]
    side_low = -half_depth + bevel
    side_high = half_depth - bevel
    seam = config["phone"].get("body_seam")
    if seam:
        seam_center = float(seam["offset_mm"]) * 0.001
        seam_half = float(seam["width_mm"]) * 0.0005
        seam_low = max(side_low, seam_center - seam_half)
        seam_high = min(side_high, seam_center + seam_half)
        parts.extend([
            _strip_part("BodySideBack", outer, outer, side_low, seam_low, materials["body"]),
            _strip_part("BodySeam", outer, outer, seam_low, seam_high, materials["port"]),
            _strip_part("BodySideFront", outer, outer, seam_high, side_high, materials["body"]),
        ])
    else:
        parts.append(_strip_part("BodySide", outer, outer, side_low, side_high, materials["body"]))
    parts.append(_strip_part("FrontBevel", outer, inset, half_depth - bevel, half_depth, materials["body"]))

    bump = config["phone"].get("camera_bump")
    if bump:
        center = tuple(float(value) * 0.001 for value in bump["center_mm"])
        bump_width, bump_height = (float(value) * 0.001 for value in bump["size_mm"])
        bump_depth = float(bump["protrusion_mm"]) * 0.001
        bump_radius = float(bump["corner_radius_mm"]) * 0.001
        back_side = bump["side"] == "back"
        overlap = min(0.00015, depth * 0.025)
        if back_side:
            low_z, high_z = -half_depth - bump_depth, -half_depth + overlap
        else:
            low_z, high_z = half_depth - overlap, half_depth + bump_depth
        parts.extend(
            _closed_rounded_prism(
                "CameraBump",
                center,
                bump_width,
                bump_height,
                bump_radius,
                low_z,
                high_z,
                materials["camera_bump"],
                segments,
            )
        )
        bump_outer_z = low_z if back_side else high_z
        for index, lens in enumerate(bump["lenses"]):
            lens_center = [float(value) * 0.001 for value in lens["center_mm"]]
            if lens["relative_to"] == "bump":
                lens_center[0] += center[0]
                lens_center[1] += center[1]
            lens_radius = float(lens["radius_mm"]) * 0.001
            lens_depth = float(lens["protrusion_mm"]) * 0.001
            lens_material = materials["lens"]
            if "color_rgb" in lens:
                lens_material = replace(
                    lens_material,
                    key=f"lens_{index + 1}",
                    color_rgb=tuple(int(value) for value in lens["color_rgb"]),
                )
            if back_side:
                lens_low, lens_high = bump_outer_z - lens_depth, bump_outer_z + overlap
            else:
                lens_low, lens_high = bump_outer_z - overlap, bump_outer_z + lens_depth
            parts.extend(
                _closed_cylinder(
                    f"Lens{index + 1}",
                    (lens_center[0], lens_center[1]),
                    lens_radius,
                    lens_low,
                    lens_high,
                    lens_material,
                    int(lens["segments"]),
                )
            )
    overlap = min(0.00012, depth * 0.02)
    for decoration in config["phone"].get("decorations", []):
        name = str(decoration["name"])
        material = materials[str(decoration["material"])]
        kind = decoration["type"]
        if kind == "box":
            center_xyz = tuple(float(value) * 0.001 for value in decoration["center_mm"])
            size_xyz = tuple(float(value) * 0.001 for value in decoration["size_mm"])
            parts.append(_closed_box(name, center_xyz, size_xyz, material))
            continue

        center_xy = tuple(float(value) * 0.001 for value in decoration["center_mm"])
        offset = float(decoration["offset_mm"]) * 0.001
        protrusion = float(decoration["protrusion_mm"]) * 0.001
        if decoration["side"] == "back":
            anchor = -half_depth - offset
            low_z, high_z = anchor - protrusion, anchor + overlap
        else:
            anchor = half_depth + offset
            low_z, high_z = anchor - overlap, anchor + protrusion
        decoration_segments = int(decoration["segments"])
        if kind == "annulus":
            parts.append(
                _closed_annulus(
                    name,
                    center_xy,
                    float(decoration["radius_mm"]) * 0.001,
                    float(decoration["inner_radius_mm"]) * 0.001,
                    low_z,
                    high_z,
                    material,
                    decoration_segments,
                )
            )
        elif kind == "cylinder":
            parts.extend(
                _closed_cylinder(
                    name,
                    center_xy,
                    float(decoration["radius_mm"]) * 0.001,
                    low_z,
                    high_z,
                    material,
                    decoration_segments,
                )
            )
        else:
            decoration_width, decoration_height = (
                float(value) * 0.001 for value in decoration["size_mm"]
            )
            parts.extend(
                _closed_rounded_prism(
                    name,
                    center_xy,
                    decoration_width,
                    decoration_height,
                    float(decoration["corner_radius_mm"]) * 0.001,
                    low_z,
                    high_z,
                    material,
                    decoration_segments,
                )
            )
    return parts


def _quad_mesh_part(
    name: str,
    vertices: Sequence[Sequence[float]],
    outward_faces: Sequence[Sequence[int]],
    material: Material,
    uv: Sequence[Sequence[float]] | None = None,
    texture_key: str | None = None,
    texture_bgr: np.ndarray | None = None,
) -> MeshPart:
    return MeshPart(
        name,
        np.asarray(vertices, dtype=np.float64),
        np.asarray(outward_faces, dtype=np.int64),
        material,
        None if uv is None else np.asarray(uv, dtype=np.float64),
        texture_key,
        texture_bgr,
    )


def build_book_parts(
    config: Mapping[str, Any],
    front_texture: np.ndarray,
    back_texture: np.ndarray,
) -> list[MeshPart]:
    dimensions = config["dimensions_mm"]
    width = float(dimensions["width"]) * 0.001
    height = float(dimensions["height"]) * 0.001
    depth = float(dimensions["depth"]) * 0.001
    half_width, half_height, half_depth = width / 2, height / 2, depth / 2
    materials = _materials_from_normalized(config)
    # The two observed covers need separate USD materials because each points
    # at a different texture file, even though their PBR constants match.
    front_cover = replace(materials["cover"], key="front_cover")
    back_cover = replace(materials["cover"], key="back_cover")
    # Vertices in each part are intentionally duplicated.  Exporters retain
    # material boundaries; topology validation merges coincident seams.
    front = _quad_mesh_part(
        "ObservedFrontCover",
        [
            [-half_width, -half_height, half_depth],
            [half_width, -half_height, half_depth],
            [half_width, half_height, half_depth],
            [-half_width, half_height, half_depth],
        ],
        [[0, 1, 2], [0, 2, 3]],
        front_cover,
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        "front",
        front_texture,
    )
    back = _quad_mesh_part(
        "ObservedBackCover",
        [
            [half_width, -half_height, -half_depth],
            [-half_width, -half_height, -half_depth],
            [-half_width, half_height, -half_depth],
            [half_width, half_height, -half_depth],
        ],
        [[0, 1, 2], [0, 2, 3]],
        back_cover,
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        "back",
        back_texture,
    )
    left_vertices = [
        [-half_width, -half_height, -half_depth],
        [-half_width, -half_height, half_depth],
        [-half_width, half_height, half_depth],
        [-half_width, half_height, -half_depth],
    ]
    right_vertices = [
        [half_width, -half_height, half_depth],
        [half_width, -half_height, -half_depth],
        [half_width, half_height, -half_depth],
        [half_width, half_height, half_depth],
    ]
    side_faces = [[0, 1, 2], [0, 2, 3]]
    spine_left = config["book"]["spine_side"] == "left"
    parts = [front, back]
    parts.append(
        _quad_mesh_part(
            "Spine" if spine_left else "ForeEdge",
            left_vertices,
            side_faces,
            materials["spine"] if spine_left else materials["pages"],
        )
    )
    parts.append(
        _quad_mesh_part(
            "ForeEdge" if spine_left else "Spine",
            right_vertices,
            side_faces,
            materials["pages"] if spine_left else materials["spine"],
        )
    )
    parts.extend(
        [
            _quad_mesh_part(
                "PageTop",
                [
                    [-half_width, half_height, half_depth],
                    [half_width, half_height, half_depth],
                    [half_width, half_height, -half_depth],
                    [-half_width, half_height, -half_depth],
                ],
                side_faces,
                materials["pages"],
            ),
            _quad_mesh_part(
                "PageBottom",
                [
                    [-half_width, -half_height, -half_depth],
                    [half_width, -half_height, -half_depth],
                    [half_width, -half_height, half_depth],
                    [-half_width, -half_height, half_depth],
                ],
                side_faces,
                materials["pages"],
            ),
        ]
    )
    return parts


def euler_xyz_matrix(degrees: Sequence[float]) -> np.ndarray:
    x, y, z = (math.radians(float(value)) for value in degrees)
    rx = np.asarray(
        [[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]],
        dtype=np.float64,
    )
    ry = np.asarray(
        [[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]],
        dtype=np.float64,
    )
    rz = np.asarray(
        [[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def apply_output_rotation(parts: list[MeshPart], degrees: Sequence[float]) -> None:
    rotation = euler_xyz_matrix(degrees)
    for part in parts:
        part.vertices = part.vertices @ rotation.T


def topology_metrics(parts: Sequence[MeshPart]) -> dict[str, Any]:
    meshes = [
        trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False)
        for part in parts
    ]
    joined = trimesh.util.concatenate(meshes)
    joined.merge_vertices(digits_vertex=8)
    joined.remove_unreferenced_vertices()
    extents = joined.extents
    return {
        "vertices_after_seam_merge": int(len(joined.vertices)),
        "triangles": int(len(joined.faces)),
        "watertight": bool(joined.is_watertight),
        "winding_consistent": bool(joined.is_winding_consistent),
        "euler_number": int(joined.euler_number),
        "body_count": int(joined.body_count),
        "bounds_m": [
            [round(float(value), 8) for value in row]
            for row in joined.bounds
        ],
        "extents_m": [round(float(value), 8) for value in extents],
        "volume_m3_signed": round(float(joined.volume), 12),
    }


def _trimesh_for_part(part: MeshPart, texture_paths: Mapping[str, Path]) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(vertices=part.vertices, faces=part.faces, process=False)
    if part.uv is not None and part.texture_key:
        with Image.open(texture_paths[part.texture_key]) as source:
            texture = source.convert("RGB")
        material = trimesh.visual.material.PBRMaterial(
            name=part.material.key,
            baseColorTexture=texture,
            metallicFactor=part.material.metallic,
            roughnessFactor=part.material.roughness,
        )
        mesh.visual = trimesh.visual.texture.TextureVisuals(uv=part.uv, material=material)
    else:
        material = trimesh.visual.material.PBRMaterial(
            name=part.material.key,
            baseColorFactor=np.asarray((*part.material.color_rgb, 255), dtype=np.uint8),
            metallicFactor=part.material.metallic,
            roughnessFactor=part.material.roughness,
        )
        mesh.visual = trimesh.visual.texture.TextureVisuals(material=material)
    return mesh


def export_glb(parts: Sequence[MeshPart], output: Path, texture_paths: Mapping[str, Path], root_name: str) -> None:
    scene = trimesh.Scene(base_frame=root_name)
    for part in parts:
        scene.add_geometry(
            _trimesh_for_part(part, texture_paths),
            node_name=part.name,
            geom_name=part.name,
        )
    payload = scene.export(file_type="glb")
    if not isinstance(payload, bytes):
        raise RuntimeError("GLB exporter returned an unexpected payload")
    output.write_bytes(payload)


def _usd_points(values: np.ndarray) -> str:
    return ",\n                ".join(f"({x:.9g}, {y:.9g}, {z:.9g})" for x, y, z in values)


def _usd_pairs(values: np.ndarray) -> str:
    return ",\n                ".join(f"({u:.9g}, {v:.9g})" for u, v in values)


def _usd_ints(values: np.ndarray) -> str:
    return ", ".join(str(int(value)) for value in values.ravel())


def _usd_identifier(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not identifier or identifier[0].isdigit():
        identifier = f"Asset_{identifier}"
    return identifier


def _material_usda(material: Material, texture_name: str | None, root: str) -> str:
    name = _usd_identifier(material.key)
    color = tuple(channel / 255.0 for channel in material.color_rgb)
    diffuse = f"color3f inputs:diffuseColor = ({color[0]:.6g}, {color[1]:.6g}, {color[2]:.6g})"
    texture_nodes = ""
    if texture_name:
        diffuse = f"color3f inputs:diffuseColor.connect = </{root}/Looks/{name}/Texture.outputs:rgb>"
        texture_nodes = f'''
            def Shader "PrimvarReader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @{texture_name}@
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = </{root}/Looks/{name}/PrimvarReader.outputs:result>
                float3 outputs:rgb
            }}'''
    return f'''
        def Material "{name}"
        {{
            token outputs:surface.connect = </{root}/Looks/{name}/PreviewSurface.outputs:surface>
            def Shader "PreviewSurface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                {diffuse}
                float inputs:metallic = {material.metallic:.6g}
                float inputs:roughness = {material.roughness:.6g}
                token outputs:surface
            }}{texture_nodes}
        }}'''


def _mesh_usda(part: MeshPart, root: str) -> str:
    counts = np.full(len(part.faces), 3, dtype=np.int64)
    uv = ""
    if part.uv is not None:
        uv = f'''
            texCoord2f[] primvars:st = [
                {_usd_pairs(part.uv)}
            ] (
                interpolation = "vertex"
            )'''
    return f'''
        def Mesh "{_usd_identifier(part.name)}" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {{
            uniform bool doubleSided = 0
            uniform token subdivisionScheme = "none"
            int[] faceVertexCounts = [{_usd_ints(counts)}]
            int[] faceVertexIndices = [{_usd_ints(part.faces)}]
            point3f[] points = [
                {_usd_points(part.vertices)}
            ]{uv}
            rel material:binding = </{root}/Looks/{_usd_identifier(part.material.key)}>
        }}'''


def author_usda(parts: Sequence[MeshPart], output: Path, texture_paths: Mapping[str, Path], root_name: str) -> None:
    root = _usd_identifier(root_name)
    unique: dict[str, tuple[Material, str | None]] = {}
    for part in parts:
        texture_name = texture_paths[part.texture_key].name if part.texture_key else None
        unique.setdefault(part.material.key, (part.material, texture_name))
    materials = "\n".join(
        _material_usda(material, texture_name, root)
        for material, texture_name in unique.values()
    )
    meshes = "\n".join(_mesh_usda(part, root) for part in parts)
    text = f'''#usda 1.0
(
    defaultPrim = "{root}"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "{root}" (
    kind = "component"
)
{{
    def Scope "Looks"
    {{
{materials}
    }}
{meshes}
}}
'''
    output.write_text(text, encoding="utf-8")


def _run_checked(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"command failed ({' '.join(command)}): {detail}")
    return result


def package_usdz_if_available(usda: Path, usdz: Path, enabled: bool = True) -> dict[str, Any]:
    """Package USDZ with the pure-Python deterministic packer on any platform.

    Apple's ``usdchecker --arkit`` still runs as an extra validation pass when
    the system tool exists; packaging itself no longer needs Apple tools.
    """

    if not enabled:
        return {"created": False, "reason": "disabled by caller"}
    if usdz.exists():
        usdz.unlink()
    report = package_usdz(usda, usdz)
    validation: dict[str, Any] = {"available": False}
    usdchecker = Path("/usr/bin/usdchecker")
    if usdchecker.is_file():
        checked = _run_checked([str(usdchecker), "--arkit", str(usdz)])
        validation = {
            "available": True,
            "command": "Apple usdchecker --arkit",
            "passed": True,
            "output": (checked.stdout.strip() or checked.stderr.strip())[-4000:],
        }
    report["validation"] = validation
    return report


def _camera_basis(direction: np.ndarray) -> np.ndarray:
    forward = direction.astype(np.float64)
    forward /= np.linalg.norm(forward)
    reference_up = np.asarray([0.0, 1.0, 0.0])
    if abs(float(np.dot(forward, reference_up))) > 0.98:
        reference_up = np.asarray([0.0, 0.0, -1.0])
    right = np.cross(reference_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return np.stack((right, up, forward))


def render_parts(parts: Sequence[MeshPart], direction: tuple[float, float, float], size: int = QA_SIZE) -> np.ndarray:
    """Render a small CPU-only orthographic textured QA view."""

    points = np.vstack([part.vertices for part in parts])
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    rotation = _camera_basis(np.asarray(direction, dtype=np.float64))
    projected = np.einsum("ij,kj->ki", rotation, points - center)
    span = np.ptp(projected[:, :2], axis=0)
    scale = size * 0.76 / max(float(span.max()), 1e-9)
    canvas = np.full((size, size, 3), (242, 240, 235), dtype=np.uint8)
    z_buffer = np.full((size, size), -np.inf, dtype=np.float32)
    light = np.asarray([-0.35, 0.65, 1.0], dtype=np.float64)
    light /= np.linalg.norm(light)

    for part in parts:
        camera = np.einsum("ij,kj->ki", rotation, part.vertices - center)
        xy = np.column_stack(
            (camera[:, 0] * scale + size / 2.0, -camera[:, 1] * scale + size / 2.0)
        )
        for face in part.faces:
            triangle = xy[face]
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
            if abs(float(denominator)) < 1e-12:
                continue
            grid_x, grid_y = np.meshgrid(
                np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5,
                np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5,
            )
            w0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
            w1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
            w2 = 1.0 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            depth = w0 * camera[face[0], 2] + w1 * camera[face[1], 2] + w2 * camera[face[2], 2]
            region_z = z_buffer[min_y : max_y + 1, min_x : max_x + 1]
            visible = inside & (depth > region_z)
            if not np.any(visible):
                continue
            world_triangle = part.vertices[face]
            normal = rotation @ np.cross(
                world_triangle[1] - world_triangle[0], world_triangle[2] - world_triangle[0]
            )
            length = np.linalg.norm(normal)
            if length:
                normal /= length
            intensity = 0.76 + 0.24 * abs(float(np.dot(normal, light)))
            if part.uv is not None and part.texture_bgr is not None:
                uv = (
                    w0[..., None] * part.uv[face[0]]
                    + w1[..., None] * part.uv[face[1]]
                    + w2[..., None] * part.uv[face[2]]
                )
                texture = part.texture_bgr
                px = np.clip(
                    np.rint(uv[..., 0] * (texture.shape[1] - 1)),
                    0,
                    texture.shape[1] - 1,
                ).astype(np.int32)
                py = np.clip(
                    np.rint((1.0 - uv[..., 1]) * (texture.shape[0] - 1)),
                    0,
                    texture.shape[0] - 1,
                ).astype(np.int32)
                sampled = texture[py, px]
                color = np.clip(sampled.astype(np.float32) * intensity, 0, 255).astype(np.uint8)
            else:
                base_bgr = np.asarray(part.material.color_rgb[::-1], dtype=np.float32)
                color = np.broadcast_to(
                    np.clip(base_bgr * intensity, 0, 255).astype(np.uint8),
                    (*visible.shape, 3),
                )
            region = canvas[min_y : max_y + 1, min_x : max_x + 1]
            region[visible] = color[visible]
            region_z[visible] = depth[visible]

    silhouette = np.isfinite(z_buffer).astype(np.uint8) * 255
    contours, _ = cv2.findContours(silhouette, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(canvas, contours, -1, (54, 57, 51), 2, cv2.LINE_AA)
    return canvas


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (result.shape[1], 42), (242, 240, 235), -1)
    cv2.putText(result, text, (13, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (42, 45, 40), 2, cv2.LINE_AA)
    return result


def _texture_card(image: np.ndarray, label: str, size: int = QA_SIZE) -> np.ndarray:
    card = np.full((size, size, 3), (242, 240, 235), dtype=np.uint8)
    available = size - 62
    scale = min(available / image.shape[1], available / image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    left = (size - width) // 2
    top = 48 + (available - height) // 2
    card[top : top + height, left : left + width] = resized
    return _label(card, label)


def write_qa(
    parts: Sequence[MeshPart],
    output: Path,
    front: np.ndarray,
    back: np.ndarray,
    *,
    authoring_mode: str = "reviewed",
) -> list[Path]:
    if authoring_mode not in {"reviewed", "automatic"}:
        raise ValueError("authoring_mode must be 'reviewed' or 'automatic'")
    views = [
        ("front three-quarter", (0.48, 0.24, 1.0)),
        ("profile", (1.0, 0.16, 0.10)),
        ("back three-quarter", (-0.48, 0.24, -1.0)),
    ]
    model_contact = output / "qa_model_contact.png"
    texture_contact = output / "qa_texture_contact.png"
    rendered = [_label(render_parts(parts, direction), label) for label, direction in views]
    if not cv2.imwrite(str(model_contact), np.hstack(rendered)):
        raise RuntimeError(f"could not write {model_contact}")
    texture_labels = (
        ("reviewed front / rectified", "reviewed back / rectified")
        if authoring_mode == "reviewed"
        else ("automatic front / rectified", "automatic back / rectified")
    )
    texture_cards = np.hstack(
        (_texture_card(front, texture_labels[0]), _texture_card(back, texture_labels[1]))
    )
    if not cv2.imwrite(str(texture_contact), texture_cards):
        raise RuntimeError(f"could not write {texture_contact}")
    return [model_contact, texture_contact]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_asset(config_path: Path, output: Path, allow_usdz: bool = True) -> dict[str, Any]:
    """Build an asset and return the manifest dictionary.

    ``allow_usdz`` exists for hermetic tests.  The command-line interface leaves
    it enabled unless explicitly asked to skip Apple packaging.
    """

    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except AttributeError:
        pass
    config = load_config(config_path)
    authoring_mode = str(config["authoring_mode"])
    automatic_authoring = authoring_mode == "automatic"
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not output.is_dir():
        raise ValueError(f"output is not a directory: {output}")

    dimensions = config["dimensions_mm"]
    texture_dimensions = _texture_dimensions(
        float(dimensions["width"]),
        float(dimensions["height"]),
        int(config["texture_size"]),
    )
    materials = _materials_from_normalized(config)
    front_path = Path(config["front"]["image"])
    back_path = Path(config["back"]["image"])
    front_texture, front_metrics = rectify_face(
        front_path,
        config["front"]["quad_px"],
        texture_dimensions,
        materials["front" if config["kind"] == "phone" else "cover"].color_rgb,
        int(config["front"]["rotate_quarter_turns"]),
        authoring_mode=authoring_mode,
    )
    back_texture, back_metrics = rectify_face(
        back_path,
        config["back"]["quad_px"],
        texture_dimensions,
        materials["back" if config["kind"] == "phone" else "cover"].color_rgb,
        int(config["back"]["rotate_quarter_turns"]),
        authoring_mode=authoring_mode,
    )

    front_texture_path = output / "front_rectified.png"
    back_texture_path = output / "back_rectified.png"
    if not cv2.imwrite(str(front_texture_path), front_texture):
        raise RuntimeError(f"could not write {front_texture_path}")
    if not cv2.imwrite(str(back_texture_path), back_texture):
        raise RuntimeError(f"could not write {back_texture_path}")
    quad_suffix = "quad_overlay" if automatic_authoring else "quad_review"
    front_quad_review = output / f"front_{quad_suffix}.png"
    back_quad_review = output / f"back_{quad_suffix}.png"
    write_quad_review(front_path, config["front"]["quad_px"], front_quad_review)
    write_quad_review(back_path, config["back"]["quad_px"], back_quad_review)
    texture_paths = {"front": front_texture_path, "back": back_texture_path}

    if config["kind"] == "phone":
        parts = build_phone_parts(config, front_texture, back_texture)
    else:
        parts = build_book_parts(config, front_texture, back_texture)
    apply_output_rotation(parts, config["output_rotation_deg"])
    topology = topology_metrics(parts)
    if not topology["watertight"]:
        raise RuntimeError(f"generated geometry is not watertight: {topology}")
    if topology["triangles"] > 5000:
        raise RuntimeError("generated geometry exceeded the 5,000-triangle safety ceiling")

    base_name = config["output_name"]
    glb_path = output / f"{base_name}.glb"
    usda_path = output / f"{base_name}.usda"
    usdz_path = output / f"{base_name}.usdz"
    export_glb(parts, glb_path, texture_paths, base_name)
    author_usda(parts, usda_path, texture_paths, base_name)
    usdz_report = package_usdz_if_available(usda_path, usdz_path, enabled=allow_usdz)
    qa_paths = write_qa(
        parts,
        output,
        front_texture,
        back_texture,
        authoring_mode=authoring_mode,
    )

    resolved_config_path = output / "resolved_config.json"
    resolved_config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    artifacts = [
        front_texture_path,
        back_texture_path,
        front_quad_review,
        back_quad_review,
        glb_path,
        usda_path,
        resolved_config_path,
        *qa_paths,
    ]
    if usdz_report["created"]:
        artifacts.append(usdz_path)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": None,
        "created_utc_policy": "omitted so identical inputs produce an identical manifest",
        "asset_name": config["asset_name"],
        "asset_kind": config["asset_kind"],
        "geometry_template": config["kind"],
        "authoring_mode": authoring_mode,
        "method": (
            "automatic source selection/rectification + deterministic parametric geometry"
            if automatic_authoring
            else "manual quad rectification + deterministic parametric geometry"
        ),
        "method_limit": (
            (
                "This is an automatically normalized parametric approximation, not "
                "photogrammetry or recovered depth. Face width is normalized to "
                f"{float(dimensions['width']):g} builder mm; no physical scale is inferred. "
                "Observed face pixels may include automatic mask/skin cleanup and body-color "
                "fill; material response and unobserved surfaces use deterministic priors."
            )
            if automatic_authoring
            else (
                "This is a parametric approximation, not photogrammetry or recovered depth. "
                "Metric dimensions, reviewed quads, and requested materials come from the config."
            )
        ),
        "execution": {
            "scope": "parametric builder only; upstream selection, segmentation, and source preparation excluded",
            "local_only": True,
            "cpu_only": True,
            "network_access": False,
            "downloads": False,
            "learned_inference": False,
            "large_language_model": False,
            "torch": False,
            "gpu": False,
            "opencv_threads": 1,
            "texture_long_edge_limit_px": MAX_TEXTURE_SIZE,
            "triangle_limit": 5000,
            "libraries": {
                "opencv": cv2.__version__,
                "numpy": np.__version__,
                "trimesh": trimesh.__version__,
            },
        },
        "configuration": {
            "source_path": str(Path(config["config_path"])),
            "source_sha256": sha256_file(Path(config["config_path"])),
            "resolved_copy": resolved_config_path.name,
            "output_rotation_deg_xyz": config["output_rotation_deg"],
        },
        "geometry": {
            "classification": (
                "parametric / automatically inferred relative fit"
                if automatic_authoring
                else "parametric / manually configured"
            ),
            "dimension_basis": config["dimension_basis"],
            "dimensions_mm": dimensions,
            **(
                {
                    "physical_scale_inferred": False,
                    "dimension_normalization": (
                        f"face width = {float(dimensions['width']):g} builder mm"
                    ),
                }
                if automatic_authoring
                else {}
            ),
            "topology": topology,
            "materials": config["materials"],
            "phone": config.get("phone"),
            "book": config.get("book"),
        },
        "notes": config["notes"],
        "appearance": {
            "front": {
                "classification": (
                    (
                        "observed pixels with automatic selection, rectification, mask/skin cleanup, "
                        "and body-color fill"
                        if automatic_authoring
                        else "observed pixels, manually reviewed quad"
                    )
                    if config["front"]["texture_mode"] == "source"
                    else (
                        "configured material; automatic source retained for QA only"
                        if automatic_authoring
                        else "configured material; reviewed source retained for QA only"
                    )
                ),
                "texture_mode": config["front"]["texture_mode"],
                "source_image": str(front_path),
                "source_sha256": sha256_file(front_path),
                **front_metrics,
            },
            "back": {
                "classification": (
                    (
                        "observed pixels with automatic selection, rectification, mask/skin cleanup, "
                        "and body-color fill"
                        if automatic_authoring
                        else "observed pixels, manually reviewed quad"
                    )
                    if config["back"]["texture_mode"] == "source"
                    else (
                        "configured material; automatic source retained for QA only"
                        if automatic_authoring
                        else "configured material; reviewed source retained for QA only"
                    )
                ),
                "texture_mode": config["back"]["texture_mode"],
                "source_image": str(back_path),
                "source_sha256": sha256_file(back_path),
                **back_metrics,
            },
            "unobserved_surfaces": (
                "Uniform deterministic material priors only; no generative filling or invented texture detail."
                if automatic_authoring
                else "Uniform configured materials only; no generative filling or invented texture detail."
            ),
        },
        "usd_package": usdz_report,
        "artifacts": {path.name: _artifact_record(path) for path in artifacts},
    }
    manifest_path = output / "manifest.json"
    temporary_manifest = output / ".manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="reviewed JSON build config")
    parser.add_argument("--output", required=True, type=Path, help="artifact output directory")
    parser.add_argument(
        "--skip-usdz",
        action="store_true",
        help="write GLB/USDA but skip Apple USDZ packaging",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = build_asset(args.config, args.output, allow_usdz=not args.skip_usdz)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    output = args.output.expanduser().resolve()
    # Use artifact records instead of assuming asset_name and output_name match.
    models = [name for name in manifest["artifacts"] if name.endswith((".glb", ".usdz"))]
    print(f"output: {output}")
    print(f"models: {', '.join(models)}")
    print(
        f"watertight={manifest['geometry']['topology']['watertight']} "
        f"triangles={manifest['geometry']['topology']['triangles']}"
    )
    print("execution: local CPU only; no network, learned inference, Torch, or GPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
