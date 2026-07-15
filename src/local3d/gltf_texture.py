"""Dependency-light writer for textured, multi-part binary glTF assets.

The project deliberately keeps this writer small and explicit.  It accepts
already encoded texture payloads and per-corner mesh attributes, so video and
image dependencies stay in :mod:`local3d.texture_bake`.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class TexturedMeshPart:
    """One glTF primitive with its own explicit material."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    tangents: np.ndarray
    material_class: str = "fabric"


def _finite_array(value: np.ndarray, shape_tail: tuple[int, ...], label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1:] != shape_tail or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite N x {' x '.join(map(str, shape_tail))} array")
    return array


def _validate_part(part: TexturedMeshPart) -> None:
    vertices = _finite_array(part.vertices, (3,), "vertices")
    normals = _finite_array(part.normals, (3,), "normals")
    uvs = _finite_array(part.uvs, (2,), "uvs")
    tangents = _finite_array(part.tangents, (4,), "tangents")
    faces = np.asarray(part.faces)
    if not len(vertices) or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError("each textured part needs vertices and triangular faces")
    if len(normals) != len(vertices) or len(uvs) != len(vertices) or len(tangents) != len(vertices):
        raise ValueError("normal, UV, and tangent counts must match the position count")
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("face index outside the part vertex array")
    if np.any(uvs < -1e-6) or np.any(uvs > 1.0 + 1e-6):
        raise ValueError("UV coordinates must remain inside the shared atlas")
    normal_lengths = np.linalg.norm(normals, axis=1)
    tangent_lengths = np.linalg.norm(tangents[:, :3], axis=1)
    if np.any(np.abs(normal_lengths - 1.0) > 2e-4):
        raise ValueError("normals must be unit length")
    if np.any(np.abs(tangent_lengths - 1.0) > 2e-4):
        raise ValueError("tangent XYZ values must be unit length")
    tangent_normal_dot = np.einsum("ij,ij->i", normals, tangents[:, :3])
    if np.any(np.abs(tangent_normal_dot) > 2e-4):
        raise ValueError("tangent XYZ values must be perpendicular to normals")
    if np.any(np.abs(np.abs(tangents[:, 3]) - 1.0) > 1e-6):
        raise ValueError("tangent handedness must be -1 or +1")


def _pack_floats(array: np.ndarray) -> bytes:
    values = np.asarray(array, dtype="<f4")
    return values.tobytes(order="C")


def _json_number(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite accessor bound")
    return number


def write_textured_glb(
    path: str | Path,
    parts: Sequence[TexturedMeshPart],
    *,
    base_color: bytes,
    base_color_mime: str,
    normal_map: bytes,
    metallic_roughness: bytes,
    generator: str = "local3d-texture-bake",
    extras: Mapping[str, Any] | None = None,
) -> Path:
    """Write a self-contained GLB with shared base, normal, and MR maps.

    All three images are embedded as buffer views.  A separate material is
    retained for every input part, while the atlas textures are shared.
    """

    if not parts:
        raise ValueError("textured GLB needs at least one mesh part")
    if base_color_mime not in {"image/jpeg", "image/png"}:
        raise ValueError("base color must be encoded as JPEG or PNG")
    if not base_color or not normal_map or not metallic_roughness:
        raise ValueError("all three embedded texture payloads are required")
    for part in parts:
        _validate_part(part)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "TexturedObject"}],
        "meshes": [{"name": "TexturedObject", "primitives": []}],
        "materials": [],
        "buffers": [{"byteLength": 0}],
        "bufferViews": [],
        "accessors": [],
        "samplers": [
            {
                "magFilter": 9729,
                "minFilter": 9729,
                "wrapS": 33071,
                "wrapT": 33071,
            }
        ],
        "images": [],
        "textures": [],
    }
    binary = bytearray()

    def append_view(payload: bytes, *, target: int | None = None) -> int:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(payload)
        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(payload),
        }
        if target is not None:
            view["target"] = target
        index = len(document["bufferViews"])
        document["bufferViews"].append(view)
        return index

    def append_accessor(
        payload: bytes,
        *,
        component_type: int,
        count: int,
        kind: str,
        target: int,
        minimum: Sequence[float] | None = None,
        maximum: Sequence[float] | None = None,
    ) -> int:
        view = append_view(payload, target=target)
        accessor: dict[str, Any] = {
            "bufferView": view,
            "byteOffset": 0,
            "componentType": component_type,
            "count": int(count),
            "type": kind,
        }
        if minimum is not None:
            accessor["min"] = [_json_number(item) for item in minimum]
        if maximum is not None:
            accessor["max"] = [_json_number(item) for item in maximum]
        index = len(document["accessors"])
        document["accessors"].append(accessor)
        return index

    for part_index, part in enumerate(parts):
        vertices = np.asarray(part.vertices, dtype=np.float32)
        normals = np.asarray(part.normals, dtype=np.float32)
        uvs = np.asarray(part.uvs, dtype=np.float32)
        tangents = np.asarray(part.tangents, dtype=np.float32)
        faces = np.asarray(part.faces, dtype=np.int64)

        position_accessor = append_accessor(
            _pack_floats(vertices),
            component_type=5126,
            count=len(vertices),
            kind="VEC3",
            target=34962,
            minimum=vertices.min(axis=0),
            maximum=vertices.max(axis=0),
        )
        normal_accessor = append_accessor(
            _pack_floats(normals),
            component_type=5126,
            count=len(normals),
            kind="VEC3",
            target=34962,
        )
        uv_accessor = append_accessor(
            _pack_floats(uvs),
            component_type=5126,
            count=len(uvs),
            kind="VEC2",
            target=34962,
            minimum=uvs.min(axis=0),
            maximum=uvs.max(axis=0),
        )
        tangent_accessor = append_accessor(
            _pack_floats(tangents),
            component_type=5126,
            count=len(tangents),
            kind="VEC4",
            target=34962,
        )
        maximum_index = int(faces.max())
        if maximum_index < 65535:
            index_payload = np.asarray(faces, dtype="<u2").tobytes(order="C")
            component_type = 5123
        else:
            index_payload = np.asarray(faces, dtype="<u4").tobytes(order="C")
            component_type = 5125
        index_accessor = append_accessor(
            index_payload,
            component_type=component_type,
            count=faces.size,
            kind="SCALAR",
            target=34963,
        )

        material_index = len(document["materials"])
        document["materials"].append(
            {
                "name": f"PartMaterial_{part_index}_{part.material_class}",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "baseColorTexture": {"index": 0, "texCoord": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                    "metallicRoughnessTexture": {"index": 2, "texCoord": 0},
                },
                "normalTexture": {"index": 1, "texCoord": 0, "scale": 1.0},
                "doubleSided": False,
                "extras": {
                    "partName": part.name,
                    "materialClass": part.material_class,
                },
            }
        )
        document["meshes"][0]["primitives"].append(
            {
                "attributes": {
                    "POSITION": position_accessor,
                    "NORMAL": normal_accessor,
                    "TEXCOORD_0": uv_accessor,
                    "TANGENT": tangent_accessor,
                },
                "indices": index_accessor,
                "material": material_index,
                "mode": 4,
            }
        )

    image_payloads = (
        ("BaseColorAtlas", base_color_mime, base_color),
        ("TangentSpaceNormal", "image/png", normal_map),
        ("MetallicRoughness", "image/png", metallic_roughness),
    )
    for name, mime_type, payload in image_payloads:
        view = append_view(payload)
        image_index = len(document["images"])
        document["images"].append(
            {"name": name, "bufferView": view, "mimeType": mime_type}
        )
        document["textures"].append(
            {"name": name, "sampler": 0, "source": image_index}
        )

    while len(binary) % 4:
        binary.append(0)
    document["buffers"][0]["byteLength"] = len(binary)
    if extras:
        document["asset"]["extras"] = dict(extras)

    json_chunk = json.dumps(
        document,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=False,
    ).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    with destination.open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, total_length))
        stream.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        stream.write(json_chunk)
        stream.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        stream.write(binary)
    return destination
