"""Backend contracts and dependency-free local fallbacks.

Real integrations (SAM2, DA3, ForeHOI, Apple Object Capture, and similar) can
implement these protocols without changing the job runner.  The fallbacks in
this file are intentionally *not* reconstruction models: they make the entire
local pipeline executable and produce standards-compliant artifacts while
clearly recording that their results are placeholders.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence, runtime_checkable

from .core import ArtifactSpec, PipelineContext, PipelineError, StageResult


@runtime_checkable
class SegmentationBackend(Protocol):
    """Contract for a local object/hand video segmentation implementation."""

    name: str

    def is_available(self) -> bool:
        """Return whether all local code/checkpoints needed by this backend exist."""

    def segment(self, context: PipelineContext) -> StageResult:
        """Write segmentation outputs under ``context.output_dir()``."""


@runtime_checkable
class ReconstructionBackend(Protocol):
    """Contract for a local mesh or scene reconstruction implementation."""

    name: str

    def is_available(self) -> bool:
        """Return whether all local code/checkpoints needed by this backend exist."""

    def reconstruct(self, context: PipelineContext) -> StageResult:
        """Write reconstruction outputs under ``context.output_dir()``."""


def first_available(backends: Iterable[Any]) -> Any | None:
    """Choose the first available local backend without importing model stacks."""

    for backend in backends:
        predicate = getattr(backend, "is_available", None)
        if not callable(predicate) or predicate():
            return backend
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_integer(value: Any, default: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result >= 0 else default


def _write_full_frame_pbm(path: Path, width: int, height: int) -> None:
    """Write a compact all-white binary PBM mask (white means foreground)."""

    if width <= 0 or height <= 0:
        raise ValueError("mask dimensions must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    row_bytes = (width + 7) // 8
    with path.open("wb") as stream:
        stream.write(f"P4\n{width} {height}\n".encode("ascii"))
        # PBM encodes white pixels as zero bits.  Stream rows to bound memory.
        blank_row = b"\x00" * row_bytes
        for _ in range(height):
            stream.write(blank_row)


class FullFrameSegmentationBackend:
    """Dependency-free wiring fallback that marks every pixel as object.

    This is useful only for exercising orchestration/export code.  The result's
    warning and metadata are designed to prevent it from being presented as a
    legitimate hand-object segmentation.
    """

    name = "full-frame-placeholder"

    def is_available(self) -> bool:
        return True

    def segment(self, context: PipelineContext) -> StageResult:
        analysis = context.read_analysis()
        frames: list[dict[str, Any]] = []
        global_width = 0
        global_height = 0
        if analysis:
            metadata = analysis.get("metadata", {})
            if isinstance(metadata, dict):
                global_width = _safe_integer(
                    metadata.get("display_width", metadata.get("width", 0))
                )
                global_height = _safe_integer(
                    metadata.get("display_height", metadata.get("height", 0))
                )
            selected = analysis.get("keyframes") or analysis.get("frames") or []
            if isinstance(selected, list):
                frames = [item for item in selected if isinstance(item, dict)]

        mask_entries: list[dict[str, Any]] = []
        masks_dir = context.output_path("masks")
        masks_dir.mkdir(parents=True, exist_ok=True)
        for index, frame in enumerate(frames):
            width = _safe_integer(frame.get("width"), global_width)
            height = _safe_integer(frame.get("height"), global_height)
            candidate_index = _safe_integer(frame.get("candidate_index"), index)
            mask_path: Path | None = None
            if width > 0 and height > 0:
                mask_path = masks_dir / f"mask_{candidate_index:06d}.pbm"
                _write_full_frame_pbm(mask_path, width, height)
            mask_entries.append(
                {
                    "candidate_index": candidate_index,
                    "source_frame_index": frame.get("source_frame_index"),
                    "timestamp_s": frame.get("timestamp_s"),
                    "frame_path": frame.get("path"),
                    "width": width or None,
                    "height": height or None,
                    "object_mask_path": str(mask_path.resolve()) if mask_path else None,
                    "object_mask_sha256": _sha256(mask_path) if mask_path else None,
                    "hand_mask_path": None,
                    "uncertain_mask_path": None,
                    "confidence": 0.0,
                }
            )

        manifest_path = context.output_path("masks.json")
        _write_json(
            manifest_path,
            {
                "schema_version": "1.0",
                "backend": self.name,
                "placeholder": True,
                "semantics": {
                    "white": "assumed_object",
                    "black": "background",
                    "hand_mask": "not_available",
                },
                "source_analysis": str(context.analysis_path()) if context.analysis_path() else None,
                "frames": mask_entries,
                "warning": (
                    "Full-frame masks contain the background and the user's hand. "
                    "Replace this backend before attempting a real reconstruction."
                ),
            },
        )
        return StageResult.of(
            [
                ArtifactSpec(
                    "segmentation.manifest",
                    manifest_path,
                    "application/json",
                    {
                        "placeholder": True,
                        "frame_count": len(mask_entries),
                        "mask_format": "image/x-portable-bitmap",
                    },
                )
            ],
            metadata={"placeholder": True, "frame_count": len(mask_entries)},
            warnings=[
                "Segmentation used the full-frame placeholder; no object or hand was isolated."
            ],
        )


Vertex = tuple[float, float, float]
Triangle = tuple[int, int, int]


def _validate_mesh(vertices: Sequence[Vertex], triangles: Sequence[Triangle]) -> None:
    if not vertices:
        raise PipelineError("mesh has no vertices")
    if not triangles:
        raise PipelineError("mesh has no triangles")
    for vertex in vertices:
        if len(vertex) != 3 or not all(math.isfinite(float(item)) for item in vertex):
            raise PipelineError(f"invalid mesh vertex: {vertex!r}")
    for triangle in triangles:
        if len(triangle) != 3 or any(index < 0 or index >= len(vertices) for index in triangle):
            raise PipelineError(f"invalid mesh triangle: {triangle!r}")


def write_obj_mesh(
    path: str | Path,
    vertices: Sequence[Vertex],
    triangles: Sequence[Triangle],
    *,
    comment: str | None = None,
) -> Path:
    """Write a simple triangle OBJ without requiring a geometry package."""

    _validate_mesh(vertices, triangles)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# local3d mesh\n")
        if comment:
            stream.write(f"# {comment.replace(chr(10), ' ')}\n")
        for x, y, z in vertices:
            stream.write(f"v {float(x):.9g} {float(y):.9g} {float(z):.9g}\n")
        for a, b, c in triangles:
            stream.write(f"f {a + 1} {b + 1} {c + 1}\n")
    return destination


def write_glb_mesh(
    path: str | Path,
    vertices: Sequence[Vertex],
    triangles: Sequence[Triangle],
    *,
    generator: str = "local3d",
    extras: dict[str, Any] | None = None,
    vertex_colors: Sequence[Sequence[int | float]] | None = None,
    normals: Sequence[Sequence[float]] | None = None,
) -> Path:
    """Write a minimal, standards-compliant binary glTF 2.0 triangle mesh."""

    _validate_mesh(vertices, triangles)
    if vertex_colors is not None:
        if len(vertex_colors) != len(vertices) or any(len(color) not in (3, 4) for color in vertex_colors):
            raise PipelineError("vertex_colors must contain one RGB or RGBA value per vertex")
    if normals is not None:
        if len(normals) != len(vertices) or any(
            len(normal) != 3 or not all(math.isfinite(float(value)) for value in normal)
            for normal in normals
        ):
            raise PipelineError("normals must contain one finite XYZ vector per vertex")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    position_values = [float(value) for vertex in vertices for value in vertex]
    positions = struct.pack(f"<{len(position_values)}f", *position_values)
    position_padding = (-len(positions)) % 4
    positions += b"\x00" * position_padding

    flat_indices = [int(index) for triangle in triangles for index in triangle]
    if max(flat_indices) < 65535:
        component_type = 5123  # UNSIGNED_SHORT
        indices = struct.pack(f"<{len(flat_indices)}H", *flat_indices)
    else:
        component_type = 5125  # UNSIGNED_INT
        indices = struct.pack(f"<{len(flat_indices)}I", *flat_indices)
    indices += b"\x00" * ((-len(indices)) % 4)
    binary_parts = [positions, indices]

    minimum = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
    maximum = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Object"}],
        "meshes": [
            {
                "name": "Object",
                "primitives": [
                    {
                        "attributes": {"POSITION": 0},
                        "indices": 1,
                        "mode": 4,
                    }
                ],
            }
        ],
        "buffers": [{"byteLength": 0}],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(positions) - position_padding,
                "target": 34962,
            },
            {
                "buffer": 0,
                "byteOffset": len(positions),
                "byteLength": len(flat_indices) * (2 if component_type == 5123 else 4),
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": len(vertices),
                "type": "VEC3",
                "min": minimum,
                "max": maximum,
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": component_type,
                "count": len(flat_indices),
                "type": "SCALAR",
            },
        ],
    }
    primitive = document["meshes"][0]["primitives"][0]
    byte_offset = len(positions) + len(indices)
    if vertex_colors is not None:
        color_values: list[int] = []
        for color in vertex_colors:
            values = list(color) + ([255] if len(color) == 3 else [])
            for value in values:
                numeric = float(value)
                if 0.0 <= numeric <= 1.0 and not isinstance(value, int):
                    numeric *= 255.0
                color_values.append(max(0, min(255, int(round(numeric)))))
        colors = bytes(color_values)
        colors += b"\x00" * ((-len(colors)) % 4)
        binary_parts.append(colors)
        document["bufferViews"].append({
            "buffer": 0, "byteOffset": byte_offset,
            "byteLength": len(color_values), "target": 34962,
        })
        document["accessors"].append({
            "bufferView": len(document["bufferViews"]) - 1, "byteOffset": 0,
            "componentType": 5121, "normalized": True,
            "count": len(vertices), "type": "VEC4",
        })
        primitive["attributes"]["COLOR_0"] = len(document["accessors"]) - 1
        byte_offset += len(colors)
    if normals is not None:
        normal_values = [float(value) for normal in normals for value in normal]
        normal_bytes = struct.pack(f"<{len(normal_values)}f", *normal_values)
        normal_bytes += b"\x00" * ((-len(normal_bytes)) % 4)
        binary_parts.append(normal_bytes)
        document["bufferViews"].append({
            "buffer": 0, "byteOffset": byte_offset,
            "byteLength": len(normal_values) * 4, "target": 34962,
        })
        document["accessors"].append({
            "bufferView": len(document["bufferViews"]) - 1, "byteOffset": 0,
            "componentType": 5126, "count": len(vertices), "type": "VEC3",
        })
        primitive["attributes"]["NORMAL"] = len(document["accessors"]) - 1
        byte_offset += len(normal_bytes)
    if vertex_colors is not None or normals is not None:
        document["materials"] = [{
            "name": "BakedAppearance",
            "pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 1},
            "doubleSided": False,
        }]
        primitive["material"] = 0
    binary_blob = b"".join(binary_parts)
    document["buffers"][0]["byteLength"] = len(binary_blob)
    if extras:
        document["asset"]["extras"] = extras

    json_chunk = json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_blob)
    with destination.open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, total_length))
        stream.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        stream.write(json_chunk)
        stream.write(struct.pack("<I4s", len(binary_blob), b"BIN\x00"))
        stream.write(binary_blob)
    return destination


def write_glb_material_parts(
    path: str | Path,
    parts: Sequence[tuple[Sequence[Vertex], Sequence[Triangle], Sequence[Sequence[float]], Sequence[int]]],
    *,
    generator: str = "local3d",
    extras: dict[str, Any] | None = None,
) -> Path:
    """Write multiple colored parts using explicit PBR materials.

    Explicit materials are used instead of ``COLOR_0`` because some common
    Apple preview surfaces ignore vertex colors even though glTF permits them.
    """
    if not parts:
        raise PipelineError("material-parts GLB needs at least one part")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Object"}],
        "meshes": [{"name": "Object", "primitives": []}],
        "materials": [], "buffers": [{"byteLength": 0}],
        "bufferViews": [], "accessors": [],
    }
    binary = bytearray()

    def append_view(payload: bytes, *, target: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(payload)
        index = len(document["bufferViews"])
        document["bufferViews"].append({
            "buffer": 0, "byteOffset": offset, "byteLength": len(payload), "target": target,
        })
        return index

    def srgb_to_linear(value: int) -> float:
        channel = max(0.0, min(1.0, float(value) / 255.0))
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    for part_index, (vertices, triangles, normals, color) in enumerate(parts):
        _validate_mesh(vertices, triangles)
        if len(normals) != len(vertices):
            raise PipelineError("each material part needs one normal per vertex")
        rgba = list(color) + ([255] if len(color) == 3 else [])
        if len(rgba) != 4:
            raise PipelineError("part color must be RGB or RGBA")

        position_values = [float(value) for vertex in vertices for value in vertex]
        position_payload = struct.pack(f"<{len(position_values)}f", *position_values)
        position_view = append_view(position_payload, target=34962)
        minimum = [min(vertex[axis] for vertex in vertices) for axis in range(3)]
        maximum = [max(vertex[axis] for vertex in vertices) for axis in range(3)]
        position_accessor = len(document["accessors"])
        document["accessors"].append({
            "bufferView": position_view, "byteOffset": 0, "componentType": 5126,
            "count": len(vertices), "type": "VEC3", "min": minimum, "max": maximum,
        })

        normal_values = [float(value) for normal in normals for value in normal]
        normal_view = append_view(struct.pack(f"<{len(normal_values)}f", *normal_values), target=34962)
        normal_accessor = len(document["accessors"])
        document["accessors"].append({
            "bufferView": normal_view, "byteOffset": 0, "componentType": 5126,
            "count": len(vertices), "type": "VEC3",
        })

        flat_indices = [int(value) for triangle in triangles for value in triangle]
        if max(flat_indices) < 65535:
            component_type, code = 5123, "H"
        else:
            component_type, code = 5125, "I"
        index_view = append_view(struct.pack(f"<{len(flat_indices)}{code}", *flat_indices), target=34963)
        index_accessor = len(document["accessors"])
        document["accessors"].append({
            "bufferView": index_view, "byteOffset": 0, "componentType": component_type,
            "count": len(flat_indices), "type": "SCALAR",
        })

        alpha = max(0.0, min(1.0, float(rgba[3]) / 255.0))
        material_index = len(document["materials"])
        document["materials"].append({
            "name": f"PartMaterial_{part_index}",
            "pbrMetallicRoughness": {
                "baseColorFactor": [*(srgb_to_linear(int(value)) for value in rgba[:3]), alpha],
                "metallicFactor": 0.0, "roughnessFactor": 0.94,
            },
            "doubleSided": False,
        })
        document["meshes"][0]["primitives"].append({
            "attributes": {"POSITION": position_accessor, "NORMAL": normal_accessor},
            "indices": index_accessor, "material": material_index, "mode": 4,
        })

    while len(binary) % 4:
        binary.append(0)
    document["buffers"][0]["byteLength"] = len(binary)
    if extras:
        document["asset"]["extras"] = extras
    json_chunk = json.dumps(document, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary)
    with destination.open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, total_length))
        stream.write(struct.pack("<I4s", len(json_chunk), b"JSON"))
        stream.write(json_chunk)
        stream.write(struct.pack("<I4s", len(binary), b"BIN\x00"))
        stream.write(binary)
    return destination


def read_obj_mesh(path: str | Path) -> tuple[list[Vertex], list[Triangle]]:
    """Read the vertex/face subset of OBJ and triangulate polygon fans."""

    source = Path(path)
    vertices: list[Vertex] = []
    triangles: list[Triangle] = []
    try:
        lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise PipelineError(f"could not read OBJ {source}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            try:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError as exc:
                raise PipelineError(f"invalid OBJ vertex at line {line_number}") from exc
        elif parts[0] == "f" and len(parts) >= 4:
            face: list[int] = []
            for token in parts[1:]:
                vertex_token = token.split("/", 1)[0]
                try:
                    obj_index = int(vertex_token)
                except ValueError as exc:
                    raise PipelineError(f"invalid OBJ face at line {line_number}") from exc
                index = obj_index - 1 if obj_index > 0 else len(vertices) + obj_index
                if index < 0 or index >= len(vertices):
                    raise PipelineError(f"OBJ face index out of range at line {line_number}")
                face.append(index)
            for offset in range(1, len(face) - 1):
                triangles.append((face[0], face[offset], face[offset + 1]))
    _validate_mesh(vertices, triangles)
    return vertices, triangles


def ensure_glb(mesh_path: str | Path, output_path: str | Path) -> Path:
    """Copy a GLB or convert a geometry-only OBJ using the pure-Python writer."""

    source = Path(mesh_path)
    destination = Path(output_path)
    if source.suffix.lower() == ".glb":
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        return destination
    if source.suffix.lower() == ".obj":
        vertices, triangles = read_obj_mesh(source)
        return write_glb_mesh(destination, vertices, triangles)
    raise PipelineError(
        f"dependency-free GLB export supports only .obj and .glb, got {source.suffix or 'no suffix'}"
    )


def _unit_cube() -> tuple[list[Vertex], list[Triangle]]:
    vertices: list[Vertex] = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    triangles: list[Triangle] = [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]
    return vertices, triangles


class PlaceholderReconstructionBackend:
    """Emit a valid unit-cube OBJ/GLB so local pipeline plumbing can be tested."""

    name = "unit-cube-placeholder"

    def is_available(self) -> bool:
        return True

    def reconstruct(self, context: PipelineContext) -> StageResult:
        vertices, triangles = _unit_cube()
        obj_path = context.output_path("model.obj")
        glb_path = context.output_path("model.glb")
        quality_path = context.output_path("quality.json")
        write_obj_mesh(
            obj_path,
            vertices,
            triangles,
            comment="PLACEHOLDER: not reconstructed from the source video",
        )
        write_glb_mesh(
            glb_path,
            vertices,
            triangles,
            generator="local3d placeholder backend",
            extras={
                "placeholder": True,
                "warning": "This unit cube was not reconstructed from the source video.",
            },
        )
        _write_json(
            quality_path,
            {
                "schema_version": "1.0",
                "status": "placeholder",
                "usable_reconstruction": False,
                "scale_status": "relative",
                "observed_surface_fraction": 0.0,
                "mesh": {
                    "vertices": len(vertices),
                    "triangles": len(triangles),
                    "watertight_by_construction": True,
                },
                "warning": (
                    "The selected backend is a pipeline fallback. The unit cube contains "
                    "no geometry or texture recovered from the input video."
                ),
            },
        )
        return StageResult.of(
            [
                ArtifactSpec(
                    "reconstruction.model.glb",
                    glb_path,
                    "model/gltf-binary",
                    {"placeholder": True, "scale_status": "relative"},
                ),
                ArtifactSpec(
                    "reconstruction.mesh.obj",
                    obj_path,
                    "model/obj",
                    {"placeholder": True, "scale_status": "relative"},
                ),
                ArtifactSpec(
                    "reconstruction.quality",
                    quality_path,
                    "application/json",
                    {"usable_reconstruction": False},
                ),
            ],
            metadata={
                "placeholder": True,
                "usable_reconstruction": False,
                "vertex_count": len(vertices),
                "triangle_count": len(triangles),
            },
            warnings=[
                "Reconstruction used the unit-cube placeholder; the GLB is only a plumbing artifact."
            ],
        )
