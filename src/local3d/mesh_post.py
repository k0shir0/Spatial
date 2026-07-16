"""Optional deterministic mesh post-processing: decimation and manifold repair.

Visual-hull and fitted meshes can carry more triangles than a delivery GLB or
USDZ needs.  This module wraps two small, permissively licensed native
libraries — ``fast-simplification`` (quadric decimation) and ``manifold3d``
(vertex welding to a closed manifold) — behind explicit, review-friendly
entry points.  Both are optional: install the ``mesh`` extra to enable them.

Determinism: both libraries are single-threaded and seed-free, so identical
input arrays and pinned versions reproduce identical output arrays on one
platform.  Cross-architecture (x86 vs ARM) bit-identity is not promised.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh


def _require(module_name: str) -> Any:
    try:
        return __import__(module_name)
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            f"{module_name} is not installed; install the optional mesh extra: "
            "python -m pip install 'spatial-local3d[mesh]'"
        ) from exc


def mesh_report(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    """Small topology summary used in build manifests."""

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return {
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "euler_number": int(mesh.euler_number),
    }


def decimate(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_triangles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Quadric-decimate to approximately ``target_triangles`` triangles."""

    if target_triangles < 4:
        raise ValueError("target_triangles must be at least 4")
    if target_triangles >= len(faces):
        return (
            np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.int32),
        )
    fast_simplification = _require("fast_simplification")
    out_vertices, out_faces = fast_simplification.simplify(
        np.ascontiguousarray(vertices, dtype=np.float32),
        np.ascontiguousarray(faces, dtype=np.int64),
        target_count=int(target_triangles),
    )
    return out_vertices.astype(np.float32), out_faces.astype(np.int32)


def weld_to_manifold(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Weld near-duplicate vertices along open edges into a closed manifold.

    Raises ``ValueError`` when welding cannot produce a valid solid, so a
    damaged mesh fails closed instead of shipping silently.
    """

    manifold3d = _require("manifold3d")
    mesh = manifold3d.Mesh(
        vert_properties=np.ascontiguousarray(vertices, dtype=np.float32),
        tri_verts=np.ascontiguousarray(faces, dtype=np.uint32),
    )
    mesh.merge()
    solid = manifold3d.Manifold(mesh)
    if solid.is_empty() or solid.num_tri() == 0:
        raise ValueError("mesh could not be welded into a valid manifold solid")
    out = solid.to_mesh()
    return (
        np.asarray(out.vert_properties[:, :3], dtype=np.float32),
        np.asarray(out.tri_verts, dtype=np.int32),
    )


def postprocess(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    target_triangles: int = 0,
    require_watertight: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Optionally decimate, then verify (and if needed re-weld) watertightness.

    ``target_triangles=0`` disables decimation.  Returns the processed arrays
    plus a report suitable for a build manifest.
    """

    report: dict[str, Any] = {"before": mesh_report(vertices, faces), "steps": []}
    out_vertices = np.asarray(vertices, dtype=np.float32)
    out_faces = np.asarray(faces, dtype=np.int32)

    if target_triangles:
        out_vertices, out_faces = decimate(
            out_vertices, out_faces, target_triangles=target_triangles
        )
        report["steps"].append({"step": "decimate", "target_triangles": int(target_triangles)})

    interim = mesh_report(out_vertices, out_faces)
    if require_watertight and not interim["watertight"]:
        out_vertices, out_faces = weld_to_manifold(out_vertices, out_faces)
        report["steps"].append({"step": "weld_to_manifold"})
        interim = mesh_report(out_vertices, out_faces)
        if not interim["watertight"]:
            raise ValueError("mesh is not watertight after manifold welding")

    report["after"] = interim
    return out_vertices, out_faces, report
