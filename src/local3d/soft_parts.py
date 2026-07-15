"""Dependency-light meshes for fitted soft-object assemblies."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def rotation_matrix(euler_degrees: Sequence[float]) -> np.ndarray:
    x, y, z = np.deg2rad(np.asarray(euler_degrees, dtype=np.float64))
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx


def ellipsoid_mesh(
    center: Sequence[float], radii: Sequence[float], *,
    euler_degrees: Sequence[float] = (0, 0, 0), rings: int = 20, segments: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rings < 4 or segments < 8 or any(float(value) <= 0 for value in radii):
        raise ValueError("invalid ellipsoid tessellation or radii")
    vertices = [[0.0, 1.0, 0.0]]
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        for segment in range(segments):
            theta = 2 * math.pi * segment / segments
            vertices.append([math.sin(phi) * math.cos(theta), math.cos(phi), math.sin(phi) * math.sin(theta)])
    vertices.append([0.0, -1.0, 0.0])
    top, bottom = 0, len(vertices) - 1
    faces: list[tuple[int, int, int]] = []
    for segment in range(segments):
        current, following = 1 + segment, 1 + (segment + 1) % segments
        faces.append((top, following, current))
    for ring in range(rings - 2):
        start, following_start = 1 + ring * segments, 1 + (ring + 1) * segments
        for segment in range(segments):
            a, b = start + segment, start + (segment + 1) % segments
            c, d = following_start + segment, following_start + (segment + 1) % segments
            faces.extend(((a, b, c), (b, d, c)))
    last = 1 + (rings - 2) * segments
    for segment in range(segments):
        current, following = last + segment, last + (segment + 1) % segments
        faces.append((current, following, bottom))

    unit = np.asarray(vertices, dtype=np.float64)
    rotation = rotation_matrix(euler_degrees)
    scaled = unit * np.asarray(radii, dtype=np.float64)
    transformed = scaled @ rotation.T + np.asarray(center, dtype=np.float64)
    # Ellipsoid normals transform by inverse transpose of scale+rotation.
    normals = (unit / np.asarray(radii, dtype=np.float64)) @ rotation.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    return transformed.astype(np.float32), np.asarray(faces, dtype=np.int32), normals.astype(np.float32)


def superellipsoid_mesh(
    center: Sequence[float],
    radii: Sequence[float],
    *,
    vertical_exponent: float = 0.65,
    horizontal_exponent: float = 0.75,
    euler_degrees: Sequence[float] = (0, 0, 0),
    rings: int = 24,
    segments: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a rounded-box superellipsoid for pillow/loaf soft objects.

    Exponents below one flatten the broad faces while retaining rounded edges.
    The topology matches :func:`ellipsoid_mesh`, which keeps the texture seam
    and atlas code shared between the two primitives.
    """

    values = [float(value) for value in radii]
    if (
        rings < 4
        or segments < 8
        or any(value <= 0.0 for value in values)
        or not 0.2 <= float(vertical_exponent) <= 2.0
        or not 0.2 <= float(horizontal_exponent) <= 2.0
    ):
        raise ValueError("invalid superellipsoid tessellation, radii, or exponents")

    def signed_power(value: float, exponent: float) -> float:
        return math.copysign(abs(value) ** exponent, value)

    vertices = [[0.0, 1.0, 0.0]]
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        radial = math.sin(phi) ** vertical_exponent
        vertical = signed_power(math.cos(phi), vertical_exponent)
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            vertices.append(
                [
                    radial * signed_power(math.cos(theta), horizontal_exponent),
                    vertical,
                    radial * signed_power(math.sin(theta), horizontal_exponent),
                ]
            )
    vertices.append([0.0, -1.0, 0.0])

    top, bottom = 0, len(vertices) - 1
    faces: list[tuple[int, int, int]] = []
    for segment in range(segments):
        current, following = 1 + segment, 1 + (segment + 1) % segments
        faces.append((top, following, current))
    for ring in range(rings - 2):
        start, following_start = 1 + ring * segments, 1 + (ring + 1) * segments
        for segment in range(segments):
            a, b = start + segment, start + (segment + 1) % segments
            c, d = following_start + segment, following_start + (segment + 1) % segments
            faces.extend(((a, b, c), (b, d, c)))
    last = 1 + (rings - 2) * segments
    for segment in range(segments):
        current, following = last + segment, last + (segment + 1) % segments
        faces.append((current, following, bottom))

    unit = np.asarray(vertices, dtype=np.float64)
    rotation = rotation_matrix(euler_degrees)
    transformed = unit * np.asarray(values, dtype=np.float64)
    transformed = transformed @ rotation.T + np.asarray(center, dtype=np.float64)
    face_array = np.asarray(faces, dtype=np.int32)
    face_normals = np.cross(
        transformed[face_array[:, 1]] - transformed[face_array[:, 0]],
        transformed[face_array[:, 2]] - transformed[face_array[:, 0]],
    )
    normals = np.zeros_like(transformed)
    for corner in range(3):
        np.add.at(normals, face_array[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(lengths < 1e-12):
        raise ValueError("superellipsoid produced a degenerate vertex normal")
    normals /= lengths
    return transformed.astype(np.float32), face_array, normals.astype(np.float32)


def tube_mesh(
    points: Sequence[Sequence[float]], radius: float, *, segments: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray(points, dtype=np.float64)
    if len(centers) < 2 or radius <= 0 or segments < 6:
        raise ValueError("a tube needs at least two points, positive radius, and six segments")
    tangents = np.gradient(centers, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-12)
    rings: list[np.ndarray] = []
    ring_normals: list[np.ndarray] = []
    previous_normal = np.array([1.0, 0.0, 0.0])
    for center, tangent in zip(centers, tangents):
        normal = previous_normal - tangent * np.dot(previous_normal, tangent)
        if np.linalg.norm(normal) < 1e-6:
            reference = np.array([0.0, 1.0, 0.0]) if abs(tangent[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
            normal = np.cross(tangent, reference)
        normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        directions = np.asarray([
            math.cos(2 * math.pi * index / segments) * normal
            + math.sin(2 * math.pi * index / segments) * binormal
            for index in range(segments)
        ])
        rings.append(center + radius * directions)
        ring_normals.append(directions)
        previous_normal = normal
    vertices = np.vstack(rings)
    normals = np.vstack(ring_normals)
    faces: list[tuple[int, int, int]] = []
    for ring in range(len(centers) - 1):
        for segment in range(segments):
            a, b = ring * segments + segment, ring * segments + (segment + 1) % segments
            c, d = (ring + 1) * segments + segment, (ring + 1) * segments + (segment + 1) % segments
            faces.extend(((a, b, c), (b, d, c)))
    # Cap both ends.
    start_index, end_index = len(vertices), len(vertices) + 1
    vertices = np.vstack((vertices, centers[0], centers[-1]))
    normals = np.vstack((normals, -tangents[0], tangents[-1]))
    for segment in range(segments):
        faces.append((start_index, (segment + 1) % segments, segment))
        last = (len(centers) - 1) * segments
        faces.append((end_index, last + segment, last + (segment + 1) % segments))
    return vertices.astype(np.float32), np.asarray(faces, dtype=np.int32), normals.astype(np.float32)


def combine_parts(parts: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, Sequence[int]]]):
    vertices, faces, normals, colors = [], [], [], []
    offset = 0
    for part_vertices, part_faces, part_normals, color in parts:
        vertices.append(part_vertices)
        faces.append(part_faces + offset)
        normals.append(part_normals)
        rgba = list(color) + ([255] if len(color) == 3 else [])
        colors.append(np.tile(np.asarray(rgba, dtype=np.uint8), (len(part_vertices), 1)))
        offset += len(part_vertices)
    return np.vstack(vertices), np.vstack(faces), np.vstack(normals), np.vstack(colors)
