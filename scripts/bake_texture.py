#!/usr/bin/env python3
"""Bake source-video pixels onto a reconstructed mesh (multiview, zero-ML).

For every texel: map UV -> surface point -> project into each posed source
frame; keep views where the surface faces the camera and the pixel lies
inside the object mask; take the channel-wise median of the most frontal
views (median resists finger/highlight outliers).  Texels no view observed
are filled with the mean observed color and reported as unobserved — they
are flat fill, not fabricated detail.

Inputs come from masked_sfm_hull.py: its mesh (same world frame) and its
COLMAP reconstruction directory.  Requires the sfm and mesh extras plus
xatlas for UV unwrapping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402


def load_views(reconstruction_dir: Path, frames_dir: Path, masks_dir: Path) -> tuple[list[dict], tuple[float, ...]]:
    import pycolmap

    rec = pycolmap.Reconstruction(str(reconstruction_dir))
    camera = list(rec.cameras.values())[0]
    intrinsics = tuple(float(value) for value in camera.params)
    views = []
    for image in sorted(rec.images.values(), key=lambda item: item.name):
        frame = cv2.imread(str(frames_dir / image.name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(masks_dir / f"{Path(image.name).stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if frame is None or mask is None:
            continue
        pose = image.cam_from_world()
        rotation = pose.rotation.matrix()
        translation = np.asarray(pose.translation)
        views.append(
            {
                "name": image.name,
                "frame": frame,
                "mask": mask > 127,
                "rotation": rotation,
                "translation": translation,
                "center": np.asarray(image.projection_center()),
            }
        )
    return views, intrinsics


def project(points: np.ndarray, view: dict, intrinsics: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    focal, center_x, center_y, radial = intrinsics
    camera_points = points @ view["rotation"].T + view["translation"]
    depth = camera_points[:, 2]
    valid = depth > 1e-6
    safe = np.where(valid, depth, 1.0)
    x_normalized = camera_points[:, 0] / safe
    y_normalized = camera_points[:, 1] / safe
    distortion = 1.0 + radial * (x_normalized**2 + y_normalized**2)
    u = focal * x_normalized * distortion + center_x
    v = focal * y_normalized * distortion + center_y
    return u, v, valid


def bake(
    mesh: trimesh.Trimesh,
    views: list[dict],
    intrinsics: tuple[float, ...],
    *,
    texture_size: int,
    top_k: int,
    min_frontality: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    import xatlas

    vmapping, indices, uvs = xatlas.parametrize(
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.uint32),
    )
    vertices = np.asarray(mesh.vertices, dtype=np.float64)[vmapping]
    faces = indices.astype(np.int64)

    face_vertices = vertices[faces]
    normals = np.cross(
        face_vertices[:, 1] - face_vertices[:, 0],
        face_vertices[:, 2] - face_vertices[:, 0],
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    centroid = vertices.mean(axis=0)
    # Hull faces should point outward; flip any that do not.
    outward = np.einsum("ij,ij->i", normals, face_vertices.mean(axis=1) - centroid)
    normals[outward < 0] *= -1.0

    size = texture_size
    texel_points = np.full((size * size, 3), np.nan)
    texel_normals = np.zeros((size * size, 3))
    uv_pixels = np.column_stack((uvs[:, 0] * (size - 1), (1.0 - uvs[:, 1]) * (size - 1)))
    for face_index, face in enumerate(faces):
        triangle = uv_pixels[face]
        min_xy = np.floor(triangle.min(axis=0)).astype(int)
        max_xy = np.ceil(triangle.max(axis=0)).astype(int)
        xs = np.arange(max(min_xy[0], 0), min(max_xy[0] + 1, size))
        ys = np.arange(max(min_xy[1], 0), min(max_xy[1] + 1, size))
        if not len(xs) or not len(ys):
            continue
        grid_x, grid_y = np.meshgrid(xs, ys)
        pixels = np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float64)
        origin, edge_b, edge_c = triangle[0], triangle[1] - triangle[0], triangle[2] - triangle[0]
        denom = edge_b[0] * edge_c[1] - edge_b[1] * edge_c[0]
        if abs(denom) < 1e-12:
            continue
        relative = pixels - origin
        w_b = (relative[:, 0] * edge_c[1] - relative[:, 1] * edge_c[0]) / denom
        w_c = (relative[:, 1] * edge_b[0] - relative[:, 0] * edge_b[1]) / denom
        w_a = 1.0 - w_b - w_c
        inside = (w_a >= -0.001) & (w_b >= -0.001) & (w_c >= -0.001)
        if not inside.any():
            continue
        pix = pixels[inside].astype(int)
        flat = pix[:, 1] * size + pix[:, 0]
        weights = np.column_stack((w_a[inside], w_b[inside], w_c[inside]))
        texel_points[flat] = weights @ vertices[face]
        texel_normals[flat] = normals[face_index]

    observed_mask = ~np.isnan(texel_points[:, 0])
    active = np.flatnonzero(observed_mask)
    points = texel_points[active]
    normals_active = texel_normals[active]

    samples = np.full((len(active), top_k, 3), np.nan)
    scores = np.full((len(active), top_k), -1.0)
    for view in views:
        directions = view["center"] - points
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
        frontality = np.einsum("ij,ij->i", normals_active, directions)
        u, v, valid = project(points, view, intrinsics)
        height, width = view["mask"].shape
        ui = np.clip(np.rint(u), 0, width - 1).astype(np.int64)
        vi = np.clip(np.rint(v), 0, height - 1).astype(np.int64)
        usable = (
            valid
            & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
            & (frontality > min_frontality)
            & view["mask"][vi, ui]
        )
        if not usable.any():
            continue
        colors = view["frame"][vi, ui].astype(np.float64)
        slot = np.argmin(scores, axis=1)
        better = usable & (frontality > scores[np.arange(len(active)), slot])
        rows = np.flatnonzero(better)
        scores[rows, slot[rows]] = frontality[rows]
        samples[rows, slot[rows]] = colors[rows]

    texture = np.zeros((size * size, 3))
    sampled = scores.max(axis=1) > 0
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(np.where(scores[..., None] > 0, samples, np.nan), axis=1)
    texture[active[sampled]] = median[sampled]
    observed_fraction = float(sampled.mean()) if len(active) else 0.0
    fill = median[sampled].mean(axis=0) if sampled.any() else np.array([128.0, 128.0, 128.0])
    texture[active[~sampled]] = fill
    texture = texture.reshape((size, size, 3)).astype(np.uint8)
    # Dilate colors into empty gutter texels so bilinear sampling has support.
    coverage = np.zeros((size, size), np.uint8)
    coverage.ravel()[active] = 1
    for _ in range(4):
        grown = cv2.dilate(texture, np.ones((3, 3), np.uint8))
        expand = cv2.dilate(coverage, np.ones((3, 3), np.uint8))
        texture = np.where((coverage == 0)[..., None] & (expand == 1)[..., None], grown, texture)
        coverage = expand
    return vertices.astype(np.float32), faces.astype(np.int32), uvs.astype(np.float32), texture, observed_fraction


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="reconstruction.glb from masked_sfm_hull.py")
    parser.add_argument("reconstruction", type=Path, help="COLMAP model directory (…/reconstruction/<index>)")
    parser.add_argument("frames", type=Path)
    parser.add_argument("masks", type=Path, help="masks directory from auto_masks.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--top-views", type=int, default=5)
    parser.add_argument("--min-frontality", type=float, default=0.20)
    args = parser.parse_args()

    mesh = trimesh.load(args.mesh, force="mesh")
    views, intrinsics = load_views(args.reconstruction, args.frames, args.masks)
    if len(views) < 3:
        parser.error("fewer than three posed views available for baking")

    vertices, faces, uvs, texture_bgr, observed = bake(
        mesh,
        views,
        intrinsics,
        texture_size=args.texture_size,
        top_k=args.top_views,
        min_frontality=args.min_frontality,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    texture_path = args.output / "texture.png"
    cv2.imwrite(str(texture_path), texture_bgr)

    from PIL import Image

    textured = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    textured.visual = trimesh.visual.TextureVisuals(
        uv=uvs,
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture_path), diffuse=[255, 255, 255, 255]
        ),
    )
    glb_path = args.output / "textured.glb"
    textured.export(glb_path)

    report = {
        "tool": "bake_texture (multiview median projection, zero-ML)",
        "views_used": len(views),
        "texture_size": args.texture_size,
        "observed_texel_fraction": round(observed, 4),
        "unobserved_fill": "mean observed color (flat fill, not fabricated detail)",
        "artifacts": {"glb": str(glb_path), "texture": str(texture_path)},
    }
    (args.output / "bake_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
