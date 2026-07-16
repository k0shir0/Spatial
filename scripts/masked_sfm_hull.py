#!/usr/bin/env python3
"""Reconstruct a handheld object via masked SfM + point/silhouette hull.

Zero-LLM pipeline for captures where the object rotates in front of a static
camera: object masks restrict classical SIFT features to the object, COLMAP's
seeded CPU mapper recovers object-relative camera poses, and geometry is the
intersection of (a) the inflated convex hull of well-tracked triangulated
surface points and (b) silhouette carving from the posed masks.  Every posed
model is validated by reprojection IoU against the masks and the build fails
closed below the acceptance threshold.

Honest limits: output scale is ambiguous (no board/marker), concavities are
not recovered (convex-hull + silhouette bound), and grip changes mid-clip can
fragment pose tracking into arcs — the best-validating arc wins.

Requires the sfm extra (pycolmap, scipy) plus masks from auto_masks.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from local3d.backends import write_glb_mesh  # noqa: E402
from local3d.visual_hull import taubin_smooth  # noqa: E402


def _project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, intrinsics: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    focal, center_x, center_y, radial = intrinsics
    camera_points = points @ rotation.T + translation
    depth = camera_points[:, 2]
    valid = depth > 1e-6
    safe_depth = np.where(valid, depth, 1.0)
    x_normalized = camera_points[:, 0] / safe_depth
    y_normalized = camera_points[:, 1] / safe_depth
    distortion = 1.0 + radial * (x_normalized**2 + y_normalized**2)
    with np.errstate(invalid="ignore"):
        u = np.rint(focal * x_normalized * distortion + center_x)
        v = np.rint(focal * y_normalized * distortion + center_y)
    u = np.nan_to_num(u, nan=-1.0).astype(np.int64)
    v = np.nan_to_num(v, nan=-1.0).astype(np.int64)
    return u, v, valid


def _load_view(rec: "object", image: "object", masks_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mask_path = masks_dir / f"{Path(image.name).stem}_mask.png"
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    pose = image.cam_from_world()
    return mask > 127, pose.rotation.matrix(), np.asarray(pose.translation)


def evaluate_and_carve(
    rec: "object",
    masks_dir: Path,
    *,
    resolution: int,
    inflate: float,
    max_view_violations: int | None,
    min_track_length: int,
) -> dict | None:
    from scipy.spatial import Delaunay

    points_3d = np.array(
        [p.xyz for p in rec.points3D.values() if p.track.length() >= min_track_length]
    )
    if len(points_3d) < 50:
        return None
    lower = np.percentile(points_3d, 1, axis=0)
    upper = np.percentile(points_3d, 99, axis=0)
    pad = 0.10 * (upper - lower)
    lower, upper = lower - pad, upper + pad

    camera = list(rec.cameras.values())[0]
    intrinsics = tuple(float(value) for value in camera.params)

    axes = [np.linspace(lower[i], upper[i], resolution) for i in range(3)]
    grid_z, grid_y, grid_x = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    voxels = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))

    centroid = points_3d.mean(axis=0)
    inflated = centroid + (points_3d - centroid) * inflate
    hull_vertices = inflated[np.unique(Delaunay(inflated).convex_hull.ravel())]
    occupied = Delaunay(hull_vertices).find_simplex(voxels) >= 0
    if not occupied.any():
        return None

    views = []
    for image in sorted(rec.images.values(), key=lambda item: item.name):
        view = _load_view(rec, image, masks_dir)
        if view is not None:
            views.append(view)
    if len(views) < 5:
        return None
    if max_view_violations is None:
        max_view_violations = max(1, round(0.15 * len(views)))

    violations = np.zeros(len(voxels), dtype=np.uint16)
    for binary, rotation, translation in views:
        height, width = binary.shape
        index = np.flatnonzero(occupied)
        u, v, valid = _project(voxels[index], rotation, translation, intrinsics)
        inside = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        fails = np.zeros(len(index), dtype=bool)
        fails[inside] = ~binary[v[inside], u[inside]]
        failed = index[fails]
        violations[failed] += 1
        occupied[failed[violations[failed] > max_view_violations]] = False
        if not occupied.any():
            return None

    surviving = voxels[occupied]
    ious = []
    for binary, rotation, translation in views:
        height, width = binary.shape
        u, v, valid = _project(surviving, rotation, translation, intrinsics)
        inside = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        silhouette = np.zeros((height, width), dtype=np.uint8)
        silhouette[v[inside], u[inside]] = 1
        silhouette = cv2.morphologyEx(
            cv2.dilate(silhouette, np.ones((7, 7), np.uint8)),
            cv2.MORPH_CLOSE,
            np.ones((15, 15), np.uint8),
        )
        intersection = float(np.logical_and(silhouette > 0, binary).sum())
        union = float(np.logical_or(silhouette > 0, binary).sum())
        ious.append(intersection / max(union, 1.0))

    return {
        "occupancy": occupied.reshape((resolution, resolution, resolution)),
        "bounds": (lower, upper),
        "median_iou": float(np.median(ious)),
        "p10_iou": float(np.percentile(ious, 10)),
        "view_count": len(views),
        "point_count": int(len(points_3d)),
        "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
    }


def occupancy_to_world_mesh(occupancy: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    from skimage.measure import marching_cubes

    lower, upper = bounds
    vertices_zyx, faces, _normals, _values = marching_cubes(
        np.pad(occupancy.astype(np.uint8), 1), level=0.5
    )
    vertices_zyx -= 1.0
    shape = np.asarray(occupancy.shape, dtype=np.float64)
    vertices = vertices_zyx[:, ::-1] / (shape[::-1] - 1.0)
    return (lower + vertices * (upper - lower)).astype(np.float32), faces.astype(np.int32)


def render_qa_views(vertices: np.ndarray, faces: np.ndarray, output: Path, size: int = 480) -> None:
    """Small CPU painter-sort renders for review; no renderer dependency."""

    directions = {
        "front": np.eye(3),
        "side": np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
        "top": np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]),
        "iso": None,
    }
    iso = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
    tiles = []
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    light = np.array([-0.4, 0.6, 1.0]); light /= np.linalg.norm(light)
    for name, basis in directions.items():
        if basis is None:
            forward = iso
            right = np.cross([0.0, 1.0, 0.0], forward); right /= np.linalg.norm(right)
            up = np.cross(forward, right)
            basis = np.stack((right, up, forward))
        projected = (vertices - center) @ basis.T
        span = max(float(np.ptp(projected[:, :2], axis=0).max()), 1e-9)
        scale = size * 0.8 / span
        xy = projected[:, :2] * scale + size / 2.0
        depth = projected[:, 2]
        canvas = np.full((size, size, 3), 245, dtype=np.uint8)
        tri_xy = xy[faces]
        tri_depth = depth[faces].mean(axis=1)
        order = np.argsort(tri_depth)
        edge1 = vertices[faces[:, 1]] - vertices[faces[:, 0]]
        edge2 = vertices[faces[:, 2]] - vertices[faces[:, 0]]
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-12)
        shade = np.clip(np.abs(normals @ light), 0.15, 1.0)
        for tri_index in order:
            intensity = int(90 + 140 * shade[tri_index])
            polygon = tri_xy[tri_index].astype(np.int32)
            cv2.fillConvexPoly(canvas, polygon, (intensity, int(intensity * 0.86), int(intensity * 0.62)))
        cv2.putText(canvas, name, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        tiles.append(canvas)
    sheet = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    cv2.imwrite(str(output), sheet)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path, help="directory of extracted frames")
    parser.add_argument("masks", type=Path, help="masks directory from auto_masks.py (…/masks)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=144)
    parser.add_argument("--inflate", type=float, default=1.08, help="point-hull inflation about the centroid")
    parser.add_argument(
        "--max-view-violations", type=int, default=None,
        help="views that may disagree before a region is carved away; default scales "
        "to ~15%% of posed views so transient hand occlusion cannot bite into the object",
    )
    parser.add_argument("--min-track-length", type=int, default=3)
    parser.add_argument("--min-median-iou", type=float, default=0.5, help="acceptance gate")
    parser.add_argument("--target-triangles", type=int, default=2500, help="0 disables decimation")
    parser.add_argument("--max-num-features", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--match-threads", type=int, default=1, help=">1 speeds matching but weakens strict determinism")
    args = parser.parse_args()

    try:
        import pycolmap
    except ImportError:
        parser.error("pycolmap is not installed; install the sfm extra: pip install -e '.[sfm]'")

    frame_paths = sorted(args.frames.glob("*.jpg")) + sorted(args.frames.glob("*.png"))
    if len(frame_paths) < 10:
        parser.error(f"need at least 10 frames in {args.frames}")

    work = args.output
    work.mkdir(parents=True, exist_ok=True)
    colmap_masks = work / "colmap_masks"
    colmap_masks.mkdir(exist_ok=True)
    staged = 0
    for frame in frame_paths:
        mask = args.masks / f"{frame.stem}_mask.png"
        if mask.is_file():
            shutil.copy2(mask, colmap_masks / f"{frame.name}.png")
            staged += 1
    if staged < len(frame_paths):
        print(f"warning: {len(frame_paths) - staged} frames have no mask and will contribute no features", file=sys.stderr)

    pycolmap.set_random_seed(args.seed)
    database = work / "database.db"
    if database.exists():
        database.unlink()

    reader = pycolmap.ImageReaderOptions()
    reader.mask_path = str(colmap_masks)
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.num_threads = 1
    extraction.use_gpu = False
    extraction.sift.max_num_features = args.max_num_features
    extraction.sift.estimate_affine_shape = True
    extraction.sift.domain_size_pooling = True
    pycolmap.extract_features(
        database_path=str(database),
        image_path=str(args.frames),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = args.match_threads
    matching.use_gpu = False
    pycolmap.match_exhaustive(database_path=str(database), matching_options=matching)

    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.num_threads = 1
    mapper_options.random_seed = args.seed
    mapper_options.min_num_matches = 8
    mapper_options.mapper.init_min_num_inliers = 50
    mapper_options.mapper.abs_pose_min_num_inliers = 15
    mapper_options.mapper.init_min_tri_angle = 8.0
    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database),
        image_path=str(args.frames),
        output_path=str(work / "reconstruction"),
        options=mapper_options,
    )
    if not reconstructions:
        print("no camera poses could be recovered; needs_recapture", file=sys.stderr)
        return 1

    candidates = []
    for model_index, rec in reconstructions.items():
        result = evaluate_and_carve(
            rec,
            args.masks,
            resolution=args.resolution,
            inflate=args.inflate,
            max_view_violations=args.max_view_violations,
            min_track_length=args.min_track_length,
        )
        if result is not None:
            result["model_index"] = int(model_index)
            result["registered_images"] = rec.num_reg_images()
            candidates.append(result)
    if not candidates:
        print("no posed model produced a valid hull; needs_recapture", file=sys.stderr)
        return 1

    best = max(candidates, key=lambda item: item["median_iou"])
    vertices, faces = occupancy_to_world_mesh(best.pop("occupancy"), best.pop("bounds"))
    vertices = taubin_smooth(vertices, faces, iterations=6)

    post_report = None
    if args.target_triangles:
        try:
            from local3d.mesh_post import postprocess

            vertices, faces, post_report = postprocess(
                vertices, faces, target_triangles=args.target_triangles
            )
        except (ImportError, RuntimeError) as exc:
            print(f"warning: decimation skipped ({exc})", file=sys.stderr)

    glb_path = work / "reconstruction.glb"
    write_glb_mesh(
        glb_path,
        vertices.tolist(),
        faces.tolist(),
        generator="local3d-masked-sfm-hull",
        extras={
            "scale": "ambiguous",
            "source": "masked_sfm_point_hull_with_silhouette_carving",
            "geometry": "convex-bounded observed surface; concavities and unobserved faces are not measured",
        },
    )
    render_qa_views(vertices, faces, work / "qa_views.png")

    report = {
        "tool": "masked_sfm_hull (pycolmap CPU SIFT + point/silhouette hull, zero-LLM)",
        "seed": args.seed,
        "frames_in": len(frame_paths),
        "models": [
            {key: value for key, value in candidate.items() if key not in {"occupancy", "bounds"}}
            for candidate in candidates
        ],
        "selected_model": best["model_index"],
        "median_reprojection_iou": best["median_iou"],
        "accepted": best["median_iou"] >= args.min_median_iou,
        "triangles": int(len(faces)),
        "post_processing": post_report,
        "artifacts": {"glb": str(glb_path), "qa_views": str(work / "qa_views.png")},
        "limits": [
            "scale is ambiguous (no marker or measured reference)",
            "geometry is a convex/silhouette bound of the observed arc, not a full scan",
            "masks are review input; hand pixels inside masks can distort silhouettes",
        ],
    }
    (work / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("selected_model", "median_reprojection_iou", "accepted", "triangles")}, indent=2))
    if not report["accepted"]:
        print(f"median IoU below {args.min_median_iou}; needs_recapture", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
