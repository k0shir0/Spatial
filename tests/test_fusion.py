from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d import fusion
from local3d.recon_common import make_view, rasterize_zbuffer

INTRINSICS = (300.0, 100.0, 100.0, 0.0)
IMAGE_W = IMAGE_H = 200


def _look_at_view(name, center, mask=None, depth=None, target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)):
    center = np.asarray(center, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    translation = -rotation @ center
    return make_view(
        name=name,
        image_path=Path(name),
        rotation=rotation,
        translation=translation,
        mask_tight=(None if mask is None else mask.astype(np.uint8)),
        extras={"depth_map": depth, "depth_conf": 1.0},
    )


def _ring_views(mesh, rdist, *, intrinsics=INTRINSICS, n=12):
    """Twelve cameras on a circle in the xz-plane, looking at the origin."""
    views = []
    for k in range(n):
        theta = 2.0 * np.pi * k / n
        center = (rdist * np.sin(theta), 0.0, rdist * np.cos(theta))
        pose = _look_at_view(f"view{k:03d}", center)
        zbuffer, _ = rasterize_zbuffer(
            mesh.vertices, mesh.faces, pose["rotation"], pose["translation"],
            intrinsics, IMAGE_W, IMAGE_H, scale=1.0,
        )
        finite = np.isfinite(zbuffer)
        depth = np.where(finite, zbuffer, 0.0).astype(np.float32)
        views.append(_look_at_view(f"view{k:03d}", center, mask=finite, depth=depth))
    return views


def _dented_sphere(resolution=128, radius=1.0, dent_center_z=1.15, dent_radius=0.55):
    axis = np.linspace(-1.3, 1.3, resolution)
    grid_z, grid_y, grid_x = np.meshgrid(axis, axis, axis, indexing="ij")
    dist_origin = np.sqrt(grid_x**2 + grid_y**2 + grid_z**2)
    dist_dent = np.sqrt(grid_x**2 + grid_y**2 + (grid_z - dent_center_z) ** 2)
    inside = (dist_origin <= radius) & ~(dist_dent <= dent_radius)
    verts, faces = fusion._marching_to_world(
        inside.astype(np.uint8), ([-1.3, -1.3, -1.3], [1.3, 1.3, 1.3]),
        level=0.5, pad_value=0.0,
    )
    return trimesh.Trimesh(verts, faces, process=False)


def test_carve_hull_and_extract_recovers_sphere_volume():
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    views = _ring_views(sphere, rdist=4.0)

    bounds = fusion.bounds_from_points(np.asarray(sphere.vertices))
    occupancy, report = fusion.carve_hull(views, INTRINSICS, bounds, resolution=96)
    assert report["occupied_voxels"] > 0

    result = fusion.reconstruct_geometry(
        views, INTRINSICS, np.asarray(sphere.vertices), None,
        resolution=96, target_triangles=0,
    )
    assert result["report"]["mode"] == "hull_only"
    mesh = trimesh.Trimesh(result["vertices"], result["faces"], process=False)

    true_volume = 4.0 / 3.0 * np.pi * 1.0**3
    assert 0.75 * true_volume < mesh.volume < 1.25 * true_volume

    lower, upper = bounds
    assert (result["vertices"] >= lower - 1e-4).all()
    assert (result["vertices"] <= upper + 1e-4).all()


def test_tsdf_recovers_concavity_the_hull_cannot_see():
    dented = _dented_sphere(resolution=128)
    views = _ring_views(dented, rdist=3.0)
    depths = [view["depth_map"] for view in views]

    bounds = fusion.bounds_from_points(np.asarray(dented.vertices))
    occupancy, _ = fusion.carve_hull(views, INTRINSICS, bounds, resolution=88)
    hull_verts, _ = fusion._marching_to_world(
        occupancy.astype(np.uint8), bounds, level=0.5, pad_value=0.0
    )

    tsdf, weight = fusion.fuse_tsdf(views, INTRINSICS, depths, occupancy, bounds)
    fused_verts, _, extract_report = fusion.extract_mesh(tsdf, weight, occupancy, bounds)
    assert extract_report["observed_band_voxels"] > 0

    def cone_distance(verts):
        axial = verts[:, 2]
        radial = np.sqrt(verts[:, 0] ** 2 + verts[:, 1] ** 2)
        cone = verts[(axial > 0.5) & (radial < 0.45)]
        assert len(cone) > 0
        cone = cone[:: max(len(cone) // 300, 1)]  # deterministic subsample
        _closest, distance, _tid = trimesh.proximity.closest_point_naive(dented, cone)
        return float(np.mean(distance))

    hull_error = cone_distance(hull_verts)
    fused_error = cone_distance(fused_verts)
    # The hull fills the dent; fusion must pull the surface at least 2x closer.
    assert fused_error * 2.0 <= hull_error


def test_hull_sdf_and_tsdf_share_sign_convention():
    resolution = 32
    bounds = (np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0]))
    axis = np.linspace(-1.0, 1.0, resolution)

    # Occupancy = the half-space world-z >= 0 (indexed [z, y, x]).
    occupancy = np.zeros((resolution, resolution, resolution), dtype=bool)
    occupancy[axis >= 0.0, :, :] = True

    hull = fusion.hull_sdf(occupancy)

    center = resolution // 2
    inside_iz = center + 2   # world z ~ +0.16, inside the block
    outside_iz = center - 2  # world z ~ -0.10, in front of the block
    assert hull[inside_iz, center, center] < 0.0   # inside -> negative
    assert hull[outside_iz, center, center] > 0.0  # outside -> positive

    # One camera at (0, 0, -4) looking +z sees the block's -z face at world z=0.
    focal = 64.0
    intr = (focal, 64.0, 64.0, 0.0)
    view = _look_at_view("solo", (0.0, 0.0, -4.0), target=(0.0, 0.0, 1.0))
    depth_map = np.full((128, 128), 4.0, dtype=np.float32)  # surface at world z=0
    tsdf, weight = fusion.fuse_tsdf([view], intr, [depth_map], occupancy, bounds)

    assert weight[inside_iz, center, center] > 0.0
    assert weight[outside_iz, center, center] > 0.0
    # Same convention as the hull: inside negative, outside positive.
    assert tsdf[inside_iz, center, center] < 0.0
    assert tsdf[outside_iz, center, center] > 0.0


def test_bounds_from_silhouettes_brackets_the_object():
    sphere = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    views = _ring_views(sphere, rdist=4.0)
    lower, upper = fusion.bounds_from_silhouettes(views, INTRINSICS, coarse=40)
    # The box must contain the unit sphere and stay bounded.
    assert (lower < -0.9).all() and (upper > 0.9).all()
    assert np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))
    assert (upper - lower).max() < 20.0
