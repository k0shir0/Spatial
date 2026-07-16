"""Synthetic evidence tests for the backend-independent reconstruction gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.recon_common import make_view, rasterize_zbuffer  # noqa: E402
from local3d.reconstruction_gate import assess_reconstruction  # noqa: E402


IMAGE_SIZE = 144
INTRINSICS = (170.0, IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0, 0.0)


def _mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(1.4, 1.0, 0.8)).subdivide()


def _look_at(center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=np.float64)
    forward = -center / np.linalg.norm(center)
    world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward))
    return rotation, -rotation @ center


def _views(
    mesh: trimesh.Trimesh,
    angles_degrees: np.ndarray,
    *,
    images: bool = True,
    depths: bool = True,
    pose_source: str = "sfm",
) -> list[dict]:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    palette = np.column_stack(
        (
            50 + np.arange(len(faces)) * 31 % 170,
            60 + np.arange(len(faces)) * 47 % 160,
            70 + np.arange(len(faces)) * 59 % 150,
        )
    ).astype(np.uint8)
    result: list[dict] = []
    for index, angle in enumerate(angles_degrees):
        radians = np.radians(float(angle))
        center = np.array([3.6 * np.sin(radians), 0.9, 3.6 * np.cos(radians)])
        rotation, translation = _look_at(center)
        zbuffer, face_index = rasterize_zbuffer(
            vertices,
            faces,
            rotation,
            translation,
            INTRINSICS,
            IMAGE_SIZE,
            IMAGE_SIZE,
            scale=1.0,
        )
        foreground = np.isfinite(zbuffer)
        extras: dict[str, object] = {
            "pose_source": pose_source,
            "used_for_geometry": index % 4 != 0,
        }
        if images:
            image = np.full((IMAGE_SIZE, IMAGE_SIZE, 3), 18, dtype=np.uint8)
            image[foreground] = palette[face_index[foreground]]
            extras["image_bgr"] = image
        if depths:
            extras["depth_map"] = np.where(foreground, zbuffer, 0.0).astype(np.float32)
        result.append(
            make_view(
                name=f"view_{index:03d}",
                image_path=Path(f"not_read_{index:03d}.png"),
                rotation=rotation,
                translation=translation,
                mask_tight=foreground,
                extras=extras,
            )
        )
    return result


def _assessment(mesh: trimesh.Trimesh, views: list[dict], **kwargs: object) -> dict:
    return assess_reconstruction(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        views,
        INTRINSICS,
        total_frame_count=len(views),
        sfm_registered_fraction=1.0,
        **kwargs,
    )


def test_supported_full_orbit_passes_and_report_is_strict_json() -> None:
    mesh = _mesh()
    views = _views(mesh, np.linspace(0.0, 360.0, 12, endpoint=False))

    report = _assessment(mesh, views)

    assert report["accepted"], report["reason_details"]
    assert report["reasons"] == []
    assert report["metrics"]["pose"]["direction_coverage_degrees"] > 300.0
    assert report["metrics"]["mask_agreement"]["median_iou"] > 0.95
    assert report["metrics"]["texture_support"]["observed_surface_fraction"] > 0.70
    assert report["metrics"]["depth_support"]["depth_map_views"] == len(views)
    assert report["metrics"]["geometric_evidence_mode"] == "aligned_depth_maps"
    # allow_nan=False catches NumPy scalars and accidental Infinity/NaN values.
    json.dumps(report, allow_nan=False, sort_keys=True)


def test_degenerate_camera_cluster_is_rejected_despite_perfect_masks_and_depth() -> None:
    mesh = _mesh()
    views = _views(mesh, np.linspace(-8.0, 8.0, 12))

    report = _assessment(mesh, views)

    assert not report["accepted"]
    assert "degenerate_camera_sweep" in report["reasons"]
    assert "insufficient_opposing_views" in report["reasons"]
    assert report["metrics"]["mask_agreement"]["median_iou"] > 0.95
    assert report["metrics"]["pose"]["direction_coverage_degrees"] < 30.0


def test_good_silhouettes_without_observed_appearance_or_depth_fail_closed() -> None:
    mesh = _mesh()
    views = _views(
        mesh,
        np.linspace(0.0, 360.0, 12, endpoint=False),
        images=False,
        depths=False,
    )

    report = _assessment(mesh, views)

    assert not report["accepted"]
    assert "insufficient_source_texture_views" in report["reasons"]
    assert "insufficient_observed_texture_surface" in report["reasons"]
    assert "insufficient_multiview_texture_support" in report["reasons"]
    assert "insufficient_geometric_depth_support" in report["reasons"]
    assert report["metrics"]["mask_agreement"]["median_iou"] > 0.95
    assert report["metrics"]["geometric_evidence_mode"] == "unsupported"


def test_spatially_supported_sfm_points_can_supply_depth_evidence() -> None:
    mesh = _mesh()
    views = _views(
        mesh,
        np.linspace(0.0, 360.0, 12, endpoint=False),
        depths=False,
    )
    x, y, z = np.meshgrid(
        np.linspace(-0.55, 0.55, 5),
        np.linspace(-0.38, 0.38, 5),
        np.linspace(-0.28, 0.28, 5),
        indexing="ij",
    )
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    report = _assessment(mesh, views, sfm_points=points)

    assert report["accepted"], report["reason_details"]
    assert report["metrics"]["sfm_points"]["passed"]
    assert report["metrics"]["geometric_evidence_mode"] == "sfm_points"


def test_wrong_delivery_mesh_is_rejected_by_source_view_agreement() -> None:
    truth = _mesh()
    views = _views(truth, np.linspace(0.0, 360.0, 12, endpoint=False))
    wrong = truth.copy()
    wrong.vertices = np.asarray(wrong.vertices) * np.array([1.70, 0.72, 0.62])

    report = _assessment(wrong, views)

    assert not report["accepted"]
    assert any(
        reason in report["reasons"]
        for reason in (
            "weak_median_silhouette_agreement",
            "weak_tail_silhouette_agreement",
            "too_few_silhouette_consistent_views",
            "weak_holdout_silhouette_agreement",
        )
    )
    assert report["metrics"]["mask_agreement"]["median_iou"] < 0.70


def test_silhouette_inferred_pose_ring_is_not_independent_pose_evidence() -> None:
    mesh = _mesh()
    views = _views(
        mesh,
        np.linspace(0.0, 360.0, 12, endpoint=False),
        pose_source="silhouette",
    )

    report = _assessment(mesh, views, pose_mode="turntable")

    assert not report["accepted"]
    assert "insufficient_independent_pose_evidence" in report["reasons"]
    assert report["metrics"]["pose"]["independent_pose_views"] == 0
    # This isolates the evidence problem: assigned poses can still fit perfectly.
    assert report["metrics"]["mask_agreement"]["median_iou"] > 0.95
