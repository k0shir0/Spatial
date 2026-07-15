#!/usr/bin/env python3
"""Hybrid evidence-gated video -> textured 3D (CPU-only, zero-LLM).

One command takes a continuous, static-camera clip of a hand-rotated object and
builds several auditable candidates. Object-centric masks feed deterministic
masked COLMAP SfM; independently recovered cameras may produce a silhouette
hull, optional SfM-aligned *predicted* monocular-depth fusion, and a source-view
texture atlas. Silhouette-completed cameras may help carving but are forbidden
from sourcing texture or counting as independent evidence.

The general mesh is promoted only after camera diversity, held-out/source-mask
reprojection, observed surface pixels, geometric support, topology, and the
reloaded GLB all pass. If the capture cannot support that result, conservative
rounded-slab or bilateral soft-volume fits may be selected with explicit
``parametric``/``inferred`` provenance. If none is supported, the command asks
for recapture instead of delivering plausible-looking garbage. Metric scale and
truly unobserved surfaces remain unknowable without additional capture evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _log(stage: str, message: str) -> None:
    print(f"[{stage}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_snapshot(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SystemExit(f"source video not found: {resolved}")
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "bytes": int(stat.st_size),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "modified_time_ns": int(stat.st_mtime_ns),
        "sha256": _sha256(resolved),
    }


def _verify_source_unchanged(path: Path, expected: dict) -> None:
    current = _source_snapshot(path)
    for key in ("bytes", "device", "inode", "modified_time_ns", "sha256"):
        if current[key] != expected[key]:
            raise SystemExit(f"source video changed during reconstruction ({key})")


def _prepare_output(path: Path, source: dict) -> None:
    marker = path / ".spatial-hybrid-output.json"
    if path.exists() and any(path.iterdir()):
        raise SystemExit(
            f"output directory is not empty: {path}; choose a new path so prior evidence is preserved"
        )
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / ".spatial-hybrid-output.json.tmp"
    temporary.write_text(
        json.dumps({"schema_version": 1, "source": source}, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(marker)


def stage_ingest(video: Path, out_dir: Path, *, sample_fps: float, max_frames: int) -> tuple[list[Path], object]:
    from local3d.ingest import IngestConfig, analyze_video, write_analysis_json

    config = IngestConfig(
        sample_fps=sample_fps,
        max_candidates=max_frames,
        keyframe_count=12,
        min_keyframe_gap_s=0.30,
        segmenter="none",
    )
    frames_dir = out_dir / "frames"
    analysis = analyze_video(str(video), frames_dir, config=config)
    write_analysis_json(analysis, out_dir / "analysis.json")
    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    _log("ingest", f"{len(frames)} frames -> {frames_dir}")
    return frames, analysis


def _use_object_anchor_for_general(anchor_report: dict, frame_count: int) -> bool:
    """Prefer a persistent object-specific route even when profiles are sparse."""

    accepted = int(anchor_report.get("acceptedFrames", 0))
    fraction = accepted / max(int(frame_count), 1)
    persistent_object_appearance = bool(
        anchor_report.get("persistentGreen") or anchor_report.get("persistentSoftPink")
    )
    return bool(
        (persistent_object_appearance and accepted >= 6)
        or fraction >= 0.55
    )


def stage_masks(
    frames: list[Path], out_dir: Path, *, model: str = "u2netp", depth_model: Path | None = None
) -> dict:
    from PIL import Image

    from local3d.mask_clean import clean_mask_sequence, erode_for_sfm
    from local3d.masking import load_cached_u2netp, refine_held_object_sequence

    session, remove, segmenter_provenance = load_cached_u2netp(model)
    raw_masks: list[np.ndarray] = []
    images: list[np.ndarray] = []
    for path in frames:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        alpha = remove(Image.open(path), session=session, only_mask=True, post_process_mask=False)
        mask = np.asarray(alpha, dtype=np.uint8)
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        raw_masks.append(mask >= 128)
        images.append(frame)

    disparities = None
    if depth_model is not None and Path(depth_model).is_file():
        from local3d.monodepth import DepthEstimator

        estimator = DepthEstimator(depth_model, threads=2)
        disparities = [estimator.disparity(image) for image in images]
        _log("masks", "depth-guided pruning enabled")

    cleaned_tight, _cleaned_eroded, report = clean_mask_sequence(
        raw_masks, images, disparities=disparities
    )
    cleaned_for_general: list[np.ndarray] = []
    general_exclusions: dict[str, list[int]] = {}
    for index, (mask, frame_report) in enumerate(zip(cleaned_tight, report["frames"])):
        flags = set(frame_report.get("flags", []))
        reasons: list[str] = []
        if "area_outlier" in flags:
            reasons.append("area_outlier")
        if float(frame_report.get("border_touch", 0.0)) > 0.01:
            reasons.append("border_touch")
        if int(frame_report.get("components", 0)) != 1:
            reasons.append("not_one_component")
        coverage = float(frame_report.get("coverage", 0.0))
        if coverage < 0.015 or coverage > 0.46:
            reasons.append("unsafe_coverage")
        if float(frame_report.get("depth_pruned_fraction", 0.0)) > 0.45:
            reasons.append("destructive_depth_prune")
        if reasons:
            cleaned_for_general.append(np.zeros_like(mask, dtype=bool))
            for reason in reasons:
                general_exclusions.setdefault(reason, []).append(index)
        else:
            cleaned_for_general.append(mask)
    anchored, anchor_report = refine_held_object_sequence(images, raw_masks)
    anchor_fraction = float(anchor_report["acceptedFrames"]) / max(len(frames), 1)

    # A clip-persistent object anchor wins only with majority support. Missing
    # anchor frames become empty masks; silently falling back to a person matte
    # on the difficult regrip/profile frames is what corrupted the prior runs.
    use_anchor_for_general = _use_object_anchor_for_general(anchor_report, len(frames))
    if use_anchor_for_general:
        tight = [
            (item > 0) if item is not None else np.zeros_like(raw_masks[index], dtype=bool)
            for index, item in enumerate(anchored)
        ]
    else:
        tight = cleaned_for_general
    eroded_sfm = [erode_for_sfm(mask) for mask in tight]

    # Prior/fallback routes may use the smaller conservative anchor subset. If
    # fewer than six survived, expose the generic cleaned sequence instead.
    prior_masks = (
        [(item > 0) if item is not None else None for item in anchored]
        if anchor_report["acceptedFrames"] >= 6
        else [mask for mask in cleaned_tight]
    )
    report["segmenter"] = segmenter_provenance
    report["object_anchor"] = anchor_report
    report["general_mask_selection"] = {
        "selected": "object_anchor" if use_anchor_for_general else "depth_cleaned_foreground",
        "anchor_success_fraction": round(anchor_fraction, 6),
        "minimum_anchor_success_fraction": 0.55,
        "persistent_object_appearance": bool(
            anchor_report.get("persistentGreen")
            or anchor_report.get("persistentSoftPink")
        ),
        "persistent_anchor_minimum_frames": 6,
        "missing_anchor_policy": "empty mask; never whole-frame or person-matte fallback",
        "excluded_cleaned_frames": general_exclusions,
    }

    tight_dir = out_dir / "masks_tight"
    eroded_dir = out_dir / "masks_eroded"
    sfm_dir = out_dir / "masks_sfm"
    prior_dir = out_dir / "masks_prior_candidates"
    overlay_dir = out_dir / "overlays"
    for directory in (tight_dir, eroded_dir, sfm_dir, prior_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for path, frame, tight_mask, sfm_mask, prior_mask in zip(
        frames, images, tight, eroded_sfm, prior_masks
    ):
        cv2.imwrite(str(tight_dir / f"{path.stem}_mask.png"), tight_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(eroded_dir / f"{path.stem}_mask.png"), sfm_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(sfm_dir / f"{path.stem}_mask.png"), sfm_mask.astype(np.uint8) * 255)
        if prior_mask is not None and np.any(prior_mask):
            cv2.imwrite(
                str(prior_dir / f"{path.stem}.png"),
                np.asarray(prior_mask, dtype=np.uint8) * 255,
            )
    # Review overlays for a few frames only (QA artifact, not operator gate).
    step = max(len(frames) // 12, 1)
    for path, frame, tight_mask in list(zip(frames, images, tight))[::step]:
        dimmed = (frame * 0.35).astype(np.uint8)
        overlay = np.where(tight_mask[..., None], frame, dimmed)
        cv2.imwrite(str(overlay_dir / f"{path.stem}_overlay.png"), overlay)

    (out_dir / "mask_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _log("masks", f"{len(tight)} masks cleaned -> {tight_dir}")
    return {
        "tight_dir": tight_dir,
        "eroded_dir": eroded_dir,
        "sfm_dir": sfm_dir,
        "prior_dir": prior_dir,
        "tight": tight,
        "eroded": eroded_sfm,
        "report": report,
    }


def stage_prior_inputs(video: Path, out_dir: Path, *, model: str = "u2netp") -> dict:
    """Reproduce the proven sparse-cadence prior-mask path independently.

    Dense sampling is useful for SfM but can overweight a long partial/profile
    pose in appearance-family selection. The regularized fitters therefore get
    their own deterministic 3 fps evidence stream, matching the cadence at
    which their gates and selection logic were validated.
    """

    from local3d.masking import generate_color_anchored_masks

    frames, _analysis = stage_ingest(
        video, out_dir / "ingest", sample_fps=3.0, max_frames=60
    )
    masks_dir = out_dir / "masks"
    review_dir = out_dir / "mask_review"
    report = generate_color_anchored_masks(
        frames[0].parent,
        masks_dir,
        review_dir,
        model_name=model,
    )
    accepted = len(report.get("frames", []))
    _log("prior-masks", f"{accepted} object-supported masks from {len(frames)} sparse frames")
    return {"frames": frames, "prior_dir": masks_dir, "report": report}


def load_mask_dir(directory: Path, frames: list[Path]) -> dict[str, np.ndarray]:
    masks = {}
    for frame in frames:
        path = directory / f"{frame.stem}_mask.png"
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks[frame.name] = mask > 127
    return masks


def stage_poses(
    frames: list[Path],
    mask_bundle: dict,
    work_dir: Path,
    *,
    seed: int,
    fov_deg: float,
    match_threads: int,
    min_registered_fraction: float,
) -> dict:
    from local3d.sfm_video import run_masked_sfm

    frames_dir = frames[0].parent
    result = run_masked_sfm(
        frames_dir,
        mask_bundle["sfm_dir"],
        work_dir,
        seed=seed,
        fov_deg=fov_deg,
        match_threads=match_threads,
    )
    fraction = float(result.get("registered_fraction") or 0.0)
    _log("poses", f"SfM registered fraction: {fraction:.2f}")

    if result.get("ok") and fraction >= min_registered_fraction:
        views = sorted(result["views"], key=lambda item: item["name"])
        tight = load_mask_dir(mask_bundle["tight_dir"], frames)
        eroded = load_mask_dir(mask_bundle["eroded_dir"], frames)
        for view in views:
            view["mask_tight"] = tight.get(view["name"])
            view["mask_eroded"] = eroded.get(view["name"])
            view["pose_source"] = "sfm"
            observations = result.get("track_evidence", {}).get("views", {}).get(
                view["name"]
            )
            if observations is not None:
                view["sfm_observations"] = observations
        views = [v for v in views if v["mask_tight"] is not None]

        # Keep a deterministic subset completely out of silhouette carving and
        # depth fusion.  These masks become a genuine held-out check of the
        # finished geometry instead of grading a hull against the same masks
        # that created it.
        holdout_count = min(max(3, len(views) // 6), max(0, len(views) - 8))
        holdout_names: set[str] = set()
        if holdout_count:
            indices = np.linspace(0, len(views) - 1, holdout_count + 2)[1:-1]
            holdout_names = {views[int(round(index))]["name"] for index in indices}
        for view in views:
            is_holdout = view["name"] in holdout_names
            view["evaluation_role"] = "holdout" if is_holdout else "reconstruction"
            view["used_for_geometry"] = not is_holdout
        geometry_views = [view for view in views if view["used_for_geometry"]]

        # Silhouette pose completion: frames COLMAP could not register still
        # carry good masks; recover approximate poses so carving sees the
        # full sweep.  Completed views carve; only SfM views feed TSDF/texture.
        carve_views = geometry_views
        completion_report = None
        try:
            from local3d.pose_complete import complete_poses

            frame_by_name = {frame.name: frame for frame in frames}
            completed = complete_poses(
                views, tight, tuple(result["intrinsics"]),
                points_xyz=result["points_xyz"],
            )
            completion_report = completed.get("report")
            carve_views = []
            for view in completed["views_all"]:
                if view.get("image_path") in (None, Path("")) :
                    view["image_path"] = frame_by_name.get(view["name"], view.get("image_path"))
                if view.get("mask_tight") is None:
                    view["mask_tight"] = tight.get(view["name"])
                if view["name"] in holdout_names:
                    continue
                view["used_for_geometry"] = True
                carve_views.append(view)
            _log(
                "poses",
                f"silhouette completion accepted {completed.get('accepted', 0)} "
                f"of {len(frames) - len(views)} unregistered frames",
            )
        except Exception as exc:  # completion is best-effort; carving falls back
            _log("poses", f"pose completion skipped: {exc}")

        posed_fraction = len(carve_views) / max(len(frames), 1)
        if posed_fraction < 0.5:
            _log(
                "poses",
                f"WARNING partial coverage: {len(carve_views)} of {len(frames)} "
                "frames posed; unobserved surfaces will be point-hull bounded "
                "and reported",
            )

        return {
            "mode": "sfm",
            "views": views,
            "geometry_views": geometry_views,
            "carve_views": carve_views,
            "intrinsics": tuple(result["intrinsics"]),
            "points_xyz": result["points_xyz"],
            "track_evidence": result.get("track_evidence", {"views": {}}),
            "registered_fraction": fraction,
            "report": {
                "sfm": result["report"],
                "completion": completion_report,
                "held_out_geometry_views": sorted(holdout_names),
            },
        }

    _log("poses", "SfM below acceptance gate; trying silhouette-turntable fallback")
    from local3d.turntable_pose import fit_turntable_poses

    sample = cv2.imread(str(frames[0]))
    height, width = sample.shape[:2]
    focal = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    intrinsics = (focal, width / 2.0, height / 2.0, 0.0)
    tt = fit_turntable_poses(mask_bundle["tight"], intrinsics, seed=seed)
    if not tt.get("ok"):
        raise SystemExit(
            "needs_recapture: SfM registered too few frames and the silhouette "
            f"turntable fit scored {tt.get('score', 0):.2f} (<0.66). "
            f"SfM report: {json.dumps(result.get('report', {}), default=str)[:500]}"
        )
    views = tt["views"]
    for index, view in enumerate(views):
        view["image_path"] = frames[index]
        view["name"] = frames[index].name
        view["mask_tight"] = mask_bundle["tight"][index]
        view["mask_eroded"] = mask_bundle["eroded"][index]
    _log("poses", f"turntable fit accepted (IoU {tt['score']:.2f}, sweep {tt['sweep_deg']:.0f} deg)")
    return {
        "mode": "turntable",
        "views": views,
        "intrinsics": intrinsics,
        "points_xyz": None,
        "registered_fraction": fraction,
        "report": {"sfm": result.get("report"), "turntable": tt.get("report"), "score": tt.get("score")},
    }


def stage_geometry(
    pose_bundle: dict,
    depth_model: Path | None,
    *,
    resolution: int,
    target_triangles: int,
    intersect_point_hull: bool = False,
) -> dict:
    from local3d.fusion import reconstruct_geometry

    result = reconstruct_geometry(
        pose_bundle.get("geometry_views", pose_bundle["views"]),
        pose_bundle["intrinsics"],
        pose_bundle["points_xyz"],
        depth_model if (depth_model and depth_model.is_file()) else None,
        carve_views=pose_bundle.get("carve_views"),
        resolution=resolution,
        target_triangles=target_triangles,
        depth_threads=2,
        intersect_point_hull=intersect_point_hull,
        use_precomputed_depths=any(
            view.get("depth_map") is not None
            for view in pose_bundle.get("geometry_views", pose_bundle["views"])
        ),
    )
    _log("geometry", f"{len(result['faces'])} triangles")
    return result


def stage_texture(geometry: dict, pose_bundle: dict, *, atlas_size: int) -> dict:
    from local3d.texturing import bake_texture_atlas

    # Only bundle-adjusted SfM poses may source texture. Silhouette-completed
    # poses are deliberately approximate and are restricted to hull carving.
    texture_views = [
        view
        for view in pose_bundle["views"]
        if view.get("image_path") and view.get("mask_tight") is not None
    ]
    result = bake_texture_atlas(
        geometry["vertices"],
        geometry["faces"],
        texture_views,
        pose_bundle["intrinsics"],
        atlas_size=atlas_size,
        min_frontality=0.15,
    )
    unobserved = result["report"].get("unobserved_face_fraction")
    _log("texture", f"atlas {atlas_size}, unobserved faces: {unobserved}")
    return result


def stage_export(texture: dict, out_dir: Path) -> dict:
    import trimesh
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    texture_path = out_dir / "texture.png"
    cv2.imwrite(str(texture_path), texture["texture_bgr"])

    mesh = trimesh.Trimesh(
        vertices=texture["vertices"], faces=texture["faces"], process=False
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=texture["uvs"],
        material=trimesh.visual.material.SimpleMaterial(
            image=Image.open(texture_path), diffuse=[255, 255, 255, 255]
        ),
    )
    glb_path = out_dir / "model.glb"
    mesh.export(glb_path)
    _log("export", str(glb_path))
    return {"glb": glb_path, "texture": texture_path}


def stage_qa(texture: dict, geometry: dict, out_dir: Path) -> dict:
    from local3d.qa_render import (
        geometry_gate,
        render_geometry_views,
        render_textured_views,
        save_contact_sheet,
    )

    sheet = render_textured_views(
        texture["vertices"], texture["faces"], texture["uvs"], texture["texture_bgr"]
    )
    save_contact_sheet(sheet, out_dir / "qa_textured_turntable.png")
    geo_sheet = render_geometry_views(geometry["vertices"], geometry["faces"])
    save_contact_sheet(geo_sheet, out_dir / "qa_geometry_turntable.png")
    gate = geometry_gate(geometry["vertices"], geometry["faces"])
    _log("qa", f"geometry gate: {'pass' if gate.get('pass') else gate.get('reasons')}")
    return gate


def validate_exported_glb(path: Path, *, expected_triangles: int) -> dict:
    """Reload the actual delivery file and validate its post-export topology."""

    import trimesh

    from local3d.qa_render import geometry_gate

    reasons: list[str] = []
    try:
        loaded = trimesh.load(path, force="scene", process=False)
        meshes = []
        for node_name in sorted(loaded.graph.nodes_geometry):
            transform, geometry_name = loaded.graph[node_name]
            geometry = loaded.geometry.get(geometry_name)
            if not isinstance(geometry, trimesh.Trimesh):
                continue
            mesh = geometry.copy()
            mesh.apply_transform(np.asarray(transform, dtype=np.float64))
            meshes.append(mesh)
        if not meshes:
            raise ValueError("GLB contains no triangle mesh")
        combined = trimesh.util.concatenate(tuple(meshes))
        attribute_topology = {
            "vertices": int(len(combined.vertices)),
            "triangles": int(len(combined.faces)),
            "components": int(combined.body_count),
            "watertight": bool(combined.is_watertight),
            "interpretation": (
                "attribute-indexed topology may be open where glTF duplicates positions "
                "for atlas UV or normal seams"
            ),
        }
        combined.merge_vertices(merge_tex=True, merge_norm=True, digits_vertex=8)
        combined.remove_unreferenced_vertices()
        gate = geometry_gate(
            np.asarray(combined.vertices), np.asarray(combined.faces), min_triangles=500
        )
        if not gate.get("pass"):
            reasons.extend(str(reason) for reason in gate.get("reasons", []))
        actual_triangles = int(len(combined.faces))
        if actual_triangles != int(expected_triangles):
            reasons.append(
                f"post-export triangle count changed ({actual_triangles} != {expected_triangles})"
            )
        textured_meshes = sum(
            1 for mesh in meshes if getattr(getattr(mesh, "visual", None), "kind", None) == "texture"
        )
        if textured_meshes == 0:
            reasons.append("reloaded GLB has no textured mesh primitive")
        return {
            "pass": not reasons,
            "reloaded": True,
            "bytes": int(path.stat().st_size),
            "mesh_primitives": len(meshes),
            "textured_mesh_primitives": textured_meshes,
            "triangles": actual_triangles,
            "topology_basis": (
                "scene transforms baked; coincident positions welded to 8 decimal digits "
                "while intentionally ignoring UV/normal seams"
            ),
            "attribute_indexed_before_position_weld": attribute_topology,
            "topology": gate,
            "reasons": reasons,
        }
    except Exception as exc:  # fail closed on malformed/unsupported delivery
        return {
            "pass": False,
            "reloaded": False,
            "reasons": [f"{type(exc).__name__}: {exc}"],
        }


def stage_evidence_gate(
    geometry: dict,
    texture: dict,
    pose_bundle: dict,
    topology_gate: dict,
    artifacts: dict,
    *,
    total_frame_count: int,
) -> dict:
    """Combine object-agnostic source evidence with delivery-file checks."""

    from local3d.reconstruction_gate import assess_reconstruction

    evidence = assess_reconstruction(
        geometry["vertices"],
        geometry["faces"],
        pose_bundle["views"],
        pose_bundle["intrinsics"],
        sfm_points=pose_bundle.get("points_xyz"),
        pose_mode=pose_bundle.get("mode"),
        total_frame_count=total_frame_count,
        sfm_registered_fraction=pose_bundle.get("registered_fraction"),
    )
    delivery = validate_exported_glb(
        Path(artifacts["glb"]), expected_triangles=len(texture["faces"])
    )
    reasons = list(evidence.get("reasons", []))
    if pose_bundle.get("mode") != "sfm":
        reasons.append("poses_are_not_independently_recovered_sfm")
    if not topology_gate.get("pass"):
        reasons.append("pre_export_topology_gate_failed")
    if not delivery.get("pass"):
        reasons.append("post_export_delivery_gate_failed")

    texture_sources = {
        str(view.get("pose_source", "unspecified")).strip().lower()
        for view in pose_bundle.get("views", [])
    }
    inferred_sources = sorted(
        source for source in texture_sources if source not in {"sfm", "charuco", "calibrated"}
    )
    if inferred_sources:
        reasons.append("texture_uses_inferred_camera_poses")

    unobserved = texture.get("report", {}).get("unobserved_face_fraction")
    if unobserved is not None and float(unobserved) > 0.40:
        reasons.append("too_much_delivery_texture_is_unobserved")

    reasons = list(dict.fromkeys(reasons))
    return {
        "pass": not reasons,
        "accepted": not reasons,
        "status": "accepted" if not reasons else "rejected",
        "reasons": reasons,
        "independent_evidence": evidence,
        "pre_export_topology": topology_gate,
        "post_export_delivery": delivery,
        "texture_pose_sources": sorted(texture_sources),
        "unobserved_face_fraction": unobserved,
        "policy": (
            "general geometry is deliverable only when independent SfM, held-out/source-mask "
            "agreement, source-pixel support, geometric depth evidence, and the reloaded GLB pass"
        ),
    }


def stage_prior_candidates(frames: list[Path], mask_bundle: dict, out_dir: Path) -> dict:
    """Build explicitly labelled regularized fallback candidates.

    These candidates never outrank a general reconstruction that passes the
    evidence gate. They exist because rigid slabs and deforming soft objects in
    held-object clips are often underconstrained for classical photogrammetry.
    """

    from local3d.auto_parametric import AutoFitError, fit_rounded_slab
    from local3d.auto_soft import AutoSoftError, fit_soft_volume

    source_masks_dir = mask_bundle["prior_dir"]
    inputs_dir = out_dir / "candidate_inputs"
    frames_dir = inputs_dir / "frames"
    masks_dir = inputs_dir / "masks"
    frames_dir.mkdir(parents=True, exist_ok=False)
    masks_dir.mkdir(parents=True, exist_ok=False)
    staged_names: list[str] = []
    for frame in frames:
        mask = source_masks_dir / f"{frame.stem}.png"
        if not mask.is_file():
            continue
        shutil.copyfile(frame, frames_dir / frame.name)
        shutil.copyfile(mask, masks_dir / mask.name)
        staged_names.append(frame.name)
    input_report = {
        "source_frames": len(frames),
        "object_supported_frames": len(staged_names),
        "missing_mask_policy": "exclude the frame; never substitute a foreground/person matte",
        "frames": staged_names,
    }
    (inputs_dir / "report.json").write_text(
        json.dumps(input_report, indent=2) + "\n", encoding="utf-8"
    )
    candidates: dict[str, dict] = {}

    slab_dir = out_dir / "rounded_slab"
    try:
        slab_report = fit_rounded_slab(frames_dir, masks_dir, slab_dir)
        candidates["rounded_slab"] = {
            "ok": True,
            "classification": "evidence-fitted parametric regularizer; not recovered photogrammetry",
            "report": slab_report,
            "input_summary": input_report,
            "artifacts": {
                "glb": slab_dir / "parametric_model.glb",
                "usdz": slab_dir / "parametric_model.usdz",
                "qa_model": slab_dir / "qa_model_contact.png",
                "qa_texture": slab_dir / "qa_texture_contact.png",
            },
        }
        _log("candidate", "rounded-slab regularizer passed its source/topology gates")
    except (AutoFitError, ValueError, RuntimeError, OSError) as exc:
        candidates["rounded_slab"] = {
            "ok": False,
            "classification": "evidence-fitted parametric regularizer",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _log("candidate", f"rounded-slab regularizer rejected: {exc}")

    soft_dir = out_dir / "soft_volume"
    try:
        soft_report = fit_soft_volume(frames_dir, masks_dir, soft_dir)
        candidates["soft_volume"] = {
            "ok": True,
            "classification": "inferred bilateral 2.5D soft volume; not recovered photogrammetry",
            "report": soft_report,
            "input_summary": input_report,
            "artifacts": {
                "glb": soft_dir / "soft_model.glb",
                "usdz": soft_dir / "soft_model.usdz",
                "qa_model": soft_dir / "qa_soft_contact.png",
            },
        }
        _log("candidate", "soft-volume fallback passed its source/topology gates")
    except (AutoSoftError, ValueError, RuntimeError, OSError) as exc:
        candidates["soft_volume"] = {
            "ok": False,
            "classification": "inferred bilateral 2.5D soft volume",
            "error": f"{type(exc).__name__}: {exc}",
        }
        _log("candidate", f"soft-volume fallback rejected: {exc}")
    return candidates


def select_candidate(general: dict, priors: dict) -> dict:
    """Select recovered geometry first, then the narrowest valid fallback."""

    if general.get("ok") and general.get("evidence_gate", {}).get("pass"):
        return {
            "name": "general_reconstruction",
            "classification": (
                "source-supported masked SfM reconstruction; geometry is silhouette-hull "
                "bounded and may include SfM-aligned monocular predicted-depth fusion"
            ),
            "reason": "the general candidate passed topology, pose, support, texture, and reprojection gates",
            "candidate": general,
        }
    slab = priors.get("rounded_slab", {})
    if slab.get("ok"):
        return {
            "name": "rounded_slab",
            "classification": slab["classification"],
            "reason": (
                "the general candidate failed its evidence gate; a regular rounded-slab fit "
                "passed conservative source and delivery gates"
            ),
            "candidate": slab,
        }
    soft = priors.get("soft_volume", {})
    if soft.get("ok"):
        return {
            "name": "soft_volume",
            "classification": soft["classification"],
            "reason": (
                "the general candidate failed its evidence gate and no rigid-slab fit was "
                "supported; the explicitly inferred soft-volume fallback passed"
            ),
            "candidate": soft,
        }
    reasons = list(general.get("evidence_gate", {}).get("reasons", []))
    if general.get("error"):
        reasons.append(str(general["error"]))
    for name, candidate in priors.items():
        if candidate.get("error"):
            reasons.append(f"{name}: {candidate['error']}")
    raise SystemExit("needs_recapture: no candidate passed evidence gates: " + "; ".join(reasons))


def promote_candidate(selection: dict, out_dir: Path) -> dict:
    """Atomically publish the selected candidate to stable delivery names."""

    source_artifacts = selection["candidate"]["artifacts"]
    source_glb = source_artifacts.get("glb")
    if source_glb is None or not Path(source_glb).is_file():
        raise SystemExit("selected candidate did not produce a GLB")
    if out_dir.exists():
        raise FileExistsError(f"delivery directory already exists: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = out_dir.with_name(f".{out_dir.name}.tmp")
    if temporary_dir.exists():
        raise FileExistsError(f"staging delivery directory already exists: {temporary_dir}")
    temporary_dir.mkdir()
    promoted: dict[str, str | int] = {}
    mappings = {
        "glb": "model.glb",
        "usdz": "model.usdz",
        "texture": "texture.png",
        "qa_model": "qa_model.png",
        "qa_texture": "qa_texture.png",
        "qa_geometry": "qa_geometry.png",
    }
    for key, destination_name in mappings.items():
        source = source_artifacts.get(key)
        if source is None:
            continue
        source = Path(source)
        if not source.is_file():
            continue
        destination = out_dir / destination_name
        temporary = temporary_dir / destination_name
        shutil.copyfile(source, temporary)
        promoted[key] = str(destination)
        promoted[f"{key}_bytes"] = temporary.stat().st_size
        promoted[f"{key}_sha256"] = _sha256(temporary)
    if "glb" not in promoted:
        raise SystemExit("selected candidate did not produce a GLB")
    temporary_dir.replace(out_dir)
    return promoted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-fps", type=float, default=12.0)
    parser.add_argument("--max-frames", type=int, default=170)
    parser.add_argument("--fov-degrees", type=float, default=65.0)
    parser.add_argument("--resolution", type=int, default=256, help="fusion voxel grid resolution")
    parser.add_argument("--target-triangles", type=int, default=20000)
    parser.add_argument("--atlas-size", type=int, default=2048)
    parser.add_argument(
        "--depth-model",
        type=Path,
        default=ROOT / "runs" / "models" / "depth_anything_v2_small_int8.onnx",
    )
    parser.add_argument(
        "--min-registered-fraction", type=float, default=0.20,
        help="minimum independently registered source-frame fraction before a "
        "general reconstruction may be attempted",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--match-threads", type=int, default=1,
        help=">1 speeds SIFT matching; weakens strict cross-run determinism "
        "and lets the SfM solve vary between runs",
    )
    args = parser.parse_args()

    cv2.setNumThreads(1)
    started = time.time()
    video = args.video.expanduser().resolve()
    out = args.output.expanduser().resolve()
    source = _source_snapshot(video)
    _prepare_output(out, source)

    frames, analysis = stage_ingest(
        video,
        out / "ingest",
        sample_fps=args.sample_fps,
        max_frames=args.max_frames,
    )
    if len(frames) < 24:
        raise SystemExit(f"needs_recapture: only {len(frames)} frames extracted")

    from local3d.source_guard import assess_source_video

    source_preflight = assess_source_video(analysis)
    (out / "source_preflight.json").write_text(
        json.dumps(source_preflight, indent=2, default=str) + "\n", encoding="utf-8"
    )
    if not source_preflight["accepted"]:
        raise SystemExit(
            "needs_recapture: source preflight failed: "
            + "; ".join(source_preflight["hard_failures"])
        )

    depth_model = args.depth_model.expanduser().resolve() if args.depth_model else None
    if depth_model is not None and not depth_model.is_file():
        _log("depth", f"model not found; continuing with silhouette/SfM evidence only: {depth_model}")
        depth_model = None

    prior_bundle = stage_prior_inputs(video, out / "prior_inputs")
    priors = stage_prior_candidates(
        prior_bundle["frames"], prior_bundle, out / "candidates"
    )
    mask_bundle = stage_masks(frames, out / "masks", depth_model=depth_model)

    general: dict = {
        "ok": False,
        "classification": "general masked-SfM reconstruction candidate",
        "evidence_gate": {
            "pass": False,
            "accepted": False,
            "status": "not_built",
            "reasons": ["general_candidate_was_not_built"],
        },
    }
    pose_bundle: dict | None = None
    try:
        pose_bundle = stage_poses(
            frames,
            mask_bundle,
            out / "sfm",
            seed=args.seed,
            fov_deg=args.fov_degrees,
            match_threads=args.match_threads,
            min_registered_fraction=args.min_registered_fraction,
        )
        if pose_bundle["mode"] != "sfm":
            general = {
                "ok": False,
                "classification": "general reconstruction rejected before geometry",
                "pose_mode": pose_bundle["mode"],
                "pose_report": pose_bundle["report"],
                "evidence_gate": {
                    "pass": False,
                    "accepted": False,
                    "status": "rejected",
                    "reasons": [
                        "poses_are_silhouette_assigned_not_independently_recovered",
                        "inferred_camera_poses_are_for_carving_only_and_may_not_source_texture",
                    ],
                },
            }
        else:
            geometry = stage_geometry(
                pose_bundle,
                depth_model,
                resolution=args.resolution,
                target_triangles=args.target_triangles,
                intersect_point_hull=False,
            )
            texture = stage_texture(geometry, pose_bundle, atlas_size=args.atlas_size)
            general_dir = out / "candidates" / "general_reconstruction"
            artifacts = stage_export(texture, general_dir)
            topology_gate = stage_qa(texture, geometry, general_dir)
            evidence_gate = stage_evidence_gate(
                geometry,
                texture,
                pose_bundle,
                topology_gate,
                artifacts,
                total_frame_count=len(frames),
            )
            general = {
                "ok": True,
                "classification": (
                    "masked SfM + held-out silhouette hull + source-view atlas; "
                    "monocular depth, when used, is predicted rather than measured"
                ),
                "pose_mode": pose_bundle["mode"],
                "pose_report": pose_bundle["report"],
                "geometry_report": geometry.get("report"),
                "texture_report": texture.get("report"),
                "evidence_gate": evidence_gate,
                "artifacts": {
                    **artifacts,
                    "qa_model": general_dir / "qa_textured_turntable.png",
                    "qa_geometry": general_dir / "qa_geometry_turntable.png",
                },
            }
    except (Exception, SystemExit) as exc:  # candidate failure must not suppress valid fallbacks
        general = {
            "ok": False,
            "classification": "general reconstruction failed closed",
            "error": f"{type(exc).__name__}: {exc}",
            "pose_mode": pose_bundle.get("mode") if pose_bundle else None,
            "pose_report": pose_bundle.get("report") if pose_bundle else None,
            "evidence_gate": {
                "pass": False,
                "accepted": False,
                "status": "rejected",
                "reasons": ["general_candidate_stage_failed"],
            },
        }
        _log("general", f"candidate rejected: {exc}")

    _verify_source_unchanged(video, source)
    selection: dict | None = None
    selection_error: str | None = None
    promoted: dict = {}
    try:
        selection = select_candidate(general, priors)
        promoted = promote_candidate(selection, out / "model")
    except SystemExit as exc:
        selection_error = str(exc)

    selection_report = None
    if selection is not None:
        selection_report = {
            key: value for key, value in selection.items() if key != "candidate"
        }
    report = {
        "schema_version": 2,
        "tool": "reconstruct_object hybrid evidence-gated pipeline (CPU-only, zero-LLM)",
        "status": "delivered" if selection is not None else "needs_recapture",
        "source": source,
        "source_preflight": source_preflight,
        "configuration": {
            "sample_fps": args.sample_fps,
            "max_frames": args.max_frames,
            "fov_degrees": args.fov_degrees,
            "resolution": args.resolution,
            "target_triangles": args.target_triangles,
            "atlas_size": args.atlas_size,
            "depth_model": str(depth_model) if depth_model else None,
            "seed": args.seed,
            "match_threads": args.match_threads,
            "minimum_registered_fraction": args.min_registered_fraction,
        },
        "elapsed_s": round(time.time() - started, 1),
        "frames": len(frames),
        "mask_report_summary": {
            key: value
            for key, value in mask_bundle["report"].items()
            if key != "frames"
        },
        "prior_mask_report_summary": {
            key: value
            for key, value in prior_bundle["report"].items()
            if key != "frames"
        },
        "candidates": {"general_reconstruction": general, **priors},
        "selection": selection_report,
        "selection_error": selection_error,
        "artifacts": promoted,
        "provenance_policy": [
            "a general mesh outranks priors only after independent pose, reprojection, surface-support, topology, and reloaded-GLB gates pass",
            "silhouette-completed poses may carve but never texture or count as independent camera evidence",
            "rounded-slab and soft-volume results are explicitly classified as evidence-fitted/inferred fallbacks, not photogrammetry",
            "monocular depth is a scale-aligned prediction, not measured range data",
        ],
        "limits": [
            "metric scale is ambiguous without a marker or measured reference",
            "unobserved geometry cannot be recovered from this source clip",
            "soft objects can deform between frames and violate rigid reconstruction assumptions",
            "specular or low-texture surfaces can prevent object-only SfM",
        ],
    }
    report_path = out / "report.json"
    temporary_report = out / ".report.json.tmp"
    temporary_report.write_text(
        json.dumps(report, indent=2, default=str, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_path)
    _log("done", f"{round(time.time() - started, 1)}s -> {report_path}")
    if selection is None:
        raise SystemExit(selection_error or "needs_recapture: no candidate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
