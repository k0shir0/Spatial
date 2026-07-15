#!/usr/bin/env python3
"""Automatic video -> textured 3D object reconstruction (CPU-only, zero-LLM).

One command takes a short clip of a hand-rotated object (static camera) to a
textured GLB plus QA renders, fully automatically:

1.  ingest      — decode frames at a dense sample rate (local3d.ingest).
2.  masks       — u2netp object masks (rembg, CPU) cleaned of hands/arms
                  (local3d.mask_clean), eroded copies for SfM.
3.  poses       — masked COLMAP CPU SfM with sequential + loop matching
                  (local3d.sfm_video); silhouette-turntable fallback when SfM
                  cannot register enough frames (local3d.turntable_pose).
4.  geometry    — silhouette hull + hull-constrained TSDF fusion of aligned
                  monocular depth (local3d.fusion, local3d.monodepth).
5.  texture     — per-face single-best-view atlas with seam leveling
                  (local3d.texturing) — no cross-view blending.
6.  export + QA — GLB, textured turntable contact sheet, geometry gate,
                  report.json (local3d.qa_render).

Honest limits (recorded in the report): scale is ambiguous without a board or
measured reference; surfaces never observed stay hull-bounded and their texels
are filled from neighbouring observed regions; the turntable fallback assumes
a roughly single-axis rotation and fails closed otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
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


def stage_ingest(video: Path, out_dir: Path, *, sample_fps: float, max_frames: int) -> list[Path]:
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
    return frames


def stage_masks(frames: list[Path], out_dir: Path, *, model: str = "u2netp") -> dict:
    from rembg import new_session, remove
    from PIL import Image

    from local3d.mask_clean import clean_mask_sequence, erode_for_sfm

    session = new_session(model)
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

    tight, eroded, report = clean_mask_sequence(raw_masks, images)

    tight_dir = out_dir / "masks_tight"
    eroded_dir = out_dir / "masks_eroded"
    overlay_dir = out_dir / "overlays"
    for directory in (tight_dir, eroded_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for path, frame, tight_mask, eroded_mask in zip(frames, images, tight, eroded):
        cv2.imwrite(str(tight_dir / f"{path.stem}_mask.png"), tight_mask.astype(np.uint8) * 255)
        cv2.imwrite(str(eroded_dir / f"{path.stem}_mask.png"), eroded_mask.astype(np.uint8) * 255)
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
        "tight": tight,
        "eroded": eroded,
        "report": report,
    }


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
        mask_bundle["eroded_dir"],
        work_dir,
        seed=seed,
        fov_deg=fov_deg,
        match_threads=match_threads,
    )
    fraction = float(result.get("registered_fraction") or 0.0)
    _log("poses", f"SfM registered fraction: {fraction:.2f}")

    if result.get("ok") and fraction >= min_registered_fraction:
        views = result["views"]
        tight = load_mask_dir(mask_bundle["tight_dir"], frames)
        eroded = load_mask_dir(mask_bundle["eroded_dir"], frames)
        for view in views:
            view["mask_tight"] = tight.get(view["name"])
            view["mask_eroded"] = eroded.get(view["name"])
        views = [v for v in views if v["mask_tight"] is not None]
        return {
            "mode": "sfm",
            "views": views,
            "intrinsics": tuple(result["intrinsics"]),
            "points_xyz": result["points_xyz"],
            "report": result["report"],
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
        "report": {"sfm": result.get("report"), "turntable": tt.get("report"), "score": tt.get("score")},
    }


def stage_geometry(pose_bundle: dict, depth_model: Path | None, *, resolution: int, target_triangles: int) -> dict:
    from local3d.fusion import reconstruct_geometry

    result = reconstruct_geometry(
        pose_bundle["views"],
        pose_bundle["intrinsics"],
        pose_bundle["points_xyz"],
        depth_model if (depth_model and depth_model.is_file()) else None,
        resolution=resolution,
        target_triangles=target_triangles,
    )
    _log("geometry", f"{len(result['faces'])} triangles")
    return result


def stage_texture(geometry: dict, pose_bundle: dict, *, atlas_size: int) -> dict:
    from local3d.texturing import bake_texture_atlas

    result = bake_texture_atlas(
        geometry["vertices"],
        geometry["faces"],
        pose_bundle["views"],
        pose_bundle["intrinsics"],
        atlas_size=atlas_size,
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
    parser.add_argument("--min-registered-fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--match-threads", type=int, default=4,
        help=">1 speeds SIFT matching; weakens strict cross-run determinism",
    )
    args = parser.parse_args()

    cv2.setNumThreads(1)
    started = time.time()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    frames = stage_ingest(args.video, out / "ingest", sample_fps=args.sample_fps, max_frames=args.max_frames)
    if len(frames) < 24:
        raise SystemExit(f"needs_recapture: only {len(frames)} frames extracted")

    mask_bundle = stage_masks(frames, out / "masks")
    pose_bundle = stage_poses(
        frames,
        mask_bundle,
        out / "sfm",
        seed=args.seed,
        fov_deg=args.fov_degrees,
        match_threads=args.match_threads,
        min_registered_fraction=args.min_registered_fraction,
    )
    geometry = stage_geometry(
        pose_bundle, args.depth_model, resolution=args.resolution, target_triangles=args.target_triangles
    )
    texture = stage_texture(geometry, pose_bundle, atlas_size=args.atlas_size)
    artifacts = stage_export(texture, out / "model")
    gate = stage_qa(texture, geometry, out / "model")

    report = {
        "tool": "reconstruct_object (masked SfM/turntable + hull-TSDF + best-view texture, zero-LLM)",
        "video": str(args.video),
        "seed": args.seed,
        "elapsed_s": round(time.time() - started, 1),
        "frames": len(frames),
        "pose_mode": pose_bundle["mode"],
        "pose_report": pose_bundle["report"],
        "mask_report_summary": {
            key: value
            for key, value in mask_bundle["report"].items()
            if not isinstance(value, list)
        },
        "geometry_report": geometry.get("report"),
        "texture_report": texture.get("report"),
        "geometry_gate": gate,
        "artifacts": {str(k): str(v) for k, v in artifacts.items()},
        "limits": [
            "scale is ambiguous (no marker or measured reference)",
            "unobserved surfaces are hull-bounded; their texels are filled from neighbours",
            "turntable fallback assumes near single-axis rotation",
        ],
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    _log("done", f"{round(time.time() - started, 1)}s -> {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
