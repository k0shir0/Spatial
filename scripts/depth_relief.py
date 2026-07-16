#!/usr/bin/env python3
"""Textured depth-relief mesh from one masked frame (small local depth model).

For objects classical SfM cannot track (featureless plush, fabric), a small
monocular depth model (Depth Anything V2 Small, Apache-2.0, ONNX on CPU)
provides per-pixel *relative* depth.  This tool turns one reviewed frame +
mask into a watertight textured relief mesh: the front surface is the masked
relative depth, the back is a mirrored copy.

Honest limits, recorded in the report: relative depth has no absolute scale
(the relief amplitude is a display parameter, not a measurement); the back
side is assumed by mirror symmetry, not observed; within-object depth
ordering is learned inference, not verified measurement.  This is a
recognizable-shape path, not a scan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from local3d.visual_hull import taubin_smooth  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def predict_depth(model_path: Path, crop_bgr: np.ndarray, threads: int) -> np.ndarray:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (518, 518), interpolation=cv2.INTER_CUBIC)
    tensor = ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)[None]
    prediction = session.run(None, {session.get_inputs()[0].name: tensor.astype(np.float32)})[0][0]
    prediction = prediction.reshape(prediction.shape[-2], prediction.shape[-1])
    return cv2.resize(prediction, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame", type=Path)
    parser.add_argument("mask", type=Path, help="binary object mask PNG for the frame")
    parser.add_argument("--model", type=Path, required=True, help="Depth Anything V2 Small ONNX path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--relief", type=float, default=0.35, help="front relief amplitude as a fraction of object width (display parameter, not measured)")
    parser.add_argument("--back-scale", type=float, default=0.7, help="mirrored back amplitude relative to front")
    parser.add_argument("--resolution", type=int, default=160, help="voxel grid resolution")
    parser.add_argument("--target-triangles", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    frame = cv2.imread(str(args.frame), cv2.IMREAD_COLOR)
    mask_image = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    if frame is None or mask_image is None:
        parser.error("could not decode frame or mask")
    mask = mask_image > 127
    if mask.mean() < 0.005:
        parser.error("mask is nearly empty")

    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    margin = int(0.12 * max(x1 - x0, y1 - y0))
    x0, y0 = max(x0 - margin, 0), max(y0 - margin, 0)
    x1, y1 = min(x1 + margin, mask.shape[1] - 1), min(y1 + margin, mask.shape[0] - 1)
    crop = frame[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]

    depth = predict_depth(args.model, crop, args.threads)
    inside = depth[crop_mask]
    low, high = np.percentile(inside, 2), np.percentile(inside, 98)
    relief_map = np.clip((depth - low) / max(high - low, 1e-9), 0.0, 1.0)
    relief_map[~crop_mask] = 0.0
    relief_map = cv2.GaussianBlur(relief_map, (7, 7), 0)

    height, width = crop_mask.shape
    object_width = float(x1 - x0)
    front_amplitude = args.relief * object_width
    back_amplitude = args.back_scale * front_amplitude

    # Solid occupancy between mirrored back and relief front, then marching cubes.
    res = args.resolution
    grid_mask = cv2.resize(crop_mask.astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST) > 0
    grid_relief = cv2.resize(relief_map, (res, res), interpolation=cv2.INTER_LINEAR)
    depth_res = max(int(res * (front_amplitude + back_amplitude) / max(width, height)), 8)
    z_axis = np.linspace(-back_amplitude, front_amplitude, depth_res)
    front = grid_relief * front_amplitude
    back = -grid_relief * back_amplitude
    occupancy = (
        grid_mask[None, :, :]
        & (z_axis[:, None, None] <= front[None, :, :])
        & (z_axis[:, None, None] >= back[None, :, :])
    )

    from skimage.measure import marching_cubes

    volume = np.pad(occupancy.astype(np.uint8), 1)
    vertices_zyx, faces, _normals, _values = marching_cubes(volume, level=0.5)
    vertices_zyx -= 1.0
    scale_x = width / (res - 1)
    scale_y = height / (res - 1)
    scale_z = (front_amplitude + back_amplitude) / max(depth_res - 1, 1)
    vertices = np.column_stack(
        (
            vertices_zyx[:, 2] * scale_x,
            height - vertices_zyx[:, 1] * scale_y,  # image rows grow downward
            vertices_zyx[:, 0] * scale_z - back_amplitude,
        )
    )
    faces = faces[:, ::-1].astype(np.int64)  # one axis flipped, so restore outward winding
    vertices = taubin_smooth(vertices.astype(np.float32), faces, iterations=6)

    post_report = None
    if args.target_triangles:
        try:
            from local3d.mesh_post import postprocess

            vertices, faces, post_report = postprocess(vertices, faces, target_triangles=args.target_triangles)
        except (ImportError, RuntimeError) as exc:
            print(f"warning: decimation skipped ({exc})", file=sys.stderr)

    # UVs: orthographic projection back into the crop (texture = source pixels).
    uv = np.column_stack((vertices[:, 0] / width, vertices[:, 1] / height))
    uv = np.clip(uv, 0.0, 1.0)

    args.output.mkdir(parents=True, exist_ok=True)
    texture_path = args.output / "texture.png"
    cv2.imwrite(str(texture_path), cv2.flip(crop, 0))  # v=0 at image bottom

    from PIL import Image

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=uv,
        material=trimesh.visual.material.SimpleMaterial(image=Image.open(texture_path), diffuse=[255, 255, 255, 255]),
    )
    glb_path = args.output / "relief.glb"
    mesh.export(glb_path)

    report = {
        "tool": "depth_relief (Depth Anything V2 Small ONNX on CPU)",
        "model_file": str(args.model),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "frame": str(args.frame),
        "triangles": int(len(faces)),
        "watertight": bool(trimesh.Trimesh(vertices=vertices, faces=faces, process=False).is_watertight),
        "post_processing": post_report,
        "limits": [
            "relief amplitude is a display parameter; relative depth has no measured scale",
            "back side is mirrored front relief, assumed not observed",
            "depth ordering is learned inference from one frame, not verified measurement",
        ],
        "artifacts": {"glb": str(glb_path), "texture": str(texture_path)},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("triangles", "watertight")} | {"glb": str(glb_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
