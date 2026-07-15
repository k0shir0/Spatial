#!/usr/bin/env python3
"""Run an explicitly supplied local Hunyuan3D-2mv checkout and checkpoint.

This optional learned-model adapter is not used by Spatial's deterministic
builders. Hunyuan code and weights are not distributed or licensed here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--back", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True, help="local Hunyuan3D-2 checkout")
    parser.add_argument("--model", type=Path, required=True, help="local model snapshot")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--octree-resolution", type=int, default=192)
    parser.add_argument("--chunks", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    sys.path.insert(0, str(args.repo))
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    dtype = torch.float16 if device == "mps" else torch.float32
    images = {
        "front": Image.open(args.front).convert("RGBA"),
        "left": Image.open(args.left).convert("RGBA"),
        "back": Image.open(args.back).convert("RGBA"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f"loading local Hunyuan3D-2mv on {device} ({dtype})", flush=True)
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(args.model),
        subfolder="hunyuan3d-dit-v2-mv",
        variant="fp16",
        device=device,
        dtype=dtype,
    )
    loaded = time.time()
    print(f"pipeline loaded in {loaded - started:.1f}s", flush=True)
    mesh = pipe(
        image=images,
        num_inference_steps=args.steps,
        octree_resolution=args.octree_resolution,
        num_chunks=args.chunks,
        generator=torch.manual_seed(args.seed),
        output_type="trimesh",
    )[0]
    generated = time.time()
    mesh.export(args.output)
    report = {
        "backend": "Hunyuan3D-2mv",
        "local_only": True,
        "device": device,
        "dtype": str(dtype),
        "steps": args.steps,
        "octree_resolution": args.octree_resolution,
        "seed": args.seed,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "load_seconds": round(loaded - started, 3),
        "inference_seconds": round(generated - loaded, 3),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
