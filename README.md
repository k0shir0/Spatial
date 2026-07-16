# Spatial

[![CI](https://github.com/JustinGamer191/Spatial/actions/workflows/ci.yml/badge.svg)](https://github.com/JustinGamer191/Spatial/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Spatial is an early-stage, local-first toolkit for turning object videos and
reviewed images into auditable GLB/USDZ assets. Its automatic core attempts
masked structure-from-motion, evidence-backed geometry, and source-view
texturing, then either promotes the result, selects an explicitly labelled
regularized fallback, or asks for a recapture. Deterministic parametric, soft
part, visual-hull, and native macOS helpers are also included.

> [!IMPORTANT]
> Spatial is not a universal arbitrary-video scanner. The automatic path is
> experimental and expects one continuous, mostly static-camera clip of one
> hand-rotated object. Hands, deformation, reflections, missing viewpoints, and
> weak texture can force an inferred fallback or `needs_recapture`. Metric scale
> is unknown unless the capture provides a measurement.

## What is included

| Path | Status | Learned inference | Repeatability |
|---|---|---:|---|
| Evidence-gated video reconstruction | Experimental | Fixed local U2NetP; optional Depth Anything | Seeded/single-threaded; strongest on the pinned platform/toolchain |
| Parametric phone/book/rounded-slab builder | Stable prototype | None | GLB and USDZ tested byte-for-byte in the reference toolchain |
| Fitted soft parts and deterministic texture bake | Prototype | None | Deterministic GLB/maps in the reference toolchain |
| Silhouette visual hull | Prototype | None | Seedless, deterministic geometry |
| Procedural shape-family router | Review-only | Tiny locally trained NumPy classifier | Seeded training and deterministic inference |
| Apple Object Capture and Vision helpers | Optional, macOS-only | Apple system frameworks | Platform-dependent |
| Standalone SAM and Hunyuan adapters | Optional experiments | Yes | Not part of the core guarantee |

The automatic router records the selected geometry class and why every other
candidate was rejected. Rounded-slab and bilateral soft-volume outputs are
never reported as recovered photogrammetry.

## Determinism and LLMs

The core automatic and fitted runtime paths call no **LLM, generative model,
hosted API, or network**. Standalone experimental adapters are outside this
contract. The automatic video command consumes the video and ordinary CLI
settings; it does not require object-specific JSON, reviewed frame choices, or
cached masks, poses, depth, geometry, or textures.

That command does use fixed, generic learned vision weights: U2NetP for
foreground masks and, when requested, Depth Anything V2 Small for predicted
relative depth. These are local computer-vision models, not language models.
Fallback routing and geometry use deterministic color, silhouette, and shape
priors, so unseen geometry remains inferred even when every operation is
repeatable.

The reviewed parametric builder has the stronger byte contract: identical input
bytes, JSON configuration, and pinned reference toolchain rebuild identical GLB
and USDZ files. The full video pipeline fixes its seed and defaults to one
matching thread, but codec, ONNX, numeric-library, and pycolmap differences can
change decisions across machines. Sidecar reports also record elapsed time and
absolute paths. See [Determinism](docs/DETERMINISM.md) for the exact boundary.

## Quick start

Python 3.10 or newer is required. FFmpeg/ffprobe is needed only for video
ingest. USDZ packaging is pure Python and works on every platform; Apple's
`usdchecker` adds an extra validation pass when present on macOS.

```bash
git clone https://github.com/JustinGamer191/Spatial.git
cd Spatial
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
```

Install the automatic video dependencies and run a new clip into an empty
output directory:

```bash
python -m pip install -e '.[video]'

python scripts/reconstruct_object.py capture.mov \
  --output runs/capture-001 \
  --depth-model /absolute/path/depth_anything_v2_small_int8.onnx
```

`--depth-model` is optional. The pipeline requires `u2netp.onnx` in rembg's
local model cache (`$U2NET_HOME` or `~/.u2net`) and never downloads missing
weights implicitly. FFmpeg/ffprobe must be installed separately. Use
`--match-threads 1`—the default—for the strictest same-machine repeatability.

Run the privacy-safe synthetic phone demo:

```bash
python scripts/make_demo_inputs.py --output examples/demo
spatial-parametric \
  --config examples/parametric_phone.json \
  --output examples/output
```

The output includes a GLB, USDA, a deterministically packaged USDZ,
provenance manifest, reviewed-quad overlays, and CPU-only QA contact sheets.
`--skip-usdz` remains available to skip packaging entirely; on macOS the
system `usdchecker` also validates the USDZ when present.

For the exact versions used by the repeatability checks:

```bash
python -m pip install -r requirements/reference.txt
python -m pip install -e . --no-deps
```

## Workflows

### Evidence-gated video reconstruction

`scripts/reconstruct_object.py` is the automatic hybrid path for a short clip
of a hand-rotated object. It extracts frames locally, builds clip-consistent
object masks, and removes or rejects hand/arm-contaminated and otherwise unsafe
frames before feature matching. Masked COLMAP then attempts a true SfM general
candidate. Poses inferred from silhouettes may extend visual-hull carving, but
they never provide texture pixels or count as independent camera evidence.

The general candidate is delivered only if an independent fidelity gate accepts
its recovered poses, held-out silhouette agreement, source-surface support,
texture coverage, topology, and the reloaded GLB. If it is rejected, the router
tries two explicitly labelled regularizers: an evidence-fitted rounded slab for
compatible rigid objects, then an inferred bilateral 2.5D soft volume. These
fallbacks are not reported as photogrammetry. If no candidate has adequate
evidence, the command fails closed with `needs_recapture` instead of promoting a
plausible-looking but unsupported mesh.

The current command requires FFmpeg, COLMAP, and a locally cached `u2netp`
rembg model; it does not download weights implicitly. The Depth Anything model
is optional and supplies predicted—not measured—depth:

```bash
python scripts/reconstruct_object.py capture.mov \
  --output runs/capture-001 \
  --depth-model runs/models/depth_anything_v2_small_int8.onnx
```

For the Mac-only, existing-video Depth Anything/Apple Depth Pro comparison,
including its frozen-input evidence gate, offline setup boundary, and upstream
license notes, see [Mac existing-video depth comparison](docs/MAC_VIDEO_DEPTH.md).

Use a new, empty output directory for every attempt. `ingest/`, `masks/`, and
`sfm/` retain the source evidence; `candidates/` retains every attempted general
or regularized result; and `model/` contains stable names for the selected GLB,
optional USDZ/texture, and QA images. `source_preflight.json` records capture
checks, while `report.json` records candidate rejections, the selection reason,
artifact hashes, model/depth provenance, and known limits. Metric scale remains
ambiguous without a reference, hidden surfaces cannot be recovered, deformation
violates SfM assumptions, and reflective or textureless captures may require a
better-lit, hands-clear recapture.

#### Clean-room tin check

The current core was tested from the original 355-frame Mac video in an empty
worktree and run directory, with no object-specific configuration or reused
intermediates. It freshly sampled 142 reconstruction frames and a separate
36-frame fallback-evidence stream. The general 19,974-triangle mesh was rejected
for only 61.64° of camera coverage, 63.54° maximum opposing-view separation,
four direction bins, and 47.96% directly observed texture surface. The pipeline
instead selected an 816-triangle watertight rounded slab with observed front and
back pixels and deterministically inferred thickness. Its stable artifact hashes
were `ee1a5f59…a526e` (GLB) and `c25f7801…06a54` (USDZ). The private source and
textured outputs remain intentionally excluded from Git.

### Parametric assets

`scripts/build_parametric_asset.py` rectifies explicitly reviewed front/back
quads and builds bounded phone or book geometry. A face may use source pixels
or a measured uniform material. Explicit decorations include closed rings,
cylinders, rounded prisms, and controls, avoiding duplicated photographed and
geometric features.

Start from [the synthetic phone config](examples/parametric_phone.json). Source
media is resolved relative to the config file. The builder rejects oversized
textures, implausible dimensions, open topology, and assets over 5,000
triangles.

### Plush and free-form objects

- `scripts/build_soft_parts_asset.py` builds operator-fitted ellipsoids,
  superellipsoids, and tubes.
- `scripts/build_textured_soft_parts_asset.py` adds deterministic source-measured
  color, fabric normal detail, and roughness maps.
- `scripts/build_visual_hull.py` carves a CPU visual hull from reviewed binary
  silhouettes.

See [Lightweight plush reconstruction](PLUSH_RECONSTRUCTION.md).

### Board-based capture (recommended)

Print a ChArUco board (`scripts/make_charuco_board.py`), capture per
[the capture protocol](docs/CAPTURE.md), and select keyframes by measured
orbit coverage with `scripts/select_board_frames.py`. Poses, frame selection,
and gating are classical OpenCV — deterministic and zero-ML. `spatial-ingest
--segmenter u2netp` optionally scores subject coverage with a small local CV
segmenter instead of the temporal-change heuristic.

### Masked SfM reconstruction (experimental)

`scripts/masked_sfm_hull.py` reconstructs a handheld rotating object from an
ordinary clip: object masks restrict classical CPU SIFT to the object, a
seeded COLMAP mapper recovers object-relative poses, and geometry is the
intersection of the triangulated-point convex hull with silhouette carving.
Every candidate pose model is validated by reprojection IoU against the
masks and the build fails closed below the acceptance gate. Output scale is
ambiguous and concavities are not recovered. Requires the `sfm` extra.

### Shape routing

`scripts/shape_router.py` measures reviewed silhouettes and returns a
review-only geometry-family candidate. Its small classifier is trained entirely
from seeded procedural masks; it has no downloaded weights. See
[Shape-family router](SHAPE_ROUTER.md).

### Native macOS helpers

The `native/` directory contains Swift packages for:

- RealityKit Object Capture with optional explicit masks;
- Apple Vision foreground masks; and
- advisory Apple Vision semantic hints.

These use operating-system frameworks, are not LLMs, and are not required by
the deterministic Python builders.

## Capture and accuracy limits

- Hands, deformation, reflections, and permanently hidden surfaces reduce
  recoverable geometry.
- A source-fitted asset is an approximation unless camera calibration, depth,
  and a scale reference are provided.
- Photo textures can contain private screen content, faces, or background
  pixels. Inspect every output before sharing it.
- Object Capture should be previewed at low detail first; increasing detail
  does not repair bad registration or masks.

## Privacy

Spatial processes source media locally. The repository intentionally excludes
all real recordings, extracted frames, generated runs, and fitted assets.
Those files can contain faces, filenames, absolute paths, and source pixels
embedded in GLB/USDZ textures. The synthetic demo is the only public example
input.

## Repository layout

```text
src/local3d/       Python library and deterministic builders
scripts/           Command-line entry points and utilities
tests/             Geometry, packaging, safety, and repeatability tests
models/            Small procedural shape-router JSON models
native/            Optional macOS Swift packages
examples/          Synthetic public configurations
requirements/      Pinned reference environment
```

## Contributing and security

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). Please do not
attach private captures or generated textured assets to public issues. Report
security problems using [SECURITY.md](SECURITY.md).

Spatial is released under the [MIT License](LICENSE). Third-party tools and
model weights used by optional experiments retain their own licenses and are
not distributed by this repository.
