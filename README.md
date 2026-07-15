# Spatial

[![CI](https://github.com/JustinGamer191/Spatial/actions/workflows/ci.yml/badge.svg)](https://github.com/JustinGamer191/Spatial/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

Spatial is an early-stage, local-first toolkit for turning reviewed object
videos and images into lightweight GLB/USDZ assets. It includes deterministic
parametric fitting, fitted soft parts, silhouette visual hulls, a conservative
shape-family router, and optional native macOS capture helpers.

> [!IMPORTANT]
> Spatial is not a one-click arbitrary video-to-3D system. Current fitted
> workflows require reviewed frames, masks, dimensions, or primitive layouts.
> Geometry and metric scale may be inferred when the capture does not measure
> them.

## What is included

| Path | Status | Learned inference | Repeatability |
|---|---|---:|---|
| Parametric phone/book/rounded-slab builder | Stable prototype | None | GLB and USDZ tested byte-for-byte in the reference toolchain |
| Fitted soft parts and deterministic texture bake | Prototype | None | Deterministic GLB/maps in the reference toolchain |
| Silhouette visual hull | Prototype | None | Seedless, deterministic geometry |
| Procedural shape-family router | Review-only | Tiny locally trained NumPy classifier | Seeded training and deterministic inference |
| Apple Object Capture and Vision helpers | Optional, macOS-only | Apple system frameworks | Platform-dependent |
| SAM/rembg and Hunyuan adapters | Optional experiments | Yes | Not part of the core guarantee |

The router never dispatches reconstruction automatically. Unsupported or
low-confidence inputs fail closed for review.

## Determinism and LLMs

The core fitted builders have **no LLM, generative-model, API, network, Torch,
or GPU dependency**. A build consumes local source images plus an explicit
reviewed JSON configuration. The parametric path fixes OpenCV to one thread,
disables OpenCL, omits wall-clock metadata, normalizes USDZ ZIP timestamps, and
records artifact hashes.

For identical input bytes, configuration, and the pinned reference toolchain,
the parametric GLB and USDZ rebuild byte-for-byte. Configurations can be written
manually or with any editor/assistant; build and runtime never call an LLM.

That does not make fitting automatic. Choosing frames, quads, masks, primitive
parts, dimensions, and materials is currently an operator-reviewed step. Also,
library/encoder upgrades and Apple system-tool changes can alter bytes, so the
guarantee is intentionally scoped to the reference environment. See
[Determinism](docs/DETERMINISM.md).

## Quick start

Python 3.10 or newer is required. FFmpeg/ffprobe is needed only for video
ingest. Apple USD tools are optional; on other platforms use `--skip-usdz`.

```bash
git clone https://github.com/JustinGamer191/Spatial.git
cd Spatial
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
```

Run the privacy-safe synthetic phone demo:

```bash
python scripts/make_demo_inputs.py --output examples/demo
spatial-parametric \
  --config examples/parametric_phone.json \
  --output examples/output \
  --skip-usdz
```

The output includes a GLB, USDA, provenance manifest, reviewed-quad overlays,
and CPU-only QA contact sheets. On macOS, omit `--skip-usdz` to package and
validate an Apple USDZ when the system tools are available.

For the exact versions used by the repeatability checks:

```bash
python -m pip install -r requirements/reference.txt
python -m pip install -e . --no-deps
```

## Workflows

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
