# Determinism contract

Spatial separates deterministic building from operator-reviewed fitting.

## Parametric builder

For the same source-image bytes, normalized JSON configuration, and reference
toolchain, `local3d.parametric_assets` is expected to emit byte-identical GLB
and USDZ files.

The implementation:

- uses no randomness, learned inference, network access, or wall clock;
- runs OpenCV with one thread and disables OpenCL;
- uses fixed reviewed quads and configured geometry without automatic fitting;
- caps source size, texture resolution, and triangle count;
- validates watertight/winding-consistent topology;
- packages USDZ with a pure-Python writer that stores entries uncompressed,
  64-byte aligned, with DOS-epoch timestamps, on every platform; and
- hashes every emitted artifact in a provenance manifest.

Regression tests rebuild GLB and USDZ in independent directories and compare
their bytes.

## Automatic video reconstruction

`scripts/reconstruct_object.py` has no LLM, generative-model, hosted-API, or
runtime-network step. It starts from the video in a new output directory and
does not require an object-specific authored configuration or cached frames,
masks, poses, depth maps, geometry, or textures.

It is not a zero-ML path. U2NetP supplies a generic foreground matte, and an
optional Depth Anything V2 Small ONNX model supplies predicted relative depth.
Both must already exist locally; the pipeline records their provenance and
refuses implicit downloads. Mask cleanup, CPU SIFT/SfM, candidate construction,
evidence gates, fallback fitting, export, and promotion are ordinary code with
fixed ordering. The default seed is zero and matching uses one thread.

That makes the command repeatable under a fixed local environment, but the full
video path does not carry the parametric builder's cross-location byte contract.
Video/JPEG/PNG codec versions, ONNX and numeric kernels, CPU architecture, and
pycolmap can perturb values near a gate. The SfM preflight also contains a
wall-time performance probe that can choose a different SIFT octave under load.
Use the same commit, source and weight hashes, dependency versions, platform,
seed, and `--match-threads 1` for the strongest comparison. Compare the stable
model artifacts; `report.json` intentionally records elapsed time, absolute
paths, and source filesystem metadata.

Rounded-slab and bilateral soft-volume fallbacks are deterministic in the same
environment, but deterministic inference is still inference. Their unobserved
geometry comes from explicit shape priors and is labelled as parametric or
inferred rather than recovered photogrammetry.

## Reference toolchain

The pinned versions are recorded in `requirements/reference.txt`. The checked
reference environment uses Python 3.10.16. USDZ packaging is pure Python and
platform-independent; when macOS supplies `/usr/bin/usdchecker`, it runs as an
extra ARKit-compliance validation pass but is not required to build.

Changing Python packages, image codecs, CPU architecture, or Apple system tools
may change bytes without changing visible geometry. A manifest also records
absolute source/config paths for provenance, so manifest bytes can differ
between checkout locations even when model bytes match.

## Other paths

- Soft-part geometry, visual-hull geometry, and texture-map generation are
  deterministic algorithms, but not every sidecar report has a
  byte-repeatability contract.
- Video job orchestration uses UUIDs and timestamps by design.
- Native Apple Vision/Object Capture behavior is controlled by OS frameworks
  and carries no byte-repeatability guarantee.
- U2NetP/rembg is a fixed learned-vision dependency of the automatic video
  extra. Optional SAM and Hunyuan utilities also use learned weights. None is an
  LLM; only the reviewed parametric builder carries the stronger byte contract.

## Human review

A deterministic builder does not imply an evidence-backed scan. Reviewed
builders still require frame choices, quads, masks, dimensions, primitives, or
materials serialized in a configuration. The automatic video path removes that
object-specific authoring step, but its capture and selected output still need
human review. Neither path requires an LLM at build or runtime.
