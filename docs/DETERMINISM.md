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
  deterministic algorithms, but not every sidecar report or USDZ package has a
  byte-repeatability test yet.
- Video job orchestration uses UUIDs and timestamps by design.
- Native Apple Vision/Object Capture behavior is controlled by OS frameworks
  and carries no byte-repeatability guarantee.
- Optional SAM/rembg and Hunyuan utilities use learned weights. They are not
  LLMs, are not installed by default, and are outside this contract.

## Human review

A deterministic builder does not imply an automatic fit. New captures still
require reviewed frame choices, quads, masks, dimensions, primitives, or
materials. Once those decisions are serialized in a configuration, rebuilding
does not require the person—or any LLM—that authored it.
