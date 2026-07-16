# Mac existing-video depth comparison

This workflow is for an existing video file processed locally on a Mac. It does
not require an iPhone, LiDAR, ARKit, RealityKit Object Capture, or Apple's
Foundation Models framework. A fixed Mac camera with a rigid object rotated in
front of it is a supported capture pattern: masked object features let SfM solve
an object-centric camera orbit.

Depth remains supporting evidence, not ground truth. Both backends discussed
here predict depth from RGB pixels. Neither one measures range, and neither can
recover a permanently hidden surface or repair an incorrect SfM solution.

## The strategy

The comparison is deliberately not “run two models and choose the nicer depth
image.” It uses one frozen reconstruction problem for both models:

1. Decode the video, construct object-only masks, and run masked SfM once.
2. Freeze the accepted masks, camera intrinsics, independently recovered poses,
   sparse 3D tracks, held-out split, and view-pair list.
3. Run Depth Anything V2 Small and Apple Depth Pro on the same registered source
   frames.
4. Align predictions to the common SfM coordinate system using only calibration
   tracks. Point IDs are split globally, so a 3D point used to calibrate one
   frame cannot reappear as validation evidence in another frame.
5. Score the aligned maps against held-out tracks and by symmetric cross-view
   reprojection. The latter backprojects one frame's predicted surface, projects
   it into another independently posed frame, and checks whether the other map
   describes the same visible surface.
6. Fuse a backend only if it passes every evidence threshold. If both pass, one
   needs a material quality advantage; otherwise they must agree closely enough
   to form a conservative consensus map.

Masks and SfM must not be recomputed per backend. Doing so would compare two
different reconstruction problems and could reward a model merely because its
mask pruning produced an easier camera solution.

## How the two predictions are treated

Depth Anything V2 Small supplies relative disparity: larger values mean nearer,
but values have no metric scale. Each registered view receives a robust affine
inverse-depth calibration from its calibration tracks.

Depth Pro supplies predicted optical-axis camera-Z depth in metres. The
comparison preserves that stronger contract by fitting one positive scale for
the clip, not a new scale and offset for every frame. The scale maps Depth
Pro's metres into SfM's otherwise arbitrary world units. No ray-length
conversion is applied. A Depth Pro result is still a monocular prediction; it
must not be reported as LiDAR, sensor depth, or independently measured object
scale.

The held-out and multiview checks are both required:

- Held-out tracks catch a model that fits its calibration points but orders
  nearby and far surfaces incorrectly.
- Multiview reprojection catches frame-to-frame shape drift between sparse
  tracks—the failure mode that can turn excellent per-frame fit numbers into a
  lumpy TSDF mesh.
- Angular pair bins prevent a long run of nearly duplicate adjacent frames from
  masquerading as broad multiview support.
- Mask-boundary and sharp depth-discontinuity pixels are downweighted or
  excluded because one-pixel segmentation or resampling errors are not reliable
  geometry evidence.
- A reprojected point hidden behind a nearer target surface is treated as an
  occlusion. A point asserted in front of the target surface is a free-space
  contradiction.

Thresholds are versioned in the JSON report rather than duplicated here. The
current implementation is in `src/local3d/depth_consistency.py`; it reports
aligned-view coverage, held-out median and tail errors, angular pair coverage,
bidirectional reprojection coverage, consistency, free-space contradictions,
per-view failures, and a final quality score.

## Fail-closed decisions

There are four possible depth decisions:

- `selected`: exactly one backend passes, or one passing backend has a material
  independent-evidence advantage.
- `consensus`: multiple backends pass with similar scores and their aligned
  depths agree within the configured tolerance. Only agreeing pixels survive.
- `reject / no_backend_passed_independent_evidence`: neither prediction has
  enough held-out and cross-view support.
- `reject / ambiguous_backend_disagreement`: both appear plausible, but the
  evidence cannot distinguish them and their geometry disagrees.

A rejected depth result is never fused. The reconstruction may still attempt a
silhouette/SfM-only candidate, an explicitly labelled rounded-slab or soft-volume
regularizer, or return `needs_recapture`. Depth success also cannot override the
independent camera-coverage, held-out silhouette, texture-support, topology, and
reloaded-file gates.

Typical fail-closed causes include too few registered object views, too few real
SfM track observations per frame, a narrow angular sweep, deformation, motion
blur, hands included in the object mask, reflective or transparent surfaces,
and predictions that are internally sharp but inconsistent between views.

## Offline setup

The host Spatial environment and the Depth Pro environment are intentionally
separate. Spatial does not vendor, install, or silently download Depth Pro or
its weights.

### Spatial-side requirements

Install Spatial's normal video dependencies, FFmpeg/ffprobe, masked-SfM support,
the locally cached `u2netp` rembg model, and a local Depth Anything V2 Small ONNX
file. Keep the exact ONNX path and SHA-256 in the run provenance. Reconstruction
must not download missing weights.

Depth Anything V2's official repository licenses the **Small** model under
Apache-2.0; its Base, Large, and Giant models are CC-BY-NC-4.0. This workflow is
specifically written for Small. An ONNX conversion may add a separate converter
license and provenance chain, so verify the exact file you distribute rather
than assuming every file named “Depth Anything V2” has the same terms.

### External Depth Pro environment

Apple's official setup currently recommends a Python 3.9 environment and an
editable install. Pin an exact 40-character source commit rather than following
the moving default branch:

```bash
git clone https://github.com/apple/ml-depth-pro.git /absolute/path/ml-depth-pro
cd /absolute/path/ml-depth-pro
git checkout <FULL_40_CHARACTER_COMMIT_SHA>

python3.9 -m venv /absolute/path/depth-pro-venv
source /absolute/path/depth-pro-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

# Intentional online setup step supplied by the official repository.
./get_pretrained_models.sh
shasum -a 256 /absolute/path/ml-depth-pro/checkpoints/depth_pro.pt
```

Review the checked-out `LICENSE` and `ACKNOWLEDGEMENTS.md` before installing or
redistributing anything. After setup, the Spatial adapter requires:

- an absolute path to that environment's Python executable;
- an absolute path to `depth_pro.pt`;
- the exact full source commit SHA;
- MPS on an Apple-silicon Mac by default; and
- a new, empty output directory for every prediction batch.

The adapter hashes the checkpoint, passes the already recovered focal length to
Depth Pro, runs a fixed repository-owned batch script without a shell, forces
common model clients into offline mode, validates every output and provenance
record, and atomically publishes the batch. CPU or CUDA execution is diagnostic
and requires an explicit opt-in; the Mac production expectation is MPS.

The intentional network boundary is setup: clone the pinned source, install its
dependencies, and obtain the checkpoint while online. Video reconstruction
itself should run with the required files already present and must fail rather
than fetch anything.

## Command-line contract

The existing single-backend option remains:

```bash
python scripts/reconstruct_object.py /absolute/path/capture.mov \
  --output /absolute/path/runs/capture-001 \
  --depth-model /absolute/path/models/depth_anything_v2_small_int8.onnx
```

That command creates the frozen frames, masks, and SfM model. Run the A/B stage
against those exact artifacts with a separate empty output and cache directory:

```bash
python scripts/compare_depth_backends.py \
  --sfm-model /absolute/path/runs/capture-001/sfm/model \
  --frames-dir /absolute/path/runs/capture-001/ingest/frames \
  --masks-dir /absolute/path/runs/capture-001/masks/masks_tight \
  --output-dir /absolute/path/runs/capture-001/depth-comparison \
  --cache-dir /absolute/path/runs/capture-001/depth-cache \
  --depth-anything-model /absolute/path/models/depth_anything_v2_small.onnx \
  --depth-pro-python /absolute/path/depth-pro-venv/bin/python \
  --depth-pro-checkpoint /absolute/path/depth_pro.pt \
  --depth-pro-commit <FULL_40_CHARACTER_COMMIT_SHA> \
  --emit-geometry-inputs
```

The driver chooses an angularly distributed subset rather than a temporal
sample, verifies that the subset preserves the recovered camera support, hashes
the frozen inputs, and caches exact-frame predictions. `geometry_inputs` is
written only when the SfM preflight and depth-selection gate both pass. A run
with inadequate SfM normally stops before either model is invoked; use
`--diagnostic-on-inadequate-sfm` only to investigate the models, because its
result is unconditionally non-promotable.

Do not assume that merely installing the Depth Pro package makes it active. The
comparison command explicitly requires the external Python, checkpoint, and
pinned commit; it never searches a home directory, uses an unpinned checkout,
or downloads a checkpoint. The raw-video reconstruction and frozen A/B stages
remain separate so a model cannot silently alter masks or camera poses during
the comparison.

For adapter diagnostics, inspect the real internal runner interface with:

```bash
python scripts/depth_pro_batch.py --help
```

The runner is normally invoked by `DepthProSubprocessBackend`, which creates the
hashed input manifest and supplies the checkpoint hash. Calling it manually is
not a substitute for the frozen-input comparison or its evidence gate.

Every production A/B report should contain:

- source-video, frame, mask, SfM-model, model-weight, and checkpoint hashes;
- the Depth Pro source commit, package versions, device, and precision;
- the exact shared frame IDs, track split, view pairs, and thresholds;
- a report for each backend, including rejection reasons;
- the selection or consensus reason; and
- the identities of the depth maps actually admitted to fusion.

Use a new output directory for every run. Reusing a directory would make it
unclear which model, source bytes, or evidence split produced an artifact.

## Capture expectations for Mac video

For the strongest existing-video result:

- keep the object rigid and rotate it slowly through a broad orbit;
- keep it fully in frame and make the hand occupy as little of its silhouette as
  practical;
- include opposing views and some modest elevation change;
- use diffuse, stable lighting and avoid motion blur or focus pumping; and
- prefer opaque, textured surfaces.

The pipeline does not require phone metadata or a moving physical camera. It
does require enough repeatable object texture for masked SfM. Soft deformation,
featureless faces, glare, transparency, and permanently hidden surfaces remain
fundamental limitations. A metre-labelled monocular prediction does not remove
the need for a scale reference when dimensional accuracy matters.

## Sources and license boundary

- [Apple `ml-depth-pro`](https://github.com/apple/ml-depth-pro) is the official
  public reference implementation accompanying the Depth Pro paper. Its README
  says the repository model was retrained and is close to, but not identical to,
  the paper model; it also says both sample code and model weights are released
  under the repository's license.
- [Apple's Depth Pro license](https://github.com/apple/ml-depth-pro/blob/main/LICENSE)
  grants a personal, non-exclusive copyright license to use, reproduce, modify,
  and redistribute the Apple software subject to its terms. It does **not** state
  a research-only or non-commercial restriction. It is nevertheless Apple's own
  license, not Spatial's MIT license and not Apache-2.0. The terms include notice
  retention for an entire unmodified redistribution, restrictions on using
  Apple names or marks for endorsement, no implied patent grant, warranty and
  liability disclaimers, and separate terms for listed subcomponents.
- [Apple's acknowledgements](https://github.com/apple/ml-depth-pro/blob/main/ACKNOWLEDGEMENTS.md)
  carries those third-party notices and terms.
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
  documents its model-specific licensing: Small is Apache-2.0, while
  Base/Large/Giant are CC-BY-NC-4.0.

Depth Pro accompanies a research publication, but “reference implementation”
does not mean “research-use-only.” Conversely, the absence of a research-only
label does not merge its code or weights into Spatial's MIT license. Keep the
external installation, provenance, notices, and license review explicit. This
section summarizes the upstream files; it is not legal advice.
