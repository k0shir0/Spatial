# Open-source landscape for a Chromebook-class Spatial

Research date: **2026-07-15**. Priority constraint: **lightweight above all — the
pipeline must run on a Chromebook** (ChromeOS Linux VM, 4–8 GB total RAM,
x86_64 or aarch64, no GPU). All licenses, release dates, wheel tags, and model
sizes were verified against PyPI/GitHub/Hugging Face on the research date;
several claims (USDZ determinism, MobileSAM latency) were tested empirically.
Latency figures for Chromebook-class CPUs are extrapolations unless noted.

> Same contract as `research.md`: design-time research only. The shipped
> runtime stays local, offline, and review-gated.

## Headline conclusions

1. **OpenMVS is the closest maintained open-source CPU-capable end-to-end
   pipeline.** Its documented path connects video keyframes, SfM, dense
   reconstruction, mesh, texture, and GLB, with CUDA optional. It does not,
   however, demonstrate Spatial's full target intersection: arbitrary handheld
   monocular object video, faithful rather than generated geometry, 4–8 GB
   Chromebook operation, Linux x86_64 and aarch64 delivery, permissive
   commercial integration, and deterministic GLB plus USDZ. Its prebuilt Linux
   binary is x64, USDZ is not documented, and AGPL plus the bundled
   research-only IBFS component requires licensing review. COLMAP dense and
   Meshroom need CUDA; WebODM targets aerial jobs with a larger memory floor;
   and splat pipelines remain GPU-oriented. Spatial's visual-hull and fitted
   regularizers therefore remain the practical low-memory fallback, while
   OpenMVS should be benchmarked externally before integration.
2. **USDZ without a Mac is solved with ~40 lines of stdlib code.** A pure-Python
   `zipfile` packer (ZIP_STORED, fixed 1980-01-01 DOS timestamps, 64-byte data
   alignment via extra-field padding) was validated this session: byte-identical
   across runs, `dataOffset % 64 == 0` for every entry, opens in
   `Usd.Stage.Open()`, and passes `UsdUtils.ComplianceChecker` with no packaging
   errors. three.js's `USDZExporter` proves the pattern (AR-QuickLook-valid USDZ
   from plain zip writing, no USD library). Spatial already authors USDA and
   already normalizes ZIP timestamps — this deletes the Apple-only
   `usdzip` dependency and makes USDZ a first-class Linux/Chromebook output.
3. **The CUDA-heavy candidates in `plan.md`/`research.md` (SAM 2 large, ForeHOI,
   DA3-giant, MapAnything) are incompatible with the Chromebook constraint.**
   Every pipeline stage has a lightweight substitute, catalogued below.
4. **The hardest problem in `plan.md` — object-relative 6-DoF pose — has a
   zero-ML, zero-new-dependency answer:** a printed ChArUco board via
   `cv2.aruco` (in the existing `opencv-python-headless` dep, main modules
   since 4.7, aarch64 wheels included). Deterministic, sub-mm-class accuracy at
   turntable distances, and the board frame gives background exclusion for free
   (clip reconstruction to the volume above the board).
5. **Cross-cutting version conflict:** the repo declares `requires-python >=3.10`
   but current onnxruntime (1.27) and rembg (2.0.76) require ≥ 3.11. Bump the
   floor to 3.11 before adding either.

## Target hardware, defined

| Tier | Typical 2026 spec |
|---|---|
| Entry-level | 4 GB RAM, 64 GB eMMC, MediaTek Kompanio 520 (arm64) or Intel N100-class (x86_64) |
| Chromebook Plus (certified floor) | ≥ 8 GB RAM, ≥ 128 GB, 12th-gen i3 / Ryzen 3 / Kompanio Ultra |

Working targets: **two ISAs (x86_64 + aarch64), ≤ 2 GB peak pipeline RSS** so
4 GB devices survive, ≤ ~500 MB installed toolchain, no GPU ever (ChromeOS 147
"Baguette" officially dropped Linux GPU acceleration).

## Chromebook architecture

**Primary: Crostini/Baguette CLI + Chrome for everything the VM can't do.**

- **Capture** happens outside the VM — the Linux container still cannot access
  the webcam (Chromium issue 907701, open since 2018, verified still true).
  Use the ChromeOS Camera app → Files → "Share with Linux" (`/mnt/chromeos/`),
  or later a getUserMedia web page.
- **Process** inside the VM (Debian 13 under Baguette): every core dep ships
  official manylinux aarch64 wheels (opencv-python-headless 5.0.0.93 is
  36.5 MB on aarch64; numpy/Pillow/scikit-image likewise; trimesh is pure
  Python); FFmpeg 7.1.5 via apt. Distribute with `uv tool install` (uv is a
  static binary, first-class on both ISAs).
- **Review** in the Chrome browser: a `spatial review` command serving a static
  page on `localhost:<port>` (Crostini forwards it) using **`<model-viewer>`**
  4.3.1 for GLB — and the same GLB/USDZ pair gives free phone AR preview.

**Phase-2 option: a fully-local Chrome PWA** — getUserMedia capture, WebCodecs
hardware-decoded frame extraction (~10× faster than ffmpeg.wasm; ffmpeg.wasm
has a ~2 GB ceiling and 5–10× slowdown — avoid), File System Access API project
folders, OpenCV.js (~9 MB wasm) for silhouette ops, carve core compiled to
wasm. Zero-install and works on school-managed devices where Linux is
admin-disabled, but it is a second codebase with a ~2 GB wasm32 working-set
ceiling. No prior art exists (no maintained WASM photogrammetry project) —
an open niche, not a reuse opportunity. Do it after the Crostini path, not
before.

## Recommended stack by pipeline stage

### Pose (object-relative 6-DoF) — Tier 0, zero new deps

- **ChArUco board via `cv2.aruco`** (Apache-2.0, existing dep, aarch64 OK).
  Generate the board with `CharucoBoard.generateImage`; per frame
  `CharucoDetector.detectBoard` → `Board.matchImagePoints` → `cv2.solvePnP` +
  `solvePnPRefineLM`; require ≥ 6 corners, gate on reprojection error and
  temporal consistency. A board (not a single marker) avoids the two-fold IPPE
  pose ambiguity and tolerates the object occluding its center. Deterministic
  with `cv2.setNumThreads(1)` and a pinned wheel.
- Optional extra `[apriltag]`: **pupil-apriltags** (BSD-2, active) — but no
  Linux aarch64 wheel, so never a core dep. Other AprilTag bindings are dead.
- Markerless fallback `[sfm]`, **x86_64 only**: **pycolmap ≥ 4.1** (BSD-3;
  CPU SIFT is BSD-2 VLFeat; the PyPI wheel is CPU-only). Key wins verified in
  the 4.1.0 wheel: native `ImageReaderOptions.mask_path` masking (solves
  rotating-object/static-background at extraction), a `random_seed` parameter
  making reconstruction fully deterministic (new in COLMAP 4.0), and GLOMAP's
  global mapper absorbed upstream (drift-free 360° orbits). ~15–45 min,
  < 2 GB for 100–150 frames on a modest 4-core x86_64. No aarch64 wheels exist.
- Optional learned matching tier: **LightGlue + DISK/ALIKED ONNX**
  (Apache-2.0/BSD-3, ~48 MB) — slow on Chromebooks (~4–8 s/pair), keep
  experimental. **SuperPoint weights are non-commercial — never ship.**

### Segmentation and masks

- **Core (torch-free, ~110 MB): onnxruntime + MobileSAM ONNX + PyMatting.**
  - **onnxruntime** 1.27 (MIT, 16–19 MB wheels, official aarch64, Python
    ≥ 3.11). CPU EP is run-to-run bit-identical on the same machine; pin
    `intra_op_num_threads=1` for strict reproducibility.
  - **MobileSAM** (Apache-2.0; encoder 28 MB + decoder 16 MB fp32, ~20 MB
    int8). Measured on a fast laptop: 263 ms encode, **15 ms per click** after
    encoding — ideal for the review-gated flow: encode each keyframe once, let
    the reviewer click interactively, persist prompt coordinates + model hash
    in the run manifest. `samexporter` (MIT) is the model-prep tool.
    Quality upgrade option: EfficientViT-SAM L0 (Apache-2.0, 140 MB).
    Smallest option: SlimSAM-77 int8 (~14 MB total, but slower than MobileSAM).
  - **PyMatting** (MIT, 54 KB + numba) for mask → alpha edges: non-neural
    closed-form matting, fully deterministic, no weights; trimap derives
    mechanically from the binary mask via erode/dilate. ~1 s at 512² measured.
    rembg already depends on it for exactly this step.
- **Keep and extend the existing rembg extra** (verified MIT, v2.0.76
  June 2026, actively maintained): **u2netp (4.6 MB)** automatic-preview tier,
  **isnet-general-use (179 MB)** automatic-quality tier. Pre-fetch models from
  rembg's release bucket — pooch pins MD5 checksums, satisfying
  no-network-at-runtime with integrity checking for free. Could replace the
  temporal-change coverage heuristic in `ingest.py` with real object masks.
  **Never enable the `bria-rmbg` session (CC-BY-NC).**
- **Video mask propagation `[video]` extra: EdgeTAM** (Apache-2.0, 56 MB,
  Meta, pushed 2026-01). Hard fact: **no torch-free full-video ONNX/ncnn VOS
  port exists as of mid-2026** (SAM2 memory attention fails ONNX export), so
  true propagation costs the PyTorch CPU wheel (+192 MB x86_64 / +155 MB
  aarch64, from the pytorch.org CPU index). EdgeTAM is a SAM2 drop-in with
  official CPU support, 22× faster than SAM 2; fallbacks in the same API:
  EfficientTAM-ti/512 (68 MB, proven CPU-only in Sammie-Roto), SAM 2.1
  hiera_tiny (156 MB, reference tier). Downsample to ≤ 125 frames at 480p
  (SAM2-family loads every frame as ~12.6 MB fp32 tensor). Strict-MIT
  alternative: Cutie-small (109 MB, memory-bounded, needs a small CPU patch).
  Study `muggled_sam` (Apache-2.0) for a RAM-bounded reimplementation.
  - **Torch-free degraded mode for core:** per-frame MobileSAM with prompt
    carryover (previous mask centroid/bbox seeds the next frame). Drift-prone
    but deterministic and frame-by-frame reviewable — fits the review-gated
    ethos.
- **Hand exclusion:** nothing is simultaneously pixel-accurate, < 50 MB,
  permissive, and maintained. Workable pattern: **PINTO0309's Apache-2.0/MIT
  ONNX ports of MediaPipe palm-detection + hand landmarks** (runs on
  onnxruntime, solves the aarch64 gap — the official mediapipe wheel has had
  no Linux aarch64 build since Nov 2024) for localization, converted to pixels
  via PINTO zoo #380 skin-segmentation ONNX (MIT) or ROI-seeded skin
  thresholding; subtract from the object mask and flag heavily-occluded frames
  for review. Note the propagated object mask already excludes hands most of
  the time — this is a correction pass.

### Depth and surface refinement

- **Tier 1, zero new deps (recommended before any ML):**
  - **Photo-consistency carving** (Kutulakos–Seitz variance test on the
    existing hull voxel grid, pure numpy, seconds–minutes) — recovers
    concavities silhouettes can't see.
  - **`cv2.StereoSGBM` pairwise depth + grid/TSDF fusion** for texture-rich
    objects: rectify orbit pairs (i, i±3..6), disparity-bounded by the hull,
    fuse ~100 depth maps, marching cubes. Deterministic, est. 2–10 min,
    < 1 GB. No pip package fills this niche — it is exactly what Spatial can
    own. (Reference code to imitate: OpenSfM's CPU dense module, BSD-2.)
- **`[depth]` extra: Depth Anything V2 — Small** via onnxruntime.
  **License trap, verified per-variant: only Small is Apache-2.0; Base, Large,
  and Giant are CC-BY-NC-4.0.** `onnx-community/depth-anything-v2-small`:
  int8 **27.3 MB**, q4f16 19.1 MB; est. 0.3–1 s/frame at ~392 px on a
  Chromebook. Treat as uncertain evidence for hull refinement, per `plan.md`.
  Drop-in quality upgrade to evaluate: Distill-Any-Depth Small (verified
  Apache-2.0, identical architecture/cost). Watch DA3-SMALL (Apache-2.0, no
  quantized ONNX yet). MiDaS v2.1 small (66.8 MB, MIT, repo archived 2025) is
  a faster/rougher fallback.
- **Visual hull:** keep the in-repo numpy implementation; borrow two ideas —
  Open3D's conservative voxel-**corner** projection and vacancy's
  **carve-to-SDF-instead-of-bool** trick (smoother marching cubes at the same
  grid), plus octree coarse-to-fine (carve 64³, subdivide boundary voxels →
  effective 512³ surface in O(surface) work). Open3D itself is ruled out as a
  dep (100–450 MB wheels, x86_64-only, release stale since Jan 2025).

### Mesh processing and export

- **Repair/watertight:** trimesh built-ins → **manifold3d** (Apache-2.0,
  1.3 MB, aarch64 wheels, very active; already trimesh's boolean engine).
  pymeshfix is AGPL — ruled out.
- **Decimation:** **fast-simplification** (MIT, ~1.7 MB, aarch64 abi3 wheels,
  pushed 2026-07) — it is the backend behind
  `trimesh.Trimesh.simplify_quadric_decimation()`. Equivalent alternative:
  pyfqmr (MIT, 0.9 MB). Pick one and pin. PyMeshLab ruled out (GPL + 100 MB);
  libigl ruled out (heavy, patchy ARM wheels).
- **UV unwrapping:** **xatlas-python** (MIT, ~0.3 MB) — the only serious
  permissive option. No aarch64 wheel; document the sdist build on ARM or
  contribute a cibuildwheel aarch64 job upstream (its CI is already
  cibuildwheel).
- **GLB:** trimesh export (pinned — byte layout can shift between releases),
  **pygltflib** (MIT, pure Python) for post-export metadata surgery.
  **No Draco, no meshopt compression, no KTX2 by default** — QuickLook never
  reads glTF, USDZ forbids GPU texture compression entirely, and compressed
  GLB breaks the "one deterministic texture set shared by both outputs"
  property. PNG/JPEG only.
- **USDZ (the big one):** pure-Python packer as described in headline #2 —
  `zipfile`, ZIP_STORED, fixed `date_time=(1980,1,1,0,0,0)`, 64-byte alignment
  via extra-field padding. Retire/fold in `_normalize_zip_timestamps`.
  - Optional `[usd]` extra (**x86_64 only** — usd-core has no aarch64 wheel):
    **usd-core 26.5** (TOST-1.0 permissive, 28.8 MB wheel) for deterministic
    usda→usdc conversion (verified byte-stable; smaller, faster-loading USDZ)
    and `UsdUtils.ComplianceChecker(arkit=True)` — the engine behind
    `usdchecker --arkit`, callable from Python (currently deprecation-warned
    toward the new Usd Validation Framework; plan the migration).
  - Watchlist: **TinyUSDZ** (Apache-2.0, 5.5 MB wheels incl. aarch64, very
    active — best lightweight USD *reader* today; adopt for packaging if its
    USDZ *writing* graduates from experimental) and NVIDIA `usd-exchange`
    (Apache-2.0, ~20 MB aarch64 wheels, unverified authoring API).
- **CI validation (x86_64 runners):** Khronos glTF-Validator binary (no ARM
  build) for every GLB; usd-core ComplianceChecker for every USDZ; keep the
  existing golden-hash tests but scope byte-reproducibility per
  (platform, pinned-versions) pair — native FP stages (xatlas, decimation,
  manifold3d) should not be advertised as bit-identical across x86 vs ARM.

## License rule-outs (record these; several are technically excellent)

| Project | Why ruled out |
|---|---|
| EdgeSAM, MatAnyone | S-Lab non-commercial |
| DEVA | CC-BY-NC-SA |
| FastSAM / anything ultralytics | AGPL-3.0 (FastSAM also architecturally wrong — no true point prompts) |
| BRIA RMBG 1.4/2.0 | non-commercial weights (rembg session `bria-rmbg`) |
| RobustVideoMatting | GPL-3.0 + human-only |
| SuperPoint weights | Magic Leap research-only, no redistribution |
| VGGT, DUSt3R, MASt3R | non-commercial (and far too heavy anyway) |
| Depth Anything V2 Base/Large/Giant | CC-BY-NC (Small is Apache-2.0 — the only safe variant) |
| Metric3D v2 | ambiguous weights license + heavy |
| Apple Depth Pro | research-only weights, 2.2 GB |
| OpenMVS | AGPL-3.0 + bundled IBFS is "research purposes only" |
| PyMeshLab | GPL-3.0 (+ ~100 MB) |
| pymeshfix | AGPL-3.0 |
| hands-segmentation-pytorch | no license at all |

Technology rule-outs: COLMAP dense stereo (CUDA-only, verified compiled out of
the PyPI wheel), NanoSAM (TensorRT-only, dead), XMem (superseded by Cutie),
MODNet (portrait-only), EgoHOS (5 GB, dead stack), Open3D/PyTorch3D as deps
(packaging weight), OpenSfM (not on PyPI, dominated by pycolmap), TheiaSfM
(dead), Meshroom/AliceVision (CUDA MVS), OpenSplat/gsplat (GPU-bound in
practice), ffmpeg.wasm for full videos (2 GB ceiling, 5–10× slowdown — use
WebCodecs in-browser instead), glTF-Transform at runtime (Node dep; fine
for CI).

## Suggested adoption order

1. **Pure-Python USDZ packer** — small, deletes the Apple-only dependency,
   immediately makes the full GLB+USDZ pipeline Chromebook-viable. Add
   usd-core `[usd]` extra + CI compliance checks on x86_64.
2. **ChArUco pose module** (`cv2.aruco`, zero new deps) + capture-contract
   docs for a printed board — unlocks calibrated multi-view for the visual
   hull and photo-consistency carving.
3. **fast-simplification + manifold3d** (~3 MB combined, aarch64-ready) —
   decimation + watertight repair for hull output.
4. **Python ≥ 3.11 floor bump**, then **MobileSAM + PyMatting on onnxruntime**
   as the click-to-mask review tool; extend the rembg extra with pinned
   pre-fetched u2netp/isnet models.
5. **SGBM pairwise refinement + photo-consistency carving** (zero new deps).
6. Optional extras as demand appears: `[video]` EdgeTAM (torch-cpu),
   `[depth]` Depth Anything V2 Small int8, `[sfm]` pycolmap (x86_64).
7. **`spatial review` localhost UI** with `<model-viewer>`; `uv tool install`
   as the documented Chromebook install path.
