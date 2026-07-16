# Board-based capture protocol (no LLM anywhere)

This is the capture contract that makes reconstruction a measurement problem.
Every step below is deterministic classical computer vision or plain
photography; no step consults an LLM, the network, or an object catalog. The
only optional learned component is a small fixed-weight local segmenter for
mask scoring, and it can be skipped.

## Why a printed board

A ChArUco board in frame turns "is this a good frame?" and "where is the
camera?" into measurements:

- **Pose**: chessboard corners + embedded markers give a 6-DoF board pose per
  frame via PnP (`local3d.charuco_pose`). Frames with too few corners or high
  reprojection error fail closed.
- **Frame selection**: keyframes are chosen by azimuth coverage of the orbit
  (`scripts/select_board_frames.py`) — geometry, not heuristics.
- **Scale**: the printed square size is known, so poses are metric when the
  print is verified and intrinsics are calibrated.
- **Background exclusion**: the board plane defines the volume above it;
  anything outside that volume is not the object.

## One-time setup

```bash
python scripts/make_charuco_board.py --output board/charuco_board.png
```

Print at 100% scale (the PNG embeds 300 DPI). Verify with a ruler that one
square measures exactly 30 mm — if it does not, poses will be silently wrong.
Tape the sheet completely flat to something rigid.

## Capture rules

1. **Object stationary on the board** (or board + object rotating together on
   a turntable). Never rotate the object in your hand: hands occlude
   silhouettes and break the rigid-scene assumption.
2. **Hands fully out of frame** while recording. Pause, regrip, resume if you
   must touch the object.
3. **Camera on a stand or held very steady**; move in slow arcs. Record one
   level orbit, one high orbit (30-45 degrees down), and, after flipping the
   object onto a different resting face, a second pass for the previously
   hidden side.
4. **Lock focus and exposure** before recording; avoid mixed/backlit light.
5. Keep the object 40-70% of frame height, board corners visible around it.
6. Matte objects work; glass, mirrors, and fur violate silhouette and
   photo-consistency assumptions (see README capture limits).

## Processing pipeline

```bash
# 1. Extract candidates; score with real object masks (optional segmenter).
spatial-ingest capture.mov --output run/ingest --segmenter u2netp

# 2. Estimate board poses and select keyframes by orbit coverage (zero-ML).
python scripts/select_board_frames.py run/ingest/frames --output run/poses

# 3. Object masks for the selected keyframes (review the contact sheet!).
python scripts/auto_masks.py run/poses/keyframes --output run/masks

# 4. After operator review of masks and pose overlays: carve, decimate, export.
python scripts/build_visual_hull.py run/masks/masks --output run/hull.glb \
    --views-json <derived from run/poses/poses.json> --target-triangles 2000
```

Every stage writes JSON with gates and provenance; every stage fails closed
when support is missing (no board → no pose → no reconstruction). Masks are
review input: inspect `mask_contact_sheet.png` before carving, and never ship
an asset whose texture you have not looked at.

## What each stage depends on

| Stage | Dependency class |
|---|---|
| Board render/print | OpenCV drawing (deterministic) |
| Ingest scoring | classical CV; optional 4.6 MB u2netp via onnxruntime CPU (fixed weights, no LLM) |
| Pose + frame selection | OpenCV ChArUco + PnP (deterministic, zero-ML) |
| Masks | u2netp/isnet fixed weights, local, deterministic across runs |
| Hull + decimation + export | numpy/scikit-image/trimesh + optional fast-simplification/manifold3d |
| GLB/USDZ packaging | pure Python, byte-reproducible |
