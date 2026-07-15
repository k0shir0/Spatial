# VisionMaskCLI

A small, dependency-free macOS command-line tool that creates object masks from
foreground instances. It uses only Apple's built-in Vision, Core Graphics, and
Image I/O frameworks. There are no downloaded weights, Python environments,
PyTorch, or MPS processes.

The default mode retains the original conservative green-tin selector. For
other objects, pass a normalized point that remains on the object throughout
the selected frames:

```bash
.build/release/vision-mask \
  --input <keyframes> --output <masks> \
  --include 'frame_*.jpg' --object-seed 0.5,0.45
```

Seeded mode is still fail-closed. It rejects frames when Vision merges the
object with the holder, the seed lands on background, occluder subtraction
removes the seed, or coverage is unsafe. It is not a video mask propagator;
every emitted mask must still pass temporal and visual review.

The tool runs one frame at a time. For each frame it:

1. Requests foreground instances and a person-confidence mask from Vision.
2. Chooses the near-center instance with the strongest green evidence.
3. Removes high-confidence person pixels and skin-colored pixels with a conservative inward margin.
4. Retains the connected green object component.
5. Writes a binary, one-channel, 8-bit PNG with the same filename stem and pixel dimensions as its keyframe.
6. Decodes and validates the saved PNG immediately.

It fails rather than writing a mask when green evidence, foreground coverage, format, or dimensions are unsafe.

## Requirements

- macOS 14 or newer.
- Xcode command-line tools.

## Build and preflight

```bash
cd native/VisionMaskCLI
swift build -c release

.build/release/vision-mask \
  --input ../../runs/sample/object_capture/keyframes_ffmpeg \
  --output ../../runs/sample/object_capture/masks_vision \
  --include 'frame_*.jpg' \
  --dry-run
```

`--dry-run` decodes and checks every matching input and detects output collisions. It does not issue Vision requests, create the output directory, or write files. The explicit include pattern matters in the sample directory because `contact_sheet.jpg` is a review artifact, not a reconstruction frame.

## Generate masks

```bash
.build/release/vision-mask \
  --input ../../runs/sample/object_capture/keyframes_ffmpeg \
  --output ../../runs/sample/object_capture/masks_vision \
  --include 'frame_*.jpg'
```

Outputs are `frame_0001.png`, `frame_0002.png`, and so on, plus a small `mask_report.json` with selection and exclusion metrics. Existing outputs are never replaced unless `--overwrite` is passed.

For a visual review set, add a separate debug directory:

```bash
  --debug-dir ../../runs/sample/object_capture/masks_vision_review
```

Each frame then gets a transparent `*_cutout.png` and an `*_overlay.png`. The overlay dims excluded pixels, tints included pixels green, and draws the final boundary yellow. Review images are deliberately kept outside the mask directory so Object Capture cannot ingest them as masks.

The defaults use a person confidence threshold of `0.82`, a six-pixel occluder margin, and one-pixel object-boundary erosion. Increase `--occluder-margin` if skin fringe remains. Do not lower `--person-threshold` casually: overly broad person segmentation can remove valid tin pixels.

## Validate existing masks

```bash
.build/release/vision-mask \
  --input ../../runs/sample/object_capture/keyframes_ffmpeg \
  --output ../../runs/sample/object_capture/masks_vision \
  --include 'frame_*.jpg' \
  --validate-only
```

Validation requires exactly one matching PNG per selected input, original pixel dimensions, one 8-bit grayscale channel, strictly binary values, and plausible foreground coverage. It performs no Vision inference and writes nothing.

Inspect the masks before reconstruction, especially every hand/object contact edge. Native segmentation cannot recover a surface that is physically hidden by a finger; a second capture with a different grip is still needed for complete geometry.
