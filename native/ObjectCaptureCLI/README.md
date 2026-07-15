# ObjectCaptureCLI

A dependency-free, local macOS backend for reconstructing a textured USDZ model from segmented keyframes. It wraps RealityKit's `PhotogrammetrySession`; no images leave the Mac and no third-party model weights are required.

## Requirements

- macOS 15 or newer.
- A Mac on which `PhotogrammetrySession.isSupported` is true. The Apple M5 development machine in this project reports `true` when checked from a normal Terminal process.
- Xcode command-line tools (`xcrun swift --version`).
- At least three images; 20–60 sharp, overlapping views are a better practical range.

Build:

```bash
cd native/ObjectCaptureCLI
swift build -c release
```

Smoke-test with RealityKit's automatic foreground masking:

```bash
.build/release/object-capture \
  --input /absolute/path/to/keyframes \
  --output /absolute/path/to/output/preview.usdz \
  --detail preview
```

For hand-held video, supply explicit object-only masks. Automatic masking may include the hands, making them part of the reconstructed mesh:

```bash
.build/release/object-capture \
  --input /absolute/path/to/keyframes \
  --masks /absolute/path/to/masks \
  --output /absolute/path/to/output/model.usdz \
  --detail reduced
```

The keyframe and mask names must share a stem, and every mask must be the same pixel size as its image:

```text
keyframes/frame_0001.jpg    masks/frame_0001.png
keyframes/frame_0002.jpg    masks/frame_0002.png
```

Masks are one-channel images in which black pixels are excluded and nonblack pixels are object. Use a small inward erosion at hand/object contact rather than allowing skin pixels into the mask.

The defaults are suited to video-derived frames: sequential sample ordering, high feature sensitivity, and object masking. Start with `preview`, inspect registration and hand contamination, then request `reduced` or `medium`. Use `--force` only when intentionally replacing an existing output.

## Extracting sample keyframes

This example produces about 35 ordered frames from the first 11.5 seconds
without altering the source video:

```bash
mkdir -p work/keyframes
ffmpeg \
  -i '/path/to/capture.mov' \
  -t 11.5 \
  -vf 'fps=3' \
  -q:v 2 \
  work/keyframes/frame_%04d.jpg
```

Do not expect a no-mask smoke test to produce a clean asset from a hand-held
clip. The first meaningful test is segmented keyframes with hand pixels
excluded.

## Output and conversion

USDZ is the native Object Capture result and can be previewed directly in macOS. `/usr/bin/usdzip` and `/usr/bin/usdcat` are present for inspection/unpacking. A GLB exporter is not installed on the development machine; add a local conversion stage later rather than weakening the native reconstruction path.

## Sandbox note

`PhotogrammetrySession.isSupported` can return `false` inside a restricted automation sandbox even though it returns `true` from Terminal on the same Mac. Build and run the reconstruction binary directly from a normal Terminal session.
