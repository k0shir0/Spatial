# VisionClassifyCLI

`vision-classify` is an optional, local semantic-hint helper for the shape
router. It uses Apple's built-in `VNClassifyImageRequest`; the package contains
no model weights and performs no downloads, network access, or Torch work.
The classification itself is learned inference provided by the operating
system, not an LLM and not a model bundled with this repository.

The output is deliberately advisory. It can emit only a high-confidence
`phone`, `tin`, `bottle`, cylindrical `can`, or `book` hint. It never chooses,
overrides, or dispatches geometry. A phone, tin, and closed book can all have a
slab-like geometry family, so silhouette classification remains the source of
truth. Book evidence uses a narrow allowlist such as `book`, `paperback`, and
`textbook`; generic labels such as `document`, `paper`, and `printed page` do
not map to a book.

## Build and test

```bash
cd native/VisionClassifyCLI
swift test
swift build -c release
```

## Classify masked keyframes

Masks are optional but strongly recommended. They must be binary, same-size PNG
files with the same filename stem as each source image. Missing masks cause the
corresponding images to be skipped and recorded rather than classified with
hands and background present.

```bash
.build/release/vision-classify \
  --input /path/to/keyframes \
  --masks /path/to/masks \
  --include 'frame_*.jpg' \
  --output /path/to/semantic-hints.json
```

Use `--dry-run` to validate discovery and mask matching without issuing Vision
requests or writing files. Existing output is preserved unless `--overwrite`
is passed.

The JSON includes per-frame Vision labels, conservative mapped evidence,
cross-frame aggregate scores, skipped images, thresholds, and an explicit
safety contract:

```json
{
  "contract": {
    "advisory_only": true,
    "dispatches_geometry": false,
    "overrides_geometry": false
  },
  "semantic_hint": {
    "status": "accepted | abstained",
    "hint": "phone | tin | bottle | can | book | null"
  }
}
```

Abstention is a correct outcome. It means the built-in classifier did not
provide repeatable, sufficiently separated semantic evidence; it does not mean
that silhouette geometry classification failed.
