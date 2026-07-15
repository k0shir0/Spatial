# Contributing

Thanks for helping improve Spatial.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
pytest -q
```

FFmpeg/ffprobe is needed for the ingest tests. Apple USD tests skip when the
macOS tools are unavailable.

## Pull requests

- Keep source media, extracted frames, generated assets, and model checkpoints
  out of commits.
- Add tests for schema, topology, export, and safety changes.
- Preserve resource ceilings and fail-closed validation.
- Do not add wall-clock metadata or unseeded randomness to deterministic
  artifact paths.
- Document whether a feature uses learned inference, network access, GPU work,
  or third-party weights.
- Record hashes when adding a curated deterministic fixture.
- Update public documentation when configuration contracts change.

Run the synthetic demo twice when changing the parametric exporter:

```bash
python scripts/make_demo_inputs.py --output examples/demo
spatial-parametric --config examples/parametric_phone.json --output /tmp/spatial-a --skip-usdz
spatial-parametric --config examples/parametric_phone.json --output /tmp/spatial-b --skip-usdz
cmp /tmp/spatial-a/spatial_demo_phone.glb /tmp/spatial-b/spatial_demo_phone.glb
```

## Privacy

Never post a user's capture, face, screen, absolute local path, or textured
output in a pull request or public issue without explicit permission. Prefer
programmatically generated fixtures.
