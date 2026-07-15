#!/usr/bin/env python3
"""Audit exact-frame decoding across a directory of texture-source videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.texture_bake import audit_video_directory, load_texture_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--texture-config",
        type=Path,
        help="mark the single video whose hash matches this fitted-model config",
    )
    args = parser.parse_args()
    source_hash = None
    if args.texture_config:
        source_hash = load_texture_config(args.texture_config)["source_video_sha256"]
    report = audit_video_directory(
        args.directory,
        args.output,
        texture_source_sha256=source_hash,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
