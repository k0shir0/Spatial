#!/usr/bin/env python3
"""Repository-local entry point for the safe parametric asset builder."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.parametric_assets import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
