#!/usr/bin/env python3
"""Repository-local entry point for :mod:`local3d.shape_router`."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from local3d.shape_router import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
