"""Repository root used by the direct application and experiment scripts."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = (
    PACKAGE_DIR.parents[1] if (PACKAGE_DIR.parents[1] / "pyproject.toml").exists() else Path.cwd()
)
