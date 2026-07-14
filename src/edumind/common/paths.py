"""Stable package and local-state path helpers."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = (
    PACKAGE_DIR.parents[1] if (PACKAGE_DIR.parents[1] / "pyproject.toml").exists() else Path.cwd()
)
SRC_DIR = PACKAGE_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
ARTIFACTS_DIR = (
    Path(os.getenv("EDUMIND_ARTIFACTS", PROJECT_ROOT / "artifacts")).expanduser().resolve()
)


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve an explicit config; packaged defaults are handled by common.config."""
    if config_path is None:
        from .config import default_config_path

        return default_config_path()
    candidate = Path(config_path).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    return candidate


def artifact_path(*parts: str, create: bool = True) -> Path:
    """Build a path inside the configured local artifact directory."""
    path = ARTIFACTS_DIR.joinpath(*parts)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
