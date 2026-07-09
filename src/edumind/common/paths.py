"""Project path helpers."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve a config path against the repository config directory."""
    if config_path is None:
        return CONFIG_DIR / "base.yaml"

    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    return (CONFIG_DIR / candidate).resolve()


def artifact_path(*parts: str, create: bool = True) -> Path:
    """Build a path inside the artifacts directory."""
    path = ARTIFACTS_DIR.joinpath(*parts)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
