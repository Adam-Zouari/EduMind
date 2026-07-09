"""Configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .paths import resolve_config_path


def load_yaml_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the shared YAML config file."""
    resolved = resolve_config_path(config_path)
    with resolved.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
