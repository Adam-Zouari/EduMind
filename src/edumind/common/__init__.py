"""Shared helpers for EduMind-AI."""

from .config import load_yaml_config
from .paths import (
    ARTIFACTS_DIR,
    CONFIG_DIR,
    DATA_DIR,
    DOCS_DIR,
    PROJECT_ROOT,
    artifact_path,
    resolve_config_path,
)

__all__ = [
    "ARTIFACTS_DIR",
    "CONFIG_DIR",
    "DATA_DIR",
    "DOCS_DIR",
    "PROJECT_ROOT",
    "artifact_path",
    "load_yaml_config",
    "resolve_config_path",
]
