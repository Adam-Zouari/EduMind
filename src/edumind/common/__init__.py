"""Shared helpers for EduMind-AI."""

from .config import load_yaml_config
from .paths import PROJECT_ROOT

__all__ = [
    "PROJECT_ROOT",
    "load_yaml_config",
]
