"""Shared runtime helpers for audio and video OCR extractors."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

from ..config import FFMPEG_PATH, WHISPER_DEVICE

torch: Any | None = None
whisper: Any | None = None
WHISPER_AVAILABLE = False
_IMPORT_ERROR: Exception | None = None
_RUNTIME_READY = False
_MODEL_CACHE: dict[tuple[str, str], object] = {}
_MODEL_LOCK = Lock()


def _configure_runtime() -> None:
    global _RUNTIME_READY, WHISPER_AVAILABLE, torch, whisper, _IMPORT_ERROR
    if _RUNTIME_READY:
        return

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    if FFMPEG_PATH != "ffmpeg" and Path(FFMPEG_PATH).exists():
        ffmpeg_dir = str(Path(FFMPEG_PATH).parent)
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    try:
        import torch as torch_module
        import whisper as whisper_module

        torch_module.set_num_threads(1)
        torch = torch_module
        whisper = whisper_module
        WHISPER_AVAILABLE = True
        _IMPORT_ERROR = None
    except Exception as exc:  # pragma: no cover - environment-dependent optional stack
        torch = None
        whisper = None
        WHISPER_AVAILABLE = False
        _IMPORT_ERROR = exc

    _RUNTIME_READY = True


def get_whisper_device() -> str:
    """Resolve the configured Whisper device with safe fallback behavior."""
    _configure_runtime()

    requested = WHISPER_DEVICE or "cpu"
    if requested == "auto":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        if torch is not None and getattr(getattr(torch, "backends", None), "mps", None):
            if torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    if requested.startswith("cuda"):
        if torch is not None and torch.cuda.is_available():
            return requested
        return "cpu"

    if requested == "mps":
        if torch is not None and getattr(getattr(torch, "backends", None), "mps", None):
            if torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    return "cpu"


def load_whisper_model(model_name: str) -> tuple[object | None, str | None]:
    """Load and cache a Whisper model for the configured runtime."""
    _configure_runtime()

    if not WHISPER_AVAILABLE or whisper is None:
        detail = f": {_IMPORT_ERROR}" if _IMPORT_ERROR else "."
        return None, f"Whisper is not available{detail}"

    device = get_whisper_device()
    cache_key = (model_name, device)

    with _MODEL_LOCK:
        if cache_key not in _MODEL_CACHE:
            try:
                _MODEL_CACHE[cache_key] = whisper.load_model(model_name, device=device)
            except Exception as exc:  # pragma: no cover - environment-dependent optional stack
                return None, f"Failed to load Whisper model '{model_name}' on {device}: {exc}"

    return _MODEL_CACHE[cache_key], None
