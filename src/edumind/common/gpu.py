"""Serialize application model work and release unused runtime allocations."""

from __future__ import annotations

import gc
import sys
from contextlib import contextmanager
from threading import RLock

_WORKLOAD_LOCK = RLock()


@contextmanager
def exclusive_model_workload():
    """Serialize uploads and answers across the shared in-process application."""
    with _WORKLOAD_LOCK:
        yield


def release_runtime_memory() -> None:
    """Collect released models without importing or initializing GPU frameworks."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    paddle = sys.modules.get("paddle")
    if (
        paddle is not None
        and paddle.is_compiled_with_cuda()
        and paddle.device.get_device().startswith("gpu")
    ):
        paddle.device.cuda.empty_cache()
