"""GPU monitoring utilities for maintained experiments."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import pynvml

    _PYNVML_IMPORTED = True
except ImportError:
    _PYNVML_IMPORTED = False

_NVML_INITIALIZED = False


def _ensure_nvml() -> bool:
    """Initialize NVML lazily when utilization data is requested."""
    global _NVML_INITIALIZED

    if not _PYNVML_IMPORTED:
        return False
    if _NVML_INITIALIZED:
        return True

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        logger.debug("Could not initialize NVML: %s", exc)
        return False

    _NVML_INITIALIZED = True
    return True


def is_cuda_available() -> bool:
    """Return whether CUDA is available through torch."""
    return bool(_TORCH_AVAILABLE and torch.cuda.is_available())


def get_gpu_memory_usage(device_id: int = 0) -> dict[str, float]:
    """Return current GPU memory metrics in megabytes."""
    metrics = {
        "allocated_mb": 0.0,
        "reserved_mb": 0.0,
        "free_mb": 0.0,
        "total_mb": 0.0,
    }
    if not is_cuda_available():
        return metrics

    try:
        metrics["allocated_mb"] = torch.cuda.memory_allocated(device_id) / (1024 ** 2)
        metrics["reserved_mb"] = torch.cuda.memory_reserved(device_id) / (1024 ** 2)
        if _ensure_nvml():
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["total_mb"] = memory_info.total / (1024 ** 2)
            metrics["free_mb"] = memory_info.free / (1024 ** 2)
    except Exception as exc:
        logger.warning("Error getting GPU memory usage: %s", exc)
    return metrics


def get_gpu_utilization(device_id: int = 0) -> dict[str, float]:
    """Return current GPU and memory utilization percentages."""
    metrics = {
        "gpu_utilization_percent": 0.0,
        "memory_utilization_percent": 0.0,
    }
    if not _ensure_nvml():
        return metrics

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        metrics["gpu_utilization_percent"] = float(utilization.gpu)
        metrics["memory_utilization_percent"] = float(utilization.memory)
    except Exception as exc:
        logger.warning("Error getting GPU utilization: %s", exc)
    return metrics


def get_gpu_info(device_id: int = 0) -> dict[str, Any]:
    """Return a compact GPU-information payload."""
    info: dict[str, Any] = {
        "cuda_available": is_cuda_available(),
        "device_id": device_id,
        "device_name": "N/A",
        "driver_version": "N/A",
        "cuda_version": "N/A",
    }
    if not is_cuda_available():
        return info

    try:
        if _TORCH_AVAILABLE:
            info["device_name"] = torch.cuda.get_device_name(device_id)
            info["cuda_version"] = torch.version.cuda
        if _ensure_nvml():
            driver_version = pynvml.nvmlSystemGetDriverVersion()
            info["driver_version"] = (
                driver_version.decode("utf-8")
                if isinstance(driver_version, bytes)
                else str(driver_version)
            )
    except Exception as exc:
        logger.warning("Error getting GPU info: %s", exc)
    return info


def monitor_gpu_during_execution(
    func: Callable[..., T],
) -> Callable[..., tuple[T, dict[str, float | bool]]]:
    """Wrap a function and return its result together with GPU metrics."""

    @wraps(func)
    def wrapper(*args: object, **kwargs: object) -> tuple[T, dict[str, float | bool]]:
        if not is_cuda_available():
            return func(*args, **kwargs), {"cuda_available": False}

        initial_memory = get_gpu_memory_usage()
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        execution_time = time.perf_counter() - start_time
        final_memory = get_gpu_memory_usage()
        utilization = get_gpu_utilization()
        metrics = {
            "execution_time_sec": execution_time,
            "peak_allocated_mb": final_memory["allocated_mb"],
            "peak_reserved_mb": final_memory["reserved_mb"],
            "memory_delta_mb": final_memory["allocated_mb"] - initial_memory["allocated_mb"],
            "gpu_utilization_percent": utilization["gpu_utilization_percent"],
            "memory_utilization_percent": utilization["memory_utilization_percent"],
        }
        return result, metrics

    return wrapper


def measure_throughput(
    func: Callable[..., T],
    num_items: int,
    *args: object,
    **kwargs: object,
) -> dict[str, float]:
    """Measure batch throughput and optional GPU metrics."""
    start_time = time.perf_counter()
    func(*args, **kwargs)
    execution_time = time.perf_counter() - start_time
    metrics = {
        "num_items": float(num_items),
        "execution_time_sec": execution_time,
        "throughput_items_per_sec": num_items / execution_time if execution_time > 0 else 0.0,
        "latency_per_item_ms": (execution_time * 1000 / num_items) if num_items > 0 else 0.0,
    }
    if is_cuda_available():
        metrics["gpu_memory_mb"] = get_gpu_memory_usage()["allocated_mb"]
        metrics["gpu_utilization_percent"] = get_gpu_utilization()["gpu_utilization_percent"]
    return metrics


def reset_peak_memory_stats(device_id: int = 0) -> None:
    """Reset peak-memory tracking when CUDA is available."""
    if is_cuda_available() and _TORCH_AVAILABLE:
        torch.cuda.reset_peak_memory_stats(device_id)
        torch.cuda.empty_cache()


def get_peak_memory_stats(device_id: int = 0) -> dict[str, float]:
    """Return peak allocated and reserved GPU memory in megabytes."""
    metrics = {
        "peak_allocated_mb": 0.0,
        "peak_reserved_mb": 0.0,
    }
    if not is_cuda_available():
        return metrics

    try:
        metrics["peak_allocated_mb"] = torch.cuda.max_memory_allocated(device_id) / (1024 ** 2)
        metrics["peak_reserved_mb"] = torch.cuda.max_memory_reserved(device_id) / (1024 ** 2)
    except Exception as exc:
        logger.warning("Error getting peak memory stats: %s", exc)
    return metrics
