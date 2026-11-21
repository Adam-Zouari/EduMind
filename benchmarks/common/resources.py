"""Resident process-tree RAM and raw assigned-device NVML memory sampling."""

from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Self


class ResourceMonitor:
    def __init__(
        self,
        interval_seconds: float = 0.05,
        *,
        temporary_directory: Path | None = None,
        require_vram: bool = False,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if require_vram else "cpu")
        if self.device not in {"cpu", "cuda"} or (
            require_vram and self.device != "cuda"
        ):
            raise ValueError(
                "Resource monitoring requires an explicit CPU/CUDA contract"
            )
        if not math.isfinite(interval_seconds) or interval_seconds <= 0:
            raise ValueError("Resource sampling interval must be finite and positive")
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._sample_lock = threading.Lock()
        self._vram_errors = 0
        self._thread: threading.Thread | None = None
        self._peak_ram_bytes = 0
        self._ram_sampled = False
        self._peak_vram_bytes = 0
        self._vram_sampled = False
        self._peak_temporary_bytes = 0
        self._temporary_sampled = False
        self._temporary_directory = temporary_directory
        self._require_vram = require_vram
        self._pynvml: Any | None = None
        self._gpu_handle: Any | None = None
        self._started_at = 0.0
        self._samples: list[dict[str, object]] = []
        self.gpu_identity: dict[str, object] = {}

    def __enter__(self) -> Self:
        if self._started_at:
            raise RuntimeError("ResourceMonitor instances cannot be reused")
        self._started_at = time.perf_counter()
        if self.device == "cuda":
            try:
                import pynvml

                pynvml.nvmlInit()
                self._pynvml = pynvml
                visible = (
                    os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
                )
                self._gpu_handle = (
                    pynvml.nvmlDeviceGetHandleByUUID(visible)
                    if visible.startswith(("GPU-", "MIG-"))
                    else pynvml.nvmlDeviceGetHandleByIndex(int(visible))
                )
                identity = (
                    pynvml.nvmlDeviceGetUUID(self._gpu_handle)
                    if hasattr(pynvml, "nvmlDeviceGetUUID")
                    else visible
                )
                self.gpu_identity["uuid"] = (
                    identity.decode() if isinstance(identity, bytes) else identity
                )
            except Exception as exc:
                self._shutdown()
                if self._require_vram:
                    raise RuntimeError(
                        "CUDA benchmarks require working assigned-device NVML measurement"
                    ) from exc
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds * 4))
        self._sample()
        self._shutdown()
        if exc is not None:
            exc.resource_evidence = self._resource_evidence()

    def _resource_evidence(self):
        return {
            "peak_process_tree_ram_mb": self._peak_ram_bytes / (1024**2)
            if self._ram_sampled
            else None,
            "peak_vram_mb": self.peak_vram_mb,
            "vram_measurement_method": self.vram_measurement_method,
            "resource_samples": self.samples(),
            "gpu_identity": dict(self.gpu_identity),
        }

    def _shutdown(self) -> None:
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001, S110 - cleanup preserves the result
                pass
        self._pynvml = None

    def metrics(self) -> dict[str, float | None]:
        reason = None
        if self._require_vram and (not self._vram_sampled or self._vram_errors):
            reason = "CUDA resource monitoring has missing device-memory observations"
        elif not self._ram_sampled:
            reason = "Resource monitoring has no process-tree RAM observations"
        if reason:
            error = RuntimeError(reason)
            error.resource_evidence = self._resource_evidence()
            raise error
        values = {
            "peak_process_tree_ram_mb": self._peak_ram_bytes / (1024**2)
            if self._ram_sampled
            else None,
            "peak_vram_mb": self.peak_vram_mb,
        }
        if self._temporary_directory is not None:
            values["peak_temporary_disk_mb"] = (
                self._peak_temporary_bytes / (1024**2)
                if self._temporary_sampled
                else None
            )
        return values

    def samples(self) -> list[dict[str, object]]:
        return [dict(row) for row in self._samples]

    @property
    def peak_vram_mb(self) -> float | None:
        return self._peak_vram_bytes / (1024**2) if self._vram_sampled else None

    def sample_now(self) -> None:
        self._sample()

    @property
    def vram_measurement_method(self) -> str:
        if self.device == "cpu":
            return "not-applicable"
        return (
            "nvml-device-total"
            if self._vram_sampled and not self._vram_errors
            else "unavailable"
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        with self._sample_lock:
            self._sample_locked()

    def _sample_locked(self) -> None:
        ram_bytes = vram_bytes = None
        try:
            import psutil

            root = psutil.Process(os.getpid())
            resident = 0
            for process in [root, *root.children(recursive=True)]:
                try:
                    resident += int(process.memory_info().rss)
                except psutil.NoSuchProcess:
                    continue
            self._ram_sampled = True
            self._peak_ram_bytes = max(self._peak_ram_bytes, resident)
            ram_bytes = resident
        except Exception:  # noqa: BLE001, S110 - unavailable RAM remains null
            pass
        if self._pynvml is not None and self._gpu_handle is not None:
            try:
                info = self._pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                vram_bytes = int(info.used)
                if vram_bytes < 0:
                    raise ValueError("Invalid NVML memory usage")
                self._vram_sampled = True
                self._peak_vram_bytes = max(self._peak_vram_bytes, vram_bytes)
                self.gpu_identity.setdefault("baseline_used_mb", vram_bytes / (1024**2))
                for field in ("total", "free"):
                    if hasattr(info, field):
                        self.gpu_identity[field + "_mb"] = int(getattr(info, field)) / (
                            1024**2
                        )
            except Exception:  # noqa: BLE001 - absent observations stay unavailable
                vram_bytes = None
                self._vram_errors += 1
        if self._temporary_directory is not None:
            try:
                size = sum(
                    path.stat().st_size
                    for path in self._temporary_directory.rglob("*")
                    if path.is_file()
                )
                self._temporary_sampled = True
                self._peak_temporary_bytes = max(self._peak_temporary_bytes, size)
            except OSError:
                pass
        self._samples.append(
            {
                "elapsed_seconds": max(0.0, time.perf_counter() - self._started_at),
                "process_tree_ram_mb": ram_bytes / (1024**2)
                if ram_bytes is not None
                else None,
                "vram_mb": vram_bytes / (1024**2) if vram_bytes is not None else None,
                "vram_measurement_method": self.vram_measurement_method,
            }
        )
