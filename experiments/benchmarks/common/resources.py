"""Low-overhead process RAM/VRAM sampling for measured candidate runs."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any, Self

_GPU_FLOOR_LOCK = threading.Lock()
_GPU_FLOOR_BYTES: dict[int, int] = {}


def _device_memory_floor(index: int, observed_bytes: int) -> int:
    """Keep the lowest device baseline seen by this benchmark process.

    Windows WDDM can retain a terminated CUDA child's allocation briefly while
    reporting per-process memory as unavailable. Re-basing every candidate on
    that transient allocation makes the next child appear to use zero VRAM.
    The parent process is long-lived, so its lowest observed pre-run value is a
    stable device floor for all sequential children.
    """

    with _GPU_FLOOR_LOCK:
        previous = _GPU_FLOOR_BYTES.get(index)
        floor = observed_bytes if previous is None else min(previous, observed_bytes)
        _GPU_FLOOR_BYTES[index] = floor
        return floor


class ResourceMonitor:
    def __init__(
        self,
        interval_seconds: float = 0.05,
        *,
        temporary_directory: Path | None = None,
        require_vram: bool = False,
        report_zero_vram: bool = False,
        zero_vram_measurement_method: str = "cpu-zero",
    ) -> None:
        if not zero_vram_measurement_method:
            raise ValueError("zero_vram_measurement_method cannot be empty")
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_ram_bytes = 0
        self._ram_sampled = False
        self._peak_vram_bytes = 0
        self._vram_sampled = False
        self._process_vram_sampled = False
        self._device_delta_vram_sampled = False
        self._peak_temporary_bytes = 0
        self._temporary_directory = temporary_directory
        self._require_vram = require_vram
        self._report_zero_vram = report_zero_vram
        self._zero_vram_measurement_method = zero_vram_measurement_method
        self._pynvml: Any | None = None
        self._gpu_handles: list[Any] = []
        self._gpu_baseline_bytes: list[int] = []
        self._started_at = 0.0
        self._samples: list[dict[str, object]] = []

    def __enter__(self) -> Self:
        self._started_at = time.perf_counter()
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
            self._gpu_baseline_bytes = [
                _device_memory_floor(
                    index,
                    int(pynvml.nvmlDeviceGetMemoryInfo(handle).used),
                )
                for index, handle in enumerate(self._gpu_handles)
            ]
            if self._require_vram and not self._gpu_handles:
                raise RuntimeError("CUDA benchmark requested but NVML found no GPU")
        except Exception as exc:  # optional except when a CUDA profile requires it
            self._pynvml = None
            if self._require_vram:
                raise RuntimeError(
                    "CUDA benchmarks require working NVML VRAM measurement"
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
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001, S110 - cleanup must not mask results
                pass

    def metrics(self) -> dict[str, float]:
        if self._require_vram and (
            not self._vram_sampled or self._peak_vram_bytes <= 0
        ):
            raise RuntimeError(
                "CUDA resource monitoring did not capture process VRAM usage"
            )
        values = {}
        if self._ram_sampled:
            values["peak_process_tree_ram_mb"] = self._peak_ram_bytes / (1024**2)
        if self._peak_vram_bytes or self._report_zero_vram:
            values["peak_vram_mb"] = self._peak_vram_bytes / (1024**2)
        if self._temporary_directory is not None:
            values["peak_temporary_disk_mb"] = self._peak_temporary_bytes / (1024**2)
        return values

    def samples(self) -> list[dict[str, object]]:
        """Return timestamped resource observations for benchmark artifacts."""

        return [dict(row) for row in self._samples]

    @property
    def vram_measurement_method(self) -> str:
        if self._process_vram_sampled:
            return "nvml-process-tree"
        if self._device_delta_vram_sampled:
            return "nvml-device-delta-wddm"
        if self._report_zero_vram and not self._require_vram:
            return self._zero_vram_measurement_method
        return "unavailable"

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        process_ids = {os.getpid()}
        ram_bytes: int | None = None
        vram_bytes: int | None = None
        try:
            import psutil

            self._ram_sampled = True
            root = psutil.Process(os.getpid())
            processes = [root, *root.children(recursive=True)]
            process_ids.update(process.pid for process in processes)
            resident = 0
            for process in processes:
                try:
                    resident += int(process.memory_info().rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            self._peak_ram_bytes = max(
                self._peak_ram_bytes,
                resident,
            )
            ram_bytes = resident
        except (ImportError, OSError):
            pass
        if self._pynvml is not None:
            sampled_vram = 0
            for index, handle in enumerate(self._gpu_handles):
                try:
                    processes = list(
                        self._pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    )
                    try:
                        processes.extend(
                            self._pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                        )
                    except Exception:  # noqa: BLE001, S110 - process listing is optional
                        pass
                    by_pid = {int(process.pid): process for process in processes}
                    selected = [
                        process for pid, process in by_pid.items() if pid in process_ids
                    ]
                    self._vram_sampled = True
                    reported = [
                        int(process.usedGpuMemory)
                        for process in selected
                        if process.usedGpuMemory not in (None, 0)
                    ]
                    if reported:
                        self._process_vram_sampled = True
                        sampled_vram += sum(reported)
                    elif selected:
                        # Windows WDDM exposes the target PID through NVML but
                        # returns usedGpuMemory=None. In that case, measure the
                        # increase in device memory from the pre-run baseline.
                        used = int(self._pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                        delta = max(0, used - self._gpu_baseline_bytes[index])
                        if delta:
                            self._device_delta_vram_sampled = True
                            sampled_vram += delta
                except Exception:  # noqa: BLE001, S112 - metrics() rejects missing CUDA samples
                    continue
            if self._vram_sampled:
                self._peak_vram_bytes = max(self._peak_vram_bytes, sampled_vram)
                vram_bytes = sampled_vram
        if self._temporary_directory is not None:
            try:
                size = sum(
                    path.stat().st_size
                    for path in self._temporary_directory.rglob("*")
                    if path.is_file()
                )
                self._peak_temporary_bytes = max(self._peak_temporary_bytes, size)
            except OSError:
                pass
        self._samples.append(
            {
                "elapsed_seconds": max(0.0, time.perf_counter() - self._started_at),
                "process_tree_ram_mb": (
                    ram_bytes / (1024**2) if ram_bytes is not None else None
                ),
                "vram_mb": vram_bytes / (1024**2) if vram_bytes is not None else None,
                "vram_measurement_method": self.vram_measurement_method,
            }
        )
