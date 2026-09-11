"""Low-overhead process RAM/VRAM sampling for measured candidate runs."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import TracebackType
from typing import Any


class ResourceMonitor:
    def __init__(
        self,
        interval_seconds: float = 0.05,
        *,
        temporary_directory: Path | None = None,
        require_vram: bool = False,
        report_zero_vram: bool = False,
    ) -> None:
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
        self._pynvml: Any | None = None
        self._gpu_handles: list[Any] = []
        self._gpu_baseline_bytes: list[int] = []

    def __enter__(self) -> ResourceMonitor:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
            self._gpu_baseline_bytes = [
                int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                for handle in self._gpu_handles
            ]
            if self._require_vram and not self._gpu_handles:
                raise RuntimeError("CUDA ASR benchmark requested but NVML found no GPU")
        except Exception as exc:  # optional except when a CUDA profile requires it
            self._pynvml = None
            if self._require_vram:
                raise RuntimeError(
                    "CUDA ASR benchmarks require working NVML VRAM measurement"
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
            except Exception:  # pragma: no cover - driver-specific cleanup
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

    @property
    def vram_measurement_method(self) -> str:
        if self._process_vram_sampled:
            return "nvml-process-tree"
        if self._device_delta_vram_sampled:
            return "nvml-device-delta-wddm"
        if self._report_zero_vram and not self._require_vram:
            return "cpu-zero"
        return "unavailable"

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        process_ids = {os.getpid()}
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
        except (ImportError, OSError):
            pass
        if self._pynvml is not None:
            for index, handle in enumerate(self._gpu_handles):
                try:
                    processes = list(
                        self._pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    )
                    try:
                        processes.extend(
                            self._pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                        )
                    except Exception:
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
                        self._peak_vram_bytes = max(
                            self._peak_vram_bytes, sum(reported)
                        )
                    elif selected:
                        # Windows WDDM exposes the target PID through NVML but
                        # returns usedGpuMemory=None. In that case, measure the
                        # increase in device memory from the pre-run baseline.
                        used = int(self._pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                        delta = max(0, used - self._gpu_baseline_bytes[index])
                        if delta:
                            self._device_delta_vram_sampled = True
                            self._peak_vram_bytes = max(self._peak_vram_bytes, delta)
                except Exception:  # a required CUDA run is rejected by metrics()
                    continue
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
