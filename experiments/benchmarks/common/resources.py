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
    ) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_ram_bytes = 0
        self._ram_sampled = False
        self._peak_vram_bytes = 0
        self._peak_temporary_bytes = 0
        self._temporary_directory = temporary_directory
        self._pynvml: Any | None = None
        self._gpu_handles: list[Any] = []

    def __enter__(self) -> ResourceMonitor:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(index)
                for index in range(pynvml.nvmlDeviceGetCount())
            ]
        except Exception:  # optional monitoring must not invalidate quality results
            self._pynvml = None
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
        values = {}
        if self._ram_sampled:
            values["peak_process_tree_ram_mb"] = self._peak_ram_bytes / (1024**2)
        if self._peak_vram_bytes:
            values["peak_vram_mb"] = self._peak_vram_bytes / (1024**2)
        if self._temporary_directory is not None:
            values["peak_temporary_disk_mb"] = self._peak_temporary_bytes / (1024**2)
        return values

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
            try:
                for handle in self._gpu_handles:
                    processes = self._pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    self._peak_vram_bytes = max(
                        self._peak_vram_bytes,
                        sum(
                            int(process.usedGpuMemory)
                            for process in processes
                            if process.pid in process_ids and process.usedGpuMemory
                        ),
                    )
            except Exception:  # optional driver/API differences omit VRAM measurements
                pass
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
