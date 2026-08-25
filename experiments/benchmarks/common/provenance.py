"""Git and hardware provenance recorded for benchmark runs."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path

from edumind.common.artifacts import sha256_file


def git_provenance(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        digest = hashlib.sha256()
        digest.update(
            subprocess.run(
                ["git", "diff", "--binary", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
            ).stdout
        )
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        for relative in sorted(untracked):
            path = root / relative
            if path.is_file():
                digest.update(relative.encode("utf-8"))
                digest.update(bytes.fromhex(sha256_file(path)))
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None, "dirty_hash": None}
    return {
        "commit": commit,
        "dirty": bool(status),
        "dirty_hash": digest.hexdigest() if status else None,
    }


def hardware_summary() -> Mapping[str, object]:
    summary: dict[str, object] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        import psutil

        summary["physical_cpu_count"] = psutil.cpu_count(logical=False)
        summary["ram_bytes"] = int(psutil.virtual_memory().total)
    except (ImportError, OSError):
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        summary["nvidia_driver"] = str(pynvml.nvmlSystemGetDriverVersion())
        summary["gpus"] = [
            {
                "name": str(pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(index))),
                "memory_bytes": int(
                    pynvml.nvmlDeviceGetMemoryInfo(
                        pynvml.nvmlDeviceGetHandleByIndex(index)
                    ).total
                ),
            }
            for index in range(pynvml.nvmlDeviceGetCount())
        ]
        pynvml.nvmlShutdown()
    except Exception:
        summary["gpus"] = []
    return summary

