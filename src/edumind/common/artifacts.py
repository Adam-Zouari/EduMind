"""Atomic files, fingerprints, and local process coordination."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


@contextmanager
def local_file_lock(path: Path, *, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Acquire an inter-process lock file with bounded waiting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out acquiring local lock: {path}")
            time.sleep(0.05)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        yield
    finally:
        os.close(descriptor)
        path.unlink(missing_ok=True)


def git_provenance(root: Path) -> dict[str, object]:
    """Collect a commit plus dirty-content hash without logging file contents."""
    try:
        import hashlib

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
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
                    pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(index)).total
                ),
            }
            for index in range(pynvml.nvmlDeviceGetCount())
        ]
        pynvml.nvmlShutdown()
    except Exception:  # hardware fingerprint remains useful without an NVIDIA driver
        summary["gpus"] = []
    return summary
