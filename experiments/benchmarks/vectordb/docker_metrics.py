"""Measured Docker memory and persistent storage for one known benchmark service."""

from __future__ import annotations

import re
import json
import subprocess
import threading

from edumind.common.paths import PROJECT_ROOT

CONTAINERS = {
    "chroma": "edumind-benchmark-chroma",
    "qdrant": "edumind-benchmark-qdrant",
    "weaviate": "edumind-benchmark-weaviate",
    "pgvector": "edumind-benchmark-pgvector",
}
DATA_PATHS = {
    "chroma": "/data",
    "qdrant": "/qdrant/storage",
    "weaviate": "/var/lib/weaviate",
    "pgvector": "/var/lib/postgresql/data",
}
VOLUMES = {
    "chroma": "edumind-vector-benchmark_chroma-data",
    "qdrant": "edumind-vector-benchmark_qdrant-data",
    "weaviate": "edumind-vector-benchmark_weaviate-data",
    "pgvector": "edumind-vector-benchmark_pgvector-data",
}


class DockerMonitor:
    def __init__(self, candidate: str) -> None:
        self.candidate = candidate
        self.peak_bytes = 0
        self.sampled = False
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def __enter__(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        self._sample()

    def metrics(self) -> dict[str, float]:
        result = {}
        if self.sampled:
            result["peak_server_memory_bytes"] = float(self.peak_bytes)
        storage = _storage_bytes(self.candidate)
        if storage is not None:
            result["persistent_storage_bytes"] = float(storage)
        return result

    def _run(self) -> None:
        while not self.stop_event.wait(0.25):
            self._sample()

    def _sample(self) -> None:
        try:
            process = subprocess.run(
                [
                    "docker",
                    "stats",
                    "--no-stream",
                    "--format",
                    "{{.MemUsage}}",
                    CONTAINERS[self.candidate],
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if process.returncode == 0 and process.stdout.strip():
                value = _bytes(process.stdout.split("/")[0].strip())
                if value is not None:
                    self.sampled = True
                    self.peak_bytes = max(self.peak_bytes, value)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return


def _storage_bytes(candidate: str) -> int | None:
    try:
        process = subprocess.run(
            ["docker", "exec", CONTAINERS[candidate], "du", "-sb", DATA_PATHS[candidate]],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if process.returncode != 0:
        return _helper_storage_bytes(candidate)
    match = re.match(r"(\d+)", process.stdout.strip())
    return int(match.group(1)) if match else _helper_storage_bytes(candidate)


def _helper_storage_bytes(candidate: str) -> int | None:
    lock = PROJECT_ROOT / "data/benchmarks/models/vectordb.json"
    try:
        payload = json.loads(lock.read_text(encoding="utf-8"))
        image = str(payload["images"]["inspector"])
        process = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--mount",
                f"type=volume,source={VOLUMES[candidate]},target=/volume,readonly",
                image,
                "du",
                "-sk",
                "/volume",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, KeyError, OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return None
    match = re.match(r"(\d+)", process.stdout.strip()) if process.returncode == 0 else None
    return int(match.group(1)) * 1024 if match else None


def verify_image(candidate: str, expected_digest: str) -> None:
    """Fail when a running benchmark container does not match its locked image digest."""
    try:
        expected = subprocess.run(
            ["docker", "image", "inspect", expected_digest, "--format", "{{.Id}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        running = subprocess.run(
            ["docker", "inspect", CONTAINERS[candidate], "--format", "{{.Image}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Cannot verify the pinned Docker image for {candidate}; prepare and start "
            "the vector benchmark servers first"
        ) from exc
    if expected != running:
        raise RuntimeError(
            f"{candidate} is running image {running}, expected locked image {expected}. "
            "Recreate the benchmark containers with the documented --env-file command."
        )


def image_lock() -> dict[str, str]:
    path = PROJECT_ROOT / "data/benchmarks/models/vectordb.json"
    if not path.is_file():
        raise RuntimeError(
            f"Missing {path}; run `python experiments/benchmarks/prepare.py vectordb`"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images")
    expected_images = {"chroma", "qdrant", "weaviate", "pgvector", "inspector"}
    if not isinstance(images, dict) or set(images) != expected_images:
        raise RuntimeError("Vector database image lock is malformed or incomplete")
    clients = payload.get("clients")
    if not isinstance(clients, dict) or not clients:
        raise RuntimeError("Vector database client lock is malformed or incomplete")
    return {
        **{f"image:{name}": str(value) for name, value in images.items()},
        **{f"client:{name}": str(value) for name, value in clients.items()},
    }


def _bytes(value: str) -> int | None:
    match = re.fullmatch(r"([0-9.]+)([KMG]i?B|B)", value)
    if not match:
        return None
    factors = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    return int(float(match.group(1)) * factors[match.group(2)])
