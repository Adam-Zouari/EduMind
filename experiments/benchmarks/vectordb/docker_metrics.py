"""Measured Docker memory and persistent storage for one known benchmark service."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from itertools import pairwise

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
    def __init__(
        self,
        candidate: str,
        *,
        interval_seconds: float,
        stats_timeout_seconds: float,
        storage_timeout_seconds: float,
        helper_timeout_seconds: float,
        monitor_memory: bool,
        monitor_storage: bool,
    ) -> None:
        self.candidate = candidate
        self.peak_bytes = 0
        self.sampled = False
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.interval_seconds = interval_seconds
        self.stats_timeout_seconds = stats_timeout_seconds
        self.storage_timeout_seconds = storage_timeout_seconds
        self.helper_timeout_seconds = helper_timeout_seconds
        self.monitor_memory = monitor_memory
        self.monitor_storage = monitor_storage
        self.samples = []
        self.checkpoints = []
        self.started_at = None
        self.ended_at = None

    def __enter__(self):
        self.started_at = time.time()
        if self.monitor_memory:
            self._sample()
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop_event.set()
        if self.thread:
            # Each Docker request is already bounded by stats_timeout_seconds.
            # Finish the sampler before the final sample and metric snapshot.
            self.thread.join()
        if self.monitor_memory:
            self._sample()
        self.ended_at = time.time()

    def checkpoint(self, corpus_id, *, complete=True):
        value = (
            _storage_bytes(
                self.candidate,
                timeout_seconds=self.storage_timeout_seconds,
                helper_timeout_seconds=self.helper_timeout_seconds,
            )
            if self.monitor_storage
            else None
        )
        row = {
            "corpus_id": corpus_id,
            "timestamp": time.time(),
            "checkpoint": "completed-build" if complete else "partial-build",
            "apparent_bytes": value,
            "method": "du-apparent-bytes",
            "complete": complete and value is not None,
        }
        self.checkpoints.append(row)
        return row

    def metrics(self) -> dict[str, float]:
        result = {}
        if self.sampled:
            result["peak_server_memory_bytes"] = float(self.peak_bytes)
        return result

    def evidence(self):
        usable = [row for row in self.samples if row["bytes"] is not None]
        timestamps = [row["timestamp"] for row in self.samples]
        return {
            "samples": self.samples,
            "checkpoints": self.checkpoints,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "coverage": "complete"
            if self.samples and len(usable) == len(self.samples) and self.ended_at
            else "partial"
            if usable
            else "unavailable",
            "sample_count": len(usable),
            "gap_count": len(self.samples) - len(usable),
            "interval_seconds": self.interval_seconds,
            "observed_max_gap_seconds": max(
                (right - left for left, right in pairwise(timestamps)),
                default=None,
            ),
            "method": "docker-stats-cli-cache-subtracted",
            "scope": "server-lifecycle",
        }

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval_seconds):
            self._sample()

    def _sample(self) -> None:
        sample = {"timestamp": time.time(), "bytes": None, "error": None}
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
                timeout=self.stats_timeout_seconds,
                check=False,
            )
            if process.returncode == 0 and process.stdout.strip():
                value = _bytes(process.stdout.split("/")[0].strip())
                if value is not None:
                    self.sampled = True
                    self.peak_bytes = max(self.peak_bytes, value)
                    sample["bytes"] = value
            if sample["bytes"] is None:
                sample["error"] = process.stderr.strip() or "unusable stats response"
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            sample["error"] = str(exc)
        self.samples.append(sample)


def _storage_bytes(
    candidate: str, *, timeout_seconds: float, helper_timeout_seconds: float
) -> int | None:
    try:
        process = subprocess.run(
            [
                "docker",
                "exec",
                CONTAINERS[candidate],
                "du",
                "-sb",
                DATA_PATHS[candidate],
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if process.returncode != 0:
        return _helper_storage_bytes(candidate, timeout_seconds=helper_timeout_seconds)
    match = re.match(r"(\d+)", process.stdout.strip())
    return (
        int(match.group(1))
        if match
        else _helper_storage_bytes(candidate, timeout_seconds=helper_timeout_seconds)
    )


def _helper_storage_bytes(candidate: str, *, timeout_seconds: float) -> int | None:
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
                "-sb",
                "/volume",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ):
        return None
    match = (
        re.match(r"(\d+)", process.stdout.strip()) if process.returncode == 0 else None
    )
    return int(match.group(1)) if match else None


def verify_image(
    candidate: str, expected_digest: str, *, timeout_seconds: float
) -> None:
    """Fail when a running benchmark container does not match its locked image digest."""
    try:
        expected = subprocess.run(
            ["docker", "image", "inspect", expected_digest, "--format", "{{.Id}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        ).stdout.strip()
        running = subprocess.run(
            ["docker", "inspect", CONTAINERS[candidate], "--format", "{{.Image}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        ).stdout.strip()
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
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
            f"Missing {path}; run `python -m experiments.benchmarks.prepare vectordb`"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(payload) != {"schema_version", "images", "clients"}
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
    ):
        raise RuntimeError("Unsupported vector deployment-lock schema")
    images = payload.get("images")
    expected_images = {"chroma", "qdrant", "weaviate", "pgvector", "inspector"}
    if not isinstance(images, dict) or set(images) != expected_images:
        raise RuntimeError("Vector database image lock is malformed or incomplete")
    if any(
        not isinstance(value, str)
        or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", value)
        for value in images.values()
    ):
        raise RuntimeError("Vector image lock requires immutable SHA256 digests")
    clients = payload.get("clients")
    if not isinstance(clients, dict) or set(clients) != {
        "chromadb-client",
        "qdrant-client",
        "weaviate-client",
        "psycopg",
        "psycopg-pool",
    }:
        raise RuntimeError("Vector database client lock is malformed or incomplete")
    for name, expected in clients.items():
        try:
            actual = version(name)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"Missing locked vector client: {name}") from exc
        if actual != expected:
            raise RuntimeError(f"Vector client {name} is {actual}, expected {expected}")
    return {
        **{f"image:{name}": str(value) for name, value in images.items()},
        **{f"client:{name}": str(value) for name, value in clients.items()},
    }


def _bytes(value: str) -> int | None:
    match = re.fullmatch(r"([0-9.]+)([KMG]i?B|B)", value)
    if not match:
        return None
    factors = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "KiB": 1024,
        "MiB": 1024**2,
        "GiB": 1024**3,
    }
    return int(float(match.group(1)) * factors[match.group(2)])
