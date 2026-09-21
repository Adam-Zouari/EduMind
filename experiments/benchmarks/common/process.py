"""Fresh-process helpers shared by benchmark workers."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from edumind.common.artifacts import atomic_write_json
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    DatasetManifest,
    SampleResult,
)


def worker_environment(device: str) -> dict[str, str]:
    environment = os.environ.copy()
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["NVIDIA_VISIBLE_DEVICES"] = "none"
    return environment


def run_json_worker(
    script: Path,
    payload: Mapping[str, object],
    *,
    device: str,
    prefix: str,
    error_label: str,
    temporary_root: Path | None = None,
) -> dict[str, object]:
    if temporary_root is not None:
        temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=temporary_root) as raw:
        directory = Path(raw)
        input_path = directory / "input.json"
        output_path = directory / "output.json"
        atomic_write_json(input_path, dict(payload))
        completed = subprocess.run(
            [sys.executable, str(script), str(input_path), str(output_path)],
            cwd=PROJECT_ROOT,
            env=worker_environment(device),
            capture_output=True,
            text=True,
        )
        if completed.returncode or not output_path.is_file():
            detail = (
                completed.stderr or completed.stdout or "no worker output"
            ).strip()
            raise RuntimeError(f"{error_label} failed: {detail[-4000:]}")
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError(f"{error_label} returned a non-object result")
        return result


def json_worker_main(
    execute: Callable[[dict[str, object]], dict[str, object]],
) -> int:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} PAYLOAD_JSON RESULT_JSON")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Worker payload must be a JSON object")
    atomic_write_json(Path(sys.argv[2]), execute(payload))
    return 0


def payload_mapping(value: object) -> dict[str, object]:
    """Normalize an untrusted JSON value to an object mapping."""

    return dict(value) if isinstance(value, Mapping) else {}


def numeric_payload(value: object) -> dict[str, float]:
    """Read finite-shape numeric JSON fields while excluding booleans."""

    return {
        str(name): float(number)
        for name, number in payload_mapping(value).items()
        if isinstance(number, (int, float)) and not isinstance(number, bool)
    }


def sample_result_from_payload(value: object) -> SampleResult:
    row = payload_mapping(value)
    return SampleResult(
        str(row["sample_id"]),
        numeric_payload(row["metrics"]),
        float(row["latency_seconds"]),
        payload_mapping(row.get("metadata")),
    )


def benchmark_objects_from_payload(
    payload: Mapping[str, object],
) -> tuple[
    str,
    DatasetManifest,
    dict[str, dict[str, object]],
    BenchmarkPlan,
]:
    candidate = str(payload["candidate"])
    manifest_payload = payload_mapping(payload["manifest"])
    manifest = DatasetManifest(
        **{
            **manifest_payload,
            "samples": tuple(
                dict(row) for row in manifest_payload.get("samples", [])
            ),
        }
    )
    plan_payload = payload_mapping(payload["plan"])
    plan = BenchmarkPlan(
        **{**plan_payload, "candidates": tuple(plan_payload.get("candidates", []))}
    )
    model_lock = {
        str(name): dict(entry)
        for name, entry in payload_mapping(payload["model_lock"]).items()
        if isinstance(entry, Mapping)
    }
    return candidate, manifest, model_lock, plan


def candidate_execution_payload(exc: CandidateExecutionError) -> dict[str, object]:
    return {
        "status": "failed",
        "error": str(exc),
        "samples": [asdict(sample) for sample in exc.samples],
        "operational": exc.operational,
        "metrics": exc.metrics,
        "intervals": exc.intervals,
        "parameters": exc.parameters,
        "artifacts": exc.artifacts,
    }


def successful_execution_payload(evaluated: Sequence[object]) -> dict[str, object]:
    samples = evaluated[0]
    return {
        "status": "success",
        "samples": [asdict(sample) for sample in samples],
        "operational": evaluated[1],
        "metrics": evaluated[2],
        "parameters": evaluated[3],
        "intervals": evaluated[4],
        "artifacts": evaluated[5],
    }


def decode_worker_result(result: Mapping[str, object]):
    status = str(result.get("status", ""))
    samples = tuple(
        sample_result_from_payload(row) for row in result.get("samples", [])
    )
    if status == "failed":
        raise CandidateExecutionError(
            str(result.get("error", "worker failed")),
            samples=samples,
            operational=numeric_payload(result.get("operational")),
            metrics=numeric_payload(result.get("metrics")),
            parameters=payload_mapping(result.get("parameters")),
            intervals=payload_mapping(result.get("intervals")),
            artifacts=payload_mapping(result.get("artifacts")),
        )
    if status != "success":
        raise RuntimeError(f"Worker returned unsupported status {status!r}")
    return (
        list(samples),
        numeric_payload(result.get("operational")),
        numeric_payload(result.get("metrics")),
        payload_mapping(result.get("parameters")),
        payload_mapping(result.get("intervals")),
        payload_mapping(result.get("artifacts")),
    )


def seed_deterministically(seed: int) -> None:
    """Seed benchmark workers consistently, including available CUDA devices."""

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ModuleNotFoundError:
        pass
