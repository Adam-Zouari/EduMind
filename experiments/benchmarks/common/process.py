"""Fresh-process helpers shared by benchmark workers."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import time
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
from experiments.benchmarks.common.resources import ResourceMonitor


class WorkerResourceLimitError(RuntimeError):
    """A worker was stopped after crossing an enforced hardware limit."""

    def __init__(self, peak_vram_mb: float, limit_mb: float, samples) -> None:
        super().__init__(
            f"worker exceeded the {limit_mb:.0f} MiB VRAM limit "
            f"(observed {peak_vram_mb:.1f} MiB)"
        )
        self.peak_vram_mb = peak_vram_mb
        self.limit_mb = limit_mb
        self.samples = tuple(dict(row) for row in samples)


class WorkerMeasurementError(RuntimeError):
    """A worker could not be qualified because resource telemetry was unavailable."""


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
    vram_limit_mb: float | None = None,
    require_vram_measurement: bool = False,
    telemetry_interval_seconds: float = 0.05,
    poll_interval_seconds: float = 0.05,
    timeout_seconds: float | None = None,
) -> dict[str, object]:
    module = ".".join(
        script.resolve().relative_to(PROJECT_ROOT.resolve()).with_suffix("").parts
    )
    if temporary_root is not None:
        temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=temporary_root) as raw:
        directory = Path(raw)
        input_path = directory / "input.json"
        output_path = directory / "output.json"
        atomic_write_json(input_path, dict(payload))
        command = [sys.executable, "-m", module, str(input_path), str(output_path)]
        supervision: dict[str, object] | None = None
        if vram_limit_mb is None and not require_vram_measurement:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=worker_environment(device),
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds,
            )
            stdout, stderr, returncode = (
                completed.stdout,
                completed.stderr,
                completed.returncode,
            )
        else:
            monitor = ResourceMonitor(
                interval_seconds=telemetry_interval_seconds,
                require_vram=True,
            )
            process: subprocess.Popen[str] | None = None
            exceeded: tuple[float, tuple[dict[str, object], ...]] | None = None
            started = time.perf_counter()
            try:
                with monitor:
                    # Capture the whole-device fallback baseline before the child can
                    # allocate CUDA memory. This is required on Windows/WDDM, where
                    # NVML may expose the PID but not its per-process byte count.
                    process = subprocess.Popen(
                        command,
                        cwd=PROJECT_ROOT,
                        env=worker_environment(device),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    while process.poll() is None:
                        monitor.sample_now()
                        peak = monitor.peak_vram_mb
                        if (
                            vram_limit_mb is not None
                            and peak is not None
                            and peak > vram_limit_mb
                        ):
                            exceeded = (peak, tuple(monitor.samples()))
                            _terminate_process_tree(process.pid)
                            break
                        if (
                            timeout_seconds is not None
                            and time.perf_counter() - started > timeout_seconds
                        ):
                            _terminate_process_tree(process.pid)
                            raise TimeoutError(
                                f"{error_label} exceeded the {timeout_seconds:g}s timeout"
                            )
                        time.sleep(poll_interval_seconds)
            except RuntimeError as exc:
                if process is not None:
                    _terminate_process_tree(process.pid)
                    process.communicate()
                raise WorkerMeasurementError(str(exc)) from exc
            except BaseException:
                if process is not None:
                    _terminate_process_tree(process.pid)
                    process.communicate()
                raise
            if process is None:  # defensive: Popen either returns or raises
                raise RuntimeError(f"{error_label} did not start")
            final_peak = monitor.peak_vram_mb
            if (
                exceeded is None
                and vram_limit_mb is not None
                and final_peak is not None
                and final_peak > vram_limit_mb
            ):
                exceeded = (final_peak, tuple(monitor.samples()))
            stdout, stderr = process.communicate()
            returncode = process.returncode
            if exceeded is not None:
                raise WorkerResourceLimitError(exceeded[0], vram_limit_mb, exceeded[1])
            try:
                resource_metrics = monitor.metrics()
            except RuntimeError as exc:
                supervision = {
                    "vram_measurement_method": "unavailable",
                    "measurement_error": str(exc),
                    "resource_samples": monitor.samples(),
                }
            else:
                supervision = {
                    **resource_metrics,
                    "vram_measurement_method": monitor.vram_measurement_method,
                    "resource_samples": monitor.samples(),
                }
        if returncode or not output_path.is_file():
            detail = (stderr or stdout or "no worker output").strip()
            raise RuntimeError(f"{error_label} failed: {detail[-4000:]}")
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError(f"{error_label} returned a non-object result")
        reported_peak = _reported_peak_vram_mb(result)
        if reported_peak is not None and supervision is not None:
            supervision["worker_reported_peak_vram_mb"] = reported_peak
            reported_method = result.get("vram_measurement_method")
            if (
                supervision.get("vram_measurement_method") == "unavailable"
                and isinstance(reported_method, str)
                and reported_method != "unavailable"
            ):
                supervision["vram_measurement_method"] = (
                    f"worker-{reported_method}"
                )
            supervised_peak = supervision.get("peak_vram_mb")
            if not isinstance(supervised_peak, (int, float)) or isinstance(
                supervised_peak, bool
            ):
                supervision["peak_vram_mb"] = reported_peak
            else:
                supervision["peak_vram_mb"] = max(
                    float(supervised_peak), reported_peak
                )
        if (
            vram_limit_mb is not None
            and reported_peak is not None
            and reported_peak > vram_limit_mb
        ):
            samples = list((supervision or {}).get("resource_samples", []))
            samples.append(
                {"source": "worker-report", "vram_mb": reported_peak}
            )
            raise WorkerResourceLimitError(reported_peak, vram_limit_mb, samples)
        if supervision is not None:
            result["_worker_supervision"] = supervision
        return result


def _terminate_process_tree(pid: int) -> None:
    try:
        import psutil

        try:
            root = psutil.Process(pid)
            children = root.children(recursive=True)
        except psutil.NoSuchProcess:
            return
        except psutil.Error:
            _taskkill(pid)
            return
        for process in reversed(children):
            try:
                process.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        try:
            root.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        _, alive = psutil.wait_procs([*children, root], timeout=2)
        for process in alive:
            try:
                process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except (ImportError, OSError):
        _taskkill(pid)


def _taskkill(pid: int) -> None:
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
    except OSError:
        pass


def _reported_peak_vram_mb(result: Mapping[str, object]) -> float | None:
    values: list[float] = []
    for container in (result, result.get("operational"), result.get("metrics")):
        if not isinstance(container, Mapping):
            continue
        for name in ("peak_vram_mb", "peak_visual_vram_mb"):
            value = container.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
    return max(values) if values else None


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
            "samples": tuple(dict(row) for row in manifest_payload.get("samples", [])),
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
