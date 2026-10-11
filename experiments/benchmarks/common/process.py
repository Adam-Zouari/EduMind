"""Fresh-process helpers shared by benchmark workers."""

from __future__ import annotations

import json
import math
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


class WorkerResourceLimitError(CandidateExecutionError):
    """A worker was stopped after crossing an enforced hardware limit."""

    def __init__(self, peak_vram_mb: float, limit_mb: float, samples) -> None:
        self.peak_vram_mb = peak_vram_mb
        self.limit_mb = limit_mb
        self.resource_samples = tuple(dict(row) for row in samples)
        self.resource_evidence = {
            "peak_vram_mb": peak_vram_mb,
            "vram_limit_mb": limit_mb,
            "vram_measurement_method": "nvml-device-total",
            "resource_samples": list(self.resource_samples),
        }
        super().__init__(
            f"worker exceeded the {limit_mb:.0f} MiB VRAM limit "
            f"(observed {peak_vram_mb:.1f} MiB)",
            operational={"peak_vram_mb": peak_vram_mb},
            artifacts={"resource_limit": self.resource_evidence},
        )


class WorkerExecutionError(CandidateExecutionError):
    """A worker setup/infrastructure failure with any captured resource evidence."""

    def __init__(self, message, evidence):
        self.resource_evidence = dict(evidence)
        super().__init__(
            message,
            operational={
                key: evidence[key]
                for key in ("peak_process_tree_ram_mb", "peak_vram_mb")
                if key in evidence
            },
            artifacts={"worker_error": {"error": message, **evidence}},
        )


class WorkerMeasurementError(WorkerExecutionError):
    """A worker could not be qualified because resource telemetry was unavailable."""

    def __init__(self, message, evidence=None):
        super().__init__(message, evidence or {})


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
    if temporary_root is None:
        temporary_root = Path(os.environ.get("TEMP") or tempfile.gettempdir())
    if temporary_root is not None:
        temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=prefix, dir=temporary_root) as raw:
        directory = Path(raw)
        input_path = directory / "input.json"
        output_path = directory / "output.json"
        command = [sys.executable, "-m", module, str(input_path), str(output_path)]

        def launch(**streams):
            atomic_write_json(
                input_path, {**payload, "_worker_started_at": time.perf_counter()}
            )
            return subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=worker_environment(device),
                text=True,
                **streams,
            )

        supervision: dict[str, object] | None = None
        if vram_limit_mb is None and not require_vram_measurement:
            process = launch(
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                stdout, stderr = process.communicate(timeout=timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                _terminate_process_tree(process.pid)
                stdout, stderr = process.communicate()
                raise WorkerExecutionError(
                    f"{error_label} exceeded the {timeout_seconds:g}s timeout",
                    {"stdout": stdout[-4000:], "stderr": stderr[-4000:]},
                ) from exc
            returncode = process.returncode
        else:
            monitor = ResourceMonitor(
                interval_seconds=telemetry_interval_seconds,
                require_vram=True,
            )
            process: subprocess.Popen[str] | None = None
            exceeded: tuple[float, tuple[dict[str, object], ...]] | None = None
            started = time.perf_counter()
            try:
                # Files cannot fill a pipe buffer while the supervisor polls the GPU.
                with (
                    monitor,
                    (directory / "stdout.log").open(
                        "w", encoding="utf-8"
                    ) as stdout_log,
                    (directory / "stderr.log").open(
                        "w", encoding="utf-8"
                    ) as stderr_log,
                ):
                    # Sample raw assigned-device usage throughout the child lifecycle.
                    process = launch(
                        stdout=stdout_log,
                        stderr=stderr_log,
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
                            raise TimeoutError(
                                f"{error_label} exceeded the {timeout_seconds:g}s timeout"
                            )
                        time.sleep(poll_interval_seconds)
            except TimeoutError as exc:
                if process is not None:
                    _terminate_process_tree(process.pid)
                    process.communicate()
                raise WorkerExecutionError(
                    str(exc), getattr(exc, "resource_evidence", {})
                ) from exc
            except RuntimeError as exc:
                if process is not None:
                    _terminate_process_tree(process.pid)
                    process.communicate()
                raise WorkerMeasurementError(
                    str(exc), getattr(exc, "resource_evidence", {})
                ) from exc
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
            process.communicate()
            stdout = (directory / "stdout.log").read_text(
                encoding="utf-8", errors="replace"
            )
            stderr = (directory / "stderr.log").read_text(
                encoding="utf-8", errors="replace"
            )
            returncode = process.returncode
            if exceeded is not None:
                raise WorkerResourceLimitError(exceeded[0], vram_limit_mb, exceeded[1])
            try:
                resource_metrics = monitor.metrics()
            except RuntimeError as exc:
                raise WorkerMeasurementError(
                    str(exc), getattr(exc, "resource_evidence", {})
                ) from exc
            else:
                supervision = {
                    **resource_metrics,
                    "vram_measurement_method": monitor.vram_measurement_method,
                    "resource_samples": monitor.samples(),
                }
        if returncode or not output_path.is_file():
            detail = (stderr or stdout or "no worker output").strip()
            evidence = dict(supervision or {})
            if output_path.is_file():
                failed = json.loads(output_path.read_text(encoding="utf-8"))
                if isinstance(failed, Mapping) and failed.get("worker_error"):
                    detail = str(failed["worker_error"])
                    evidence.update(payload_mapping(failed.get("resource_evidence")))
            raise WorkerExecutionError(
                f"{error_label} failed: {detail[-4000:]}", evidence
            )
        result = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise RuntimeError(f"{error_label} returned a non-object result")
        reported_peak = _reported_peak_vram_mb(result)
        if reported_peak is not None and supervision is not None:
            supervision["worker_reported_peak_vram_mb"] = reported_peak
            supervised_peak = supervision.get("peak_vram_mb")
            if not isinstance(supervised_peak, (int, float)) or isinstance(
                supervised_peak, bool
            ):
                supervision["peak_vram_mb"] = reported_peak
            else:
                supervision["peak_vram_mb"] = max(float(supervised_peak), reported_peak)
        if (
            vram_limit_mb is not None
            and reported_peak is not None
            and reported_peak > vram_limit_mb
        ):
            samples = list((supervision or {}).get("resource_samples", []))
            samples.append({"source": "worker-report", "vram_mb": reported_peak})
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
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0
            ):
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
    try:
        result = execute(payload)
    except Exception as exc:  # noqa: BLE001 - retain setup/measurement failures without inventing attempts
        atomic_write_json(
            Path(sys.argv[2]),
            {
                "worker_error": f"{type(exc).__name__}: {exc}",
                "resource_evidence": getattr(exc, "resource_evidence", {}),
            },
        )
        return 1
    atomic_write_json(Path(sys.argv[2]), result)
    return 0


def payload_mapping(value: object) -> dict[str, object]:
    """Normalize an untrusted JSON value to an object mapping."""

    return dict(value) if isinstance(value, Mapping) else {}


def numeric_payload(value: object) -> dict[str, float | None]:
    """Read finite-shape numeric JSON fields while excluding booleans."""

    result = {}
    for name, number in payload_mapping(value).items():
        if number is None:
            result[str(name)] = None
        elif isinstance(number, (int, float)) and not isinstance(number, bool):
            if not math.isfinite(number):
                raise ValueError(f"Worker metric {name} must be finite or null")
            result[str(name)] = float(number)
        else:
            raise ValueError(f"Worker metric {name} must be numeric or null")
    return result


def sample_result_from_payload(value: object) -> SampleResult:
    row = payload_mapping(value)
    return SampleResult(
        str(row["sample_id"]),
        numeric_payload(row["metrics"]),
        float(row["latency_seconds"]) if row["latency_seconds"] is not None else None,
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
    if "paddle" in sys.modules:
        sys.modules["paddle"].seed(seed)
