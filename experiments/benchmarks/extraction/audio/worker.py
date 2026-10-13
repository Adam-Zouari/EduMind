"""Execute every scheduled ASR attempt in one fresh, measured model lifecycle."""

from __future__ import annotations

import random
import time
from pathlib import Path

from edumind.common.artifacts import stable_hash
from edumind.common.model_placement import inspect_model_placement
from experiments.benchmarks.common.process import (
    WorkerExecutionError,
    WorkerMeasurementError,
    json_worker_main,
    seed_deterministically,
)
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.extraction.audio.adapters import build_runtime
from experiments.benchmarks.extraction.audio.evaluate import (
    aggregate,
    metric_statuses,
    normalize_transcript,
    score_nonspeech,
    score_speech,
    validate_prediction,
)
from experiments.benchmarks.extraction.audio.protocol import protocol_from_worker
from experiments.benchmarks.extraction.scoring import attempt_rates, source_id


def synchronize(device):
    if device == "cuda":
        import torch

        torch.cuda.synchronize()


def execute(payload):
    device = str(payload["device"])
    protocol = protocol_from_worker(payload["protocol"])
    preflight = payload.get("mode") == "preflight"
    if payload.get("mode") not in {None, "preflight"}:
        raise ValueError("Unknown ASR worker mode")
    execution = protocol.profile(str(payload["profile"]))
    if device not in (execution.devices or (execution.device,)):
        raise ValueError("ASR worker device violates the frozen profile")
    runtime = build_runtime(
        str(payload["candidate"]), payload["model_lock"], device, protocol
    )
    speech, controls = list(payload["speech"]), list(payload["reliability"])
    if preflight:
        if device != "cuda" or len(speech) != 1 or controls:
            raise ValueError("ASR preflight requires CUDA and exactly one speech input")
        execution = protocol.preflight
    random.Random(protocol.meta.seed).shuffle(speech)
    if not speech:
        raise ValueError("ASR worker has no speech inputs")
    timing_rows, outputs = [], {}
    cold = execution_error = None
    parameters = {}
    phase = "load"
    before = after = None
    monitor = ResourceMonitor(require_vram=device == "cuda", device=device)
    lifecycle_started = time.perf_counter()
    try:
        with monitor:
            started = time.perf_counter()
            seed_deterministically(protocol.meta.seed)
            runtime.load()
            synchronize(device)
            cold = time.perf_counter() - started
            phase = "placement_before"
            before = (
                inspect_model_placement(
                    *runtime.placement_models(), expected_device="cuda"
                )
                if device == "cuda"
                else None
            )
            if before and before["status"] != "qualified":
                raise RuntimeError(
                    "ASR CUDA weight placement violates the no-offload contract"
                )
            phase = "warmup"
            for _ in range(execution.warmups):
                seed_deterministically(protocol.meta.seed)
                transcript = runtime.transcribe(Path(str(speech[0]["canonical_path"])))
                validate_prediction(
                    transcript.text,
                    transcript.segments,
                    float(speech[0]["duration_seconds"]),
                    protocol.timestamp_tolerance_seconds,
                )
                synchronize(device)
            phase = "inference"
            for sample_type, items in (("speech", speech), ("nonspeech", controls)):
                for item in items:
                    repetitions = (
                        execution.repetitions if sample_type == "speech" else 1
                    )
                    for repetition in range(1, repetitions + 1):
                        seed_deterministically(protocol.meta.seed)
                        started = time.perf_counter()
                        transcript = error = None
                        try:
                            transcript = runtime.transcribe(
                                Path(str(item["canonical_path"]))
                            )
                            validate_prediction(
                                transcript.text,
                                transcript.segments,
                                float(item["duration_seconds"]),
                                float(
                                    protocol.audio[
                                        "manifest_duration_tolerance_seconds"
                                    ]
                                ),
                            )
                            synchronize(device)
                        except Exception as exc:  # noqa: BLE001 - retain and continue scheduled attempts
                            error = f"{type(exc).__name__}: {exc}"
                            transcript = None
                        elapsed = time.perf_counter() - started
                        timing_rows.append(
                            {
                                "sample_id": str(item["id"]),
                                "sample_type": sample_type,
                                "repetition": repetition,
                                "executed": True,
                                "success": error is None,
                                "error": error,
                                "latency_seconds": elapsed,
                                "duration_seconds": float(item["duration_seconds"]),
                                "device": device,
                                "projected_transcript_sha256": stable_hash(
                                    normalize_transcript(transcript.text)
                                )
                                if transcript
                                else None,
                            }
                        )
                        outputs.setdefault(str(item["id"]), []).append(transcript)
            phase = "telemetry"
        phase = "placement_after"
        after = (
            inspect_model_placement(*runtime.placement_models(), expected_device="cuda")
            if device == "cuda"
            else None
        )
        if after and after["status"] != "qualified":
            raise RuntimeError("ASR inference changed CUDA weight placement")
    except Exception as exc:  # noqa: BLE001 - preserve completed observations
        before = getattr(exc, "placement_report", before)
        execution_error = {
            "phase": phase,
            "error": f"{type(exc).__name__}: {exc}",
            "lifecycle_elapsed_seconds": time.perf_counter() - lifecycle_started,
        }
    finally:
        try:
            parameters = runtime.parameters()
        except Exception as exc:  # noqa: BLE001 - keep lifecycle evidence when metadata is unavailable
            parameter_error = f"{type(exc).__name__}: {exc}"
            if execution_error is None:
                execution_error = {
                    "phase": "parameters",
                    "error": parameter_error,
                    "lifecycle_elapsed_seconds": time.perf_counter()
                    - lifecycle_started,
                }
            else:
                execution_error["parameter_error"] = parameter_error
        try:
            runtime.close()
        except Exception as exc:  # noqa: BLE001 - cleanup must not erase the original error
            cleanup_error = f"{type(exc).__name__}: {exc}"
            if execution_error is None:
                execution_error = {
                    "phase": "close",
                    "error": cleanup_error,
                    "lifecycle_elapsed_seconds": time.perf_counter()
                    - lifecycle_started,
                }
            else:
                execution_error["cleanup_error"] = cleanup_error
    try:
        resources = monitor.metrics()
    except Exception as exc:  # noqa: BLE001 - retain partial telemetry and attempts
        execution_error = execution_error or {
            "phase": "telemetry",
            "error": f"{type(exc).__name__}: {exc}",
        }
        execution_error["resource_evidence"] = getattr(exc, "resource_evidence", {})
        resources = {"peak_process_tree_ram_mb": None, "peak_vram_mb": None}
    if device == "cpu":
        resources["peak_vram_mb"] = None
    elif resources["peak_vram_mb"] is None:
        execution_error = execution_error or {
            "phase": "telemetry",
            "error": "ASR CUDA telemetry is unavailable",
        }
    if preflight:
        placement = after if after and after["status"] != "qualified" else before
        evidence = {
            **resources,
            "placement": placement,
            "placement_before": before,
            "placement_after": after,
            "resource_samples": monitor.samples(),
            "vram_measurement_method": monitor.vram_measurement_method,
            "stress_sample_id": str(speech[0]["id"]),
            "stress_duration_seconds": float(speech[0]["duration_seconds"]),
            "parameters": parameters,
            "timings": timing_rows,
            "execution_error": execution_error,
        }
        if placement and placement["status"] != "qualified":
            return evidence
        if timing_rows and not timing_rows[0]["success"]:
            raise WorkerExecutionError(timing_rows[0]["error"], evidence)
        if execution_error:
            error_type = (
                WorkerMeasurementError
                if monitor.vram_measurement_method == "unavailable"
                or execution_error["phase"] == "telemetry"
                else WorkerExecutionError
            )
            raise error_type(
                execution_error["error"],
                {**evidence, "execution_error": execution_error},
            )
        return evidence
    observed = {(row["sample_id"], row["repetition"]) for row in timing_rows}
    for sample_type, items in (("speech", speech), ("nonspeech", controls)):
        for item in items:
            for repetition in range(
                1, (execution.repetitions if sample_type == "speech" else 1) + 1
            ):
                if (str(item["id"]), repetition) not in observed:
                    timing_rows.append(
                        {
                            "sample_id": str(item["id"]),
                            "sample_type": sample_type,
                            "repetition": repetition,
                            "executed": False,
                            "success": None,
                            "error": None,
                            "latency_seconds": None,
                            "duration_seconds": float(item["duration_seconds"]),
                            "device": device,
                            "projected_transcript_sha256": None,
                            "not_executed_reason": execution_error["phase"]
                            if execution_error
                            else "missing_attempt",
                        }
                    )
    timing_rows.sort(key=lambda row: (row["sample_id"], row["repetition"]))
    sample_rows = []
    scoring_evidence = {
        **resources,
        "cold_model_load_seconds": cold,
        "execution_error": execution_error,
        "samples": sample_rows,
        "timings": timing_rows,
        "resource_samples": monitor.samples(),
        "placement_before": before,
        "placement_after": after,
        "parameters": parameters,
    }
    for sample_type, items in (("speech", speech), ("nonspeech", controls)):
        for item in items:
            attempts = outputs.get(str(item["id"]), [])
            timings = [
                row for row in timing_rows if row["sample_id"] == str(item["id"])
            ]
            first = attempts[0] if attempts else None
            if first is None:
                row = {
                    "sample_id": str(item["id"]),
                    "sample_type": sample_type,
                    "conditions": list(item.get("conditions", [])),
                    "duration_seconds": float(item["duration_seconds"]),
                    "quality_latency_seconds": None,
                    "first_attempt_success": False if timings[0]["executed"] else None,
                    "metric_status": "unavailable"
                    if timings[0]["executed"]
                    else "incomplete",
                    "reason": "first_attempt_failed"
                    if timings[0]["executed"]
                    else "first_attempt_not_executed",
                }
            elif sample_type == "speech":
                row = _evaluate(
                    score_speech,
                    scoring_evidence,
                    item,
                    first.text,
                    first.segments,
                    quality_latency_seconds=float(timings[0]["latency_seconds"]),
                    warnings=first.warnings,
                    alignment_threshold=protocol.alignment_threshold,
                    timestamp_tolerance_seconds=float(
                        protocol.audio["manifest_duration_tolerance_seconds"]
                    ),
                )
            else:
                row = _evaluate(
                    score_nonspeech,
                    scoring_evidence,
                    item,
                    first.text,
                    latency_seconds=float(timings[0]["latency_seconds"]),
                    warnings=first.warnings,
                )
            rates = (
                attempt_rates(
                    [
                        normalize_transcript(value.text) if value is not None else None
                        for value in attempts
                    ]
                )
                if attempts
                else {"repeatability_success_rate": None, "attempt_failure_rate": None}
            )
            row.update(
                {
                    "first_attempt_executed": timings[0]["executed"],
                    "scheduled_attempt_count": len(timings),
                    "executed_attempt_count": sum(
                        value["executed"] for value in timings
                    ),
                    "failed_attempt_count": sum(
                        value["success"] is False for value in timings
                    ),
                    "unexecuted_attempt_count": sum(
                        not value["executed"] for value in timings
                    ),
                    "source_group_id": source_id(item),
                    "transcript_repeatability_success_rate": rates[
                        "repeatability_success_rate"
                    ]
                    if all(value["executed"] for value in timings)
                    else None,
                    "attempt_failure_rate": rates["attempt_failure_rate"]
                    if all(value["executed"] for value in timings)
                    else None,
                    "planned_reference_word_count": len(
                        normalize_transcript(str(item.get("reference", ""))).split()
                    ),
                    "planned_reference_character_count": len(
                        normalize_transcript(str(item.get("reference", "")))
                    ),
                    "planned_reference_timed_segment_count": len(
                        [
                            segment
                            for segment in item.get("reference_segments", [])
                            if normalize_transcript(str(segment["text"]))
                        ]
                    ),
                }
            )
            sample_rows.append(row)
    metrics, intervals = _evaluate(
        aggregate,
        scoring_evidence,
        sample_rows,
        timing_rows,
        cold_model_load_seconds=cold,
        peak_process_tree_ram_mb=resources["peak_process_tree_ram_mb"],
        peak_vram_mb=resources["peak_vram_mb"],
        resamples=execution.bootstrap_resamples,
        seed=protocol.meta.seed,
        confidence=protocol.confidence_level,
        minimum_sources=protocol.minimum_ci_sources,
        minimum_latency_sources=protocol.minimum_latency_ci_sources,
    )
    return {
        "samples": sample_rows,
        "timings": timing_rows,
        "metrics": metrics,
        "metric_statuses": _evaluate(
            metric_statuses,
            scoring_evidence,
            metrics,
            sample_rows,
            timing_rows,
            device,
            execution_error,
        ),
        "intervals": intervals,
        "resource_samples": monitor.samples(),
        "gpu_identity": monitor.gpu_identity,
        "placement_before": before,
        "placement_after": after,
        "execution_error": execution_error,
        "parameters": {
            **parameters,
            "protocol_version": protocol.meta.version,
            "protocol_checksum": protocol.meta.checksum,
            "vram_measurement_method": monitor.vram_measurement_method,
            "seed": protocol.meta.seed,
            "warmups": execution.warmups,
            "repetitions": execution.repetitions,
        },
    }


def _evaluate(operation, evidence, *args, **kwargs):
    """Retain observed attempts if offline scoring or aggregation cannot finish."""
    started = time.perf_counter()
    try:
        return operation(*args, **kwargs)
    except Exception as exc:
        failure = {
            "phase": "offline_evaluation",
            "operation": operation.__name__,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.perf_counter() - started,
        }
        if evidence.get("execution_error") is not None:
            failure["preceding_execution_error"] = evidence["execution_error"]
        raise WorkerExecutionError(
            f"ASR offline evaluation failed: {failure['error']}",
            {**evidence, "execution_error": failure},
        ) from exc


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute))
