"""Execute every scheduled ASR attempt in one fresh, measured model lifecycle."""

from __future__ import annotations

import random
import time
from pathlib import Path

from edumind.common.artifacts import stable_hash
from edumind.common.model_placement import inspect_model_placement
from experiments.benchmarks.common.process import (
    json_worker_main,
    seed_deterministically,
)
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.extraction.audio.adapters import build_runtime
from experiments.benchmarks.extraction.audio.evaluate import (
    aggregate,
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
    runtime = build_runtime(
        str(payload["candidate"]), payload["model_lock"], device, protocol
    )
    if payload.get("mode") == "preflight":
        return _preflight(runtime, payload, device, protocol)
    execution = protocol.profile(str(payload["profile"]))
    if device not in (execution.devices or (execution.device,)):
        raise ValueError("ASR worker device violates the frozen profile")
    speech, controls = list(payload["speech"]), list(payload["reliability"])
    random.Random(protocol.meta.seed).shuffle(speech)
    if not speech:
        raise ValueError("ASR worker has no speech inputs")
    timing_rows, outputs = [], {}
    before = after = None
    monitor = ResourceMonitor(require_vram=device == "cuda", device=device)
    try:
        with monitor:
            started = time.perf_counter()
            seed_deterministically(protocol.meta.seed)
            runtime.load()
            synchronize(device)
            cold = time.perf_counter() - started
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
            for _ in range(execution.warmups):
                runtime.transcribe(Path(str(speech[0]["canonical_path"])))
                synchronize(device)
            for sample_type, items in (("speech", speech), ("nonspeech", controls)):
                for item in items:
                    attempts = []
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
                        attempts.append(transcript)
                        timing_rows.append(
                            {
                                "sample_id": str(item["id"]),
                                "sample_type": sample_type,
                                "repetition": repetition,
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
                    outputs[str(item["id"])] = attempts
            after = (
                inspect_model_placement(
                    *runtime.placement_models(), expected_device="cuda"
                )
                if device == "cuda"
                else None
            )
            if after and after["status"] != "qualified":
                raise RuntimeError("ASR inference changed CUDA weight placement")
    finally:
        runtime.close()
    resources = monitor.metrics()
    sample_rows = []
    for sample_type, items in (("speech", speech), ("nonspeech", controls)):
        for item in items:
            attempts = outputs[str(item["id"])]
            timings = [
                row for row in timing_rows if row["sample_id"] == str(item["id"])
            ]
            first = attempts[0]
            if first is None:
                row = {
                    "sample_id": str(item["id"]),
                    "sample_type": sample_type,
                    "conditions": list(item.get("conditions", [])),
                    "duration_seconds": float(item["duration_seconds"]),
                    "quality_latency_seconds": None,
                    "first_attempt_success": False,
                    "metric_status": "unavailable",
                    "reason": "first_attempt_failed",
                }
            elif sample_type == "speech":
                row = score_speech(
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
                row = score_nonspeech(
                    item,
                    first.text,
                    latency_seconds=float(timings[0]["latency_seconds"]),
                    warnings=first.warnings,
                )
            rates = attempt_rates(
                [
                    normalize_transcript(value.text) if value is not None else None
                    for value in attempts
                ]
            )
            row.update(
                {
                    "source_group_id": source_id(item),
                    "transcript_repeatability_success_rate": rates[
                        "repeatability_success_rate"
                    ],
                    "attempt_failure_rate": rates["attempt_failure_rate"],
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
    metrics, intervals = aggregate(
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
        "intervals": intervals,
        "resource_samples": monitor.samples(),
        "gpu_identity": monitor.gpu_identity,
        "placement_before": before,
        "placement_after": after,
        "parameters": {
            **runtime.parameters(),
            "vram_measurement_method": monitor.vram_measurement_method,
            "seed": protocol.meta.seed,
            "warmups": execution.warmups,
            "repetitions": execution.repetitions,
        },
    }


def _preflight(
    runtime, payload: dict[str, object], device: str, protocol
) -> dict[str, object]:
    if device != "cuda":
        raise ValueError("ASR preflight requires CUDA")
    speech = list(payload["speech"])  # type: ignore[arg-type]
    if not speech:
        raise ValueError("ASR preflight requires one canonical speech sample")
    monitor = ResourceMonitor(require_vram=True)
    try:
        with monitor:
            seed_deterministically(protocol.meta.seed)
            runtime.load()
            synchronize(device)
            before = inspect_model_placement(
                *runtime.placement_models(), expected_device="cuda"
            )
            output = runtime.transcribe(Path(str(speech[0]["canonical_path"])))
            validate_prediction(
                output.text,
                output.segments,
                float(speech[0]["duration_seconds"]),
                protocol.timestamp_tolerance_seconds,
            )
            synchronize(device)
            after = inspect_model_placement(
                *runtime.placement_models(), expected_device="cuda"
            )
    finally:
        runtime.close()
    resources = monitor.metrics()
    placement = after if after["status"] != "qualified" else before
    if before["status"] == "qualified" and after["status"] == "qualified":
        placement = after
    return {
        "placement": placement,
        "placement_before": before,
        "placement_after": after,
        "peak_vram_mb": float(resources["peak_vram_mb"]),
        "peak_process_tree_ram_mb": resources["peak_process_tree_ram_mb"],
        "vram_measurement_method": monitor.vram_measurement_method,
        "stress_sample_id": str(speech[0]["id"]),
        "stress_duration_seconds": float(speech[0]["duration_seconds"]),
        "parameters": runtime.parameters(),
    }


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute))
