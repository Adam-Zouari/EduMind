"""Execute one ASR candidate in a fresh process."""

from __future__ import annotations

import random
import time
from pathlib import Path

from experiments.benchmarks.common.process import json_worker_main
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.extraction.audio.adapters import build_runtime
from experiments.benchmarks.extraction.audio.evaluate import (
    aggregate,
    normalize_transcript,
    score_nonspeech,
    score_speech,
)
from experiments.benchmarks.extraction.audio.protocol import protocol_from_worker


def execute(payload: dict[str, object]) -> dict[str, object]:
    device = str(payload["device"])
    protocol = protocol_from_worker(payload["protocol"])
    runtime = build_runtime(
        str(payload["candidate"]),
        payload["model_lock"],  # type: ignore[arg-type]
        device,
        protocol,
    )
    speech = list(payload["speech"])  # type: ignore[arg-type]
    reliability = list(payload["reliability"])  # type: ignore[arg-type]
    random.Random(int(payload["seed"])).shuffle(speech)
    sample_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    monitor = ResourceMonitor(
        require_vram=device == "cuda",
        report_zero_vram=device == "cpu",
    )
    try:
        with monitor:
            started = time.perf_counter()
            runtime.load()
            cold_model_load_seconds = time.perf_counter() - started

            for _ in range(int(payload["warmups"])):
                runtime.transcribe(Path(str(speech[0]["canonical_path"])))

            for item in speech:
                outputs = []
                for repetition in range(int(payload["repetitions"])):
                    started = time.perf_counter()
                    transcript = runtime.transcribe(Path(str(item["canonical_path"])))
                    latency = time.perf_counter() - started
                    outputs.append((transcript, latency))
                    timing_rows.append(
                        {
                            "sample_id": str(item["id"]),
                            "repetition": repetition + 1,
                            "latency_seconds": latency,
                            "duration_seconds": float(item["duration_seconds"]),
                            "real_time_factor": latency
                            / float(item["duration_seconds"]),
                            "device": device,
                        }
                    )
                quality, quality_latency = outputs[0]
                normalized_outputs = [
                    normalize_transcript(transcript.text) for transcript, _ in outputs
                ]
                sample_rows.append(
                    score_speech(
                        item,
                        quality.text,
                        quality.segments,
                        quality_latency_seconds=quality_latency,
                        repeat_transcript_agreement=len(set(normalized_outputs)) == 1,
                        warnings=quality.warnings,
                        alignment_threshold=protocol.alignment_threshold,
                        timestamp_tolerance_seconds=float(
                            protocol.audio["manifest_duration_tolerance_seconds"]
                        ),
                    )
                )

            for item in reliability:
                started = time.perf_counter()
                transcript = runtime.transcribe(Path(str(item["canonical_path"])))
                sample_rows.append(
                    score_nonspeech(
                        item,
                        transcript.text,
                        latency_seconds=time.perf_counter() - started,
                        warnings=transcript.warnings,
                    )
                )
    finally:
        runtime.close()

    resources = monitor.metrics()
    if "peak_process_tree_ram_mb" not in resources:
        raise RuntimeError("ASR worker could not measure process-tree RAM")
    metrics, intervals = aggregate(
        sample_rows,
        timing_rows,
        cold_model_load_seconds=cold_model_load_seconds,
        peak_process_tree_ram_mb=float(resources["peak_process_tree_ram_mb"]),
        peak_vram_mb=0.0 if device == "cpu" else float(resources["peak_vram_mb"]),
        resamples=int(payload["bootstrap_resamples"]),
        seed=int(payload["seed"]),
        confidence=protocol.confidence_level,
    )
    return {
        "samples": sample_rows,
        "timings": timing_rows,
        "metrics": metrics,
        "intervals": intervals,
        "parameters": {
            **runtime.parameters(),
            "vram_measurement_method": monitor.vram_measurement_method,
            "seed": int(payload["seed"]),
            "warmups": int(payload["warmups"]),
            "repetitions": int(payload["repetitions"]),
        },
    }


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute))
