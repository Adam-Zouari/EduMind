"""Run one visual video candidate in a fresh process; no ASR is imported or called."""

from __future__ import annotations

import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from edumind.extraction import ExtractionProfile, ExtractionRequest, SourceKind
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.common.provenance import package_versions
from experiments.benchmarks.extraction.registry import build_experiment_registry
from experiments.benchmarks.extraction.document.profiles import parse_document_profile
from experiments.benchmarks.extraction.document.protocol import (
    protocol_from_worker as document_protocol_from_worker,
)
from experiments.benchmarks.common.process import json_worker_main
from experiments.benchmarks.extraction.video.candidates import frame_command, parse_candidate
from experiments.benchmarks.extraction.video.metrics import (
    aggregate_quality,
    bootstrap_quality,
    score_video,
)
from experiments.benchmarks.extraction.video.protocol import protocol_from_worker


def execute(payload: dict[str, object]) -> dict[str, object]:
    protocol = protocol_from_worker(payload["protocol"])
    document_protocol = document_protocol_from_worker(payload["document_protocol"])
    candidate = parse_candidate(str(payload["candidate"]), protocol)
    device = str(payload["device"])
    items = list(payload["items"])  # type: ignore[arg-type]
    random.Random(int(payload["seed"])).shuffle(items)
    image_engine = str(payload["image_engine"])
    image_profile = parse_document_profile(str(payload["image_candidate"]))
    if image_profile.runtime_engine != image_engine:
        raise ValueError("Visual worker image candidate and engine disagree")
    document_protocol.validate_candidate_factors(image_profile.factors)
    image_revision = str(payload["image_revision"])
    image_options = dict(payload["image_options"])  # type: ignore[arg-type]
    expected_image_options = {
        **document_protocol.parser_options(image_engine),
        **image_profile.options,
    }
    if any(image_options.get(name) != value for name, value in expected_image_options.items()):
        raise ValueError("Visual worker image options differ from the document protocol")
    extractor = build_experiment_registry().create(image_engine, SourceKind.IMAGE)
    timing_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    ffmpeg_commands: list[dict[str, object]] = []
    monitor = ResourceMonitor(require_vram=device == "cuda", report_zero_vram=device == "cpu")
    with tempfile.TemporaryDirectory(prefix="edumind-video-visual-") as raw_temp:
        temporary = Path(raw_temp)
        try:
            with monitor:
                cold_frames, cold_command = _extract_frames(
                    candidate, Path(str(items[0]["source_path"])), temporary / "cold"
                )
                ffmpeg_commands.append({"phase": "cold", "command": cold_command})
                request = _image_request(
                    cold_frames[0][0], image_engine, image_revision, device, image_options
                )
                initializer = getattr(extractor, "initialize_image_pipeline", None)
                if not callable(initializer):
                    raise RuntimeError(
                        f"Visual parser {image_engine} lacks an explicit image initialization hook"
                    )
                started = time.perf_counter()
                initializer(request)
                cold_load_seconds = time.perf_counter() - started

                for warmup in range(int(payload["warmups"])):
                    _process_video(
                        extractor,
                        candidate,
                        items[0],
                        temporary / f"warmup-{warmup}",
                        image_engine,
                        image_revision,
                        device,
                        image_options,
                    )

                for item in items:
                    outputs = []
                    for repetition in range(int(payload["repetitions"])):
                        started = time.perf_counter()
                        predictions, command, selected_frame_count = _process_video(
                            extractor,
                            candidate,
                            item,
                            temporary / f"{item['id']}-{repetition}",
                            image_engine,
                            image_revision,
                            device,
                            image_options,
                        )
                        latency = time.perf_counter() - started
                        outputs.append((predictions, latency, selected_frame_count))
                        timing_rows.append(
                            {
                                "sample_id": str(item["id"]),
                                "repetition": repetition + 1,
                                "latency_seconds": latency,
                                "duration_seconds": float(item["duration_seconds"]),
                                "visual_real_time_factor": latency
                                / float(item["duration_seconds"]),
                                "selected_frame_count": selected_frame_count,
                                "device": device,
                            }
                        )
                        ffmpeg_commands.append(
                            {
                                "sample_id": str(item["id"]),
                                "repetition": repetition + 1,
                                "command": command,
                            }
                        )
                    quality_predictions, quality_latency, quality_frame_count = outputs[0]
                    row = score_video(
                        item,
                        quality_predictions,
                        protocol.occurrence_matching,
                    )
                    row["quality_latency_seconds"] = quality_latency
                    row["predictions"] = quality_predictions
                    row["selected_frame_count"] = quality_frame_count
                    sample_rows.append(row)
        finally:
            resources = monitor.metrics()

    quality = aggregate_quality(sample_rows)
    intervals = bootstrap_quality(
        sample_rows,
        resamples=int(payload["bootstrap_resamples"]),
        seed=int(payload["seed"]),
        confidence=protocol.confidence_level,
    )
    latencies_by_video: dict[str, list[float]] = {}
    for row in timing_rows:
        latencies_by_video.setdefault(str(row["sample_id"]), []).append(
            float(row["latency_seconds"])
        )
    medians = [float(np.median(values)) for values in latencies_by_video.values()]
    measured_seconds = sum(float(row["latency_seconds"]) for row in timing_rows)
    measured_video_seconds = sum(float(row["duration_seconds"]) for row in timing_rows)
    operational = {
        "visual_real_time_factor": measured_seconds / measured_video_seconds,
        "p50_warm_visual_latency_seconds": float(np.quantile(medians, 0.50)),
        "p95_warm_visual_latency_seconds": float(np.quantile(medians, 0.95)),
        "cold_visual_pipeline_load_seconds": cold_load_seconds,
        "peak_visual_process_tree_ram_mb": float(resources["peak_process_tree_ram_mb"]),
        "peak_visual_vram_mb": (
            0.0 if device == "cpu" else float(resources["peak_vram_mb"])
        ),
        "mean_selected_frames_per_video": float(
            np.mean([float(row["selected_frame_count"]) for row in sample_rows])
        ),
    }
    intervals.update(
        _bootstrap_operational(
            sample_rows,
            timing_rows,
            operational,
            resamples=int(payload["bootstrap_resamples"]),
            seed=int(payload["seed"]),
            confidence=protocol.confidence_level,
        )
    )
    return {
        "samples": sample_rows,
        "timings": timing_rows,
        "metrics": quality,
        "operational": operational,
        "intervals": intervals,
        "ffmpeg_commands": ffmpeg_commands,
        "parameters": {
            "candidate": candidate.candidate,
            "strategy": candidate.strategy,
            "interval_seconds": candidate.interval_seconds,
            "scene_threshold": candidate.scene_threshold,
            "maximum_gap_seconds": candidate.maximum_gap_seconds,
            "ffmpeg_filter": candidate.ffmpeg_filter,
            "ffmpeg_frame_sync": protocol.frame_sync,
            "includes_frame_zero": protocol.include_frame_zero,
            "image_engine": image_engine,
            "image_revision": image_revision,
            "image_options": image_options,
            "device": device,
            "seed": int(payload["seed"]),
            "warmups": int(payload["warmups"]),
            "repetitions": int(payload["repetitions"]),
            "vram_measurement_method": monitor.vram_measurement_method,
            "package_versions": package_versions(
                (
                    "docling",
                    "paddleocr",
                    "paddlepaddle",
                    "paddlex",
                    "torch",
                    "transformers",
                    "onnxruntime",
                    "easyocr",
                    "rapidocr",
                )
            ),
        },
    }


def _process_video(
    extractor,
    candidate,
    item,
    directory,
    image_engine,
    image_revision,
    device,
    image_options,
):
    frames, command = _extract_frames(
        candidate, Path(str(item["source_path"])), directory
    )
    predictions = []
    for frame, timestamp in frames:
        request = _image_request(
            frame, image_engine, image_revision, device, image_options
        )
        document = extractor.extract(request, SourceKind.IMAGE)
        text = document.text.strip()
        if text:
            predictions.append(
                {
                    "text": text,
                    "timestamp": timestamp,
                    "warnings": len(document.warnings),
                }
            )
    return predictions, command, len(frames)


def _bootstrap_operational(
    rows, timings, estimates, *, resamples, seed, confidence
):
    if not resamples or len(rows) < 2:
        return {}
    by_sample: dict[str, list[dict[str, object]]] = {}
    for timing in timings:
        by_sample.setdefault(str(timing["sample_id"]), []).append(timing)
    names = (
        "visual_real_time_factor",
        "p50_warm_visual_latency_seconds",
        "p95_warm_visual_latency_seconds",
        "mean_selected_frames_per_video",
    )
    draws = {name: [] for name in names}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        sampled = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        sampled_timings = [
            timing
            for row in sampled
            for timing in by_sample[str(row["sample_id"])]
        ]
        per_video = [
            float(np.median([float(value["latency_seconds"]) for value in by_sample[str(row["sample_id"])] ]))
            for row in sampled
        ]
        draws["visual_real_time_factor"].append(
            sum(float(value["latency_seconds"]) for value in sampled_timings)
            / sum(float(value["duration_seconds"]) for value in sampled_timings)
        )
        draws["p50_warm_visual_latency_seconds"].append(
            float(np.quantile(per_video, 0.50))
        )
        draws["p95_warm_visual_latency_seconds"].append(
            float(np.quantile(per_video, 0.95))
        )
        draws["mean_selected_frames_per_video"].append(
            float(np.mean([float(row["selected_frame_count"]) for row in sampled]))
        )
    alpha = (1.0 - confidence) / 2.0
    return {
        name: {
            "estimate": estimates[name],
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
            "confidence": confidence,
            "resamples": len(values),
        }
        for name, values in draws.items()
    }


def _image_request(path, engine, revision, device, options):
    profile = ExtractionProfile(
        name=f"video-visual-{engine}",
        engine=engine,
        engine_revision=revision,
        preprocessing="raw",
        device=device,
        routing="video-keyframe",
        normalization="none",
        options=options,
    )
    return ExtractionRequest.from_path(
        path,
        source_kind=SourceKind.IMAGE,
        profile=profile,
        options=options,
    )


def _extract_frames(candidate, source: Path, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    pattern = directory / "frame-%05d.png"
    command = frame_command(candidate, str(source), str(pattern))
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    timestamps = [
        float(value)
        for value in re.findall(
            r"showinfo.*?pts_time:([0-9]+(?:\.[0-9]+)?)", completed.stderr
        )
    ]
    frames = sorted(directory.glob("frame-*.png"))
    if not frames or len(frames) != len(timestamps):
        raise RuntimeError("FFmpeg frame timestamps did not match extracted frames")
    if abs(timestamps[0]) > 1e-6:
        raise RuntimeError("Video selector did not include frame zero")
    return list(zip(frames, timestamps, strict=True)), command


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute))
