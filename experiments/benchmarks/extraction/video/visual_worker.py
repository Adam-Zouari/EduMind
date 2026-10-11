"""Fresh visual-only extraction lifecycle; no ASR model is loaded here."""

from __future__ import annotations

import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from edumind.common.artifacts import stable_hash
from edumind.common.model_placement import inspect_model_placement
from edumind.extraction import ExtractionProfile, ExtractionRequest, SourceKind
from experiments.benchmarks.common.process import (
    json_worker_main,
    seed_deterministically,
)
from experiments.benchmarks.common.provenance import package_versions
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.extraction.audio.protocol import (
    protocol_from_worker as audio_protocol_from_worker,
)
from experiments.benchmarks.extraction.document.metrics import _document_fingerprint
from experiments.benchmarks.extraction.document.profiles import parse_document_profile
from experiments.benchmarks.extraction.document.protocol import (
    protocol_from_worker as document_protocol_from_worker,
)
from experiments.benchmarks.extraction.document.worker import (
    pipeline_parameters,
    synchronize,
)
from experiments.benchmarks.extraction.registry import build_experiment_registry
from experiments.benchmarks.extraction.scoring import (
    attempt_rates,
    bootstrap_sources,
    source_id,
)
from experiments.benchmarks.extraction.video.candidates import (
    frame_command,
    parse_candidate,
)
from experiments.benchmarks.extraction.video.frozen_asr import load_frozen_asr_artifact
from experiments.benchmarks.extraction.video.metrics import (
    QUALITY_DIRECTIONS,
    aggregate_quality,
    bootstrap_quality,
    quality_statuses,
    score_video,
)
from experiments.benchmarks.extraction.video.protocol import protocol_from_worker


class VisualAttemptFailure(RuntimeError):
    def __init__(self, message, selected_frame_count, command):
        super().__init__(message)
        self.selected_frame_count = selected_frame_count
        self.command = command


def execute(payload):
    protocol = protocol_from_worker(payload["protocol"])
    document_protocol = document_protocol_from_worker(payload["document_protocol"])
    candidate = parse_candidate(str(payload["candidate"]), protocol)
    device = str(payload["device"])
    preflight = payload.get("mode") == "preflight"
    execution = (
        protocol.preflight if preflight else protocol.profile(str(payload["profile"]))
    )
    if device not in (
        ("cuda",) if preflight else execution.devices or (execution.device,)
    ):
        raise ValueError("Visual worker device violates the frozen execution profile")
    items = list(payload["items"])
    random.Random(protocol.meta.seed).shuffle(items)
    if not items:
        raise ValueError("Visual worker requires videos")
    if not preflight:
        audio_protocol = audio_protocol_from_worker(payload["audio_protocol"])
        load_frozen_asr_artifact(
            Path(payload["frozen_asr_path"]),
            manifest_checksum=payload["manifest_checksum"],
            protocol_checksum=protocol.meta.checksum,
            audio_protocol_checksum=audio_protocol.meta.checksum,
            timestamp_tolerance_seconds=audio_protocol.timestamp_tolerance_seconds,
            sample_ids=[str(item["id"]) for item in items],
            expected_identity=payload["frozen_asr_identity"],
            expected_device=device,
            expected_profile=str(payload["profile"]),
            expected_checksum=payload["frozen_asr_checksum"],
            expected_durations={
                str(item["id"]): float(item["duration_seconds"]) for item in items
            },
        )
    image_profile = parse_document_profile(str(payload["image_candidate"]))
    image_engine = image_profile.runtime_engine
    if image_engine != str(payload["image_engine"]):
        raise ValueError("Visual worker image candidate and engine disagree")
    document_protocol.validate_candidate_factors(image_profile.factors)
    image_options = dict(payload["image_options"])
    expected = {
        **document_protocol.parser_options(image_engine),
        **image_profile.options,
    }
    if any(image_options.get(name) != value for name, value in expected.items()):
        raise ValueError("Visual image options differ from the document protocol")
    image_revision = str(payload["image_revision"])
    timings, commands, outputs = [], [], {}
    monitor = ResourceMonitor(require_vram=device == "cuda", device=device)
    with tempfile.TemporaryDirectory(prefix="video-visual-") as raw:
        temporary = Path(raw)
        with monitor:
            seed_deterministically(protocol.meta.seed)
            cold_frames, command = _extract_frames(
                candidate, Path(str(items[0]["source_path"])), temporary / "cold"
            )
            commands.append({"phase": "initialize", "command": command})
            request = _image_request(
                cold_frames[0][0], image_engine, image_revision, device, image_options
            )
            # Frame preparation is not model loading.
            started = time.perf_counter()
            extractor = build_experiment_registry().create(
                image_engine, SourceKind.IMAGE
            )
            initializer = getattr(extractor, "initialize_image_pipeline", None)
            if not callable(initializer):
                raise RuntimeError("Visual parser lacks an explicit readiness hook")
            initializer(request)
            synchronize(device, image_engine)
            cold = time.perf_counter() - started
            before = (
                inspect_model_placement(extractor, expected_device=device)
                if preflight or device == "cuda"
                else None
            )
            if before and not preflight and before["status"] != "qualified":
                raise RuntimeError(f"Visual backend violates CUDA placement: {before}")
            for warmup in range(execution.warmups):
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
                synchronize(device, image_engine)
            for index, item in enumerate(items):
                attempts = []
                for repetition in range(1, execution.repetitions + 1):
                    seed_deterministically(protocol.meta.seed)
                    started = time.perf_counter()
                    predictions = fingerprint = error = count = command = None
                    try:
                        predictions, command, count, fingerprint = _process_video(
                            extractor,
                            candidate,
                            item,
                            temporary / f"{index}-{repetition}",
                            image_engine,
                            image_revision,
                            device,
                            image_options,
                        )
                        synchronize(device, image_engine)
                    except Exception as exc:  # noqa: BLE001 - every scheduled attempt remains visible
                        error = f"{type(exc).__name__}: {exc}"
                        count = getattr(exc, "selected_frame_count", None)
                        command = getattr(exc, "command", None)
                        predictions = fingerprint = None
                    elapsed = time.perf_counter() - started
                    attempts.append(
                        {"predictions": predictions, "fingerprint": fingerprint}
                    )
                    timings.append(
                        {
                            "sample_id": str(item["id"]),
                            "repetition": repetition,
                            "latency_seconds": elapsed,
                            "duration_seconds": float(item["duration_seconds"]),
                            "success": error is None,
                            "error": error,
                            "selected_frame_count": count,
                        }
                    )
                    if command is not None:
                        commands.append(
                            {
                                "sample_id": str(item["id"]),
                                "repetition": repetition,
                                "command": command,
                            }
                        )
                outputs[str(item["id"])] = attempts
            after = (
                inspect_model_placement(extractor, expected_device=device)
                if preflight or device == "cuda"
                else None
            )
            if after and not preflight and after["status"] != "qualified":
                raise RuntimeError(f"Visual backend changed CUDA placement: {after}")
    resources = monitor.metrics()
    if preflight:
        if any(not row["success"] for row in timings):
            raise RuntimeError("Visual stress inference failed")
        return {
            "placement": after if after["status"] != "qualified" else before,
            "placement_before": before,
            "placement_after": after,
            **resources,
            "vram_measurement_method": monitor.vram_measurement_method,
            "stress_sample_ids": [str(item["id"]) for item in items],
        }
    rows = []
    for item in items:
        attempts = outputs[str(item["id"])]
        timing = [value for value in timings if value["sample_id"] == str(item["id"])]
        first = attempts[0]
        row = (
            score_video(item, first["predictions"], protocol.occurrence_matching)
            if first["predictions"] is not None
            else {
                **{name: None for name in QUALITY_DIRECTIONS},
                "sample_id": str(item["id"]),
                "duration_seconds": float(item["duration_seconds"]),
                "metric_status": "unavailable",
                "reason": "first_attempt_failed",
            }
        )
        rates = attempt_rates([attempt["fingerprint"] for attempt in attempts])
        row.update(
            {
                "repeatability_success_rate": rates["repeatability_success_rate"],
                "attempt_failure_rate": rates["attempt_failure_rate"],
                "source_group_id": source_id(item),
                "first_attempt_success": bool(timing[0]["success"]),
                "quality_latency_seconds": timing[0]["latency_seconds"]
                if timing[0]["success"]
                else None,
                "selected_frame_count": timing[0]["selected_frame_count"],
                "predictions": first["predictions"],
                "_timings": timing,
            }
        )
        row["metric_statuses"] = quality_statuses(
            row, has_occurrences=bool(item["visual_occurrences"])
        )
        rows.append(row)
    operational = _operational_statistics(rows)
    operational.update(
        {
            "cold_visual_pipeline_load_seconds": cold,
            "peak_visual_process_tree_ram_mb": resources["peak_process_tree_ram_mb"],
            "peak_visual_vram_mb": resources["peak_vram_mb"],
        }
    )
    intervals = bootstrap_quality(
        rows,
        resamples=execution.bootstrap_resamples,
        seed=protocol.meta.seed,
        confidence=protocol.confidence_level,
        minimum_sources=protocol.minimum_ci_sources,
    )
    intervals.update(
        bootstrap_sources(
            rows,
            _operational_statistics,
            resamples=execution.bootstrap_resamples,
            seed=protocol.meta.seed,
            confidence=protocol.confidence_level,
            minimum_sources=protocol.minimum_ci_sources,
        )
    )
    latency_intervals = bootstrap_sources(
        rows,
        _operational_statistics,
        resamples=execution.bootstrap_resamples,
        seed=protocol.meta.seed,
        confidence=protocol.confidence_level,
        minimum_sources=protocol.minimum_latency_ci_sources,
    )
    for name in ("p50_warm_visual_latency_seconds", "p95_warm_visual_latency_seconds"):
        if name in latency_intervals:
            intervals[name] = latency_intervals[name]
    metrics = aggregate_quality(rows)
    for name in QUALITY_DIRECTIONS:
        metrics[name + ".scheduled_count"] = float(len(rows))
        metrics[name + ".eligible_count"] = float(
            sum(
                row["metric_statuses"][name]["status"] != "inapplicable" for row in rows
            )
        )
        metrics[name + ".contributing_count"] = float(
            sum(row.get(name) is not None for row in rows)
        )
    return {
        "samples": [
            {key: value for key, value in row.items() if key != "_timings"}
            for row in rows
        ],
        "timings": timings,
        "outputs": {"attempts": outputs},
        "metrics": metrics,
        "operational": operational,
        "intervals": intervals,
        "ffmpeg_commands": commands,
        "resource_samples": monitor.samples(),
        "gpu_identity": monitor.gpu_identity,
        "placement_before": before,
        "placement_after": after,
        "parameters": {
            "initialized_pipeline_options": pipeline_parameters(extractor),
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
            "seed": protocol.meta.seed,
            "warmups": execution.warmups,
            "repetitions": execution.repetitions,
            "canonical_numeric_precision": "full-precision",
            "vram_measurement_method": monitor.vram_measurement_method,
            "package_versions": package_versions(
                ("docling", "paddleocr", "paddlepaddle", "torch", "transformers")
            ),
        },
    }


def _operational_statistics(rows):
    completed = [value for row in rows for value in row["_timings"] if value["success"]]
    duration = sum(float(value["duration_seconds"]) for value in completed)
    result = {
        "visual_real_time_factor": sum(
            float(value["latency_seconds"]) for value in completed
        )
        / duration
        if duration
        else None
    }
    medians = [
        float(
            np.median(
                [
                    value["latency_seconds"]
                    for value in row["_timings"]
                    if value["success"]
                ]
            )
        )
        for row in rows
        if any(value["success"] for value in row["_timings"])
    ]
    for percentile in (50, 95):
        result[f"p{percentile}_warm_visual_latency_seconds"] = (
            float(np.quantile(medians, percentile / 100)) if medians else None
        )
    counts = [
        row["selected_frame_count"]
        for row in rows
        if row["selected_frame_count"] is not None
    ]
    result["mean_selected_frames_per_video"] = (
        float(np.mean(counts)) if counts else None
    )
    return result


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
    if any(
        not np.isfinite(timestamp)
        or not 0 <= timestamp <= float(item["duration_seconds"])
        for _, timestamp in frames
    ):
        raise VisualAttemptFailure(
            "Selected frame timestamp is outside the video", len(frames), command
        )
    predictions, signatures = [], []
    for frame, timestamp in frames:
        request = _image_request(
            frame, image_engine, image_revision, device, image_options
        )
        try:
            document = extractor.extract(request, SourceKind.IMAGE)
        except Exception as exc:
            raise VisualAttemptFailure(str(exc), len(frames), command) from exc
        signatures.append(
            {"timestamp": timestamp, "document": _document_fingerprint(document)}
        )
        text = document.text.strip()
        if text:
            predictions.append(
                {
                    "text": text,
                    "timestamp": timestamp,
                    "warnings": len(document.warnings),
                }
            )
    return predictions, command, len(frames), stable_hash(signatures)


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
        raise VisualAttemptFailure(
            "FFmpeg frame timestamps did not match extracted frames",
            len(frames),
            command,
        )
    if abs(timestamps[0]) > 1e-6:
        raise VisualAttemptFailure(
            "Video selector did not include frame zero", len(frames), command
        )
    return list(zip(frames, timestamps, strict=True)), command


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute))
