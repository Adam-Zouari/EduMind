"""Fresh visual-only extraction lifecycle; no ASR model is loaded here."""

from __future__ import annotations

import random
import re
import subprocess
import tempfile
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from edumind.common.artifacts import stable_hash
from edumind.common.model_placement import inspect_model_placement
from edumind.extraction import ExtractionProfile, ExtractionRequest, SourceKind
from experiments.benchmarks.common.process import (
    WorkerExecutionError,
    WorkerMeasurementError,
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
    _error_text,
    pipeline_parameters,
    synchronize,
    validate_document_payload,
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
    OPERATIONAL_DIRECTIONS,
    aggregate_quality,
    aggregate_quality_statuses,
    bootstrap_quality,
    quality_statuses,
    quality_support_counts,
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
    if payload.get("mode") not in {None, "preflight"}:
        raise ValueError("Unknown visual worker mode")
    preflight = payload.get("mode") == "preflight"
    profile_name = str(payload.get("profile", "development" if preflight else ""))
    execution = protocol.preflight if preflight else protocol.profile(profile_name)
    if device not in (
        ("cuda",) if preflight else execution.devices or (execution.device,)
    ):
        raise ValueError("Visual worker device violates the frozen execution profile")
    if (
        candidate.strategy == "hybrid"
        and candidate.scene_threshold
        != protocol.hybrid_threshold("development" if preflight else profile_name)
    ):
        raise ValueError("Hybrid worker threshold differs from the frozen selection")
    items = list(payload["items"])
    random.Random(protocol.meta.seed).shuffle(items)
    if not items or len({str(item["id"]) for item in items}) != len(items):
        raise ValueError("Visual worker requires unique videos")
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
            expected_profile=profile_name,
            expected_checksum=payload["frozen_asr_checksum"],
            expected_durations={
                str(item["id"]): float(item["duration_seconds"]) for item in items
            },
            expected_ffmpeg_version=payload["ffmpeg_version"],
            expected_canonical_checksums=payload["canonical_audio_checksums"],
        )
    image_profile = parse_document_profile(str(payload["image_candidate"]))
    image_engine = image_profile.runtime_engine
    if image_engine != str(payload["image_engine"]):
        raise ValueError("Visual worker image candidate and engine disagree")
    if device not in document_protocol.backend_devices[image_engine]:
        raise ValueError("Visual parser does not support this device")
    document_protocol.validate_candidate_factors(image_profile.factors)
    image_options = dict(payload["image_options"])
    expected = {
        **document_protocol.parser_options(image_engine),
        **image_profile.options,
    }
    if any(image_options.get(name) != value for name, value in expected.items()):
        raise ValueError("Visual image options differ from the document protocol")
    image_revision = str(payload["image_revision"])
    timings = [
        {
            "sample_id": str(item["id"]),
            "repetition": repetition,
            "executed": False,
            "success": None,
            "error": None,
            "latency_seconds": None,
            "duration_seconds": float(item["duration_seconds"]),
            "selected_frame_count": None,
            "output_fingerprint": None,
            "not_executed_reason": "lifecycle_interrupted",
        }
        for item in items
        for repetition in range(1, execution.repetitions + 1)
    ]
    commands = []
    outputs = {
        str(item["id"]): [
            {"predictions": None, "fingerprint": None, "frame_evidence": None}
            for _ in range(execution.repetitions)
        ]
        for item in items
    }
    monitor = ResourceMonitor(
        require_vram=device == "cuda",
        device=device,
        interval_seconds=protocol.preflight.telemetry_interval_seconds,
    )
    cold = before = after = extractor = execution_error = None
    initialized_options = {}
    phase = "initialize_frame"
    lifecycle_started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="video-visual-") as raw:
            temporary = Path(raw)
            with monitor:
                seed_deterministically(protocol.meta.seed)
                cold_frames, command = _extract_frames(
                    candidate,
                    Path(str(items[0]["source_path"])),
                    temporary / "cold",
                    frame_limit=1,
                )
                commands.append({"phase": "initialize", "command": command})
                request = _image_request(
                    cold_frames[0][0],
                    image_engine,
                    image_revision,
                    device,
                    image_options,
                )
                phase = "load"
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
                phase = "placement_before"
                before = (
                    inspect_model_placement(extractor, expected_device=device)
                    if device == "cuda"
                    else None
                )
                if before and before["status"] != "qualified":
                    raise RuntimeError(
                        "Visual CUDA weight placement violates the no-offload contract"
                    )
                phase = "warmup"
                for warmup in range(execution.warmups):
                    seed_deterministically(protocol.meta.seed)
                    _, command, _, _ = _process_video(
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
                    commands.append(
                        {
                            "phase": "warmup",
                            "repetition": warmup + 1,
                            "command": command,
                        }
                    )
                phase = "inference"
                for index, item in enumerate(items):
                    for repetition in range(1, execution.repetitions + 1):
                        timing = timings[index * execution.repetitions + repetition - 1]
                        seed_deterministically(protocol.meta.seed)
                        started = time.perf_counter()
                        predictions = fingerprint = error = count = command = None
                        frame_evidence = {}
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
                                frame_evidence,
                            )
                            synchronize(device, image_engine)
                        except Exception as exc:  # noqa: BLE001 - continue accounted request failures
                            error = _error_text(exc)
                            count = getattr(exc, "selected_frame_count", count)
                            command = getattr(exc, "command", command)
                            predictions = fingerprint = None
                        timing.update(
                            executed=True,
                            success=error is None,
                            error=error,
                            latency_seconds=time.perf_counter() - started,
                            selected_frame_count=count,
                            output_fingerprint=fingerprint,
                            not_executed_reason=None,
                        )
                        outputs[str(item["id"])][repetition - 1] = {
                            "predictions": predictions,
                            "fingerprint": fingerprint,
                            "frame_evidence": frame_evidence,
                        }
                        if command is not None:
                            commands.append(
                                {
                                    "sample_id": str(item["id"]),
                                    "repetition": repetition,
                                    "command": command,
                                }
                            )
                phase = "telemetry"
            phase = "placement_after"
            after = (
                inspect_model_placement(extractor, expected_device=device)
                if device == "cuda"
                else None
            )
            if after and after["status"] != "qualified":
                raise RuntimeError("Visual inference changed CUDA weight placement")
    except Exception as exc:  # noqa: BLE001 - retain completed and unexecuted observations
        before = getattr(exc, "placement_report", before)
        execution_error = {
            "phase": phase,
            "error": _error_text(exc),
            "lifecycle_elapsed_seconds": time.perf_counter() - lifecycle_started,
        }
        command = getattr(exc, "command", None)
        if command:
            commands.append({"phase": phase, "command": command})
    if extractor is not None:
        try:
            initialized_options = pipeline_parameters(extractor)
        except Exception as exc:  # noqa: BLE001 - metadata must not erase inference evidence
            error = _error_text(exc)
            if execution_error is None:
                execution_error = {"phase": "parameters", "error": error}
            else:
                execution_error["parameter_error"] = error
    try:
        resources = monitor.metrics()
    except Exception as exc:  # noqa: BLE001 - retain partial telemetry
        execution_error = execution_error or {
            "phase": "telemetry",
            "error": _error_text(exc),
        }
        resources = {"peak_process_tree_ram_mb": None, "peak_vram_mb": None}
    if device == "cpu":
        resources["peak_vram_mb"] = None
    elif resources["peak_vram_mb"] is None:
        execution_error = execution_error or {
            "phase": "telemetry",
            "error": "Visual CUDA telemetry is unavailable",
        }
    for timing in timings:
        if not timing["executed"]:
            timing["not_executed_reason"] = (
                execution_error["phase"] if execution_error else "missing_attempt"
            )
    evidence = {
        **resources,
        "cold_visual_pipeline_load_seconds": cold,
        "execution_error": execution_error,
        "timings": timings,
        "outputs": {
            "attempts": outputs,
            "inputs": items,
            "protocol": payload["protocol"],
            "document_protocol": payload["document_protocol"],
        },
        "ffmpeg_commands": commands,
        "resource_samples": monitor.samples(),
        "gpu_identity": monitor.gpu_identity,
        "placement_before": before,
        "placement_after": after,
        "resource_measurement_scope": "partial_lifecycle"
        if execution_error
        else "complete_lifecycle",
    }
    if preflight:
        placement = after if after and after["status"] != "qualified" else before
        evidence.update(
            placement=placement,
            stress_sample_ids=[str(item["id"]) for item in items],
            vram_measurement_method=monitor.vram_measurement_method,
            initialized_pipeline_options=initialized_options,
        )
        if placement and placement["status"] != "qualified":
            return evidence
        failed = next((row for row in timings if row["success"] is False), None)
        if failed:
            raise WorkerExecutionError(failed["error"], evidence)
        if execution_error:
            error_type = (
                WorkerMeasurementError
                if execution_error["phase"] == "telemetry"
                else WorkerExecutionError
            )
            raise error_type(execution_error["error"], evidence)
        return evidence
    rows = []
    try:
        for item in items:
            attempts = outputs[str(item["id"])]
            timing = [
                value for value in timings if value["sample_id"] == str(item["id"])
            ]
            first = attempts[0]
            row = score_video(item, first["predictions"], protocol.occurrence_matching)
            complete = all(value["executed"] for value in timing)
            rates = (
                attempt_rates([attempt["fingerprint"] for attempt in attempts])
                if complete
                else {"repeatability_success_rate": None, "attempt_failure_rate": None}
            )
            row.update(
                {
                    **rates,
                    "source_group_id": source_id(item),
                    "first_attempt_executed": timing[0]["executed"],
                    "first_attempt_success": timing[0]["success"],
                    "quality_latency_seconds": timing[0]["latency_seconds"]
                    if timing[0]["success"]
                    else None,
                    "selected_frame_count": timing[0]["selected_frame_count"],
                    "scheduled_attempt_count": len(timing),
                    "executed_attempt_count": sum(
                        value["executed"] for value in timing
                    ),
                    "failed_attempt_count": sum(
                        value["success"] is False for value in timing
                    ),
                    "unexecuted_attempt_count": sum(
                        not value["executed"] for value in timing
                    ),
                    "predictions": first["predictions"],
                    "_timings": timing,
                }
            )
            row["metric_statuses"] = quality_statuses(row)
            rows.append(row)
        operational = _operational_statistics(rows)
        operational.update(
            {
                "cold_visual_pipeline_load_seconds": cold,
                "peak_visual_process_tree_ram_mb": resources[
                    "peak_process_tree_ram_mb"
                ],
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
        if protocol.minimum_latency_ci_sources != protocol.minimum_ci_sources:
            latency_intervals = bootstrap_sources(
                rows,
                _operational_statistics,
                resamples=execution.bootstrap_resamples,
                seed=protocol.meta.seed,
                confidence=protocol.confidence_level,
                minimum_sources=protocol.minimum_latency_ci_sources,
            )
            for name in (
                "p50_warm_visual_latency_seconds",
                "p95_warm_visual_latency_seconds",
            ):
                if name in latency_intervals:
                    intervals[name] = latency_intervals[name]
        metrics = aggregate_quality(rows)
        metrics.update(quality_support_counts(rows))
    except Exception as exc:
        raise WorkerExecutionError(
            _error_text(exc), {**evidence, "samples": rows}
        ) from exc
    reports = aggregate_quality_statuses(rows)
    for name, report in reports.items():
        report.update(
            value=metrics[name],
            support={
                key.removeprefix(name + "."): value
                for key, value in metrics.items()
                if key.startswith(name + ".")
            },
        )
    successful = [
        row for row in rows if any(timing["success"] for timing in row["_timings"])
    ]
    selected = [row for row in rows if row["selected_frame_count"] is not None]
    for name in OPERATIONAL_DIRECTIONS:
        contributors = (
            selected if name == "mean_selected_frames_per_video" else successful
        )
        lifecycle_metric = name in {
            "cold_visual_pipeline_load_seconds",
            "peak_visual_process_tree_ram_mb",
            "peak_visual_vram_mb",
        }
        support = {
            "scheduled_count": 1 if lifecycle_metric else len(rows),
            "contributing_count": int(operational[name] is not None)
            if lifecycle_metric
            else len(contributors),
            "independent_source_count": None
            if lifecycle_metric
            else len({row["source_group_id"] for row in rows}),
            "contributing_source_count": None
            if lifecycle_metric
            else len({row["source_group_id"] for row in contributors}),
        }
        support["sample_count"] = (
            int(operational[name] is not None)
            if lifecycle_metric
            else len({row["source_group_id"] for row in contributors})
            if operational[name] is not None
            else 0
        )
        if not lifecycle_metric:
            metrics[f"{name}.sample_count"] = float(support["sample_count"])
        reports[name] = {
            "value": operational[name],
            "support": support,
            "status": "inapplicable"
            if name == "peak_visual_vram_mb" and device == "cpu"
            else "scored"
            if operational[name] is not None
            else "unavailable",
            "reason": "cpu_profile"
            if name == "peak_visual_vram_mb" and device == "cpu"
            else None
            if operational[name] is not None
            else "no_completed_measurement",
        }
    for report in reports.values():
        report.update(
            comparison_status="incomplete" if execution_error else "complete",
            comparison_reason="execution_error" if execution_error else None,
        )
    return {
        **evidence,
        "samples": [
            {key: value for key, value in row.items() if key != "_timings"}
            for row in rows
        ],
        "metrics": metrics,
        "metric_statuses": reports,
        "operational": operational,
        "intervals": intervals,
        "parameters": {
            "initialized_pipeline_options": initialized_options,
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
                (
                    "docling",
                    "paddleocr",
                    "paddlepaddle-gpu",
                    "onnxruntime-gpu",
                    "rapidocr",
                    "torch",
                    "transformers",
                )
            ),
        },
    }


def _operational_statistics(rows):
    completed = [value for row in rows for value in row["_timings"] if value["success"]]
    valid_timing = all(
        type(value.get(name)) in (int, float)
        and np.isfinite(value[name])
        and (value[name] > 0 if name == "duration_seconds" else value[name] >= 0)
        for value in completed
        for name in ("duration_seconds", "latency_seconds")
    )
    if not valid_timing:
        completed = []
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
                    if value["success"] and valid_timing
                ]
            )
        )
        for row in rows
        if valid_timing and any(value["success"] for value in row["_timings"])
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
    evidence=None,
):
    frames, command = _extract_frames(
        candidate, Path(str(item["source_path"])), directory
    )
    if evidence is None:
        evidence = {}
    evidence.update(
        selected_frame_timestamps=[timestamp for _, timestamp in frames], frames=[]
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
    try:
        for index, (frame, timestamp) in enumerate(frames):
            request = _image_request(
                frame, image_engine, image_revision, device, image_options
            )
            record = {
                "frame_index": index,
                "timestamp": timestamp,
                "input_sha256": request.checksum,
                "document": None,
            }
            evidence["frames"].append(record)
            record["document"] = extractor.extract(request, SourceKind.IMAGE).to_dict()
            document = validate_document_payload(record["document"])
            if any(not warning.recoverable for warning in document.warnings):
                raise ValueError(
                    "Visual parser reported a nonrecoverable extraction failure"
                )
            if (
                document.source_kind != SourceKind.IMAGE
                or document.source_checksum != request.checksum
            ):
                raise ValueError("Visual output differs from its frame request")
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
    except Exception as exc:
        raise VisualAttemptFailure(_error_text(exc), len(frames), command) from exc
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


def _extract_frames(candidate, source: Path, directory: Path, *, frame_limit=None):
    directory.mkdir(parents=True, exist_ok=True)
    pattern = directory / "frame-%05d.png"
    command = frame_command(candidate, str(source), str(pattern))
    if frame_limit is not None:
        command[-1:-1] = ["-frames:v", str(frame_limit)]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise VisualAttemptFailure(
            f"FFmpeg selection failed: {exc.stderr}", None, command
        ) from exc
    frames = sorted(directory.glob("frame-*.png"))
    try:
        timestamps = [
            float(value)
            for value in re.findall(r"showinfo.*?pts_time:(\S+)", completed.stderr)
        ]
    except ValueError as exc:
        raise VisualAttemptFailure(
            "Invalid FFmpeg frame timestamps", len(frames), command
        ) from exc
    # FFmpeg may log the next selected frame before the one-frame output limit.
    if frame_limit is not None:
        timestamps = timestamps[:frame_limit]
    if not frames or len(frames) != len(timestamps):
        raise VisualAttemptFailure(
            "FFmpeg frame timestamps did not match extracted frames",
            len(frames),
            command,
        )
    if (
        any(not np.isfinite(value) for value in timestamps)
        or any(right <= left for left, right in pairwise(timestamps))
        or abs(timestamps[0]) > 1e-6
    ):
        raise VisualAttemptFailure(
            "Video selector timestamps are invalid or omit frame zero",
            len(frames),
            command,
        )
    return list(zip(frames, timestamps, strict=True)), command


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute))
