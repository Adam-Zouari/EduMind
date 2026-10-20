"""Video comparisons: one audio-preparation child followed by visual-only children."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

from edumind.common.artifacts import sha256_file
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import (
    LIFECYCLE_PROFILES,
    default_decision_path,
    execution_devices,
    without_option_values,
)
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.data_validation import (
    manifest_path as default_manifest_path,
)
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.preflight import eligible_candidates
from experiments.benchmarks.common.preflight_reports import resolve_preflight_report
from experiments.benchmarks.common.process import WorkerExecutionError, run_json_worker
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.extraction.audio.adapters import profiles as audio_profiles
from experiments.benchmarks.extraction.audio.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_AUDIO_PROTOCOL_PATH,
)
from experiments.benchmarks.extraction.audio.protocol import AudioProtocol
from experiments.benchmarks.extraction.audio.protocol import (
    load_protocol as load_audio_protocol,
)
from experiments.benchmarks.extraction.document.profiles import (
    lock_paths,
    parse_document_profile,
)
from experiments.benchmarks.extraction.document.protocol import (
    DEFAULT_PROTOCOL_PATH as DEFAULT_DOCUMENT_PROTOCOL_PATH,
)
from experiments.benchmarks.extraction.document.protocol import DocumentProtocol
from experiments.benchmarks.extraction.document.protocol import (
    load_protocol as load_document_protocol,
)
from experiments.benchmarks.extraction.document.runner import (
    validate_prepared_components,
)
from experiments.benchmarks.extraction.evidence import (
    public_asr_evidence,
    retain_prediction,
)
from experiments.benchmarks.extraction.media import ffmpeg_version
from experiments.benchmarks.extraction.scoring import attempt_rates
from experiments.benchmarks.extraction.video.candidates import (
    all_candidates,
    scene_candidates,
)
from experiments.benchmarks.extraction.video.frozen_asr import (
    create_frozen_asr_artifact,
    load_frozen_asr_artifact,
    model_identity,
)
from experiments.benchmarks.extraction.video.frozen_asr_worker import (
    METRIC_DIRECTIONS as ASR_DIRECTIONS,
)
from experiments.benchmarks.extraction.video.metrics import (
    METRIC_DIRECTIONS,
    QUALITY_DIRECTIONS,
    aggregate_quality,
)
from experiments.benchmarks.extraction.video.preflight import (
    qualification_inputs,
    run_video_preflight,
    validate_input_envelope,
    video_qualification_identity,
)
from experiments.benchmarks.extraction.video.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
    protocol_from_worker,
)
from experiments.benchmarks.extraction.video.validation import verified_inputs
from experiments.benchmarks.preparation.models import load_selected_model_lock

PROFILE_STAGE = {
    "smoke": "video-smoke",
    "development": "video-development",
    "validation": "video-validation",
    "locked": "video-locked-test",
}


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark frozen-ASR inputs and visual video policies"
    )
    parser.add_argument("--profile", choices=LIFECYCLE_PROFILES, default="smoke")
    parser.add_argument("--phase", choices=("all", "scene-selection"), default="all")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--audio-protocol", type=Path, default=DEFAULT_AUDIO_PROTOCOL_PATH
    )
    parser.add_argument(
        "--document-protocol", type=Path, default=DEFAULT_DOCUMENT_PROTOCOL_PATH
    )
    parser.add_argument(
        "--frozen-asr",
        type=Path,
        help="Explicitly reuse an exactly matching existing artifact",
    )
    parser.add_argument("--shortlist", type=Path)
    parser.add_argument("--document-selection", type=Path)
    parser.add_argument("--audio-selection", type=Path)
    parser.add_argument("--image-candidate", help="Smoke-only image parser override")
    parser.add_argument("--audio-candidate", help="Smoke-only ASR override")
    parser.add_argument("--device", choices=("cpu", "cuda", "both"))
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--preflight-run-id")
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()
    if arguments.phase == "scene-selection" and arguments.profile != "development":
        parser.error("Preliminary scene selection is a development-only comparison")
    if arguments.profile == "smoke" and (
        arguments.preflight_report or arguments.preflight_run_id
    ):
        parser.error("Smoke does not consume hardware qualification")
    if arguments.profile == "preflight" and any(
        (
            arguments.frozen_asr,
            arguments.shortlist,
            arguments.image_candidate,
            arguments.audio_candidate,
            arguments.preflight_report,
            arguments.preflight_run_id,
        )
    ):
        parser.error(
            "Preflight qualifies the declared roster; selection/reuse overrides are not evaluation inputs"
        )
    protocol, audio_protocol, document_protocol = (
        load_protocol(arguments.protocol),
        load_audio_protocol(arguments.audio_protocol),
        load_document_protocol(arguments.document_protocol),
    )
    if arguments.profile == "smoke" and arguments.device in {None, "both"}:
        return _run_smoke_devices(arguments, protocol.profile("smoke").devices)
    profile = "development" if arguments.profile == "preflight" else arguments.profile
    device = execution_devices(
        arguments.profile,
        arguments.device,
        smoke_devices=protocol.profile("smoke").devices,
        authoritative_device=protocol.profile(profile).device,
    )[0]
    if profile != "smoke":
        if arguments.image_candidate or arguments.audio_candidate:
            parser.error("Direct model overrides are smoke-only")
        arguments.audio_selection = arguments.audio_selection or default_decision_path(
            "audio", "locked"
        )
        arguments.document_selection = (
            arguments.document_selection
            or default_decision_path("document-image", "locked")
        )
        if profile in {"validation", "locked"}:
            arguments.shortlist = arguments.shortlist or default_decision_path(
                "video", profile
            )
    manifest_path = (
        arguments.manifest or default_manifest_path("video", profile)
    ).resolve()
    validation_path, validation = verified_inputs(manifest_path, profile, protocol)
    manifest = load_manifest(manifest_path, validate_annotations=False)
    items = [
        {**item, "source_path": str((PROJECT_ROOT / item["source_path"]).resolve())}
        for item in manifest.samples
        if item.get("kind") == "video"
    ]
    audio_candidate, audio_decision = _selected_audio(arguments, audio_protocol)
    image_candidate, image_decision = _selected_image(arguments, document_protocol)
    if arguments.profile == "preflight":
        result = run_video_preflight(
            items,
            manifest=manifest,
            manifest_path=manifest_path,
            protocol=protocol,
            audio_protocol=audio_protocol,
            document_protocol=document_protocol,
            audio_candidate=audio_candidate,
            audio_decision=audio_decision,
            image_candidate=image_candidate,
            image_decision=image_decision,
            data_validation_report=validation_path,
            no_mlflow=arguments.no_mlflow,
        )
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "ready_for_development": result.ready_for_development,
                    "artifacts": str(result.artifact_directory),
                },
                indent=2,
            )
        )
        return 0 if result.ready_for_development else 2
    qualification = qualification_path = None
    if profile != "smoke":
        qualification_path, qualification = _qualification(
            profile,
            manifest_path,
            items,
            validation,
            protocol,
            audio_protocol,
            document_protocol,
            audio_candidate,
            image_candidate,
            explicit=arguments.preflight_report,
            run_id=arguments.preflight_run_id,
        )
    candidates, decisions = _candidates(
        arguments,
        protocol,
        expected_protocols={
            "video": protocol.meta.checksum,
            "audio": audio_protocol.meta.checksum,
            "document": document_protocol.meta.checksum,
        },
        expected_qualification=qualification.get("qualification_fingerprint")
        if qualification
        else None,
    )
    if qualification is not None:
        candidates = tuple(
            value.removeprefix("visual|")
            for value in eligible_candidates(
                tuple(f"visual|{candidate}" for candidate in candidates),
                qualification,
                profile=profile,
                label="Video visual",
            )
        )
    result = run_visual_benchmark(
        profile,
        candidates,
        items=items,
        manifest_path=manifest_path,
        manifest_name=manifest.name,
        manifest_checksum=manifest.fingerprint,
        protocol=protocol,
        audio_protocol=audio_protocol,
        document_protocol=document_protocol,
        frozen_asr_path=arguments.frozen_asr,
        image_candidate=image_candidate,
        image_decision=image_decision,
        audio_candidate=audio_candidate,
        audio_decision=audio_decision,
        decision_files=decisions,
        device=device,
        ffmpeg_version=ffmpeg_version(),
        no_mlflow=arguments.no_mlflow,
        preflight_report=qualification_path,
        preflight=qualification,
        data_validation_report=validation_path,
        scene_selection=arguments.phase == "scene-selection",
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "complete": result.complete,
                "artifacts": str(result.artifact_directory),
            },
            indent=2,
        )
    )
    return 0 if result.complete else 2


def run_visual_benchmark(
    profile,
    candidates,
    *,
    items,
    manifest_path,
    manifest_name,
    manifest_checksum,
    protocol,
    audio_protocol,
    document_protocol,
    frozen_asr_path,
    image_candidate,
    image_decision,
    audio_candidate,
    audio_decision,
    decision_files,
    device,
    ffmpeg_version,
    no_mlflow,
    preflight_report=None,
    preflight=None,
    data_validation_report=None,
    scene_selection=False,
):
    report_path, validation = verified_inputs(manifest_path, profile, protocol)
    if (
        data_validation_report is not None
        and Path(data_validation_report).resolve() != report_path.resolve()
    ):
        raise ValueError(
            "Video data-validation report differs from the verified report"
        )
    data_validation_report = report_path
    canonical_checksums = {
        sample_id: metadata["canonical_sha256"]
        for sample_id, metadata in validation["metadata"]["samples"].items()
    }
    current_manifest = load_manifest(manifest_path, validate_annotations=False)
    expected_items = [
        {**item, "source_path": str((PROJECT_ROOT / item["source_path"]).resolve())}
        for item in current_manifest.samples
        if item.get("kind") == "video"
    ]
    if (
        manifest_checksum != current_manifest.fingerprint
        or manifest_name != current_manifest.name
        or sorted(items, key=lambda row: str(row["id"]))
        != sorted(expected_items, key=lambda row: str(row["id"]))
    ):
        raise ValueError("Video execution inputs differ from the validated manifest")
    execution = protocol.profile(profile)
    if profile != "smoke":
        selection = SimpleNamespace(
            profile=profile,
            audio_selection=audio_decision,
            document_selection=image_decision,
        )
        if (
            _selected_audio(selection, audio_protocol)[0] != audio_candidate
            or _selected_image(selection, document_protocol)[0] != image_candidate
        ):
            raise ValueError(
                "Video models differ from their reviewed selection decisions"
            )
        if preflight is None or preflight_report is None:
            raise ValueError(
                "Authoritative video execution requires matching preflight evidence"
            )
        _, verified_qualification = _qualification(
            profile,
            manifest_path,
            items,
            validation,
            protocol,
            audio_protocol,
            document_protocol,
            audio_candidate,
            image_candidate,
            explicit=preflight_report,
        )
        if verified_qualification != preflight:
            raise ValueError("Video preflight object differs from its verified report")
        eligible_candidates(
            tuple(f"visual|{candidate}" for candidate in candidates),
            preflight,
            profile="validation",
            label="Video visual",
        )
        if profile in {"validation", "locked"}:
            selected, _ = _candidates(
                SimpleNamespace(
                    profile=profile,
                    phase="all",
                    shortlist=decision_files.get("shortlist"),
                ),
                protocol,
                expected_protocols={
                    "video": protocol.meta.checksum,
                    "audio": audio_protocol.meta.checksum,
                    "document": document_protocol.meta.checksum,
                },
                expected_qualification=preflight["qualification_fingerprint"],
            )
            if tuple(candidates) != selected:
                raise ValueError("Video comparison differs from its reviewed shortlist")
    if scene_selection and profile != "development":
        raise ValueError("Scene selection is development-only")
    declared = (
        scene_candidates(protocol)
        if scene_selection
        else all_candidates(protocol, protocol.hybrid_threshold(profile))
    )
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or not set(candidates) <= set(declared)
    ):
        raise ValueError("Video comparison requires unique declared candidates")
    if profile == "locked" and len(candidates) != 1:
        raise ValueError("Locked video comparison requires exactly one candidate")
    if (
        protocol.window_length_seconds
        > audio_protocol.audio["maximum_duration_seconds"]
    ):
        raise ValueError("Video ASR windows exceed the selected audio protocol limit")
    document_profile = parse_document_profile(image_candidate)
    document_protocol.validate_candidate_factors(document_profile.factors)
    engine = document_profile.runtime_engine
    if device not in document_protocol.backend_devices[engine]:
        raise ValueError(f"{engine} does not support {device}")
    if device not in (execution.devices or (execution.device,)):
        raise ValueError("Video device violates the execution profile")
    image_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=(document_profile.lock_candidate,),
    )
    entry = image_lock[document_profile.lock_candidate]
    validate_prepared_components(image_candidate, entry, document_protocol)
    image_options = {
        **document_protocol.parser_options(engine),
        **document_profile.options,
        **lock_paths(entry),
    }
    _, expected_identity = model_identity(
        audio_candidate, audio_decision, audio_protocol
    )
    stage = (
        "video-development-scene-selection"
        if scene_selection
        else PROFILE_STAGE[profile]
    )
    preparation = f"frozen-asr — {audio_candidate}"
    ordered = list(candidates)
    random.Random(protocol.meta.seed).shuffle(ordered)
    plan = BenchmarkPlan(
        "extraction",
        stage,
        profile,
        manifest_name,
        (preparation, *ordered),
        seed=protocol.meta.seed,
        repetitions=execution.repetitions,
        warmups=execution.warmups,
        bootstrap_resamples=execution.bootstrap_resamples,
        settings={
            "device": device,
            "audio_candidate": audio_candidate,
            "image_candidate": image_candidate,
            "ffmpeg_version": ffmpeg_version,
            "selected_scene_source_run_id": protocol.selected_scene_source_run_id,
            "preflight_run_id": preflight.get("mlflow_run_id") if preflight else None,
            "preflight_fingerprint": preflight.get("qualification_fingerprint")
            if preflight
            else None,
            "hardware_exclusions": preflight.get("excluded_candidates", [])
            if preflight
            else [],
        },
    )
    frozen = {}

    def evaluate(candidate, context):
        if candidate == preparation:
            # Output directory belongs to this child; default execution never selects a stale/latest file.
            path = (
                frozen_asr_path
                or Path(context["artifact_directory"]) / "frozen_asr.json"
            )
            if frozen_asr_path is None:
                _preserve_worker_failure(
                    create_frozen_asr_artifact,
                    path,
                    items=[
                        {
                            **item,
                            "canonical_sha256": canonical_checksums[str(item["id"])],
                        }
                        for item in items
                    ],
                    manifest_checksum=manifest_checksum,
                    protocol=protocol,
                    audio_protocol=audio_protocol,
                    audio_candidate=audio_candidate,
                    audio_decision_path=audio_decision,
                    device=device,
                    ffmpeg_version=ffmpeg_version,
                    profile_name=profile,
                    producer_run_id=None if no_mlflow else context["mlflow_run_id"],
                )
            artifact, checksum = _preserve_worker_failure(
                load_frozen_asr_artifact,
                path,
                manifest_checksum=manifest_checksum,
                protocol_checksum=protocol.meta.checksum,
                audio_protocol_checksum=audio_protocol.meta.checksum,
                timestamp_tolerance_seconds=audio_protocol.timestamp_tolerance_seconds,
                sample_ids=[str(item["id"]) for item in items],
                expected_identity=expected_identity,
                expected_device=device,
                expected_profile=profile,
                expected_durations={
                    str(item["id"]): float(item["duration_seconds"]) for item in items
                },
                expected_ffmpeg_version=ffmpeg_version,
                expected_canonical_checksums=canonical_checksums,
            )
            frozen.update(path=path, checksum=checksum, artifact=artifact)
            rows = artifact["videos"]
            samples = [
                SampleResult(
                    str(row["sample_id"]),
                    row.get("metric_values", {}),
                    row["latency_seconds"] if row["success"] else None,
                    {
                        "success": row["success"],
                        "reason": row["error"],
                        "window_count": len(row["windows"]),
                        "metric_statuses": row.get("metric_statuses", {}),
                    },
                )
                for row in rows
            ]
            return (
                samples,
                {},
                artifact["metrics"],
                {
                    **artifact["parameters"],
                    "candidate": preparation,
                    "run_type": "input_preparation",
                    "input_preparation_reused": frozen_asr_path is not None,
                    **expected_identity,
                    "frozen_asr_run_id": artifact["run_id"],
                    "frozen_asr_checksum": checksum,
                    "prediction_evidence": {
                        "path": str(Path(path).resolve()),
                        "sha256": checksum,
                        "uploaded": False,
                    },
                },
                artifact["intervals"],
                {
                    "outputs": {
                        "prediction_evidence": {
                            "path": str(Path(path).resolve()),
                            "sha256": checksum,
                            "uploaded": False,
                        }
                    },
                    "samples": _flat_rows(
                        rows, ("windows", "segments", "transcript", "nonspeech_units")
                    ),
                    "windows": [
                        {
                            "sample_id": row["sample_id"],
                            **{
                                key: value
                                for key, value in window.items()
                                if key
                                not in {"segments", "transcript", "raw_prediction"}
                            },
                        }
                        for row in rows
                        for window in row["windows"]
                    ],
                    "resource_samples": artifact["resource_samples"],
                    "metric_statuses": artifact["metric_statuses"],
                    "gpu_identity": artifact["gpu_identity"],
                    "ffmpeg_commands": artifact["ffmpeg_commands"],
                },
            )
        if not frozen:
            raise RuntimeError(
                "Frozen-ASR preparation did not deliver a validated artifact"
            )
        output = _preserve_worker_failure(
            _run_visual_worker,
            candidate,
            items,
            image_engine=engine,
            image_candidate=image_candidate,
            image_revision=str(entry["revision"]),
            image_options=image_options,
            device=device,
            profile=profile,
            protocol=protocol.meta.worker_payload(),
            document_protocol=document_protocol.meta.worker_payload(),
            audio_protocol=audio_protocol.meta.worker_payload(),
            frozen_asr_path=str(frozen["path"]),
            frozen_asr_checksum=frozen["checksum"],
            manifest_checksum=manifest_checksum,
            frozen_asr_identity=expected_identity,
            ffmpeg_version=ffmpeg_version,
            canonical_audio_checksums=canonical_checksums,
        )
        try:
            _validate_visual_output(output, items, execution.repetitions, device)
        except (KeyError, TypeError, ValueError) as exc:
            raise CandidateExecutionError(
                f"Invalid visual worker result: {exc}",
                artifacts={"worker_output": output},
            ) from exc
        rows = output["samples"]
        samples = [
            SampleResult(
                str(row["sample_id"]),
                {name: row.get(name) for name in QUALITY_DIRECTIONS},
                row["quality_latency_seconds"],
                {
                    key: value
                    for key, value in row.items()
                    if key not in QUALITY_DIRECTIONS and key != "predictions"
                },
            )
            for row in rows
        ]
        artifacts = {
            "samples": _flat_rows(rows, ("predictions",)),
            **{
                name: output[name]
                for name in (
                    "timings",
                    "outputs",
                    "ffmpeg_commands",
                    "resource_samples",
                    "gpu_identity",
                    "metric_statuses",
                )
            },
            "placement": {
                "before": output.get("placement_before"),
                "after": output.get("placement_after"),
            },
        }
        if output.get("execution_error"):
            raise CandidateExecutionError(
                output["execution_error"]["error"],
                samples=samples,
                metrics=output["metrics"],
                operational=output["operational"],
                parameters=output["parameters"],
                intervals=output["intervals"],
                artifacts={**artifacts, "execution_error": output["execution_error"]},
            )
        return (
            samples,
            output["operational"],
            output["metrics"],
            {
                **output["parameters"],
                "image_model_identity": entry,
                "frozen_asr_run_id": frozen["artifact"]["run_id"],
                "frozen_asr_checksum": frozen["checksum"],
            },
            output["intervals"],
            artifacts,
        )

    directions = {**METRIC_DIRECTIONS, **ASR_DIRECTIONS}
    contracts = {
        preparation: tuple(ASR_DIRECTIONS),
        **{candidate: tuple(METRIC_DIRECTIONS) for candidate in candidates},
    }
    decisions = {
        **decision_files,
        "audio": audio_decision,
        **({"document": image_decision} if image_decision else {}),
    }
    return run_benchmark(
        plan,
        evaluate,
        dataset_checksum=manifest_checksum,
        directions=directions,
        primary_metric=(
            "visual_content_f1",
            "mean_visual_first_detection_delay_seconds",
            "timed_visual_occurrence_coverage",
        ),
        required_metrics=tuple(directions),
        nullable_metrics=tuple(directions),
        candidate_metric_contracts=contracts,
        candidate_run_types={preparation: "input_preparation"},
        candidate_run_names={
            candidate: candidate.removeprefix("video-") for candidate in candidates
        },
        paired_metrics=(),
        paired_comparisons=False,
        operational_prefix="",
        monitor_resources=False,
        evaluator_receives_context=True,
        shuffle_candidates=False,
        candidate_artifact_name="candidate.json",
        revisions={
            "image_parser": str(entry.get("selection_revision", entry["revision"])),
            "ffmpeg": ffmpeg_version,
        },
        decision_files=decisions,
        input_artifacts={
            "manifest": manifest_path,
            **({"preflight_report": preflight_report} if preflight_report else {}),
            **(
                {"data_validation": data_validation_report}
                if data_validation_report
                else {}
            ),
        },
        protocols={
            "video": protocol.meta,
            "audio": audio_protocol.meta,
            "document": document_protocol.meta,
        },
        no_mlflow=no_mlflow,
        run_name_prefix=f"smoke-{device}"
        if profile == "smoke"
        else "development-scene-selection"
        if scene_selection
        else profile,
    )


def _flat_rows(rows, excluded):
    return [
        {key: value for key, value in row.items() if key not in excluded}
        for row in rows
    ]


def _qualification(
    profile,
    manifest_path,
    items,
    validation,
    protocol,
    audio_protocol,
    document_protocol,
    audio_candidate,
    image_candidate,
    *,
    explicit=None,
    run_id=None,
):
    development_path = (
        manifest_path
        if profile == "development"
        else default_manifest_path("video", "development")
    )
    _, development_validation = verified_inputs(
        development_path, "development", protocol
    )
    development = load_manifest(development_path, validate_annotations=False)
    stress = qualification_inputs(development, development_validation)
    declared, fingerprint, _ = video_qualification_identity(
        protocol,
        audio_protocol,
        document_protocol,
        audio_candidate,
        image_candidate,
        development,
        stress,
        development_validation,
    )
    path, qualification = resolve_preflight_report(
        benchmark="video",
        fingerprint=fingerprint,
        candidates=declared,
        explicit=explicit,
        run_id=run_id,
    )
    eligible_candidates(
        (f"frozen-asr|{audio_candidate}",),
        qualification,
        profile="validation",
        label="Frozen ASR",
    )
    validate_input_envelope(items, validation, qualification)
    return path, qualification


def _preserve_worker_failure(function, *args, **kwargs):
    """Keep request tables visible when the subprocess cannot deliver a result."""
    try:
        return function(*args, **kwargs)
    except WorkerExecutionError as exc:
        evidence = exc.resource_evidence
        body = evidence.get("worker_result", {})
        worker = (
            body.get("resource_evidence", body)
            if isinstance(body, Mapping) and body
            else (
                evidence
                if "worker_result" not in evidence
                and any(
                    name in evidence
                    for name in (
                        "artifact_type",
                        "videos",
                        "timings",
                        "cold_model_load_seconds",
                        "cold_visual_pipeline_load_seconds",
                        "operational",
                    )
                )
                else {}
            )
        )
        worker = worker if isinstance(worker, Mapping) else {}
        if isinstance(body, Mapping):
            evidence = {**evidence, **body, **worker}
        exc.artifacts.update(
            {
                name: evidence[name]
                for name in (
                    "samples",
                    "videos",
                    "timings",
                    "outputs",
                    "resource_samples",
                    "ffmpeg_commands",
                    "execution_error",
                )
                if name in evidence
            }
        )
        if "supervisor_resource_evidence" in exc.resource_evidence:
            exc.artifacts["supervisor_resource_evidence"] = exc.resource_evidence[
                "supervisor_resource_evidence"
            ]
        elif not worker:
            exc.artifacts["supervisor_resource_evidence"] = {
                name: evidence[name]
                for name in (
                    "peak_process_tree_ram_mb",
                    "peak_vram_mb",
                    "resource_samples",
                    "vram_measurement_method",
                )
                if name in evidence
            }
        exc.parameters.update(evidence.get("parameters", {}))
        if function in {create_frozen_asr_artifact, load_frozen_asr_artifact}:
            # The canonical child ID is already logged. The runtime alias remains
            # in raw preparation evidence, not under MLflow's immutable key.
            exc.parameters.pop("candidate", None)
            if isinstance(evidence.get("metrics"), Mapping):
                exc.metrics.update(
                    {
                        name: value
                        for name, value in evidence["metrics"].items()
                        if value is None
                        or (type(value) in (int, float) and math.isfinite(value))
                    }
                )
                exc.intervals.update(evidence.get("intervals", {}))
                exc.artifacts["metric_statuses"] = {
                    name: {
                        **report,
                        "comparison_status": "incomplete",
                        "comparison_reason": "preparation_failed",
                    }
                    for name, report in evidence.get("metric_statuses", {}).items()
                }
            for name in (
                "cold_model_load_seconds",
                "peak_process_tree_ram_mb",
                "peak_vram_mb",
            ):
                value = worker.get("metrics", {}).get(name, worker.get(name))
                exc.operational[name] = (
                    value
                    if value is None
                    or (type(value) in (int, float) and math.isfinite(value))
                    else None
                )
        if (
            function is _run_visual_worker
            and isinstance(body, dict)
            and {
                "samples",
                "timings",
                "operational",
                "metrics",
                "outputs",
                "metric_statuses",
            }
            <= body.keys()
        ):
            try:
                execution = protocol_from_worker(kwargs["protocol"]).profile(
                    kwargs["profile"]
                )
                _validate_visual_output(
                    body, args[1], execution.repetitions, kwargs["device"]
                )
            except (KeyError, TypeError, ValueError) as validation_error:
                exc.artifacts["output_validation_error"] = {
                    "error": str(validation_error)
                }
            else:
                exc.metrics.update(body["metrics"])
                exc.intervals.update(body.get("intervals", {}))
                exc.operational.update(body["operational"])
                exc.artifacts["metric_statuses"] = {
                    name: {
                        **report,
                        "comparison_status": "incomplete",
                        "comparison_reason": "worker_measurement_failed",
                    }
                    for name, report in body["metric_statuses"].items()
                }
        if function is _run_visual_worker:
            measured = worker.get("operational", {})
            exc.operational = {
                **exc.operational,
                "peak_visual_process_tree_ram_mb": measured.get(
                    "peak_visual_process_tree_ram_mb",
                    worker.get("peak_process_tree_ram_mb"),
                ),
                "peak_visual_vram_mb": measured.get(
                    "peak_visual_vram_mb", worker.get("peak_vram_mb")
                ),
            }
        if "cold_visual_pipeline_load_seconds" in worker:
            exc.operational["cold_visual_pipeline_load_seconds"] = worker[
                "cold_visual_pipeline_load_seconds"
            ]
        if function in {create_frozen_asr_artifact, load_frozen_asr_artifact}:
            # Common persistence uploads every artifact, including worker_error's
            # nested transport envelope. Keep bodies only in a durable local file.
            if args:
                path = Path(args[0])
                reference = (
                    {
                        "path": str(path.resolve()),
                        "sha256": sha256_file(path),
                        "uploaded": False,
                    }
                    if function is load_frozen_asr_artifact and path.is_file()
                    else retain_prediction(
                        path.with_suffix(".failure.json"), exc.resource_evidence
                    )
                )
                exc.parameters["prediction_evidence"] = reference
            exc.artifacts = public_asr_evidence(exc.artifacts)
        raise


def _validate_visual_output(output, items, repetitions, device):
    """Reject partial or contradictory transport results before publishing them."""
    rows, timings = output["samples"], output["timings"]
    ids = {str(item["id"]) for item in items}
    durations = {str(item["id"]): float(item["duration_seconds"]) for item in items}
    if len(rows) != len(ids) or {row["sample_id"] for row in rows} != ids:
        raise ValueError("Visual worker sample inventory differs from the manifest")
    expected = {
        (sample_id, repetition)
        for sample_id in ids
        for repetition in range(1, repetitions + 1)
    }
    if (
        len(timings) != len(expected)
        or {(row["sample_id"], row["repetition"]) for row in timings} != expected
    ):
        raise ValueError("Visual worker attempt inventory is incomplete")
    if not set(METRIC_DIRECTIONS) <= output["metric_statuses"].keys():
        raise ValueError("Visual worker lacks metric status evidence")
    for mapping in (output["metrics"], output["operational"]):
        if any(
            value is not None
            and (type(value) not in (int, float) or not math.isfinite(value))
            for value in mapping.values()
        ):
            raise ValueError("Visual worker metrics must be finite numbers or null")
    for row in timings:
        if (
            type(row["repetition"]) is not int
            or row["duration_seconds"] != durations[row["sample_id"]]
        ):
            raise ValueError("Visual attempt identity differs from the manifest")
        if type(row["executed"]) is not bool or (
            row["executed"] and type(row["success"]) is not bool
        ):
            raise ValueError("Visual attempt outcome is invalid")
        elapsed = row["latency_seconds"]
        if row["executed"] and (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("Executed visual attempt lacks a valid latency")
        if not row["executed"] and (row["success"] is not None or elapsed is not None):
            raise ValueError("Unexecuted visual attempts cannot contain outcomes")
        count = row["selected_frame_count"]
        if count is not None and (type(count) is not int or count < 0):
            raise ValueError("Visual selected-frame count is invalid")
        if row["success"] and (not count or not row["output_fingerprint"]):
            raise ValueError(
                "Successful visual attempt lacks selected frames or canonical output"
            )
        if row["success"] is False and not row.get("error"):
            raise ValueError("Failed visual attempt lacks its error")
        if row["success"] is not True and row["output_fingerprint"] is not None:
            raise ValueError(
                "Failed or unexecuted visual attempt cannot retain a success fingerprint"
            )
    attempts = output["outputs"]["attempts"]
    if set(attempts) != ids or any(
        len(attempts[sample_id]) != repetitions for sample_id in ids
    ):
        raise ValueError("Visual output inventory is incomplete")
    for timing in timings:
        attempt = attempts[timing["sample_id"]][timing["repetition"] - 1]
        predictions = attempt["predictions"]
        if attempt["fingerprint"] != timing["output_fingerprint"] or (
            (predictions is not None) != (timing["success"] is True)
        ):
            raise ValueError("Visual output and timing outcomes disagree")
        if predictions is not None and (
            not isinstance(predictions, list)
            or any(
                not isinstance(prediction, Mapping)
                or not isinstance(prediction.get("text"), str)
                or type(prediction.get("timestamp")) not in (int, float)
                or not math.isfinite(prediction["timestamp"])
                or not 0 <= prediction["timestamp"] <= durations[timing["sample_id"]]
                for prediction in predictions
            )
        ):
            raise ValueError("Visual output has invalid timed text")
    for row in rows:
        for name in QUALITY_DIRECTIONS:
            value = row.get(name)
            if value is not None and (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value < 0
                or (name != "mean_visual_first_detection_delay_seconds" and value > 1)
            ):
                raise ValueError("Visual per-video quality value is invalid")
        sample_id = row["sample_id"]
        first = next(
            timing
            for timing in timings
            if timing["sample_id"] == sample_id and timing["repetition"] == 1
        )
        if (
            row["first_attempt_success"] is not first["success"]
            or row["first_attempt_executed"] is not first["executed"]
        ):
            raise ValueError("Visual quality does not belong to the first attempt")
        if row["selected_frame_count"] != first["selected_frame_count"] or row[
            "quality_latency_seconds"
        ] != (first["latency_seconds"] if first["success"] else None):
            raise ValueError("Visual first-attempt descriptors disagree")
        if row["predictions"] != attempts[sample_id][0]["predictions"]:
            raise ValueError("Visual quality output differs from the first attempt")
        if not first["success"] and any(
            row.get(name) is not None
            for name in QUALITY_DIRECTIONS
            if name not in {"attempt_failure_rate", "repeatability_success_rate"}
        ):
            raise ValueError("Failed first attempt cannot receive quality scores")
        scheduled = [timing for timing in timings if timing["sample_id"] == sample_id]
        rates = (
            attempt_rates([attempt["fingerprint"] for attempt in attempts[sample_id]])
            if all(timing["executed"] for timing in scheduled)
            else {"attempt_failure_rate": None, "repeatability_success_rate": None}
        )
        if any(row.get(name) != value for name, value in rates.items()):
            raise ValueError("Visual reliability does not match its scheduled outcomes")
    if any(
        output["metrics"].get(name) != value
        for name, value in aggregate_quality(rows).items()
    ):
        raise ValueError("Visual aggregate differs from its per-video values")
    if device == "cpu" and output["operational"]["peak_visual_vram_mb"] is not None:
        raise ValueError("CPU visual VRAM must be inapplicable")
    if device == "cuda" and not output.get("execution_error"):
        if output["operational"]["peak_visual_vram_mb"] is None:
            raise ValueError("Visual CUDA telemetry is unavailable")
        if any(
            not output.get(name) or output[name].get("status") != "qualified"
            for name in ("placement_before", "placement_after")
        ):
            raise ValueError("Visual CUDA placement was not verified")


def _run_visual_worker(candidate, items, **settings):
    protocol = protocol_from_worker(settings["protocol"])
    return run_json_worker(
        Path(__file__).with_name("visual_worker.py"),
        {"candidate": candidate, "items": items, **settings},
        device=str(settings["device"]),
        prefix="edumind-video-candidate-",
        error_label="Visual video worker",
        require_vram_measurement=settings["device"] == "cuda",
        telemetry_interval_seconds=protocol.preflight.telemetry_interval_seconds,
        poll_interval_seconds=protocol.preflight.poll_interval_seconds,
        timeout_seconds=protocol.preflight.worker_timeout_seconds,
    )


def _run_smoke_devices(arguments, devices):
    forwarded = without_option_values(sys.argv[1:], ("--device", "--frozen-asr"))
    codes = []
    for device in devices:
        command = [
            sys.executable,
            "-m",
            "experiments.benchmarks.extraction.video.run",
            *forwarded,
            "--device",
            device,
        ]
        if arguments.frozen_asr:
            path = arguments.frozen_asr.with_name(
                f"{arguments.frozen_asr.stem}-{device}{arguments.frozen_asr.suffix}"
            )
            command.extend(("--frozen-asr", str(path)))
        codes.append(subprocess.run(command, check=False).returncode)
    return 0 if all(code == 0 for code in codes) else 2


def _candidates(
    arguments, protocol, *, expected_protocols=None, expected_qualification=None
):
    if arguments.profile in {"validation", "locked"}:
        if arguments.phase != "all" or arguments.shortlist is None:
            raise ValueError(
                "Validation/locked requires the complete comparison and its engineer shortlist decision"
            )
        decision = load_engineer_decision(
            arguments.shortlist,
            exact=1 if arguments.profile == "locked" else None,
            maximum=1 if arguments.profile == "locked" else protocol.maximum_finalists,
            expected_source=(
                "extraction",
                "video-development"
                if arguments.profile == "validation"
                else "video-validation",
                "development" if arguments.profile == "validation" else "validation",
            ),
            expected_protocols=expected_protocols,
            expected_qualification=expected_qualification,
        )
        declared = set(
            all_candidates(protocol, protocol.hybrid_threshold(arguments.profile))
        )
        if not set(decision.selected_candidates) <= declared:
            raise ValueError(
                "Selected video candidate is outside the frozen protocol roster"
            )
        return decision.selected_candidates, {"shortlist": arguments.shortlist}
    if arguments.shortlist:
        raise ValueError("Development/smoke grids do not accept a shortlist")
    if arguments.phase == "scene-selection":
        return scene_candidates(protocol), {}
    return all_candidates(protocol, protocol.hybrid_threshold(arguments.profile)), {}


def _selected_audio(arguments, protocol: AudioProtocol):
    candidates = audio_profiles(protocol)
    if arguments.profile == "smoke" and arguments.audio_candidate:
        if arguments.audio_selection or arguments.audio_candidate not in candidates:
            raise ValueError("Choose one valid smoke ASR override")
        return arguments.audio_candidate, protocol.meta.source_path
    if arguments.profile == "smoke" and arguments.audio_selection is None:
        return next(iter(candidates)), protocol.meta.source_path
    decision = load_engineer_decision(
        arguments.audio_selection,
        exact=1,
        expected_source=("extraction", "audio-validation", "validation")
        if arguments.profile != "smoke"
        else None,
        expected_protocols={"audio": protocol.meta.checksum},
    )
    if decision.selected_candidates[0] not in candidates:
        raise ValueError("Selected audio candidate is not declared")
    return decision.selected_candidates[0], arguments.audio_selection


def _selected_image(arguments, protocol: DocumentProtocol):
    if arguments.profile == "smoke" and arguments.image_candidate:
        if arguments.document_selection:
            raise ValueError("Choose either --image-candidate or --document-selection")
        parse_document_profile(arguments.image_candidate)
        return arguments.image_candidate, None
    if arguments.profile == "smoke" and arguments.document_selection is None:
        return protocol.configuration_candidates("smoke", image=True)[0], None
    decision = load_engineer_decision(
        arguments.document_selection,
        exact=1,
        expected_source=(
            "extraction",
            "document-architecture-validation-image",
            "validation",
        )
        if arguments.profile != "smoke"
        else None,
        expected_protocols={"document": protocol.meta.checksum},
    )
    parse_document_profile(decision.selected_candidates[0])
    return decision.selected_candidates[0], arguments.document_selection
