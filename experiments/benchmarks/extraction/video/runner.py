"""Video comparisons: one audio-preparation child followed by visual-only children."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

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
    manifest_path as _default_manifest,
)
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.preflight import eligible_candidates
from experiments.benchmarks.common.preflight_reports import resolve_preflight_report
from experiments.benchmarks.common.process import run_json_worker
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
from experiments.benchmarks.extraction.media import ffmpeg_version
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
)
from experiments.benchmarks.extraction.video.preflight import (
    run_video_preflight,
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
    manifest_path = (arguments.manifest or _manifest(profile)).resolve()
    validation_path, _ = verified_inputs(manifest_path, profile, protocol)
    manifest = load_manifest(manifest_path)
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
    candidates, decisions = _candidates(arguments, protocol)
    qualification = qualification_path = None
    if profile != "smoke":
        development_path = (
            manifest_path if profile == "development" else _manifest("development")
        )
        verified_inputs(development_path, "development", protocol)
        development = load_manifest(development_path)
        stress = max(
            (item for item in development.samples if item.get("kind") == "video"),
            key=lambda item: float(item["duration_seconds"]),
        )
        declared, fingerprint, _ = video_qualification_identity(
            protocol,
            audio_protocol,
            document_protocol,
            audio_candidate,
            image_candidate,
            development,
            stress,
        )
        qualification_path, qualification = resolve_preflight_report(
            benchmark="video",
            fingerprint=fingerprint,
            candidates=declared,
            explicit=arguments.preflight_report,
            run_id=arguments.preflight_run_id,
        )
        if f"frozen-asr|{audio_candidate}" not in qualification["qualified_candidates"]:
            raise ValueError("Selected frozen ASR did not pass video preflight")
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
    report_path, _ = verified_inputs(manifest_path, profile, protocol)
    if (
        data_validation_report is not None
        and Path(data_validation_report).resolve() != report_path.resolve()
    ):
        raise ValueError(
            "Video data-validation report differs from the verified report"
        )
    data_validation_report = report_path
    current_manifest = load_manifest(manifest_path)
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
                create_frozen_asr_artifact(
                    path,
                    items=items,
                    manifest_checksum=manifest_checksum,
                    protocol=protocol,
                    audio_protocol=audio_protocol,
                    audio_candidate=audio_candidate,
                    audio_decision_path=audio_decision,
                    device=device,
                    ffmpeg_version=ffmpeg_version,
                    profile_name=profile,
                    producer_run_id=context["mlflow_run_id"],
                )
            artifact, checksum = load_frozen_asr_artifact(
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
            )
            if any(
                len(row["windows"]) != row["scheduled_window_count"]
                for row in artifact["videos"]
            ):
                raise CandidateExecutionError(
                    "Frozen ASR preparation has an incomplete window inventory",
                    metrics=artifact["metrics"],
                    artifacts={"outputs": artifact},
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
                    "run_type": "preparation",
                    **expected_identity,
                    "frozen_asr_run_id": artifact["run_id"],
                    "frozen_asr_checksum": checksum,
                },
                artifact["intervals"],
                {
                    "outputs": artifact,
                    "samples": _flat_rows(
                        rows, ("windows", "segments", "nonspeech_units")
                    ),
                    "windows": [
                        {
                            "sample_id": row["sample_id"],
                            **{
                                key: value
                                for key, value in window.items()
                                if key != "segments"
                            },
                        }
                        for row in rows
                        for window in row["windows"]
                    ],
                    "resource_samples": artifact["resource_samples"],
                    "gpu_identity": artifact["gpu_identity"],
                    "ffmpeg_commands": artifact["ffmpeg_commands"],
                },
            )
        if not frozen:
            raise RuntimeError(
                "Frozen-ASR preparation did not deliver a validated artifact"
            )
        output = _run_visual_worker(
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
        )
        rows = output["samples"]
        if sorted(row["sample_id"] for row in rows) != sorted(
            str(item["id"]) for item in items
        ):
            raise ValueError("Visual worker sample inventory differs from the manifest")
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
            {
                "samples": _flat_rows(rows, ("predictions",)),
                "timings": output["timings"],
                "outputs": output["outputs"],
                "ffmpeg_commands": output["ffmpeg_commands"],
                "resource_samples": output["resource_samples"],
                "gpu_identity": output["gpu_identity"],
                "placement": {
                    "before": output.get("placement_before"),
                    "after": output.get("placement_after"),
                },
            },
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
            **({"frozen_asr": frozen_asr_path} if frozen_asr_path else {}),
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


def _run_visual_worker(candidate, items, **settings):
    protocol = protocol_from_worker(settings["protocol"])
    return run_json_worker(
        Path(__file__).with_name("visual_worker.py"),
        {"candidate": candidate, "items": items, **settings},
        device=str(settings["device"]),
        prefix="edumind-video-candidate-",
        error_label="Visual video worker",
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


def _candidates(arguments, protocol):
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
    )
    parse_document_profile(decision.selected_candidates[0])
    return decision.selected_candidates[0], arguments.document_selection


def _manifest(profile):
    return _default_manifest("video", profile)
