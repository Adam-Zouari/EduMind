"""Direct ASR benchmark orchestration."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path

from edumind.common.artifacts import stable_hash
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.arguments import (
    LIFECYCLE_PROFILES,
    default_decision_path,
    execution_devices,
)
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.data_validation import (
    manifest_path as default_manifest_path,
)
from experiments.benchmarks.common.datasets import (
    load_manifest,
    require_manifest_split,
)
from experiments.benchmarks.common.decisions import load_engineer_decision
from experiments.benchmarks.common.preflight import (
    current_qualification_fingerprint,
    eligible_candidates,
    model_lock_fingerprints,
    run_preflight,
    stress_input_identity,
)
from experiments.benchmarks.common.preflight_reports import resolve_preflight_report
from experiments.benchmarks.common.process import (
    WorkerExecutionError,
    WorkerMeasurementError,
    run_json_worker,
)
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.extraction.audio.evaluate import (
    INTERVAL_METRICS,
    METRIC_DIRECTIONS,
    PRIMARY_METRICS,
    TIMING_METRICS,
)
from experiments.benchmarks.extraction.audio.protocol import (
    DEFAULT_PROTOCOL_PATH,
    AudioProtocol,
    load_protocol,
)
from experiments.benchmarks.extraction.audio.validation import (
    canonicalize,
    verified_inputs,
    verify_canonical_inputs,
)
from experiments.benchmarks.extraction.audio.validation import (
    reliability_manifest as default_reliability_manifest,
)
from experiments.benchmarks.extraction.media import (
    ffmpeg_version,
)
from experiments.benchmarks.preparation.models import (
    load_selected_model_lock,
)

PROFILE_STAGE = {
    "smoke": "audio-smoke",
    "development": "audio-development",
    "validation": "audio-validation",
    "locked": "audio-locked-test",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark English audio extraction")
    parser.add_argument(
        "--profile",
        choices=LIFECYCLE_PROFILES,
        default="smoke",
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--reliability-manifest", type=Path)
    parser.add_argument(
        "--shortlist",
        type=Path,
        help="engineer decision selecting finalists or one ASR",
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "both"))
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--preflight-run-id")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()
    protocol = load_protocol(arguments.protocol)
    execution_profile = (
        "development" if arguments.profile == "preflight" else arguments.profile
    )
    execution = protocol.profile(execution_profile)
    devices = execution_devices(
        arguments.profile,
        arguments.device,
        smoke_devices=protocol.profile("smoke").devices,
        authoritative_device=execution.device,
    )
    shortlist = arguments.shortlist or default_decision_path("audio", arguments.profile)
    if arguments.profile == "preflight":
        ignored = []
        if arguments.reliability_manifest is not None:
            ignored.append("--reliability-manifest")
        if arguments.preflight_report is not None:
            ignored.append("--preflight-report")
        if arguments.preflight_run_id is not None:
            ignored.append("--preflight-run-id")
        if arguments.shortlist is not None:
            ignored.append("--shortlist")
        if ignored:
            parser.error("ASR preflight does not accept: " + ", ".join(ignored))
        result = run_preflight_profile(
            manifest_path=arguments.manifest,
            device=devices[0],
            no_mlflow=arguments.no_mlflow,
            protocol_path=arguments.protocol,
        )
        payload = {
            "run_id": result.run_id,
            "ready_for_development": result.ready_for_development,
            "qualified_candidates": result.qualified_candidates,
            "excluded_candidates": result.excluded_candidates,
            "blocked_candidates": result.blocked_candidates,
            "artifacts": str(result.artifact_directory),
        }
        print(json.dumps(payload, indent=2))
        return 0 if result.ready_for_development else 2
    results = [
        run(
            arguments.profile,
            manifest_path=arguments.manifest,
            reliability_path=arguments.reliability_manifest,
            device=device,
            no_mlflow=arguments.no_mlflow,
            decision_file=shortlist,
            preflight_report=arguments.preflight_report,
            preflight_run_id=arguments.preflight_run_id,
            protocol_path=arguments.protocol,
        )
        for device in devices
    ]
    print(
        json.dumps(
            [
                {
                    "device": device,
                    "run_id": result.run_id,
                    "complete": result.complete,
                    "artifacts": str(result.artifact_directory),
                }
                for device, result in zip(devices, results, strict=True)
            ],
            indent=2,
        )
    )
    return 0 if all(result.complete for result in results) else 2


def run(
    profile: str,
    *,
    manifest_path: Path | None,
    reliability_path: Path | None,
    device: str,
    no_mlflow: bool,
    decision_file: Path | None,
    preflight_report: Path | None = None,
    preflight_run_id: str | None = None,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
):
    protocol = load_protocol(protocol_path)
    execution = protocol.profile(profile)
    if profile == "smoke" and (
        preflight_report is not None or preflight_run_id is not None
    ):
        raise ValueError("ASR smoke does not consume hardware qualification overrides")
    if execution.hardware_required and device != execution.device:
        raise ValueError(
            f"Authoritative ASR profile requires device {execution.device}"
        )
    speech_path = (manifest_path or default_manifest_path("audio", profile)).resolve()
    controls_path = (
        reliability_path or default_reliability_manifest(profile)
    ).resolve()
    data_report_path, data_report = verified_inputs(
        speech_path, profile, protocol, controls_path=controls_path
    )
    speech_manifest = load_manifest(speech_path, validate_annotations=False)
    reliability_manifest = load_manifest(controls_path, validate_annotations=False)
    speech = [item for item in speech_manifest.samples if item.get("kind") == "audio"]
    split = "locked-test" if profile == "locked" else profile
    require_manifest_split(speech_manifest, profile, split)
    controls = [
        item
        for item in reliability_manifest.samples
        if item.get("kind") == "audio_reliability" and item.get("split") == split
    ]
    declared = tuple(protocol.candidates)
    required_models = tuple(
        protocol.candidate(candidate).model_id for candidate in declared
    )
    model_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=required_models,
    )
    qualification_path = None
    qualification = None
    fingerprint = None
    if profile != "smoke":
        qualification_manifest = speech_manifest
        qualification_metadata = data_report["metadata"]
        if profile != "development":
            development_path = default_manifest_path("audio", "development")
            _, development_report = verified_inputs(
                development_path, "development", protocol
            )
            qualification_metadata = development_report["metadata"]
            qualification_manifest = load_manifest(
                development_path, validate_annotations=False
            )
        qualification_speech = [
            item
            for item in qualification_manifest.samples
            if item.get("kind") == "audio"
        ]
        qualification_stress = max(
            qualification_speech,
            key=lambda item: float(item["duration_seconds"]),
        )
        fingerprint, _ = _qualification_identity(
            protocol,
            model_lock,
            qualification_manifest,
            qualification_stress,
            qualification_metadata["samples"][str(qualification_stress["id"])],
        )
        qualification_path, qualification = resolve_preflight_report(
            benchmark="audio",
            fingerprint=fingerprint,
            candidates=declared,
            explicit=preflight_report,
            run_id=preflight_run_id,
        )
    candidates = _candidates(profile, decision_file, protocol, fingerprint)
    if qualification is not None:
        candidates = eligible_candidates(
            candidates, qualification, profile=profile, label="ASR"
        )

    temporary_root = PROJECT_ROOT / "artifacts/benchmarks/asr-canonical"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as raw_directory:
        canonical_directory = Path(raw_directory)
        canonical_speech = canonicalize(
            speech, canonical_directory / "speech", protocol
        )
        canonical_controls = canonicalize(
            controls, canonical_directory / "reliability", protocol
        )
        verify_canonical_inputs(
            [*canonical_speech, *canonical_controls], data_report["metadata"]
        )
        if profile != "smoke":
            _verify_input_envelope(
                [*canonical_speech, *canonical_controls],
                qualification_metadata["samples"][str(qualification_stress["id"])][
                    "actual_duration_seconds"
                ],
            )
        ffmpeg_identity = ffmpeg_version()
        plan = BenchmarkPlan(
            "extraction",
            PROFILE_STAGE[profile],
            profile,
            speech_manifest.name,
            candidates,
            seed=protocol.meta.seed,
            repetitions=execution.repetitions,
            bootstrap_resamples=execution.bootstrap_resamples,
            warmups=execution.warmups,
            settings={
                "device": device,
                "ffmpeg_version": ffmpeg_identity,
                "ffmpeg_commands": {
                    str(item["id"]): item["ffmpeg_command"]
                    for item in [*canonical_speech, *canonical_controls]
                },
                "speech_manifest_checksum": speech_manifest.fingerprint,
                "reliability_manifest_checksum": reliability_manifest.fingerprint,
                "canonical_checksums": {
                    str(item["id"]): item["canonical_sha256"]
                    for item in [*canonical_speech, *canonical_controls]
                },
                "preflight_run_id": (
                    qualification.get("mlflow_run_id") if qualification else None
                ),
                "preflight_fingerprint": (
                    qualification.get("qualification_fingerprint")
                    if qualification
                    else None
                ),
                "hardware_exclusions": (
                    qualification.get("excluded_candidates", [])
                    if qualification
                    else []
                ),
            },
        )

        def evaluate(candidate: str, context):
            worker_error = None
            try:
                output = _run_worker(
                    candidate,
                    model_lock,
                    canonical_speech,
                    canonical_controls,
                    device=device,
                    profile=profile,
                    directory=canonical_directory,
                    protocol=protocol,
                    prediction_directory=Path(context["artifact_directory"])
                    / "predictions",
                )
            except WorkerExecutionError as exc:
                output = exc.resource_evidence.get("worker_result")
                if (
                    not isinstance(exc, WorkerMeasurementError)
                    or not isinstance(output, dict)
                    or not {"timings", "samples", "metrics"} <= output.keys()
                ):
                    evidence = exc.resource_evidence
                    exc.artifacts.update(
                        {
                            name: evidence[name]
                            for name in ("samples", "timings", "execution_error")
                            if name in evidence
                        }
                    )
                    exc.parameters.update(evidence.get("parameters", {}))
                    exc.parameters["prediction_evidence"] = [
                        row["prediction"]
                        for row in evidence.get("timings", [])
                        if row.get("prediction")
                    ]
                    exc.operational["cold_model_load_seconds"] = evidence.get(
                        "cold_model_load_seconds"
                    )
                    exc.metrics.update(dict.fromkeys(METRIC_DIRECTIONS))
                    exc.artifacts["metric_statuses"] = {
                        name: {
                            "value": exc.operational.get(name),
                            "status": "inapplicable"
                            if name == "peak_vram_mb" and device == "cpu"
                            else "scored"
                            if exc.operational.get(name) is not None
                            else "incomplete",
                            "reason": "cpu_profile"
                            if name == "peak_vram_mb" and device == "cpu"
                            else None
                            if exc.operational.get(name) is not None
                            else "worker_execution_failed",
                            "comparison_status": "incomplete",
                            "comparison_reason": "execution_error",
                            "execution_error_phase": (
                                evidence.get("execution_error") or {}
                            ).get("phase", "worker_execution"),
                            "support": {
                                "recorded_attempt_count": len(
                                    evidence.get("timings", [])
                                )
                            },
                        }
                        for name in METRIC_DIRECTIONS
                    }
                    raise
                worker_error = exc
            _validate_output(
                output,
                canonical_speech,
                canonical_controls,
                execution.repetitions,
                device,
            )
            sample_results = [
                SampleResult(
                    str(row["sample_id"]),
                    row["metric_values"],
                    float(row["quality_latency_seconds"])
                    if row["quality_latency_seconds"] is not None
                    else None,
                    {
                        key: value
                        for key, value in row.items()
                        if key
                        not in {"sample_id", "metric_values", "quality_latency_seconds"}
                    },
                )
                for row in output["samples"]
            ]
            metrics = dict(output["metrics"])
            parameters = {
                **output["parameters"],
                "data_split": split,
                "ffmpeg_version": ffmpeg_identity,
                "speech_manifest_checksum": speech_manifest.fingerprint,
                "reliability_manifest_checksum": reliability_manifest.fingerprint,
            }
            operational = {
                name: metrics.pop(name)
                for name in (
                    "real_time_factor",
                    "p50_warm_clip_latency_seconds",
                    "p95_warm_clip_latency_seconds",
                    "cold_model_load_seconds",
                    "peak_process_tree_ram_mb",
                    "peak_vram_mb",
                )
            }
            artifacts = {
                key: output[key]
                for key in (
                    "samples",
                    "timings",
                    "resource_samples",
                    "gpu_identity",
                    "metric_statuses",
                )
            }
            artifacts["placement"] = {
                "before": output["placement_before"],
                "after": output["placement_after"],
            }
            if worker_error is not None:
                artifacts["worker_error"] = worker_error.resource_evidence
            if output.get("execution_error") or worker_error is not None:
                failure = output.get("execution_error") or {
                    "phase": "supervisor_telemetry",
                    "error": str(worker_error),
                }
                artifacts["execution_error"] = failure
                raise CandidateExecutionError(
                    f"ASR {failure['phase']}: {failure['error']}",
                    samples=tuple(sample_results),
                    metrics=metrics,
                    operational=operational,
                    parameters=parameters,
                    intervals=output["intervals"],
                    artifacts=artifacts,
                )
            return (
                sample_results,
                operational,
                metrics,
                parameters,
                output["intervals"],
                artifacts,
            )

        revisions = {
            candidate: str(
                model_lock[protocol.candidate(candidate).model_id].get(
                    "selection_revision", ""
                )
            )
            for candidate in candidates
        }
        return run_benchmark(
            plan,
            evaluate,
            dataset_checksum=stable_hash(
                {
                    "speech": speech_manifest.fingerprint,
                    "reliability": reliability_manifest.fingerprint,
                    "canonical": plan.settings["canonical_checksums"],
                }
            ),
            directions=METRIC_DIRECTIONS,
            primary_metric=PRIMARY_METRICS,
            required_metrics=tuple(METRIC_DIRECTIONS),
            paired_metrics=(),
            revisions=revisions,
            decision_files={"shortlist": decision_file} if decision_file else None,
            input_artifacts={
                "speech": speech_path,
                "data_validation": data_report_path,
                "reliability": controls_path,
                **(
                    {"preflight_report": qualification_path}
                    if qualification_path is not None
                    else {}
                ),
            },
            protocols={"audio": protocol.meta},
            no_mlflow=no_mlflow,
            monitor_resources=False,
            evaluator_receives_context=True,
            operational_prefix="",
            paired_comparisons=False,
            candidate_artifact_name="candidate.json",
            nullable_metrics=tuple(METRIC_DIRECTIONS),
            run_name_prefix=(f"smoke-{device}" if profile == "smoke" else profile),
        )


def run_preflight_profile(
    *,
    manifest_path: Path | None,
    device: str,
    no_mlflow: bool,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
):
    if device != "cuda":
        raise ValueError("ASR preflight requires CUDA")
    protocol = load_protocol(protocol_path)
    candidates = tuple(protocol.candidates)
    speech_path = (
        manifest_path or default_manifest_path("audio", "development")
    ).resolve()
    data_report_path, data_report = verified_inputs(
        speech_path, "development", protocol
    )
    speech_manifest = load_manifest(speech_path, validate_annotations=False)
    require_manifest_split(speech_manifest, "development", "development")
    speech = [item for item in speech_manifest.samples if item.get("kind") == "audio"]
    stress = max(speech, key=lambda item: float(item["duration_seconds"]))
    model_ids = tuple(
        protocol.candidate(candidate).model_id for candidate in candidates
    )
    model_lock = load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json", candidates=model_ids
    )
    fingerprint, context = _qualification_identity(
        protocol,
        model_lock,
        speech_manifest,
        stress,
        data_report["metadata"]["samples"][str(stress["id"])],
    )
    temporary_root = PROJECT_ROOT / "artifacts/benchmarks/asr-preflight-canonical"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=temporary_root) as raw_directory:
        directory = Path(raw_directory)
        canonical = canonicalize((stress,), directory / "speech", protocol)
        verify_canonical_inputs(canonical, data_report["metadata"])

        def probe(candidate: str):
            return _run_worker(
                candidate,
                model_lock,
                canonical,
                (),
                device="cuda",
                profile="development",
                directory=directory,
                protocol=protocol,
                mode="preflight",
            )

        context = {
            **context,
            "stress_manifest": str(speech_path),
            "stress_manifest_checksum": speech_manifest.fingerprint,
            "stress_sample_id": stress["id"],
            "data_validation_report": str(data_report_path),
        }
        return run_preflight(
            benchmark="audio",
            candidates=candidates,
            fingerprint=fingerprint,
            context=context,
            probe=probe,
            no_mlflow=no_mlflow,
        )


def _run_worker(
    candidate,
    model_lock,
    speech,
    controls,
    *,
    device,
    profile,
    directory,
    protocol,
    mode=None,
    prediction_directory=None,
):
    safe = "".join(character if character.isalnum() else "-" for character in candidate)
    return run_json_worker(
        Path(__file__).with_name("worker.py"),
        {
            "candidate": candidate,
            "model_lock": model_lock,
            "speech": speech,
            "reliability": controls,
            "device": device,
            "profile": profile,
            "protocol": protocol.meta.worker_payload(),
            "prediction_directory": str(
                prediction_directory
                or PROJECT_ROOT
                / "artifacts/benchmarks/asr-predictions"
                / uuid.uuid4().hex
            ),
            **({"mode": mode} if mode else {}),
        },
        device=device,
        prefix=f"{safe}-",
        error_label="ASR worker",
        temporary_root=directory,
        require_vram_measurement=device == "cuda",
        telemetry_interval_seconds=protocol.preflight.telemetry_interval_seconds,
        poll_interval_seconds=protocol.preflight.poll_interval_seconds,
        timeout_seconds=protocol.preflight.worker_timeout_seconds,
    )


def _validate_candidates(candidates, protocol: AudioProtocol) -> None:
    if not candidates:
        raise ValueError("ASR candidate list is empty")
    if len(set(candidates)) != len(candidates):
        raise ValueError("ASR candidate list contains duplicates")
    unknown = sorted(set(candidates) - set(protocol.candidates))
    if unknown:
        raise ValueError("Unknown ASR candidates: " + ", ".join(unknown))


def _verify_input_envelope(samples, tested_duration):
    for item in samples:
        if item["duration_seconds"] > tested_duration + 1e-6:
            raise ValueError(
                f"ASR input {item.get('id')} ({item['duration_seconds']:g}s) exceeds the decoded preflight duration envelope ({tested_duration:g}s); provide a covering reviewed development stress clip and rerun data validation/preflight"
            )


def _validate_output(output, speech, controls, repetitions, device) -> None:
    """Reject missing or contradictory worker evidence before publishing results."""

    def finite(value, *, nullable=False):
        return (nullable and value is None) or (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and value >= 0
        )

    def reject(message):
        raise CandidateExecutionError(message, artifacts={"worker_evidence": output})

    def valid_status(record, value):
        return (
            isinstance(record, dict)
            and record.get("status")
            in {"scored", "inapplicable", "unavailable", "incomplete"}
            and (
                record["status"] == "scored"
                and value is not None
                and record.get("reason") is None
                or record["status"] != "scored"
                and value is None
                and isinstance(record.get("reason"), str)
                and bool(record["reason"])
            )
        )

    items = {str(item["id"]): (item, "speech") for item in speech}
    items.update({str(item["id"]): (item, "nonspeech") for item in controls})
    expected = {
        (sample_id, repetition)
        for sample_id, (_, kind) in items.items()
        for repetition in range(1, (repetitions if kind == "speech" else 1) + 1)
    }
    observed = {}
    for row in output.get("timings", []):
        repetition = row.get("repetition")
        key = (row.get("sample_id"), repetition)
        if (
            not isinstance(repetition, int)
            or isinstance(repetition, bool)
            or key not in expected
            or key in observed
        ):
            reject("Invalid ASR attempt inventory")
        item, kind = items[key[0]]
        if (
            row.get("sample_type") != kind
            or row.get("device") != device
            or not finite(row.get("duration_seconds"))
            or row.get("duration_seconds") != item["duration_seconds"]
        ):
            reject("ASR attempt identity differs from its input")
        executed, success = row.get("executed"), row.get("success")
        fingerprint = row.get("projected_transcript_sha256")
        if not isinstance(executed, bool):
            reject("ASR attempt lacks explicit execution status")
        if executed:
            if not isinstance(success, bool) or not finite(row.get("latency_seconds")):
                reject("Invalid ASR measured attempt")
            if success:
                if (
                    row.get("error") is not None
                    or not isinstance(fingerprint, str)
                    or len(fingerprint) != 64
                    or any(c not in "0123456789abcdef" for c in fingerprint)
                ):
                    reject("ASR successful attempt lacks a valid output fingerprint")
            elif (
                not isinstance(row.get("error"), str)
                or not row["error"]
                or fingerprint is not None
            ):
                reject("Invalid ASR failed attempt evidence")
        elif any(
            row.get(field) is not None
            for field in (
                "success",
                "error",
                "latency_seconds",
                "projected_transcript_sha256",
            )
        ) or not row.get("not_executed_reason"):
            reject("ASR unexecuted attempt contains fabricated observations")
        observed[key] = row
    if observed.keys() != expected:
        reject("Incomplete ASR attempt inventory")
    execution_error = output.get("execution_error")
    if execution_error is not None and (
        not isinstance(execution_error, dict)
        or not all(
            isinstance(execution_error.get(key), str) and execution_error[key]
            for key in ("phase", "error")
        )
    ):
        reject("Invalid ASR execution error evidence")
    if (
        any(not row["executed"] for row in observed.values())
        and execution_error is None
    ):
        reject("ASR unexecuted attempts require an incomplete execution error")
    seen = set()
    for row in output.get("samples", []):
        sample_id = row.get("sample_id")
        if sample_id not in items or sample_id in seen:
            reject("Invalid ASR sample inventory")
        seen.add(sample_id)
        first = observed[(sample_id, 1)]
        if (
            row.get("first_attempt_executed") is not first["executed"]
            or row.get("first_attempt_success") is not first["success"]
        ):
            reject("ASR sample quality does not follow the designated first attempt")
        if (
            not finite(row.get("quality_latency_seconds"), nullable=True)
            or (
                first["success"] is not True
                and row.get("quality_latency_seconds") is not None
            )
            or (
                first["success"] is True
                and row.get("quality_latency_seconds") != first["latency_seconds"]
            )
        ):
            reject("Invalid ASR quality latency")
        if not isinstance(row.get("metric_values"), dict) or not isinstance(
            row.get("metric_statuses"), dict
        ):
            reject("ASR sample lacks metric values/statuses")
        required = set(INTERVAL_METRICS) - TIMING_METRICS
        if (
            row["metric_values"].keys() != required
            or row["metric_statuses"].keys() != required
        ):
            reject("ASR sample lacks required metric values/statuses")
        if any(
            not finite(value, nullable=True) for value in row["metric_values"].values()
        ):
            reject("Invalid ASR sample metric")
        if any(
            not valid_status(row["metric_statuses"][name], value)
            for name, value in row["metric_values"].items()
        ):
            reject("Invalid ASR sample metric status/reason")
        if first["success"] is not True and any(
            value is not None
            for name, value in row["metric_values"].items()
            if name
            not in {"attempt_failure_rate", "transcript_repeatability_success_rate"}
        ):
            reject("ASR first attempt failure cannot produce quality metrics")
    if seen != items.keys():
        reject("Incomplete ASR sample inventory")
    metrics = output.get("metrics", {})
    if not METRIC_DIRECTIONS.keys() <= metrics.keys() or any(
        not finite(value, nullable=True) for value in metrics.values()
    ):
        reject("Invalid ASR aggregate metric contract")
    if device == "cpu" and metrics["peak_vram_mb"] is not None:
        reject("ASR CPU VRAM must be unavailable, not zero")
    statuses = output.get("metric_statuses")
    if not isinstance(statuses, dict) or statuses.keys() != METRIC_DIRECTIONS.keys():
        reject("ASR aggregate metric statuses are missing")
    for name, record in statuses.items():
        if (
            not valid_status(record, metrics[name])
            or not finite(record.get("value"), nullable=True)
            or record.get("value") != metrics[name]
            or record.get("comparison_status")
            != ("incomplete" if execution_error is not None else "complete")
            or "comparison_reason" not in record
            or record["comparison_reason"]
            != ("execution_error" if execution_error is not None else None)
            or "execution_error_phase" not in record
            or record["execution_error_phase"]
            != (execution_error["phase"] if execution_error is not None else None)
            or not isinstance(record.get("support"), dict)
            or any(not finite(value) for value in record["support"].values())
        ):
            reject("ASR aggregate metric status differs from recorded execution")


def _candidates(
    profile: str,
    shortlist: Path | None,
    protocol: AudioProtocol,
    qualification_fingerprint: str | None = None,
) -> tuple[str, ...]:
    if profile in {"smoke", "preflight", "development"}:
        if shortlist is not None:
            raise ValueError(
                f"ASR {profile} does not accept a manual shortlist; its roster "
                "comes from the protocol and applicable preflight"
            )
        return tuple(protocol.candidates)
    if shortlist is None:
        raise ValueError(f"ASR {profile} requires --shortlist DECISION_JSON")
    if qualification_fingerprint is None:
        raise ValueError("ASR selection requires current hardware qualification")
    candidates = load_engineer_decision(
        shortlist,
        exact=1 if profile == "locked" else None,
        maximum=1 if profile == "locked" else protocol.maximum_finalists,
        expected_protocols={"audio": protocol.meta.checksum},
        expected_qualification=qualification_fingerprint,
        expected_source=(
            "extraction",
            "audio-development" if profile == "validation" else "audio-validation",
            "development" if profile == "validation" else "validation",
        ),
    ).selected_candidates
    _validate_candidates(candidates, protocol)
    return candidates


def _qualification_identity(
    protocol: AudioProtocol, model_lock, manifest, stress, canonical_identity
):
    execution = protocol.profile("development")
    locked_models = model_lock_fingerprints(model_lock)
    revisions = {
        alias: locked_models[protocol.candidate(alias).model_id]
        for alias in protocol.candidates
    }
    return current_qualification_fingerprint(
        benchmark="audio",
        candidates=tuple(protocol.candidates),
        protocols={"audio": protocol.meta},
        revisions=revisions,
        execution={
            "device": "cuda",
            "dtype_by_candidate": {
                alias: protocol.dtype(alias, "cuda") for alias in protocol.candidates
            },
            "batch_size": execution.batch_size,
            "preflight": asdict(protocol.preflight),
            "ffmpeg_version": ffmpeg_version(),
        },
        input_envelope={
            **stress_input_identity(manifest, (stress,)),
            "kind": "canonical-audio",
            "canonical_input": dict(canonical_identity),
            "sample_rate_hz": protocol.audio["sample_rate_hz"],
            "channels": protocol.audio["channels"],
            "maximum_duration_seconds": protocol.audio["maximum_duration_seconds"],
            "tested_duration_seconds": canonical_identity["actual_duration_seconds"],
        },
    )
