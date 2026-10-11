"""Document benchmark orchestration and direct extraction plumbing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.contracts import BenchmarkPlan, BenchmarkResult
from experiments.benchmarks.common.datasets import load_manifest, require_manifest_split
from experiments.benchmarks.common.preflight import (
    current_qualification_fingerprint,
    eligible_candidates,
    model_lock_fingerprints,
    run_preflight,
    stress_input_identity,
)
from experiments.benchmarks.common.preflight_reports import resolve_preflight_report
from experiments.benchmarks.common.process import run_json_worker
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.extraction.document import runner
from experiments.benchmarks.extraction.document.metrics import (
    METRIC_DIRECTIONS,
    load_reference_data,
    validate_official_evaluators,
)
from experiments.benchmarks.extraction.document.official_metrics import (
    official_image_digest,
)
from experiments.benchmarks.extraction.document.profiles import (
    parse_document_profile,
)
from experiments.benchmarks.extraction.document.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)
from experiments.benchmarks.extraction.document.validation import verified_inputs
from experiments.benchmarks.preparation.evaluators import OMNIDOCBENCH_REVISION
from experiments.benchmarks.preparation.models import load_selected_model_lock


def run(
    profile: str,
    candidates: tuple[str, ...],
    *,
    manifest_path: Path | None = None,
    no_mlflow: bool = False,
    component_options: Mapping[str, object] | None = None,
    decision_files: Mapping[str, Path] | None = None,
    document_kind: str | None = None,
    document_comparison: str | None = None,
    preflight_report: Path | None = None,
    preflight_run_id: str | None = None,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> BenchmarkResult:
    protocol = load_protocol(protocol_path)
    execution = protocol.profile(profile)
    data_report_path, data_report = verified_inputs(
        manifest_path or _manifest(profile), profile, protocol
    )
    manifest = load_manifest(manifest_path or _manifest(profile))
    require_manifest_split(
        manifest,
        profile,
        "locked-test" if profile == "locked" else profile,
    )
    selected = [
        {**item, **data_report["metadata"]["samples"][str(item["id"])]}
        for item in manifest.samples
        if item.get("kind") in {"image", "pdf", "docx"}
        and (document_kind is None or item.get("kind") == document_kind)
    ]
    if not selected:
        raise ValueError(f"Manifest {manifest.name} has no document samples")
    minimum = protocol.minimum_samples(profile, document_kind)
    if minimum and len(selected) < minimum:
        raise ValueError(
            f"Document {profile} requires at least {minimum} frozen samples; "
            f"manifest contains {len(selected)}"
        )
    loaded_references = {
        str(item["id"]): load_reference_data(item) for item in selected
    }
    references = {
        sample_id: reference for sample_id, (_, reference) in loaded_references.items()
    }
    uses_official_evaluators = validate_official_evaluators(
        tuple(references.values()), timeout_seconds=protocol.evaluator_timeout_seconds
    )
    component_options = dict(component_options or {})
    unknown_component_options = set(component_options) - {"device"}
    if unknown_component_options:
        raise ValueError(
            "Document runtime options belong in protocol.yaml: "
            + ", ".join(sorted(unknown_component_options))
        )
    component_options.setdefault("device", execution.device)
    if str(component_options["device"]) not in (
        execution.devices or (execution.device,)
    ):
        raise ValueError(
            "Document device override violates the frozen execution profile"
        )
    comparison = document_comparison or (
        f"architecture-{profile}"
        if profile in {"validation", "locked"}
        else "configuration"
    )
    plan = BenchmarkPlan(
        "extraction",
        f"document-{comparison}-{document_kind or 'all'}",
        profile,
        manifest.name,
        candidates,
        seed=protocol.meta.seed,
        repetitions=execution.repetitions,
        bootstrap_resamples=execution.bootstrap_resamples,
        warmups=execution.warmups,
        settings=component_options,
    )
    declared = declared_document_candidates(protocol)
    qualification_path = None
    qualification = None
    if profile != "smoke":
        if profile != "development":
            verified_inputs(_manifest("development"), "development", protocol)
        qualification_lock = _model_lock(declared)
        qualification_manifest = (
            manifest
            if profile == "development"
            else load_manifest(_manifest("development"))
        )
        qualification_stress = _stress_documents(qualification_manifest)
        fingerprint, _ = _qualification_identity(
            protocol,
            qualification_lock,
            declared,
            qualification_manifest,
            qualification_stress,
        )
        qualification_path, qualification = resolve_preflight_report(
            benchmark="document",
            fingerprint=fingerprint,
            candidates=declared,
            explicit=preflight_report,
            run_id=preflight_run_id,
        )
        candidates = eligible_candidates(
            candidates, qualification, profile=profile, label="Document"
        )
        plan = BenchmarkPlan(
            **{
                **vars(plan),
                "candidates": candidates,
                "settings": {
                    **plan.settings,
                    "preflight_run_id": qualification.get("mlflow_run_id"),
                    "preflight_fingerprint": qualification.get(
                        "qualification_fingerprint"
                    ),
                    "hardware_exclusions": qualification.get("excluded_candidates", []),
                },
            }
        )
    model_lock = _model_lock(candidates)
    requested_device = str(component_options.get("device", execution.device))
    for candidate in candidates:
        document_profile = parse_document_profile(candidate)
        allowed_devices = protocol.backend_devices[document_profile.runtime_engine]
        if requested_device not in allowed_devices:
            raise ValueError(
                f"{document_profile.runtime_engine} cannot run on {requested_device}; "
                f"allowed devices: {', '.join(allowed_devices)}"
            )
        protocol.validate_candidate_factors(document_profile.factors)
        runner.validate_prepared_components(
            candidate,
            model_lock.get(document_profile.lock_candidate, {}),
            protocol,
        )

    def evaluate(candidate: str):
        return runner.evaluate_candidate(
            candidate,
            selected,
            plan,
            model_lock,
            component_options,
            references,
            protocol,
        )

    directions = runner.directions_for(tuple(references.values()), METRIC_DIRECTIONS)
    return run_benchmark(
        plan,
        evaluate,
        dataset_checksum=manifest.fingerprint,
        directions=directions,
        primary_metric=runner.primary_metrics(directions),
        required_metrics=runner.required_metrics(directions),
        nullable_metrics=runner.required_metrics(directions),
        paired_metrics=runner.paired_metrics(directions),
        paired_group_key="source_group_id",
        monitor_resources=False,
        candidate_artifact_name="candidate.json",
        revisions={
            **{
                name: str(value.get("revision", ""))
                for name, value in model_lock.items()
            },
            **(
                {
                    "omnidocbench-evaluator": OMNIDOCBENCH_REVISION,
                    "omnidocbench-image": official_image_digest(),
                }
                if uses_official_evaluators
                else {}
            ),
        },
        decision_files=decision_files,
        input_artifacts={
            "manifest": (manifest_path or _manifest(profile)).resolve(),
            "data_validation": data_report_path,
            **(
                {"preflight_report": qualification_path}
                if qualification_path is not None
                else {}
            ),
        },
        protocols={"document": protocol.meta},
        no_mlflow=no_mlflow,
        run_name_prefix=(
            f"smoke-{requested_device}-{document_kind or 'all'}"
            if profile == "smoke"
            else (
                f"development-{'config' if comparison == 'configuration' else 'parsers'}-{document_kind or 'all'}"
                if profile == "development" and document_kind != "docx"
                else f"{profile}-{document_kind or 'all'}"
            )
        ),
    )


def run_preflight_profile(
    *,
    manifest_path: Path | None,
    no_mlflow: bool,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
):
    protocol = load_protocol(protocol_path)
    path = (manifest_path or _manifest("development")).resolve()
    manifest = load_manifest(path)
    data_report_path, _ = verified_inputs(path, "development", protocol)
    require_manifest_split(manifest, "development", "development")
    stress = _stress_documents(manifest)
    candidates = declared_document_candidates(protocol)
    model_lock = _model_lock(candidates)
    for candidate in candidates:
        profile = parse_document_profile(candidate)
        runner.validate_prepared_components(
            candidate, model_lock.get(profile.lock_candidate, {}), protocol
        )
    fingerprint, context = _qualification_identity(
        protocol, model_lock, candidates, manifest, stress
    )

    def probe(candidate: str):
        candidate_items = _preflight_items(candidate, stress)
        model_backed = candidate != "docling-standard-native"
        result = run_json_worker(
            Path(__file__).with_name("worker.py"),
            {
                "candidate": candidate,
                "items": candidate_items,
                "model_lock": model_lock,
                "protocol": protocol.meta.worker_payload(),
                "profile": "development",
                "mode": "preflight",
                "device": "cuda",
            },
            device="cuda",
            prefix="edumind-document-preflight-",
            error_label=f"document preflight worker {candidate}",
            require_vram_measurement=model_backed,
            telemetry_interval_seconds=protocol.preflight.telemetry_interval_seconds,
            poll_interval_seconds=protocol.preflight.poll_interval_seconds,
            timeout_seconds=protocol.preflight.worker_timeout_seconds,
        )
        supervision = result.pop("_worker_supervision", {})
        if isinstance(supervision, Mapping):
            result.update(supervision)
        if not model_backed:
            result.setdefault("vram_measurement_method", "not-applicable")
        return result

    return run_preflight(
        benchmark="document",
        candidates=candidates,
        fingerprint=fingerprint,
        context={
            **context,
            "stress_manifest": str(path),
            "stress_manifest_checksum": manifest.fingerprint,
            "stress_sample_ids": [str(item["id"]) for item in stress],
            "data_validation_report": str(data_report_path),
        },
        probe=probe,
        required_groups={
            kind: tuple(
                candidate
                for candidate in candidates
                if kind
                in {str(item["kind"]) for item in _preflight_items(candidate, stress)}
            )
            for kind in ("pdf", "image", "docx")
        },
        no_mlflow=no_mlflow,
    )


def _preflight_items(candidate: str, stress):
    by_kind = {str(item["kind"]): item for item in stress}
    profile = parse_document_profile(candidate)
    if profile.requested_engine == "docling-standard-native":
        kinds = ("docx",)
    elif profile.factors.get("mode") == "pdf_aware_layout_regions":
        kinds = ("pdf",)
    else:
        kinds = ("pdf", "image")
    return tuple(by_kind[kind] for kind in kinds)


def declared_document_candidates(protocol):
    return tuple(
        dict.fromkeys(
            (
                *protocol.configuration_candidates("development"),
                *protocol.configuration_candidates("development", image=True),
                "docling-standard-native",
                "docling-vlm-granite-258m",
                "paddleocr-vl-1.6",
            )
        )
    )


def _qualification_identity(protocol, model_lock, candidates, manifest, stress):
    execution = protocol.profile("development")
    return current_qualification_fingerprint(
        benchmark="document",
        candidates=candidates,
        protocols={"document": protocol.meta},
        revisions=model_lock_fingerprints(model_lock),
        execution={
            "device": execution.device,
            "dtype": execution.dtype,
            "batch_size": execution.batch_size,
            "resource_policy": "backend-specific-reporting",
            "preflight": asdict(protocol.preflight),
        },
        input_envelope={
            **stress_input_identity(manifest, stress),
            "source_types": ["image", "pdf", "docx"],
            "source_sizes_bytes": {
                str(item["kind"]): (PROJECT_ROOT / str(item["source_path"]))
                .stat()
                .st_size
                for item in stress
            },
        },
    )


def _stress_documents(manifest):
    items = [
        item
        for item in manifest.samples
        if item.get("kind") in {"image", "pdf", "docx"}
    ]
    return tuple(
        max(
            (item for item in items if item.get("kind") == kind),
            key=lambda item: (PROJECT_ROOT / str(item["source_path"])).stat().st_size,
        )
        for kind in ("image", "pdf", "docx")
    )


def _model_lock(candidates: tuple[str, ...]) -> dict[str, dict[str, object]]:
    required = tuple(
        dict.fromkeys(
            parse_document_profile(candidate).lock_candidate for candidate in candidates
        )
    )
    return load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=required,
    )


def _manifest(profile: str) -> Path:
    if profile == "smoke":
        return PROJECT_ROOT / "data/benchmarks/extraction/smoke.json"
    split = "locked-test" if profile == "locked" else profile
    return PROJECT_ROOT / f"data/benchmarks/extraction/document-{split}.json"
