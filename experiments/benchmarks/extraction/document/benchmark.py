"""Document benchmark orchestration and direct extraction plumbing."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

from edumind.common.artifacts import sha256_file
from edumind.common.paths import PROJECT_ROOT
from edumind.extraction import ExtractionPipeline, ExtractionProfile, SourceKind
from experiments.benchmarks.common.contracts import BenchmarkPlan, BenchmarkResult
from experiments.benchmarks.common.datasets import load_manifest, require_manifest_split
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.extraction.document import runner
from experiments.benchmarks.extraction.document.metrics import (
    METRIC_DIRECTIONS,
    load_reference_data,
    validate_official_evaluators,
    validate_reference,
)
from experiments.benchmarks.extraction.document.official_metrics import (
    official_image_digest,
)
from experiments.benchmarks.extraction.document.profiles import (
    lock_paths,
    parse_document_profile,
)
from experiments.benchmarks.extraction.document.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
)
from experiments.benchmarks.extraction.registry import build_experiment_registry
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
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> BenchmarkResult:
    protocol = load_protocol(protocol_path)
    execution = protocol.profile(profile)
    manifest = load_manifest(manifest_path or _manifest(profile))
    require_manifest_split(
        manifest,
        profile,
        profile,
    )
    selected = [
        item
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
    _validate_assets(
        selected,
        require_checksums=True,
        require_provenance=profile in {"development", "validation"},
    )
    loaded_references = {
        str(item["id"]): load_reference_data(item) for item in selected
    }
    for item in selected:
        payload, reference = loaded_references[str(item["id"])]
        validate_reference(
            item,
            authoritative=profile in {"development", "validation"},
            payload=payload,
            reference=reference,
        )
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
    comparison = document_comparison or (
        "architecture-validation" if profile == "validation" else "configuration"
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
        pipeline = ExtractionPipeline(registry=build_experiment_registry())
        return runner.evaluate_candidate(
            candidate,
            selected,
            plan,
            model_lock,
            component_options,
            references,
            pipeline,
            lambda *args: extract_once(*args, protocol=protocol),
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
        paired_metrics=runner.paired_metrics(directions),
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
        input_artifacts={"manifest": (manifest_path or _manifest(profile)).resolve()},
        protocols={"document": protocol.meta},
        no_mlflow=no_mlflow,
    )


def extract_once(candidate, item, model_lock, component_options, pipeline, *, protocol):
    started = time.perf_counter()
    kind = SourceKind(str(item["kind"]))
    document_profile = parse_document_profile(candidate)
    lock_entry = model_lock.get(document_profile.lock_candidate, {})
    options = dict(component_options)
    options.update(protocol.parser_options(document_profile.runtime_engine))
    options.update(document_profile.options)
    options.update(lock_paths(lock_entry))
    document = pipeline.extract(
        PROJECT_ROOT / str(item["source_path"]),
        source_kind=kind,
        profile=ExtractionProfile(
            name=f"benchmark-{candidate}",
            engine=document_profile.runtime_engine,
            engine_revision=str(lock_entry.get("revision", "system")),
            preprocessing="raw",
            normalization="none",
            routing="direct",
            device=str(component_options["device"]),
            options=options,
        ),
        use_cache=False,
    )
    return document, time.perf_counter() - started


def _validate_assets(
    samples, *, require_checksums: bool, require_provenance: bool
) -> None:
    for item in samples:
        path = PROJECT_ROOT / str(item.get("source_path", ""))
        expected = item.get("asset_sha256")
        if require_checksums and not expected:
            raise ValueError(
                f"Extraction sample {item.get('id')} has no asset_sha256. Prepare smoke "
                "fixtures or the licensed public-asset manifest before running it."
            )
        if require_provenance:
            missing = [
                key
                for key in ("source_license", "source_revision", "document_family")
                if not item.get(key)
            ]
            if missing:
                raise ValueError(
                    f"Extraction sample {item.get('id')} lacks authoritative provenance: "
                    f"{', '.join(missing)}"
                )
        if not path.is_file():
            raise FileNotFoundError(f"Extraction asset is missing: {path}")
        if expected and sha256_file(path) != str(expected):
            raise ValueError(f"Extraction asset checksum mismatch: {path}")
        reference_path = item.get("reference_path")
        if reference_path:
            reference = PROJECT_ROOT / str(reference_path)
            if not reference.is_file():
                raise FileNotFoundError(f"Extraction reference is missing: {reference}")
            expected_reference = item.get("reference_sha256")
            if require_checksums and not expected_reference:
                raise ValueError(
                    f"Extraction sample {item.get('id')} has no reference_sha256"
                )
            if expected_reference and sha256_file(reference) != str(expected_reference):
                raise ValueError(f"Extraction reference checksum mismatch: {reference}")


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
    split = profile
    return PROJECT_ROOT / f"data/benchmarks/extraction/document-{split}.json"
