"""Shared execution plumbing for direct extraction experiments."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path

from edumind.common.artifacts import sha256_file
from edumind.common.paths import PROJECT_ROOT
from edumind.extraction import ExtractionPipeline, ExtractionProfile, SourceKind
from experiments.benchmarks.common.contracts import BenchmarkPlan, BenchmarkResult
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.extraction.document import runner as document_runner
from experiments.benchmarks.extraction.registry import build_experiment_registry
from experiments.benchmarks.preparation.evaluators import OMNIDOCBENCH_REVISION
from experiments.benchmarks.preparation.models import load_selected_model_lock

STAGES = {"document"}
LOCK_CANDIDATES = {
    "docling-standard": "docling-standard",
    "docling-vlm-granite-258m": "ibm-granite/granite-docling-258M",
    "paddleocr-vl-1.6": "PaddlePaddle/PaddleOCR-VL-1.6",
}


def run(
    stage: str,
    profile: str,
    candidates: tuple[str, ...],
    *,
    manifest_path: Path | None = None,
    no_mlflow: bool = False,
    component_options: Mapping[str, object] | None = None,
    decision_files: Mapping[str, Path] | None = None,
    document_kind: str | None = None,
    document_comparison: str | None = None,
) -> BenchmarkResult:
    if stage not in STAGES:
        raise ValueError(f"Unknown extraction stage: {stage}")
    from experiments.benchmarks.extraction.document import evaluate as evaluator
    manifest = load_manifest(manifest_path or _manifest(stage, profile))
    selected = [
        item
        for item in manifest.samples
        if item.get("kind") in {"image", "pdf", "docx"}
        and (document_kind is None or item.get("kind") == document_kind)
    ]
    if not selected:
        raise ValueError(f"Manifest {manifest.name} has no samples for {stage}")
    minimum = _minimum_samples(profile, document_kind)
    if minimum and len(selected) < minimum:
        raise ValueError(
            f"{stage} {profile} requires at least {minimum} frozen samples; "
            f"manifest contains {len(selected)}"
        )
    _validate_assets(
        selected,
        require_checksums=True,
        require_provenance=profile in {"standard", "full"},
    )
    component_options = dict(component_options or {})
    for item in selected:
        evaluator.validate_reference(item, authoritative=profile in {"standard", "full"})
    uses_official_evaluators = evaluator.validate_official_evaluators(selected)
    comparison = document_comparison or (
        "architecture-validation" if profile == "full" else "configuration"
    )
    plan_stage = f"document-{comparison}-{document_kind or 'all'}"
    plan = BenchmarkPlan(
        "extraction",
        plan_stage,
        profile,
        manifest.name,
        candidates,
        repetitions=1 if profile == "smoke" else 3,
        bootstrap_resamples=0 if profile == "smoke" else 10_000,
        settings=component_options,
    )
    model_lock = _model_lock(candidates)
    for candidate in candidates:
        lock_name = _lock_candidate(candidate)
        document_runner.validate_prepared_components(
            candidate, model_lock.get(lock_name, {})
        )
    def evaluate(candidate: str):
        pipeline = ExtractionPipeline(registry=build_experiment_registry())
        return document_runner.evaluate_candidate(
            candidate,
            selected,
            plan,
            model_lock,
            component_options,
            pipeline,
            _extract_once,
        )

    directions = document_runner.directions_for(selected, evaluator.directions())
    primary_metrics = document_runner.primary_metrics(directions)
    required_metrics = document_runner.required_metrics(directions)
    return run_benchmark(
        plan,
        evaluate,
        dataset_checksum=manifest.fingerprint,
        directions=directions,
        primary_metric=primary_metrics,
        required_metrics=required_metrics,
        paired_metrics=document_runner.paired_metrics(directions),
        revisions={
            **{name: str(value.get("revision", "")) for name, value in model_lock.items()},
            **(
                {
                    "omnidocbench-evaluator": OMNIDOCBENCH_REVISION,
                    "omnidocbench-image": evaluator.official_image_digest(),
                }
                if uses_official_evaluators
                else {}
            ),
        },
        decision_files=decision_files,
        no_mlflow=no_mlflow,
    )


def _minimum_samples(profile: str, document_kind: str | None) -> int:
    if profile == "smoke":
        return 0
    targets = {
        "standard": {"image": 72, "pdf": 36, "docx": 27},
        "full": {"image": 24, "pdf": 12, "docx": 9},
    }
    if document_kind:
        return targets[profile][document_kind]
    return sum(targets[profile].values())


def _extract_once(stage, candidate, item, model_lock, component_options, pipeline):
    if stage != "document":
        raise ValueError("The shared extraction path supports document benchmarks only")
    started = time.perf_counter()
    kind = SourceKind(str(item["kind"]))
    engine = candidate.partition("|")[0]
    engine = "docling-standard" if engine == "docling-standard-native" else engine
    lock_name = _lock_candidate(candidate)
    lock_entry = model_lock.get(lock_name, {})
    document = pipeline.extract(
        PROJECT_ROOT / str(item["source_path"]),
        source_kind=kind,
        profile=ExtractionProfile(
            name=f"benchmark-{candidate}",
            engine=engine,
            engine_revision=str(lock_entry.get("revision", "system")),
            preprocessing="raw",
            normalization="none",
            routing="direct",
            device=_device(component_options, item),
            options=_options(item, lock_entry, component_options, candidate),
        ),
        use_cache=False,
    )
    return document.text, document, time.perf_counter() - started


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


def _device(component_options, item) -> str:
    return str(item.get("device") or component_options.get("device") or "cpu")


def _options(
    item: Mapping[str, object],
    lock_entry: Mapping[str, object],
    component_options: Mapping[str, object],
    candidate: str,
) -> dict[str, object]:
    raw = item.get("options", {})
    result = dict(raw) if isinstance(raw, Mapping) else {}
    result.update(component_options)
    if candidate.startswith("docling-standard|"):
        for factor in candidate.split("|")[1:]:
            key, value = factor.split("=", 1)
            result[
                {
                    "ocr": "ocr_engine",
                    "mode": "ocr_mode",
                    "table": "table_mode",
                    "formula": "formula_enrichment",
                }[key]
            ] = value == "on" if key == "formula" else value
    result.update(_lock_paths(lock_entry))
    return result


def _lock_candidate(candidate: str) -> str:
    engine = candidate.partition("|")[0]
    if engine == "docling-standard-native":
        engine = "docling-standard"
    return LOCK_CANDIDATES.get(engine, engine)


def _model_lock(candidates: tuple[str, ...]) -> dict[str, dict[str, object]]:
    required = tuple(dict.fromkeys(_lock_candidate(candidate) for candidate in candidates))
    return load_selected_model_lock(
        PROJECT_ROOT / "data/benchmarks/models/selected.json",
        candidates=required,
    )


def _manifest(stage: str, profile: str) -> Path:
    if profile == "smoke":
        return PROJECT_ROOT / "data/benchmarks/extraction/smoke.json"
    split = "development" if profile == "standard" else "validation"
    return PROJECT_ROOT / f"data/benchmarks/extraction/{stage}-{split}.json"


def _lock_paths(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): value
        for key, value in entry.items()
        if str(key).endswith("_path") or str(key).endswith("_dir")
    }
