"""Shared execution plumbing for direct extraction experiments."""

from __future__ import annotations

import importlib
import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from edumind.common.artifacts import sha256_file
from edumind.common.paths import PROJECT_ROOT
from edumind.extraction import ExtractionPipeline, ExtractionProfile, SourceKind
from experiments.benchmarks.common.contracts import BenchmarkPlan, BenchmarkResult, SampleResult
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.extraction.registry import build_experiment_registry
from experiments.benchmarks.preparation.models import load_selected_model_lock

STAGES = {"document", "audio", "video"}
LOCK_CANDIDATES = {
    "docling-standard": "docling-standard",
    "docling-vlm-granite-258m": "ibm-granite/granite-docling-258M",
    "paddleocr-vl-1.6": "PaddlePaddle/PaddleOCR-VL-1.6",
    "whisper-small-en-control": "openai/whisper-small.en",
    "canary-180m": "nvidia/canary-180m-flash",
    "parakeet-tdt-0.6b-v2": "nvidia/parakeet-tdt-0.6b-v2",
    "moss-transcribe-diarize": "OpenMOSS-Team/MOSS-Transcribe-Diarize",
    "qwen3-asr-1.7b-aligned": "Qwen/Qwen3-ASR-1.7B-hf",
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
) -> BenchmarkResult:
    if stage not in STAGES:
        raise ValueError(f"Unknown extraction stage: {stage}")
    evaluator = importlib.import_module(
        f"experiments.benchmarks.extraction.{stage}.evaluate"
    )
    manifest = load_manifest(manifest_path or _manifest(stage, profile))
    selected = [
        item
        for item in manifest.samples
        if (stage == "document" and item.get("kind") in {"image", "pdf", "docx"})
        or item.get("kind") == stage
    ]
    if not selected:
        raise ValueError(f"Manifest {manifest.name} has no samples for {stage}")
    minimum = {"document": 45, "audio": 18, "video": 6}
    if profile in {"standard", "full"} and len(selected) < minimum[stage]:
        raise ValueError(
            f"{stage} {profile} requires at least {minimum[stage]} frozen samples; "
            f"manifest contains {len(selected)}"
        )
    _validate_assets(
        selected,
        require_checksums=True,
        require_provenance=profile in {"standard", "full"},
    )
    component_options = dict(component_options or {})
    architecture_comparison = stage == "document" and any(
        candidate in {"docling-vlm-granite-258m", "paddleocr-vl-1.6"}
        for candidate in candidates
    )
    evaluation_items = (
        [item for item in selected if item.get("kind") in {"image", "pdf"}]
        if architecture_comparison
        else selected
    )
    plan = BenchmarkPlan(
        "extraction",
        stage,
        profile,
        manifest.name,
        candidates,
        repetitions=1 if profile == "smoke" else 3,
        bootstrap_resamples=500 if profile == "smoke" else 10_000,
        settings={
            **component_options,
            "document_scope": (
                "image-pdf-architecture" if architecture_comparison else "all-supported"
            ),
        },
    )
    model_lock = _model_lock(stage)
    prepared_options = _prepared_component_options(component_options, model_lock)

    def evaluate(candidate: str):
        pipeline = ExtractionPipeline(registry=build_experiment_registry())
        samples: list[SampleResult] = []
        latencies: list[float] = []
        ordered = list(evaluation_items)
        random.Random(plan.seed).shuffle(ordered)
        cold_load_seconds = None
        _, _, cold_load_seconds = _extract_once(
            stage, candidate, ordered[0], model_lock, prepared_options, pipeline
        )
        for _ in range(plan.warmups):
            _extract_once(stage, candidate, ordered[0], model_lock, prepared_options, pipeline)
        for item in ordered:
            outputs = [
                _extract_once(stage, candidate, item, model_lock, prepared_options, pipeline)
                for _ in range(plan.repetitions)
            ]
            hypothesis, document, _ = outputs[0]
            item_latencies = [value[2] for value in outputs]
            latency = float(np.median(item_latencies))
            latencies.extend(item_latencies)
            quality = evaluator.metrics(
                str(item["reference"]), hypothesis, item, document
            )
            quality["determinism"] = float(all(value[0] == hypothesis for value in outputs))
            samples.append(
                SampleResult(
                    str(item["id"]),
                    quality,
                    latency,
                    {
                        "kind": item.get("kind"),
                        "candidate": candidate,
                        "warnings": len(document.warnings) if document else 0,
                    },
                )
            )
        operational = {
            "p50_latency_seconds": float(np.median(latencies)),
            "p95_latency_seconds": float(np.quantile(latencies, 0.95)),
            "items_per_minute": 60.0 * len(latencies) / max(sum(latencies), 1e-9),
        }
        if cold_load_seconds is not None:
            operational["cold_load_seconds"] = cold_load_seconds
        durations = [
            float(item["duration_seconds"])
            for item in evaluation_items
            if item.get("duration_seconds")
        ]
        if durations and len(durations) == len(evaluation_items):
            operational["real_time_factor"] = sum(latencies) / (
                sum(durations) * plan.repetitions
            )
        return samples, operational

    return run_benchmark(
        plan,
        evaluate,
        dataset_checksum=manifest.fingerprint,
        directions=_metric_directions(stage, profile, evaluator.directions()),
        primary_metric={
            "document": "content_f1",
            "audio": "word_error_rate",
            "video": "complete_content_recall",
        }[stage],
        revisions={name: str(value.get("revision", "")) for name, value in model_lock.items()},
        decision_files=decision_files,
        no_mlflow=no_mlflow,
    )


def _metric_directions(
    stage: str, profile: str, directions: Mapping[str, str]
) -> dict[str, str]:
    """Declare smoke metrics explicitly; standard/full require the complete contract."""

    if profile != "smoke":
        return dict(directions)
    common = {
        "character_error_rate",
        "word_error_rate",
        "content_f1",
        "reading_order_accuracy",
        "operational.p95_latency_seconds",
        "operational.peak_ram_mb",
    }
    stage_specific = {
        "document": {"page_coverage"},
        "audio": set(),
        "video": {"transcript_word_error_rate", "complete_content_recall"},
    }[stage]
    required = common | stage_specific
    return {name: direction for name, direction in directions.items() if name in required}


def _extract_once(stage, candidate, item, model_lock, component_options, pipeline):
    started = time.perf_counter()
    kind = SourceKind(str(item["kind"]))
    engine, preprocessing = _engine_and_preprocessing(stage, candidate, item)
    lock_name = (
        LOCK_CANDIDATES[
            str(component_options.get("audio_candidate", "whisper-small-en-control"))
        ]
        if stage == "video"
        else LOCK_CANDIDATES.get(engine, engine)
    )
    lock_entry = model_lock.get(lock_name, {})
    document = pipeline.extract(
        PROJECT_ROOT / str(item["source_path"]),
        source_kind=kind,
        profile=ExtractionProfile(
            name=f"benchmark-{candidate}",
            engine=engine,
            engine_revision=str(lock_entry.get("revision", "system")),
            preprocessing=preprocessing,
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
            if item.get("kind") == "pdf":
                reference_pages = item.get("reference_page_texts")
                if (
                    not isinstance(reference_pages, Sequence)
                    or isinstance(reference_pages, (str, bytes))
                    or not reference_pages
                    or len(reference_pages) != int(item.get("reference_pages", 0))
                ):
                    raise ValueError(
                        f"Authoritative PDF sample {item.get('id')} requires "
                        "reference_page_texts matching reference_pages"
                    )
        if not path.is_file():
            raise FileNotFoundError(f"Extraction asset is missing: {path}")
        if expected and sha256_file(path) != str(expected):
            raise ValueError(f"Extraction asset checksum mismatch: {path}")


def _engine_and_preprocessing(stage, candidate, item) -> tuple[str, str]:
    if stage == "document":
        return candidate.partition("|")[0], "raw"
    if stage == "audio":
        return candidate.partition("|")[0], str(item.get("preprocessing", "raw"))
    return candidate, str(item.get("preprocessing", "raw"))


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


def _model_lock(stage: str) -> dict[str, dict[str, object]]:
    del stage
    return load_selected_model_lock(PROJECT_ROOT / "data/benchmarks/models/selected.json")


def _prepared_component_options(
    options: Mapping[str, object], model_lock: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    result = dict(options)
    selected = [options.get("image_engine"), options.get("audio_candidate")]
    for candidate in selected:
        if not candidate:
            continue
        candidate_name = str(candidate)
        entry = model_lock.get(LOCK_CANDIDATES.get(candidate_name, candidate_name), {})
        if not entry:
            raise RuntimeError(
                f"Selected component {candidate_name} is absent from the extraction model lock"
            )
        prefix = "image_" if candidate_name == options.get("image_engine") else "audio_"
        result.update({f"{prefix}{key}": value for key, value in _lock_paths(entry).items()})
        if candidate_name == options.get("image_engine"):
            result["image_revision"] = str(entry.get("revision", "unpinned"))
        else:
            result["audio_revision"] = str(entry.get("revision", "unpinned"))
    return result


def _manifest(stage: str, profile: str) -> Path:
    if profile == "smoke":
        return PROJECT_ROOT / "data/benchmarks/extraction/smoke.json"
    split = "validation" if profile == "standard" else "locked-test"
    return PROJECT_ROOT / f"data/benchmarks/extraction/{stage}-{split}.json"


def _lock_paths(entry: Mapping[str, object]) -> dict[str, object]:
    result = {
        str(key): value
        for key, value in entry.items()
        if str(key).endswith("_path") or str(key).endswith("_dir")
    }
    submodels = entry.get("submodels", [])
    if isinstance(submodels, Sequence) and not isinstance(submodels, (str, bytes)):
        for submodel in submodels:
            if isinstance(submodel, Mapping) and submodel.get("role") == "forced-aligner":
                result["aligner_model_path"] = str(submodel.get("model_path", ""))
    return result
