"""Shared evaluator for the seven direct extraction experiments."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from edumind.common.artifacts import sha256_file
from edumind.common.paths import PROJECT_ROOT
from edumind.extraction import ExtractionPipeline, ExtractionProfile, SourceKind
from edumind.extraction.normalization import normalize_text

from experiments.benchmarks.common.contracts import BenchmarkPlan, BenchmarkResult, SampleResult
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.common.document_metrics import structured_document_scores
from experiments.benchmarks.common.metrics import (
    character_error_rate,
    levenshtein,
    word_error_rate,
)
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.common.text_metrics import (
    block_scores,
    content_scores,
    reading_order_accuracy,
    timestamp_mae,
)
from experiments.benchmarks.prepare import load_extraction_model_lock

STAGES = {"image", "pdf", "docx", "audio", "video", "normalization", "routing"}


def run(
    stage: str,
    profile: str,
    candidates: tuple[str, ...],
    *,
    manifest_path: Path | None = None,
    no_mlflow: bool = False,
    component_options: Mapping[str, object] | None = None,
) -> BenchmarkResult:
    if stage not in STAGES:
        raise ValueError(f"Unknown extraction stage: {stage}")
    manifest = load_manifest(manifest_path or _manifest(stage, profile))
    selected = [
        item
        for item in manifest.samples
        if stage == "normalization"
        or (stage == "routing" and item.get("kind") == "pdf")
        or item.get("kind") == stage
    ]
    if not selected:
        raise ValueError(f"Manifest {manifest.name} has no samples for {stage}")
    minimum = {
        "image": 24,
        "pdf": 12,
        "routing": 12,
        "docx": 9,
        "audio": 18,
        "video": 6,
        "normalization": 40,
    }
    if profile in {"standard", "full"} and len(selected) < minimum[stage]:
        raise ValueError(
            f"{stage} {profile} requires at least {minimum[stage]} frozen samples; "
            f"manifest contains {len(selected)}"
        )
    if stage == "normalization" and profile in {"standard", "full"}:
        missing_family = [item.get("id") for item in selected if not item.get("document_family")]
        if missing_family:
            raise ValueError(
                "Authoritative normalization cases require document_family for split isolation: "
                + ", ".join(str(value) for value in missing_family[:5])
            )
    if stage != "normalization":
        _validate_assets(
            selected,
            require_checksums=True,
            require_provenance=profile in {"standard", "full"},
        )
    component_options = dict(component_options or {})
    plan = BenchmarkPlan(
        "extraction",
        stage,
        profile,
        manifest.name,
        candidates,
        repetitions=1 if profile == "smoke" else 3,
        bootstrap_resamples=500 if profile == "smoke" else 10_000,
        settings=component_options,
    )
    model_lock = _model_lock(stage, candidates, component_options)
    prepared_component_options = _prepared_component_options(component_options, model_lock)

    def evaluate(candidate: str):
        pipeline = ExtractionPipeline()
        samples: list[SampleResult] = []
        latencies: list[float] = []
        ordered = list(selected)
        random.Random(plan.seed).shuffle(ordered)
        cold_load_seconds = None
        if stage != "normalization":
            _, _, cold_load_seconds = _extract_once(
                stage, candidate, ordered[0], model_lock, prepared_component_options, pipeline
            )
            for _ in range(plan.warmups):
                _extract_once(
                    stage, candidate, ordered[0], model_lock, prepared_component_options, pipeline
                )
        for item in ordered:
            outputs = [
                _extract_once(
                    stage, candidate, item, model_lock, prepared_component_options, pipeline
                )
                for _ in range(plan.repetitions)
            ]
            hypothesis, document, _ = outputs[0]
            item_latencies = [value[2] for value in outputs]
            latency = float(np.median(item_latencies))
            latencies.extend(item_latencies)
            metrics = _metrics(stage, str(item["reference"]), hypothesis, item, document)
            metrics["determinism"] = float(all(value[0] == hypothesis for value in outputs))
            samples.append(
                SampleResult(
                    str(item["id"]),
                    metrics,
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
        durations = [float(item["duration_seconds"]) for item in selected if item.get("duration_seconds")]
        if durations and len(durations) == len(selected):
            operational["real_time_factor"] = sum(latencies) / (sum(durations) * plan.repetitions)
        return samples, operational

    return run_benchmark(
        plan,
        evaluate,
        dataset_checksum=manifest.fingerprint,
        directions=_directions(stage),
        gates={"determinism": ("max", 1.0)},
        revisions={name: str(value.get("revision", "")) for name, value in model_lock.items()},
        no_mlflow=no_mlflow,
    )


def _metrics(stage, reference, hypothesis, item, document) -> dict[str, float]:
    scores = content_scores(reference, hypothesis)
    scores.update(block_scores(reference, hypothesis))
    scores.update(
        {
            "character_error_rate": character_error_rate(reference, hypothesis),
            "word_error_rate": word_error_rate(reference, hypothesis),
            "reading_order_accuracy": reading_order_accuracy(reference, hypothesis),
            "empty_output_rate": float(not hypothesis.strip()),
            "word_accuracy": max(0.0, 1.0 - min(1.0, word_error_rate(reference, hypothesis))),
            "duplicate_text_rate": _duplicate_line_rate(hypothesis),
        }
    )
    if stage in {"image", "pdf", "routing"}:
        scores.update(_page_scores(item, document))
    if stage in {"image", "pdf", "docx", "routing"} and document is not None:
        scores.update(structured_document_scores(item.get("reference_elements"), document))
    if stage in {"audio", "video"}:
        reference_timestamps = item.get("reference_timestamps")
        if isinstance(reference_timestamps, Sequence) and not isinstance(reference_timestamps, str):
            predicted = [
                segment.timestamp_start
                for segment in document.segments
                if segment.timestamp_start is not None
            ]
            if reference_timestamps and len(reference_timestamps) == len(predicted):
                scores["timestamp_mae_seconds"] = timestamp_mae(
                    [float(value) for value in reference_timestamps], predicted
                )
            scores["timestamp_alignment_coverage"] = min(
                len(reference_timestamps), len(predicted)
            ) / max(1, len(reference_timestamps), len(predicted))
        reference_segments = item.get("reference_segments")
        if isinstance(reference_segments, Sequence) and not isinstance(reference_segments, str):
            expected_boundaries = [
                float(segment[key])
                for segment in reference_segments
                if isinstance(segment, Mapping)
                for key in ("start", "end")
                if segment.get(key) is not None
            ]
            predicted_boundaries = [
                value
                for segment in document.segments
                for value in (segment.timestamp_start, segment.timestamp_end)
                if value is not None
            ]
            if expected_boundaries and len(expected_boundaries) == len(predicted_boundaries):
                scores["segment_boundary_mae_seconds"] = timestamp_mae(
                    expected_boundaries, predicted_boundaries
                )
    if stage == "video":
        audio_count = int(document.metadata.get("audio_segment_count", 0))
        audio_text = "\n".join(segment.text for segment in document.segments[:audio_count])
        visual_segments = document.segments[audio_count:]
        visual_text = "\n".join(segment.text for segment in visual_segments)
        transcript_reference = str(item.get("reference_transcript", reference))
        scores["transcript_word_error_rate"] = word_error_rate(
            transcript_reference, audio_text
        )
        scores["complete_content_recall"] = scores["content_recall"]
        if "reference_visual_text" in item:
            visual_reference = item["reference_visual_text"]
            if isinstance(visual_reference, Sequence) and not isinstance(visual_reference, str):
                visual_reference = "\n".join(str(value) for value in visual_reference)
            visual_scores = content_scores(str(visual_reference), visual_text)
            scores.update(
                {
                    "visual_text_precision": visual_scores["content_precision"],
                    "visual_text_recall": visual_scores["content_recall"],
                    "visual_text_f1": visual_scores["content_f1"],
                    "duplicate_visual_text_rate": _duplicate_line_rate(visual_text),
                }
            )
        visual_timestamps = item.get("reference_visual_timestamps")
        predicted_visual_timestamps = [
            segment.timestamp_start
            for segment in visual_segments
            if segment.timestamp_start is not None
        ]
        if (
            isinstance(visual_timestamps, Sequence)
            and not isinstance(visual_timestamps, str)
            and visual_timestamps
            and len(visual_timestamps) == len(predicted_visual_timestamps)
        ):
            scores["audio_visual_alignment_mae_seconds"] = timestamp_mae(
                [float(value) for value in visual_timestamps], predicted_visual_timestamps
            )
    if stage == "routing":
        oracle = str(item.get("oracle_engine", ""))
        selected_engine = document.profile.engine
        scores["router_selection_accuracy"] = float(not oracle or selected_engine == oracle)
        oracle_quality = float(item.get("oracle_quality", 1.0))
        scores["quality_regret"] = max(0.0, oracle_quality - scores["content_f1"])
        scores["fallback_success_rate"] = float(bool(hypothesis.strip()))
    if stage == "normalization":
        observed = str(item["observed"])
        before_distance = levenshtein(list(reference), list(observed))
        after_distance = levenshtein(list(reference), list(hypothesis))
        changes = levenshtein(list(observed), list(hypothesis))
        corrected = max(0, before_distance - after_distance)
        scores["content_preservation_recall"] = scores["content_recall"]
        scores["corruption_removal_recall"] = corrected / before_distance if before_distance else 1.0
        scores["corruption_removal_precision"] = corrected / changes if changes else float(not before_distance)
        precision, recall = (
            scores["corruption_removal_precision"],
            scores["corruption_removal_recall"],
        )
        scores["corruption_removal_f1"] = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return scores


def _extract_once(stage, candidate, item, model_lock, component_options, pipeline):
    started = time.perf_counter()
    if stage == "normalization":
        return normalize_text(str(item["observed"]), candidate), None, time.perf_counter() - started
    kind = SourceKind(str(item["kind"]))
    engine, preprocessing = _engine_and_preprocessing(stage, candidate, item)
    lock_name = (
        str(component_options.get("audio_candidate", "faster-whisper-base-int8"))
        if stage == "video"
        else engine
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
            normalization=str(item.get("normalization", "conservative")),
            routing=candidate if stage == "routing" else "direct",
            device=_device(stage, candidate, component_options, item),
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


def _duplicate_line_rate(text: str) -> float:
    lines = [" ".join(line.casefold().split()) for line in text.splitlines() if line.strip()]
    return (len(lines) - len(set(lines))) / len(lines) if lines else 0.0


def _page_scores(item: Mapping[str, object], document) -> dict[str, float]:
    raw_references = item.get("reference_page_texts")
    if isinstance(raw_references, Sequence) and not isinstance(raw_references, (str, bytes)):
        references = [str(value) for value in raw_references]
    elif item.get("kind") == "image":
        references = [str(item.get("reference", ""))]
    else:
        expected = max(1, int(item.get("reference_pages", 1)))
        pages = {
            segment.page_number
            for segment in document.segments
            if segment.page_number is not None and segment.text.strip()
        }
        return {"page_coverage": min(1.0, len(pages) / expected)}

    predicted = {
        page: "\n".join(
            segment.text
            for segment in document.segments
            if segment.page_number == page and segment.text.strip()
        )
        for page in range(1, len(references) + 1)
    }
    populated = [value for value in predicted.values() if value.strip()]
    correct_attribution = 0
    same_page_scores: list[float] = []
    for page, reference in enumerate(references, 1):
        hypothesis = predicted[page]
        same_page_scores.append(content_scores(reference, hypothesis)["content_f1"])
        if not hypothesis.strip():
            continue
        similarities = [content_scores(value, hypothesis)["content_f1"] for value in references]
        if int(np.argmax(similarities)) + 1 == page:
            correct_attribution += 1
    normalized = [" ".join(value.casefold().split()) for value in populated]
    duplicate_rate = (
        (len(normalized) - len(set(normalized))) / len(normalized) if normalized else 0.0
    )
    coverage = len(populated) / len(references)
    return {
        "page_coverage": coverage,
        "page_content_f1": float(np.mean(same_page_scores)),
        "page_attribution_accuracy": correct_attribution / len(references),
        "missing_page_rate": 1.0 - coverage,
        "duplicate_page_rate": duplicate_rate,
    }

def _directions(stage: str) -> dict[str, str]:
    result = {
        "character_error_rate": "min",
        "word_error_rate": "min",
        "content_f1": "max",
        "reading_order_accuracy": "max",
        "operational.p95_latency_seconds": "min",
        "operational.peak_ram_mb": "min",
    }
    if stage in {"image", "pdf", "routing"}:
        result.update(
            {
                "page_coverage": "max",
                "page_content_f1": "max",
                "page_attribution_accuracy": "max",
                "missing_page_rate": "min",
                "duplicate_page_rate": "min",
                "block_structure_f1": "max",
                "table_detection_f1": "max",
                "table_content_f1": "max",
                "table_structure_f1": "max",
                "formula_detection_f1": "max",
                "formula_latex_similarity": "max",
            }
        )
    if stage == "docx":
        result.update(
            {
                "content_precision": "max",
                "content_recall": "max",
                "block_structure_f1": "max",
                "table_detection_f1": "max",
                "table_content_f1": "max",
                "table_structure_f1": "max",
                "formula_detection_f1": "max",
                "formula_latex_similarity": "max",
            }
        )
    if stage == "audio":
        result.update(
            {
                "missing_text_rate": "min",
                "hallucinated_text_rate": "min",
                "timestamp_mae_seconds": "min",
                "segment_boundary_mae_seconds": "min",
                "timestamp_alignment_coverage": "max",
                "operational.real_time_factor": "min",
            }
        )
    if stage == "video":
        result.update(
            {
                "transcript_word_error_rate": "min",
                "visual_text_precision": "max",
                "visual_text_recall": "max",
                "visual_text_f1": "max",
                "audio_visual_alignment_mae_seconds": "min",
                "complete_content_recall": "max",
                "operational.real_time_factor": "min",
            }
        )
    if stage == "routing":
        result.update({"router_selection_accuracy": "max", "quality_regret": "min"})
    if stage == "normalization":
        result.update(
            {"content_preservation_recall": "max", "corruption_removal_f1": "max"}
        )
    return result


def _engine_and_preprocessing(stage, candidate, item) -> tuple[str, str]:
    if stage == "image":
        engine, separator, preprocessing = candidate.partition("|")
        return engine, preprocessing if separator else "raw"
    if stage == "routing":
        if candidate == "always-native":
            return "pypdf", "raw"
        if candidate == "always-ocr":
            return "ocr-pdf", "document"
        if candidate == "document-router":
            layout = str(item.get("layout", "digital"))
            if layout == "digital":
                return "pypdf", "raw"
            if layout == "scanned":
                return "ocr-pdf", "document"
            return "hybrid-pdf", "document"
        if candidate == "page-hybrid-router":
            return "hybrid-pdf", "document"
        raise ValueError(f"Unknown routing candidate: {candidate}")
    if stage == "audio":
        return candidate.partition("|")[0], str(item.get("preprocessing", "raw"))
    return candidate, str(item.get("preprocessing", "raw"))


def _device(stage, candidate, component_options, item) -> str:
    if item.get("device"):
        return str(item["device"])
    audio_candidate = (
        str(component_options.get("audio_candidate", "faster-whisper-base-int8"))
        if stage == "video"
        else candidate.partition("|")[0]
    )
    if stage in {"audio", "video"} and audio_candidate in {
        "openai-whisper-small-en",
        "faster-whisper-small-int8",
        "faster-whisper-small-float16",
        "faster-whisper-turbo-int8",
        "distil-whisper-large-v3.5",
        "parakeet-tdt-0.6b-v3",
        "canary-qwen-2.5b",
    }:
        return "cuda"
    return "cpu"


def _options(
    item: Mapping[str, object],
    lock_entry: Mapping[str, object],
    component_options: Mapping[str, object],
    candidate: str,
) -> dict[str, object]:
    raw = item.get("options", {})
    result = dict(raw) if isinstance(raw, Mapping) else {}
    result.update(component_options)
    if "|vad=" in candidate:
        result["vad"] = candidate.endswith("vad=on")
    result.update(
        {key: value for key, value in lock_entry.items() if key.endswith("_path") or key.endswith("_dir")}
    )
    return result


def _model_lock(
    stage: str,
    candidates: tuple[str, ...],
    component_options: Mapping[str, object],
) -> dict[str, dict[str, str]]:
    prepared_engines = (
        "docling",
        "paddleocr",
        "pp-structure",
        "glm-ocr",
        "mineru",
        "olmocr",
    )
    needs_lock = stage in {"audio", "video"} or any(
        name.startswith(prepared_engines) for name in candidates
    ) or str(component_options.get("image_engine", "")).startswith(prepared_engines)
    if not needs_lock:
        return {}
    return load_extraction_model_lock(PROJECT_ROOT / "data/benchmarks/models/extraction.json")


def _prepared_component_options(
    options: Mapping[str, object], model_lock: Mapping[str, Mapping[str, str]]
) -> dict[str, object]:
    result = dict(options)
    selected = [options.get("image_engine"), options.get("audio_candidate")]
    for candidate in selected:
        if not candidate:
            continue
        candidate_name = str(candidate)
        entry = model_lock.get(candidate_name, {})
        if candidate_name == "tesseract-5" and not entry:
            result["image_revision"] = "5"
            continue
        if not entry:
            raise RuntimeError(
                f"Selected component {candidate_name} is absent from the extraction model lock"
            )
        result.update(
            {
                key: value
                for key, value in entry.items()
                if key.endswith("_path") or key.endswith("_dir")
            }
        )
        if candidate_name == options.get("image_engine"):
            result["image_revision"] = str(entry.get("revision", "unpinned"))
    return result


def _manifest(stage: str, profile: str) -> Path:
    if profile == "smoke":
        return PROJECT_ROOT / "data/benchmarks/extraction/smoke.json"
    split = "validation" if profile == "standard" else "locked-test"
    return PROJECT_ROOT / f"data/benchmarks/extraction/{stage}-{split}.json"
