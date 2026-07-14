"""Modality extraction, normalization, and routing benchmark runner."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from edumind.common.config import load_settings
from edumind.common.paths import PROJECT_ROOT
from edumind.extraction import (
    ExtractedDocument,
    ExtractionPipeline,
    ExtractionProfile,
    ExtractionRequest,
    SourceKind,
)
from edumind.extraction.extractors.base import build_document
from edumind.extraction.normalization import normalize_text
from edumind.extraction.registry import ExtractorRegistration, ExtractorRegistry

from .contracts import BenchmarkPlan, BenchmarkResult, SampleResult
from .datasets import load_manifest
from .harness import BenchmarkHarness
from .metrics import character_error_rate, word_error_rate
from .prepare import load_extraction_model_lock
from .registry import candidates_for


def run_extraction_stage(
    stage: str,
    profile: str,
    *,
    artifact_root: Path | None = None,
    manifest_path: Path | None = None,
) -> BenchmarkResult:
    if stage not in {"image", "pdf", "docx", "audio", "video", "normalization", "routing"}:
        raise ValueError(f"Unsupported extraction benchmark stage: {stage}")
    settings = load_settings()
    manifest = load_manifest(
        manifest_path
        or (
            PROJECT_ROOT / "data/benchmarks/extraction/smoke.json"
            if profile == "smoke"
            else PROJECT_ROOT / f"data/benchmarks/extraction/{stage}-validation.json"
        )
    )
    base_candidates = candidates_for("extraction", stage)
    model_lock = (
        load_extraction_model_lock(PROJECT_ROOT / "data/benchmarks/models/extraction.json")
        if profile != "smoke" and stage in {"image", "audio", "video"}
        else {}
    )
    candidates = (
        tuple(
            f"{engine}|{preprocessing}"
            for engine in base_candidates
            for preprocessing in ("raw", "document", "photo")
        )
        if stage == "image" and profile in {"standard", "full"}
        else base_candidates
    )
    plan = BenchmarkPlan(
        "extraction",
        stage,
        profile,
        manifest.name,
        candidates,
        bootstrap_resamples=min(500, settings.benchmark.bootstrap_resamples)
        if profile == "smoke"
        else settings.benchmark.bootstrap_resamples,
    )
    harness = BenchmarkHarness(
        artifact_root or settings.benchmark.artifact_directory,
        tracking_uri=settings.benchmark.tracking_uri,
    )
    pipeline = (
        ExtractionPipeline(settings=settings, registry=_smoke_registry())
        if profile == "smoke"
        else ExtractionPipeline(settings=settings)
    )

    def evaluate(candidate: str):
        samples: list[SampleResult] = []
        latencies: list[float] = []
        selected = [
            item
            for item in manifest.samples
            if stage in {"normalization", "routing"} or item.get("kind") == stage
        ]
        random.Random(plan.seed).shuffle(selected)
        for item in selected:
            started = time.perf_counter()
            reference = str(item["reference"])
            if stage == "normalization":
                normalization = candidate
                hypothesis = normalize_text(str(item["observed"]), normalization)
            else:
                if profile == "smoke":
                    engine = "smoke-fixture"
                    preprocessing = "raw"
                elif stage == "image":
                    engine, preprocessing = candidate.split("|", 1)
                else:
                    engine = _routing_engine(candidate, item) if stage == "routing" else candidate
                    preprocessing = str(item.get("preprocessing", "raw"))
                kind = SourceKind(str(item["kind"]))
                raw_options = item.get("options")
                resolved_options = dict(raw_options) if isinstance(raw_options, Mapping) else {}
                lock_candidate = "faster-whisper-base-int8" if stage == "video" else engine
                lock_entry = model_lock.get(lock_candidate, {})
                resolved_options.update(
                    {
                        key: value
                        for key, value in lock_entry.items()
                        if key.endswith("_path") or key.endswith("_dir")
                    }
                )
                extraction_profile = ExtractionProfile(
                    f"benchmark-{candidate}",
                    engine,
                    str(lock_entry.get("revision", item.get("engine_revision", "system"))),
                    preprocessing=preprocessing,
                    normalization=str(item.get("normalization", "conservative")),
                    routing=candidate if stage == "routing" else "direct",
                    options=resolved_options,
                )
                hypothesis = pipeline.extract(
                    PROJECT_ROOT / str(item["source_path"]),
                    source_kind=kind,
                    profile=extraction_profile,
                    use_cache=False,
                ).text
            latency = time.perf_counter() - started
            latencies.append(latency)
            cer = character_error_rate(reference, hypothesis)
            wer = word_error_rate(reference, hypothesis)
            samples.append(
                SampleResult(
                    str(item["id"]),
                    {
                        "character_error_rate": cer,
                        "word_error_rate": wer,
                        "page_coverage": float(bool(hypothesis.strip())),
                        "content_preservation_recall": max(0.0, 1.0 - cer),
                        "empty_output_rate": float(not hypothesis.strip()),
                        "duplicate_text_rate": 0.0,
                    },
                    latency,
                    {"kind": item.get("kind"), "smoke_fake_engine": profile == "smoke"},
                )
            )
        return samples, {
            "p50_latency_seconds": float(np.median(latencies)),
            "p95_latency_seconds": float(np.quantile(latencies, 0.95)),
            "peak_ram_mb": 0.0,
        }

    return harness.run(
        plan,
        evaluate,
        dataset_checksum=manifest.checksum,
        directions={
            "character_error_rate": "min",
            "word_error_rate": "min",
            "page_coverage": "max",
            "content_preservation_recall": "max",
            "operational.p95_latency_seconds": "min",
        },
    )


class _SmokeFixtureExtractor:
    """Deterministic fake model that still exercises the complete production pipeline."""

    name = "smoke-fixture"
    revision = "fixture-v1"
    supported_kinds = frozenset(SourceKind)

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        observed = request.source_path.read_text(encoding="utf-8")
        pages: list[int | None] | None = (
            [1] if kind in {SourceKind.IMAGE, SourceKind.PDF, SourceKind.DOCX} else None
        )
        timestamps = [(0.0, 1.0)] if kind in {SourceKind.AUDIO, SourceKind.VIDEO} else None
        return build_document(
            request,
            kind,
            request.profile,
            [observed],
            pages=pages,
            timestamps=timestamps,
            metadata={"smoke_fake_engine": True},
        )


def _smoke_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    registry.register(
        ExtractorRegistration(
            "smoke-fixture",
            frozenset(SourceKind),
            _SmokeFixtureExtractor,
            "",
            "smoke",
        )
    )
    return registry


def _routing_engine(candidate: str, sample: Mapping[str, object]) -> str:
    if candidate == "always-native":
        return "pypdf"
    if candidate == "always-ocr":
        return "hybrid-pdf"
    if candidate in {"document-router", "page-hybrid-router"}:
        return (
            "hybrid-pdf"
            if sample.get("layout") in {"scanned", "mixed", "broken-encoding"}
            else "pypdf"
        )
    raise ValueError(f"Unknown routing candidate: {candidate}")
