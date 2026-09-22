"""Execution and metric contracts specific to document extraction."""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from edumind.common.artifacts import atomic_write_json
from edumind.common.paths import PROJECT_ROOT
from experiments.benchmarks.common.contracts import SampleResult
from experiments.benchmarks.common.provenance import package_versions

from .metrics import (
    aggregate_evaluations,
    apply_official_metrics,
    score_document,
)
from .profiles import parse_document_profile


def evaluate_candidate(
    candidate,
    items,
    plan,
    model_lock,
    component_options,
    references,
    pipeline,
    extract_once,
    protocol,
):
    ordered = list(items)
    random.Random(plan.seed).shuffle(ordered)
    first_latency = _cold_latency(
        candidate, ordered[0], model_lock, component_options, protocol
    )
    for _ in range(plan.warmups):
        extract_once(candidate, ordered[0], model_lock, component_options, pipeline)

    samples, evaluations = [], []
    timing_rows: list[dict[str, object]] = []
    document_latencies, page_latencies = [], []
    group_latencies: dict[str, list[float]] = {}
    successful_pages = 0
    measured_seconds = 0.0
    for item in ordered:
        documents, latencies, successful_latencies = [], [], []
        failures: list[Exception] = []
        for repetition in range(plan.repetitions):
            started = time.perf_counter()
            try:
                document, latency = extract_once(
                    candidate, item, model_lock, component_options, pipeline
                )
                documents.append(document)
                latencies.append(latency)
                successful_latencies.append(latency)
                timing_rows.append(
                    {
                        "sample_id": str(item["id"]),
                        "repetition": repetition + 1,
                        "latency_seconds": latency,
                        "success": True,
                        "error": None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - retain failed sample rows
                failures.append(exc)
                failed_latency = time.perf_counter() - started
                latencies.append(failed_latency)
                timing_rows.append(
                    {
                        "sample_id": str(item["id"]),
                        "repetition": repetition + 1,
                        "latency_seconds": failed_latency,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        all_succeeded = not failures and len(documents) == plan.repetitions
        # Quality is intentionally all-or-nothing across measured repetitions.
        document = documents[0] if all_succeeded else None
        latency = float(np.median(latencies))
        measured_seconds += sum(latencies)
        # Successful attempts still contribute processed pages even when a different
        # repetition failed and quality is deliberately replaced by an empty output.
        successful_page_counts = [_processed_pages(item, value) for value in documents]
        if successful_latencies:
            complete_latency = float(np.median(successful_latencies))
            document_latencies.append(complete_latency)
        if successful_latencies and item["kind"] in {"image", "pdf"}:
            latency_per_page = [
                latency / pages
                for latency, pages in zip(
                    successful_latencies, successful_page_counts, strict=True
                )
                if pages
            ]
            if latency_per_page:
                page_latencies.append(float(np.median(latency_per_page)))
            successful_pages += sum(successful_page_counts)
        evaluation = score_document(
            item,
            document,
            reference=references[str(item["id"])],
            repeated_documents=documents if all_succeeded else (),
            failed=not all_succeeded,
            element_matching_threshold=protocol.element_matching_threshold,
            duplicate_content_threshold=protocol.duplicate_content_threshold,
        )
        evaluations.append(evaluation)
        if successful_latencies:
            for group in evaluation.groups:
                group_latencies.setdefault(group, []).append(complete_latency)
        samples.append(
            SampleResult(
                str(item["id"]),
                evaluation.metrics,
                latency,
                {
                    "kind": item.get("kind"),
                    "document_groups": list(evaluation.groups),
                    "document_family": item.get("document_family"),
                    "has_table": item.get("has_table"),
                    "has_formula": item.get("has_formula"),
                    "layout_difficulty": item.get("layout_difficulty"),
                    "candidate": candidate,
                    "warnings": len(document.warnings) if document else 0,
                    "failure": (
                        "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures)
                        if failures
                        else None
                    ),
                    "successful_repetitions": len(documents),
                    "attempted_repetitions": plan.repetitions,
                },
            )
        )

    apply_official_metrics(
        evaluations, timeout_seconds=protocol.evaluator_timeout_seconds
    )
    resamples = 0 if plan.profile == "smoke" else plan.bootstrap_resamples
    aggregate, intervals = aggregate_evaluations(
        evaluations,
        resamples=resamples,
        seed=plan.seed,
        confidence=protocol.confidence_level,
    )
    operational = {"first_item_latency_seconds": first_latency}
    if document_latencies:
        operational.update(
            {
                "p50_complete_document_latency_seconds": float(
                    np.quantile(document_latencies, 0.50)
                ),
                "p95_complete_document_latency_seconds": float(
                    np.quantile(document_latencies, 0.95)
                ),
            }
        )
    if page_latencies:
        operational.update(
            {
                "p50_warm_latency_per_page_seconds": float(
                    np.quantile(page_latencies, 0.50)
                ),
                "p95_warm_latency_per_page_seconds": float(
                    np.quantile(page_latencies, 0.95)
                ),
                "batch_pages_per_minute": 60.0
                * successful_pages
                / max(measured_seconds, 1e-9),
            }
        )
    if resamples:
        intervals.update(
            _latency_intervals(
                document_latencies,
                page_latencies,
                resamples,
                plan.seed,
                protocol.confidence_level,
            )
        )
    for group, values in sorted(group_latencies.items()):
        operational[f"{group}.p50_complete_document_latency_seconds"] = float(
            np.quantile(values, 0.50)
        )
        operational[f"{group}.p95_complete_document_latency_seconds"] = float(
            np.quantile(values, 0.95)
        )
        if resamples:
            intervals.update(
                {
                    name.replace("operational.", f"operational.{group}.", 1): interval
                    for name, interval in _latency_intervals(
                        values,
                        [],
                        resamples,
                        plan.seed,
                        protocol.confidence_level,
                    ).items()
                }
            )
    document_profile = parse_document_profile(candidate)
    engine = document_profile.requested_engine
    lock_entry = model_lock.get(document_profile.lock_candidate, {})
    executed_parser_options = {
        **protocol.parser_options(document_profile.runtime_engine),
        **document_profile.options,
    }
    parameters = {
        "engine": engine,
        "engine_revision": lock_entry.get("revision", "system"),
        "model_path": lock_entry.get("model_path", ""),
        "device": component_options["device"],
        "seed": plan.seed,
        "warmups": plan.warmups,
        "repetitions": plan.repetitions,
        "normalization": "none",
        "resolved_parser_options": executed_parser_options,
        "element_matching_threshold": protocol.element_matching_threshold,
        "duplicate_content_threshold": protocol.duplicate_content_threshold,
        "evaluator_timeout_seconds": protocol.evaluator_timeout_seconds,
        "model_cache_manifest_sha256": lock_entry.get(
            "model_cache_manifest_sha256", ""
        ),
        "prepared_components": lock_entry.get("prepared_components", []),
        "system_components": lock_entry.get("system_components", {}),
        "package_versions": package_versions(
            (
                "docling",
                "paddleocr",
                "paddlepaddle",
                "paddlex",
                "torch",
                "transformers",
                "onnxruntime",
                "easyocr",
                "rapidocr",
            )
        ),
    }
    parameters.update(
        {f"parser.{name}": value for name, value in executed_parser_options.items()}
    )
    if engine == "paddleocr-vl-1.6":
        parameters.update(
            {
                "paddle_cache_path": lock_entry.get("paddle_cache_path", ""),
                "paddle_cache_manifest_sha256": lock_entry.get(
                    "paddle_cache_manifest_sha256", ""
                ),
            }
        )
    for key, value in document_profile.factors.items():
        parameters[key] = value
    return (
        samples,
        operational,
        aggregate,
        parameters,
        intervals,
        {
            "samples": [
                {
                    "sample_id": sample.sample_id,
                    **dict(sample.metrics),
                    "latency_seconds": sample.latency_seconds,
                    **dict(sample.metadata),
                }
                for sample in samples
            ],
            "timings": timing_rows,
        },
    )


def validate_prepared_components(candidate, lock_entry, protocol) -> None:
    """Reject Docling configurations whose parser dependencies were not locked."""

    profile = parse_document_profile(candidate)
    if profile.requested_engine != "docling-standard":
        return
    options = {
        **protocol.parser_options(profile.runtime_engine),
        **profile.options,
    }
    required = {
        "layout",
        "tableformer",
        {
            "rapidocr": "rapidocr",
            "easyocr": "easyocr",
            "tesseract": "tesseract-cli",
        }[str(options["ocr_engine"])],
    }
    if bool(options["formula_enrichment"]):
        required.add("code_formula")
    raw_prepared = lock_entry.get("prepared_components", [])
    if not isinstance(raw_prepared, (list, tuple)):
        raise RuntimeError("Docling prepared_components must be a list")
    prepared = set(raw_prepared)
    missing = sorted(required - prepared)
    if missing:
        raise RuntimeError(
            f"Document candidate {candidate} lacks prepared components: "
            + ", ".join(missing)
        )


def directions_for(references, available):
    names = {
        "reliability.empty_output_rate",
        "reliability.structured_output_determinism",
        "reliability.candidate_failure_rate",
        "operational.first_item_latency_seconds",
        "operational.p50_complete_document_latency_seconds",
        "operational.p95_complete_document_latency_seconds",
        "operational.peak_process_tree_ram_mb",
        "operational.peak_vram_mb",
        "operational.peak_temporary_disk_mb",
    }
    capabilities = set().union(*(reference.capabilities for reference in references))
    if "text" in capabilities:
        names.update(
            {
                "text.content_precision",
                "text.content_recall",
                "text.content_f1",
                "text.character_error_rate",
                "text.word_error_rate",
                "reliability.duplicate_content_rate",
            }
        )
    if "pages" in capabilities:
        names.update(
            {
                "pages.page_coverage",
                "pages.page_content_f1",
                "pages.page_attribution_accuracy",
                "pages.duplicate_page_rate",
                "operational.p50_warm_latency_per_page_seconds",
                "operational.p95_warm_latency_per_page_seconds",
                "operational.batch_pages_per_minute",
            }
        )
    if capabilities & {"reading_order", "layout_boxes", "element_types", "hierarchy"}:
        names.update(
            {
                "layout.element_precision",
                "layout.element_recall",
                "layout.element_f1",
            }
        )
    if "reading_order" in capabilities:
        names.add("text.reading_order_accuracy")
    if "element_types" in capabilities:
        names.add("layout.element_type_accuracy")
    if "hierarchy" in capabilities:
        names.add("layout.hierarchy_accuracy")
    if "layout_boxes" in capabilities:
        names.add("layout.mean_bounding_box_iou")
    has_table_references = any(
        "tables" in reference.capabilities
        and any(element.kind.value == "table" for element in reference.elements)
        for reference in references
    )
    if "tables" in capabilities and has_table_references:
        names.update(name for name in available if name.startswith("tables.detection_"))
        names.update(
            {
                "tables.content_precision",
                "tables.content_recall",
                "tables.content_f1",
                "tables.teds",
                "tables.teds_s",
            }
        )
    has_formula_references = any(
        "formulas" in reference.capabilities
        and any(element.kind.value == "formula" for element in reference.elements)
        for reference in references
    )
    if "formulas" in capabilities and has_formula_references:
        names.update(
            name for name in available if name.startswith("formulas.detection_")
        )
        names.update({"formulas.recognition_similarity", "formulas.exact_match"})
    return {name: available[name] for name in available if name in names}


def primary_metrics(directions):
    preferred = (
        "text.content_f1",
        "text.reading_order_accuracy",
        "pages.page_content_f1",
        "layout.element_f1",
        "layout.element_type_accuracy",
        "layout.hierarchy_accuracy",
        "tables.detection_f1",
        "tables.content_f1",
        "tables.teds",
        "formulas.detection_f1",
        "formulas.exact_match",
    )
    return tuple(name for name in preferred if name in directions)


def required_metrics(directions):
    conditional = {
        "text.reading_order_accuracy",
        "pages.duplicate_page_rate",
        "pages.page_attribution_accuracy",
        "layout.element_type_accuracy",
        "layout.hierarchy_accuracy",
        "layout.mean_bounding_box_iou",
        "reliability.duplicate_content_rate",
        "operational.peak_vram_mb",
    }
    return tuple(name for name in directions if name not in conditional)


def paired_metrics(directions):
    pooled = {
        "layout.element_precision",
        "layout.element_recall",
        "layout.element_f1",
        "tables.detection_precision",
        "tables.detection_recall",
        "tables.detection_f1",
        "formulas.detection_precision",
        "formulas.detection_recall",
        "formulas.detection_f1",
    }
    return tuple(
        name
        for name in directions
        if name not in pooled and not name.startswith("operational.")
    )


def _cold_latency(candidate, item, model_lock, component_options, protocol):
    payload_path = (
        Path(os.environ.get("TEMP", ".")) / f"document-cold-worker-{os.getpid()}.json"
    )
    atomic_write_json(
        payload_path,
        {
            "candidate": candidate,
            "item": dict(item),
            "model_lock": model_lock,
            "component_options": dict(component_options),
            "protocol": protocol.meta.worker_payload(),
        },
    )
    started = time.perf_counter()
    try:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.benchmarks.extraction.document.cold_worker",
                    str(payload_path),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "no worker output").strip()
            raise RuntimeError(
                f"Fresh document worker failed: {detail[-2000:]}"
            ) from exc
    finally:
        payload_path.unlink(missing_ok=True)
    if "EDUMIND_FIRST_ITEM_COMPLETE" not in completed.stdout:
        raise RuntimeError("Fresh document worker completed without a timing result")
    return time.perf_counter() - started


def _latency_intervals(document_values, page_values, resamples, seed, confidence):
    result = {}
    rng = np.random.default_rng(seed)
    tail = (1.0 - confidence) / 2.0
    for suffix, values in (
        ("complete_document_latency_seconds", document_values),
        ("warm_latency_per_page_seconds", page_values),
    ):
        if len(values) < 2:
            continue
        observed = np.asarray(values, dtype=np.float64)
        draws = {0.50: [], 0.95: []}
        for _ in range(resamples):
            sample = observed[rng.integers(0, len(observed), len(observed))]
            for quantile, estimates in draws.items():
                estimates.append(float(np.quantile(sample, quantile)))
        for quantile, estimates in draws.items():
            result[f"operational.p{int(quantile * 100)}_{suffix}"] = {
                "estimate": float(np.quantile(observed, quantile)),
                "lower": float(np.quantile(estimates, tail)),
                "upper": float(np.quantile(estimates, 1.0 - tail)),
                "confidence": confidence,
            }
    return result


def _processed_pages(item, document):
    if document is None:
        return 0
    if item.get("kind") == "image":
        return 1
    count = document.metadata.get("page_count", 0)
    return (
        int(count)
        if count
        else len(
            {
                segment.page_number
                for segment in document.segments
                if segment.page_number
            }
        )
    )
