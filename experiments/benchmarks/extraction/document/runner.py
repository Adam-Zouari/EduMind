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

from .metrics import aggregate_evaluations, load_reference, score_document


def evaluate_candidate(
    candidate, items, plan, model_lock, component_options, pipeline, extract_once
):
    ordered = list(items)
    random.Random(plan.seed).shuffle(ordered)
    first_latency = _cold_latency(candidate, ordered[0], model_lock, component_options)
    for _ in range(plan.warmups):
        extract_once("document", candidate, ordered[0], model_lock, component_options, pipeline)

    samples, evaluations = [], []
    document_latencies, page_latencies = [], []
    group_latencies: dict[str, list[float]] = {}
    successful_pages = 0
    measured_seconds = 0.0
    for item in ordered:
        documents, latencies, successful_latencies = [], [], []
        failure = None
        for _ in range(plan.repetitions):
            started = time.perf_counter()
            try:
                _, document, latency = extract_once(
                    "document", candidate, item, model_lock, component_options, pipeline
                )
                documents.append(document)
                latencies.append(latency)
                successful_latencies.append(latency)
            except Exception as exc:  # failed inputs remain explicit sample rows
                failure = exc
                latencies.append(time.perf_counter() - started)
                break
        document = documents[0] if documents else None
        latency = float(np.median(latencies))
        measured_seconds += sum(latencies)
        pages = _processed_pages(item, document)
        if successful_latencies:
            complete_latency = float(np.median(successful_latencies))
            document_latencies.append(complete_latency)
        if successful_latencies and pages and item["kind"] in {"image", "pdf"}:
            page_latencies.append(complete_latency / pages)
            successful_pages += pages * len(documents)
        evaluation = score_document(
            item, document, repeated_documents=documents, failed=failure is not None
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
                    "failure": f"{type(failure).__name__}: {failure}" if failure else None,
                },
            )
        )

    resamples = 0 if plan.profile == "smoke" else plan.bootstrap_resamples
    aggregate, intervals = aggregate_evaluations(
        evaluations, resamples=resamples, seed=plan.seed
    )
    operational = {
        "first_item_latency_seconds": first_latency,
        "p50_complete_document_latency_seconds": float(np.quantile(document_latencies, 0.50)),
        "p95_complete_document_latency_seconds": float(np.quantile(document_latencies, 0.95)),
    }
    if page_latencies:
        operational.update(
            {
                "p50_warm_latency_per_page_seconds": float(np.quantile(page_latencies, 0.50)),
                "p95_warm_latency_per_page_seconds": float(np.quantile(page_latencies, 0.95)),
                "batch_pages_per_minute": 60.0 * successful_pages / max(measured_seconds, 1e-9),
            }
        )
    if resamples:
        intervals.update(
            _latency_intervals(document_latencies, page_latencies, resamples, plan.seed)
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
                        values, [], resamples, plan.seed
                    ).items()
                }
            )
    parameters = {}
    for factor in candidate.split("|")[1:]:
        key, value = factor.split("=", 1)
        parameters[key] = value
    return samples, operational, aggregate, parameters, intervals


def directions_for(items, document_kind, available):
    names = {
        "text.content_precision", "text.content_recall", "text.content_f1",
        "text.character_error_rate", "text.word_error_rate",
        "reliability.empty_output_rate", "reliability.duplicate_content_rate",
        "reliability.structured_output_determinism", "reliability.candidate_failure_rate",
        "operational.first_item_latency_seconds",
        "operational.p50_complete_document_latency_seconds",
        "operational.p95_complete_document_latency_seconds",
        "operational.peak_process_tree_ram_mb", "operational.peak_temporary_disk_mb",
    }
    if document_kind in {"image", "pdf"}:
        names.update(
            {
                "pages.page_coverage", "pages.page_content_f1",
                "pages.page_attribution_accuracy", "pages.duplicate_page_rate",
                "layout.mean_bounding_box_iou",
                "operational.p50_warm_latency_per_page_seconds",
                "operational.p95_warm_latency_per_page_seconds",
                "operational.batch_pages_per_minute",
            }
        )
    references = [element for item in items for element in load_reference(item).elements]
    if any(element.kind.value not in {"table", "formula"} for element in references):
        names.update(
            {
                "text.reading_order_accuracy", "layout.element_precision",
                "layout.element_recall", "layout.element_f1",
                "layout.element_type_accuracy", "layout.hierarchy_accuracy",
            }
        )
    has_tables = any(element.kind.value == "table" for element in references)
    if has_tables:
        names.update(name for name in available if name.startswith("tables.detection_"))
        names.update({"tables.content_f1", "tables.structure_score"})
    has_formulas = any(element.kind.value == "formula" for element in references)
    if has_formulas:
        names.update(name for name in available if name.startswith("formulas.detection_"))
        names.update({"formulas.recognition_similarity", "formulas.exact_match"})
    return {name: available[name] for name in available if name in names}


def primary_metrics(directions):
    preferred = (
        "text.content_f1", "text.reading_order_accuracy", "pages.page_content_f1",
        "layout.element_f1", "layout.element_type_accuracy", "layout.hierarchy_accuracy",
        "tables.detection_f1", "tables.content_f1", "tables.structure_score",
        "formulas.detection_f1", "formulas.exact_match",
    )
    return tuple(name for name in preferred if name in directions)


def required_metrics(directions):
    conditional = {
        "text.reading_order_accuracy", "pages.duplicate_page_rate",
        "pages.page_attribution_accuracy",
        "layout.element_type_accuracy", "layout.hierarchy_accuracy",
        "layout.mean_bounding_box_iou", "reliability.duplicate_content_rate",
        "operational.peak_vram_mb",
    }
    return tuple(name for name in directions if name not in conditional)


def paired_metrics(directions):
    pooled = {
        "layout.element_precision", "layout.element_recall", "layout.element_f1",
        "tables.detection_precision", "tables.detection_recall", "tables.detection_f1",
        "formulas.detection_precision", "formulas.detection_recall",
        "formulas.detection_f1",
    }
    return tuple(
        name
        for name in directions
        if name not in pooled and not name.startswith("operational.")
    )


def _cold_latency(candidate, item, model_lock, component_options):
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
        },
    )
    started = time.perf_counter()
    try:
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("cold_worker.py")),
                    str(payload_path),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "no worker output").strip()
            raise RuntimeError(f"Fresh document worker failed: {detail[-2000:]}") from exc
    finally:
        payload_path.unlink(missing_ok=True)
    if "EDUMIND_FIRST_ITEM_COMPLETE" not in completed.stdout:
        raise RuntimeError("Fresh document worker completed without a timing result")
    return time.perf_counter() - started


def _latency_intervals(document_values, page_values, resamples, seed):
    result = {}
    rng = np.random.default_rng(seed)
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
            for quantile in draws:
                draws[quantile].append(float(np.quantile(sample, quantile)))
        for quantile, estimates in draws.items():
            result[f"operational.p{int(quantile * 100)}_{suffix}"] = {
                "estimate": float(np.quantile(observed, quantile)),
                "lower": float(np.quantile(estimates, 0.025)),
                "upper": float(np.quantile(estimates, 0.975)),
                "confidence": 0.95,
            }
    return result


def _processed_pages(item, document):
    if document is None:
        return 0
    if item.get("kind") == "image":
        return 1
    count = document.metadata.get("page_count", 0)
    return int(count) if count else len(
        {segment.page_number for segment in document.segments if segment.page_number}
    )
