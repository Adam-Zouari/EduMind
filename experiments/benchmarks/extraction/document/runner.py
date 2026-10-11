"""Document attempt scoring; model execution belongs to the fresh worker."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from edumind.extraction import ExtractedDocument
from experiments.benchmarks.common.contracts import (
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.process import run_json_worker
from experiments.benchmarks.common.provenance import package_versions
from experiments.benchmarks.extraction.scoring import bootstrap_sources, source_id

from .metrics import aggregate_evaluations, apply_official_metrics, score_document
from .profiles import parse_document_profile


def evaluate_candidate(
    candidate, items, plan, model_lock, component_options, references, protocol
):
    result = run_json_worker(
        Path(__file__).with_name("worker.py"),
        {
            "candidate": candidate,
            "items": list(items),
            "profile": plan.profile,
            "model_lock": model_lock,
            "device": component_options["device"],
            "protocol": protocol.meta.worker_payload(),
        },
        device=str(component_options["device"]),
        prefix="edumind-document-",
        error_label=f"document worker {candidate}",
        timeout_seconds=protocol.preflight.worker_timeout_seconds,
    )
    attempts = result["attempts"]
    expected = {
        (str(item["id"]), rep)
        for item in items
        for rep in range(1, plan.repetitions + 1)
    }
    actual = [(str(row["sample_id"]), int(row["repetition"])) for row in attempts]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise CandidateExecutionError(
            "Incomplete document attempt inventory", artifacts={"timings": attempts}
        )
    evaluations, samples, latency_rows = [], [], []
    for item in items:
        rows = sorted(
            (row for row in attempts if str(row["sample_id"]) == str(item["id"])),
            key=lambda row: int(row["repetition"]),
        )
        documents = [
            ExtractedDocument.from_dict(row["document"]) if row["success"] else None
            for row in rows
        ]
        evaluation = score_document(
            item,
            documents[0],
            reference=references[str(item["id"])],
            repeated_documents=documents,
            failed=documents[0] is None,
            element_matching_threshold=protocol.element_matching_threshold,
            duplicate_content_threshold=protocol.duplicate_content_threshold,
        )
        evaluations.append(evaluation)
        successful = [float(row["latency_seconds"]) for row in rows if row["success"]]
        pages = item.get("physical_page_count")
        if item["kind"] == "image":
            pages = 1
        elif item["kind"] == "pdf" and pages is None:
            raise ValueError(
                "PDF physical page count must come from validated input metadata"
            )
        latency = float(np.median(successful)) if successful else None
        latency_rows.append(
            {
                "source_group_id": source_id(item),
                "latency": latency,
                "page_latency": latency / pages
                if latency is not None and pages
                else None,
                "successful_pages": (pages or 0) * len(successful),
                "attempt_seconds": sum(float(row["latency_seconds"]) for row in rows),
                "groups": evaluation.groups,
            }
        )
        samples.append(
            SampleResult(
                str(item["id"]),
                evaluation.metrics,
                latency,
                {
                    "kind": item["kind"],
                    "source_group_id": source_id(item),
                    "document_groups": list(evaluation.groups),
                    "metric_statuses": evaluation.statuses,
                    "first_attempt_success": documents[0] is not None,
                    "successful_repetitions": len(successful),
                    "attempted_repetitions": len(rows),
                    "warnings": list(rows[0]["document"].get("warnings", []))
                    if documents[0]
                    else [],
                    "failure": [row["error"] for row in rows if not row["success"]],
                },
            )
        )
    scorer_error = None
    try:
        apply_official_metrics(
            evaluations, timeout_seconds=protocol.evaluator_timeout_seconds
        )
    except RuntimeError as exc:
        scorer_error = str(exc)
        for evaluation in evaluations:
            for name in (
                "tables.teds",
                "tables.teds_s",
                "formulas.recognition_similarity",
                "formulas.exact_match",
            ):
                if evaluation.statuses[name]["reason"] == "evaluation_pending":
                    evaluation.statuses[name] = {
                        "status": "incomplete",
                        "reason": "evaluator_failed",
                    }
    resamples = 0 if plan.profile == "smoke" else plan.bootstrap_resamples
    aggregate, intervals = aggregate_evaluations(
        evaluations,
        resamples=resamples,
        seed=plan.seed,
        confidence=protocol.confidence_level,
        minimum_sources=protocol.minimum_ci_sources,
    )
    operational = dict(result["operational"])
    operational.update(_latency_statistics(latency_rows))
    if any(item["kind"] in {"image", "pdf"} for item in items):
        elapsed = float(result["measured_batch_seconds"])
        operational["batch_pages_per_minute"] = (
            60 * sum(row["successful_pages"] for row in latency_rows) / elapsed
            if elapsed > 0
            else None
        )
    intervals.update(
        {
            "operational." + name: bounds
            for name, bounds in bootstrap_sources(
                latency_rows,
                _latency_statistics,
                resamples=resamples,
                seed=plan.seed,
                confidence=protocol.confidence_level,
                minimum_sources=protocol.minimum_latency_ci_sources,
            ).items()
        }
    )
    profile = parse_document_profile(candidate)
    entry = model_lock.get(profile.lock_candidate, {})
    parameters = {
        **result["parameters"],
        "engine": profile.requested_engine,
        "engine_revision": entry.get("revision", "system"),
        "model_path": entry.get("model_path", ""),
        "element_matching_threshold": protocol.element_matching_threshold,
        "duplicate_content_threshold": protocol.duplicate_content_threshold,
        "evaluator_timeout_seconds": protocol.evaluator_timeout_seconds,
        "model_cache_manifest_sha256": entry.get("model_cache_manifest_sha256", ""),
        "prepared_components": entry.get("prepared_components", []),
        "system_components": entry.get("system_components", {}),
        "paddle_cache_manifest_sha256": entry.get("paddle_cache_manifest_sha256"),
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
                "rapidfuzz",
            )
        ),
    }
    artifacts = {
        "samples": [
            {
                "sample_id": sample.sample_id,
                **sample.metrics,
                **sample.metadata,
                "latency_seconds": sample.latency_seconds,
            }
            for sample in samples
        ],
        "timings": [
            {name: value for name, value in row.items() if name != "document"}
            for row in attempts
        ],
        "outputs": {"attempts": attempts},
        "resource_samples": result["resource_samples"],
        "gpu_identity": result.get("gpu_identity", {}),
        "placement": {
            "before": result.get("placement_before"),
            "after": result.get("placement_after"),
        },
    }
    if scorer_error:
        artifacts["evaluator_error"] = {"error": scorer_error}
        raise CandidateExecutionError(
            "Official scoring incomplete; saved extraction outputs can be rescored",
            samples=tuple(samples),
            operational=operational,
            metrics=aggregate,
            intervals=intervals,
            parameters=parameters,
            artifacts=artifacts,
        )
    return samples, operational, aggregate, parameters, intervals, artifacts


def _latency_statistics(rows):
    result = {}
    for label, key in (
        ("complete_document", "latency"),
        ("warm_latency_per_page", "page_latency"),
    ):
        values = [row[key] for row in rows if row[key] is not None]
        suffix = (
            label + "_latency_seconds"
            if label == "complete_document"
            else label + "_seconds"
        )
        for percentile in (50, 95):
            result[f"p{percentile}_{suffix}"] = (
                float(np.quantile(values, percentile / 100)) if values else None
            )
    return result


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
        "reliability.unexpected_empty_output_rate",
        "reliability.repeatability_success_rate",
        "reliability.attempt_failure_rate",
        "operational.cold_model_load_seconds",
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
                "pages.page_attribution_recall",
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
        names.add("text.reading_order_ned")
    if "element_types" in capabilities:
        names.add("layout.element_type_recall")
    if "hierarchy" in capabilities:
        names.add("layout.hierarchy_preservation_rate")
    if "layout_boxes" in capabilities:
        names.add("layout.mean_bounding_box_iou")
    if "tables" in capabilities:
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
    if "formulas" in capabilities:
        names.update(
            name for name in available if name.startswith("formulas.detection_")
        )
        names.update({"formulas.recognition_similarity", "formulas.exact_match"})
    return {name: available[name] for name in available if name in names}


def primary_metrics(directions):
    preferred = (
        "text.content_f1",
        "text.reading_order_ned",
        "pages.page_content_f1",
        "layout.element_f1",
        "layout.element_type_recall",
        "layout.hierarchy_preservation_rate",
        "tables.detection_f1",
        "tables.content_f1",
        "tables.teds",
        "formulas.detection_f1",
        "formulas.exact_match",
    )
    return tuple(name for name in preferred if name in directions)


def required_metrics(directions):
    return tuple(directions)


def paired_metrics(directions):
    return tuple(name for name in directions if not name.startswith("operational."))
