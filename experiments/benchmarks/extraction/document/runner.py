"""Document attempt scoring; model execution belongs to the fresh worker."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from experiments.benchmarks.common.contracts import (
    CandidateExecutionError,
    SampleResult,
)
from experiments.benchmarks.common.process import (
    WorkerMeasurementError,
    numeric_payload,
    run_json_worker,
)
from experiments.benchmarks.common.provenance import package_versions
from experiments.benchmarks.extraction.scoring import bootstrap_sources, source_id

from .metrics import (
    _document_fingerprint,
    aggregate_evaluations,
    apply_official_metrics,
    score_document,
)
from .profiles import parse_document_profile
from .worker import validate_document_payload


def evaluate_candidate(
    candidate, items, plan, model_lock, component_options, references, protocol
):
    result = {}
    try:
        try:
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
                require_vram_measurement=(
                    component_options["device"] == "cuda"
                    and candidate != "docling-standard-native"
                ),
                telemetry_interval_seconds=protocol.preflight.telemetry_interval_seconds,
                poll_interval_seconds=protocol.preflight.poll_interval_seconds,
                timeout_seconds=protocol.preflight.worker_timeout_seconds,
            )
        except WorkerMeasurementError as exc:
            result = exc.resource_evidence.get("worker_result")
            if not isinstance(result, Mapping) or "attempts" not in result:
                raise
            result = dict(result)
            result["execution_error"] = {
                "phase": "supervision",
                "error": str(exc),
                "worker_execution_error": result.get("execution_error"),
            }
            result["resource_evidence"] = {
                "worker": result.get("resource_evidence", {}),
                "supervisor": {
                    name: value
                    for name, value in exc.resource_evidence.items()
                    if name != "worker_result"
                },
            }
        attempts, decoded = _attempt_inventory(result, items, plan.repetitions)
    except CandidateExecutionError as exc:
        exc.parameters.setdefault("measurement.resource_scope", "partial")
        exc.artifacts.setdefault(
            "validation_report",
            {
                "status": "failed",
                "reason_code": "missing_worker_result",
                "scheduled_attempt_count": len(items) * plan.repetitions,
                "errors": [str(exc)],
            },
        )
        raise
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        operational = result.get("operational", {})
        partial = (
            {
                name: value
                for name, value in operational.items()
                if name.startswith("peak_")
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0
            }
            if isinstance(operational, Mapping)
            else {}
        )
        raise CandidateExecutionError(
            f"Invalid document worker result: {exc}",
            operational=partial,
            parameters={"measurement.resource_scope": "partial"},
            artifacts={
                "outputs": result,
                "validation_report": {
                    "status": "failed",
                    "reason_code": "invalid_worker_result",
                    "scheduled_attempt_count": len(items) * plan.repetitions,
                    "errors": [str(exc)],
                },
            },
        ) from exc

    evaluations, samples, latency_rows = [], [], []
    for item in items:
        rows = sorted(
            (row for row in attempts if row["sample_id"] == str(item["id"])),
            key=lambda row: row["repetition"],
        )
        documents = [decoded[(row["sample_id"], row["repetition"])] for row in rows]
        evaluation = score_document(
            item,
            documents[0],
            reference=references[str(item["id"])],
            repeated_documents=documents,
            failed=documents[0] is None,
            element_matching_threshold=protocol.element_matching_threshold,
            duplicate_content_threshold=protocol.duplicate_content_threshold,
        )
        executed = [row for row in rows if row["executed"]]
        if len(executed) != len(rows):
            incomplete = {"status": "incomplete", "reason": "missing_measured_attempts"}
            for name in (
                "reliability.attempt_failure_rate",
                "reliability.repeatability_success_rate",
            ):
                evaluation.metrics[name] = None
                evaluation.statuses[name] = incomplete
            if not rows[0]["executed"]:
                for name, status in evaluation.statuses.items():
                    if status["status"] != "inapplicable":
                        evaluation.metrics[name] = None
                        evaluation.statuses[name] = incomplete
        evaluations.append(evaluation)
        successful = [float(row["latency_seconds"]) for row in rows if row["success"]]
        pages = item.get("physical_page_count")
        if item["kind"] == "image":
            pages = 1
        elif item["kind"] == "pdf" and (type(pages) is not int or pages < 1):
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
                    **{
                        name: item[name]
                        for name in (
                            "document_family",
                            "has_table",
                            "has_formula",
                            "multi_column",
                            "layout_difficulty",
                            "physical_page_count",
                        )
                        if name in item
                    },
                    "reference_capabilities": sorted(
                        references[str(item["id"])].capabilities
                    ),
                    "source_group_id": source_id(item),
                    "document_groups": list(evaluation.groups),
                    "metric_statuses": evaluation.statuses,
                    "reference_counts": evaluation.reference_counts,
                    "first_attempt_id": 1,
                    "first_attempt_success": documents[0] is not None,
                    "successful_repetitions": len(successful),
                    "attempted_repetitions": len(executed),
                    "scheduled_repetitions": len(rows),
                    "unexecuted_repetitions": len(rows) - len(executed),
                    "warnings": list(rows[0]["document"].get("warnings", []))
                    if documents[0]
                    else [],
                    "failure": [row["error"] for row in executed if not row["success"]],
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
    for group in (
        None,
        *sorted({group for row in latency_rows for group in row["groups"]}),
    ):
        cohort = (
            latency_rows
            if group is None
            else [row for row in latency_rows if group in row["groups"]]
        )
        prefix = f"{group}." if group else ""
        operational.update(
            {
                prefix + name: value
                for name, value in _latency_statistics(cohort).items()
            }
        )
        intervals.update(
            {
                "operational." + prefix + name: bounds
                for name, bounds in bootstrap_sources(
                    cohort,
                    _latency_statistics,
                    resamples=resamples,
                    seed=plan.seed,
                    confidence=protocol.confidence_level,
                    minimum_sources=protocol.minimum_latency_ci_sources,
                ).items()
            }
        )
    if any(item["kind"] in {"image", "pdf"} for item in items):
        elapsed = result["measured_batch_seconds"]
        operational["batch_pages_per_minute"] = (
            60 * sum(row["successful_pages"] for row in latency_rows) / elapsed
            if elapsed is not None
            and elapsed > 0
            and all(row["executed"] for row in attempts)
            else None
        )
    profile = parse_document_profile(candidate)
    entry = model_lock.get(profile.lock_candidate, {})
    parameters = {
        **result["parameters"],
        "measurement.resource_scope": "partial"
        if result.get("execution_error")
        else "complete",
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
                "paddlepaddle-gpu",
                "paddlex",
                "torch",
                "transformers",
                "onnxruntime",
                "onnxruntime-gpu",
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
        "validation_report": {
            "status": "failed"
            if result.get("execution_error") or scorer_error
            else "passed",
            "scheduled_attempt_count": len(attempts),
            "executed_attempt_count": sum(row["executed"] for row in attempts),
            "failed_attempt_count": sum(row["success"] is False for row in attempts),
            "unexecuted_attempt_count": sum(not row["executed"] for row in attempts),
            "errors": [
                error
                for error in (result.get("execution_error"), scorer_error)
                if error
            ],
        },
    }
    if scorer_error:
        artifacts["evaluator_error"] = {"error": scorer_error}
    if result.get("resource_evidence"):
        artifacts["resource_evidence"] = result["resource_evidence"]
    if result.get("execution_error"):
        artifacts["execution_error"] = result["execution_error"]
    if scorer_error or result.get("execution_error"):
        raise CandidateExecutionError(
            "Official scoring incomplete; saved extraction outputs can be rescored"
            if scorer_error
            else str(result["execution_error"]),
            samples=tuple(samples),
            operational=operational,
            metrics=aggregate,
            intervals=intervals,
            parameters=parameters,
            artifacts=artifacts,
        )
    return samples, operational, aggregate, parameters, intervals, artifacts


def _attempt_inventory(result, items, repetitions):
    attempts = result["attempts"]
    if not isinstance(attempts, list):
        raise ValueError("Document attempts must be a list")
    operational = numeric_payload(result["operational"])
    if any(value is not None and value < 0 for value in operational.values()):
        raise ValueError("Document operational measurements must be nonnegative")
    if not isinstance(result["parameters"], Mapping) or not isinstance(
        result["resource_samples"], list
    ):
        raise ValueError("Document worker parameters/resource samples are malformed")
    elapsed = result["measured_batch_seconds"]
    if not (elapsed is None and result.get("execution_error")) and not _valid_duration(
        elapsed
    ):
        raise ValueError("Measured batch duration must be finite and nonnegative")
    expected = {
        (str(item["id"]), rep) for item in items for rep in range(1, repetitions + 1)
    }
    item_by_id = {str(item["id"]): item for item in items}
    documents = {}
    for row in attempts:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("sample_id"), str)
            or type(row.get("repetition")) is not int
        ):
            raise ValueError(
                "Document attempts require explicit sample IDs and integer repetitions"
            )
        key = (row["sample_id"], row["repetition"])
        if key not in expected or key in documents:
            raise ValueError("Duplicate or unexpected document attempt")
        executed = row.get("executed")
        if type(executed) is not bool:
            raise ValueError("Document executed state must be boolean")
        if not executed:
            if not result.get("execution_error") or any(
                row.get(name) is not None
                for name in (
                    "latency_seconds",
                    "success",
                    "error",
                    "document",
                    "output_fingerprint",
                )
            ):
                raise ValueError(
                    "Unexecuted document attempts cannot fabricate observations"
                )
            documents[key] = None
            continue
        duration = row["latency_seconds"]
        if not _valid_duration(duration):
            raise ValueError("Document attempt latency must be finite and nonnegative")
        if type(row.get("success")) is not bool:
            raise ValueError("Document attempt success must be boolean")
        document = None
        if row["success"]:
            payload = row["document"]
            if row.get("error") is not None:
                raise ValueError(
                    "Successful document attempt requires a complete canonical output"
                )
            document = validate_document_payload(payload)
            if row.get("output_fingerprint") != _document_fingerprint(document):
                raise ValueError("Document output fingerprint mismatch")
            item = item_by_id[key[0]]
            if document.source_kind.value != item["kind"] or (
                item.get("asset_sha256") is not None
                and document.source_checksum != item["asset_sha256"]
            ):
                raise ValueError(
                    "Document worker output does not match its input identity"
                )
        elif (
            row.get("document") is not None
            or row.get("output_fingerprint") is not None
            or not isinstance(row.get("error"), str)
            or not row["error"]
        ):
            raise ValueError(
                "Failed document attempt must retain its error and no completed output"
            )
        documents[key] = document
    if documents.keys() != expected:
        raise ValueError("Incomplete document attempt inventory")
    return attempts, documents


def _valid_duration(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _latency_statistics(rows):
    result = {}
    for key, suffix in (
        ("latency", "complete_document_latency_seconds"),
        ("page_latency", "warm_latency_per_page_seconds"),
    ):
        values = [row[key] for row in rows if row[key] is not None]
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


def paired_metrics(directions):
    return tuple(name for name in directions if not name.startswith("operational."))
