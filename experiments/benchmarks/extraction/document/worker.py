"""One fresh parser lifecycle: readiness, warmup, and all measured attempts."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

from edumind.common.model_placement import inspect_model_placement
from edumind.common.paths import PROJECT_ROOT
from edumind.extraction.contracts import (
    ExtractedDocument,
    ExtractionProfile,
    ExtractionRequest,
    ExtractionWarning,
    SourceKind,
)
from experiments.benchmarks.common.process import (
    WorkerMeasurementError,
    json_worker_main,
    seed_deterministically,
)
from experiments.benchmarks.common.resources import ResourceMonitor
from experiments.benchmarks.extraction.document.profiles import (
    lock_paths,
    parse_document_profile,
)
from experiments.benchmarks.extraction.document.protocol import protocol_from_worker
from experiments.benchmarks.extraction.registry import build_experiment_registry


def _number(value, label, minimum=0):
    try:
        valid = (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= minimum
        )
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{label} must be finite and >= {minimum}")


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _json_value(value):
    if isinstance(value, Mapping):
        if not isinstance(value, dict):
            raise ValueError("Canonical JSON objects must be dictionaries")
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Canonical JSON object keys must be strings")
        for child in value.values():
            _json_value(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _json_value(child)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical JSON numbers must be finite")
    elif value is not None and not isinstance(value, (str, int, bool)):
        raise ValueError("Canonical output must contain only JSON values")


def validate_document_payload(payload) -> ExtractedDocument:
    """Strict, side-effect-free canonical JSON validation shared with the runner.

    Unlike the cache decoder, this never fills missing document fields or drops
    malformed segment/warning entries. No parser/scorer/framework is imported.
    """
    if (
        not isinstance(payload, Mapping)
        or not {f.name for f in fields(ExtractedDocument)} <= payload.keys()
    ):
        raise ValueError("Canonical document is missing required fields")
    try:
        _json_value(payload)
    except (TypeError, RecursionError) as exc:
        raise ValueError("Canonical output must be finite, acyclic JSON") from exc
    for name in ("source_name", "source_path", "source_checksum", "text"):
        if not isinstance(payload[name], str):
            raise ValueError(f"Canonical {name} must be a string")
    if not isinstance(payload["source_kind"], str) or payload["source_kind"] not in {
        "pdf",
        "image",
        "docx",
    }:
        raise ValueError("Canonical document has an unsupported source kind")
    if payload["mime_type"] is not None and not isinstance(payload["mime_type"], str):
        raise ValueError("Canonical MIME type must be a string or null")
    if (
        not isinstance(payload["metadata"], Mapping)
        or type(payload["cache_hit"]) is not bool
    ):
        raise ValueError("Canonical document metadata/cache flag is malformed")
    if payload["cache_hit"]:
        raise ValueError("Measured extraction cannot return a cached document")
    _number(payload["extraction_seconds"], "Extraction duration")
    profile = payload["profile"]
    if (
        not isinstance(profile, Mapping)
        or not {f.name for f in fields(ExtractionProfile)} <= profile.keys()
    ):
        raise ValueError("Canonical extraction profile is incomplete")
    for name in (
        "name",
        "engine",
        "engine_revision",
        "preprocessing",
        "device",
        "routing",
        "normalization",
    ):
        if not isinstance(profile[name], str):
            raise ValueError(f"Canonical profile {name} must be a string")
    if not isinstance(profile["options"], Mapping):
        raise ValueError("Canonical profile options must be an object")
    segments, warnings = payload["segments"], payload["warnings"]
    if not isinstance(segments, (list, tuple)) or not isinstance(
        warnings, (list, tuple)
    ):
        raise ValueError("Canonical segments/warnings must be sequences")
    identifiers = set()
    parents = {}
    for segment in segments:
        required = {"text", "start", "end", "kind", "structured_content", "metadata"}
        if not isinstance(segment, Mapping) or not required <= segment.keys():
            raise ValueError("Canonical segment is missing required fields")
        if not isinstance(segment["text"], str) or not isinstance(segment["kind"], str):
            raise ValueError("Canonical segment text/kind must be strings")
        _integer(segment["start"], "Segment start")
        _integer(segment["end"], "Segment end")
        for name, minimum in (("order", 0), ("page_number", 1)):
            if segment.get(name) is not None:
                _integer(segment[name], name, minimum)
        for name in ("timestamp_start", "timestamp_end"):
            if segment.get(name) is not None:
                _number(segment[name], name)
        identifier, parent = segment.get("element_id"), segment.get("parent_id")
        for value in (identifier, parent):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(
                    "Canonical element/parent IDs must be nonempty strings or null"
                )
        if identifier is not None:
            if identifier in identifiers:
                raise ValueError("Canonical element IDs must be unique")
            identifiers.add(identifier)
            parents[identifier] = parent
        elif parent is not None:
            raise ValueError("Parented canonical elements require an element ID")
        box = segment.get("bounding_box")
        if box is not None:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError(
                    "Canonical box must contain four normalized coordinates"
                )
            for value in box:
                _number(value, "Bounding-box coordinate")
            if any(value > 1 for value in box) or box[0] > box[2] or box[1] > box[3]:
                raise ValueError(
                    "Canonical bounding box is outside the page or reversed"
                )
        structured = segment["structured_content"]
        if not isinstance(structured, Mapping) or not isinstance(
            segment["metadata"], Mapping
        ):
            raise ValueError("Canonical segment structure/metadata must be objects")
        if segment["kind"] == "table":
            rows = structured.get("rows")
            if not isinstance(rows, (list, tuple)) or any(
                not isinstance(row, (list, tuple))
                or any(not isinstance(cell, str) for cell in row)
                for row in rows
            ):
                raise ValueError("Canonical tables require rows of string cells")
            if "html" in structured and not isinstance(structured["html"], str):
                raise ValueError("Canonical table HTML must be a string")
            if "cells" in structured and (
                not isinstance(structured["cells"], (list, tuple))
                or any(
                    not isinstance(cell, Mapping)
                    or not isinstance(cell.get("text"), str)
                    for cell in structured["cells"]
                )
            ):
                raise ValueError("Canonical table cells must be text-bearing objects")
            for cell in structured.get("cells", ()):
                for name in (
                    "row",
                    "column",
                    "row_start",
                    "row_end",
                    "column_start",
                    "column_end",
                ):
                    if name in cell:
                        _integer(cell[name], f"Table cell {name}")
                for name in ("row_span", "column_span"):
                    if name in cell:
                        _integer(cell[name], f"Table cell {name}", 1)
        if segment["kind"] == "formula" and not isinstance(
            structured.get("latex"), str
        ):
            raise ValueError("Canonical formulas require a LaTeX string")
    for identifier, parent in parents.items():
        seen = {identifier}
        while parent is not None:
            if parent not in identifiers or parent in seen:
                raise ValueError(
                    "Canonical hierarchy contains a dangling parent or cycle"
                )
            seen.add(parent)
            parent = parents[parent]
    for warning in warnings:
        if (
            not isinstance(warning, Mapping)
            or not {f.name for f in fields(ExtractionWarning)} <= warning.keys()
        ):
            raise ValueError("Canonical warning is incomplete")
        if (
            not isinstance(warning["code"], str)
            or not isinstance(warning["message"], str)
            or type(warning["recoverable"]) is not bool
        ):
            raise ValueError("Canonical warning fields are malformed")
        if warning["segment_index"] is not None:
            _integer(warning["segment_index"], "Warning segment index")
    try:
        return ExtractedDocument.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Canonical document schema is invalid: {exc}") from exc


def synchronize(device, engine):
    if device != "cuda":
        return
    if engine == "paddleocr-vl-1.6":
        import paddle

        paddle.device.cuda.synchronize()
    else:
        import torch

        torch.cuda.synchronize()


def initialize(extractor, request):
    if (
        request.source_kind is SourceKind.IMAGE
        or extractor.engine == "paddleocr-vl-1.6"
    ):
        extractor.initialize_image_pipeline(request)
        return
    from docling.datamodel.base_models import InputFormat

    converter = (
        extractor._converter(request, request.source_kind)[0]
        if extractor.engine == "docling-standard"
        else extractor._docling_converter(request)
    )
    converter.initialize_pipeline(
        InputFormat.DOCX if request.source_kind is SourceKind.DOCX else InputFormat.PDF
    )


def pipeline_parameters(extractor):
    """Capture the actual initialized Docling options, including pinned defaults."""
    result = {}
    for runtime_index, runtime in enumerate(
        getattr(extractor, "_runtimes", {}).values()
    ):
        for key, pipeline in getattr(runtime, "initialized_pipelines", {}).items():
            options = getattr(pipeline, "pipeline_options", None)
            if options is not None:
                result[f"{runtime_index}:{type(pipeline).__name__}:{key[1]}"] = (
                    options.model_dump(mode="json", serialize_as_any=True)
                )
        config = getattr(runtime, "_merged_paddlex_config", None)
        if isinstance(config, Mapping):
            result[f"{runtime_index}:PaddleOCRVL"] = json.loads(json.dumps(config))
    return result


def _error_text(exc):
    parts = []
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        message = f"{type(exc).__name__}: {exc}"
        detail = getattr(exc, "detail", None)
        if detail and str(detail) not in message:
            message += f" ({detail})"
        parts.append(message)
        exc = exc.__cause__
    return " <- ".join(parts)


def _physical_pages(item):
    if item["kind"] == "docx":
        return None
    pages = item.get("physical_page_count", 1 if item["kind"] == "image" else None)
    _integer(pages, "Physical input page count", 1)
    return pages


@contextmanager
def _capture_lifecycle_failure(record_failure, resource_evidence):
    """Capture the original error before temporary-directory cleanup can mask it."""
    try:
        yield
    except Exception as exc:  # noqa: BLE001 - retain lifecycle observations
        record_failure(exc)
        resource_evidence.update(getattr(exc, "resource_evidence", {}))


def execute(payload):
    worker_started = float(payload.get("_worker_started_at", time.perf_counter()))
    protocol = protocol_from_worker(payload["protocol"])
    preflight = payload.get("mode") == "preflight"
    execution = (
        protocol.preflight if preflight else protocol.profile(str(payload["profile"]))
    )
    device = str(payload["device"])
    if device not in (
        ("cuda",) if preflight else execution.devices or (execution.device,)
    ):
        raise ValueError("Document worker device violates the execution profile")
    candidate = str(payload["candidate"])
    profile = parse_document_profile(candidate)
    model_backed = profile.requested_engine != "docling-standard-native"
    backend_device = device if model_backed else "cpu"
    protocol.validate_candidate_factors(profile.factors)
    entry = payload["model_lock"][profile.lock_candidate]
    parser_options = protocol.parser_options(profile.runtime_engine)
    if profile.runtime_engine != "docling-standard":
        applicable = (
            {"image_scale", "load_in_8bit"}
            if profile.runtime_engine == "docling-vlm-granite-258m"
            else {"pipeline_version", "recognition_backend"}
        )
        parser_options = {
            key: value for key, value in parser_options.items() if key in applicable
        }
    options = {
        **parser_options,
        **profile.options,
        **lock_paths(entry),
        "device": backend_device,
    }
    extraction_profile = ExtractionProfile(
        name=candidate,
        engine=profile.runtime_engine,
        engine_revision=str(entry.get("revision", "system")),
        normalization="none",
        device=backend_device,
        options=options,
    )
    items = list(payload["items"])
    if not items:
        raise ValueError("Document worker needs at least one input")
    random.Random(protocol.meta.seed).shuffle(items)
    requests = {
        str(item["id"]): ExtractionRequest(
            source_path=PROJECT_ROOT / str(item["source_path"]),
            checksum=str(item["asset_sha256"]),
            source_kind=SourceKind(item["kind"]),
            profile=extraction_profile,
            options=options,
        )
        for item in items
    }
    if len(requests) != len(items):
        raise ValueError("Document worker sample IDs must be unique")
    rows = [
        {
            "sample_id": str(item["id"]),
            "repetition": repetition,
            "page_count": _physical_pages(item),
            "executed": False,
            "success": None,
            "error": None,
            "document": None,
            "latency_seconds": None,
            "output_fingerprint": None,
            "not_executed_reason": "not_started",
        }
        for item in items
        for repetition in range(1, execution.repetitions + 1)
    ]
    extractor = monitor = before = after = None
    cold_load = first_latency = measured_batch_seconds = batch_started = None
    execution_error = failure = None
    phase, phase_started = "resources", time.perf_counter()
    resource_evidence = {}

    def record_failure(exc):
        nonlocal execution_error, failure, measured_batch_seconds
        error = {
            "phase": phase,
            "error": _error_text(exc),
            "phase_elapsed_seconds": time.perf_counter() - phase_started,
            "worker_elapsed_seconds": time.perf_counter() - worker_started,
        }
        if execution_error is None:
            execution_error, failure = error, exc
        else:
            execution_error.setdefault("additional_errors", []).append(error)
        if (
            phase == "measured"
            and batch_started is not None
            and measured_batch_seconds is None
        ):
            measured_batch_seconds = time.perf_counter() - batch_started

    def extract_payload(request):
        document = extractor.extract(request, request.source_kind)
        synchronize(backend_device, profile.runtime_engine)
        data = document.to_dict()
        validated = validate_document_payload(data)
        if (
            validated.source_kind != request.source_kind
            or validated.source_checksum != request.checksum
        ):
            raise ValueError("Canonical document does not match the input identity")
        return data, validated

    previous = {name: os.environ.get(name) for name in ("TMP", "TEMP", "TMPDIR")}
    previous_tempdir = tempfile.tempdir
    try:
        # Resolve the single frozen comparison before resource/latency monitoring.
        from experiments.benchmarks.extraction.document.metrics import (
            _document_fingerprint,
        )

        with (
            tempfile.TemporaryDirectory(prefix="document-work-") as raw,
            _capture_lifecycle_failure(record_failure, resource_evidence),
        ):
            for name in previous:
                os.environ[name] = raw
            tempfile.tempdir = raw
            monitor = ResourceMonitor(
                device=backend_device,
                require_vram=device == "cuda" and model_backed,
                temporary_directory=Path(raw),
            )
            with monitor:
                phase, phase_started = "load", time.perf_counter()
                seed_deterministically(protocol.meta.seed)
                extractor = build_experiment_registry().create(
                    profile.runtime_engine, SourceKind(items[0]["kind"])
                )
                initialize(extractor, requests[str(items[0]["id"])])
                synchronize(backend_device, profile.runtime_engine)
                cold_load = time.perf_counter() - phase_started
                phase, phase_started = "placement_before", time.perf_counter()
                if (preflight or device == "cuda") and model_backed:
                    before = inspect_model_placement(extractor, expected_device=device)
                    if before["status"] != "qualified":
                        raise RuntimeError(
                            f"Document backend violates CUDA placement: {before}"
                        )
                phase, phase_started = "warmup", time.perf_counter()
                for index in range(execution.warmups):
                    seed_deterministically(protocol.meta.seed)
                    extract_payload(requests[str(items[0]["id"])])
                    if index == 0:
                        first_latency = time.perf_counter() - worker_started
                phase, phase_started = "measured", time.perf_counter()
                batch_started = phase_started
                for row in rows:
                    started = time.perf_counter()
                    row.update(executed=True, not_executed_reason=None)
                    try:
                        seed_deterministically(protocol.meta.seed)
                        data, document = extract_payload(requests[row["sample_id"]])
                        fingerprint = _document_fingerprint(document)
                        row.update(
                            success=True, document=data, output_fingerprint=fingerprint
                        )
                    except Exception as exc:  # noqa: BLE001 - preserve every recoverable attempt
                        row.update(
                            success=False,
                            document=None,
                            output_fingerprint=None,
                            error=_error_text(exc),
                        )
                        if preflight and execution_error is None:
                            record_failure(exc)
                    finally:
                        row["latency_seconds"] = time.perf_counter() - started
                measured_batch_seconds = time.perf_counter() - batch_started
            # Extraction resource scope ends before qualification inspection.
            phase, phase_started = "placement_after", time.perf_counter()
            if (preflight or device == "cuda") and model_backed:
                after = inspect_model_placement(extractor, expected_device=device)
                if after["status"] != "qualified":
                    raise RuntimeError(
                        f"Document backend changed CUDA placement: {after}"
                    )
            phase, phase_started = "cleanup", time.perf_counter()
    except Exception as exc:  # noqa: BLE001 - lifecycle envelope retains completed observations
        if execution_error is not None:
            phase, phase_started = "cleanup", time.perf_counter()
        record_failure(exc)
        resource_evidence.update(getattr(exc, "resource_evidence", {}))
        if batch_started is not None and measured_batch_seconds is None:
            measured_batch_seconds = time.perf_counter() - batch_started
    finally:
        tempfile.tempdir = previous_tempdir
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    for row in rows:
        if not row["executed"]:
            row["not_executed_reason"] = (
                f"{execution_error['phase']}_failed"
                if execution_error
                else "not_started"
            )
    resources = {
        "peak_process_tree_ram_mb": None,
        "peak_vram_mb": None,
        "peak_temporary_disk_mb": None,
    }
    phase, phase_started = "resources", time.perf_counter()
    if monitor is not None:
        try:
            resources.update(monitor.metrics())
        except Exception as exc:  # noqa: BLE001 - retain partial observations without qualifying telemetry
            record_failure(exc)
            resource_evidence.update(getattr(exc, "resource_evidence", {}))
            for name in resources:
                if name == "peak_vram_mb" and backend_device == "cpu":
                    continue
                value = resource_evidence.get(name)
                try:
                    _number(value, name)
                except ValueError:
                    continue
                resources[name] = float(value)
    initialized_options = {}
    phase, phase_started = "parameters", time.perf_counter()
    try:
        initialized_options = pipeline_parameters(extractor)
    except Exception as exc:  # noqa: BLE001 - logging failure cannot erase attempts
        record_failure(exc)
    result = {
        "attempts": rows,
        "execution_error": execution_error,
        "measured_batch_seconds": measured_batch_seconds,
        "operational": {
            **resources,
            "cold_model_load_seconds": cold_load,
            "first_item_latency_seconds": first_latency,
        },
        "resource_samples": monitor.samples() if monitor else [],
        "resource_evidence": resource_evidence,
        "placement_before": before,
        "placement_after": after,
        "gpu_identity": monitor.gpu_identity if monitor else {},
        "parameters": {
            "resolved_parser_options": options,
            "initialized_pipeline_options": initialized_options,
            "device": device,
            "backend_device": backend_device,
            "dtype_policy": "backend-specific",
            "requested_dtype": None if preflight else execution.dtype_for(device),
            "seed": protocol.meta.seed,
            "warmups": execution.warmups,
            "repetitions": execution.repetitions,
            "vram_measurement_method": monitor.vram_measurement_method
            if monitor
            else "unavailable",
            "canonical_numeric_precision": "full-precision",
        },
    }
    if preflight:
        placement = next(
            (
                value
                for value in (after, before)
                if value and value["status"] != "qualified"
            ),
            after or before,
        )
        if not model_backed:
            placement = {
                "status": "qualified",
                "verification": "no-model-placement-required",
                "backend": profile.runtime_engine,
            }
        result.update(
            placement=placement,
            stress_sample_ids=[str(item["id"]) for item in items],
            vram_measurement_method=result["parameters"]["vram_measurement_method"],
            **resources,
        )
        # Shared preflight classifies OOM from exception text. Placement failures
        # instead return their precise native status, without a generic mask.
        if execution_error and execution_error["phase"] not in {
            "placement_before",
            "placement_after",
        }:
            if execution_error["phase"] == "resources":
                error = WorkerMeasurementError(execution_error["error"], result)
            else:
                error = RuntimeError(execution_error["error"])
                error.resource_evidence = result
            raise error from failure
    return result


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute))
