"""One fresh parser lifecycle: readiness, warmup, and all measured attempts."""

from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path

from edumind.common.model_placement import inspect_model_placement
from edumind.common.paths import PROJECT_ROOT
from edumind.extraction import ExtractionProfile, ExtractionRequest, SourceKind
from experiments.benchmarks.common.process import (
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
    return {
        str(format_name): option.pipeline_options.model_dump(mode="json")
        for converter in getattr(extractor, "_runtimes", {}).values()
        for format_name, option in getattr(converter, "format_to_options", {}).items()
        if option.pipeline_options is not None
    }


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
    options = {
        **protocol.parser_options(profile.runtime_engine),
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
    rows = []
    with tempfile.TemporaryDirectory(prefix="document-work-") as raw:
        directory = Path(raw)
        previous = {name: os.environ.get(name) for name in ("TMP", "TEMP", "TMPDIR")}
        previous_tempdir = tempfile.tempdir
        try:
            for name in previous:
                os.environ[name] = raw
            tempfile.tempdir = raw
            with ResourceMonitor(
                device=backend_device,
                require_vram=device == "cuda" and model_backed,
                temporary_directory=directory,
            ) as monitor:
                started = time.perf_counter()
                seed_deterministically(protocol.meta.seed)
                extractor = build_experiment_registry().create(
                    profile.runtime_engine, SourceKind(items[0]["kind"])
                )
                initialize(extractor, requests[str(items[0]["id"])])
                synchronize(backend_device, profile.runtime_engine)
                cold_load = time.perf_counter() - started
                before = (
                    inspect_model_placement(extractor, expected_device=device)
                    if (preflight or device == "cuda") and model_backed
                    else None
                )
                if before and not preflight and before["status"] != "qualified":
                    raise RuntimeError(
                        f"Document backend violates CUDA placement: {before}"
                    )
                first_latency = None
                for index in range(execution.warmups):
                    request = requests[str(items[0]["id"])]
                    extractor.extract(request, request.source_kind)
                    synchronize(backend_device, profile.runtime_engine)
                    if index == 0:
                        first_latency = time.perf_counter() - worker_started
                batch_started = time.perf_counter()
                for item in items:
                    for repetition in range(1, execution.repetitions + 1):
                        seed_deterministically(protocol.meta.seed)
                        started = time.perf_counter()
                        document = error = None
                        try:
                            request = requests[str(item["id"])]
                            document = extractor.extract(request, request.source_kind)
                            synchronize(backend_device, profile.runtime_engine)
                            document = document.to_dict()
                        except Exception as exc:  # noqa: BLE001 - each scheduled attempt survives
                            error = f"{type(exc).__name__}: {exc}"
                        rows.append(
                            {
                                "sample_id": str(item["id"]),
                                "repetition": repetition,
                                "success": error is None,
                                "error": error,
                                "document": document,
                                "latency_seconds": time.perf_counter() - started,
                            }
                        )
                measured_batch_seconds = time.perf_counter() - batch_started
                after = (
                    inspect_model_placement(extractor, expected_device=device)
                    if (preflight or device == "cuda") and model_backed
                    else None
                )
                if after and not preflight and after["status"] != "qualified":
                    raise RuntimeError(
                        f"Document backend changed CUDA placement: {after}"
                    )
        finally:
            tempfile.tempdir = previous_tempdir
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    if preflight:
        if any(not row["success"] for row in rows):
            raise RuntimeError("Document stress inference failed")
        placement = (
            (after if after["status"] != "qualified" else before)
            if model_backed
            else {
                "status": "qualified",
                "verification": "no-model-placement-required",
                "backend": profile.runtime_engine,
            }
        )
        return {
            "placement": placement,
            "placement_before": before,
            "placement_after": after,
            **monitor.metrics(),
            "vram_measurement_method": monitor.vram_measurement_method,
            "resource_samples": monitor.samples(),
            "stress_sample_ids": [str(item["id"]) for item in items],
        }
    return {
        "attempts": rows,
        "measured_batch_seconds": measured_batch_seconds,
        "operational": {
            **monitor.metrics(),
            "cold_model_load_seconds": cold_load,
            "first_item_latency_seconds": first_latency,
        },
        "resource_samples": monitor.samples(),
        "placement_before": before,
        "placement_after": after,
        "gpu_identity": monitor.gpu_identity,
        "parameters": {
            "resolved_parser_options": options,
            "initialized_pipeline_options": pipeline_parameters(extractor),
            "device": device,
            "backend_device": backend_device,
            "dtype_policy": "backend-specific",
            "requested_dtype": execution.dtype_for(device),
            "seed": protocol.meta.seed,
            "warmups": execution.warmups,
            "repetitions": execution.repetitions,
            "vram_measurement_method": monitor.vram_measurement_method,
            "canonical_numeric_precision": "full-precision",
        },
    }


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute))
