"""Fresh CUDA feasibility probe for one document parser candidate."""

from __future__ import annotations

from edumind.common.model_placement import inspect_model_placement
from edumind.extraction import ExtractionPipeline, SourceKind
from experiments.benchmarks.common.process import json_worker_main
from experiments.benchmarks.extraction.document.benchmark import extract_once
from experiments.benchmarks.extraction.document.profiles import parse_document_profile
from experiments.benchmarks.extraction.document.protocol import protocol_from_worker
from experiments.benchmarks.extraction.registry import build_experiment_registry


def execute(payload: dict[str, object]) -> dict[str, object]:
    protocol = protocol_from_worker(payload["protocol"])
    candidate = str(payload["candidate"])
    profile = parse_document_profile(candidate)
    if "cuda" not in protocol.backend_devices[profile.runtime_engine]:
        raise RuntimeError(f"{profile.runtime_engine} does not support CUDA")
    pipeline = ExtractionPipeline(registry=build_experiment_registry())
    latency = 0.0
    segment_count = warning_count = 0
    sample_ids = []
    items = list(payload["items"])
    for _ in range(int(payload["warmups"])):
        extract_once(
            candidate,
            items[0],
            payload["model_lock"],
            {"device": "cuda"},
            pipeline,
            protocol=protocol,
        )
    for _ in range(int(payload["repetitions"])):
        for item in items:
            document, item_latency = extract_once(
                candidate,
                item,
                payload["model_lock"],
                {"device": "cuda"},
                pipeline,
                protocol=protocol,
            )
            latency += item_latency
            segment_count += len(document.segments)
            warning_count += len(document.warnings)
            sample_ids.append(str(item["id"]))
    if bool(payload["placement_required"]):
        extractor = pipeline.registry.create(
            profile.runtime_engine,
            SourceKind(str(items[0]["kind"])),
        )
        placement = inspect_model_placement(extractor, expected_device="cuda")
    else:
        placement = {
            "status": "qualified",
            "verification": "no-model-placement-required",
            "backend": profile.runtime_engine,
            "expected_device": "cuda",
        }
    return {
        "placement": placement,
        "latency_seconds": latency,
        "output_segment_count": segment_count,
        "warning_count": warning_count,
        "stress_sample_ids": sample_ids,
    }


if __name__ == "__main__":
    raise SystemExit(json_worker_main(execute))
