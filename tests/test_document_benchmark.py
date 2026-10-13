"""Document orchestration, saved evidence, and source-specific measurements."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from edumind.common.artifacts import atomic_write_json
from edumind.extraction import ExtractedDocument, ExtractionProfile, SourceKind
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
)
from experiments.benchmarks.common.process import WorkerMeasurementError
from experiments.benchmarks.extraction.document import benchmark, runner, validation
from experiments.benchmarks.extraction.document.metrics import (
    _document_fingerprint,
    load_reference,
)
from experiments.benchmarks.extraction.document.protocol import load_protocol


@pytest.fixture
def worker_result():
    document = ExtractedDocument(
        "fixture",
        "fixture.png",
        SourceKind.IMAGE,
        "asset",
        None,
        "alpha",
        (),
        ExtractionProfile("fixture", "fixture", "1"),
    )
    return {
        "attempts": [
            {
                "sample_id": "sample",
                "repetition": 1,
                "executed": True,
                "success": True,
                "error": None,
                "document": document.to_dict(),
                "output_fingerprint": _document_fingerprint(document),
                "latency_seconds": 0.1,
            }
        ],
        "measured_batch_seconds": 0.2,
        "parameters": {},
        "resource_samples": [],
        "operational": {"peak_vram_mb": None, "peak_process_tree_ram_mb": 10},
    }


def evaluate(monkeypatch, result, *, kind="image", profile="smoke", protocol=None):
    def launch(*args, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(runner, "run_json_worker", launch)
    protocol = protocol or load_protocol()
    item = {"id": "sample", "kind": kind, "asset_sha256": "asset", "reference": "alpha"}
    return runner.evaluate_candidate(
        "docling-standard-native",
        [item],
        BenchmarkPlan(
            "extraction",
            "document",
            profile,
            "fixture",
            ("candidate",),
            seed=42,
            warmups=1,
            repetitions=1,
            bootstrap_resamples=20 if profile != "smoke" else 0,
        ),
        {},
        {"device": "cpu"},
        {"sample": load_reference(item)},
        protocol,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result["attempts"].append(deepcopy(result["attempts"][0])),
        lambda result: result["attempts"].clear(),
        lambda result: result["attempts"][0].update(repetition=True),
        lambda result: result["attempts"][0].update(latency_seconds=float("nan")),
        lambda result: result["attempts"][0].update(latency_seconds=-1),
        lambda result: result["attempts"][0].update(success="true"),
        lambda result: result["attempts"][0].update(output_fingerprint="different"),
        lambda result: result["attempts"][0]["document"].update(segments=[None]),
        lambda result: result["attempts"][0]["document"].update(warnings=[None]),
        lambda result: result["attempts"][0]["document"].update(
            source_checksum="different"
        ),
        lambda result: result.update(measured_batch_seconds=-1),
    ],
)
def test_malformed_worker_keeps_evidence_without_invented_samples(
    monkeypatch, worker_result, mutation
):
    mutation(worker_result)
    with pytest.raises(CandidateExecutionError) as failure:
        evaluate(monkeypatch, worker_result)
    assert failure.value.samples == ()
    assert failure.value.operational["peak_process_tree_ram_mb"] == 10
    assert failure.value.parameters["measurement.resource_scope"] == "partial"
    assert failure.value.artifacts["validation_report"]["status"] == "failed"
    assert failure.value.artifacts["outputs"] == worker_result


def test_lifecycle_failure_retains_unexecuted_rows_without_failure_rate(
    monkeypatch, worker_result
):
    worker_result["execution_error"] = {"phase": "warmup", "error": "injected"}
    worker_result["measured_batch_seconds"] = None
    worker_result["attempts"][0].update(
        executed=False,
        success=None,
        error=None,
        document=None,
        latency_seconds=None,
        output_fingerprint=None,
    )
    with pytest.raises(CandidateExecutionError) as failure:
        evaluate(monkeypatch, worker_result)
    sample = failure.value.samples[0]
    assert sample.metrics["text.content_f1"] is None
    assert sample.metrics["reliability.attempt_failure_rate"] is None
    assert (
        sample.metadata["metric_statuses"]["text.content_f1"]["status"] == "incomplete"
    )
    assert failure.value.operational["batch_pages_per_minute"] is None
    assert failure.value.artifacts["validation_report"]["unexecuted_attempt_count"] == 1


def test_post_inference_failure_keeps_completed_first_quality(
    monkeypatch, worker_result
):
    worker_result["execution_error"] = {"phase": "placement_after", "error": "offload"}
    with pytest.raises(CandidateExecutionError) as failure:
        evaluate(monkeypatch, worker_result)
    assert failure.value.samples[0].metrics["text.content_f1"] == 1
    assert failure.value.samples[0].metrics["reliability.attempt_failure_rate"] == 0
    assert failure.value.parameters["measurement.resource_scope"] == "partial"


def test_supervisor_telemetry_failure_retains_completed_worker_output(
    monkeypatch, worker_result
):
    worker_result["resource_evidence"] = {"worker_samples": [1]}

    with pytest.raises(CandidateExecutionError) as failure:
        evaluate(
            monkeypatch,
            WorkerMeasurementError(
                "telemetry gap",
                {"worker_result": worker_result, "supervisor_samples": [2]},
            ),
        )
    assert failure.value.samples[0].metrics["text.content_f1"] == 1
    assert failure.value.artifacts["execution_error"]["phase"] == "supervision"
    assert failure.value.artifacts["resource_evidence"] == {
        "worker": {"worker_samples": [1]},
        "supervisor": {"supervisor_samples": [2]},
    }


def test_child_measurement_error_is_not_a_completed_worker_result(monkeypatch):
    error = WorkerMeasurementError(
        "telemetry gap",
        {"worker_result": {"worker_error": "telemetry gap"}, "peak_vram_mb": 200},
    )
    with pytest.raises(WorkerMeasurementError) as failure:
        evaluate(monkeypatch, error)
    assert failure.value is error
    assert failure.value.operational["peak_vram_mb"] == 200
    assert failure.value.artifacts["validation_report"]["reason_code"] == (
        "missing_worker_result"
    )


def test_worker_supervisor_preserves_written_result_on_measurement_gap(
    tmp_path, monkeypatch, worker_result
):
    from types import SimpleNamespace

    from experiments.benchmarks.common import process

    class Monitor:
        peak_vram_mb = 20

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def metrics(self):
            error = RuntimeError("telemetry gap")
            error.resource_evidence = {"peak_vram_mb": 20}
            raise error

    def launch(command, **kwargs):
        atomic_write_json(Path(command[-1]), worker_result)
        return SimpleNamespace(
            pid=1, returncode=0, poll=lambda: 0, communicate=lambda: ("", "")
        )

    monkeypatch.setattr(process, "ResourceMonitor", Monitor)
    monkeypatch.setattr(process.subprocess, "Popen", launch)
    with pytest.raises(WorkerMeasurementError) as failure:
        process.run_json_worker(
            Path(runner.__file__).with_name("worker.py"),
            {},
            device="cuda",
            prefix="document-supervision-",
            error_label="test",
            require_vram_measurement=True,
            temporary_root=tmp_path,
        )
    assert failure.value.resource_evidence["worker_result"] == json.loads(
        json.dumps(worker_result)
    )
    assert failure.value.operational["peak_vram_mb"] == 20


def test_latency_groups_have_reference_fixed_source_support(monkeypatch, worker_result):
    protocol = replace(load_protocol(), minimum_latency_ci_sources=2)
    result = evaluate(
        monkeypatch, worker_result, profile="development", protocol=protocol
    )
    operational, intervals = result[1], result[4]
    key = "p50_complete_document_latency_seconds"
    assert operational[f"image.{key}"] == 0.1
    assert intervals[f"operational.image.{key}"]["independent_source_count"] == 1
    assert intervals[f"operational.image.{key}"]["lower"] is None


@pytest.mark.parametrize(
    "profile,stage",
    [
        ("development", "configuration"),
        ("validation", "architecture-development"),
        ("locked", "architecture-validation"),
    ],
)
@pytest.mark.parametrize("stale", ["protocol", "qualification", None])
def test_decisions_use_current_protocol_and_qualification(
    tmp_path, profile, stage, stale
):
    protocol = load_protocol()
    candidate = protocol.configuration_candidates("development")[0]
    summary = {
        "run_id": "source",
        "complete": True,
        "plan": {
            "suite": "extraction",
            "stage": f"document-{stage}-pdf",
            "profile": "validation" if profile == "locked" else "development",
            "settings": {
                "preflight_fingerprint": "old"
                if stale == "qualification"
                else "current"
            },
        },
        "protocols": {
            "document": {
                "checksum": "old" if stale == "protocol" else protocol.meta.checksum
            }
        },
        "candidates": [{"candidate": candidate, "status": "success"}],
    }
    atomic_write_json(tmp_path / "summary.json", summary)
    path = tmp_path / "decision.json"
    atomic_write_json(
        path,
        {
            "schema_version": 1,
            "source_summary": "summary.json",
            "source_run_id": "source",
            "selected_candidates": [candidate],
            "selected_by": "reviewer",
            "selected_date": "2026-10-13",
            "reason": "reviewed",
        },
    )

    def select():
        return benchmark.comparison_candidates(
            "pdf",
            profile,
            f"architecture-{profile}",
            protocol,
            selection_path=path,
            qualification_fingerprint="current",
        )

    if stale:
        with pytest.raises(
            ValueError, match="different .*protocol|different hardware qualification"
        ):
            select()
    else:
        selected, files = select()
        assert selected[0] == candidate
        assert len(selected) == (3 if profile == "development" else 1)
        assert files == {"pdf": path}


def test_reference_page_requires_physical_page_even_without_page_metrics(tmp_path):
    from experiments.benchmarks.common.datasets import manifest_content_checksum

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "data/benchmarks/extraction/smoke.json").read_text(encoding="utf-8")
    )
    item = next(row for row in manifest["samples"] if row["kind"] == "image")
    item.update(
        reference_capabilities=["reading_order"],
        reference_elements=[
            {
                "id": "one",
                "kind": "text",
                "text": "alpha",
                "order": 0,
                "page_number": 2,
            },
        ],
    )
    manifest["samples"] = [item]
    manifest["checksum"] = manifest_content_checksum(manifest["samples"])
    path = tmp_path / "manifest.json"
    atomic_write_json(path, manifest)
    with pytest.raises(ValueError, match="physical page inventory"):
        validation.validate(path, "smoke", load_protocol())
