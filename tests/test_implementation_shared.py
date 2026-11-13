"""Behavioral regressions for the shared implementation-audit boundaries."""

from copy import deepcopy

import pytest

from experiments.benchmarks.common.contracts import BenchmarkPlan, SampleResult
from experiments.benchmarks.common.metrics import precision_recall_f1
from experiments.benchmarks.common.runner import run_benchmark
from experiments.benchmarks.extraction.document.protocol import load_protocol


@pytest.mark.parametrize("false_positive,false_negative", [(13, 1), (1, 13)])
def test_count_f1_exact_threshold(false_positive, false_negative):
    assert precision_recall_f1(7, false_positive, false_negative)[2] == 0.5
    assert precision_recall_f1(6, false_positive, false_negative)[2] < 0.5


@pytest.mark.parametrize("version", [None, True, 999, "1"])
def test_worker_envelope_rejects_invalid_schema(version):
    protocol = load_protocol().meta
    payload = deepcopy(protocol.worker_payload())
    if version is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = version
    with pytest.raises(ValueError):
        protocol.validate_worker_payload(payload)


def test_worker_envelope_rejects_unknown_fields():
    protocol = load_protocol().meta
    with pytest.raises(ValueError):
        protocol.validate_worker_payload({**protocol.worker_payload(), "unknown": 1})
    protocol.validate_worker_payload(protocol.worker_payload())


def test_all_protocol_envelopes_use_the_same_strict_boundary():
    from test_benchmark_protocols import PROTOCOL_CASES

    for name, path, loader, _ in PROTOCOL_CASES:
        meta = loader(path).meta
        meta.validate_worker_payload(meta.worker_payload())
        for damage in (
            {"schema_version": True},
            {"schema_version": 999},
            {"unexpected": "field"},
        ):
            with pytest.raises(ValueError):
                meta.validate_worker_payload({**meta.worker_payload(), **damage})
        missing = meta.worker_payload()
        missing.pop("schema_version")
        with pytest.raises(ValueError):
            meta.validate_worker_payload(missing)


def test_completed_evidence_survives_late_telemetry_error(tmp_path, monkeypatch):
    from experiments.benchmarks.common import runner

    class Monitor:
        vram_measurement_method = "unavailable"

        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def samples(self):
            return [{"timestamp": 1, "ram_mb": 123}]

        def metrics(self):
            error = RuntimeError("late telemetry failure")
            error.resource_evidence = {
                "peak_process_tree_ram_mb": 123,
                "peak_vram_mb": None,
            }
            raise error

    monkeypatch.setattr(runner, "ResourceMonitor", Monitor)
    plan = BenchmarkPlan(
        "test", "audit", "smoke", "fixture", ("candidate",), 42, 1, 0, 0
    )

    def evaluate(candidate):
        return (
            [SampleResult("q", {"quality.score": 0.75}, 0.1)],
            {"latency_seconds": 0.1},
            {"quality.score": 0.75},
            {},
            {},
            {"predictions": [{"question": "q", "text": "retained prediction"}]},
        )

    result = run_benchmark(
        plan,
        evaluate,
        dataset_checksum="fixture",
        directions={"quality.score": "max"},
        primary_metric="quality.score",
        no_mlflow=True,
        artifact_root=tmp_path,
        resource_artifact_name="resource_samples",
    )
    candidate = result.candidates[0]
    assert candidate.status == "failed"
    assert candidate.metrics["quality.score"] == 0.75
    assert candidate.operational["peak_process_tree_ram_mb"] == 123
    assert candidate.operational["peak_vram_mb"] is None
    assert len(candidate.samples) == 1
    import pandas as pd

    predictions = list(result.artifact_directory.rglob("predictions.parquet"))
    assert len(predictions) == 1
    assert pd.read_parquet(predictions[0]).to_dict("records") == [
        {"question": "q", "text": "retained prediction"}
    ]


def test_preflight_parent_and_child_use_canonical_dataset_identity(
    tmp_path, monkeypatch
):
    from experiments.benchmarks.common import preflight
    from experiments.benchmarks.common.tracking import NoTracking

    tags = []

    class Tracking(NoTracking):
        def tags(self, values):
            tags.append(values)

    monkeypatch.setattr(preflight, "tracker", lambda **kwargs: Tracking())
    preflight.run_preflight(
        benchmark="generation",
        candidates=("test-candidate",),
        fingerprint="test-only",
        context={
            "input_envelope": {"stress_manifest_checksum": "actual-fixture-identity"}
        },
        probe=lambda _: {
            "placement": {"status": "qualified"},
            "vram_measurement_method": "nvml-device-total",
        },
        no_mlflow=True,
        artifact_root=tmp_path,
    )
    identities = [row["dataset_checksum"] for row in tags if "dataset_checksum" in row]
    assert identities == ["actual-fixture-identity", "actual-fixture-identity"]


def test_generic_rows_never_claim_independent_source_intervals():
    from experiments.benchmarks.common.statistics import aggregate_samples

    samples = [
        SampleResult("q1", {"quality": 1.0}, None, {"source_id": "same-paper"}),
        SampleResult("q2", {"quality": 0.0}, None, {"source_id": "same-paper"}),
        SampleResult("q3", {"quality": None}, None, {"source_id": "other-paper"}),
    ]
    metrics, intervals = aggregate_samples(samples, resamples=100)
    assert metrics["quality"] == 0.5
    assert intervals["quality"]["lower"] is None
    assert intervals["quality"]["reason"] == "benchmark_estimator_required"
    assert intervals["quality"]["eligible_count"] == 3
    assert intervals["quality"]["unavailable_count"] == 1
