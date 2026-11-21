"""Behavioral regressions for the shared implementation-audit boundaries."""

from copy import deepcopy

import pytest

from benchmarks.common.metrics import precision_recall_f1
from benchmarks.extraction.document.protocol import load_protocol


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


def test_preflight_parent_and_child_use_canonical_dataset_identity(
    tmp_path, monkeypatch
):
    from benchmarks.common import preflight
    from benchmarks.common.tracking import NoTracking

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
