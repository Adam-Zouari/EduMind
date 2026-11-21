"""Selected-server Final integration regressions, with test-only backends."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from test_generation_benchmark import Backend, Monitor

from benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
)
from benchmarks.common.datasets import load_manifest
from benchmarks.rag.evaluation import Chunk
from benchmarks.rag.final import benchmark, validation
from benchmarks.rag.final.composition import reserve_locked
from benchmarks.rag.final.protocol import load_protocol as final_protocol
from benchmarks.rag.final.validation import (
    manifest_path,
    prepare_report,
    verify_inputs,
)
from benchmarks.rag.generation.protocol import load_protocol
from benchmarks.rag.retrieval_reranking.protocol import (
    load_protocol as retrieval_protocol,
)
from edumind.common.artifacts import stable_hash


def test_locked_consumption_is_atomic_and_not_reopened_after_failure(tmp_path):
    marker = tmp_path / "locked.json"
    reserve_locked(marker, {"manifest": "test-only"})
    with pytest.raises(ValueError, match="already consumed"):
        reserve_locked(marker, {"manifest": "different configuration"})


def test_final_data_check_accepts_missing_supplied_gold_context(tmp_path, monkeypatch):
    import json

    source = manifest_path("smoke")
    payload = json.loads(source.read_text(encoding="utf-8"))
    for row in payload["samples"]:
        if row.get("kind") == "question":
            row.pop("supplied_evidence", None)
    from benchmarks.common.datasets import manifest_content_checksum

    payload["checksum"] = manifest_content_checksum(payload["samples"])
    target = tmp_path / "final.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    _, report = prepare_report(
        target, "smoke", final_protocol(), report_root=tmp_path / "reports"
    )
    assert report["status"] == "passed", report["errors"]
    verify_inputs(target, "smoke", final_protocol(), report_root=tmp_path / "reports")
    changed_validator = tmp_path / "changed_generation_validator.py"
    changed_validator.write_text(
        "# changed validation implementation", encoding="utf-8"
    )
    monkeypatch.setattr(
        validation.generation_validation, "__file__", str(changed_validator)
    )
    with pytest.raises(ValueError, match="Missing or stale data validation"):
        verify_inputs(
            target, "smoke", final_protocol(), report_root=tmp_path / "reports"
        )


@pytest.mark.parametrize("fail_first", [False, True])
def test_live_server_execution_and_first_ranking_quality(tmp_path, fail_first):
    manifest = load_manifest(manifest_path("smoke"))
    protocol = load_protocol()
    chunks = [
        Chunk(row["id"] + ":whole", row["id"], row["text"], 0, len(row["text"]), 10)
        for row in manifest.samples
        if row.get("kind") == "document"
    ]
    server_calls = []

    class Server:
        identity = "test-live-server"
        reranker = None
        index = SimpleNamespace(chunks=chunks, embedder=None)

        def retrieve(self, question, *, evidence):
            server_calls.append(question["id"])
            # Warmup succeeds; the first measured request can fail.
            if fail_first and len(server_calls) == 2:
                raise RuntimeError("test database timeout")
            evidence.update(
                pool=list(range(len(chunks))),
                ranking=list(reversed(range(len(chunks)))),
                database_latency_seconds=0.001,
                identity=self.identity,
            )

    composition = SimpleNamespace(
        checksum="test-composition",
        protocols={"final": "fixture"},
        system={"residency": "resident", "top_k": 3},
    )
    backend = Backend()
    plan = BenchmarkPlan(
        "rag",
        "final",
        "smoke",
        manifest.name,
        ("selected-system",),
        42,
        1,
        0,
        1,
        {"device": "cpu", "dtype": "float32"},
    )
    raw = benchmark.execute(
        "qwen3-0.6b-direct",
        manifest,
        {},
        plan,
        protocol,
        composition,
        Server(),
        backend_factory=lambda *args: backend,
        monitor_factory=Monitor,
        evidence_directory=tmp_path,
    )
    assert raw["error"] is None
    assert len(server_calls) == 1 + len(raw["records"])
    assert backend.closed
    raw["chunk_inventory"] = [asdict(chunk) for chunk in chunks]
    # The frozen schedule remains independent from retrieved contexts.
    assert raw["schedule_checksum"] == stable_hash(raw["schedule"])
    if fail_first:
        first = raw["records"][0]["attempts"][0]
        assert first["failure_stage"] == "retrieval"
        assert first["generation_status"] == "not_started"
        with pytest.raises(CandidateExecutionError) as caught:
            benchmark.score(raw, manifest, protocol, retrieval_protocol())
        assert caught.value.metrics["reliability.request_failure_rate"] > 0
        assert caught.value.metrics["diagnostic.evidence_unit_recall_at_3"] is None
    else:
        for record in raw["records"]:
            blocks = record["attempts"][0]["live_question"]["blocks"]
            assert [block["text"] for block in blocks] == [
                chunk.text for chunk in chunks[::-1][:3]
            ]
        result = benchmark.score(raw, manifest, protocol, retrieval_protocol())
        assert result[2]["diagnostic.evidence_unit_recall_at_3"] == 1
        assert result[2]["reliability.request_failure_rate"] == 0
        assert result[1]["server_call_p50_seconds"] == 0.001
