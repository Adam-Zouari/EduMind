"""Bounded Final integrity regressions; all runtime/judge fixtures are test-only."""

import json
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from test_generation_benchmark import Backend, Monitor
from test_generation_recovery import calibrated_inputs as calibrated_fixture
from test_generation_recovery import fixture_transport

from edumind.common.artifacts import stable_hash
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    DatasetManifest,
)
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.rag.evaluation import Chunk
from experiments.benchmarks.rag.final import benchmark
from experiments.benchmarks.rag.final.validation import manifest_path
from experiments.benchmarks.rag.generation.judge import SemanticJudge
from experiments.benchmarks.rag.generation.metrics import score_pairs
from experiments.benchmarks.rag.generation.protocol import load_protocol
from experiments.benchmarks.rag.generation.validation import questions_from_manifest
from experiments.benchmarks.rag.retrieval_reranking.protocol import (
    load_protocol as retrieval_protocol,
)


def _offline_score(monkeypatch, attempts, *, documents=None, sources=None):
    """Exercise Final's actual RR aggregation; unrelated generation is isolated."""
    monkeypatch.setattr(
        benchmark, "score_execution", lambda *a, **kw: ([], {}, {}, {}, {}, {})
    )
    documents = documents or [f"doc-{i}" for i in range(len(attempts))]
    sources = sources or ["reviewed-shared-source"] * len(attempts)
    questions = [
        {
            "kind": "question",
            "id": f"q-{i}",
            "document_id": document,
            "source_id": source,
            "evidence_type": "text",
            "answerable": False,
        }
        for i, (document, source) in enumerate(zip(documents, sources, strict=True))
    ]
    manifest = DatasetManifest(
        "test-only-final",
        "1",
        "rag",
        "smoke",
        "test-only",
        "CC0",
        "1",
        "test-only",
        "test-only",
        42,
        tuple(questions),
    )
    raw = {
        "schedule": {"profile": "smoke"},
        "composition": {"top_k": 3},
        "composition_checksum": "test-only",
        "chunk_inventory": [],
        "records": [
            {"question": q, "attempts": rows}
            for q, rows in zip(questions, attempts, strict=True)
        ],
    }
    before = deepcopy(raw)
    try:
        result = benchmark.score(raw, manifest, load_protocol(), retrieval_protocol())
    except CandidateExecutionError as exc:
        result = (
            exc.samples,
            exc.operational,
            exc.metrics,
            exc.parameters,
            exc.intervals,
            exc.artifacts,
        )
    assert raw == before, "Offline scoring must not rewrite retained execution evidence"
    return result


def _attempt(*, success=True, outcome=None, seconds=2.0, schema_valid=True):
    return {
        "seed": 42,
        "request_status": "completed" if success else "failed",
        "outcome": outcome or ("completed" if success else "timeout"),
        "generation_latency_seconds": seconds,
        "request_latency_seconds": seconds if success else None,
        "visible_text": json.dumps(
            {"status": "answered", "answer": "test-only", "citations": ["E1"]}
        )
        if schema_valid
        else "{",
        "retrieval": {"latency_seconds": 0.25, "database_latency_seconds": 0.1},
    }


def test_rr_projection_keeps_reviewed_source_and_document_macro_weights(monkeypatch):
    result = _offline_score(
        monkeypatch,
        [[_attempt(success=False)], [_attempt()], [_attempt(success=False)]],
        documents=["doc-A", "doc-A", "doc-B"],
    )
    key = "reliability.request_failure_rate"
    support = result[5]["retrieval_metric_support"][key]
    assert result[2][key] == 0.75  # Document means 0.5 and 1, not pooled 2/3.
    assert support["eligible_question_count"] == 3
    assert support["eligible_document_count"] == 2
    assert support["independent_source_count"] == 1
    assert support["contributing_source_count"] == 1
    assert result[4][key]["independent_source_count"] == 1


def test_rr_projection_preserves_unavailable_source_inventory(monkeypatch):
    result = _offline_score(
        monkeypatch,
        [[_attempt(success=False)], [{"seed": 42, "outcome": "not_executed"}]],
        sources=["reviewed-A", "reviewed-B"],
    )
    support = result[5]["retrieval_metric_support"]["reliability.request_failure_rate"]
    assert support["independent_source_count"] == 2
    assert support["contributing_source_count"] == 1
    assert support["unavailable_question_count"] == 1
    assert not support["complete"]


def _citations(seed, cited, mapping, *, supplied=("E1", "E2")):
    return {
        "seed": seed,
        "outcome": "completed",
        "response": {"status": "answered", "answer": "test-only", "citations": cited},
        "citation_identities": mapping,
        "live_question": {"blocks": [{"id": label} for label in supplied]},
    }


def _citation_pair(left, right):
    return score_pairs(
        {"answerable": True},
        [left, right],
        citation_identity=benchmark.stable_citations,
    )[0]["metrics"]["repeatability.citation_agreement"]


@pytest.mark.parametrize("identity", [None, "", 123])
def test_known_citation_requires_usable_stable_identity(identity):
    left = _citations(42, ["E1"], {"E1": identity})
    with pytest.raises((ValueError, KeyError), match="identity|mapping"):
        benchmark.stable_citations(left)


def test_missing_known_mapping_makes_pair_unavailable_not_unknown_agreement():
    left = _citations(42, ["E1"], {})
    right = _citations(43, ["E1"], {})
    cell = _citation_pair(left, right)
    assert cell["value"] is None
    assert cell["status"] == "unavailable"
    assert "citation_identity_unavailable" in cell["reason"]


def test_relabelled_known_evidence_and_unknown_labels_remain_distinct():
    left = _citations(42, ["E1", "unknown-label"], {"E1": "chunk-A", "E2": "chunk-B"})
    relabelled = _citations(
        43, ["E2", "unknown-label"], {"E1": "chunk-B", "E2": "chunk-A"}
    )
    assert _citation_pair(left, relabelled)["value"] == 1.0
    same_label_other_content = _citations(43, ["E1"], {"E1": "chunk-B"})
    assert (
        _citation_pair(
            _citations(42, ["E1"], {"E1": "chunk-A"}), same_label_other_content
        )["value"]
        == 0.0
    )
    assert benchmark.stable_citations(left) == {
        "known:chunk-A",
        "unknown:unknown-label",
    }


@pytest.mark.parametrize(
    "outcome", ["timeout", "context_limit", "runtime_error", "completed"]
)
def test_failed_generation_is_excluded_but_earlier_stage_timings_survive(
    monkeypatch, outcome
):
    result = _offline_score(
        monkeypatch,
        [[_attempt(seconds=2), _attempt(success=False, outcome=outcome, seconds=100)]],
    )
    assert result[1]["generation_p50_seconds"] == 2
    assert result[1]["generation_p95_seconds"] == 2
    assert result[1]["retrieval_p50_seconds"] == 0.25
    assert result[1]["server_call_p50_seconds"] == 0.1
    assert "100" in result[5]["final_attempts"][1]["evidence"]


def test_all_failed_generation_has_no_successful_latency(monkeypatch):
    result = _offline_score(monkeypatch, [[_attempt(success=False)]])
    for p in (50, 95):
        assert result[1][f"generation_p{p}_seconds"] is None
        bounds = result[4][f"operational.generation_p{p}_seconds"]
        assert bounds["estimate"] is None
        assert bounds["lower"] is None and bounds["upper"] is None
    assert result[2]["reliability.request_failure_rate"] == 1


def test_historical_malformed_completion_does_not_supply_successful_latency(
    monkeypatch,
):
    result = _offline_score(monkeypatch, [[_attempt(schema_valid=False)]])
    assert result[1]["generation_p50_seconds"] is None


@pytest.fixture
def execution_case(monkeypatch):
    manifest = load_manifest(manifest_path("smoke"))
    chunks = [
        Chunk(row["id"] + ":whole", row["id"], row["text"], 0, len(row["text"]), 10)
        for row in manifest.samples
        if row.get("kind") == "document"
    ]
    monkeypatch.setattr(benchmark, "release_runtime_memory", lambda: None)

    class Server:
        identity = "test-only"
        reranker = None
        index = SimpleNamespace(chunks=chunks, embedder=None)

        def retrieve(self, question, *, evidence):
            evidence.update(
                pool=list(range(len(chunks))), ranking=list(range(len(chunks)))
            )

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
    composition = SimpleNamespace(
        checksum="test-only-composition",
        protocols={},
        system={"residency": "resident", "top_k": 3},
    )

    def execute(protocol, backend=None):
        raw = benchmark.execute(
            "qwen3-0.6b-direct",
            manifest,
            {},
            plan,
            protocol,
            composition,
            Server(),
            backend_factory=lambda *a: backend or Backend(),
            monitor_factory=Monitor,
        )
        assert raw["error"] is None
        raw["chunk_inventory"] = [asdict(chunk) for chunk in chunks]
        return raw

    return manifest, execute


@pytest.mark.parametrize("outcome", ["timeout", "context_limit", "runtime_error"])
def test_execute_labels_failed_generation_elapsed_not_completed_latency(
    execution_case, outcome
):
    _, execute = execution_case
    raw = execute(load_protocol(), Backend(failure=outcome))
    for record in raw["records"]:
        attempt = record["attempts"][0]
        assert attempt["request_status"] == "failed"
        assert attempt.get("generation_latency_seconds") is None
        assert attempt["generation_elapsed_to_failure_seconds"] >= 0
        assert attempt["timing"]["ttft_seconds"] == 0.1


@pytest.fixture
def bound_execution(execution_case, tmp_path):
    manifest, execute = execution_case
    questions = questions_from_manifest(manifest, require_supplied_evidence=False)
    protocol, *_ = calibrated_fixture.__wrapped__(
        (load_protocol(), manifest, questions), tmp_path
    )
    return protocol, manifest, execute(protocol), questions


@pytest.mark.parametrize(
    "damage", ["contract_sha256", "checksum", "missing_hash", "missing_checksum"]
)
def test_judge_mismatch_is_rejected_before_any_evaluation(bound_execution, damage):
    protocol, manifest, raw, _ = bound_execution
    contract = dict(
        SemanticJudge(protocol.judge_contract_path, lambda *a, **kw: None).contract
    )
    if damage.startswith("missing_"):
        contract.pop("contract_sha256" if damage == "missing_hash" else "checksum")
    else:
        contract[damage] = "different-test-only-identity"
    judge = SimpleNamespace(
        contract=contract, response=lambda *a: pytest.fail("mismatched judge called")
    )
    with pytest.raises(ValueError, match="judge identity mismatch"):
        benchmark.score(raw, manifest, protocol, retrieval_protocol(), judge=judge)


def test_matching_bound_judge_scores_without_rewriting_saved_execution(bound_execution):
    protocol, manifest, raw, questions = bound_execution
    judge = SemanticJudge(protocol.judge_contract_path, fixture_transport(questions[0]))
    before = stable_hash(raw)
    assert benchmark.validate_final_schedule(raw, protocol, "smoke", judge=judge)
    result = benchmark.score(raw, manifest, protocol, retrieval_protocol(), judge=judge)
    assert result[2]["quality.faithfulness"] == 1
    assert stable_hash(raw) == before


def test_unbound_saved_schedule_cannot_adopt_a_configured_judge(execution_case):
    manifest, execute = execution_case
    protocol = load_protocol()
    raw = execute(protocol)
    judge = SimpleNamespace(
        contract={"contract_sha256": "test-only", "checksum": "test-only"}
    )
    with pytest.raises(ValueError, match="judge identity mismatch"):
        benchmark.score(raw, manifest, protocol, retrieval_protocol(), judge=judge)
