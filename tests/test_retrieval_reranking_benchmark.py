from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from edumind.common.artifacts import stable_hash
from edumind.common.paths import PROJECT_ROOT
from edumind.rag.contracts import PRODUCTION_EMBEDDING_MODEL, EmbeddingSpec
from edumind.rag.tokenizers import TiktokenOffsetTokenizer
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    CandidateResult,
    DatasetManifest,
    SampleResult,
)
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    load_protocol as default_chunking_protocol,
)
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    protocol_from_mapping as chunking_protocol_from_mapping,
)
from experiments.benchmarks.rag.chunking_embedding.strategies import (
    build_chunking_strategy,
)
from experiments.benchmarks.rag.evaluation import Chunk, ExactIndex
from experiments.benchmarks.rag.methods import reciprocal_rank_fusion_with_scores
from experiments.benchmarks.rag.retrieval_reranking import (
    benchmark as retrieval_benchmark,
)
from experiments.benchmarks.rag.retrieval_reranking import run as retrieval_run
from experiments.benchmarks.rag.retrieval_reranking.benchmark import (
    _require_permutation,
    evaluate_candidate,
)
from experiments.benchmarks.rag.retrieval_reranking.comparisons import (
    parent_artifact_builder,
)
from experiments.benchmarks.rag.retrieval_reranking.metrics import (
    directions_for,
    pool_evidence_unit_recall,
)
from experiments.benchmarks.rag.retrieval_reranking.profiles import (
    development_candidates,
    owner_first,
    parse_candidate,
)
from experiments.benchmarks.rag.retrieval_reranking.protocol import (
    load_protocol as default_protocol,
)
from experiments.benchmarks.rag.retrieval_reranking.protocol import (
    protocol_from_mapping,
    protocol_from_settings,
)

RETRIEVAL_CANDIDATES = owner_first(development_candidates())


def _settings(
    *,
    profile: str = "development",
    warmups: int | None = None,
    repetitions: int | None = None,
    bootstrap_resamples: int | None = None,
    **overrides: object,
) -> dict[str, object]:
    payload = deepcopy(default_protocol().meta.resolved)
    payload["statistics"]["minimum_quality_ci_documents"] = 2
    payload["statistics"]["minimum_repeatability_ci_documents"] = 2
    execution = payload["profiles"][profile]
    if warmups is not None:
        execution["warmups"] = warmups
    if repetitions is not None:
        execution["repetitions"] = repetitions
    if bootstrap_resamples is not None:
        execution["bootstrap_resamples"] = bootstrap_resamples
    protocol = protocol_from_mapping(payload)
    chunking_payload = deepcopy(default_chunking_protocol().meta.resolved)
    chunking_execution = chunking_payload["profiles"][profile]
    chunking_execution.update(
        {
            "warmups": execution["warmups"],
            "repetitions": execution["repetitions"],
            "bootstrap_resamples": execution["bootstrap_resamples"],
            "device": execution["device"],
            "dtype": execution["dtype"],
        }
    )
    chunking_protocol = chunking_protocol_from_mapping(chunking_payload)
    return {
        "retrieval_protocol": protocol.meta.worker_payload(),
        "chunking_embedding_protocol": chunking_protocol.meta.worker_payload(),
        **overrides,
    }


def test_candidate_matrix_is_full_cross_and_owners_run_first() -> None:
    candidates = RETRIEVAL_CANDIDATES
    assert len(candidates) == len(set(candidates)) == 15
    assert candidates[:3] == ("dense|none", "bm25|none", "rrf|none")
    assert {
        (parse_candidate(value).retriever, parse_candidate(value).reranker)
        for value in candidates
    } == {
        (retriever, reranker)
        for retriever in ("dense", "bm25", "rrf")
        for reranker in (
            "none",
            "gte-modernbert",
            "ettin-150m",
            "ettin-400m",
            "ettin-1b",
        )
    }


@pytest.mark.parametrize(
    ("chunker", "expected_limit"),
    (("semantic", 3584.0), ("token-256-32", None)),
)
def test_bm25_preflight_applies_gpu_gate_when_chunking_is_model_backed(
    chunker: str,
    expected_limit: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = {}

    def worker(*_args, **kwargs):
        observed.update(kwargs)
        return {}

    monkeypatch.setattr(retrieval_benchmark, "run_json_worker", worker)
    manifest = _fixture_manifest()
    plan = BenchmarkPlan(
        "rag",
        "retrieval-reranking",
        "development",
        manifest.name,
        ("bm25|none",),
        seed=42,
        repetitions=1,
        bootstrap_resamples=0,
        warmups=0,
        settings=_settings(chunker_embedding=f"{chunker}|{PRODUCTION_EMBEDDING_MODEL}"),
    )

    result = retrieval_benchmark.preflight_in_fresh_process(
        "bm25|none",
        manifest,
        {},
        plan,
        vram_limit_mb=3584.0,
        preflight=default_protocol().preflight,
    )

    assert observed["vram_limit_mb"] == expected_limit
    if expected_limit is None:
        assert result["vram_measurement_method"] == "not-applicable"


def test_validation_runs_exactly_the_engineer_selected_candidates(monkeypatch) -> None:
    selected = ("bm25|ettin-150m", "dense|ettin-1b")
    monkeypatch.setattr(
        retrieval_run,
        "load_engineer_decision",
        lambda *_args, **_kwargs: SimpleNamespace(selected_candidates=selected),
    )
    assert (
        retrieval_run._selected_candidates(
            "validation",
            Path("decision.json"),
            RETRIEVAL_CANDIDATES,
            3,
            protocols={"retrieval": "test-protocol"},
            qualification="test-qualification",
        )
        == selected
    )


def test_locked_retrieval_requires_exactly_one_candidate(monkeypatch) -> None:
    def decision(_path, **options):
        assert options["exact"] == 1
        return SimpleNamespace(selected_candidates=("rrf|ettin-400m",))

    monkeypatch.setattr(retrieval_run, "load_engineer_decision", decision)
    assert retrieval_run._selected_candidates(
        "locked",
        Path("decision.json"),
        RETRIEVAL_CANDIDATES,
        3,
        protocols={"retrieval": "test-protocol"},
        qualification="test-qualification",
    ) == ("rrf|ettin-400m",)


def test_pool_recall_requires_complete_evidence_unit() -> None:
    question = {
        "id": "q1",
        "evidence_type": "text",
        "evidence": [
            {
                "id": "e1",
                "document_id": "doc",
                "start": 2,
                "end": 8,
            }
        ],
    }
    partial = Chunk("partial", "doc", "cdef", 2, 6, 1)
    complete = Chunk("complete", "doc", "abcdefghij", 0, 10, 2)
    assert pool_evidence_unit_recall(question, [partial]) == 0.0
    assert pool_evidence_unit_recall(question, [complete]) == 1.0


def test_reranker_must_return_exact_pool_permutation() -> None:
    pool = [(0, 0.9), (1, 0.8), (2, 0.7)]
    _require_permutation([(2, 3.0), (0, 2.0), (1, 1.0)], pool, "q1")
    with pytest.raises(ValueError, match="not a pool permutation"):
        _require_permutation([(2, 3.0), (2, 2.0), (1, 1.0)], pool, "q1")


def test_reranker_input_limits_match_frozen_model_contracts() -> None:
    assert default_protocol().reranker_maximum_tokens == {
        "gte-modernbert": 8192,
        "ettin-150m": 7999,
        "ettin-400m": 7999,
        "ettin-1b": 7999,
    }


def test_protocol_is_strict_and_plan_checksum_is_verified() -> None:
    protocol = default_protocol()
    malformed = deepcopy(protocol.meta.resolved)
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="unknown unexpected"):
        protocol_from_mapping(malformed)
    nonfinite = deepcopy(protocol.meta.resolved)
    nonfinite["retrieval"]["bm25"]["k1"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        protocol_from_mapping(nonfinite)
    with pytest.raises(ValueError, match="checksum"):
        protocol_from_settings(
            {
                "retrieval_protocol": {
                    **protocol.meta.worker_payload(),
                    "checksum": "wrong",
                },
            }
        )
    wrong_version = _settings()
    wrong_version["retrieval_protocol"]["protocol_version"] = "wrong"
    with pytest.raises(ValueError, match="version"):
        protocol_from_settings(wrong_version)


def test_bm25_freezes_documented_parameters(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class FixtureBM25:
        def __init__(self, documents, **kwargs) -> None:
            observed["documents"] = documents
            observed.update(kwargs)

    import rank_bm25

    from experiments.benchmarks.rag import methods

    monkeypatch.setattr(rank_bm25, "BM25Okapi", FixtureBM25)

    protocol = default_protocol()
    methods.BM25(
        ["First document", "second document"],
        k1=protocol.bm25_k1,
        b=protocol.bm25_b,
        epsilon=protocol.bm25_epsilon,
    )
    assert observed == {
        "documents": [["first", "document"], ["second", "document"]],
        "k1": 1.5,
        "b": 0.75,
        "epsilon": 0.25,
    }


def test_rrf_uses_documented_stable_tie_breaking() -> None:
    result = reciprocal_rank_fusion_with_scores(
        [[0], [1]],
        20,
        60,
        weights=(1.0, 1.0),
        tie_keys={0: "z-chunk", 1: "a-chunk"},
    )
    assert [identifier for identifier, _ in result] == [1, 0]


def test_smoke_fixture_produces_exactly_thirty_frozen_chunks() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/smoke.json")
    tokenizer = TiktokenOffsetTokenizer("cl100k_base")
    chunker = build_chunking_strategy(
        "token-256-32", tokenizer=tokenizer, protocol=default_chunking_protocol()
    )
    chunks = [
        row
        for sample in manifest.samples
        if sample.get("kind") == "document"
        for row in chunker.split(str(sample["text"]))
    ]
    assert len(chunks) == 30


def test_metric_contract_excludes_obsolete_retrieval_metrics() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/smoke.json")
    protocol = default_protocol()
    directions, _ = directions_for(manifest, protocol)
    assert "quality.overall.ndcg_at_3" in directions
    assert "quality.overall.evidence_unit_recall_at_5" in directions
    assert "validity.ranking_agreement" not in directions
    assert directions["diagnostic.ranking_repeatability"] == "max"
    assert "validity.pool_collection_match" not in directions
    assert directions["validity.failed_query_count"] == "descriptive"
    assert (
        directions["diagnostic.pool_evidence_unit_recall_at_20.sample_count"]
        == "descriptive"
    )
    assert not any(
        obsolete in metric
        for metric in directions
        for obsolete in ("mrr", "hit_rate", "map_at", "context_recall")
    )


def test_protocol_drives_cutoff_and_pool_metric_names() -> None:
    payload = deepcopy(default_protocol().meta.resolved)
    payload["retrieval"]["pool_size"] = 4
    payload["retrieval"]["rrf"]["source_depth"] = 4
    payload["quality"]["cutoffs"] = [2]
    protocol = protocol_from_mapping(payload)
    directions, _ = directions_for(_fixture_manifest(), protocol)
    assert "quality.overall.ndcg_at_2" in directions
    assert "quality.overall.ndcg_at_3" not in directions
    assert protocol.pool_recall_metric == ("diagnostic.pool_evidence_unit_recall_at_4")


class _FixtureTokenizer:
    name = "fixture"

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(index, index + 1) for index in range(len(text))]


class _FixtureBM25:
    serialized = b"fixture-bm25"

    @property
    def storage_bytes(self) -> int:
        return len(self.serialized)

    def rank(self, query: str, limit: int) -> list[tuple[int, float]]:
        del query
        return [(position, float(limit - position)) for position in range(limit)]


class _FixtureReranker:
    model_name = "fixture-reranker"
    revision = "fixture-revision"
    model_path = "."
    maximum_length = 8192
    batch_size = 1

    def prepare(self) -> None:
        pass

    def input_token_counts(self, query: str, documents: list[str]) -> list[int]:
        del query
        return [len(document) for document in documents]

    def rank_with_scores(
        self,
        query: str,
        documents: list[str],
        *,
        validate_inputs: bool = True,
    ) -> list[tuple[int, float]]:
        del query, validate_inputs
        return [
            (position, float(position)) for position in reversed(range(len(documents)))
        ]


def _fixture_manifest() -> DatasetManifest:
    text = "verified evidence followed by ordinary context"
    return DatasetManifest(
        "retrieval-fixture",
        "1",
        "rag",
        "development",
        "fixture",
        "CC0-1.0",
        "1",
        "fixture-checksum",
        "fixture-v1",
        42,
        (
            {"id": "doc", "kind": "document", "text": text},
            {
                "id": "q1",
                "kind": "question",
                "question": "Where is the verified evidence?",
                "document_id": "doc",
                "answerable": True,
                "evidence_type": "text",
                "evidence": [
                    {
                        "id": "e1",
                        "document_id": "doc",
                        "start": 0,
                        "end": 8,
                    }
                ],
            },
        ),
    )


def _fixture_index() -> ExactIndex:
    chunks = [
        Chunk(
            f"doc:{position}",
            "doc",
            "verified evidence" if position == 0 else f"decoy {position}",
            0 if position == 0 else 18,
            17 if position == 0 else 25,
            2,
        )
        for position in range(30)
    ]
    return ExactIndex(
        chunks=chunks,
        vectors=None,
        embedder=None,
        tokenizer=_FixtureTokenizer(),
        bm25=_FixtureBM25(),
        embedding_spec=EmbeddingSpec(
            PRODUCTION_EMBEDDING_MODEL,
            "revision",
            PRODUCTION_EMBEDDING_MODEL,
            ".",
            "",
            "",
            True,
            768,
            "cosine",
            8192,
            "cls",
        ),
        chunking_fingerprint="chunking-fingerprint",
        chunking_contract={"name": "fixture"},
        corpus_build_seconds=0.0,
        phase_seconds={
            "dense_index_build_seconds": 0.03,
            "bm25_index_build_seconds": 0.07,
            "index_build_seconds": 0.1,
        },
        preflight={"status": "passed", "truncated_inputs": 0},
    )


def test_index_identity_is_exact_and_retriever_specific() -> None:
    candidate = parse_candidate("bm25|none")
    original = _fixture_index()
    changed_text = _fixture_index()
    first = changed_text.chunks[0]
    changed_text.chunks[0] = Chunk(
        first.identifier,
        first.document_id,
        "different evidence",
        first.start,
        first.end,
        first.tokens,
        first.model_tokens,
    )
    changed_strategy = _fixture_index()
    changed_strategy.chunking_fingerprint = "different-chunking-fingerprint"

    original_artifact = retrieval_benchmark._index_build_artifact(original, candidate)
    changed_text_artifact = retrieval_benchmark._index_build_artifact(
        changed_text, candidate
    )
    changed_strategy_artifact = retrieval_benchmark._index_build_artifact(
        changed_strategy, candidate
    )

    assert original_artifact["chunks_sha256"] != changed_text_artifact["chunks_sha256"]
    assert (
        original_artifact["bm25_index_sha256"]
        == changed_text_artifact["bm25_index_sha256"]
    )
    assert (
        original_artifact["chunks_sha256"] == changed_strategy_artifact["chunks_sha256"]
    )
    assert (
        original_artifact["chunking_fingerprint"]
        != changed_strategy_artifact["chunking_fingerprint"]
    )
    assert "dense_index_sha256" not in original_artifact
    assert "bm25_index_sha256" in original_artifact
    assert "index_sha256" not in original_artifact
    assert "chunk_ids_sha256" not in original_artifact

    dense = _fixture_index()
    dense.vectors = np.arange(60, dtype=np.float32).reshape(30, 2)
    dense.bm25 = None
    dense_artifact = retrieval_benchmark._index_build_artifact(
        dense, parse_candidate("dense|none")
    )
    assert "dense_index_sha256" in dense_artifact
    assert "bm25_index_sha256" not in dense_artifact

    rrf = _fixture_index()
    rrf.vectors = np.arange(60, dtype=np.float32).reshape(30, 2)
    rrf_artifact = retrieval_benchmark._index_build_artifact(
        rrf, parse_candidate("rrf|none")
    )
    assert "dense_index_sha256" in rrf_artifact
    assert "bm25_index_sha256" in rrf_artifact


def test_owner_pool_is_reused_by_every_reranker_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _fixture_manifest()
    index = _fixture_index()
    monkeypatch.setattr(
        retrieval_benchmark, "build_index", lambda *_args, **_kwargs: index
    )
    model_lock = {
        PRODUCTION_EMBEDDING_MODEL: {
            "revision": "revision",
            "model_path": ".",
            "model_cache_manifest_sha256": "embedding-sha",
        },
        "cross-encoder/ettin-reranker-150m-v1": {
            "revision": "reranker-revision",
            "model_path": ".",
            "model_cache_manifest_sha256": "reranker-sha",
        },
    }
    plan = BenchmarkPlan(
        "rag",
        "retrieval-reranking",
        "smoke",
        manifest.name,
        ("bm25|none", "bm25|ettin-150m"),
        seed=default_protocol().meta.seed,
        repetitions=2,
        bootstrap_resamples=0,
        warmups=0,
        settings=_settings(
            profile="smoke",
            warmups=0,
            repetitions=2,
            bootstrap_resamples=0,
            chunker_embedding=f"token-256-32|{PRODUCTION_EMBEDDING_MODEL}",
        ),
    )
    owner = evaluate_candidate(
        "bm25|none",
        manifest,
        model_lock,
        plan,
        device="cpu",
        dtype="float32",
        frozen_pool_collection=None,
        child_run_id="owner-run",
    )
    rows = owner[5]["pool_collection"]
    assert len(rows) == 20
    assert owner[3]["bm25_k1"] == 1.5
    assert owner[3]["bm25_b"] == 0.75
    assert owner[3]["bm25_epsilon"] == 0.25
    assert "dense_similarity" not in owner[3]
    assert "rrf_union" not in owner[3]
    assert "retrieval_contract" not in owner[3]
    assert "reranker_contract" not in owner[3]
    assert "reranker_batch_size" not in owner[3]
    index_build = owner[5]["index_build"]
    frozen_pool_collection = {
        "owner_candidate": "bm25|none",
        "owner_child_run_id": "owner-run",
        "pool_collection_checksum": stable_hash(rows),
        "retriever": index_build["retriever"],
        "chunks_sha256": index_build["chunks_sha256"],
        "bm25_index_sha256": index_build["bm25_index_sha256"],
        "rows": rows,
    }
    monkeypatch.setattr(
        retrieval_benchmark,
        "_reranker",
        lambda *_args, **_kwargs: _FixtureReranker(),
    )
    with pytest.raises(CandidateExecutionError, match="index identity mismatch"):
        evaluate_candidate(
            "bm25|ettin-150m",
            manifest,
            model_lock,
            plan,
            device="cpu",
            dtype="float32",
            frozen_pool_collection={
                **frozen_pool_collection,
                "bm25_index_sha256": "different-index",
            },
            child_run_id="child-run",
        )
    with pytest.raises(CandidateExecutionError, match="unexpected dense_index_sha256"):
        evaluate_candidate(
            "bm25|ettin-150m",
            manifest,
            model_lock,
            plan,
            device="cpu",
            dtype="float32",
            frozen_pool_collection={
                **frozen_pool_collection,
                "dense_index_sha256": "unexpected",
            },
            child_run_id="child-run",
        )
    child = evaluate_candidate(
        "bm25|ettin-150m",
        manifest,
        model_lock,
        plan,
        device="cpu",
        dtype="float32",
        frozen_pool_collection=frozen_pool_collection,
        child_run_id="child-run",
    )
    assert "pool_collection" not in child[5]
    assert child[3]["pool_owner_child_run_id"] == "owner-run"
    assert child[3]["pool_collection_checksum"] == stable_hash(rows)
    assert child[3]["chunks_sha256"] == owner[3]["chunks_sha256"]
    assert child[3]["bm25_index_sha256"] == owner[3]["bm25_index_sha256"]
    assert "dense_index_sha256" not in child[3]
    assert child[3]["reranker_model"] == "fixture-reranker"
    assert child[3]["reranker_batch_size"] == 1
    assert child[3]["reranker_input_template"] == "tokenizer(query, passage)"
    assert "reranker_contract" not in child[3]
    assert "validity.pool_collection_match" not in child[2]
    assert child[2]["validity.exact_pool_permutation"] == 1.0
    assert child[2]["diagnostic.ranking_repeatability"] == 1.0
    assert len(child[5]["timings"]) == 2


def test_measured_failure_keeps_every_timing_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _fixture_manifest()
    index = _fixture_index()
    calls = 0

    def rank(_query: str, limit: int) -> list[tuple[int, float]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        return [(position, float(limit - position)) for position in range(limit)]

    monkeypatch.setattr(index.bm25, "rank", rank)
    monkeypatch.setattr(
        retrieval_benchmark, "build_index", lambda *_args, **_kwargs: index
    )
    plan = BenchmarkPlan(
        "rag",
        "retrieval-reranking",
        "smoke",
        manifest.name,
        ("bm25|none",),
        seed=default_protocol().meta.seed,
        repetitions=3,
        bootstrap_resamples=0,
        warmups=0,
        settings=_settings(
            profile="smoke",
            warmups=0,
            repetitions=3,
            bootstrap_resamples=0,
            chunker_embedding=f"token-256-32|{PRODUCTION_EMBEDDING_MODEL}",
        ),
    )
    evaluated = evaluate_candidate(
        "bm25|none",
        manifest,
        {
            PRODUCTION_EMBEDDING_MODEL: {
                "revision": "revision",
                "model_path": ".",
                "model_cache_manifest_sha256": "embedding-sha",
            }
        },
        plan,
        device="cpu",
        dtype="float32",
        frozen_pool_collection=None,
        child_run_id="owner-run",
    )
    assert calls == 3
    assert len(evaluated[5]["timings"]) == 3
    assert evaluated[2]["validity.failed_query_count"] == 1.0
    assert evaluated[2]["diagnostic.ranking_repeatability"] == pytest.approx(1 / 3)
    assert evaluated[2]["quality.overall.ndcg_at_3"] is not None


def test_parent_comparisons_are_separated_and_csv_matches_parquet(
    tmp_path: Path,
) -> None:
    directions = {
        "quality.overall.ndcg_at_3": "max",
        "diagnostic.pool_evidence_unit_recall_at_20": "max",
        "operational.full_stack_latency_ms_p50": "min",
        "operational.retrieval_latency_ms_p50": "min",
        "workload.retrieved_context_tokens_at_3_mean": "descriptive",
        "storage.index_size_bytes": "descriptive",
    }
    pool_collection_checksums = {
        retriever: stable_hash([{"retriever": retriever}])
        for retriever in ("dense", "bm25", "rrf")
    }
    results = []
    for position, candidate in enumerate(RETRIEVAL_CANDIDATES):
        parsed = parse_candidate(candidate)
        value = 0.5 + position / 100.0
        samples = (
            SampleResult(
                "q1",
                {
                    "quality.overall.ndcg_at_3": value,
                    "diagnostic.pool_evidence_unit_recall_at_20": value,
                },
                0.01,
                {"document_id": "d1", "evidence_type": "text"},
            ),
            SampleResult(
                "q2",
                {
                    "quality.overall.ndcg_at_3": value + 0.01,
                    "diagnostic.pool_evidence_unit_recall_at_20": value + 0.01,
                },
                0.02,
                {"document_id": "d2", "evidence_type": "text"},
            ),
        )
        results.append(
            CandidateResult(
                candidate=candidate,
                status="success",
                fingerprint=f"fingerprint-{position}",
                metrics={
                    "quality.overall.ndcg_at_3": value,
                    "diagnostic.pool_evidence_unit_recall_at_20": value,
                    "workload.retrieved_context_tokens_at_3_mean": 100.0 + position,
                    "storage.index_size_bytes": 1000.0 + position,
                },
                intervals={},
                samples=samples,
                operational={
                    "full_stack_latency_ms_p50": 10.0 + position,
                    "retrieval_latency_ms_p50": 5.0 + position,
                },
                mlflow_run_id=f"run-{position}",
                parameters={
                    "pool_collection_checksum": pool_collection_checksums[
                        parsed.retriever
                    ]
                },
            )
        )
    pool_collections = {
        f"{retriever}|none": {
            "owner_child_run_id": f"owner-{retriever}",
            "pool_collection_checksum": checksum,
            "retriever": retriever,
            "chunks_sha256": f"chunks-{retriever}",
            **(
                {"dense_index_sha256": f"dense-{retriever}"}
                if retriever in {"dense", "rrf"}
                else {}
            ),
            **(
                {"bm25_index_sha256": f"bm25-{retriever}"}
                if retriever in {"bm25", "rrf"}
                else {}
            ),
            "rows": [{"retriever": retriever}],
        }
        for retriever, checksum in pool_collection_checksums.items()
    }
    plan = BenchmarkPlan(
        "rag",
        "retrieval-reranking",
        "validation",
        "fixture",
        RETRIEVAL_CANDIDATES,
        seed=default_protocol().meta.seed,
        repetitions=1,
        bootstrap_resamples=100,
        warmups=default_protocol().profile("validation").warmups,
        settings=_settings(
            profile="validation",
            repetitions=1,
            bootstrap_resamples=100,
            requested_finalists=[
                "dense|ettin-150m",
                "bm25|ettin-400m",
            ],
        ),
    )
    paths = parent_artifact_builder(
        pool_collections, directions, compare_finalists=True
    )(tmp_path, results, plan)
    assert {path.name for path in paths} == {
        "reranker_comparisons.parquet",
        "reranker_comparisons.csv",
        "retriever_comparisons.parquet",
        "retriever_comparisons.csv",
        "finalist_comparisons.parquet",
        "finalist_comparisons.csv",
        "pool_collections.json",
    }
    reranker = pd.read_parquet(tmp_path / "reranker_comparisons.parquet")
    retriever = pd.read_parquet(tmp_path / "retriever_comparisons.parquet")
    assert (
        reranker[["baseline_candidate", "candidate"]].drop_duplicates().shape[0] == 12
    )
    assert (
        retriever[["baseline_candidate", "candidate"]].drop_duplicates().shape[0] == 3
    )
    assert reranker["shared_pool_collection_checksum"].notna().all()
    assert retriever["baseline_pool_collection_checksum"].notna().all()
    assert retriever["candidate_pool_collection_checksum"].notna().all()
    pool_recall = retriever[
        retriever["metric_name"] == "diagnostic.pool_evidence_unit_recall_at_20"
    ]
    assert pool_recall["evidence_slice"].eq("overall").all()
    assert pool_recall["cutoff"].eq(20).all()
    assert pool_recall["ci_status"].eq("available").all()
    finalists = pd.read_parquet(tmp_path / "finalist_comparisons.parquet")
    assert (
        finalists[["baseline_candidate", "candidate"]].drop_duplicates().shape[0] == 1
    )
    workload = reranker[
        reranker["metric_name"] == "workload.retrieved_context_tokens_at_3_mean"
    ]
    assert workload["direction"].eq("descriptive").all()
    assert workload["favors_candidate"].isna().all()
    pool_collections_artifact = json.loads(
        (tmp_path / "pool_collections.json").read_text()
    )
    assert len(pool_collections_artifact["pool_collections"]) == 3
    by_retriever = {
        row["retriever"]: row for row in pool_collections_artifact["pool_collections"]
    }
    assert "bm25_index_sha256" not in by_retriever["dense"]
    assert "dense_index_sha256" not in by_retriever["bm25"]
    assert {
        "dense_index_sha256",
        "bm25_index_sha256",
    } <= by_retriever["rrf"].keys()
    assert all(
        "index_checksum" not in row
        for row in pool_collections_artifact["pool_collections"]
    )
    assert pool_collections_artifact["comparison_artifacts"]["reranker_comparisons"][
        "logical_table_sha256"
    ]
    assert not (tmp_path / "retrieval_protocol.json").exists()


def _rr_plan(*, warmups=0, repetitions=3):
    manifest = _fixture_manifest()
    return BenchmarkPlan(
        "rag",
        "retrieval-reranking",
        "smoke",
        manifest.name,
        ("bm25|none",),
        seed=42,
        warmups=warmups,
        repetitions=repetitions,
        bootstrap_resamples=0,
        settings=_settings(
            profile="smoke",
            warmups=warmups,
            repetitions=repetitions,
            bootstrap_resamples=0,
            chunker_embedding=f"token-256-32|{PRODUCTION_EMBEDDING_MODEL}",
        ),
    )


def _rr_lock():
    entry = {
        "revision": "revision",
        "model_path": ".",
        "model_cache_manifest_sha256": "sha",
    }
    return {
        PRODUCTION_EMBEDDING_MODEL: entry,
        "cross-encoder/ettin-reranker-150m-v1": entry,
    }


def _rr_evaluate(
    monkeypatch,
    *,
    index=None,
    plan=None,
    candidate="bm25|none",
    frozen=None,
    preparation_only=False,
):
    monkeypatch.setattr(
        retrieval_benchmark, "build_index", lambda *a, **kw: index or _fixture_index()
    )
    return evaluate_candidate(
        candidate,
        _fixture_manifest(),
        _rr_lock(),
        plan or _rr_plan(),
        device="cpu",
        dtype="float32",
        frozen_pool_collection=frozen,
        child_run_id=None if preparation_only else "run",
        preparation_only=preparation_only,
    )


def _frozen(evaluated):
    build = evaluated[5]["index_build"]
    return {
        "owner_candidate": "bm25|none",
        "owner_child_run_id": None,
        "pool_collection_checksum": evaluated[3]["pool_collection_checksum"],
        "rows": evaluated[5]["pool_collection"],
        **{
            key: build[key]
            for key in ("retriever", "chunks_sha256", "bm25_index_sha256")
        },
    }


def test_one_attempt_smoke_repeatability_is_inapplicable_not_missing(monkeypatch):
    result = _rr_evaluate(monkeypatch, plan=_rr_plan(repetitions=1))
    name = "diagnostic.ranking_repeatability"
    details = result[5]["validation_report"]["metric_support"][name]
    assert result[2][name] is None
    assert details["status"] == "inapplicable"
    assert details["unavailable_question_count"] == 0
    assert details["eligible_question_count"] == 0
    assert result[4][name]["reason"] == "repeatability_not_measured"


def test_none_owner_has_only_one_warmup_and_scheduled_retrievals(monkeypatch):
    index, calls = _fixture_index(), []
    original = index.bm25.rank
    monkeypatch.setattr(
        index.bm25,
        "rank",
        lambda query, limit: calls.append(query) or original(query, limit),
    )
    evaluated = _rr_evaluate(monkeypatch, index=index, plan=_rr_plan(warmups=1))
    assert len(calls) == 4
    assert {row["repetition"] for row in evaluated[5]["rankings"]} == {1, 2, 3}
    assert evaluated[1]["index_build_seconds"] == 0.07
    assert "storage.reranker_snapshot_bytes" not in evaluated[2]
    assert not any(name.startswith("workload.corpus") for name in evaluated[2])


@pytest.mark.parametrize("failures,expected", [({1}, 1 / 3), ({1, 2, 3}, 0.0)])
def test_failed_first_never_uses_later_quality_and_keeps_outcomes(
    monkeypatch, failures, expected
):
    index, calls = _fixture_index(), 0
    original = index.bm25.rank

    def rank(query, limit):
        nonlocal calls
        calls += 1
        if calls in failures:
            raise RuntimeError("recoverable request failure")
        return original(query, limit)

    monkeypatch.setattr(index.bm25, "rank", rank)
    with pytest.raises(CandidateExecutionError) as caught:
        _rr_evaluate(monkeypatch, index=index)
    error = caught.value
    assert calls == 3
    assert error.metrics["quality.overall.ndcg_at_3"] is None
    assert error.metrics["diagnostic.ranking_repeatability"] == pytest.approx(expected)
    assert error.metrics["validity.processed_query_count"] == 1
    assert error.metrics["validity.failed_query_count"] == 1
    assert len(error.artifacts["timings"]) == 3
    assert error.operational["index_build_seconds"] == 0.07
    assert error.metrics["storage.index_size_bytes"] == len(index.bm25.serialized)


def test_malformed_ranking_retains_raw_fragment_and_completed_build(monkeypatch):
    index = _fixture_index()
    monkeypatch.setattr(index.bm25, "rank", lambda *a: ["bad row", [0, "bad score"]])
    with pytest.raises(CandidateExecutionError) as caught:
        _rr_evaluate(monkeypatch, index=index)
    rankings = caught.value.artifacts["rankings"]
    assert any("bad row" in str(row.get("raw_output_json")) for row in rankings)
    assert caught.value.operational["index_build_seconds"] == 0.07
    assert caught.value.metrics["quality.overall.evidence_token_precision_at_3"] is None


def test_preparation_only_is_explicit_and_native_validation_is_measured(monkeypatch):
    index = _fixture_index()
    prepared = _rr_evaluate(monkeypatch, index=index, preparation_only=True)
    assert prepared[0] == []
    assert prepared[5]["timings"] == []
    assert prepared[3]["pool_owner_child_run_id"] is None
    events = []

    class Reranker(_FixtureReranker):
        def input_token_counts(self, query, documents):
            events.append("validate")
            return super().input_token_counts(query, documents)

        def rank_with_scores(self, query, documents, **kwargs):
            events.append("rank")
            return super().rank_with_scores(query, documents, **kwargs)

    monkeypatch.setattr(retrieval_benchmark, "_reranker", lambda *a, **kw: Reranker())
    child = _rr_evaluate(
        monkeypatch,
        index=index,
        candidate="bm25|ettin-150m",
        frozen=_frozen(prepared),
        plan=_rr_plan(warmups=1),
    )
    assert events == ["validate", "rank"] * 4
    assert child[5]["validation_report"]["native_inputs"]["q1"]["total"] > 0
    assert child[2]["workload.reranker_input_tokens_per_query_mean"] > 0
    assert "storage.index_size_bytes" not in child[2]


def test_repeatability_all_three_pairs_full_order_and_missing_outcomes():
    from experiments.benchmarks.rag.retrieval_reranking.metrics import repeatability

    def attempt(rep, ids=None, status="success"):
        return {
            "repetition": rep,
            "status": status,
            "valid": status == "success",
            "ordered_chunk_ids": ids,
        }

    a, b = list(range(20)), [*range(18), 19, 18]
    result = repeatability([attempt(1, a), attempt(2, a), attempt(3, b)], 3)
    assert result["value"] == pytest.approx(1 / 3)
    assert len(result["pairs"]) == 3
    assert repeatability(
        [attempt(1, status="failed"), attempt(2, a), attempt(3, a)], 3
    )["value"] == pytest.approx(1 / 3)
    assert (
        repeatability(
            [attempt(1, a), attempt(2, a), attempt(3, status="not_executed")], 3
        )["value"]
        is None
    )
    assert repeatability([attempt(1, a)], 1)["status"] == "inapplicable"


def test_repeatability_source_ci_policy_and_degenerate_bounds():
    from experiments.benchmarks.rag.retrieval_reranking.metrics import aggregate_quality

    key = "diagnostic.ranking_repeatability"
    samples = [
        SampleResult(
            str(i), {key: 1.0}, None, {"document_id": str(i), "evidence_type": "text"}
        )
        for i in range(2)
    ]
    values, intervals, _ = aggregate_quality(
        samples,
        names=[key],
        resamples=100,
        seed=42,
        confidence=0.95,
        minimum_documents=2,
    )
    assert values[key] == 1
    assert intervals[key]["lower"] == intervals[key]["upper"] == 1
    assert intervals[key]["degenerate"]
    assert intervals[key]["defined_resamples"] == 100
    _, intervals, _ = aggregate_quality(
        samples,
        names=[key],
        resamples=100,
        seed=42,
        confidence=0.95,
        minimum_documents=None,
    )
    assert intervals[key]["lower"] is None
    assert intervals[key]["reason"] == "ci_support_not_frozen"


def test_document_macro_weights_survive_reviewed_multi_document_source_groups():
    from experiments.benchmarks.rag.retrieval_reranking.metrics import aggregate_quality

    name = "quality.overall.ndcg_at_5"
    samples = [
        SampleResult(
            f"{document}-{i}",
            {name: value},
            None,
            {
                "document_id": document,
                "independent_source_id": source,
                "evidence_type": "text",
            },
        )
        for document, source, value, count in (
            ("a", "shared", 1.0, 1),
            ("b", "shared", 0.0, 5),
            ("c", "other", 1.0, 1),
        )
        for i in range(count)
    ]
    metrics, intervals, support = aggregate_quality(
        samples,
        names=[name],
        resamples=200,
        seed=42,
        confidence=0.95,
        minimum_documents=2,
    )
    assert metrics[name] == pytest.approx(2 / 3)
    assert metrics[name + ".sample_count"] == 3
    assert support[name]["eligible_question_count"] == 7
    assert intervals[name]["independent_source_count"] == 2
    assert intervals[name]["lower"] == 0.5
    assert intervals[name]["upper"] == 1


def test_paired_latency_recalculates_percentiles_not_mean_query_differences():
    from experiments.benchmarks.rag.retrieval_reranking.metrics import (
        paired_document_interval,
    )

    def rows(times):
        return [
            SampleResult(
                str(i), {}, seconds, {"document_id": str(i), "evidence_type": "text"}
            )
            for i, seconds in enumerate(times)
        ]

    bounds, questions, documents, sources = paired_document_interval(
        rows([1.0, 2.0, 100.0]),
        rows([2.0, 10.0, 101.0]),
        "operational.full_stack_latency_ms_p50",
        resamples=200,
        seed=42,
        confidence=0.95,
        minimum_documents=2,
    )
    assert bounds["estimate"] == 8000  # median 10 minus median 2, in milliseconds
    assert bounds["estimate"] != pytest.approx((1000 + 8000 + 1000) / 3)
    assert (questions, documents, sources) == (3, 3, 3)
    assert bounds["defined_resamples"] == 200


def test_interrupted_rrf_build_preserves_only_completed_dependency():
    index = _fixture_index()
    index.vectors = np.tile(np.eye(1, 768, dtype=np.float32), (len(index.chunks), 1))
    index.phase_seconds = {"dense_index_build_seconds": 0.5}
    progress = {
        "chunks": index.chunks,
        "vectors": index.vectors,
        "chunking_contract": {"fingerprint": index.chunking_fingerprint},
        "phase_seconds": index.phase_seconds,
        "build_elapsed_to_failure": 0.7,
    }
    artifact = retrieval_benchmark._interrupted_index_artifact(
        progress, parse_candidate("rrf|none")
    )
    assert artifact["status"] == "incomplete"
    assert set(artifact["dependencies"]) == {"dense"}
    assert artifact["dependencies"]["dense"]["size_bytes"] == index.vectors.nbytes
    assert artifact["dependencies"]["dense"]["build_seconds"] == 0.5
    assert "corpus_build_seconds" not in artifact


def test_rr_validator_binds_explicit_chunking_protocol_and_reuses_report(tmp_path):
    from experiments.benchmarks.rag.retrieval_reranking.validation import (
        prepare_report,
        verify_inputs,
    )

    path = PROJECT_ROOT / "data/benchmarks/rag/smoke.json"
    protocol, chunking = default_protocol(), default_chunking_protocol()
    report_path, report = prepare_report(
        path, "smoke", protocol, chunking_protocol=chunking, report_root=tmp_path
    )
    assert report["status"] == "passed"
    assert (
        verify_inputs(
            path, "smoke", protocol, chunking_protocol=chunking, report_root=tmp_path
        )[0]
        == report_path
    )
    payload = deepcopy(chunking.meta.resolved)
    payload["data"]["minimum_structured_questions"]["table"] += 1
    changed = chunking_protocol_from_mapping(payload)
    # Smoke deliberately has no authoritative allocation requirement.
    assert (
        verify_inputs(
            path, "smoke", protocol, chunking_protocol=changed, report_root=tmp_path
        )[0]
        == report_path
    )


def test_decision_is_bound_to_protocol_and_qualification(monkeypatch):
    observed = {}

    def decision(path, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(selected_candidates=("bm25|ettin-150m",))

    monkeypatch.setattr(retrieval_run, "load_engineer_decision", decision)
    retrieval_run._selected_candidates(
        "locked",
        Path("decision.json"),
        RETRIEVAL_CANDIDATES,
        3,
        protocols={"retrieval": "protocol-sha"},
        qualification="qualification-sha",
    )
    assert observed["expected_protocols"] == {"retrieval": "protocol-sha"}
    assert observed["expected_qualification"] == "qualification-sha"


def test_attempt_tables_are_durably_serializable(monkeypatch, tmp_path):
    evaluated = _rr_evaluate(monkeypatch)
    for name in (
        "query_metrics",
        "rankings",
        "evidence_matches",
        "timings",
        "resources",
    ):
        frame = pd.DataFrame(evaluated[5][name])
        destination = tmp_path / f"{name}.parquet"
        frame.to_parquet(destination, index=False)
        assert len(pd.read_parquet(destination)) == len(frame)


def test_small_corpus_pool_size_is_exact_not_padded():
    index = _fixture_index()
    index.chunks = index.chunks[:12]
    rows = [(i, float(i)) for i in range(12)]
    retrieval_benchmark._require_pool(rows, index, default_protocol(), "q1")
    index.chunks = _fixture_index().chunks
    with pytest.raises(ValueError, match="Incomplete query pool"):
        retrieval_benchmark._require_pool(rows, index, default_protocol(), "q1")
