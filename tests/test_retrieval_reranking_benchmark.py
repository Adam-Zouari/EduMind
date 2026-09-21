from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from edumind.common.artifacts import stable_hash
from edumind.common.paths import PROJECT_ROOT
from edumind.rag.contracts import EmbeddingSpec, PRODUCTION_EMBEDDING_MODEL
from edumind.rag.tokenizers import TiktokenOffsetTokenizer
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    CandidateExecutionError,
    CandidateResult,
    DatasetManifest,
    SampleResult,
)
from experiments.benchmarks.common.arguments import load_candidates
from experiments.benchmarks.common.datasets import load_manifest
from experiments.benchmarks.rag.chunking_embedding.strategies import (
    build_chunking_strategy,
)
from experiments.benchmarks.rag.evaluation import Chunk, ExactIndex
from experiments.benchmarks.rag.retrieval import benchmark as retrieval_benchmark
from experiments.benchmarks.rag.methods import reciprocal_rank_fusion_with_scores
from experiments.benchmarks.rag.retrieval.benchmark import (
    _require_permutation,
    evaluate_candidate,
)
from experiments.benchmarks.rag.retrieval.comparisons import parent_artifact_builder
from experiments.benchmarks.rag.retrieval.metrics import (
    directions_for,
    pool_evidence_unit_recall,
)
from experiments.benchmarks.rag.retrieval.profiles import (
    owner_first,
    parse_candidate,
    validation_candidates,
)
from experiments.benchmarks.rag.retrieval.protocol import (
    default_protocol,
    protocol_from_mapping,
    protocol_from_settings,
)


RETRIEVAL_CANDIDATES = owner_first(
    load_candidates(
        PROJECT_ROOT / "experiments/benchmarks/rag/retrieval/candidates.yaml",
        "smoke",
    )
)


def _settings(
    *,
    profile: str = "development",
    warmups: int | None = None,
    repetitions: int | None = None,
    bootstrap_resamples: int | None = None,
    **overrides: object,
) -> dict[str, object]:
    payload = deepcopy(default_protocol().resolved)
    execution = payload["profiles"][profile]
    if warmups is not None:
        execution["warmups"] = warmups
    if repetitions is not None:
        execution["repetitions"] = repetitions
    if bootstrap_resamples is not None:
        execution["bootstrap_resamples"] = bootstrap_resamples
    protocol = protocol_from_mapping(payload)
    return {
        "retrieval_protocol": deepcopy(protocol.resolved),
        "retrieval_protocol_checksum": protocol.checksum,
        "retrieval_protocol_version": protocol.version,
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


def test_validation_adds_matching_no_reranker_controls() -> None:
    selected = ("bm25|ettin-150m", "rrf|none", "dense|ettin-1b")
    assert validation_candidates(selected, RETRIEVAL_CANDIDATES) == (
        "dense|none",
        "bm25|none",
        "rrf|none",
        "dense|ettin-1b",
        "bm25|ettin-150m",
    )


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
    malformed = deepcopy(protocol.resolved)
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="unknown unexpected"):
        protocol_from_mapping(malformed)
    nonfinite = deepcopy(protocol.resolved)
    nonfinite["retrieval"]["bm25"]["k1"] = float("nan")
    with pytest.raises(ValueError, match="must be finite"):
        protocol_from_mapping(nonfinite)
    with pytest.raises(ValueError, match="checksum"):
        protocol_from_settings(
            {
                "retrieval_protocol": deepcopy(protocol.resolved),
                "retrieval_protocol_checksum": "wrong",
            }
        )
    wrong_version = _settings()
    wrong_version["retrieval_protocol_version"] = "wrong"
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

    methods.BM25(["First document", "second document"])
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
        tie_keys={0: "z-chunk", 1: "a-chunk"},
    )
    assert [identifier for identifier, _ in result] == [1, 0]


def test_smoke_fixture_produces_exactly_thirty_frozen_chunks() -> None:
    manifest = load_manifest(PROJECT_ROOT / "data/benchmarks/rag/smoke.json")
    tokenizer = TiktokenOffsetTokenizer("cl100k_base")
    chunker = build_chunking_strategy("token-256-32", tokenizer=tokenizer)
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
    directions, required = directions_for(manifest, protocol)
    assert "quality.overall.ndcg_at_3" in directions
    assert "quality.overall.evidence_unit_recall_at_5" in directions
    assert "validity.ranking_agreement" in required
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
    payload = deepcopy(default_protocol().resolved)
    payload["retrieval"]["pool_size"] = 4
    payload["retrieval"]["rrf"]["source_depth"] = 4
    payload["quality"]["cutoffs"] = [2]
    protocol = protocol_from_mapping(payload)
    directions, _ = directions_for(_fixture_manifest(), protocol)
    assert "quality.overall.ndcg_at_2" in directions
    assert "quality.overall.ndcg_at_3" not in directions
    assert protocol.pool_recall_metric == (
        "diagnostic.pool_evidence_unit_recall_at_4"
    )


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
    snapshot_bytes = 1234

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
            (position, float(position))
            for position in reversed(range(len(documents)))
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
        corpus_build_seconds=0.1,
        preflight={"status": "passed", "truncated_inputs": 0},
    )


def test_owner_pool_is_reused_by_every_reranker_child(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _fixture_manifest()
    index = _fixture_index()
    monkeypatch.setattr(retrieval_benchmark, "build_index", lambda *_args, **_kwargs: index)
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
        repetitions=2,
        bootstrap_resamples=0,
        warmups=0,
        settings=_settings(
            profile="smoke",
            warmups=0,
            repetitions=2,
            bootstrap_resamples=0,
            chunker_embedding=f"token-256-32|{PRODUCTION_EMBEDDING_MODEL}"
        ),
    )
    owner = evaluate_candidate(
        "bm25|none",
        manifest,
        model_lock,
        plan,
        device="cpu",
        dtype="float32",
        frozen_pool=None,
        child_run_id="owner-run",
    )
    rows = owner[5]["candidate_pool"]
    assert len(rows) == 20
    assert owner[3]["bm25_k1"] == 1.5
    assert owner[3]["bm25_b"] == 0.75
    assert owner[3]["bm25_epsilon"] == 0.25
    assert "dense_similarity" not in owner[3]
    assert "rrf_union" not in owner[3]
    assert "retrieval_contract" not in owner[3]
    assert "reranker_contract" not in owner[3]
    assert "reranker_batch_size" not in owner[3]
    frozen_pool = {
        "owner_candidate": "bm25|none",
        "owner_run_id": "owner-run",
        "pool_checksum": stable_hash(rows),
        "rows": rows,
    }
    monkeypatch.setattr(
        retrieval_benchmark,
        "_reranker",
        lambda *_args, **_kwargs: _FixtureReranker(),
    )
    child = evaluate_candidate(
        "bm25|ettin-150m",
        manifest,
        model_lock,
        plan,
        device="cpu",
        dtype="float32",
        frozen_pool=frozen_pool,
        child_run_id="child-run",
    )
    assert "candidate_pool" not in child[5]
    assert child[3]["pool_owner_run_id"] == "owner-run"
    assert child[3]["pool_checksum"] == stable_hash(rows)
    assert child[3]["reranker_model"] == "fixture-reranker"
    assert child[3]["reranker_batch_size"] == 1
    assert child[3]["reranker_input_template"] == "tokenizer(query, passage)"
    assert "reranker_contract" not in child[3]
    assert child[2]["validity.exact_pool_permutation"] == 1.0
    assert child[2]["validity.ranking_agreement"] == 1.0
    assert len(child[5]["timings"]) == 2


def test_measured_failure_keeps_every_timing_row(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch.setattr(retrieval_benchmark, "build_index", lambda *_args, **_kwargs: index)
    plan = BenchmarkPlan(
        "rag",
        "retrieval-reranking",
        "smoke",
        manifest.name,
        ("bm25|none",),
        repetitions=3,
        bootstrap_resamples=0,
        warmups=0,
        settings=_settings(
            profile="smoke",
            warmups=0,
            repetitions=3,
            bootstrap_resamples=0,
            chunker_embedding=f"token-256-32|{PRODUCTION_EMBEDDING_MODEL}"
        ),
    )
    with pytest.raises(CandidateExecutionError) as captured:
        evaluate_candidate(
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
            frozen_pool=None,
            child_run_id="owner-run",
        )
    assert calls == 3
    assert len(captured.value.artifacts["timings"]) == 3
    assert captured.value.metrics["validity.failed_query_count"] == 1.0
    assert captured.value.metrics["validity.ranking_agreement"] < 1.0


def test_parent_comparisons_are_separated_and_csv_matches_parquet(
    tmp_path: Path,
) -> None:
    directions = {
        "quality.overall.ndcg_at_3": "max",
        "diagnostic.pool_evidence_unit_recall_at_20": "max",
        "operational.full_stack_latency_ms_p50": "min",
        "operational.first_stage_latency_ms_p50": "min",
        "workload.retrieved_tokens_at_3_mean": "descriptive",
        "storage.required_index_bytes": "descriptive",
    }
    pool_checksums = {
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
                    "workload.retrieved_tokens_at_3_mean": 100.0 + position,
                    "storage.required_index_bytes": 1000.0 + position,
                },
                intervals={},
                samples=samples,
                operational={
                    "full_stack_latency_ms_p50": 10.0 + position,
                    "first_stage_latency_ms_p50": 5.0 + position,
                },
                mlflow_run_id=f"run-{position}",
                parameters={"pool_checksum": pool_checksums[parsed.retriever]},
            )
        )
    pools = {
        f"{retriever}|none": {
            "owner_run_id": f"owner-{retriever}",
            "pool_checksum": checksum,
            "index_checksum": f"index-{retriever}",
            "rows": [{"retriever": retriever}],
        }
        for retriever, checksum in pool_checksums.items()
    }
    plan = BenchmarkPlan(
        "rag",
        "retrieval-reranking",
        "validation",
        "fixture",
        RETRIEVAL_CANDIDATES,
        bootstrap_resamples=100,
        settings=_settings(
            profile="validation",
            repetitions=1,
            bootstrap_resamples=100,
            requested_finalists=[
                "dense|ettin-150m",
                "bm25|ettin-400m",
            ]
        ),
    )
    paths = parent_artifact_builder(
        pools, directions, compare_finalists=True
    )(tmp_path, results, plan)
    assert {path.name for path in paths} == {
        "reranker_comparisons.parquet",
        "reranker_comparisons.csv",
        "retriever_comparisons.parquet",
        "retriever_comparisons.csv",
        "finalist_comparisons.parquet",
        "finalist_comparisons.csv",
        "pool_index.json",
        "retrieval_protocol.json",
    }
    reranker = pd.read_parquet(tmp_path / "reranker_comparisons.parquet")
    retriever = pd.read_parquet(tmp_path / "retriever_comparisons.parquet")
    assert reranker[
        ["baseline_candidate", "candidate"]
    ].drop_duplicates().shape[0] == 12
    assert retriever[
        ["baseline_candidate", "candidate"]
    ].drop_duplicates().shape[0] == 3
    pool_recall = retriever[
        retriever["metric_name"]
        == "diagnostic.pool_evidence_unit_recall_at_20"
    ]
    assert pool_recall["evidence_slice"].eq("overall").all()
    assert pool_recall["cutoff"].eq(20).all()
    assert pool_recall["ci_status"].eq("available").all()
    finalists = pd.read_parquet(tmp_path / "finalist_comparisons.parquet")
    assert finalists[
        ["baseline_candidate", "candidate"]
    ].drop_duplicates().shape[0] == 1
    workload = reranker[
        reranker["metric_name"] == "workload.retrieved_tokens_at_3_mean"
    ]
    assert workload["direction"].eq("descriptive").all()
    assert workload["favors_candidate"].isna().all()
    pool_index = json.loads((tmp_path / "pool_index.json").read_text())
    assert len(pool_index["pools"]) == 3
    assert pool_index["comparison_artifacts"]["reranker_comparisons"][
        "logical_table_sha256"
    ]
    protocol_artifact = json.loads(
        (tmp_path / "retrieval_protocol.json").read_text()
    )
    assert protocol_artifact["checksum"] == protocol_from_settings(
        plan.settings
    ).checksum
    assert protocol_artifact["resolved"]["retrieval"]["pool_size"] == 20
