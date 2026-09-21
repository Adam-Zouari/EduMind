from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    DatasetManifest,
    SampleResult,
)
from experiments.benchmarks.common.datasets import (
    DatasetValidationError,
    validate_evidence,
)
from experiments.benchmarks.rag.chunking_embedding import benchmark
from experiments.benchmarks.rag.chunking_embedding.metrics import (
    aggregate_quality,
    alpha_ndcg_at_k,
    latency_intervals,
    score_question,
)
from experiments.benchmarks.rag.chunking_embedding.profiles import embedding_spec
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    load_protocol as default_chunking_protocol,
    protocol_from_mapping as chunking_protocol_from_mapping,
)
from experiments.benchmarks.rag.chunking_embedding.strategies import (
    RecursiveCharacterChunkingStrategy,
    SemanticChunkingStrategy,
    StructureAwareChunkingStrategy,
    build_chunking_strategy,
)
from experiments.benchmarks.rag.evaluation import (
    ExactIndex,
    InputCompatibilityError,
    _chunking_contract,
    dense_rank_with_scores,
    retrieval_metrics,
)
from experiments.benchmarks.preparation.datasets import _answers_and_evidence
from edumind.rag.contracts import EmbeddingSpec
from edumind.rag.embedder import Embedder
from edumind.rag.errors import RAGConfigurationError
from edumind.rag.text_chunker import TokenChunkingStrategy
from edumind.rag.tokenizers import TiktokenOffsetTokenizer
from edumind.rag.types import RetrievalHit


CHUNKING_PROTOCOL = default_chunking_protocol()


def _protocol_settings(*, warmups=1, repetitions=1, bootstrap_resamples=0):
    payload = deepcopy(CHUNKING_PROTOCOL.meta.resolved)
    payload["profiles"]["smoke"].update(
        {
            "warmups": warmups,
            "repetitions": repetitions,
            "bootstrap_resamples": bootstrap_resamples,
        }
    )
    protocol = chunking_protocol_from_mapping(payload)
    return {"chunking_embedding_protocol": protocol.meta.worker_payload()}


@dataclass(frozen=True)
class Chunk:
    identifier: str
    document_id: str
    text: str
    start: int
    end: int
    tokens: int = 0
    model_tokens: int | None = None


class WordTokenizer:
    name = "fixture:words"

    @staticmethod
    def spans(text: str) -> list[tuple[int, int]]:
        result = []
        cursor = 0
        for word in text.split():
            start = text.index(word, cursor)
            result.append((start, start + len(word)))
            cursor = start + len(word)
        return result

    @classmethod
    def count(cls, text: str) -> int:
        return len(cls.spans(text))


class CharacterTokenizer:
    name = "fixture:characters"

    @staticmethod
    def spans(text: str) -> list[tuple[int, int]]:
        return [
            (index, index + 1)
            for index, character in enumerate(text)
            if not character.isspace()
        ]

    @classmethod
    def count(cls, text: str) -> int:
        return len(cls.spans(text))


def test_qasper_preparation_emits_typed_units_and_rejects_ambiguous_offsets() -> None:
    question = {
        "answers": [
            {
                "answer": {
                    "free_form_answer": "answer",
                    "highlighted_evidence": ["supporting text"],
                }
            }
        ]
    }

    _, evidence, _, answerable = _answers_and_evidence(
        question, "before supporting text after", "paper"
    )

    assert answerable is True
    assert evidence == [
        {
            "id": "paper:7:22",
            "evidence_type": "text",
            "document_id": "paper",
            "start": 7,
            "end": 22,
        }
    ]
    with pytest.raises(ValueError, match="ambiguous source offset"):
        _answers_and_evidence(
            question,
            "supporting text then supporting text",
            "paper",
        )


def test_alpha_ndcg_penalizes_duplicate_evidence_before_novel_evidence() -> None:
    corpus = [{"a"}, {"a"}, {"b"}]
    duplicate_first = alpha_ndcg_at_k(corpus, corpus, 3, alpha=0.5)
    novel_first = alpha_ndcg_at_k([{"a"}, {"b"}, {"a"}], corpus, 3, alpha=0.5)
    assert 0.0 <= duplicate_first < novel_first <= 1.0


def test_quality_uses_three_and_five_cutoffs_and_alpha_requires_multiple_units() -> None:
    question = {
        "id": "q1",
        "document_id": "doc",
        "evidence_type": "text",
        "evidence": [
            {
                "id": "unit-a",
                "evidence_type": "text",
                "document_id": "doc",
                "start": 0,
                "end": 5,
            },
            {
                "id": "unit-b",
                "evidence_type": "text",
                "document_id": "doc",
                "start": 6,
                "end": 10,
            },
        ],
    }
    ranking = [
        Chunk("a", "doc", "alpha", 0, 5),
        Chunk("d1", "other", "x", 0, 1),
        Chunk("d2", "other", "y", 1, 2),
        Chunk("b", "doc", "beta", 6, 10),
        Chunk("d3", "other", "z", 2, 3),
    ]

    score = score_question(
        question, ranking, ranking, WordTokenizer(), cutoffs=(3, 5), alpha=0.5
    )

    assert score.metrics["evidence_unit_recall_at_3"] == 0.5
    assert score.metrics["evidence_unit_recall_at_5"] == 1.0
    assert score.metrics["evidence_token_precision_at_3"] == pytest.approx(1 / 3)
    assert score.metrics["evidence_token_precision_at_5"] == pytest.approx(2 / 5)
    assert set(score.metrics) == {
        "ndcg_at_3",
        "ndcg_at_5",
        "evidence_unit_recall_at_3",
        "evidence_unit_recall_at_5",
        "evidence_token_precision_at_3",
        "evidence_token_precision_at_5",
        "alpha_ndcg_at_3",
        "alpha_ndcg_at_5",
    }
    assert score.matches[0]["recovered_at_3"] is True
    assert score.matches[1]["recovered_at_3"] is False
    assert score.matches[1]["recovered_at_5"] is True


def test_single_unit_questions_omit_alpha_ndcg_and_its_metric_contract() -> None:
    manifest = _manifest()
    question = next(row for row in manifest.samples if row.get("kind") == "question")
    chunk = Chunk("a", "doc", "alpha", 0, 5)

    score = score_question(
        question, [chunk], [chunk], WordTokenizer(), cutoffs=(3, 5), alpha=0.5
    )
    directions, _ = benchmark.directions_for(manifest, CHUNKING_PROTOCOL)

    assert not any(name.startswith("alpha_ndcg") for name in score.metrics)
    assert not any("alpha_ndcg" in name for name in directions)

    rows = list(manifest.samples)
    rows[-1] = {
        **rows[-1],
        "evidence": [
            *rows[-1]["evidence"],
            {
                "id": "q-e2",
                "evidence_type": "text",
                "document_id": "doc",
                "start": 0,
                "end": 5,
            },
        ],
    }
    multi_evidence = replace(manifest, samples=tuple(rows))
    multi_directions, _ = benchmark.directions_for(
        multi_evidence, CHUNKING_PROTOCOL
    )
    assert "quality.overall.alpha_ndcg_at_3" in multi_directions
    assert "quality.text.alpha_ndcg_at_5" in multi_directions


def test_retrieval_metrics_use_complete_units_and_evaluation_tokens() -> None:
    text = "alpha beta gamma"
    question = {
        "id": "q1",
        "document_id": "doc",
        "evidence_type": "text",
        "evidence": [
            {
                "id": "unit-a",
                "evidence_type": "text",
                "intervals": [
                    {"document_id": "doc", "start": 0, "end": 5},
                    {"document_id": "doc", "start": 6, "end": 10},
                ],
            }
        ],
    }
    partial = Chunk("partial", "doc", "alpha", 0, 5)
    complete = Chunk("complete", "doc", "alpha beta gamma", 0, len(text))
    distractor = Chunk("other", "other", "alpha beta", 0, 10)

    partial_score = score_question(
        question,
        [partial, distractor],
        [partial, complete, distractor],
        WordTokenizer(),
        cutoffs=(3, 5),
        alpha=0.5,
    )
    assert partial_score.metrics["evidence_unit_recall_at_5"] == 0.0

    complete_score = score_question(
        question,
        [complete],
        [partial, complete, distractor],
        WordTokenizer(),
        cutoffs=(3, 5),
        alpha=0.5,
    )
    assert complete_score.metrics["evidence_unit_recall_at_5"] == 1.0
    assert complete_score.metrics["evidence_token_precision_at_5"] == pytest.approx(2 / 3)
    assert complete_score.matches[0]["first_rank"] == 1


def test_downstream_retrieval_metrics_accept_multi_interval_evidence_units() -> None:
    question = {
        "id": "q1",
        "document_id": "doc",
        "evidence_type": "text",
        "evidence": [
            {
                "id": "unit-a",
                "evidence_type": "text",
                "intervals": [
                    {"document_id": "doc", "start": 0, "end": 5},
                    {"document_id": "doc", "start": 6, "end": 10},
                ],
            }
        ],
    }
    chunk = Chunk("complete", "doc", "alpha beta", 0, 10, 2)
    metrics, _ = retrieval_metrics(
        question, [chunk], [chunk], WordTokenizer(), cutoffs=(1, 3, 5, 10)
    )
    assert metrics["context_recall_at_5"] == 1.0


def test_downstream_citation_support_accepts_multi_interval_evidence_units() -> None:
    from experiments.benchmarks.rag.generation.evaluate import _supported_contexts

    question = {
        "id": "q1",
        "document_id": "doc",
        "evidence_type": "text",
        "evidence": [
            {
                "id": "unit-a",
                "evidence_type": "text",
                "intervals": [
                    {"document_id": "doc", "start": 0, "end": 5},
                    {"document_id": "doc", "start": 20, "end": 25},
                ],
            }
        ],
    }
    hit = RetrievalHit(
        "chunk", "evidence", {"document_id": "doc", "start": 18, "end": 24}, 1.0
    )
    assert _supported_contexts(question, [hit]) == {1}


def test_exact_cosine_ranking_uses_stable_corpus_order_for_ties() -> None:
    class QueryEmbedder:
        @staticmethod
        def embed_query(_query):
            return np.asarray([2.0, 0.0])

    chunks = [
        Chunk("first", "doc", "a", 0, 1),
        Chunk("second", "doc", "b", 1, 2),
        Chunk("third", "doc", "c", 2, 3),
    ]
    index = ExactIndex(
        chunks,
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        QueryEmbedder(),
        WordTokenizer(),
        None,
        SimpleNamespace(dimension=2),
        "chunker",
        {},
        1.0,
        {},
    )
    ranking = dense_rank_with_scores(index, "query", 3)
    assert [position for position, _ in ranking] == [0, 1, 2]
    assert ranking[0][1] == ranking[1][1] == 1.0


def test_quality_aggregation_uses_documents_as_statistical_units() -> None:
    samples = [
        SampleResult("q1", {"quality.overall.alpha_ndcg_at_5": 1.0}, 0.1, {"document_id": "a", "evidence_type": "text"}),
        SampleResult("q2", {"quality.overall.alpha_ndcg_at_5": 1.0}, 0.1, {"document_id": "a", "evidence_type": "text"}),
        SampleResult("q3", {"quality.overall.alpha_ndcg_at_5": 0.0}, 0.1, {"document_id": "b", "evidence_type": "text"}),
    ]
    aggregate, intervals = aggregate_quality(
        samples, resamples=100, seed=42, confidence=0.95
    )
    assert aggregate["quality.overall.alpha_ndcg_at_5"] == 0.5
    assert aggregate["quality.overall.alpha_ndcg_at_5.sample_count"] == 2
    assert intervals["quality.overall.alpha_ndcg_at_5"]["estimate"] == 0.5


def test_latency_intervals_resample_documents_not_query_rows() -> None:
    samples = [
        SampleResult(
            f"q{index}",
            {},
            index / 1000,
            {"document_id": f"doc-{index}", "evidence_type": "text"},
        )
        for index in range(1, 21)
    ]
    intervals = latency_intervals(
        samples,
        resamples=100,
        seed=42,
        minimum_documents=20,
        confidence=0.95,
    )
    assert set(intervals) == {
        "operational.query_latency_ms_p50",
        "operational.query_latency_ms_p95",
    }
    assert intervals["operational.query_latency_ms_p50"]["estimate"] == 10.5


def test_sentence_transformer_preflight_uses_resolved_role_prompt(monkeypatch) -> None:
    calls = []

    class Model:
        prompts = {"query": "query: ", "document": "passage: "}
        default_prompt_name = None

        def preprocess(self, inputs, **options):
            calls.append((inputs, options))
            return {"input_ids": np.ones((1, 7), dtype=int)}

    spec = EmbeddingSpec(
        "fixture",
        "revision",
        "fixture",
        None,
        "",
        "",
        True,
        2,
        "cosine",
        32,
        "mean",
        interface="query-document",
    )
    runtime = Embedder(spec)
    monkeypatch.setattr(runtime, "_model", lambda _device: Model())

    assert runtime.input_token_counts(["question"], role="query") == [7]
    assert calls[0][1]["prompt"] == "query: "
    assert calls[0][1]["task"] == "query"
    assert calls[0][1]["processing_kwargs"]["text"]["truncation"] is False
    assert runtime.input_configuration("document")["prompt"] == "passage: "


def test_embedding_runtime_rejects_contract_above_prepared_model_limit(
    monkeypatch,
) -> None:
    import sentence_transformers

    class Model:
        max_seq_length = 128
        device = "cpu"

    monkeypatch.setattr(
        sentence_transformers, "SentenceTransformer", lambda *_args, **_kwargs: Model()
    )
    spec = EmbeddingSpec(
        "fixture",
        "revision",
        "fixture",
        None,
        "",
        "",
        True,
        2,
        "cosine",
        256,
        "mean",
    )
    with pytest.raises(RAGConfigurationError, match="prepared model exposes 128"):
        Embedder(spec).prepare()


def test_octen_profiles_use_saved_query_and_document_prompts() -> None:
    assert embedding_spec(
        "Octen/Octen-Embedding-0.6B", revision="rev", local_path="path"
    ).interface == "query-document"


def test_all_chunking_strategies_return_exact_nonempty_source_spans() -> None:
    text = (
        "# Section\n\nFirst sentence. Second sentence.\n\n"
        "| Header | Value |\n|---|---|\n| alpha | 1 |\n\n$$x^2$$\n"
        + "token " * 600
    )
    names = (
        "recursive-character",
        "token-256-32",
        "token-384-64",
        "token-512-64",
        "sentence-8-2",
        "semantic",
        "section-aware-512-64",
        "structure-aware-512-64",
    )

    def embed_sentences(sentences):
        return np.eye(len(sentences), dtype=float)

    for name in names:
        strategy = build_chunking_strategy(
            name,
            tokenizer=CharacterTokenizer(),
            embed_sentences=embed_sentences,
            semantic_embedding_fingerprint="fixture-embedding",
            protocol=CHUNKING_PROTOCOL,
        )
        chunks = strategy.split(text)
        assert chunks
        assert all(
            0 <= start < end <= len(text)
            and tokens == CharacterTokenizer.count(text[start:end])
            and tokens > 0
            for start, end, tokens in chunks
        )


def test_recursive_chunker_does_not_emit_whitespace_only_chunks() -> None:
    strategy = RecursiveCharacterChunkingStrategy(
        CharacterTokenizer(),
        size=10,
        overlap=2,
        name="recursive-character",
        separators=("\n\n", "\n", ". ", " "),
        minimum_boundary_ratio=0.5,
    )
    chunks = strategy.split("a" + " " * 40 + "b")
    assert chunks
    assert all(tokens > 0 for _, _, tokens in chunks)


def test_recursive_chunker_fingerprint_includes_tokenizer() -> None:
    arguments = {
        "size": 1000,
        "overlap": 200,
        "name": "recursive-character",
        "separators": ("\n\n", "\n", ". ", " "),
        "minimum_boundary_ratio": 0.5,
    }
    first = RecursiveCharacterChunkingStrategy(WordTokenizer(), **arguments)
    second_tokenizer = WordTokenizer()
    second_tokenizer.name = "fixture:different"
    second = RecursiveCharacterChunkingStrategy(second_tokenizer, **arguments)

    assert first.fingerprint != second.fingerprint


def test_recursive_chunker_uses_frozen_separator_priority() -> None:
    strategy = RecursiveCharacterChunkingStrategy(
        CharacterTokenizer(),
        size=16,
        overlap=2,
        name="recursive-character",
        separators=("\n\n", "\n", ". ", " "),
        minimum_boundary_ratio=0.5,
    )

    chunks = strategy.split("aaaa aaaa\n\nbbbb bbbb")
    contract = _chunking_contract(strategy)

    assert chunks[0][1] == 11
    assert contract["separators"] == ["\n\n", "\n", ". ", " "]
    assert contract["minimum_boundary_ratio"] == 0.5


def test_semantic_chunker_does_not_invent_boundaries_for_equal_similarity() -> None:
    strategy = SemanticChunkingStrategy(
        WordTokenizer(),
        lambda sentences: np.ones((len(sentences), 2), dtype=float),
        "fixture-embedding",
        maximum_tokens=384,
        percentile=0.2,
    )
    assert len(strategy.split("First sentence. Second sentence.")) == 1


def test_semantic_chunker_bounds_one_oversized_sentence() -> None:
    strategy = SemanticChunkingStrategy(
        WordTokenizer(),
        lambda sentences: np.ones((len(sentences), 2), dtype=float),
        "fixture-embedding",
        maximum_tokens=3,
        percentile=0.2,
    )

    chunks = strategy.split("one two three four five six")

    assert [tokens for _, _, tokens in chunks] == [3, 3]


def test_structure_chunker_enforces_ceiling_for_oversized_table_rows() -> None:
    strategy = StructureAwareChunkingStrategy(
        CharacterTokenizer(),
        size=10,
        overlap=2,
        name="structure-aware-fixture",
        parser_identity="markdown-table-formula-v1",
    )
    chunks = strategy.split(
        "| Header | Value |\n|---|---|\n| " + "x" * 40 + " | 1 |\n| b | 2 |"
    )
    assert chunks
    assert max(tokens for _, _, tokens in chunks) <= 10


def test_rag_evidence_requires_stable_units_and_valid_mixed_types() -> None:
    base = DatasetManifest(
        "fixture",
        "1",
        "rag",
        "development",
        "fixture",
        "CC0",
        "1",
        "checksum",
        "v1",
        42,
        (
            {"id": "doc", "kind": "document", "text": "alpha beta"},
            {
                "id": "q",
                "kind": "question",
                "document_id": "doc",
                "question": "question",
                "answerable": True,
                "evidence_type": "text",
                "evidence": [{"document_id": "doc", "start": 0, "end": 5}],
            },
        ),
    )
    with pytest.raises(DatasetValidationError, match="stable ID"):
        validate_evidence(base)

    mixed = DatasetManifest(
        **{
            **base.__dict__,
            "samples": (
                base.samples[0],
                {
                    **base.samples[1],
                    "evidence_type": "mixed",
                    "evidence": [
                        {"id": "a", "evidence_type": "text", "document_id": "doc", "start": 0, "end": 5},
                        {"id": "b", "evidence_type": "formula", "document_id": "doc", "start": 6, "end": 10},
                    ],
                },
            ),
        }
    )
    validate_evidence(mixed)

    non_integer = DatasetManifest(
        **{
            **base.__dict__,
            "samples": (
                base.samples[0],
                {
                    **base.samples[1],
                    "evidence": [
                        {
                            "id": "a",
                            "evidence_type": "text",
                            "document_id": "doc",
                            "start": 0.5,
                            "end": 5,
                        }
                    ],
                },
            ),
        }
    )
    with pytest.raises(DatasetValidationError, match="non-integer offsets"):
        validate_evidence(non_integer)

    duplicate_unit = DatasetManifest(
        **{
            **mixed.__dict__,
            "samples": (
                mixed.samples[0],
                {
                    **mixed.samples[1],
                    "evidence": [
                        {
                            "id": identifier,
                            "evidence_type": "text",
                            "document_id": "doc",
                            "start": 0,
                            "end": 5,
                        }
                        for identifier in ("a", "b")
                    ],
                    "evidence_type": "text",
                },
            ),
        }
    )
    with pytest.raises(DatasetValidationError, match="duplicate evidence units"):
        validate_evidence(duplicate_unit)


def test_build_index_rejects_model_inputs_before_embedding(monkeypatch) -> None:
    import experiments.benchmarks.rag.evaluation as evaluation

    class Embedder:
        batch_size = 1

        def prepare(self):
            pass

        def input_token_counts(self, texts, *, role):
            del role
            return [300 for _ in texts]

        @staticmethod
        def input_configuration(role):
            return {"role": role}

        def embed_texts(self, texts):
            raise AssertionError("incompatible inputs must not be embedded")

    class Strategy:
        name = "fixture"
        fingerprint = "chunker-fingerprint"
        tokenizer = WordTokenizer()

        @staticmethod
        def split(text):
            return [(0, len(text), 1)]

    monkeypatch.setattr(evaluation, "TiktokenOffsetTokenizer", lambda *_: WordTokenizer())
    monkeypatch.setattr(
        evaluation,
        "embedding_spec",
        lambda *_args, **_kwargs: SimpleNamespace(
            tokenizer="fixture",
            maximum_length=256,
            dimension=2,
            fingerprint="embedding-fingerprint",
        ),
    )
    monkeypatch.setattr(
        evaluation, "Embedder", lambda _spec, **_kwargs: Embedder()
    )
    monkeypatch.setattr(
        evaluation, "build_chunking_strategy", lambda *_args, **_kwargs: Strategy()
    )
    manifest = _manifest()
    with pytest.raises(InputCompatibilityError) as caught:
        evaluation.build_index(
            manifest,
            "fixture",
            "embedding",
            {"embedding": {"revision": "rev", "model_path": "path"}},
            with_bm25=False,
            device="cpu",
            dtype="float32",
            chunking_protocol=CHUNKING_PROTOCOL,
        )
    assert caught.value.report["offending_document_count"] == 2


def test_cl100k_boundaries_send_canonical_text_not_token_ids_to_embedder(
    monkeypatch,
) -> None:
    import experiments.benchmarks.rag.evaluation as evaluation

    embedded_inputs: list[tuple[str, ...]] = []

    class NativeEmbedder:
        batch_size = 1

        @staticmethod
        def prepare() -> None:
            pass

        @staticmethod
        def input_token_counts(texts, *, role):
            del role
            assert all(isinstance(text, str) for text in texts)
            return [7 for _ in texts]

        @staticmethod
        def input_configuration(role):
            return {"role": role, "tokenizer": "native"}

        @staticmethod
        def embed_texts(texts):
            assert all(isinstance(text, str) for text in texts)
            embedded_inputs.append(tuple(texts))
            return np.tile(np.asarray([[1.0, 0.0]]), (len(texts), 1))

    monkeypatch.setattr(
        evaluation,
        "embedding_spec",
        lambda *_args, **_kwargs: SimpleNamespace(
            tokenizer="native",
            maximum_length=8192,
            dimension=2,
            fingerprint="embedding-fingerprint",
        ),
    )
    monkeypatch.setattr(evaluation, "Embedder", lambda _spec, **_kwargs: NativeEmbedder())
    monkeypatch.setattr(
        evaluation,
        "build_chunking_strategy",
        lambda _name, *, tokenizer, **_kwargs: TokenChunkingStrategy(
            tokenizer, 5, 0, "token-5-0"
        ),
    )
    text = "I am a superman fan"
    manifest = DatasetManifest(
        "fixture",
        "1",
        "rag",
        "smoke",
        "fixture",
        "CC0",
        "1",
        "checksum",
        "v1",
        42,
        (
            {"id": "doc", "kind": "document", "text": text},
            {
                "id": "q",
                "kind": "question",
                "document_id": "doc",
                "question": "Who is the fan?",
                "answerable": True,
                "evidence_type": "text",
                "evidence": [],
            },
        ),
    )

    index = evaluation.build_index(
        manifest,
        "token-5-0",
        "embedding",
        {"embedding": {"revision": "rev", "model_path": "path"}},
        with_bm25=False,
        device="cpu",
        dtype="float32",
        chunking_protocol=CHUNKING_PROTOCOL,
    )

    assert isinstance(index.tokenizer, TiktokenOffsetTokenizer)
    assert "".join(chunk.text for chunk in index.chunks) == text
    assert embedded_inputs == [tuple(chunk.text for chunk in index.chunks)]
    assert all(chunk.model_tokens == 7 for chunk in index.chunks)
    assert all(chunk.tokens <= 5 for chunk in index.chunks)


def test_worker_marks_oversized_input_as_failed_with_reason_code(monkeypatch) -> None:
    embedding_name = "Alibaba-NLP/gte-modernbert-base"
    candidate = f"fixture|{embedding_name}"
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        "smoke",
        "fixture",
        (candidate,),
        seed=CHUNKING_PROTOCOL.meta.seed,
        repetitions=1,
        warmups=0,
        bootstrap_resamples=0,
        settings=_protocol_settings(warmups=0),
    )

    def reject(*_args, **_kwargs):
        raise InputCompatibilityError(
            "too long",
            {
                "status": "failed",
                "reason_code": "input_length_exceeded",
                "maximum_length": 8192,
                "chunking_contract": {},
            },
        )

    monkeypatch.setattr(benchmark, "evaluate_candidate", reject)
    result = benchmark.execute_payload(
        {
            "candidate": candidate,
            "manifest": _manifest().__dict__,
            "model_lock": {
                embedding_name: {"revision": "rev", "model_path": "path"}
            },
            "plan": plan.__dict__,
            "device": "cpu",
            "dtype": "float32",
        }
    )

    assert result["status"] == "failed"
    report = result["artifacts"]["validation_report"]
    assert report["status"] == "failed"
    assert report["reason_code"] == "input_length_exceeded"
    assert report["preflight"]["maximum_length"] == 8192


def test_candidate_emits_new_metrics_and_auditable_top_twenty(monkeypatch) -> None:
    chunks = [
        Chunk(f"chunk-{index}", "doc" if index == 0 else "other", "alpha", 0, 5, 1, 3)
        for index in range(25)
    ]
    index = ExactIndex(
        chunks,
        np.ones((25, 2), dtype=np.float32),
        object(),
        WordTokenizer(),
        None,
        SimpleNamespace(dimension=2),
        "chunker-fingerprint",
        {"name": "fixture", "fingerprint": "chunker-fingerprint"},
        2.0,
        {"status": "passed", "truncated_inputs": 0},
    )
    monkeypatch.setattr(benchmark, "build_index", lambda *_args, **_kwargs: index)
    monkeypatch.setattr(
        benchmark,
        "dense_rank_with_scores",
        lambda *_args, **_kwargs: [(position, 1.0 - position / 100) for position in range(20)],
    )
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        "smoke",
        "fixture",
        ("fixture|embedding",),
        seed=CHUNKING_PROTOCOL.meta.seed,
        repetitions=2,
        warmups=0,
        bootstrap_resamples=0,
        settings=_protocol_settings(warmups=0, repetitions=2),
    )
    result = benchmark.evaluate_candidate(
        "fixture|embedding",
        _manifest(),
        {"embedding": {"revision": "rev", "model_path": "path"}},
        plan,
        device="cpu",
        dtype="float32",
    )
    samples, operational, aggregate, _parameters, intervals, artifacts = result
    assert len(samples) == 1
    assert aggregate["quality.overall.ndcg_at_3"] == 1.0
    assert aggregate["quality.overall.evidence_unit_recall_at_5"] == 1.0
    assert "quality.overall.alpha_ndcg_at_3" not in aggregate
    assert (
        aggregate[
            "validity.quality.overall.alpha_ndcg_eligible_question_count"
        ]
        == 0.0
    )
    assert aggregate["workload.embedding_matrix_bytes"] == 25 * 2 * 4
    assert operational["corpus_build_source_tokens_per_second"] == 1.0
    assert _parameters["manifest_checksum"] == "checksum"
    assert _parameters["manifest_fingerprint"]
    assert _parameters["quality_cutoffs"] == (3, 5)
    assert set(CHUNKING_PROTOCOL.primary_quality_metrics) == {
        "ndcg_at_3",
        "ndcg_at_5",
        "evidence_unit_recall_at_3",
        "evidence_unit_recall_at_5",
        "evidence_token_precision_at_3",
        "evidence_token_precision_at_5",
    }
    assert intervals == {}
    assert len(artifacts["retrievals"]) == 20
    assert len(artifacts["timings"]) == 2
    assert set(artifacts) == {
        "chunk_manifest",
        "query_metrics",
        "retrievals",
        "evidence_matches",
        "timings",
        "validation_report",
    }


def test_measured_query_failure_runs_every_repetition(monkeypatch) -> None:
    index = ExactIndex(
        [Chunk("chunk", "doc", "alpha", 0, 5, 1, 3)],
        np.ones((1, 2), dtype=np.float32),
        object(),
        WordTokenizer(),
        None,
        SimpleNamespace(dimension=2),
        "chunker-fingerprint",
        {"name": "fixture", "fingerprint": "chunker-fingerprint"},
        1.0,
        {"status": "passed", "truncated_inputs": 0},
    )
    calls = 0

    def rank(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected")
        return [(0, 1.0)]

    monkeypatch.setattr(benchmark, "build_index", lambda *_args, **_kwargs: index)
    monkeypatch.setattr(benchmark, "dense_rank_with_scores", rank)
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        "smoke",
        "fixture",
        ("fixture|embedding",),
        seed=CHUNKING_PROTOCOL.meta.seed,
        repetitions=3,
        warmups=0,
        bootstrap_resamples=0,
        settings=_protocol_settings(warmups=0, repetitions=3),
    )
    with pytest.raises(benchmark.CandidateExecutionError) as caught:
        benchmark.evaluate_candidate(
            "fixture|embedding",
            _manifest(),
            {"embedding": {"revision": "rev", "model_path": "path"}},
            plan,
            device="cpu",
            dtype="float32",
        )
    assert calls == 3
    assert len(caught.value.artifacts["timings"]) == 3


def test_worker_serializes_unexpected_python_failures_with_validation_report(
    monkeypatch,
) -> None:
    candidate = "token-256-32|Alibaba-NLP/gte-modernbert-base"
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        "smoke",
        "fixture",
        (candidate,),
        seed=CHUNKING_PROTOCOL.meta.seed,
        repetitions=1,
        warmups=0,
        bootstrap_resamples=0,
        settings=_protocol_settings(warmups=0),
    )
    def fail(*_args, **_kwargs):
        raise RuntimeError("injected")

    monkeypatch.setattr(benchmark, "evaluate_candidate", fail)
    result = benchmark.execute_payload(
        {
            "candidate": candidate,
            "manifest": _manifest().__dict__,
            "model_lock": {
                "Alibaba-NLP/gte-modernbert-base": {
                    "revision": "rev",
                    "model_path": "path",
                }
            },
            "plan": plan.__dict__,
            "device": "cpu",
            "dtype": "float32",
        }
    )
    assert result["status"] == "failed"
    assert result["artifacts"]["validation_report"]["errors"] == [
        "RuntimeError: injected"
    ]


def _manifest() -> DatasetManifest:
    return DatasetManifest(
        "fixture",
        "1",
        "rag",
        "smoke",
        "fixture",
        "CC0",
        "1",
        "checksum",
        "v1",
        42,
        (
            {"id": "doc", "kind": "document", "text": "alpha"},
            {
                "id": "other",
                "kind": "document",
                "text": "irrelevant",
            },
            {
                "id": "q",
                "kind": "question",
                "document_id": "doc",
                "question": "Where is alpha?",
                "answerable": True,
                "evidence_type": "text",
                "evidence": [
                    {
                        "id": "q-e1",
                        "evidence_type": "text",
                        "document_id": "doc",
                        "start": 0,
                        "end": 5,
                    }
                ],
            },
        ),
    )
