from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from edumind.rag.contracts import EmbeddingSpec
from edumind.rag.embedder import Embedder
from edumind.rag.errors import RAGConfigurationError
from edumind.rag.text_chunker import TokenChunkingStrategy
from edumind.rag.tokenizers import TiktokenOffsetTokenizer
from experiments.benchmarks.common.contracts import (
    BenchmarkPlan,
    DatasetManifest,
    SampleResult,
)
from experiments.benchmarks.common.datasets import (
    DatasetValidationError,
    validate_evidence,
)
from experiments.benchmarks.preparation.datasets import _answers_and_evidence
from experiments.benchmarks.rag.chunking_embedding import benchmark
from experiments.benchmarks.rag.chunking_embedding.metrics import (
    aggregate_quality,
    aggregate_repeatability,
    latency_intervals,
    paired_quality,
    ranking_repeatability,
    score_question,
)
from experiments.benchmarks.rag.chunking_embedding.profiles import embedding_spec
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    load_protocol as default_chunking_protocol,
)
from experiments.benchmarks.rag.chunking_embedding.protocol import (
    protocol_from_mapping as chunking_protocol_from_mapping,
)
from experiments.benchmarks.rag.chunking_embedding.strategies import (
    RecursiveCharacterChunkingStrategy,
    SemanticChunkingStrategy,
    StructureAwareChunkingStrategy,
    build_chunking_strategy,
)
from experiments.benchmarks.rag.evaluation import (
    Chunk,
    ExactIndex,
    InputCompatibilityError,
    _chunking_contract,
    dense_rank_with_scores,
    retrieval_metrics,
)

CHUNKING_PROTOCOL = default_chunking_protocol()
FIXTURE_EMBEDDING = replace(
    embedding_spec("Alibaba-NLP/gte-modernbert-base", revision="r", local_path="p"),
    dimension=2,
)
FIXTURE_PHASE_SECONDS = {
    "cold_model_load_seconds": 0.5,
    "chunking_seconds": 0.2,
    "chunk_embedding_seconds": 0.6,
    "dense_index_build_seconds": 0.6,
    "index_build_seconds": 0.6,
}


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
    question = {
        **_manifest().samples[-1],
        "evidence": [
            {
                "id": name,
                "evidence_type": "text",
                "document_id": "doc",
                "start": start,
                "end": end,
            }
            for name, start, end in (("a", 0, 5), ("b", 6, 10))
        ],
    }
    corpus = [
        Chunk("a", "doc", "alpha", 0, 5, 1),
        Chunk("repeat-a", "doc", "alpha ", 0, 6, 1),
        Chunk("b", "doc", "beta", 6, 10, 1),
    ]

    def score(ranking):
        return score_question(
            question, ranking, corpus, WordTokenizer(), cutoffs=(3,), alpha=0.5
        ).metrics["alpha_ndcg_at_3"]

    duplicate_first = score(corpus)
    novel_first = score([corpus[0], corpus[2], corpus[1]])
    assert 0.0 <= duplicate_first < novel_first <= 1.0


def test_quality_uses_three_and_five_cutoffs() -> None:
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
        Chunk("a", "doc", "alpha", 0, 5, 1),
        Chunk("d1", "other", "x", 0, 1, 1),
        Chunk("d2", "other", "y", 1, 2, 1),
        Chunk("b", "doc", "beta", 6, 10, 1),
        Chunk("d3", "other", "z", 2, 3, 1),
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


def test_single_unit_questions_receive_alpha_ndcg_and_its_metric_contract() -> None:
    manifest = _manifest()
    question = next(row for row in manifest.samples if row.get("kind") == "question")
    chunk = Chunk("a", "doc", "alpha", 0, 5, 1)

    score = score_question(
        question, [chunk], [chunk], WordTokenizer(), cutoffs=(3, 5), alpha=0.5
    )
    directions, _ = benchmark.directions_for(manifest, CHUNKING_PROTOCOL)

    assert score.metrics["alpha_ndcg_at_3"] == 1.0
    assert "quality.overall.alpha_ndcg_at_3" in directions

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
    multi_directions, _ = benchmark.directions_for(multi_evidence, CHUNKING_PROTOCOL)
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
    partial = Chunk("partial", "doc", "alpha", 0, 5, 1)
    complete = Chunk("complete", "doc", "alpha beta gamma", 0, len(text), 3)
    distractor = Chunk("other", "other", "alpha beta", 0, 10, 2)

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
    assert complete_score.metrics["evidence_token_precision_at_5"] == pytest.approx(
        2 / 3
    )
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


@pytest.mark.parametrize(
    ("counts", "vector", "error"),
    [
        ([1], [2.0, 0.0], None),
        ([True], None, "Native-token counts"),
        ([1.5], None, "Native-token counts"),
        ([float("nan")], None, "Native-token counts"),
        ([], None, "Native-token counts"),
        ([0], None, "Native-token counts"),
        ([1, 2], None, "Native-token counts"),
        ([1], [0.0, 0.0], "zero-norm"),
        ([1], [1e30, 1e30], "non-finite norms"),
        ([1], [float("nan"), 0.0], "non-finite values"),
    ],
)
def test_exact_cosine_ranking_validates_inputs_and_breaks_ties(counts, vector, error):
    class QueryEmbedder:
        @staticmethod
        def input_token_counts(texts, *, role):
            return counts

        @staticmethod
        def embed_query(_query):
            assert vector is not None, "invalid counts must fail before embedding"
            return np.asarray(vector)

    chunks = [
        Chunk("first", "doc", "a", 0, 1, 1),
        Chunk("second", "doc", "b", 1, 2, 1),
        Chunk("third", "doc", "c", 2, 3, 1),
    ]
    index = ExactIndex(
        chunks,
        np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        QueryEmbedder(),
        WordTokenizer(),
        None,
        SimpleNamespace(dimension=2, maximum_length=8192, query_device="cpu"),
        "chunker",
        {},
        1.0,
        {},
    )
    if error:
        with pytest.raises(RuntimeError, match=error):
            dense_rank_with_scores(index, "query", 3)
        return
    ranking = dense_rank_with_scores(index, "query", 3)
    assert [position for position, _ in ranking] == [0, 1, 2]
    assert ranking[0][1] == ranking[1][1] == 1.0


@pytest.mark.parametrize("shape", [(0, 768), (2, 768), (1, 2), (768,)])
def test_embedding_runtime_rejects_wrong_output_cardinality_and_dimensions(shape):
    runtime = Embedder(
        embedding_spec("Alibaba-NLP/gte-modernbert-base", revision="r", local_path="p")
    )
    runtime._models["cpu"] = SimpleNamespace(
        encode=lambda *args, **kwargs: np.ones(shape, dtype=np.float32)
    )
    with pytest.raises(RAGConfigurationError, match="shape mismatch"):
        runtime.embed_query("question")


def test_quality_aggregation_uses_documents_as_statistical_units() -> None:
    samples = [
        SampleResult(
            "q1",
            {"quality.overall.alpha_ndcg_at_5": 1.0},
            0.1,
            {"document_id": "a", "evidence_type": "text"},
        ),
        SampleResult(
            "q2",
            {"quality.overall.alpha_ndcg_at_5": 1.0},
            0.1,
            {"document_id": "a", "evidence_type": "text"},
        ),
        SampleResult(
            "q3",
            {"quality.overall.alpha_ndcg_at_5": 0.0},
            0.1,
            {"document_id": "b", "evidence_type": "text"},
        ),
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
        prompts: ClassVar[dict[str, str]] = {
            "query": "query: ",
            "document": "passage: ",
        }
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
    assert (
        embedding_spec(
            "Octen/Octen-Embedding-0.6B", revision="rev", local_path="path"
        ).interface
        == "query-document"
    )


def test_all_chunking_strategies_return_exact_nonempty_source_spans() -> None:
    text = (
        "# Section\n\nFirst sentence. Second sentence.\n\n"
        "| Header | Value |\n|---|---|\n| alpha | 1 |\n\n$$x^2$$\n" + "token " * 600
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
                        {
                            "id": "a",
                            "evidence_type": "text",
                            "document_id": "doc",
                            "start": 0,
                            "end": 5,
                        },
                        {
                            "id": "b",
                            "evidence_type": "formula",
                            "document_id": "doc",
                            "start": 6,
                            "end": 10,
                        },
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
    from experiments.benchmarks.rag import evaluation

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

    monkeypatch.setattr(
        evaluation, "TiktokenOffsetTokenizer", lambda *_: WordTokenizer()
    )
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
    monkeypatch.setattr(evaluation, "Embedder", lambda _spec, **_kwargs: Embedder())
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
    from experiments.benchmarks.rag import evaluation

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
    monkeypatch.setattr(
        evaluation, "Embedder", lambda _spec, **_kwargs: NativeEmbedder()
    )
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


@pytest.mark.parametrize("text", ["A🙂B", "漢字", "école", "مرحبا"])
def test_token_truncation_preserves_unicode_source_prefix(text):
    tokenizer = TiktokenOffsetTokenizer()
    for maximum in range(tokenizer.count(text) + 1):
        truncated = tokenizer.truncate(text, maximum)
        assert text.startswith(truncated)
        assert tokenizer.count(truncated) <= maximum


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

    monkeypatch.setattr(benchmark, "build_index", reject)
    result = benchmark.execute_payload(
        {
            "candidate": candidate,
            "manifest": _manifest().__dict__,
            "model_lock": {embedding_name: {"revision": "rev", "model_path": "path"}},
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
        FIXTURE_EMBEDDING,
        "chunker-fingerprint",
        {"name": "fixture", "fingerprint": "chunker-fingerprint"},
        2.0,
        {"status": "passed", "truncated_inputs": 0},
        FIXTURE_PHASE_SECONDS,
    )
    monkeypatch.setattr(benchmark, "build_index", lambda *_args, **_kwargs: index)

    def rank(*args, timings, **kwargs):
        timings.update(query_embedding_latency_ms=0.001, vector_search_latency_ms=0.001)
        return [(position, 1.0 - position / 100) for position in range(20)]

    monkeypatch.setattr(benchmark, "dense_rank_with_scores", rank)
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
    assert aggregate["quality.overall.alpha_ndcg_at_3"] == 1.0
    assert aggregate["validity.quality.overall.eligible_question_count"] == 1.0
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
    assert len(artifacts["retrievals"]) == 40
    assert {row["repetition"] for row in artifacts["retrievals"]} == {1, 2}
    assert len(artifacts["timings"]) == 6  # four phases plus measured queries
    assert set(artifacts) == {
        "chunk_manifest",
        "query_metrics",
        "retrievals",
        "evidence_matches",
        "timings",
        "validation_report",
        "resources",
    }


def test_measured_query_failure_runs_every_repetition(monkeypatch) -> None:
    index = ExactIndex(
        [Chunk("chunk", "doc", "alpha", 0, 5, 1, 3)],
        np.ones((1, 2), dtype=np.float32),
        object(),
        WordTokenizer(),
        None,
        FIXTURE_EMBEDDING,
        "chunker-fingerprint",
        {"name": "fixture", "fingerprint": "chunker-fingerprint"},
        1.0,
        {"status": "passed", "truncated_inputs": 0},
        FIXTURE_PHASE_SECONDS,
    )
    calls = 0

    def rank(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("injected")
        _kwargs["timings"].update(
            query_embedding_latency_ms=0.001, vector_search_latency_ms=0.001
        )
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
    assert len(caught.value.artifacts["timings"]) == 7
    assert caught.value.samples[0].latency_seconds is not None
    assert caught.value.samples[0].metrics["quality.overall.ndcg_at_3"] is None
    assert caught.value.metrics["quality.overall.ndcg_at_3"] is None
    assert (
        caught.value.artifacts["query_metrics"][0]["first_attempt_status"] == "failed"
    )


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

    monkeypatch.setattr(benchmark, "build_index", fail)
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
    assert (
        "RuntimeError: injected" in result["artifacts"]["validation_report"]["errors"]
    )


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


def test_linear_ndcg_credits_multiple_units_without_exponential_gain():
    question = _manifest().samples[-1]
    question = {
        **question,
        "evidence": [
            *question["evidence"],
            {
                "id": "e2",
                "evidence_type": "text",
                "document_id": "doc",
                "start": 6,
                "end": 10,
            },
        ],
    }
    one = Chunk("one", "doc", "alpha", 0, 5, 1)
    two = Chunk("two", "doc", "alpha beta", 0, 10, 2)
    score = score_question(
        question, [one, two], [one, two], WordTokenizer(), cutoffs=(3,), alpha=0.5
    )
    discount = 1 / np.log2(3)
    assert score.metrics["ndcg_at_3"] == pytest.approx(
        (1 + 2 * discount) / (2 + discount)
    )


@pytest.mark.parametrize("ranking", [[], [Chunk("empty", "doc", "", 0, 0, 0)]])
def test_token_free_retrieval_is_not_a_zero_quality_observation(ranking):
    with pytest.raises(ValueError, match="tokens|token-free"):
        score_question(
            _manifest().samples[-1],
            ranking,
            ranking,
            WordTokenizer(),
            cutoffs=(3,),
            alpha=0.5,
        )


def test_partial_quality_stays_diagnostic_and_failed_documents_enter_draws():
    name = "quality.overall.ndcg_at_3"
    samples = [
        SampleResult(
            "a", {name: 1.0}, 0.1, {"document_id": "a", "evidence_type": "text"}
        ),
        SampleResult(
            "b", {name: 0.0}, 0.2, {"document_id": "b", "evidence_type": "text"}
        ),
        SampleResult(
            "c",
            {name: None},
            None,
            {
                "document_id": "c",
                "evidence_type": "text",
                "first_attempt_status": "failed",
            },
        ),
    ]
    diagnostics = {}
    aggregate, intervals = aggregate_quality(
        samples, resamples=100, seed=42, confidence=0.95, diagnostics=diagnostics
    )
    assert aggregate[name] is None
    assert name not in intervals
    assert diagnostics[name]["estimate"] == 0.5
    assert diagnostics[name]["interval"]["independent_source_count"] == 3
    assert diagnostics[name]["interval"]["undefined_resamples"] > 0
    assert diagnostics[name]["failed_question_count"] == 1


def test_supported_degenerate_ci_is_retained_and_one_document_has_no_bounds():
    name = "quality.overall.ndcg_at_3"
    samples = [
        SampleResult(
            str(i), {name: 0.0}, 0.1, {"document_id": str(i), "evidence_type": "text"}
        )
        for i in range(2)
    ]
    _, intervals = aggregate_quality(samples, resamples=100, seed=42, confidence=0.95)
    assert intervals[name]["lower"] == intervals[name]["upper"] == 0.0
    assert intervals[name]["degenerate"] is True
    _, intervals = aggregate_quality(
        samples[:1], resamples=100, seed=42, confidence=0.95
    )
    assert intervals[name]["lower"] is None
    assert intervals[name]["reason"] == "insufficient_independent_sources"


@pytest.mark.parametrize("failed_attempt", [1, 2, 3])
def test_first_quality_and_successful_latency_have_separate_support(
    monkeypatch, failed_attempt
):
    chunks = [Chunk("a", "doc", "alpha", 0, 5, 1, 3)]
    index = ExactIndex(
        chunks,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        object(),
        WordTokenizer(),
        None,
        FIXTURE_EMBEDDING,
        "fixture",
        {},
        1.0,
        {},
        FIXTURE_PHASE_SECONDS,
    )
    monkeypatch.setattr(benchmark, "build_index", lambda *args, **kwargs: index)
    calls = 0

    def rank(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == failed_attempt:
            raise RuntimeError("injected")
        kwargs["timings"].update(
            query_embedding_latency_ms=0.001, vector_search_latency_ms=0.001
        )
        return [(0, 1.0)]

    monkeypatch.setattr(benchmark, "dense_rank_with_scores", rank)
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        "smoke",
        "fixture",
        ("fixture|embedding",),
        42,
        3,
        0,
        0,
        _protocol_settings(warmups=0, repetitions=3),
    )
    try:
        evaluated = benchmark.evaluate_candidate(
            "fixture|embedding",
            _manifest(),
            {"embedding": {"revision": "r", "model_path": "p"}},
            plan,
            device="cpu",
            dtype="float32",
        )
    except benchmark.CandidateExecutionError as exc:
        assert failed_attempt == 1
        result = exc
    else:
        assert failed_attempt != 1
        result = SimpleNamespace(
            **dict(
                zip(
                    (
                        "samples",
                        "operational",
                        "metrics",
                        "parameters",
                        "intervals",
                        "artifacts",
                    ),
                    evaluated,
                    strict=True,
                )
            )
        )
    assert calls == 3
    assert result.metrics["quality.overall.ndcg_at_3"] == (
        None if failed_attempt == 1 else 1.0
    )
    assert result.samples[0].latency_seconds is not None
    assert result.operational["peak_vram_mb"] is None
    assert result.parameters["measurement.resource_scope"] == "complete"


@pytest.mark.parametrize("text", ["A🙂B漢字é", "...!!!", "\n\nFirst.\nSecond?!"])
def test_all_strategies_preserve_source_characters_and_actual_token_counts(text):
    tokenizer = TiktokenOffsetTokenizer()
    for name in CHUNKING_PROTOCOL.strategies:
        strategy = build_chunking_strategy(
            name,
            protocol=CHUNKING_PROTOCOL,
            tokenizer=tokenizer,
            embed_sentences=lambda rows: np.ones((len(rows), 2)),
            semantic_embedding_fingerprint="fixture",
        )
        spans = strategy.split(text)
        assert spans
        assert all(
            count == tokenizer.count(text[start:end]) and count > 0
            for start, end, count in spans
        )
        assert all(
            any(start <= i < end for start, end, _ in spans)
            for i, char in enumerate(text)
            if not char.isspace()
        )


def test_protocol_rejects_missing_chunker_and_single_document_ci_support():
    payload = deepcopy(CHUNKING_PROTOCOL.meta.resolved)
    del payload["strategies"]["semantic"]
    with pytest.raises(ValueError, match="all eight"):
        chunking_protocol_from_mapping(payload)
    payload = deepcopy(CHUNKING_PROTOCOL.meta.resolved)
    payload["statistics"]["minimum_latency_ci_documents"] = 1
    with pytest.raises(ValueError):
        chunking_protocol_from_mapping(payload)


def test_sealed_data_report_is_reused_then_rejects_changed_input(tmp_path, monkeypatch):
    from edumind.common.artifacts import atomic_write_json
    from edumind.common.paths import PROJECT_ROOT
    from experiments.benchmarks.common.datasets import load_manifest
    from experiments.benchmarks.rag.chunking_embedding.validation import (
        prepare_report,
        verify_inputs,
    )

    source = PROJECT_ROOT / "data/benchmarks/rag/smoke.json"
    manifest = load_manifest(source)
    path = tmp_path / "smoke.json"
    atomic_write_json(path, manifest.__dict__)
    root = tmp_path / "reports"
    _, report = prepare_report(path, "smoke", CHUNKING_PROTOCOL, report_root=root)
    assert report["status"] == "passed"
    from experiments.benchmarks.common import datasets

    monkeypatch.setattr(
        datasets,
        "validate_evidence",
        lambda *_: pytest.fail("sealed verification must not reevaluate annotations"),
    )
    assert (
        verify_inputs(path, "smoke", CHUNKING_PROTOCOL, report_root=root)[1] == report
    )
    payload = {**manifest.__dict__, "revision": "changed"}
    atomic_write_json(path, payload)
    with pytest.raises(ValueError, match="Missing or stale"):
        verify_inputs(path, "smoke", CHUNKING_PROTOCOL, report_root=root)


@pytest.mark.parametrize("identifier", [" unit ", 1, None])
def test_invalid_evidence_ids_produce_a_failed_data_report(tmp_path, identifier):
    from edumind.common.artifacts import atomic_write_json
    from experiments.benchmarks.common.datasets import manifest_content_checksum
    from experiments.benchmarks.rag.chunking_embedding.validation import prepare_report

    manifest = deepcopy(_manifest().__dict__)
    manifest["samples"][-1]["evidence"][0]["id"] = identifier
    manifest["checksum"] = manifest_content_checksum(manifest["samples"])
    path = tmp_path / "smoke.json"
    atomic_write_json(path, manifest)
    destination, report = prepare_report(
        path, "smoke", CHUNKING_PROTOCOL, report_root=tmp_path / "reports"
    )
    assert destination.is_file()
    assert report["status"] == "failed"
    assert "stable ID" in report["errors"][0]


@pytest.mark.parametrize("payload", [[], None, {"samples": [None]}])
def test_malformed_manifest_structure_produces_a_failed_data_report(tmp_path, payload):
    from edumind.common.artifacts import atomic_write_json
    from experiments.benchmarks.rag.chunking_embedding.validation import prepare_report

    if isinstance(payload, dict):
        payload = {**_manifest().__dict__, **payload}
    path = tmp_path / "smoke.json"
    atomic_write_json(path, payload)
    destination, report = prepare_report(
        path, "smoke", CHUNKING_PROTOCOL, report_root=tmp_path / "reports"
    )
    assert destination.is_file()
    assert report["status"] == "failed"
    assert "JSON object" in report["errors"][0]


def test_alpha_greedy_exceedance_keeps_raw_ratio_and_caps_question_score():
    text = "abcdefghijklmnopqrstuvwxyz1234"
    evidence = [
        (7, 16),
        (14, 25),
        (13, 15),
        (6, 17),
        (9, 20),
        (3, 7),
        (22, 24),
        (20, 29),
    ]
    spans = [(10, 23), (11, 25), (13, 15), (13, 19), (10, 19), (14, 29), (5, 19)]
    chunks = [
        Chunk(str(i), "doc", text[start:end], start, end, 1)
        for i, (start, end) in enumerate(spans)
    ]
    score = score_question(
        {
            **_manifest().samples[-1],
            "evidence": [
                {
                    "id": str(i),
                    "evidence_type": "text",
                    "document_id": "doc",
                    "start": start,
                    "end": end,
                }
                for i, (start, end) in enumerate(evidence)
            ],
        },
        [chunks[5], chunks[6], chunks[1]],
        chunks,
        WordTokenizer(),
        cutoffs=(3,),
        alpha=0.5,
    )
    assert score.metrics["alpha_ndcg_at_3"] == 1
    assert score.diagnostics["alpha_ndcg_at_3.raw_ratio"] > 1
    assert score.diagnostics["alpha_ndcg_at_3.greedy_baseline_exceeded"] is True


@pytest.mark.parametrize(
    "ranking", [[], [(0, 1.0), (0, 0.9)], [(2, 1.0)], [(0, float("nan"))]]
)
def test_invalid_rankings_are_integrity_errors(ranking):
    from experiments.benchmarks.rag.evaluation import validate_ranking

    with pytest.raises(RuntimeError):
        validate_ranking(SimpleNamespace(chunks=[object()]), ranking, 20)


def test_partial_build_preserves_inventory_but_no_final_bytes_or_duration(monkeypatch):
    model = "Alibaba-NLP/gte-modernbert-base"

    def build(*args, progress, **kwargs):
        progress.update(
            chunks=[Chunk("a", "doc", "alpha", 0, 5, 1)],
            generation_complete=True,
            build_started=0.0,
            chunking_contract={"fingerprint": "actual-chunker", "size": 256},
            phase_seconds={"cold_model_load_seconds": 0.5, "chunking_seconds": 0.2},
            phase_elapsed_to_failure_seconds={"chunk_embedding_seconds": 0.3},
        )
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(benchmark, "build_index", build)
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        "smoke",
        "fixture",
        (f"token-256-32|{model}",),
        42,
        1,
        0,
        0,
        _protocol_settings(warmups=0),
    )
    with pytest.raises(benchmark.CandidateExecutionError) as caught:
        benchmark.evaluate_candidate(
            plan.candidates[0],
            _manifest(),
            {model: {"revision": "r", "model_path": "p"}},
            plan,
            device="cpu",
            dtype="float32",
        )
    result = caught.value
    assert result.metrics["workload.chunk_count"] == 1.0
    assert result.metrics["workload.embedding_matrix_bytes"] is None
    assert result.operational["corpus_build_seconds"] is None
    assert result.operational["cold_model_load_seconds"] == 0.5
    assert result.operational["chunking_seconds"] == 0.2
    assert result.operational["chunk_embedding_seconds"] is None
    phase = next(
        row for row in result.artifacts["timings"] if row["phase"] == "chunk_embedding"
    )
    assert phase["elapsed_to_failure_seconds"] == 0.3
    assert (
        next(row for row in result.artifacts["timings"] if row["phase"] == "query")[
            "status"
        ]
        == "unexecuted"
    )
    assert result.parameters["measurement.resource_scope"] == "partial"
    assert result.operational["peak_process_tree_ram_mb"] > 0
    assert result.parameters["chunking_fingerprint"] == "actual-chunker"
    assert (
        result.artifacts["chunk_manifest"][0]["chunking_fingerprint"]
        == "actual-chunker"
    )


def test_paired_ci_does_not_invent_support_before_data_review():
    from experiments.benchmarks.common.contracts import CandidateResult
    from experiments.benchmarks.common.runner import _paired_comparisons

    name = "quality.overall.ndcg_at_3"
    samples = tuple(
        SampleResult(str(i), {name: 0.5}, 0.1, {"document_id": str(i)})
        for i in range(3)
    )
    results = [
        CandidateResult(candidate, "success", "hash", {}, {}, samples, {})
        for candidate in ("a", "b")
    ]
    metric = _paired_comparisons(
        results,
        {name: "max"},
        resamples=100,
        seed=42,
        group_key="document_id",
        confidence=0.95,
        minimum_ci_units=None,
    )[0]["metrics"][name]
    assert metric["left_minus_right"] == 0
    assert metric["lower"] is None
    assert metric["reason"] == "ci_support_not_frozen"


def test_snowflake_protocol_config_is_executed_and_fingerprinted(monkeypatch):
    import sentence_transformers
    import torch

    calls = []

    class Rotary(torch.nn.Module):
        def __init__(self, dim=2, max_position_embeddings=8, base=10000, device=None):
            super().__init__()
            self.dim = dim
            self.max_position_embeddings = max_position_embeddings
            self.base = base
            self.register_buffer("inv_freq", torch.ones(dim, device=device))

    embeddings = torch.nn.Module()
    embeddings.word_embeddings = torch.nn.Embedding(3, 2)
    embeddings.register_buffer("position_ids", torch.full((8,), -1))
    embeddings.rotary_emb = Rotary()
    embeddings.rotary_emb.inv_freq.fill_(float("nan"))
    original_weights = embeddings.word_embeddings.weight.detach().clone()

    class Model:
        device = "cpu"
        max_seq_length = 8192

        def __getitem__(self, index):
            assert index == 0
            return SimpleNamespace(
                model=SimpleNamespace(
                    config=SimpleNamespace(
                        model_type="gte", rope_scaling=None, max_position_embeddings=8
                    ),
                    embeddings=embeddings,
                )
            )

    def load(_source, **kwargs):
        calls.append(kwargs)
        return Model()

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", load)
    name = "Snowflake/snowflake-arctic-embed-m-v2.0"
    spec = embedding_spec(
        name,
        revision="rev",
        local_path="path",
        config_options=CHUNKING_PROTOCOL.model_config_options(name),
    )
    Embedder(spec, dtype="float32").prepare()
    assert calls[0]["config_kwargs"] == {
        "unpad_inputs": False,
        "use_memory_efficient_attention": False,
    }
    assert spec.fingerprint != replace(spec, config_options=()).fingerprint
    assert embeddings.position_ids.tolist() == list(range(8))
    assert torch.isfinite(embeddings.rotary_emb.inv_freq).all()
    assert torch.equal(embeddings.word_embeddings.weight, original_weights)


def test_protocol_rejects_unpadding_without_its_attention_backend():
    payload = deepcopy(CHUNKING_PROTOCOL.meta.resolved)
    payload["index"]["model_config"]["Snowflake/snowflake-arctic-embed-m-v2.0"][
        "unpad_inputs"
    ] = True
    with pytest.raises(ValueError, match="unpadding requires"):
        chunking_protocol_from_mapping(payload)


def test_protocol_rejects_implicit_snowflake_configuration():
    payload = deepcopy(CHUNKING_PROTOCOL.meta.resolved)
    payload["index"]["model_config"] = {}
    with pytest.raises(ValueError, match="must be explicit"):
        chunking_protocol_from_mapping(payload)


def test_structure_chunker_does_not_split_a_formula_at_a_hash_heading():
    text = "# Topic\n\n$$\n# literal formula content\nx+y\n$$\n"
    strategy = build_chunking_strategy(
        "structure-aware-512-64", protocol=CHUNKING_PROTOCOL, tokenizer=WordTokenizer()
    )
    chunks = strategy.split(text)
    assert any(text[start:end].count("$$") == 2 for start, end, _ in chunks)


def test_zero_generated_chunks_are_known_but_not_a_completed_index(monkeypatch):
    from experiments.benchmarks.rag import evaluation

    monkeypatch.setattr(
        evaluation,
        "build_chunking_strategy",
        lambda *args, **kwargs: SimpleNamespace(
            name="empty",
            fingerprint="empty",
            tokenizer=WordTokenizer(),
            split=lambda _: [],
        ),
    )
    progress = {}
    with pytest.raises(RuntimeError, match="empty corpus"):
        evaluation.build_index(
            _manifest(),
            "token-256-32",
            "Alibaba-NLP/gte-modernbert-base",
            {"Alibaba-NLP/gte-modernbert-base": {"revision": "r", "model_path": "p"}},
            with_bm25=False,
            with_dense=False,
            device="cpu",
            dtype="float32",
            chunking_protocol=CHUNKING_PROTOCOL,
            progress=progress,
        )
    assert progress["generation_complete"] is True
    assert progress["chunks"] == []
    assert progress["processed_documents"] == 2


def test_fatal_worker_retains_missing_records_without_inventing_attempts(monkeypatch):
    from experiments.benchmarks.common.process import WorkerExecutionError

    def crash(*args, **kwargs):
        raise WorkerExecutionError("injected process exit", {"peak_vram_mb": 600})

    monkeypatch.setattr(benchmark, "run_json_worker", crash)
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        "smoke",
        "fixture",
        ("fixture|embedding",),
        42,
        3,
        0,
        0,
        _protocol_settings(warmups=0, repetitions=3),
    )
    with pytest.raises(WorkerExecutionError) as caught:
        benchmark.run_in_fresh_process(
            "fixture|embedding", _manifest(), {}, plan, device="cuda", dtype="float16"
        )
    assert caught.value.samples == ()
    report = caught.value.artifacts["validation_report"]
    assert report["missing_attempt_record_count"] == 3
    assert len(caught.value.artifacts["timings"]) == 3
    assert all(row["status"] == "missing" for row in caught.value.artifacts["timings"])
    assert report["completeness"]["outcome_ledger_complete"] is False
    assert "failed_attempt_count" not in report
    assert caught.value.operational["peak_vram_mb"] == 600
    assert caught.value.parameters["measurement.resource_scope"] == "partial"


@pytest.mark.parametrize(
    "invalid_field",
    [
        "metrics",
        "operational",
        "samples",
        "negative_peaks",
        "nan_latency",
        "negative_latency",
        "bool_latency",
    ],
)
def test_invalid_worker_schema_retains_missing_records_and_available_peaks(
    monkeypatch, invalid_field
):
    result = {
        "status": "success",
        "samples": [],
        "metrics": {},
        "operational": {"peak_vram_mb": 600},
    }
    result[invalid_field] = (
        [{"sample_id": "q"}] if invalid_field == "samples" else {"bad": "text"}
    )
    if invalid_field == "negative_peaks":
        result["metrics"] = {"bad": "text"}
        result["operational"] = {"peak_vram_mb": -1, "peak_process_tree_ram_mb": -2}
    if invalid_field.endswith("_latency"):
        result["samples"] = [
            {
                "sample_id": "q",
                "metrics": {},
                "latency_seconds": {
                    "nan_latency": float("nan"),
                    "negative_latency": -1,
                    "bool_latency": True,
                }[invalid_field],
            }
        ]
    monkeypatch.setattr(benchmark, "run_json_worker", lambda *args, **kwargs: result)
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        "smoke",
        "fixture",
        ("fixture|embedding",),
        42,
        1,
        0,
        1,
        settings=_protocol_settings(),
    )
    with pytest.raises(benchmark.WorkerExecutionError) as caught:
        benchmark.run_in_fresh_process(
            "fixture|embedding", _manifest(), {}, plan, device="cpu", dtype="float32"
        )
    error = caught.value
    assert (
        error.artifacts["validation_report"]["reason_code"] == "missing_worker_result"
    )
    assert error.parameters["measurement.resource_scope"] == "partial"
    if invalid_field not in {"operational", "negative_peaks"}:
        assert error.operational["peak_vram_mb"] == 600
    else:
        assert error.operational == {}


def test_successful_preflight_merges_supervisor_evidence_without_legacy_helper(
    monkeypatch,
):
    from experiments.benchmarks.common.preflight import _probe_candidate

    monkeypatch.setattr(
        benchmark,
        "run_json_worker",
        lambda *args, **kwargs: {
            "placement": {"status": "qualified"},
            "_worker_supervision": {
                "vram_measurement_method": "nvml-device-total",
                "peak_vram_mb": 600,
            },
        },
    )
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        "development",
        "fixture",
        ("fixture|embedding",),
        42,
        1,
        0,
        0,
        settings={
            "chunking_embedding_protocol": CHUNKING_PROTOCOL.meta.worker_payload()
        },
    )
    result = _probe_candidate(
        "fixture|embedding",
        lambda candidate: benchmark.preflight_in_fresh_process(
            candidate,
            _manifest(),
            {},
            plan,
            vram_limit_mb=3584,
            preflight=CHUNKING_PROTOCOL.preflight,
        ),
    )
    assert result["status"] == "qualified"
    assert result["peak_vram_mb"] == 600
    assert "_worker_supervision" not in result


def test_authoritative_data_validation_checks_complete_inventory_and_leakage(tmp_path):
    from edumind.common.artifacts import atomic_write_json
    from experiments.benchmarks.common.datasets import manifest_content_checksum
    from experiments.benchmarks.rag.chunking_embedding.validation import prepare_report

    payload = deepcopy(CHUNKING_PROTOCOL.meta.resolved)
    payload["data"]["qasper_papers"] = dict.fromkeys(
        ("development", "validation", "locked"), 1
    )
    payload["data"]["minimum_structured_questions"] = dict.fromkeys(
        ("table", "formula", "mixed"), 1
    )
    protocol = chunking_protocol_from_mapping(payload)
    manifests = []
    for split in ("development", "validation", "locked-test"):
        document = {
            "id": split,
            "kind": "document",
            "text": f"alpha beta {split}",
            "source_dataset": "allenai/qasper",
            "source_license": "CC-BY-4.0",
            "source_revision": "fixture",
            "independent_source_id": split,
        }
        questions = []
        for scope in ("table", "formula", "mixed"):
            types = ("text", "formula") if scope == "mixed" else (scope,)
            questions.append(
                {
                    "id": split + scope,
                    "kind": "question",
                    "question": "Which evidence?",
                    "document_id": split,
                    "answerable": True,
                    "evidence_type": scope,
                    "evidence": [
                        {
                            "id": f"{scope}-{i}",
                            "evidence_type": kind,
                            "document_id": split,
                            "start": i * 6,
                            "end": i * 6 + 4,
                        }
                        for i, kind in enumerate(types)
                    ],
                }
            )
        samples = [document, *questions]
        manifest = {
            **_manifest().__dict__,
            "split": split,
            "samples": samples,
            "checksum": manifest_content_checksum(samples),
        }
        path = tmp_path / f"{split}.json"
        atomic_write_json(path, manifest)
        manifests.append(path)
    root = tmp_path / "reports"
    options = {"inventory": manifests, "report_root": root}
    assert (
        prepare_report(manifests[0], "development", protocol, **options)[1]["status"]
        == "passed"
    )
    import json

    leaked = json.loads(manifests[1].read_text(encoding="utf-8"))
    leaked["samples"][0]["independent_source_id"] = "development"
    leaked["checksum"] = manifest_content_checksum(leaked["samples"])
    atomic_write_json(manifests[1], leaked)
    report = prepare_report(manifests[0], "development", protocol, **options)[1]
    assert report["status"] == "failed"
    assert "Cross-split source leakage" in report["errors"][0]


@pytest.mark.parametrize(
    "profile,source_profile", [("validation", "development"), ("locked", "validation")]
)
def test_phase_decisions_reject_stale_protocols_and_select_exact_winners(
    tmp_path, profile, source_profile
):
    from edumind.common.artifacts import atomic_write_json
    from experiments.benchmarks.rag.chunking_embedding.run import _selected_candidates

    payload = deepcopy(CHUNKING_PROTOCOL.meta.resolved)
    payload["statistics"].update(
        minimum_quality_ci_documents=2, minimum_repeatability_ci_documents=2
    )
    protocol = chunking_protocol_from_mapping(payload)
    candidate = protocol.smoke_pair
    summary = {
        "run_id": "source-run",
        "complete": True,
        "plan": {
            "suite": "rag",
            "stage": "chunking-embedding",
            "profile": source_profile,
            "settings": {"preflight_fingerprint": "current-qualification"},
        },
        "protocols": {"chunking_embedding": {"checksum": protocol.meta.checksum}},
        "candidates": [{"candidate": candidate, "status": "success"}],
    }
    source = tmp_path / "summary.json"
    decision = tmp_path / "decision.json"
    atomic_write_json(source, summary)
    atomic_write_json(
        decision,
        {
            "schema_version": 1,
            "source_summary": str(source),
            "source_run_id": "source-run",
            "selected_candidates": [candidate],
            "selected_by": "reviewer",
            "selected_date": "2026-10-13",
            "reason": "fixture selection",
        },
    )
    options = {"qualification_fingerprint": "current-qualification"}
    assert _selected_candidates(profile, decision, protocol, **options) == (candidate,)
    summary["plan"]["settings"]["preflight_fingerprint"] = "old-qualification"
    atomic_write_json(source, summary)
    with pytest.raises(ValueError, match="different hardware qualification"):
        _selected_candidates(profile, decision, protocol, **options)
    summary["plan"]["settings"]["preflight_fingerprint"] = "current-qualification"
    summary["protocols"]["chunking_embedding"]["checksum"] = "old-protocol"
    atomic_write_json(source, summary)
    with pytest.raises(ValueError, match="different chunking_embedding protocol"):
        _selected_candidates(profile, decision, protocol, **options)


def test_unchanged_production_embedding_keeps_its_persistent_fingerprint():
    from dataclasses import asdict

    from edumind.common.artifacts import stable_hash

    spec = embedding_spec(
        "Alibaba-NLP/gte-modernbert-base", revision="rev", local_path="path"
    )
    previous_contract = asdict(spec)
    del previous_contract["config_options"]
    del previous_contract["buffer_initialization"]
    assert spec.fingerprint == stable_hash(previous_contract)


@pytest.mark.parametrize(
    "outcomes,expected,status",
    [
        (["A", "A", "A"], 1, "scored"),
        (["A", "A", "B"], 1 / 3, "scored"),
        ([None, "A", "A"], 1 / 3, "scored"),
        ([None, None, None], 0, "scored"),
        (["A", "B", "C"], 0, "scored"),
        (["A"], None, "inapplicable"),
        (["A", "A", "unexecuted"], None, "incomplete"),
    ],
)
def test_ce_repeatability_uses_every_scheduled_pair(outcomes, expected, status):
    attempts = [
        {
            "repetition": rep,
            "status": "failed"
            if value is None
            else "unexecuted"
            if value == "unexecuted"
            else "success",
            "ranking_ids": [value] if value is not None else None,
        }
        for rep, value in enumerate(outcomes, 1)
    ]
    result = ranking_repeatability(attempts, depth=5)
    assert result["value"] == expected
    assert result["status"] == status
    assert len(result["pairs"]) == len(outcomes) * (len(outcomes) - 1) // 2


def test_ce_repeatability_depth_and_source_bootstrap_keep_query_weighting():
    attempts = [
        {"repetition": rep, "status": "success", "ranking_ids": [*range(5), tail]}
        for rep, tail in enumerate((6, 7, 8), 1)
    ]
    assert ranking_repeatability(attempts, depth=5)["value"] == 1
    name = "diagnostic.ranking_repeatability"
    samples = [
        SampleResult(
            str(i),
            {name: value},
            None,
            {
                "document_id": source,
                "repeatability_status": "scored",
                "scheduled_pair_count": 3,
                "successful_pair_count": int(value * 3),
            },
        )
        for i, (source, value) in enumerate((("a", 1), ("a", 0), ("b", 1)))
    ]
    aggregate, intervals, support = aggregate_repeatability(
        samples,
        resamples=100,
        seed=42,
        minimum_documents=2,
        confidence=0.95,
    )
    assert aggregate[name] == 0.75  # not the pooled query mean 2/3
    assert aggregate[name + ".sample_count"] == 2
    assert support["sample_unit"] == "source_document_macro_value"
    assert support["scored_question_count"] == 3
    assert intervals[name]["independent_source_count"] == 2
    assert intervals[name]["defined_resamples"] == 100
    assert intervals[name]["lower"] == 0.5
    assert intervals[name]["upper"] == 1
    _, intervals, _ = aggregate_repeatability(
        samples,
        resamples=100,
        seed=42,
        minimum_documents=None,
        confidence=0.95,
    )
    assert intervals[name]["reason"] == "ci_support_not_frozen"
    samples[-1].metadata["repeatability_status"] = "incomplete"
    samples[-1].metrics[name] = None
    aggregate, intervals, support = aggregate_repeatability(
        samples,
        resamples=100,
        seed=42,
        minimum_documents=2,
        confidence=0.95,
    )
    assert aggregate[name] is None and intervals == {}
    assert support["status"] == "incomplete"


def _ce_timed_fixture(
    monkeypatch, outcomes, *, warmups=0, phases=None, manifest=None, repetitions=None
):
    """No model loading, GPU access, or downloads; distinct completed stage clocks."""
    lengths = (1, 2, 3, 4, 5, 6)
    chunks = [
        Chunk(str(i), "doc", "alpha " * size, 0, 6 * size, size, size)
        for i, size in enumerate(lengths)
    ]
    index = ExactIndex(
        chunks,
        np.ones((6, 2), dtype=np.float32),
        object(),
        WordTokenizer(),
        None,
        FIXTURE_EMBEDDING,
        "fixture",
        {},
        1.0,
        {},
        FIXTURE_PHASE_SECONDS if phases is None else phases,
    )
    monkeypatch.setattr(benchmark, "seed_deterministically", lambda seed: None)
    monkeypatch.setattr(benchmark, "build_index", lambda *args, **kwargs: index)
    scheduled = iter(outcomes)

    def rank(*args, timings=None, **kwargs):
        if timings is None:  # warmup fails outside measured requests when requested
            raise RuntimeError("warmup failed")
        outcome = next(scheduled)
        timings.update(outcome.get("timings", {}))
        if outcome.get("error") is not None:
            raise outcome["error"]
        return [
            (position, 0.9 - i / 100)
            for i, position in enumerate(outcome.get("order", range(6)))
        ]

    monkeypatch.setattr(benchmark, "dense_rank_with_scores", rank)
    repetitions = len(outcomes) if repetitions is None else repetitions
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        "smoke",
        "fixture",
        ("fixture|embedding",),
        42,
        repetitions,
        0,
        warmups,
        _protocol_settings(warmups=warmups, repetitions=repetitions),
    )
    if manifest is None:
        base = _manifest()
        manifest = replace(
            base, samples=({**base.samples[0], "text": "alpha " * 6}, *base.samples[1:])
        )
    return benchmark.evaluate_candidate(
        "fixture|embedding",
        manifest,
        {"embedding": {"revision": "r", "model_path": "p"}},
        plan,
        device="cpu",
        dtype="float32",
    )


def test_ce_timers_context_and_saved_attempts_use_independent_cohorts(monkeypatch):
    samples, operational, metrics, parameters, _, artifacts = _ce_timed_fixture(
        monkeypatch,
        [
            {
                "timings": {
                    "query_embedding_latency_ms": 10,
                    "vector_search_latency_ms": 4,
                }
            },
            {
                "timings": {"query_embedding_latency_ms": 100},
                "error": RuntimeError("search failed"),
            },
            {
                "timings": {
                    "query_embedding_latency_ms": 20,
                    "vector_search_latency_ms": 6,
                }
            },
        ],
    )
    assert operational["cold_model_load_seconds"] == 0.5
    assert operational["chunking_seconds"] == 0.2
    assert operational["chunk_embedding_seconds"] == 0.6
    assert operational["corpus_build_seconds"] == 1
    assert operational["query_embedding_latency_ms_p50"] == 20
    assert operational["vector_search_latency_ms_p50"] == 5
    assert metrics["diagnostic.ranking_repeatability"] == pytest.approx(1 / 3)
    assert metrics["workload.retrieved_context_tokens_at_3_mean"] == 6
    assert metrics["workload.retrieved_context_tokens_at_5_p95"] == 15
    assert parameters["completeness"]["accepted"] is True
    assert parameters["measurement.resource_scope"] == "complete"
    assert (
        parameters["latency_support"]["query_embedding_latency_ms"][
            "contributing_question_count"
        ]
        == 1
    )
    assert (
        samples[0].metadata["query_embedding_latency_ms.completed_attempt_count"] == 3
    )
    failed = next(row for row in artifacts["timings"] if row.get("repetition") == 2)
    assert failed["latency_ms"] is None and failed["elapsed_to_failure_ms"] is not None
    assert failed["query_embedding_status"] == "success"
    assert failed["vector_search_status"] == "failed"
    assert {row["repetition"] for row in artifacts["retrievals"]} == {1, 2, 3}
    assert (
        len([row for row in artifacts["retrievals"] if row["status"] == "success"])
        == 12
    )
    assert artifacts["validation_report"]["status"] == "passed"
    assert len(artifacts["validation_report"]["request_errors"]) == 1


def test_ce_valid_changed_order_is_diagnostic_not_integrity_failure(monkeypatch):
    durations = {"query_embedding_latency_ms": 1, "vector_search_latency_ms": 2}
    _, _, metrics, parameters, _, artifacts = _ce_timed_fixture(
        monkeypatch,
        [
            {"timings": durations},
            {"timings": durations},
            {"timings": durations, "order": [1, 0, 2, 3, 4, 5]},
        ],
    )
    assert metrics["diagnostic.ranking_repeatability"] == pytest.approx(1 / 3)
    assert parameters["completeness"]["accepted"]
    assert artifacts["query_metrics"][0]["retrieved_context_tokens_at_3"] == 6
    assert metrics["quality.overall.ndcg_at_3"] == 1
    assert len(artifacts["retrievals"]) == 18


def test_ce_failed_first_does_not_borrow_later_quality_or_context(monkeypatch):
    with pytest.raises(benchmark.CandidateExecutionError) as caught:
        _ce_timed_fixture(
            monkeypatch,
            [
                {"error": RuntimeError("first failed")},
                {
                    "timings": {
                        "query_embedding_latency_ms": 1,
                        "vector_search_latency_ms": 2,
                    }
                },
                {
                    "timings": {
                        "query_embedding_latency_ms": 1,
                        "vector_search_latency_ms": 2,
                    }
                },
            ],
        )
    result = caught.value
    assert result.metrics["quality.overall.ndcg_at_3"] is None
    assert result.metrics["workload.retrieved_context_tokens_at_3_mean"] is None
    assert result.metrics["diagnostic.ranking_repeatability"] == pytest.approx(1 / 3)
    assert result.operational["cold_model_load_seconds"] == 0.5
    assert result.parameters["completeness"] == {
        "outcome_ledger_complete": True,
        "execution_complete": True,
        "quality_complete": False,
        "integrity_passed": True,
        "accepted": False,
    }


def test_ce_warmup_failure_accounts_for_unexecuted_not_failed_attempts(monkeypatch):
    with pytest.raises(benchmark.CandidateExecutionError) as caught:
        _ce_timed_fixture(monkeypatch, [{}, {}, {}], warmups=1)
    result = caught.value
    assert result.metrics["validity.failed_attempt_count"] == 0
    assert result.metrics["validity.unexecuted_attempt_count"] == 3
    assert result.metrics["diagnostic.ranking_repeatability"] is None
    assert result.parameters["repeatability_support"]["status"] == "incomplete"
    assert result.parameters["measurement.resource_scope"] == "partial"
    assert len(result.artifacts["retrievals"]) == 3
    assert result.operational["chunking_seconds"] == 0.2


@pytest.mark.parametrize("missing", ["phases", "component"])
def test_ce_missing_completed_timing_never_becomes_zero_or_accepted(
    monkeypatch, missing
):
    with pytest.raises(benchmark.CandidateExecutionError) as caught:
        _ce_timed_fixture(
            monkeypatch,
            [
                {
                    "timings": {}
                    if missing == "component"
                    else {
                        "query_embedding_latency_ms": 1,
                        "vector_search_latency_ms": 2,
                    }
                }
            ],
            phases={} if missing == "phases" else None,
        )
    name = (
        "cold_model_load_seconds"
        if missing == "phases"
        else "query_embedding_latency_ms_p50"
    )
    assert caught.value.operational[name] is None
    assert caught.value.metrics["quality.overall.ndcg_at_3"] == 1
    assert caught.value.parameters["completeness"]["integrity_passed"] is False


def test_ce_component_latency_intervals_keep_all_sources_and_own_query_medians():
    samples = [
        SampleResult(
            str(i),
            {},
            None,
            {"document_id": source, "query_embedding_latency_ms": value},
        )
        for i, (source, value) in enumerate(
            (("a", 1), ("a", 3), ("b", 10), ("c", None))
        )
    ]
    intervals = latency_intervals(
        samples,
        resamples=100,
        seed=42,
        minimum_documents=2,
        confidence=0.95,
        metric="query_embedding_latency_ms",
    )
    assert intervals["operational.query_embedding_latency_ms_p50"]["estimate"] == 3
    bounds = intervals["operational.query_embedding_latency_ms_p95"]
    assert bounds["estimate"] == pytest.approx(9.3)
    assert bounds["independent_source_count"] == 3
    assert bounds["contributing_document_count"] == 2
    assert bounds["undefined_resamples"] > 0


@pytest.mark.parametrize(
    "field,value",
    [("depth", 0), ("depth", 21), ("minimum_repeatability_ci_documents", 1)],
)
def test_ce_repeatability_protocol_rejects_invalid_controls(field, value):
    payload = deepcopy(CHUNKING_PROTOCOL.meta.resolved)
    payload["repeatability" if field == "depth" else "statistics"][field] = value
    with pytest.raises(ValueError):
        chunking_protocol_from_mapping(payload)


def test_ce_reviewed_groups_preserve_document_macro_and_cluster_draw_copies():
    from experiments.benchmarks.common.contracts import CandidateResult

    quality, repeat = "quality.overall.ndcg_at_3", "diagnostic.ranking_repeatability"
    samples = [
        SampleResult(
            str(i),
            {quality: value, repeat: value},
            latency / 1000,
            {
                "document_id": doc,
                "evidence_type": "text",
                "repeatability_status": "scored",
                "query_embedding_latency_ms": latency,
                # Exercise the reviewed alias, not just canonical independent IDs.
                "source_group_id" if doc == "a2" else "independent_source_id": group,
            },
        )
        for i, (doc, group, value, latency) in enumerate(
            (
                ("a1", "A", 1, 10),
                ("a1", "A", 0, 20),
                ("a2", "A", 1, 30),
                ("a3", "A", 1, 40),
                ("b", "B", 0, 50),
            )
        )
    ]
    options = {"resamples": 400, "seed": 42, "confidence": 0.95, "minimum_documents": 2}
    aggregate, intervals = aggregate_quality(samples, metric_names=[quality], **options)
    assert aggregate[quality] == 0.625  # four document means, not two group means
    assert aggregate[quality + ".sample_count"] == 4
    assert intervals[quality]["eligible_document_count"] == 4
    assert intervals[quality]["independent_source_count"] == 2
    assert intervals[quality]["lower"] == 0
    assert intervals[quality]["upper"] == pytest.approx(5 / 6)
    aggregate, intervals, _ = aggregate_repeatability(samples, **options)
    assert aggregate[repeat] == 0.625
    assert intervals[repeat]["upper"] == pytest.approx(5 / 6)
    latency = latency_intervals(samples, metric="query_embedding_latency_ms", **options)
    p50 = latency["operational.query_embedding_latency_ms_p50"]
    assert p50["estimate"] == 30
    assert p50["lower"] == 25 and p50["upper"] == 50
    # Requiring three independent units cannot be satisfied by four documents.
    _, intervals = aggregate_quality(
        samples,
        metric_names=[quality],
        **{**options, "minimum_documents": 3},
    )
    assert intervals[quality]["reason"] == "insufficient_independent_sources"
    left = CandidateResult("left", "success", "left", {}, {}, tuple(samples), {})
    right = replace(
        left,
        candidate="right",
        samples=tuple(replace(row, metrics={quality: 0}) for row in samples),
    )
    comparison = paired_quality([left, right], [quality], **options)[0]["metrics"][
        quality
    ]
    assert comparison["left_minus_right"] == 0.625
    assert comparison["paired_resampling_units"] == 2
    assert comparison["paired_documents"] == 4
    assert comparison["paired_samples"] == 5
    assert comparison["upper"] == pytest.approx(5 / 6)


def test_ce_context_descriptors_are_query_weighted_and_partial_is_diagnostic(
    monkeypatch,
):
    base = _manifest()
    queries = [{**base.samples[-1], "id": f"q{i}"} for i in range(3)]
    manifest = replace(
        base,
        samples=(
            {**base.samples[0], "text": "alpha " * 6, "source_group_id": "reviewed"},
            base.samples[1],
            *queries,
        ),
    )
    duration = {"query_embedding_latency_ms": 1, "vector_search_latency_ms": 2}
    outcomes = [
        {"timings": duration},
        {"timings": duration},
        {"timings": duration, "order": [5, 4, 3, 2, 1, 0]},
    ]
    _, _, metrics, _, _, artifacts = _ce_timed_fixture(
        monkeypatch,
        outcomes,
        manifest=manifest,
        repetitions=1,
    )
    assert metrics["workload.retrieved_context_tokens_at_3_mean"] == 9
    assert metrics["workload.retrieved_context_tokens_at_3_p95"] == pytest.approx(14.1)
    assert metrics["operational.query_embedding_latency_ms_p50.sample_count"] == 3
    assert all(
        row["independent_source_id"] == "reviewed" for row in artifacts["query_metrics"]
    )
    outcomes[-1] = {"error": RuntimeError("missing first")}
    with pytest.raises(benchmark.CandidateExecutionError) as caught:
        _ce_timed_fixture(monkeypatch, outcomes, manifest=manifest, repetitions=1)
    result = caught.value
    assert result.metrics["workload.retrieved_context_tokens_at_3_mean"] is None
    assert (
        result.parameters["workload_support"]["retrieved_context_tokens_at_3"]["mean"]
        == 6
    )
    assert (
        result.parameters["workload_support"]["retrieved_context_tokens_at_3"]["status"]
        == "partial"
    )


def test_ce_context_survives_independent_offline_scorer_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("offline scoring failed")

    monkeypatch.setattr(benchmark, "score_question", fail)
    with pytest.raises(benchmark.CandidateExecutionError) as caught:
        _ce_timed_fixture(
            monkeypatch,
            [
                {
                    "timings": {
                        "query_embedding_latency_ms": 1,
                        "vector_search_latency_ms": 2,
                    }
                }
            ],
        )
    result = caught.value
    assert result.metrics["workload.retrieved_context_tokens_at_3_mean"] == 6
    assert result.metrics["quality.overall.ndcg_at_3"] is None
    assert result.parameters["measurement.resource_scope"] == "complete"


def test_ce_paired_parent_hook_writes_grouped_comparison_evidence(tmp_path):
    from experiments.benchmarks.rag.chunking_embedding.run import _paired_artifact

    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        "development",
        "fixture",
        (),
        42,
        3,
        10000,
        1,
        {"chunking_embedding_protocol": CHUNKING_PROTOCOL.meta.worker_payload()},
    )
    paths = _paired_artifact(tmp_path, [], plan)
    assert len(paths) == 1 and paths[0].name == "paired_comparisons.json"
    assert paths[0].read_text(encoding="utf-8").strip() == "[]"
    assert _paired_artifact(tmp_path, [], replace(plan, profile="smoke")) == []


@pytest.mark.parametrize("kind", ["invalid_ranking", "typed_integrity", "oom"])
def test_ce_later_integrity_and_hardware_failures_still_fail_gates(monkeypatch, kind):
    error = RuntimeError("invalid vector")
    error.reason_code = "output_integrity_failed"

    class OutOfMemoryError(RuntimeError):
        pass

    durations = {"query_embedding_latency_ms": 1, "vector_search_latency_ms": 2}
    bad = (
        {"timings": durations, "order": [0, 0, 2, 3, 4, 5]}
        if kind == "invalid_ranking"
        else {
            "error": error
            if kind == "typed_integrity"
            else OutOfMemoryError("gpu_oom"),
        }
    )
    with pytest.raises(benchmark.CandidateExecutionError) as caught:
        _ce_timed_fixture(
            monkeypatch, [{"timings": durations}, bad, {"timings": durations}]
        )
    result = caught.value
    assert result.metrics["quality.overall.ndcg_at_3"] == 1
    assert result.metrics["diagnostic.ranking_repeatability"] == pytest.approx(1 / 3)
    assert result.parameters["completeness"]["quality_complete"]
    assert not result.parameters["completeness"]["integrity_passed"]
    if kind == "invalid_ranking":
        assert any(
            row.get("raw_ranking") not in {None, "null"}
            for row in result.artifacts["retrievals"]
        )


def test_ce_unfrozen_ci_support_is_explicit_and_cannot_advance(monkeypatch):
    from experiments.benchmarks.rag.chunking_embedding import run

    assert CHUNKING_PROTOCOL.unfrozen_ci_policies == (
        "quality",
        "ranking_repeatability",
    )
    assert CHUNKING_PROTOCOL.meta.version == "chunking-embedding-v6"
    plan = run._plan(
        CHUNKING_PROTOCOL,
        _manifest(),
        (CHUNKING_PROTOCOL.smoke_pair,),
        "development",
        "cuda",
        "float16",
    )
    assert plan.settings["ci_support_policy_frozen"] is False
    assert plan.settings["reporting_interpretation"] == "exploratory"
    monkeypatch.setattr(
        run,
        "load_engineer_decision",
        lambda *args, **kwargs: SimpleNamespace(
            selected_candidates=(CHUNKING_PROTOCOL.smoke_pair,)
        ),
    )
    with pytest.raises(ValueError, match="frozen CI support policies"):
        run._selected_candidates(
            "validation",
            "unused",
            CHUNKING_PROTOCOL,
            qualification_fingerprint="current",
        )
