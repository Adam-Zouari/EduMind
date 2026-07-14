from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from edumind.common.config import load_settings
from edumind.extraction import ExtractedDocument, ExtractedSegment, ExtractionProfile, SourceKind
from edumind.rag.errors import RAGConfigurationError
from edumind.rag.rag_pipeline import RAGPipeline
from edumind.rag.tokenizers import RegexOffsetTokenizer
from edumind.rag.types import RetrievalHit


class FakeEmbedder:
    model_loaded = True

    def embed_chunks(self, chunks):
        return [replace(chunk, embedding=[1.0, 0.0]) for chunk in chunks]

    def embed_query(self, text):
        assert text.strip()
        return np.asarray([1.0, 0.0], dtype=np.float32)


class FakeStore:
    collection_name = "test"

    def __init__(self, tmp_path):
        self.persist_directory = tmp_path
        self.chunks = []
        self.manifest = None
        self.reset_called = False
        self.no_hits = False

    def ensure_manifest(self, manifest):
        self.manifest = manifest
        return manifest

    def replace_document(self, source_id, chunks):
        replaced = len(self.chunks)
        self.chunks = list(chunks)
        return replaced

    def query_dense(self, vector, *, top_k, filter_metadata=None):
        if self.no_hits:
            return []
        return [
            RetrievalHit(
                "dense",
                "alpha beta gamma",
                {"source": "notes.pdf", "page": 2},
                0.9,
                1,
                "dense",
                3,
            )
        ]

    def query_lexical(self, query, *, top_k, filter_metadata=None):
        if self.no_hits:
            return []
        return [
            RetrievalHit(
                "lexical",
                "beta evidence",
                {"source": "notes.pdf", "page": 3},
                2.0,
                1,
                "bm25",
                2,
            )
        ]

    def load_index_manifest(self):
        return self.manifest

    def get_collection_count(self):
        return len(self.chunks)

    def reset_collection(self):
        self.reset_called = True
        self.chunks = []


class FakeGenerator:
    def __init__(self):
        self.answer = "Alpha is supported [1]."

    def generate_with_results(self, query, results, **kwargs):
        assert query and results
        return self.answer

    def health_check(self):
        return True


class FakeReranker:
    name = "fake-reranker"

    def rerank(self, query, hits, limit):
        assert query and limit == 20
        return [replace(hit, retrieval_method=self.name) for hit in reversed(hits)]


def _pipeline(tmp_path):
    store = FakeStore(tmp_path)
    generator = FakeGenerator()
    pipeline = RAGPipeline(
        tokenizer=RegexOffsetTokenizer(),
        embedder=FakeEmbedder(),
        vector_store=store,
        generator=generator,
    )
    return pipeline, store, generator


def test_rag_pipeline_ingest_query_generate_stats_and_reset(tmp_path) -> None:
    pipeline, store, generator = _pipeline(tmp_path)
    first = pipeline.ingest_document(
        {"text": "alpha beta gamma", "source": "notes.pdf", "metadata": {"course": "ml"}}
    )
    assert first.chunks_created == 1 and first.chunks_replaced == 0
    extracted = ExtractedDocument(
        "lecture.pdf",
        "lecture.pdf",
        SourceKind.PDF,
        "checksum",
        "application/pdf",
        "delta epsilon",
        (ExtractedSegment("delta epsilon", 0, 13, page_number=1),),
        ExtractionProfile("default", "pypdf", "1"),
    )
    assert pipeline.ingest_documents([extracted])[0].chunks_replaced == 1
    hits = pipeline.query("alpha", top_k=1)
    assert len(hits) == 1 and hits[0].retrieval_method == "rrf"
    answer = pipeline.generate_answer("alpha", top_k=1)
    assert answer.answer.endswith("[1].") and not answer.warnings
    generator.answer = "Unsupported without a citation."
    assert pipeline.generate_answer("alpha").warnings == ("missing_or_invalid_citations",)
    stats = pipeline.get_stats()
    assert stats["model_loaded"] is True and stats["index_compatibility_key"]
    pipeline.reset()
    assert store.reset_called is True and store.manifest is not None


def test_rag_pipeline_empty_query_no_evidence_and_generator_gate(tmp_path) -> None:
    pipeline, store, _ = _pipeline(tmp_path)
    with pytest.raises(ValueError, match="must not be empty"):
        pipeline.query("  ")
    store.no_hits = True
    refusal = pipeline.generate_answer("unknown")
    assert refusal.warnings == ("no_retrieval_evidence",)
    pipeline.llm_generator = None
    with pytest.raises(ValueError, match="not initialized"):
        pipeline.generate_answer("alpha")


def test_rag_pipeline_uses_configured_production_reranker_stack(tmp_path) -> None:
    settings = load_settings(
        overrides={
            "retrieval": {
                "strategy": "rrf-minilm-reranker",
                "reranker_revision": "abc123",
            }
        }
    )
    pipeline = RAGPipeline(
        settings=settings,
        tokenizer=RegexOffsetTokenizer(),
        embedder=FakeEmbedder(),
        vector_store=FakeStore(tmp_path),
        generator=FakeGenerator(),
        reranker=FakeReranker(),
    )
    assert pipeline.query("alpha", top_k=1)[0].retrieval_method == "fake-reranker"
    assert pipeline.get_stats()["retrieval_strategy"] == "rrf-minilm-reranker"


def test_rag_pipeline_requires_and_uses_ollama_digest_lock(tmp_path) -> None:
    missing_settings = load_settings(
        overrides={"generation": {"model_lock_path": str(tmp_path / "missing.json")}}
    )
    with pytest.raises(RAGConfigurationError, match="Missing Ollama model lock"):
        RAGPipeline(
            settings=missing_settings,
            use_llm=True,
            tokenizer=RegexOffsetTokenizer(),
            embedder=FakeEmbedder(),
            vector_store=FakeStore(tmp_path),
        )

    lock = tmp_path / "ollama.json"
    lock.write_text(json.dumps({"models": {"qwen3:1.7b": "sha256:locked"}}), encoding="utf-8")
    settings = load_settings(overrides={"generation": {"model_lock_path": str(lock)}})
    pipeline = RAGPipeline(
        settings=settings,
        use_llm=True,
        tokenizer=RegexOffsetTokenizer(),
        embedder=FakeEmbedder(),
        vector_store=FakeStore(tmp_path),
    )
    assert pipeline.llm_generator is not None
    assert pipeline.llm_generator.profile.digest == "sha256:locked"
