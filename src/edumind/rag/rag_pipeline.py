"""Production indexing, token-budget retrieval, and grounded generation."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from edumind.common.artifacts import stable_hash
from edumind.common.config import Settings, load_settings
from edumind.extraction import ExtractedDocument

from .contracts import (
    ChunkingStrategy,
    GenerationProfile,
    IndexManifest,
    Reranker,
    embedding_spec,
    load_recommendation_manifest,
)
from .document_processor import normalize_ingest_document
from .embedder import Embedder
from .errors import RAGConfigurationError
from .llm_generator import OllamaGenerator
from .retrieval import StoreRetrieval, base_retrieval_strategy, build_reranker
from .text_chunker import (
    SemanticChunkingStrategy,
    TextChunker,
    TokenChunkingStrategy,
    build_chunking_strategy,
)
from .tokenizers import LazyHuggingFaceOffsetTokenizer, OffsetTokenizer, TiktokenOffsetTokenizer
from .types import AnswerResult, IngestDocument, IngestReport, RetrievalHit, VectorStoreSettings
from .vector_store import VectorStore


class RAGPipeline:
    """The sole production path used by product services and benchmark runners."""

    def __init__(
        self,
        config_path: str | None = None,
        use_llm: bool = False,
        *,
        settings: Settings | None = None,
        tokenizer: OffsetTokenizer | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        generator: OllamaGenerator | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.settings = settings or load_settings(config_path)
        self.recommendation = load_recommendation_manifest()
        self.embedding_spec = embedding_spec(
            self.settings.embedding.model_name,
            document_device=self.settings.embedding.indexing_device,
            query_device=self.settings.embedding.query_device,
            revision=self.settings.embedding.revision,
        )
        configured_contract = {
            "dimension": self.settings.embedding.dimension,
            "query_prefix": self.settings.embedding.query_prefix,
            "document_prefix": self.settings.embedding.document_prefix,
            "normalize": self.settings.embedding.normalize,
            "similarity": self.settings.embedding.similarity,
            "maximum_length": self.settings.embedding.maximum_length,
        }
        expected_contract = {key: getattr(self.embedding_spec, key) for key in configured_contract}
        if configured_contract != expected_contract:
            mismatches = [
                key for key, value in configured_contract.items() if value != expected_contract[key]
            ]
            raise RAGConfigurationError(
                "Embedding configuration violates the audited model contract for "
                f"{self.embedding_spec.model_name}: {', '.join(mismatches)}"
            )
        self.tokenizer = tokenizer or (
            LazyHuggingFaceOffsetTokenizer(
                self.embedding_spec.tokenizer,
                self.embedding_spec.revision,
            )
            if self.settings.chunking.tokenizer == "embedding"
            else TiktokenOffsetTokenizer(self.settings.chunking.tokenizer)
        )
        self.embedder = embedder or Embedder(spec=self.embedding_spec)
        strategy: ChunkingStrategy
        if self.settings.chunking.strategy in {"token", "token-384-64"}:
            strategy = TokenChunkingStrategy(
                self.tokenizer,
                self.settings.chunking.chunk_size,
                self.settings.chunking.chunk_overlap,
                f"token-{self.settings.chunking.chunk_size}-{self.settings.chunking.chunk_overlap}",
            )
        elif self.settings.chunking.strategy == "semantic":
            strategy = SemanticChunkingStrategy(
                self.tokenizer,
                self.embedder.embed_texts,
                maximum_tokens=self.settings.chunking.chunk_size,
            )
        else:
            strategy = build_chunking_strategy(
                self.settings.chunking.strategy, tokenizer=self.tokenizer
            )
        self.text_chunker = TextChunker(strategy=strategy, embedder=self.embedder)
        self.vector_store = vector_store or VectorStore(
            VectorStoreSettings(
                self.settings.vector.collection_name,
                self.settings.vector.persist_directory,
                self.settings.vector.distance_metric,
            )
        )
        self.index_manifest = IndexManifest(
            schema_version=1,
            content_checksum=stable_hash([]),
            embedding_contract=self.embedding_spec.fingerprint,
            chunking_contract=strategy.fingerprint,
            backend=self.settings.vector.backend,
            collection_name=self.settings.vector.collection_name,
        )
        self.vector_store.ensure_manifest(self.index_manifest)
        self.retrieval = StoreRetrieval(
            self.vector_store,
            base_retrieval_strategy(self.settings.retrieval.strategy),
            self.settings.retrieval.rrf_k,
        )
        self.retrieval_stack_name = self.settings.retrieval.strategy
        self.reranker = reranker or build_reranker(
            self.settings.retrieval.strategy,
            revision=self.settings.retrieval.reranker_revision,
            device=self.settings.retrieval.reranker_device,
        )
        generation_digest = self.settings.generation.digest
        if use_llm and generator is None and generation_digest == "unpinned":
            generation_digest = _locked_ollama_digest(
                self.settings.generation.model_lock_path,
                self.settings.generation.model_name,
            )
        generation_profile = GenerationProfile(
            self.settings.generation.model_name,
            generation_digest,
            self.settings.generation.thinking,
            self.settings.generation.temperature,
            self.settings.generation.seed,
            self.settings.generation.context_tokens,
            self.settings.generation.maximum_answer_tokens,
            self.settings.generation.keep_alive,
        )
        self.llm_generator = generator or (
            OllamaGenerator(
                generation_profile,
                base_url=self.settings.generation.base_url,
                timeout_seconds=self.settings.generation.timeout_seconds,
            )
            if use_llm
            else None
        )
        self.top_k = self.settings.retrieval.top_k

    def ingest_document(
        self, document: IngestDocument | Mapping[str, object] | ExtractedDocument
    ) -> IngestReport:
        started = time.perf_counter()
        normalized = self._normalize_document(document)
        chunks = self.text_chunker.chunk_document(normalized)
        embedded = self.embedder.embed_chunks(chunks)
        replaced = self.vector_store.replace_document(normalized.source_id, embedded)
        return IngestReport(
            normalized.source_id,
            normalized.source,
            len(embedded),
            chunks_replaced=replaced,
            elapsed_seconds=time.perf_counter() - started,
        )

    def ingest_documents(
        self, documents: Sequence[IngestDocument | Mapping[str, object] | ExtractedDocument]
    ) -> list[IngestReport]:
        return [self.ingest_document(document) for document in documents]

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        filter_metadata: Mapping[str, object] | None = None,
    ) -> list[RetrievalHit]:
        if not query_text.strip():
            raise ValueError("Query must not be empty")
        limit = top_k or self.top_k
        query_vector = self.embedder.embed_query(query_text)
        candidates = self.retrieval.retrieve(
            query_text,
            query_vector.tolist(),
            self.settings.retrieval.candidate_k,
            filter_metadata,
        )
        if self.reranker is not None:
            candidates = self.reranker.rerank(
                query_text, candidates, self.settings.retrieval.candidate_k
            )
        return self._pack_hits(candidates, limit, self.settings.retrieval.context_token_budget)

    def _pack_hits(
        self, candidates: Sequence[RetrievalHit], limit: int, token_budget: int
    ) -> list[RetrievalHit]:
        packed: list[RetrievalHit] = []
        used = 0
        for candidate in candidates:
            if len(packed) >= limit:
                break
            overhead = self.tokenizer.count(
                f"[{len(packed) + 1}] source={candidate.source}; page={candidate.page}\n"
            )
            available = token_budget - used - overhead
            if available <= 0:
                break
            tokens = candidate.token_count or self.tokenizer.count(candidate.document)
            if tokens > available:
                truncated = self.tokenizer.truncate(candidate.document, available)
                if not truncated.strip():
                    break
                candidate = replace(
                    candidate,
                    document=truncated,
                    token_count=self.tokenizer.count(truncated),
                    metadata={**candidate.metadata, "context_truncated": True},
                )
            packed.append(candidate)
            used += overhead + candidate.token_count
        return packed

    def generate_context(self, results: Sequence[RetrievalHit]) -> str:
        return OllamaGenerator.build_context(results)

    def generate_answer(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: Mapping[str, object] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
    ) -> AnswerResult:
        if self.llm_generator is None:
            raise ValueError("LLM generator not initialized. Use RAGPipeline(use_llm=True).")
        retrieval_started = time.perf_counter()
        results = self.query(query, top_k=top_k, filter_metadata=filter_metadata)
        retrieval_seconds = time.perf_counter() - retrieval_started
        if not results:
            return AnswerResult(
                "I don't have enough evidence to answer.",
                [],
                "",
                retrieval_seconds=retrieval_seconds,
                warnings=("no_retrieval_evidence",),
            )
        context = self.generate_context(results)
        generation_started = time.perf_counter()
        answer = self.llm_generator.generate_with_results(
            query, results, system_prompt=system_prompt, stream=stream
        )
        generation_seconds = time.perf_counter() - generation_started
        warnings = (
            () if _citations_valid(answer, len(results)) else ("missing_or_invalid_citations",)
        )
        return AnswerResult(
            answer,
            list(results),
            context,
            retrieval_seconds,
            generation_seconds,
            self.tokenizer.count(context) + self.tokenizer.count(query),
            self.tokenizer.count(answer),
            warnings,
        )

    def get_stats(self) -> dict[str, object]:
        manifest = self.vector_store.load_index_manifest()
        return {
            "ready": True,
            "total_chunks": self.vector_store.get_collection_count(),
            "embedding_model": self.embedding_spec.model_name,
            "embedding_dimension": self.embedding_spec.dimension,
            "chunking_strategy": self.text_chunker.strategy.name,
            "retrieval_strategy": self.retrieval_stack_name,
            "reranker": self.reranker.name if self.reranker is not None else None,
            "context_token_budget": self.settings.retrieval.context_token_budget,
            "collection_name": self.vector_store.collection_name,
            "persist_directory": str(self.vector_store.persist_directory),
            "model_loaded": self.embedder.model_loaded,
            "llm_enabled": self.llm_generator is not None,
            "index_compatibility_key": manifest.compatibility_key if manifest else None,
            "recommendation_status": self.recommendation.status,
            "recommendation_run_ids": self.recommendation.benchmark_run_ids,
            "recommendation_authoritative": self.recommendation.authoritative,
        }

    def reset(self) -> None:
        self.vector_store.reset_collection()
        self.vector_store.ensure_manifest(self.index_manifest)

    @staticmethod
    def _normalize_document(
        document: IngestDocument | Mapping[str, object] | ExtractedDocument,
    ) -> IngestDocument:
        if isinstance(document, IngestDocument):
            return document
        if isinstance(document, ExtractedDocument):
            return normalize_ingest_document(
                {
                    "text": document.text,
                    "source": document.source_name,
                    "file_path": document.source_path,
                    "format_type": document.source_kind.value,
                    "metadata": {
                        **document.metadata,
                        "source_checksum": document.source_checksum,
                        "extraction_profile": document.profile.fingerprint,
                    },
                }
            )
        return normalize_ingest_document(document)


def _citations_valid(answer: str, context_count: int) -> bool:
    import re

    citations = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    refusal = "don't have enough evidence" in answer.lower()
    return refusal or bool(citations) and all(1 <= value <= context_count for value in citations)


def _locked_ollama_digest(path: Path, model_name: str) -> str:
    if not path.is_file():
        raise RAGConfigurationError(
            f"Missing Ollama model lock {path}; run `edumind benchmark prepare ollama-models`"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RAGConfigurationError(f"Cannot read Ollama model lock {path}: {exc}") from exc
    models = payload.get("models", {})
    digest = models.get(model_name) if isinstance(models, Mapping) else None
    if not digest:
        raise RAGConfigurationError(f"Ollama model lock has no digest for {model_name}")
    return str(digest)
