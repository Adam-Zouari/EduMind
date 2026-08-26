"""Production indexing, token-budget retrieval, and grounded generation."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from edumind.common.config import Settings, load_settings
from edumind.common.models import load_model_lock, require_model
from edumind.extraction import ExtractedDocument

from .contracts import GenerationProfile, IndexManifest, embedding_spec
from .document_processor import normalize_ingest_document
from .embedder import Embedder
from .errors import RAGConfigurationError
from .llm_generator import HuggingFaceGenerator
from .text_chunker import TextChunker, TokenChunkingStrategy
from .tokenizers import LazyHuggingFaceOffsetTokenizer, OffsetTokenizer, TiktokenOffsetTokenizer
from .types import AnswerResult, IngestDocument, IngestReport, RetrievalHit, VectorStoreSettings
from .vector_store import VectorStore


class RAGPipeline:
    """The sole production RAG path used by the direct Streamlit pipeline."""

    def __init__(
        self,
        config_path: str | None = None,
        use_llm: bool = False,
        *,
        settings: Settings | None = None,
        tokenizer: OffsetTokenizer | None = None,
        embedder: Embedder | None = None,
        vector_store: VectorStore | None = None,
        generator: HuggingFaceGenerator | None = None,
    ) -> None:
        self.settings = settings or load_settings(config_path)
        model_lock = load_model_lock(self.settings.models.lock_path)
        embedding_snapshot = require_model(
            model_lock, self.settings.embedding.model_name
        )
        self.embedding_spec = embedding_spec(
            self.settings.embedding.model_name,
            document_device=self.settings.embedding.indexing_device,
            query_device=self.settings.embedding.query_device,
            revision=embedding_snapshot.revision,
            local_path=str(embedding_snapshot.path),
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
                self.embedding_spec.local_path,
            )
            if self.settings.chunking.tokenizer == "embedding"
            else TiktokenOffsetTokenizer(self.settings.chunking.tokenizer)
        )
        self.embedder = embedder or Embedder(
            spec=self.embedding_spec, batch_size=self.settings.embedding.batch_size
        )
        strategy = TokenChunkingStrategy(
            self.tokenizer,
            self.settings.chunking.chunk_size,
            self.settings.chunking.chunk_overlap,
            f"token-{self.settings.chunking.chunk_size}-{self.settings.chunking.chunk_overlap}",
        )
        self.text_chunker = TextChunker(strategy)
        self.vector_store = vector_store or VectorStore(
            VectorStoreSettings(
                self.settings.vector.collection_name,
                self.settings.vector.endpoint,
                self.settings.vector.distance_metric,
            )
        )
        self.index_manifest = IndexManifest(
            schema_version=1,
            embedding_contract=self.embedding_spec.fingerprint,
            chunking_contract=strategy.fingerprint,
            backend=self.settings.vector.backend,
            collection_name=self.settings.vector.collection_name,
        )
        self.vector_store.ensure_manifest(self.index_manifest)
        self.retrieval_stack_name = "dense"
        self.llm_generator = generator
        if self.llm_generator is None and use_llm:
            generation_snapshot = require_model(
                model_lock, self.settings.generation.model_name
            )
            generation_profile = GenerationProfile(
                self.settings.generation.model_name,
                generation_snapshot.revision,
                str(generation_snapshot.path),
                self.settings.generation.device,
                self.settings.generation.dtype,
                self.settings.generation.reasoning,
                self.settings.generation.temperature,
                self.settings.generation.seed,
                self.settings.generation.context_tokens,
                self.settings.generation.maximum_answer_tokens,
            )
            self.llm_generator = HuggingFaceGenerator(generation_profile)
        self.top_k = self.settings.retrieval.top_k

    def ingest_document(
        self, document: IngestDocument | Mapping[str, object] | ExtractedDocument
    ) -> IngestReport:
        started = time.perf_counter()
        normalized = self._normalize_document(document)
        chunks = self.text_chunker.chunk_document(normalized)
        if not chunks:
            raise ValueError("Cannot index a document that produced no text chunks")
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
        candidates = self.vector_store.query_dense(
            query_vector.tolist(),
            top_k=self.settings.retrieval.candidate_k,
            filter_metadata=filter_metadata,
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
        return HuggingFaceGenerator.build_context(results)

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
        embedding_ready = bool(self.embedding_spec.local_path) and Path(
            str(self.embedding_spec.local_path)
        ).is_dir()
        vector_ready = self.vector_store.health_check()
        total_chunks: int | None = None
        if vector_ready:
            try:
                total_chunks = self.vector_store.get_collection_count()
            except Exception:
                vector_ready = False
        generation_ready = (
            self.llm_generator.health_check() if self.llm_generator is not None else False
        )
        problems = []
        if not embedding_ready:
            problems.append("The pinned embedding snapshot is unavailable")
        if not vector_ready:
            problems.append(f"Chroma is unavailable at {self.vector_store.endpoint}")
        if manifest is None:
            problems.append("The vector index manifest is unavailable")
        if self.llm_generator is not None and not generation_ready:
            problems.append("The pinned generator snapshot is unavailable")
        return {
            "ready": not problems,
            "problems": problems,
            "total_chunks": total_chunks,
            "embedding_model": self.embedding_spec.model_name,
            "embedding_dimension": self.embedding_spec.dimension,
            "embedding_ready": embedding_ready,
            "chunking_strategy": self.text_chunker.strategy.name,
            "retrieval_strategy": self.retrieval_stack_name,
            "context_token_budget": self.settings.retrieval.context_token_budget,
            "collection_name": self.vector_store.collection_name,
            "vector_endpoint": self.vector_store.endpoint,
            "model_loaded": self.embedder.model_loaded,
            "llm_enabled": self.llm_generator is not None,
            "generation_ready": generation_ready,
            "vector_ready": vector_ready,
            "index_compatibility_key": manifest.compatibility_key if manifest else None,
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
