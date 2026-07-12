"""End-to-end OCR-to-RAG pipeline."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from edumind.common.config import load_yaml_config
from edumind.common.paths import resolve_config_path

from .embedder import Embedder
from .llm_generator import OllamaGenerator
from .ocr_processor import OCRProcessor
from .text_chunker import TextChunker
from .types import AnswerResult, IngestDocument, IngestReport, RAGConfig, RetrievalHit
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

try:
    import mlflow as _mlflow
except ImportError:  # pragma: no cover - optional runtime dependency
    mlflow: Any | None = None
else:
    mlflow = _mlflow


class RAGPipeline:
    """Process text into a vector store and answer questions with local LLMs."""

    def __init__(self, config_path: str | None = None, use_llm: bool = False) -> None:
        self.config_path = resolve_config_path(config_path)
        raw_config = load_yaml_config(self.config_path)
        self.config = RAGConfig.from_mapping(raw_config)
        self._configure_mlflow()

        self.ocr_processor = OCRProcessor()
        self.embedder = Embedder(settings=self.config.embedding)
        self.text_chunker = TextChunker(settings=self.config.chunking, embedder=self.embedder)
        self.vector_store = VectorStore(settings=self.config.vector_store)

        self.top_k = self.config.rag.top_k
        self.score_threshold = self.config.rag.score_threshold
        self.llm_generator = OllamaGenerator(settings=self.config.llm) if use_llm else None

    def ingest_document(self, document: IngestDocument | Mapping[str, object]) -> IngestReport:
        """Normalize, chunk, embed, and upsert one document into the retrieval store."""
        normalized_document = self._normalize_document(document)
        chunks = self.text_chunker.chunk_document(normalized_document)
        embedded_chunks = self.embedder.embed_chunks(chunks)
        self.vector_store.upsert_chunks(embedded_chunks)
        self._log_active_mlflow_ingest(normalized_document, len(embedded_chunks))
        return IngestReport(
            source_id=normalized_document.source_id,
            source=normalized_document.source,
            chunks_created=len(embedded_chunks),
        )

    def ingest_documents(
        self,
        documents: Sequence[IngestDocument | Mapping[str, object]],
    ) -> list[IngestReport]:
        """Ingest many documents and return one report per document."""
        return [self.ingest_document(document) for document in documents]

    def ingest_from_json(self, json_path: str | Path) -> list[IngestReport]:
        """Load OCR JSON and ingest all normalized documents."""
        documents = self.ocr_processor.load_from_json(json_path)
        return self.ingest_documents(documents)

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        filter_metadata: Mapping[str, object] | None = None,
    ) -> list[RetrievalHit]:
        """Run retrieval and enforce the configured score threshold."""
        limit = top_k or self.top_k
        results = self.vector_store.query_by_text(
            query_text,
            self.embedder,
            top_k=limit,
            filter_metadata=filter_metadata,
        )
        return [result for result in results if result.score >= self.score_threshold]

    def generate_context(self, results: Sequence[RetrievalHit]) -> str:
        """Build display context from one already-computed result set."""
        return "\n\n".join(
            f"[Document {index}]\n{result.document}\nSource: {result.source}, Page: {result.page}"
            for index, result in enumerate(results, start=1)
        )

    def generate_answer(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: Mapping[str, object] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
    ) -> AnswerResult:
        """Retrieve once and reuse that result set for answer, context, and sources."""
        if self.llm_generator is None:
            raise ValueError("LLM generator not initialized. Use RAGPipeline(use_llm=True).")

        results = self.query(query, top_k=top_k, filter_metadata=filter_metadata)
        if not results:
            return AnswerResult(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                context="",
            )

        context = self.generate_context(results)
        answer = self.llm_generator.generate_with_results(
            query=query,
            results=results,
            system_prompt=system_prompt,
            stream=stream,
        )
        self._log_active_mlflow_query(query, results, answer)
        return AnswerResult(answer=answer, sources=list(results), context=context)

    def get_stats(self) -> dict[str, object]:
        """Return lightweight RAG runtime statistics."""
        total_chunks = self.vector_store.get_collection_count()
        return {
            "total_chunks": total_chunks,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.embedding_dim,
            "chunk_size": self.text_chunker.chunk_size,
            "chunk_overlap": self.text_chunker.chunk_overlap,
            "collection_name": self.vector_store.collection_name,
            "persist_directory": str(self.vector_store.persist_directory),
            "model_loaded": self.embedder.model_loaded,
            "llm_enabled": self.llm_generator is not None,
        }

    def reset(self) -> None:
        """Clear the dense and lexical retrieval stores."""
        self.vector_store.reset_collection()

    def _normalize_document(
        self,
        document: IngestDocument | Mapping[str, object],
    ) -> IngestDocument:
        """Normalize one supported ingest input into an IngestDocument."""
        if isinstance(document, IngestDocument):
            return document
        return self.ocr_processor.normalize_document(document)

    def _configure_mlflow(self) -> None:
        """Configure the MLflow tracking target without forcing logging side effects."""
        if mlflow is None:
            return
        tracking_uri = os.getenv("EDUMIND_MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_name = os.getenv("EDUMIND_MLFLOW_EXPERIMENT", "EduMind-AI-RAG")
        mlflow.set_experiment(experiment_name)

    def _log_active_mlflow_ingest(self, document: IngestDocument, chunk_count: int) -> None:
        """Log ingest metrics only when an MLflow run is already active."""
        if mlflow is None or mlflow.active_run() is None:
            return
        mlflow.log_params({"rag_source": document.source, "rag_source_id": document.source_id})
        mlflow.log_metric("chunks_created", chunk_count)
        mlflow.log_metric("source_text_length", len(document.text))

    def _log_active_mlflow_query(
        self,
        query: str,
        results: Sequence[RetrievalHit],
        answer: str,
    ) -> None:
        """Log query metrics only when an MLflow run is already active."""
        if mlflow is None or mlflow.active_run() is None:
            return
        mlflow.log_params({"query": query, "top_k": len(results)})
        mlflow.log_metric("retrieved_documents", len(results))
        mlflow.log_text(answer, "generated_answer.txt")
