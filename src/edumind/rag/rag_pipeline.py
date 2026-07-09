"""End-to-end OCR-to-RAG pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from edumind.common.config import load_yaml_config
from edumind.common.paths import resolve_config_path

from .embedder import Embedder
from .llm_generator import OllamaGenerator
from .ocr_processor import OCRProcessor
from .text_chunker import TextChunker
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

try:
    import mlflow
except Exception:  # pragma: no cover - optional runtime dependency
    mlflow = None


class RAGPipeline:
    """Process text into a vector store and answer questions with local LLMs."""

    def __init__(self, config_path: str | None = None, use_llm: bool = False):
        self.config_path = str(resolve_config_path(config_path))
        self.config = load_yaml_config(self.config_path)
        self._configure_mlflow()

        self.ocr_processor = OCRProcessor()
        self.text_chunker = TextChunker(self.config_path)
        self.embedder = Embedder(self.config_path)
        self.vector_store = VectorStore(self.config_path)

        self.rag_config = self.config["rag"]
        self.top_k = self.rag_config["top_k"]
        self.score_threshold = self.rag_config["score_threshold"]

        self.llm_generator = None
        if use_llm:
            llm_config = self.config.get("llm", {})
            self.llm_generator = OllamaGenerator(
                model_name=llm_config.get("model_name", "qwen3:1.7b"),
                base_url=llm_config.get("base_url", "http://localhost:11434"),
                temperature=llm_config.get("temperature", 0.7),
                max_tokens=llm_config.get("max_tokens", 2048),
            )

        self._log_initialization()

    def _configure_mlflow(self) -> None:
        if mlflow is None:
            return
        tracking_uri = os.getenv("EDUMIND_MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        experiment_name = os.getenv("EDUMIND_MLFLOW_EXPERIMENT", "EduMind-AI-RAG")
        mlflow.set_experiment(experiment_name)

    def _log_initialization(self) -> None:
        if mlflow is None:
            return
        try:
            with mlflow.start_run(run_name="pipeline_initialization"):
                mlflow.log_params(
                    {
                        "embedding_model": self.embedder.model_name,
                        "chunk_size": self.text_chunker.chunk_size,
                        "top_k": self.top_k,
                        "score_threshold": self.score_threshold,
                        "llm_model": self.llm_generator.model_name if self.llm_generator else "None",
                    }
                )
        except Exception as exc:
            logger.warning(f"Failed to log initialization to MLflow: {exc}")

    def ingest_document(self, document: dict[str, Any]) -> int:
        text = document.get("text", "")
        metadata = {key: value for key, value in document.items() if key != "text"}
        source = metadata.get("source", "unknown")

        if mlflow is None:
            return self._ingest_document_chunks(text, metadata)

        with mlflow.start_run(run_name=f"ingest_{Path(source).name}", nested=True):
            return self._ingest_document_chunks(text, metadata)

    def _ingest_document_chunks(self, text: str, metadata: dict[str, Any]) -> int:
        chunks = self.text_chunker.chunk_text(text, metadata)
        if mlflow is not None:
            mlflow.log_metric("chunks_created", len(chunks))
            mlflow.log_metric("source_text_length", len(text))
        chunks_with_embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.add_documents(chunks_with_embeddings)
        return len(chunks)

    def ingest_documents(self, documents: list[dict[str, Any]]) -> int:
        return sum(self.ingest_document(document) for document in documents)

    def ingest_from_json(self, json_path: str) -> int:
        return self.ingest_documents(self.ocr_processor.load_from_json(json_path))

    def query(self, query_text: str, top_k: int | None = None, filter_metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        limit = top_k or self.top_k
        results = self.vector_store.query_by_text(query_text, self.embedder, top_k=limit, filter_metadata=filter_metadata)
        if results and "distance" in results[0]:
            results = [result for result in results if result["distance"] <= (1 - self.score_threshold)]
        return results

    def generate_context(self, query_text: str, top_k: int | None = None) -> str:
        results = self.query(query_text, top_k)
        return "\n".join(f"[Document {index + 1}]\n{result['document']}\n" for index, result in enumerate(results))

    def generate_answer(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        if self.llm_generator is None:
            raise ValueError("LLM generator not initialized. Use RAGPipeline(use_llm=True).")

        limit = top_k or self.top_k
        results = self.query(query, limit, filter_metadata)
        if not results:
            return {
                "answer": "I couldn't find any relevant information to answer your question.",
                "sources": [],
                "context": "",
            }

        answer = self.llm_generator.generate_with_results(
            query=query,
            results=results,
            system_prompt=system_prompt,
            stream=stream,
        )

        if mlflow is not None:
            try:
                with mlflow.start_run(run_name="query_generation", nested=True):
                    mlflow.log_params({"query": query, "top_k": limit})
                    mlflow.log_metric("retrieved_documents", len(results))
                    mlflow.log_text(answer, "generated_answer.txt")
            except Exception as exc:
                logger.warning(f"Failed to log query generation to MLflow: {exc}")

        return {
            "answer": answer,
            "sources": [
                {
                    "source": result.get("metadata", {}).get("source", "Unknown"),
                    "page": result.get("metadata", {}).get("page", "N/A"),
                    "similarity": f"{(1 - result.get('distance', 1)) * 100:.1f}%",
                    "text": result.get("document", ""),
                }
                for result in results
            ],
            "context": self.generate_context(query, limit),
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_documents": self.vector_store.get_collection_count(),
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.embedding_dim,
            "chunk_size": self.text_chunker.chunk_size,
            "collection_name": self.vector_store.collection_name,
            "persist_directory": str(self.vector_store.persist_directory),
        }

    def reset(self) -> None:
        self.vector_store.reset_collection()
