"""Application orchestration between OCR and RAG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from edumind.ocr.core.pipeline import DataIngestionPipeline
from edumind.rag.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class OCRRAGOrchestrator:
    """Coordinate OCR extraction and RAG ingestion/querying."""

    def __init__(self, use_llm: bool = True, rag_config_path: str | None = None):
        self.ocr_pipeline = DataIngestionPipeline()
        self.rag_pipeline = RAGPipeline(config_path=rag_config_path, use_llm=use_llm)

    def process_file(
        self,
        file_path: str | Path,
        ingest_to_rag: bool = True,
        clean_text: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ocr_result = self.ocr_pipeline.process_file(
            file_path=file_path,
            clean_text=clean_text,
            **kwargs,
        )

        result: dict[str, Any] = {
            "ocr_success": ocr_result.success,
            "ocr_error": ocr_result.error,
            "text": ocr_result.text,
            "metadata": ocr_result.metadata,
            "file_path": ocr_result.file_path,
            "format_type": ocr_result.format_type,
            "extraction_time": ocr_result.extraction_time,
            "rag_ingested": False,
            "rag_chunks": 0,
        }

        if ingest_to_rag and ocr_result.success and ocr_result.text:
            try:
                num_chunks = self.rag_pipeline.ingest_document(ocr_result.to_dict())
                result["rag_ingested"] = True
                result["rag_chunks"] = num_chunks
            except Exception as exc:
                logger.error(f"RAG ingestion failed: {exc}")
                result["rag_error"] = str(exc)

        return result

    def process_batch(self, file_paths: list[str | Path], ingest_to_rag: bool = True, **kwargs: Any) -> list[dict[str, Any]]:
        return [
            self.process_file(file_path, ingest_to_rag=ingest_to_rag, **kwargs)
            for file_path in file_paths
        ]

    def query(self, query_text: str, top_k: int = 5, generate_answer: bool = True, **kwargs: Any) -> dict[str, Any]:
        if generate_answer and self.rag_pipeline.llm_generator is not None:
            return self.rag_pipeline.generate_answer(query=query_text, top_k=top_k, **kwargs)
        return {
            "query": query_text,
            "results": self.rag_pipeline.query(query_text=query_text, top_k=top_k, **kwargs),
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "rag": self.rag_pipeline.get_stats(),
            "ocr_extractors": list(self.ocr_pipeline.extractors.keys()),
        }

    def reset_rag(self) -> None:
        self.rag_pipeline.reset()
