"""Application orchestration between OCR and RAG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from edumind.ocr.core.pipeline import DataIngestionPipeline
from edumind.rag.rag_pipeline import RAGPipeline
from edumind.rag.serializers import serialize_answer_result, serialize_query_results

logger = logging.getLogger(__name__)


class OCRRAGOrchestrator:
    """Coordinate OCR extraction and RAG ingestion/querying."""

    def __init__(self, use_llm: bool = True, rag_config_path: str | None = None) -> None:
        self.ocr_pipeline = DataIngestionPipeline()
        self.rag_pipeline = RAGPipeline(config_path=rag_config_path, use_llm=use_llm)

    def process_file(
        self,
        file_path: str | Path,
        ingest_to_rag: bool = True,
        clean_text: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run OCR for one file and optionally ingest the result into RAG."""
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
            "rag_source_id": None,
        }

        if ingest_to_rag and ocr_result.success and ocr_result.text:
            try:
                report = self.rag_pipeline.ingest_document(
                    {
                        "text": ocr_result.text,
                        "source": self._resolve_source_name(ocr_result.file_path),
                        "format_type": ocr_result.format_type,
                        "file_path": ocr_result.file_path,
                        "metadata": dict(ocr_result.metadata),
                    }
                )
                result["rag_ingested"] = True
                result["rag_chunks"] = report.chunks_created
                result["rag_source_id"] = report.source_id
            except Exception as exc:
                logger.error("RAG ingestion failed: %s", exc)
                result["rag_error"] = str(exc)

        return result

    def process_batch(
        self,
        file_paths: list[str | Path],
        ingest_to_rag: bool = True,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Process many files through OCR and optional RAG ingest."""
        return [
            self.process_file(file_path, ingest_to_rag=ingest_to_rag, **kwargs)
            for file_path in file_paths
        ]

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        generate_answer: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Query the RAG subsystem and return a JSON-friendly payload."""
        if generate_answer and self.rag_pipeline.llm_generator is not None:
            answer_result = self.rag_pipeline.generate_answer(
                query=query_text,
                top_k=top_k,
                **kwargs,
            )
            return cast(dict[str, Any], _model_dump(serialize_answer_result(answer_result)))

        results = self.rag_pipeline.query(query_text=query_text, top_k=top_k, **kwargs)
        return cast(
            dict[str, Any],
            _model_dump(serialize_query_results(query_text, results)),
        )

    def get_stats(self) -> dict[str, Any]:
        """Return combined OCR and RAG runtime stats."""
        return {
            "rag": self.rag_pipeline.get_stats(),
            "ocr_extractors": list(self.ocr_pipeline.extractors.keys()),
        }

    def reset_rag(self) -> None:
        """Clear the RAG index while keeping the OCR runtime ready."""
        self.rag_pipeline.reset()

    def _resolve_source_name(self, file_path: str) -> str:
        """Build a stable source display name for downstream RAG ingest."""
        if file_path:
            return Path(file_path).name
        return "uploaded-document"


def _model_dump(model: object) -> dict[str, object]:
    """Return a dict from either a Pydantic v1 or v2 model."""
    if hasattr(model, "model_dump"):
        return getattr(model, "model_dump")()
    if hasattr(model, "dict"):
        return getattr(model, "dict")()
    raise TypeError(f"Unsupported model type: {type(model).__name__}")
