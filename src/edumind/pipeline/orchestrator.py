"""Application orchestration between OCR and RAG."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict, cast

from edumind.ocr.core.base_extractor import ExtractionResult
from edumind.ocr.core.pipeline import DataIngestionPipeline
from edumind.rag.rag_pipeline import RAGPipeline
from edumind.rag.serializers import serialize_answer_result, serialize_query_results

logger = logging.getLogger(__name__)


class ProcessedDocumentPayload(TypedDict):
    """JSON-friendly OCR plus RAG orchestration result."""

    ocr_success: bool
    ocr_error: str | None
    text: str
    metadata: dict[str, object]
    file_path: str
    format_type: str
    extraction_time: float
    rag_ingested: bool
    rag_chunks: int
    rag_source_id: str | None
    rag_error: str | None


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
    ) -> ProcessedDocumentPayload:
        """Run OCR for one file and optionally ingest the result into RAG."""
        ocr_result = self.ocr_pipeline.process_file(
            file_path=file_path,
            clean_text=clean_text,
            **kwargs,
        )
        return self._build_processed_payload(ocr_result, ingest_to_rag=ingest_to_rag)

    def process_batch(
        self,
        file_paths: list[str | Path],
        ingest_to_rag: bool = True,
        **kwargs: Any,
    ) -> list[ProcessedDocumentPayload]:
        """Process many files through OCR and optional RAG ingest."""
        ocr_results = self.ocr_pipeline.process_batch(file_paths, **kwargs)
        return [
            self._build_processed_payload(result, ingest_to_rag=ingest_to_rag)
            for result in ocr_results
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
        ocr_formats = sorted(self.ocr_pipeline.extractors.keys())
        return {
            "rag": self.rag_pipeline.get_stats(),
            "ocr_extractors": ocr_formats,
            "ocr_formats": ocr_formats,
        }

    def reset_rag(self) -> None:
        """Clear the RAG index while keeping the OCR runtime ready."""
        self.rag_pipeline.reset()

    def reset_database(self) -> None:
        """Backward-compatible alias for resetting the local RAG index."""
        self.reset_rag()

    def _build_processed_payload(
        self,
        ocr_result: ExtractionResult,
        *,
        ingest_to_rag: bool,
    ) -> ProcessedDocumentPayload:
        """Convert one OCR result into the combined orchestrator payload."""
        result: ProcessedDocumentPayload = {
            "ocr_success": ocr_result.success,
            "ocr_error": ocr_result.error,
            "text": ocr_result.text,
            "metadata": dict(ocr_result.metadata),
            "file_path": ocr_result.file_path,
            "format_type": ocr_result.format_type,
            "extraction_time": ocr_result.extraction_time,
            "rag_ingested": False,
            "rag_chunks": 0,
            "rag_source_id": None,
            "rag_error": None,
        }
        if not (ingest_to_rag and ocr_result.success and ocr_result.text.strip()):
            return result

        try:
            report = self.rag_pipeline.ingest_document(self._build_ingest_payload(ocr_result))
        except Exception as exc:
            logger.error(
                "RAG ingestion failed for %s: %s",
                ocr_result.file_path or "<unknown>",
                exc,
            )
            result["rag_error"] = str(exc)
            return result

        result["rag_ingested"] = True
        result["rag_chunks"] = report.chunks_created
        result["rag_source_id"] = report.source_id
        return result

    def _build_ingest_payload(self, ocr_result: ExtractionResult) -> dict[str, object]:
        """Build the normalized nested ingest contract for the RAG pipeline."""
        return {
            "text": ocr_result.text,
            "source": self._resolve_source_name(ocr_result.file_path),
            "format_type": ocr_result.format_type,
            "file_path": ocr_result.file_path,
            "metadata": dict(ocr_result.metadata),
        }

    def _resolve_source_name(self, file_path: str | None) -> str:
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
