"""Typed application orchestration over extraction and RAG."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from edumind.common.config import load_settings
from edumind.extraction import ExtractedDocument, ExtractionPipeline, ExtractionProfile
from edumind.rag.rag_pipeline import RAGPipeline
from edumind.rag.types import AnswerResult, IngestReport, RetrievalHit


class PipelineStage(str, Enum):
    CLASSIFYING = "classifying"
    EXTRACTING = "extracting"
    INDEXING = "indexing"
    RETRIEVING = "retrieving"
    GENERATING = "generating"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ProgressEvent:
    stage: PipelineStage
    message: str
    progress: float


@dataclass(frozen=True)
class DocumentProcessResult:
    extraction: ExtractedDocument
    ingest: IngestReport | None
    timings: Mapping[str, float]
    warnings: tuple[str, ...] = ()

@dataclass(frozen=True)
class PipelineQueryResult:
    query: str
    hits: tuple[RetrievalHit, ...]
    answer: AnswerResult | None
    timings: Mapping[str, float]
    warnings: tuple[str, ...] = ()


ProgressCallback = Callable[[ProgressEvent], None]


class EduMindPipeline:
    def __init__(
        self,
        *,
        use_llm: bool = True,
        extraction: ExtractionPipeline | None = None,
        rag: RAGPipeline | None = None,
        config_path: str | None = None,
    ) -> None:
        settings = load_settings(config_path) if extraction is None or rag is None else None
        self.extraction = extraction or ExtractionPipeline(settings)
        self.rag = rag or RAGPipeline(settings=settings, use_llm=use_llm)

    def process_file(
        self,
        file_path: str | Path,
        *,
        ingest: bool = True,
        source_name: str | None = None,
        profile: ExtractionProfile | None = None,
        progress: ProgressCallback | None = None,
    ) -> DocumentProcessResult:
        started = time.perf_counter()
        self._emit(progress, PipelineStage.CLASSIFYING, "Classifying source", 0.05)
        extraction_started = time.perf_counter()
        self._emit(progress, PipelineStage.EXTRACTING, "Extracting source content", 0.15)
        document = self.extraction.extract(file_path, profile=profile)
        if source_name is not None:
            logical_name = Path(source_name).name
            document = replace(document, source_name=logical_name, source_path=logical_name)
        extraction_seconds = time.perf_counter() - extraction_started
        ingest_report = None
        indexing_seconds = 0.0
        if ingest:
            self._emit(progress, PipelineStage.INDEXING, "Indexing normalized content", 0.65)
            indexing_started = time.perf_counter()
            ingest_report = self.rag.ingest_document(document)
            indexing_seconds = time.perf_counter() - indexing_started
        self._emit(progress, PipelineStage.COMPLETE, "Document ready", 1.0)
        return DocumentProcessResult(
            document,
            ingest_report,
            {
                "extraction_seconds": extraction_seconds,
                "indexing_seconds": indexing_seconds,
                "total_seconds": time.perf_counter() - started,
            },
            tuple(warning.message for warning in document.warnings),
        )

    def query(
        self,
        query: str,
        *,
        top_k: int = 5,
        generate_answer: bool = True,
        filters: Mapping[str, object] | None = None,
        progress: ProgressCallback | None = None,
    ) -> PipelineQueryResult:
        started = time.perf_counter()
        self._emit(progress, PipelineStage.RETRIEVING, "Retrieving evidence", 0.2)
        if generate_answer:
            self._emit(progress, PipelineStage.GENERATING, "Generating cited answer", 0.55)
            answer = self.rag.generate_answer(query, top_k=top_k, filter_metadata=filters)
            hits = tuple(answer.sources)
            timings = {
                "retrieval_seconds": answer.retrieval_seconds,
                "generation_seconds": answer.generation_seconds,
                "total_seconds": time.perf_counter() - started,
            }
            warnings = answer.warnings
        else:
            retrieval_started = time.perf_counter()
            hits = tuple(self.rag.query(query, top_k=top_k, filter_metadata=filters))
            answer = None
            timings = {
                "retrieval_seconds": time.perf_counter() - retrieval_started,
                "generation_seconds": 0.0,
                "total_seconds": time.perf_counter() - started,
            }
            warnings = ()
        self._emit(progress, PipelineStage.COMPLETE, "Query complete", 1.0)
        return PipelineQueryResult(query, hits, answer, timings, warnings)

    def readiness(self) -> dict[str, object]:
        stats = self.rag.get_stats()
        return {
            "ready": True,
            "extraction_sources": self.extraction.supported_sources(),
            "rag": stats,
            "generation_ready": self.rag.llm_generator.health_check()
            if self.rag.llm_generator
            else False,
        }

    def reset_index(self) -> None:
        self.rag.reset()

    @staticmethod
    def _emit(
        callback: ProgressCallback | None, stage: PipelineStage, message: str, value: float
    ) -> None:
        if callback:
            callback(ProgressEvent(stage, message, value))
