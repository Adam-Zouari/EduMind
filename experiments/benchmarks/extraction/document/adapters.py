"""Document parsers that remain experimental until benchmark promotion."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from typing import Any

from edumind.extraction.contracts import ExtractedDocument, ExtractionRequest, SourceKind
from edumind.extraction.errors import ExtractionBackendError, MissingDependencyError
from edumind.extraction.extractors.document import docling_pages, required_directory
from edumind.extraction.structured import build_markdown_document


class ExperimentalDocumentExtractor:
    supported_kinds = frozenset({SourceKind.IMAGE, SourceKind.PDF})

    def __init__(self, engine: str, revision: str) -> None:
        if engine not in {"docling-vlm-granite-258m", "paddleocr-vl-1.6"}:
            raise ValueError(f"Unknown experimental document parser: {engine}")
        self.engine = engine
        self.name = engine
        self.revision = revision
        self._runtimes: dict[str, Any] = {}

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        started = time.perf_counter()
        try:
            if self.engine == "docling-vlm-granite-258m":
                pages, canonical = self._docling_vlm(request, kind)
            else:
                pages, canonical = self._paddle_vl(request)
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ExtractionBackendError(
                f"Document extraction failed with {self.engine}", detail=str(exc)
            ) from exc
        return build_markdown_document(
            request,
            kind,
            request.profile,
            pages,
            metadata={
                "engine": self.engine,
                "engine_revision": request.profile.engine_revision,
                "canonical_document": canonical,
            },
            seconds=time.perf_counter() - started,
        )

    def _docling_vlm(self, request: ExtractionRequest, kind: SourceKind):
        try:
            from docling.datamodel import vlm_model_specs
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import VlmPipelineOptions
            from docling.document_converter import (
                DocumentConverter,
                ImageFormatOption,
                PdfFormatOption,
            )
            from docling.pipeline.vlm_pipeline import VlmPipeline
        except (ImportError, ModuleNotFoundError) as exc:
            raise MissingDependencyError("Docling 2.117.0 VLM dependencies are required") from exc
        model_path = required_directory(request, "model_path", "Granite Docling")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.model_copy(deep=True)
        options.repo_id = str(model_path)
        options.revision = request.profile.engine_revision
        options.load_in_8bit = False
        pipeline = VlmPipelineOptions(
            vlm_options=options,
            artifacts_path=model_path.parent,
            accelerator_options=AcceleratorOptions(device=request.profile.device),
        )
        key = request.profile.fingerprint
        if key not in self._runtimes:
            self._runtimes[key] = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline, pipeline_options=pipeline
                    ),
                    InputFormat.IMAGE: ImageFormatOption(
                        pipeline_cls=VlmPipeline, pipeline_options=pipeline
                    ),
                }
            )
        document = self._runtimes[key].convert(str(request.source_path)).document
        return docling_pages(document, kind), document.export_to_dict()

    def _paddle_vl(self, request: ExtractionRequest):
        model_path = required_directory(request, "model_path", "PaddleOCR-VL-1.6")
        paddle_cache = required_directory(request, "paddle_cache_path", "PaddleOCR-VL")
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddle_cache)
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        try:
            from paddleocr import PaddleOCRVL
        except (ImportError, ModuleNotFoundError) as exc:
            raise MissingDependencyError("PaddleX OCR extras are required") from exc
        key = request.profile.fingerprint
        if key not in self._runtimes:
            self._runtimes[key] = PaddleOCRVL(
                pipeline_version="v1.6",
                engine="transformers",
                vl_rec_model_dir=str(model_path),
                device="gpu" if request.profile and request.profile.device == "cuda" else "cpu",
            )
        pages: list[str] = []
        canonical: list[object] = []
        for result in self._runtimes[key].predict(str(request.source_path)):
            markdown = getattr(result, "markdown", {})
            if callable(markdown):
                markdown = markdown()
            pages.append(_markdown_text(markdown))
            payload = getattr(result, "json", None)
            canonical.append(payload() if callable(payload) else payload or {})
        if not pages:
            raise RuntimeError("PaddleOCR-VL-1.6 produced no pages")
        return pages, {"pages": canonical, "model_path": str(model_path)}


def _markdown_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        for key in ("markdown_texts", "markdown_text", "markdown"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    raise RuntimeError("Parser result did not expose Markdown text")

