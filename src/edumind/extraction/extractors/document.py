"""Provisional production Docling Standard document parser."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..contracts import ExtractedDocument, ExtractionRequest, SourceKind
from ..errors import ExtractionBackendError, MissingDependencyError
from ..structured import build_markdown_document


class DoclingExtractor:
    supported_kinds = frozenset({SourceKind.IMAGE, SourceKind.PDF, SourceKind.DOCX})
    engine = "docling-standard"
    name = engine

    def __init__(self, revision: str) -> None:
        self.revision = revision
        self._runtimes: dict[str, Any] = {}

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        started = time.perf_counter()
        try:
            pages, canonical = self._convert(request, kind)
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ExtractionBackendError(
                "Document extraction failed with Docling Standard", detail=str(exc)
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

    def _convert(
        self, request: ExtractionRequest, kind: SourceKind
    ) -> tuple[list[str], Mapping[str, object]]:
        try:
            from docling.datamodel.accelerator_options import AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                EasyOcrOptions,
                OcrMode,
                PdfPipelineOptions,
                RapidOcrOptions,
                TableFormerMode,
                TesseractCliOcrOptions,
            )
            from docling.document_converter import (
                DocumentConverter,
                ImageFormatOption,
                PdfFormatOption,
            )
        except (ImportError, ModuleNotFoundError) as exc:
            raise MissingDependencyError("Docling 2.117.0 is required") from exc
        artifacts = required_directory(request, "model_path", "Docling Standard")
        options = request.options
        ocr_engine = str(options.get("ocr_engine", "rapidocr"))
        ocr_mode = (
            OcrMode.FULL_PAGE
            if str(options.get("ocr_mode", "pdf_aware_layout_regions")) == "full_page"
            else OcrMode.PDF_AWARE_LAYOUT_REGIONS
        )
        ocr_options: Any
        if ocr_engine == "rapidocr":
            ocr_options = RapidOcrOptions(lang=["english"], mode=ocr_mode)
        elif ocr_engine == "tesseract":
            ocr_options = TesseractCliOcrOptions(lang=["eng"], mode=ocr_mode)
        elif ocr_engine == "easyocr":
            ocr_options = EasyOcrOptions(
                lang=["en"],
                mode=ocr_mode,
                use_gpu=request.profile.device == "cuda",
                download_enabled=False,
            )
        else:
            raise ValueError(f"Unsupported Docling OCR engine: {ocr_engine}")
        pipeline = PdfPipelineOptions(
            artifacts_path=artifacts,
            accelerator_options=AcceleratorOptions(device=request.profile.device),
            images_scale=3.0,
            do_ocr=True,
            ocr_options=ocr_options,
            do_table_structure=True,
            do_code_enrichment=False,
            do_formula_enrichment=bool(options.get("formula_enrichment", False)),
        )
        pipeline.table_structure_options.mode = (
            TableFormerMode.ACCURATE
            if str(options.get("table_mode", "fast")) == "accurate"
            else TableFormerMode.FAST
        )
        pipeline.table_structure_options.do_cell_matching = True
        key = request.profile.fingerprint
        if key not in self._runtimes:
            formats: dict[Any, Any] = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline),
            }
            self._runtimes[key] = DocumentConverter(format_options=formats)
        document = self._runtimes[key].convert(str(request.source_path)).document
        return docling_pages(document, kind), document.export_to_dict()


def docling_pages(document: Any, kind: SourceKind) -> list[str]:
    if kind is SourceKind.PDF and getattr(document, "pages", None):
        pages = [
            str(document.export_to_markdown(page_no=int(page_number))).strip()
            for page_number in sorted(document.pages)
        ]
        if any(pages):
            return pages
    return [str(document.export_to_markdown()).strip()]


def required_directory(request: ExtractionRequest, key: str, label: str) -> Path:
    path = Path(str(request.options.get(key, ""))).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} is not prepared locally; run `python "
            "experiments/benchmarks/prepare.py app-models`."
        )
    return path

