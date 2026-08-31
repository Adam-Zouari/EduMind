"""Provisional production Docling Standard document parser."""

from __future__ import annotations

import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from ..contracts import ExtractedDocument, ExtractionRequest, SourceKind
from ..errors import ExtractionBackendError, MissingDependencyError
from ..structured import build_docling_document

DOCLING_VERSION = "2.117.0"


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
            document = self._convert(request, kind)
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ExtractionBackendError(
                "Document extraction failed with Docling Standard", detail=str(exc)
            ) from exc
        return build_docling_document(
            request,
            kind,
            request.profile,
            document,
            metadata={
                "engine": self.engine,
                "engine_revision": request.profile.engine_revision,
            },
            seconds=time.perf_counter() - started,
        )

    def _convert(
        self, request: ExtractionRequest, kind: SourceKind
    ) -> Any:
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
            raise MissingDependencyError(f"Docling {DOCLING_VERSION} is required") from exc
        installed = version("docling")
        if installed != DOCLING_VERSION:
            raise MissingDependencyError(
                f"Docling {DOCLING_VERSION} is required; found {installed}"
            )
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
        return self._runtimes[key].convert(str(request.source_path)).document


def required_directory(request: ExtractionRequest, key: str, label: str) -> Path:
    path = Path(str(request.options.get(key, ""))).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} is not prepared locally; run `python "
            "experiments/benchmarks/prepare.py app-models`."
        )
    return path
