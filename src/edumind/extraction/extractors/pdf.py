"""Native and page-level hybrid PDF extraction."""

from __future__ import annotations

import tempfile
import time
from dataclasses import replace
from pathlib import Path

from ..contracts import ExtractedDocument, ExtractionRequest, ExtractionWarning, SourceKind
from ..errors import ExtractionBackendError, MissingDependencyError
from .base import build_document
from .image import ImageExtractor


class PDFExtractor:
    supported_kinds = frozenset({SourceKind.PDF})

    def __init__(self, engine: str, revision: str = "unpinned") -> None:
        self.engine = engine
        self.name = engine
        self.revision = revision

    def extract(self, request: ExtractionRequest, kind: SourceKind) -> ExtractedDocument:
        if request.profile is None:
            raise ValueError("Resolved extraction profile is required")
        started = time.perf_counter()
        try:
            if self.engine == "pypdf":
                pages = self._pypdf(request.source_path)
            elif self.engine == "pdfplumber":
                pages = self._pdfplumber(request.source_path)
            elif self.engine == "docling-pdf":
                pages = self._docling(request.source_path)
            elif self.engine == "hybrid-pdf":
                pages = self._hybrid(request)
            elif self.engine == "ocr-pdf":
                pages = self._ocr_pages(request, force=True)
            else:
                raise ValueError(f"Unknown PDF engine: {self.engine}")
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ExtractionBackendError(
                f"PDF extraction failed with {self.engine}", detail=str(exc)
            ) from exc
        empty_pages = sum(not page.strip() for page in pages)
        warnings = []
        if empty_pages:
            warnings.append(
                ExtractionWarning("empty_pages", f"{empty_pages} PDF pages produced no text")
            )
        return build_document(
            request,
            kind,
            request.profile,
            [page.strip() for page in pages],
            pages=list(range(1, len(pages) + 1)),
            metadata={"engine": self.engine, "page_count": len(pages)},
            warnings=warnings,
            seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _pypdf(path: Path) -> list[str]:
        try:
            from pypdf import PdfReader
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("pypdf is required; install .[extraction]") from exc
        return [page.extract_text() or "" for page in PdfReader(path).pages]

    @staticmethod
    def _pdfplumber(path: Path) -> list[str]:
        try:
            import pdfplumber
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("pdfplumber is required; install .[extraction]") from exc
        with pdfplumber.open(path) as pdf:
            return [page.extract_text(layout=True) or "" for page in pdf.pages]

    @staticmethod
    def _docling(path: Path) -> list[str]:
        try:
            from docling.document_converter import DocumentConverter
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("Docling is required for this PDF candidate") from exc
        document = DocumentConverter().convert(str(path)).document
        return [document.export_to_markdown()]

    def _hybrid(self, request: ExtractionRequest) -> list[str]:
        pages = self._pypdf(request.source_path)
        raw_threshold = request.options.get("native_page_minimum_characters", 40)
        threshold = int(raw_threshold) if isinstance(raw_threshold, (str, int, float)) else 40
        if all(len(page.strip()) >= threshold for page in pages):
            return pages
        return self._ocr_pages(request, force=False, native_pages=pages, threshold=threshold)

    def _ocr_pages(
        self,
        request: ExtractionRequest,
        *,
        force: bool,
        native_pages: list[str] | None = None,
        threshold: int = 40,
    ) -> list[str]:
        pages = native_pages if native_pages is not None else self._pypdf(request.source_path)
        try:
            import fitz
        except ModuleNotFoundError as exc:
            raise MissingDependencyError("PyMuPDF is required for hybrid PDF routing") from exc
        ocr_engine = str(request.options.get("image_engine", "tesseract-5"))
        ocr = ImageExtractor(ocr_engine, "5" if ocr_engine == "tesseract-5" else "unpinned")
        profile = request.profile
        if profile is None:
            raise ValueError("Resolved extraction profile is required")
        with (
            fitz.open(request.source_path) as pdf,
            tempfile.TemporaryDirectory(prefix="edumind-pdf-") as temp,
        ):
            for index, page in enumerate(pdf):
                if not force and len(pages[index].strip()) >= threshold:
                    continue
                image_path = Path(temp) / f"page-{index + 1}.png"
                page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(image_path)
                image_request = ExtractionRequest.from_path(
                    image_path,
                    source_kind=SourceKind.IMAGE,
                    profile=replace(
                        profile,
                        engine=ocr_engine,
                        engine_revision=str(request.options.get("image_revision", "unpinned")),
                        preprocessing=str(
                            request.options.get("image_preprocessing", profile.preprocessing)
                        ),
                        routing="pdf-page-fallback",
                    ),
                )
                pages[index] = ocr.extract(image_request, SourceKind.IMAGE).text
        return pages
