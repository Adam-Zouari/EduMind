"""Document parsers that remain experimental until benchmark promotion."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from html.parser import HTMLParser
from importlib.metadata import version
from typing import Any

from edumind.extraction.contracts import (
    ExtractedDocument,
    ExtractionRequest,
    ExtractionWarning,
    SourceKind,
)
from edumind.extraction.errors import ExtractionBackendError, MissingDependencyError
from edumind.extraction.extractors.document import DOCLING_VERSION, required_directory
from edumind.extraction.structured import build_docling_document, build_structured_document


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
        warnings: tuple[ExtractionWarning, ...] = ()
        try:
            if self.engine == "docling-vlm-granite-258m":
                document = self._docling_vlm(request)
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
            elements, warnings = self._paddle_vl(request)
        except MissingDependencyError:
            raise
        except Exception as exc:
            raise ExtractionBackendError(
                f"Document extraction failed with {self.engine}", detail=str(exc)
            ) from exc
        return build_structured_document(
            request,
            kind,
            request.profile,
            elements,
            metadata={
                "engine": self.engine,
                "engine_revision": request.profile.engine_revision,
            },
            warnings=warnings,
            seconds=time.perf_counter() - started,
        )

    def _docling_vlm(self, request: ExtractionRequest):
        converter = self._docling_converter(request)
        return converter.convert(str(request.source_path)).document

    def initialize_image_pipeline(self, request: ExtractionRequest) -> None:
        """Initialize exactly the visual parser lifecycle measured by video."""

        if self.engine == "docling-vlm-granite-258m":
            from docling.datamodel.base_models import InputFormat

            converter = self._docling_converter(request)
            converter.initialize_pipeline(InputFormat.IMAGE)
            return
        self._paddle_runtime(request)

    def _docling_converter(self, request: ExtractionRequest):
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
            raise MissingDependencyError(
                f"Docling {DOCLING_VERSION} VLM dependencies are required"
            ) from exc
        installed = version("docling")
        if installed != DOCLING_VERSION:
            raise MissingDependencyError(
                f"Docling {DOCLING_VERSION} is required; found {installed}"
            )
        model_path = required_directory(request, "model_path", "Granite Docling")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        options = vlm_model_specs.GRANITEDOCLING_TRANSFORMERS.model_copy(deep=True)
        options.repo_id = str(model_path)
        options.revision = request.profile.engine_revision
        options.load_in_8bit = bool(request.options["load_in_8bit"])
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
        return self._runtimes[key]

    def _paddle_vl(
        self, request: ExtractionRequest
    ) -> tuple[list[dict[str, object]], tuple[ExtractionWarning, ...]]:
        runtime = self._paddle_runtime(request)
        elements: list[dict[str, object]] = []
        warnings: list[ExtractionWarning] = []
        for result in runtime.predict(str(request.source_path)):
            payload = getattr(result, "json", None)
            payload = payload() if callable(payload) else payload or {}
            blocks = _paddle_blocks(payload, warnings=warnings)
            if not blocks:
                raise RuntimeError("PaddleOCR-VL result contains no native parsing blocks")
            elements.extend(blocks)
        if not elements:
            raise RuntimeError("PaddleOCR-VL-1.6 produced no pages")
        for order, element in enumerate(elements):
            element["order"] = order
        return elements, tuple(warnings)

    def _paddle_runtime(self, request: ExtractionRequest):
        model_path = required_directory(request, "model_path", "PaddleOCR-VL-1.6")
        paddle_cache = required_directory(request, "paddle_cache_path", "PaddleOCR-VL")
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddle_cache)
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        try:
            # On Windows, Paddle adds DLL search paths whose shared-library names
            # conflict with PyTorch. PaddleX imports ModelScope (and therefore
            # PyTorch), so load PyTorch's DLLs before importing Paddle.
            import torch
            import paddle
            from paddleocr import PaddleOCRVL
        except (ImportError, ModuleNotFoundError) as exc:
            raise MissingDependencyError("PaddleX OCR extras are required") from exc
        _ = torch.__version__
        paddleocr_version = version("paddleocr")
        if paddleocr_version != "3.7.0" or paddle.__version__ != "3.3.1":
            raise MissingDependencyError(
                "PaddleOCR 3.7.0 with PaddlePaddle 3.3.1 is required; found "
                f"PaddleOCR {paddleocr_version} with PaddlePaddle {paddle.__version__}"
            )
        key = request.profile.fingerprint
        if key not in self._runtimes:
            self._runtimes[key] = PaddleOCRVL(
                pipeline_version=str(request.options["pipeline_version"]),
                vl_rec_backend=str(request.options["recognition_backend"]),
                vl_rec_model_dir=str(model_path),
                device="gpu" if request.profile and request.profile.device == "cuda" else "cpu",
            )
        return self._runtimes[key]


def _paddle_blocks(
    value: object,
    *,
    warnings: list[ExtractionWarning] | None = None,
) -> list[dict[str, object]]:
    """Read native Paddle blocks; do not infer structure from rendered Markdown."""

    if not isinstance(value, Mapping):
        return []
    if isinstance(value.get("res"), Mapping):
        value = value["res"]
    raw_page_index = value.get("page_index")
    page_number = 1 if raw_page_index is None else int(raw_page_index) + 1
    raw_blocks = value.get("parsing_res_list", [])
    if not isinstance(raw_blocks, list):
        return []
    page_size = _page_size(value)
    result: list[dict[str, object]] = []
    for index, raw in enumerate(raw_blocks):
        if not isinstance(raw, Mapping):
            if warnings is not None:
                warnings.append(
                    ExtractionWarning(
                        "paddle_block_conversion_failed",
                        f"Skipped Paddle block {index}: block is not an object",
                        segment_index=len(result),
                    )
                )
            continue
        try:
            label = str(raw.get("block_label", "text")).casefold()
            native_content = str(raw.get("block_content", ""))
            text = native_content
            kind = {
                "title": "title",
                "heading": "heading",
                "section_header": "heading",
                "table": "table",
                "formula": "formula",
                "equation": "formula",
                "figure": "figure",
                "image": "figure",
                "caption": "caption",
                "code": "code",
                "list_item": "list_item",
            }.get(label, "text")
            structured: dict[str, object] = {}
            if kind == "table":
                rows, cells = _parse_table_html(native_content)
                if not cells:
                    # Keep the candidate alive and make the degraded conversion explicit.
                    fallback = _plain_html_text(native_content)
                    rows = [[fallback]] if fallback else [[]]
                    cells = (
                        [{"text": fallback, "row": 0, "column": 0,
                          "row_span": 1, "column_span": 1}]
                        if fallback
                        else []
                    )
                    if warnings is not None:
                        warnings.append(
                            ExtractionWarning(
                                "paddle_table_html_malformed",
                                f"Paddle table block {index} had no parseable cells",
                                segment_index=len(result),
                            )
                        )
                text = "\n".join("\t".join(cell for cell in row) for row in rows)
                structured.update({"rows": rows, "cells": cells, "html": native_content})
            elif kind == "formula":
                # Formula text is an exact LaTeX payload, not prose.
                structured["latex"] = native_content
            result.append(
                {
                    "text": text,
                    "element_id": str(raw.get("block_id", f"page-{page_number}-{index}")),
                    "page_number": page_number,
                    "bounding_box": _paddle_box(raw.get("block_bbox"), page_size),
                    "kind": kind,
                    "structured_content": structured,
                    "metadata": {
                        "label": label,
                        "block_order": raw.get("block_order"),
                        "group_id": raw.get("group_id"),
                    },
                }
            )
        except (TypeError, ValueError) as exc:
            if warnings is not None:
                warnings.append(
                    ExtractionWarning(
                        "paddle_block_conversion_failed",
                        f"Skipped Paddle block {index}: {exc}",
                        segment_index=len(result),
                    )
                )
    return result


class _TableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.cells: list[dict[str, object]] = []
        self._row: list[str] | None = None
        self._cell_parts: list[str] | None = None
        self._cell_attrs: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.casefold() == "tr":
            self._row = []
        elif tag.casefold() in {"td", "th"}:
            if self._row is None:
                self._row = []
            self._cell_parts = []
            self._cell_attrs = {str(key).casefold(): str(value) for key, value in attrs}

    def handle_data(self, data: str) -> None:
        if self._cell_parts is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.casefold()
        if lowered in {"td", "th"} and self._cell_parts is not None:
            value = " ".join("".join(self._cell_parts).split())
            assert self._row is not None
            column = len(self._row)
            row = len(self.rows)
            self._row.append(value)
            self.cells.append(
                {
                    "text": value,
                    "row": row,
                    "column": column,
                    "row_span": _positive_span(self._cell_attrs.get("rowspan")),
                    "column_span": _positive_span(self._cell_attrs.get("colspan")),
                    "header": lowered == "th",
                }
            )
            self._cell_parts = None
            self._cell_attrs = {}
        elif lowered == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def close(self) -> None:
        super().close()
        if self._cell_parts is not None:
            self.handle_endtag("td")
        if self._row is not None:
            self.rows.append(self._row)
            self._row = None


def _parse_table_html(value: str) -> tuple[list[list[str]], list[dict[str, object]]]:
    parser = _TableHTMLParser()
    parser.feed(value)
    parser.close()
    return parser.rows, parser.cells


def _plain_html_text(value: str) -> str:
    class _TextParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []

        def handle_data(self, data: str) -> None:
            self.parts.append(data)

    parser = _TextParser()
    parser.feed(value)
    parser.close()
    return " ".join(" ".join(parser.parts).split())


def _positive_span(value: str | None) -> int:
    try:
        return max(1, int(value or 1))
    except ValueError:
        return 1


def _paddle_box(
    value: object, page_size: tuple[float, float] | None
) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    values = [float(item) for item in value]
    if all(0.0 <= item <= 1.0 for item in values):
        return values
    if page_size and page_size[0] > 0 and page_size[1] > 0:
        width, height = page_size
        normalized = [values[0] / width, values[1] / height, values[2] / width, values[3] / height]
        if all(0.0 <= item <= 1.0 for item in normalized):
            return normalized
    return None


def _page_size(value: Mapping[str, object]) -> tuple[float, float] | None:
    if value.get("width") and value.get("height"):
        return float(value["width"]), float(value["height"])
    return None
