"""Document parsers that remain experimental until benchmark promotion."""

from __future__ import annotations

import os
import time
from collections.abc import Mapping
from importlib.metadata import version
from typing import Any

from edumind.extraction.contracts import ExtractedDocument, ExtractionRequest, SourceKind
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
            elements = self._paddle_vl(request)
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
            seconds=time.perf_counter() - started,
        )

    def _docling_vlm(self, request: ExtractionRequest):
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
        return document

    def _paddle_vl(self, request: ExtractionRequest):
        model_path = required_directory(request, "model_path", "PaddleOCR-VL-1.6")
        paddle_cache = required_directory(request, "paddle_cache_path", "PaddleOCR-VL")
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddle_cache)
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        try:
            import paddle
            from paddleocr import PaddleOCRVL
        except (ImportError, ModuleNotFoundError) as exc:
            raise MissingDependencyError("PaddleX OCR extras are required") from exc
        paddleocr_version = version("paddleocr")
        if paddleocr_version != "3.7.0" or paddle.__version__ != "3.3.1":
            raise MissingDependencyError(
                "PaddleOCR 3.7.0 with PaddlePaddle 3.3.1 is required; found "
                f"PaddleOCR {paddleocr_version} with PaddlePaddle {paddle.__version__}"
            )
        key = request.profile.fingerprint
        if key not in self._runtimes:
            self._runtimes[key] = PaddleOCRVL(
                pipeline_version="v1.6",
                vl_rec_backend="native",
                vl_rec_model_dir=str(model_path),
                device="gpu" if request.profile and request.profile.device == "cuda" else "cpu",
            )
        elements: list[dict[str, object]] = []
        for result in self._runtimes[key].predict(str(request.source_path)):
            payload = getattr(result, "json", None)
            payload = payload() if callable(payload) else payload or {}
            blocks = _paddle_blocks(payload)
            if not blocks:
                raise RuntimeError("PaddleOCR-VL result contains no native parsing blocks")
            elements.extend(blocks)
        if not elements:
            raise RuntimeError("PaddleOCR-VL-1.6 produced no pages")
        for order, element in enumerate(elements):
            element["order"] = order
        return elements


def _paddle_blocks(
    value: object,
) -> list[dict[str, object]]:
    """Read native Paddle blocks; do not infer structure from rendered Markdown."""

    if not isinstance(value, Mapping):
        return []
    if isinstance(value.get("res"), Mapping):
        value = value["res"]
    page_number = int(value["page_index"]) + 1
    raw_blocks = value.get("parsing_res_list", [])
    if not isinstance(raw_blocks, list):
        return []
    page_size = _page_size(value)
    result: list[dict[str, object]] = []
    for index, raw in enumerate(raw_blocks):
        if not isinstance(raw, Mapping):
            continue
        label = str(raw.get("block_label", "text")).casefold()
        text = str(raw.get("block_content", ""))
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
            structured["html"] = text
        elif kind == "formula":
            structured["latex"] = text
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
    return result


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
