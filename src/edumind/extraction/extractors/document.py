"""Provisional production Docling Standard document parser."""

from __future__ import annotations

import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

from edumind.common.onnx_placement import cuda_shape_session, verified_cuda_graph

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

    def extract(
        self, request: ExtractionRequest, kind: SourceKind
    ) -> ExtractedDocument:
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

    def _convert(self, request: ExtractionRequest, kind: SourceKind) -> Any:
        converter, _ = self._converter(request, kind)
        return converter.convert(str(request.source_path)).document

    def initialize_image_pipeline(self, request: ExtractionRequest) -> None:
        """Initialize the image pipeline without converting a benchmark frame."""

        converter, input_format = self._converter(request, SourceKind.IMAGE)
        converter.initialize_pipeline(input_format)

    def _converter(self, request: ExtractionRequest, kind: SourceKind):
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
            raise MissingDependencyError(
                f"Docling {DOCLING_VERSION} is required"
            ) from exc
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
            ocr_options = RapidOcrOptions(
                lang=[str(options.get("language", "english"))],
                mode=ocr_mode,
                # RapidOCR 3.x reads acceleration from EngineConfig, not the
                # per-stage use_cuda flags emitted by this pinned Docling release.
                rapidocr_params={
                    "EngineConfig.onnxruntime.use_cuda": request.profile.device
                    == "cuda",
                },
            )
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
            images_scale=float(options.get("image_scale", 3.0)),
            do_ocr=bool(options.get("do_ocr", True)),
            ocr_options=ocr_options,
            do_table_structure=bool(options.get("do_table_structure", True)),
            do_code_enrichment=bool(options.get("do_code_enrichment", False)),
            do_formula_enrichment=bool(options.get("formula_enrichment", False)),
        )
        pipeline.table_structure_options.mode = (
            TableFormerMode.ACCURATE
            if str(options.get("table_mode", "fast")) == "accurate"
            else TableFormerMode.FAST
        )
        pipeline.table_structure_options.do_cell_matching = bool(
            options.get("do_cell_matching", True)
        )
        key = request.profile.fingerprint
        if key not in self._runtimes:
            formats: dict[Any, Any] = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline),
            }
            self._runtimes[key] = DocumentConverter(format_options=formats)
        input_format = (
            InputFormat.IMAGE if kind is SourceKind.IMAGE else InputFormat.PDF
        )
        converter = self._runtimes[key]
        if (
            request.profile.device == "cuda"
            and ocr_engine == "rapidocr"
            and bool(options.get("do_ocr", True))
            and kind is not SourceKind.DOCX
        ):
            _require_cuda_ocr(converter, input_format)
        return converter, input_format


def _require_cuda_ocr(converter, input_format) -> None:
    """Require CUDA computation, allowing only verified CPU shape bookkeeping."""
    import onnxruntime as ort
    import torch

    _ = torch.__version__  # load the matching CUDA/cuDNN DLLs before ONNX sessions
    ort.preload_dlls()
    converter.initialize_pipeline(input_format)
    readers = []
    for pipeline in converter.initialized_pipelines.values():
        if not hasattr(pipeline, "ocr_model"):
            continue  # A converter may also cache a native DOCX pipeline.
        reader = getattr(pipeline.ocr_model, "reader", None)
        if reader is None:
            raise RuntimeError("Initialized RapidOCR pipeline exposes no reader")
        readers.append(reader)
    if not readers:
        raise RuntimeError("Initialized RapidOCR pipeline exposes no reader")
    for reader in readers:
        for name in ("text_det", "text_cls", "text_rec"):
            runtime = getattr(reader, name).session
            session = runtime.session
            if verified_cuda_graph(session) is not None:
                continue
            # Do not replace or cache a session until graph verification succeeds.
            runtime.session = cuda_shape_session(session)


def required_directory(request: ExtractionRequest, key: str, label: str) -> Path:
    path = Path(str(request.options.get(key, ""))).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} is not prepared locally; run `python "
            "experiments/benchmarks/prepare.py app-models`."
        )
    return path
