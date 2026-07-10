"""Main pipeline for content extraction."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from threading import Semaphore
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from ..config import OCR_USE_GPU, PRESERVE_LATEX
from ..extractors.docx_extractor import DOCXExtractor
from ..extractors.ocr_extractor import OCRExtractor
from ..extractors.pdf_extractor import PDFExtractor
from ..extractors.web_extractor import WebExtractor
from ..processors.form_recognizer import FormRecognizer
from ..processors.layout_analyzer import LayoutAnalyzer
from ..processors.math_extractor import MathExtractor
from ..processors.text_cleaner import TextCleaner
from ..utils.file_handler import FileHandler
from ..utils.logger import get_logger
from .base_extractor import ExtractionResult
from .format_detector import FormatDetector

logger = get_logger(__name__)


class DataIngestionPipeline:
    """Main pipeline for extracting content from various file formats."""

    _THREAD_SAFE_FORMATS = {"pdf", "docx", "image", "web"}
    _SEQUENTIAL_FORMATS = {"audio", "video"}

    def __init__(self):
        self.format_detector = FormatDetector()
        self.text_cleaner = TextCleaner()
        self.math_extractor = MathExtractor()
        self.layout_analyzer = LayoutAnalyzer()
        self.form_recognizer = FormRecognizer()
        self.extractors = {
            "pdf": PDFExtractor(),
            "docx": DOCXExtractor(),
            "image": OCRExtractor(),
            "web": WebExtractor(),
            "audio": None,
            "video": None,
        }

        logger.info("Data Ingestion Pipeline initialized")

    def process_file(
        self,
        file_path: str | Path,
        clean_text: bool = True,
        preserve_latex: bool = PRESERVE_LATEX,
        pdf_ocr_mode: str = "auto",
        include_layout: bool = False,
        include_form_fields: bool = False,
        profile: bool = False,
        include_file_hash: bool = True,
        languages: list[str] | None = None,
        strict_format_detection: bool = False,
        **kwargs: Any,
    ) -> ExtractionResult:
        """Process a file and extract its content."""
        file_path = Path(file_path)
        logger.info(f"Processing file: {file_path}")
        total_start = time.perf_counter()

        if not FileHandler.validate_file(file_path):
            return self._build_error_result(
                file_path=file_path,
                error="File validation failed",
                total_start=total_start,
                profile=profile,
            )

        try:
            detect_start = time.perf_counter()
            format_info = self.format_detector.detect(file_path, strict=strict_format_detection)
            format_detection_time = time.perf_counter() - detect_start
            return self._process_detected_file(
                file_path=file_path,
                format_info=format_info,
                clean_text=clean_text,
                preserve_latex=preserve_latex,
                pdf_ocr_mode=pdf_ocr_mode,
                include_layout=include_layout,
                include_form_fields=include_form_fields,
                profile=profile,
                include_file_hash=include_file_hash,
                languages=languages,
                total_start=total_start,
                format_detection_time=format_detection_time,
                **kwargs,
            )
        except Exception as exc:
            logger.error(f"Pipeline processing failed: {exc}", exc_info=True)
            return self._build_error_result(
                file_path=file_path,
                error=str(exc),
                total_start=total_start,
                profile=profile,
            )

    def process_batch(
        self,
        file_paths: list[str | Path],
        parallel: bool = True,
        max_workers: int = 4,
        batch_strategy: str = "auto",
        **kwargs: Any,
    ) -> list[ExtractionResult]:
        """Process multiple files with optional strategy-aware batching."""
        if not parallel:
            batch_strategy = "sequential"

        if batch_strategy not in {"auto", "threads", "sequential"}:
            raise ValueError(f"Unsupported batch_strategy: {batch_strategy}")

        logger.info(
            f"Processing batch of {len(file_paths)} files "
            f"(strategy={batch_strategy}, parallel={parallel})"
        )

        if len(file_paths) <= 1 or batch_strategy == "sequential":
            return [self.process_file(file_path, **kwargs) for file_path in file_paths]

        if batch_strategy == "threads":
            return self._process_batch_threads(file_paths, max_workers=max_workers, **kwargs)

        return self._process_batch_auto(file_paths, max_workers=max_workers, **kwargs)

    def _process_batch_threads(
        self,
        file_paths: list[str | Path],
        *,
        max_workers: int,
        **kwargs: Any,
    ) -> list[ExtractionResult]:
        """Process a batch using a plain thread pool."""
        results: list[ExtractionResult | None] = [None] * len(file_paths)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.process_file, file_path, **kwargs): index
                for index, file_path in enumerate(file_paths)
            }
            for future in tqdm(as_completed(future_to_index), total=len(file_paths), desc="Processing files"):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    logger.error(f"Error processing file {file_paths[index]}: {exc}")
                    results[index] = self._build_error_result(
                        file_path=Path(file_paths[index]),
                        error=str(exc),
                        total_start=time.perf_counter(),
                        profile=bool(kwargs.get("profile", False)),
                    )

        return [result for result in results if result is not None]

    def _process_batch_auto(
        self,
        file_paths: list[str | Path],
        *,
        max_workers: int,
        **kwargs: Any,
    ) -> list[ExtractionResult]:
        """Process thread-safe formats concurrently and media formats sequentially."""
        strict = bool(kwargs.get("strict_format_detection", False))
        profile = bool(kwargs.get("profile", False))
        results: list[ExtractionResult | None] = [None] * len(file_paths)
        detected_items: list[dict[str, Any]] = []

        for index, raw_path in enumerate(file_paths):
            file_path = Path(raw_path)
            total_start = time.perf_counter()
            if not FileHandler.validate_file(file_path):
                results[index] = self._build_error_result(
                    file_path=file_path,
                    error="File validation failed",
                    total_start=total_start,
                    profile=profile,
                )
                continue

            try:
                detect_start = time.perf_counter()
                format_info = self.format_detector.detect(file_path, strict=strict)
                detected_items.append(
                    {
                        "index": index,
                        "file_path": file_path,
                        "format_info": format_info,
                        "format_detection_time": time.perf_counter() - detect_start,
                        "total_start": total_start,
                    }
                )
            except Exception as exc:
                logger.error(f"Format detection failed for {file_path}: {exc}")
                results[index] = self._build_error_result(
                    file_path=file_path,
                    error=str(exc),
                    total_start=total_start,
                    profile=profile,
                )

        image_limiter = Semaphore(min(max_workers, 2)) if not OCR_USE_GPU else None
        thread_items = [
            item for item in detected_items if item["format_info"]["format_type"] in self._THREAD_SAFE_FORMATS
        ]
        sequential_items = [
            item for item in detected_items if item["format_info"]["format_type"] not in self._THREAD_SAFE_FORMATS
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {}
            for item in thread_items:
                future = executor.submit(
                    self._process_detected_file_with_optional_limiter,
                    file_path=item["file_path"],
                    format_info=item["format_info"],
                    format_detection_time=item["format_detection_time"],
                    total_start=item["total_start"],
                    image_limiter=image_limiter if item["format_info"]["format_type"] == "image" else None,
                    **kwargs,
                )
                future_to_index[future] = item["index"]

            for item in sequential_items:
                results[item["index"]] = self._process_detected_file(
                    file_path=item["file_path"],
                    format_info=item["format_info"],
                    format_detection_time=item["format_detection_time"],
                    total_start=item["total_start"],
                    **kwargs,
                )

            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Processing files"):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    logger.error(f"Error processing file {file_paths[index]}: {exc}")
                    results[index] = self._build_error_result(
                        file_path=Path(file_paths[index]),
                        error=str(exc),
                        total_start=time.perf_counter(),
                        profile=profile,
                    )

        return [result for result in results if result is not None]

    def _process_detected_file_with_optional_limiter(
        self,
        *,
        file_path: Path,
        format_info: dict[str, Any],
        format_detection_time: float,
        total_start: float,
        image_limiter: Semaphore | None,
        **kwargs: Any,
    ) -> ExtractionResult:
        """Process a file while optionally limiting concurrent image OCR."""
        if image_limiter is None:
            return self._process_detected_file(
                file_path=file_path,
                format_info=format_info,
                format_detection_time=format_detection_time,
                total_start=total_start,
                **kwargs,
            )

        with image_limiter:
            return self._process_detected_file(
                file_path=file_path,
                format_info=format_info,
                format_detection_time=format_detection_time,
                total_start=total_start,
                **kwargs,
            )

    def _process_detected_file(
        self,
        *,
        file_path: Path,
        format_info: dict[str, Any],
        clean_text: bool = True,
        preserve_latex: bool = PRESERVE_LATEX,
        pdf_ocr_mode: str = "auto",
        include_layout: bool = False,
        include_form_fields: bool = False,
        profile: bool = False,
        include_file_hash: bool = True,
        languages: list[str] | None = None,
        total_start: float | None = None,
        format_detection_time: float | None = None,
        **kwargs: Any,
    ) -> ExtractionResult:
        """Process a file once format detection has already been completed."""
        total_start = time.perf_counter() if total_start is None else total_start
        timings: dict[str, float] = {}
        if format_detection_time is not None:
            timings["format_detection"] = format_detection_time

        format_type = str(format_info["format_type"])
        logger.info(f"Detected format: {format_type}")

        extractor = self._get_extractor(format_type, languages=languages)
        if extractor is None:
            return self._build_error_result(
                file_path=file_path,
                error=f"No extractor available for format: {format_type}",
                total_start=total_start,
                profile=profile,
            )

        extract_kwargs = dict(kwargs)
        if languages is not None:
            extract_kwargs["languages"] = languages
        if format_type == "pdf":
            extract_kwargs["pdf_ocr_mode"] = pdf_ocr_mode
            extract_kwargs["include_layout"] = include_layout
        if format_type == "image" and include_layout:
            extract_kwargs["return_ocr_data"] = True

        extraction_start = time.perf_counter()
        result = extractor.extract(file_path, **extract_kwargs)
        timings["extraction"] = time.perf_counter() - extraction_start

        if result.success and clean_text and result.text:
            clean_start = time.perf_counter()
            if preserve_latex:
                preserved_text, math_dict = self.math_extractor.preserve_math(result.text)
                cleaned = self.text_cleaner.clean(preserved_text, preserve_latex=True)
                result.text = self.math_extractor.restore_math(cleaned, math_dict)
            else:
                result.text = self.text_cleaner.clean(result.text, preserve_latex=False)

            result.metadata["math_expressions"] = self.math_extractor.extract_latex(result.text)
            timings["cleaning"] = time.perf_counter() - clean_start

        if result.success and format_type in {"image", "pdf"}:
            self._attach_optional_metadata(
                result,
                format_type=format_type,
                include_layout=include_layout,
                include_form_fields=include_form_fields,
            )

        result.metadata["format_info"] = format_info
        result.metadata["file_size"] = FileHandler.get_file_size(file_path)

        if include_file_hash:
            hash_start = time.perf_counter()
            result.metadata["file_hash"] = FileHandler.get_file_hash(file_path)
            timings["hashing"] = time.perf_counter() - hash_start

        self._cleanup_internal_metadata(result)
        timings["total_processing"] = time.perf_counter() - total_start
        if profile:
            result.metadata["performance"] = timings

        logger.info(f"Processing completed in {timings['total_processing']:.2f}s")
        return result

    def _attach_optional_metadata(
        self,
        result: ExtractionResult,
        *,
        format_type: str,
        include_layout: bool,
        include_form_fields: bool,
    ) -> None:
        """Attach optional structured OCR metadata without rewriting the text."""
        if include_form_fields:
            fields = self.form_recognizer.extract_form_fields(result.text)
            result.metadata["structured_fields"] = self.form_recognizer.to_structured_dict(fields)

        if not include_layout:
            return

        layout_blocks: list[dict[str, Any]] = []
        if format_type == "image":
            layout_blocks.extend(self._build_layout_blocks_from_metadata(result.metadata))
        elif format_type == "pdf":
            for page in result.metadata.get("pages", []):
                page_index = int(page.get("page_index", 0))
                layout_blocks.extend(self._build_layout_blocks_from_metadata(page, page_index=page_index))

        result.metadata["layout_blocks"] = layout_blocks

    def _build_layout_blocks_from_metadata(
        self,
        metadata: dict[str, Any],
        *,
        page_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Convert cached OCR token metadata into serialized layout blocks."""
        ocr_data = metadata.get("ocr_data")
        image_shape = metadata.get("image_shape")
        if not ocr_data or not image_shape:
            return []

        image = np.zeros(tuple(image_shape), dtype=np.uint8)
        blocks = self.layout_analyzer.analyze_layout(image, ocr_data)
        serialized = []
        for block in blocks:
            payload = asdict(block)
            if page_index is not None:
                payload["page_index"] = page_index
            serialized.append(payload)
        return serialized

    def _cleanup_internal_metadata(self, result: ExtractionResult) -> None:
        """Remove internal OCR token payloads that are only needed for post-processing."""
        result.metadata.pop("ocr_data", None)
        result.metadata.pop("image_shape", None)
        for page in result.metadata.get("pages", []):
            page.pop("ocr_data", None)
            page.pop("image_shape", None)

    def _build_error_result(
        self,
        *,
        file_path: Path,
        error: str,
        total_start: float,
        profile: bool,
    ) -> ExtractionResult:
        """Build a standardized pipeline error result."""
        result = ExtractionResult(
            text="",
            file_path=str(file_path),
            success=False,
            error=error,
        )
        if profile:
            result.metadata["performance"] = {
                "total_processing": time.perf_counter() - total_start,
            }
        return result

    def _get_extractor(self, format_type: str, languages: list[str] | None = None):
        """Get or initialize extractor for a format type."""
        if format_type not in self.extractors:
            logger.error(f"Unknown format type: {format_type}")
            return None

        if format_type == "image" and languages is not None:
            return OCRExtractor(languages=languages)

        extractor = self.extractors[format_type]
        if extractor is None:
            if format_type == "audio":
                logger.info("Loading Audio Extractor (Whisper)...")
                from ..extractors.audio_extractor import AudioExtractor

                self.extractors["audio"] = AudioExtractor()
                extractor = self.extractors["audio"]
            elif format_type == "video":
                logger.info("Loading Video Extractor (Whisper + FFmpeg)...")
                from ..extractors.video_extractor import VideoExtractor

                self.extractors["video"] = VideoExtractor()
                extractor = self.extractors["video"]

        return extractor
