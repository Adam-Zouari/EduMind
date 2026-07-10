"""Main pipeline for content extraction."""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

from ..config import PRESERVE_LATEX
from ..extractors.docx_extractor import DOCXExtractor
from ..extractors.ocr_extractor import OCRExtractor
from ..extractors.pdf_extractor import PDFExtractor
from ..extractors.web_extractor import WebExtractor
from ..processors.form_recognizer import FormRecognizer
from ..processors.layout_analyzer import LayoutAnalyzer
from ..processors.math_extractor import MathExtractor
from ..processors.text_cleaner import TextCleaner
from ..utils.logger import get_logger
from .base_extractor import BaseExtractor, ExtractionResult
from .batch_processing import BatchProcessingCoordinator
from .errors import UnsupportedFormatError
from .format_detector import FormatDetector
from .metadata import (
    apply_text_post_processing,
    attach_optional_metadata,
    build_error_result,
    finalize_result_metadata,
)
from .options import BatchProcessingOptions, ProcessFileOptions, ValidBatchStrategy, ValidPdfMode
from .types import FormatInfo, PerformanceStats

logger = get_logger(__name__)


class DataIngestionPipeline:
    """Main pipeline for extracting content from various file formats."""

    def __init__(self) -> None:
        self.format_detector = FormatDetector()
        self.text_cleaner = TextCleaner()
        self.math_extractor = MathExtractor()
        self.layout_analyzer = LayoutAnalyzer()
        self.form_recognizer = FormRecognizer()
        self.extractors: dict[str, BaseExtractor | None] = {
            "pdf": PDFExtractor(),
            "docx": DOCXExtractor(),
            "image": OCRExtractor(),
            "web": WebExtractor(),
            "audio": None,
            "video": None,
        }
        self.batch_coordinator = BatchProcessingCoordinator(
            format_detector=self.format_detector,
            process_detected_file=self._process_detected_file,
            build_error_result=self._build_error_result,
        )
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
        **kwargs: object,
    ) -> ExtractionResult:
        """Process a file and extract its content."""
        logger.info(f"Processing file: {file_path}")
        options = ProcessFileOptions(
            clean_text=clean_text,
            preserve_latex=preserve_latex,
            pdf_ocr_mode=self._coerce_pdf_mode(pdf_ocr_mode),
            include_layout=include_layout,
            include_form_fields=include_form_fields,
            profile=profile,
            include_file_hash=include_file_hash,
            languages=list(languages) if languages is not None else None,
            strict_format_detection=strict_format_detection,
            extra_kwargs=dict(kwargs),
        )
        batch_options = BatchProcessingOptions(
            parallel=False,
            max_workers=1,
            batch_strategy="sequential",
            process_options=options,
        )
        return self.batch_coordinator.process_detected_or_direct(Path(file_path), batch_options)

    def process_batch(
        self,
        file_paths: list[str | Path],
        parallel: bool = True,
        max_workers: int = 4,
        batch_strategy: str = "auto",
        **kwargs: object,
    ) -> list[ExtractionResult]:
        """Process multiple files with optional strategy-aware batching."""
        raw_languages = kwargs.pop("languages", None)
        process_options = ProcessFileOptions(
            clean_text=bool(kwargs.pop("clean_text", True)),
            preserve_latex=bool(kwargs.pop("preserve_latex", PRESERVE_LATEX)),
            pdf_ocr_mode=self._coerce_pdf_mode(kwargs.pop("pdf_ocr_mode", "auto")),
            include_layout=bool(kwargs.pop("include_layout", False)),
            include_form_fields=bool(kwargs.pop("include_form_fields", False)),
            profile=bool(kwargs.pop("profile", False)),
            include_file_hash=bool(kwargs.pop("include_file_hash", True)),
            languages=self._coerce_languages(raw_languages),
            strict_format_detection=bool(kwargs.pop("strict_format_detection", False)),
            extra_kwargs=dict(kwargs),
        )
        batch_options = BatchProcessingOptions(
            parallel=parallel,
            max_workers=max_workers,
            batch_strategy=self._coerce_batch_strategy(batch_strategy),
            process_options=process_options,
        )
        return self.batch_coordinator.process(file_paths, batch_options)

    def _process_detected_file(
        self,
        file_path: Path,
        format_info: FormatInfo,
        format_detection_time: float,
        total_start: float,
        batch_options: BatchProcessingOptions,
    ) -> ExtractionResult:
        """Process a file once format detection has already been completed."""
        process_options = batch_options.process_options
        performance_stats = PerformanceStats(format_detection=format_detection_time)
        format_type = format_info.format_type
        logger.info(f"Detected format: {format_type}")

        try:
            extractor = self._get_extractor(format_type, languages=process_options.languages)
            extract_kwargs = process_options.build_extract_kwargs(format_type)

            extraction_start = time.perf_counter()
            result = extractor.extract(file_path, **extract_kwargs)
            performance_stats.extraction = time.perf_counter() - extraction_start

            apply_text_post_processing(
                result,
                clean_text=process_options.clean_text,
                preserve_latex=process_options.preserve_latex,
                text_cleaner=self.text_cleaner,
                math_extractor=self.math_extractor,
                performance_stats=performance_stats,
            )

            if result.success and format_type in {"image", "pdf"}:
                attach_optional_metadata(
                    result,
                    format_type=format_type,
                    include_layout=process_options.include_layout,
                    include_form_fields=process_options.include_form_fields,
                    layout_analyzer=self.layout_analyzer,
                    form_recognizer=self.form_recognizer,
                )

            performance_stats.total_processing = time.perf_counter() - total_start
            finalize_result_metadata(
                result,
                format_info=format_info,
                file_path=file_path,
                include_file_hash=process_options.include_file_hash,
                profile=process_options.profile,
                performance_stats=performance_stats,
            )
            logger.info(f"Processing completed in {performance_stats.total_processing:.2f}s")
            return result
        except UnsupportedFormatError as exc:
            return self._build_error_result(
                file_path,
                str(exc),
                total_start,
                process_options.profile,
            )
        except Exception as exc:
            logger.error(f"Pipeline processing failed: {exc}", exc_info=True)
            return self._build_error_result(
                file_path,
                str(exc),
                total_start,
                process_options.profile,
            )

    def _build_error_result(
        self,
        file_path: Path,
        error: str,
        total_start: float,
        profile: bool,
    ) -> ExtractionResult:
        """Build a standardized pipeline error result."""
        return build_error_result(
            file_path=file_path,
            error=error,
            total_start=total_start,
            profile=profile,
        )

    def _get_extractor(self, format_type: str, languages: list[str] | None = None) -> BaseExtractor:
        """Get or initialize the extractor for a format type."""
        if format_type not in self.extractors:
            raise UnsupportedFormatError(f"No extractor available for format: {format_type}")

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

        if extractor is None:
            raise UnsupportedFormatError(f"No extractor available for format: {format_type}")
        return extractor

    def _coerce_pdf_mode(self, value: object) -> ValidPdfMode:
        """Normalize a public PDF OCR mode string into the typed internal literal."""
        return cast(ValidPdfMode, str(value))

    def _coerce_batch_strategy(self, value: object) -> ValidBatchStrategy:
        """Normalize a public batch strategy string into the typed internal literal."""
        return cast(ValidBatchStrategy, str(value))

    def _coerce_languages(self, value: object) -> list[str] | None:
        """Normalize a public languages value into a list of strings."""
        if not isinstance(value, list):
            return None
        return [str(language) for language in value]
