"""Shared batch-processing helpers for the OCR pipeline."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Semaphore

from tqdm import tqdm

from ..config import OCR_USE_GPU
from ..utils.file_handler import FileHandler
from ..utils.logger import get_logger
from .base_extractor import ExtractionResult
from .format_detector import FormatDetector
from .options import BatchProcessingOptions
from .types import FormatInfo

logger = get_logger(__name__)

DetectedFileProcessor = Callable[
    [Path, FormatInfo, float, float, BatchProcessingOptions],
    ExtractionResult,
]
ErrorBuilder = Callable[[Path, str, float, bool], ExtractionResult]


@dataclass(frozen=True)
class DetectedFile:
    """A validated file plus its detected OCR format."""

    index: int
    file_path: Path
    format_info: FormatInfo
    format_detection_time: float
    total_start: float


class BatchProcessingCoordinator:
    """Coordinate sequential and threaded batch OCR processing."""

    _THREAD_SAFE_FORMATS = {"pdf", "docx", "image", "web"}

    def __init__(
        self,
        *,
        format_detector: FormatDetector,
        process_detected_file: DetectedFileProcessor,
        build_error_result: ErrorBuilder,
    ) -> None:
        self.format_detector = format_detector
        self.process_detected_file = process_detected_file
        self.build_error_result = build_error_result

    def process(
        self,
        file_paths: list[str | Path],
        options: BatchProcessingOptions,
    ) -> list[ExtractionResult]:
        """Process a batch of files with the requested strategy."""
        strategy = options.effective_strategy
        logger.info(
            f"Processing batch of {len(file_paths)} files "
            f"(strategy={strategy}, parallel={options.parallel})"
        )

        if len(file_paths) <= 1 or strategy == "sequential":
            return [
                self.process_detected_or_direct(Path(file_path), options)
                for file_path in file_paths
            ]

        if strategy == "threads":
            return self._process_threads(file_paths, options)

        return self._process_auto(file_paths, options)

    def process_detected_or_direct(
        self,
        file_path: Path,
        options: BatchProcessingOptions,
    ) -> ExtractionResult:
        """Process a file by validating and detecting it first."""
        total_start = time.perf_counter()
        if not FileHandler.validate_file(file_path):
            return self.build_error_result(
                file_path,
                "File validation failed",
                total_start,
                options.process_options.profile,
            )

        try:
            detect_start = time.perf_counter()
            format_info = FormatInfo.from_value(
                self.format_detector.detect(
                    file_path,
                    strict=options.process_options.strict_format_detection,
                )
            )
        except Exception as exc:
            logger.error(f"Format detection failed for {file_path}: {exc}")
            return self.build_error_result(
                file_path,
                str(exc),
                total_start,
                options.process_options.profile,
            )

        return self.process_detected_file(
            file_path,
            format_info,
            time.perf_counter() - detect_start,
            total_start,
            options,
        )

    def _process_threads(
        self,
        file_paths: list[str | Path],
        options: BatchProcessingOptions,
    ) -> list[ExtractionResult]:
        """Process a batch using a plain thread pool."""
        results: list[ExtractionResult | None] = [None] * len(file_paths)
        with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
            future_to_index = {
                executor.submit(self.process_detected_or_direct, Path(file_path), options): index
                for index, file_path in enumerate(file_paths)
            }
            for future in tqdm(
                as_completed(future_to_index),
                total=len(file_paths),
                desc="Processing files",
            ):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    logger.error(f"Error processing file {file_paths[index]}: {exc}")
                    results[index] = self.build_error_result(
                        Path(file_paths[index]),
                        str(exc),
                        time.perf_counter(),
                        options.process_options.profile,
                    )
        return [result for result in results if result is not None]

    def _process_auto(
        self,
        file_paths: list[str | Path],
        options: BatchProcessingOptions,
    ) -> list[ExtractionResult]:
        """Process thread-safe formats concurrently and media formats sequentially."""
        results: list[ExtractionResult | None] = [None] * len(file_paths)
        detected_items = self._detect_files(file_paths, options, results)
        image_limiter = Semaphore(min(options.max_workers, 2)) if not OCR_USE_GPU else None
        thread_items = [
            item
            for item in detected_items
            if item.format_info.format_type in self._THREAD_SAFE_FORMATS
        ]
        sequential_items = [
            item
            for item in detected_items
            if item.format_info.format_type not in self._THREAD_SAFE_FORMATS
        ]

        with ThreadPoolExecutor(max_workers=options.max_workers) as executor:
            future_to_index = {}
            for item in thread_items:
                future = executor.submit(
                    self._process_with_optional_image_limit,
                    item,
                    options,
                    image_limiter if item.format_info.format_type == "image" else None,
                )
                future_to_index[future] = item.index

            for item in sequential_items:
                results[item.index] = self.process_detected_file(
                    item.file_path,
                    item.format_info,
                    item.format_detection_time,
                    item.total_start,
                    options,
                )

            for future in tqdm(
                as_completed(future_to_index),
                total=len(future_to_index),
                desc="Processing files",
            ):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as exc:
                    logger.error(f"Error processing file {file_paths[index]}: {exc}")
                    results[index] = self.build_error_result(
                        Path(file_paths[index]),
                        str(exc),
                        time.perf_counter(),
                        options.process_options.profile,
                    )

        return [result for result in results if result is not None]

    def _detect_files(
        self,
        file_paths: list[str | Path],
        options: BatchProcessingOptions,
        results: list[ExtractionResult | None],
    ) -> list[DetectedFile]:
        """Validate and detect files before mixed batch scheduling."""
        detected_items: list[DetectedFile] = []
        for index, raw_path in enumerate(file_paths):
            file_path = Path(raw_path)
            total_start = time.perf_counter()
            if not FileHandler.validate_file(file_path):
                results[index] = self.build_error_result(
                    file_path,
                    "File validation failed",
                    total_start,
                    options.process_options.profile,
                )
                continue

            try:
                detect_start = time.perf_counter()
                format_info = FormatInfo.from_value(
                    self.format_detector.detect(
                        file_path,
                        strict=options.process_options.strict_format_detection,
                    )
                )
            except Exception as exc:
                logger.error(f"Format detection failed for {file_path}: {exc}")
                results[index] = self.build_error_result(
                    file_path,
                    str(exc),
                    total_start,
                    options.process_options.profile,
                )
                continue

            detected_items.append(
                DetectedFile(
                    index=index,
                    file_path=file_path,
                    format_info=format_info,
                    format_detection_time=time.perf_counter() - detect_start,
                    total_start=total_start,
                )
            )

        return detected_items

    def _process_with_optional_image_limit(
        self,
        detected_file: DetectedFile,
        options: BatchProcessingOptions,
        image_limiter: Semaphore | None,
    ) -> ExtractionResult:
        """Process a detected file while optionally limiting concurrent image OCR."""
        if image_limiter is None:
            return self.process_detected_file(
                detected_file.file_path,
                detected_file.format_info,
                detected_file.format_detection_time,
                detected_file.total_start,
                options,
            )

        with image_limiter:
            return self.process_detected_file(
                detected_file.file_path,
                detected_file.format_info,
                detected_file.format_detection_time,
                detected_file.total_start,
                options,
            )
