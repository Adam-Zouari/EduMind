"""Image OCR extraction orchestrated through focused helper components."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..config import (
    OCR_CACHE_DIR,
    OCR_CONFIDENCE_THRESHOLD,
    OCR_ENABLE_CACHING,
    OCR_LANGUAGES,
    OCR_USE_PADDLE,
)
from ..core.base_extractor import BaseExtractor, ExtractionResult
from ..core.errors import CacheReadError, CacheWriteError
from ..core.types import OCRTokenPayload
from ..utils.cache_keys import build_image_cache_key
from ._image_backends import ImageOCRBackends
from ._image_cache import ImageCacheStore
from ._image_preprocessing import ImagePreprocessor
from ._image_validation import ImageValidationRules, build_cache_status

try:
    import mlflow
except ImportError:
    mlflow = None


class OCRExtractor(BaseExtractor):
    """Extract text from images with preprocessing, validation, and caching."""

    def __init__(
        self,
        use_paddle: bool | None = None,
        confidence_threshold: float | None = None,
        languages: list[str] | None = None,
        enable_caching: bool = OCR_ENABLE_CACHING,
    ) -> None:
        super().__init__()
        resolved_threshold = OCR_CONFIDENCE_THRESHOLD if confidence_threshold is None else confidence_threshold
        self.languages = list(languages or OCR_LANGUAGES)
        self.enable_caching = enable_caching
        self.confidence_threshold = resolved_threshold
        self._cache_dir = OCR_CACHE_DIR if enable_caching else None
        self.preprocessor = ImagePreprocessor()
        self.validation_rules = ImageValidationRules(confidence_threshold=resolved_threshold)
        self.backend = ImageOCRBackends(
            use_paddle=OCR_USE_PADDLE if use_paddle is None else use_paddle,
            confidence_threshold=resolved_threshold,
            validation_rules=self.validation_rules,
        )
        self.cache_store = ImageCacheStore(self._cache_dir)

    @property
    def use_paddle(self) -> bool:
        """Expose the active backend mode for callers such as PDF fallback."""
        return self.backend.use_paddle

    @property
    def engine_name(self) -> str:
        """Expose the active OCR engine name."""
        return self.backend.engine_name

    @property
    def cache_dir(self) -> Path | None:
        """Expose the active OCR cache directory."""
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, value: Path | None) -> None:
        """Keep the cache store in sync when tests or callers swap directories."""
        self._cache_dir = value
        self.cache_store = ImageCacheStore(value)

    def extract(self, file_path: Path, **kwargs: object) -> ExtractionResult:
        """Extract text from an image file."""
        image = cv2.imread(str(file_path))
        if image is None:
            return self._create_error_result(file_path, f"Could not read image: {file_path}")

        result = self.extract_image(
            image,
            source_name=str(file_path),
            cache_key=build_image_cache_key(file_path),
            languages=self._normalize_languages(kwargs.get("languages")),
            return_ocr_data=bool(kwargs.get("return_ocr_data", False)),
            use_cache=bool(kwargs.get("use_cache", True)),
            format_type=str(kwargs.get("format_type", "image")),
        )
        result.file_path = str(file_path)
        return result

    def extract_image(
        self,
        image: np.ndarray,
        *,
        source_name: str = "",
        cache_key: str | None = None,
        languages: list[str] | None = None,
        return_ocr_data: bool = False,
        use_cache: bool = True,
        format_type: str = "image",
    ) -> ExtractionResult:
        """Extract text from an in-memory image array."""
        start_time = time.time()
        effective_languages = self._normalize_languages(languages)

        if self.enable_caching and use_cache and cache_key:
            cached_result = self._try_read_cached_result(cache_key, format_type=format_type)
            if cached_result is not None:
                return cached_result

        try:
            quality_score = self.preprocessor.assess_quality(image)
            self.logger.info(f"Image quality score: {quality_score:.2f}")
            preprocessed = self.preprocessor.preprocess(image, quality_score)
            backend_result = self.backend.extract(
                preprocessed.image,
                languages=effective_languages,
                return_ocr_data=return_ocr_data,
            )
            is_valid, validation_msg = self.validation_rules.validate(
                backend_result.text,
                backend_result.confidence,
            )
            extraction_time = time.time() - start_time

            metadata = self._build_metadata(
                image=image,
                format_type=format_type,
                cache_key=cache_key,
                quality_score=quality_score,
                preprocessing_metadata=preprocessed.metadata,
                languages=effective_languages,
                confidence=backend_result.confidence,
                attempts=backend_result.attempts,
                ocr_data=backend_result.ocr_data,
                validation_message=validation_msg,
                is_valid=is_valid,
                return_ocr_data=return_ocr_data,
            )
            self._log_optional_mlflow_metrics(
                extraction_time=extraction_time,
                confidence=backend_result.confidence,
                quality_score=quality_score,
                ocr_engine=self.engine_name,
            )

            result = ExtractionResult(
                text=backend_result.text,
                metadata=metadata,
                format_type=format_type,
                file_path=source_name,
                extraction_time=extraction_time,
                success=is_valid,
            )
            if self.enable_caching and use_cache and cache_key and is_valid:
                self._try_write_cached_result(cache_key, result)
            return result
        except Exception as exc:
            self.logger.error(f"OCR extraction failed: {exc}")
            return ExtractionResult(
                text="",
                metadata={
                    "cache": build_cache_status(hit=False, kind=format_type, key=cache_key),
                },
                format_type=format_type,
                file_path=source_name,
                success=False,
                error=str(exc),
            )

    def _normalize_languages(self, languages: list[str] | None | object) -> list[str]:
        """Normalize optional language input into a concrete list."""
        if isinstance(languages, list) and languages:
            return [str(language) for language in languages]
        return list(self.languages)

    def _try_read_cached_result(
        self,
        cache_key: str,
        *,
        format_type: str,
    ) -> ExtractionResult | None:
        """Attempt to read a cached OCR result without failing the request."""
        if not self.cache_store.exists(cache_key):
            return None
        try:
            cached_result = self.cache_store.read(cache_key)
        except CacheReadError as exc:
            self.logger.warning(f"Failed to load cache: {exc}")
            return None

        cached_result.metadata["cache"] = build_cache_status(
            hit=True,
            kind=format_type,
            key=cache_key,
        )
        return cached_result

    def _try_write_cached_result(self, cache_key: str, result: ExtractionResult) -> None:
        """Attempt to persist a cached OCR result without failing the request."""
        try:
            self.cache_store.write(cache_key, result)
        except CacheWriteError as exc:
            self.logger.warning(f"Failed to cache result: {exc}")

    def _build_metadata(
        self,
        *,
        image: np.ndarray,
        format_type: str,
        cache_key: str | None,
        quality_score: float,
        preprocessing_metadata: dict[str, object],
        languages: list[str],
        confidence: float,
        attempts: list[str],
        ocr_data: OCRTokenPayload | None,
        validation_message: str,
        is_valid: bool,
        return_ocr_data: bool,
    ) -> dict[str, object]:
        """Build the public OCR metadata payload."""
        metadata: dict[str, object] = {
            "ocr_engine": self.engine_name,
            "confidence": confidence,
            "languages": languages,
            "quality_score": quality_score,
            "preprocessing": preprocessing_metadata,
            "extraction_attempts": attempts,
            "validation": {"is_valid": is_valid, "message": validation_message},
            "extractor": "ocr",
            "cache": build_cache_status(hit=False, kind=format_type, key=cache_key),
        }
        if return_ocr_data and ocr_data is not None:
            metadata["ocr_data"] = ocr_data
            metadata["image_shape"] = list(image.shape)
        return metadata

    def _log_optional_mlflow_metrics(
        self,
        *,
        extraction_time: float,
        confidence: float,
        quality_score: float,
        ocr_engine: str,
    ) -> None:
        """Log OCR metrics to MLflow when the dependency and an active run exist."""
        if mlflow is None:
            return

        try:
            if mlflow.active_run():
                mlflow.log_metrics(
                    {
                        "ocr_processing_time": extraction_time,
                        "ocr_confidence": float(confidence),
                        "image_quality_score": quality_score,
                    }
                )
                mlflow.log_param("ocr_engine", ocr_engine)
        except Exception as exc:
            self.logger.warning(f"Failed to log to MLflow: {exc}")
