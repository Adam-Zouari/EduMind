"""Package-local exceptions for OCR processing."""

from __future__ import annotations


class OCRProcessingError(Exception):
    """Base exception for OCR package failures."""


class FormatDetectionError(OCRProcessingError):
    """Raised when the format of a file cannot be determined reliably."""


class UnsupportedFormatError(OCRProcessingError):
    """Raised when no extractor supports the detected format."""


class OptionalDependencyError(OCRProcessingError):
    """Raised when an optional OCR dependency is unavailable."""


class OCRBackendError(OCRProcessingError):
    """Raised when an OCR backend fails to produce text."""


class CacheReadError(OCRProcessingError):
    """Raised when OCR cache contents cannot be read safely."""


class CacheWriteError(OCRProcessingError):
    """Raised when OCR cache contents cannot be written safely."""


class MediaExtractionError(OCRProcessingError):
    """Raised when audio/video preprocessing fails."""
