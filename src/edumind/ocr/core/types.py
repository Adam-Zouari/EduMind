"""Typed internal contracts for the OCR package."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict


class CacheStatus(TypedDict):
    """Serialized OCR cache status metadata."""

    hit: bool
    kind: str
    key: str | None


class OCRTokenPayload(TypedDict):
    """Normalized token payload returned by OCR backends."""

    text: list[str]
    conf: list[float]
    left: list[int]
    top: list[int]
    width: list[int]
    height: list[int]


class StructuredFieldPayload(TypedDict):
    """Serialized field metadata returned by form recognition."""

    value: str
    type: str
    confidence: float


class LayoutBlockPayload(TypedDict, total=False):
    """Serialized layout block metadata."""

    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float
    block_type: str
    page_index: int


class PdfPageMetadata(TypedDict, total=False):
    """Serialized per-page PDF OCR metadata."""

    page_index: int
    source: str
    confidence: float | None
    extraction_time: float
    fallback_reason: str | None
    cache: CacheStatus
    ocr_data: OCRTokenPayload
    image_shape: list[int]
    ocr_error: str


@dataclass(frozen=True)
class FormatInfo:
    """Normalized format-detection result."""

    format_type: str
    mime_type: str | None
    extension: str

    @classmethod
    def from_value(cls, value: FormatInfo | Mapping[str, object]) -> FormatInfo:
        """Normalize dict-style or dataclass-style format metadata."""
        if isinstance(value, FormatInfo):
            return value

        mime_value = value.get("mime_type")
        return cls(
            format_type=str(value.get("format_type", "unknown")),
            mime_type=str(mime_value) if mime_value is not None else None,
            extension=str(value.get("extension", "")),
        )

    def to_metadata_dict(self) -> dict[str, str | None]:
        """Return the public metadata representation."""
        return {
            "format_type": self.format_type,
            "mime_type": self.mime_type,
            "extension": self.extension,
        }


@dataclass
class PerformanceStats:
    """Collected per-file timing metadata."""

    format_detection: float | None = None
    extraction: float | None = None
    cleaning: float | None = None
    hashing: float | None = None
    total_processing: float | None = None

    def to_metadata_dict(self) -> dict[str, float]:
        """Serialize only populated timing values."""
        values = {
            "format_detection": self.format_detection,
            "extraction": self.extraction,
            "cleaning": self.cleaning,
            "hashing": self.hashing,
            "total_processing": self.total_processing,
        }
        return {
            key: value
            for key, value in values.items()
            if value is not None
        }
