"""Base abstractions for OCR extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Normalized extraction output used by OCR and RAG flows."""

    text: str
    metadata: dict[str, object] = field(default_factory=dict)
    format_type: str = ""
    file_path: str = ""
    extraction_time: float = 0.0
    success: bool = True
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"text": self.text}
        result.update(self.metadata)
        if "format_type" not in result:
            result["format_type"] = self.format_type
        if self.file_path:
            result["source"] = Path(self.file_path).name
        result["success"] = self.success
        return result

    def to_cache_dict(self) -> dict[str, object]:
        """Serialize the full extraction result for OCR cache storage."""
        return {
            "cache_version": 1,
            "text": self.text,
            "metadata": self.metadata,
            "format_type": self.format_type,
            "file_path": self.file_path,
            "extraction_time": self.extraction_time,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_cache_dict(cls, data: dict[str, object]) -> ExtractionResult:
        """Rebuild an extraction result from the OCR cache payload."""
        raw_metadata = data.get("metadata")
        if isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
        else:
            metadata = {
                key: value
                for key, value in data.items()
                if key
                not in {
                    "cache_version",
                    "text",
                    "format_type",
                    "source",
                    "file_path",
                    "extraction_time",
                    "success",
                    "error",
                    "timestamp",
                }
            }

        timestamp = datetime.now()
        raw_timestamp = data.get("timestamp")
        if isinstance(raw_timestamp, str) and raw_timestamp:
            try:
                timestamp = datetime.fromisoformat(raw_timestamp)
            except ValueError:
                logger.warning("Ignoring invalid OCR cache timestamp: {}", raw_timestamp)

        raw_text = data.get("text", "")
        raw_format_type = data.get("format_type", "")
        raw_file_path = data.get("file_path", data.get("source", ""))
        raw_extraction_time = data.get("extraction_time", 0.0)
        raw_error = data.get("error")

        return cls(
            text=raw_text if isinstance(raw_text, str) else "",
            metadata=metadata,
            format_type=raw_format_type if isinstance(raw_format_type, str) else "",
            file_path=raw_file_path if isinstance(raw_file_path, str) else "",
            extraction_time=_coerce_float(raw_extraction_time),
            success=bool(data.get("success", True)),
            error=raw_error if isinstance(raw_error, str) else None,
            timestamp=timestamp,
        )


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def extract(self, file_path: Path, **kwargs: object) -> ExtractionResult:
        """Extract content from a file."""

    def _create_error_result(self, file_path: Path, error: str) -> ExtractionResult:
        return ExtractionResult(
            text="",
            file_path=str(file_path),
            success=False,
            error=error,
        )


def _coerce_float(value: object) -> float:
    """Safely coerce serialized numeric values to floats."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
