"""Base abstractions for OCR extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Normalized extraction output used by OCR and RAG flows."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    format_type: str = ""
    file_path: str = ""
    extraction_time: float = 0.0
    success: bool = True
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"text": self.text}
        result.update(self.metadata)
        if "format_type" not in result:
            result["format_type"] = self.format_type
        if self.file_path:
            result["source"] = Path(self.file_path).name
        result["success"] = self.success
        return result


class BaseExtractor(ABC):
    """Abstract base class for all extractors."""

    def __init__(self) -> None:
        self.logger = get_logger(self.__class__.__name__)

    @abstractmethod
    def extract(self, file_path: Path, **kwargs: Any) -> ExtractionResult:
        """Extract content from a file."""

    def _create_error_result(self, file_path: Path, error: str) -> ExtractionResult:
        return ExtractionResult(
            text="",
            file_path=str(file_path),
            success=False,
            error=error,
        )
