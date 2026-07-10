"""Validation helpers for image OCR results."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.types import CacheStatus


def build_cache_status(*, hit: bool, kind: str, key: str | None) -> CacheStatus:
    """Build normalized OCR cache metadata."""
    return {
        "hit": hit,
        "kind": kind,
        "key": key,
    }


@dataclass(frozen=True)
class ImageValidationRules:
    """Validation rules for extracted OCR text."""

    confidence_threshold: float

    def validate(self, text: str, confidence: float) -> tuple[bool, str]:
        """Validate OCR output quality."""
        if not text or len(text.strip()) == 0:
            return False, "No text extracted"

        if confidence < self.confidence_threshold:
            return False, f"Low confidence: {confidence:.2f} < {self.confidence_threshold}"

        words = text.split()
        if len(words) < 3:
            return False, f"Too few words extracted: {len(words)}"

        special_char_ratio = sum(
            1 for character in text if not character.isalnum() and not character.isspace()
        ) / len(text)
        if special_char_ratio > 0.5:
            return False, f"Too many special characters: {special_char_ratio:.2%}"

        return True, "Extraction validated successfully"

    @staticmethod
    def parse_confidence(value: object) -> float:
        """Parse OCR confidence values that may arrive as strings or numbers."""
        if not isinstance(value, (int, float, str)):
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
