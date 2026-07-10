"""Text cleaning and normalization with context-aware OCR error correction."""

from __future__ import annotations

import re

import ftfy

from ..config import MIN_TEXT_LENGTH, NORMALIZE_WHITESPACE, REMOVE_HEADERS_FOOTERS
from ..utils.logger import get_logger

logger = get_logger(__name__)

OCR_ERROR_PATTERNS = {
    "letter_to_letter": {
        r"\brn\b": "m",
        r"\bvv\b": "w",
        r"\bcl\b": "d",
        r"\bII\b": "ll",
    },
    "ambiguous_chars": {
        "O": "0",
        "l": "1",
        "I": "1",
        "S": "5",
        "Z": "2",
    },
    "common_words": {
        r"\btlie\b": "the",
        r"\btbe\b": "the",
        r"\banci\b": "and",
        r"\bwlth\b": "with",
        r"\bfrom\b": "from",
        r"\bthls\b": "this",
        r"\bthat\b": "that",
        r"\bwhlch\b": "which",
        r"\bwlll\b": "will",
        r"\bcan\b": "can",
        r"\bhas\b": "has",
        r"\bhave\b": "have",
    },
    "punctuation": {
        r"\s+([.,!?;:])": r"\1",
        r"([.,!?;:])\s*([.,!?;:])": r"\1",
        r",,": ",",
        r"\.\.": ".",
    },
}


class TextCleaner:
    """Clean and normalize extracted text with lightweight OCR heuristics."""

    @staticmethod
    def clean(text: str, preserve_latex: bool = True, aggressive_ocr_fix: bool = False) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""

        text = ftfy.fix_text(text)

        if REMOVE_HEADERS_FOOTERS:
            text = TextCleaner._remove_headers_footers(text)

        if NORMALIZE_WHITESPACE:
            text = TextCleaner._normalize_whitespace(text)

        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
        text = re.sub(r"[-_]{4,}", "", text)
        text = TextCleaner._fix_ocr_errors_advanced(text, aggressive=aggressive_ocr_fix)

        if not preserve_latex:
            text = re.sub(r"\$.*?\$", "", text)
            text = re.sub(r"\\[a-zA-Z]+\{.*?\}", "", text)

        return text.strip()

    @staticmethod
    def _remove_headers_footers(text: str) -> str:
        """Remove common header/footer patterns."""
        patterns = [
            r"^\s*page\s+\d+",
            r"^\s*\d+\s*$",
            r"confidential",
            r"proprietary",
            r"copyright",
        ]

        cleaned_lines = []
        for line in text.split("\n"):
            if any(re.search(pattern, line.lower()) for pattern in patterns):
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize repeated spaces and blank lines."""
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return "\n".join(line.strip() for line in text.split("\n"))

    @staticmethod
    def _fix_ocr_errors(text: str) -> str:
        """Legacy OCR error fix kept for backward compatibility."""
        return TextCleaner._fix_ocr_errors_advanced(text, aggressive=False)

    @staticmethod
    def _fix_ocr_errors_advanced(text: str, aggressive: bool = False) -> str:
        """Fix common OCR errors with lightweight context awareness."""
        for pattern, replacement in OCR_ERROR_PATTERNS["letter_to_letter"].items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        for pattern, replacement in OCR_ERROR_PATTERNS["common_words"].items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        for pattern, replacement in OCR_ERROR_PATTERNS["punctuation"].items():
            text = re.sub(pattern, replacement, text)

        if aggressive:
            text = TextCleaner._fix_ambiguous_characters(text)

        return text

    @staticmethod
    def _fix_ambiguous_characters(text: str) -> str:
        """Fix ambiguous OCR characters only when a token looks numeric."""
        corrected_words = []
        for word in text.split():
            if TextCleaner._is_likely_number(word):
                corrected = word
                for letter, number in OCR_ERROR_PATTERNS["ambiguous_chars"].items():
                    corrected = corrected.replace(letter, number)
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        return " ".join(corrected_words)

    @staticmethod
    def _is_likely_number(word: str) -> bool:
        """Heuristically determine whether a token is intended to be numeric."""
        cleaned = word.replace(",", "").replace(".", "").replace("-", "").replace("/", "")
        if not cleaned:
            return False

        digit_count = sum(1 for char in cleaned if char.isdigit())
        ambiguous_count = sum(1 for char in cleaned if char in "OlISZ")
        total_chars = len(cleaned)

        if (digit_count + ambiguous_count) / total_chars > 0.7:
            return True

        return any(
            re.match(pattern, word)
            for pattern in (
                r"^\d{1,2}[Ol]\d+$",
                r"^\d+[Ol]$",
                r"^[Ol]\d+$",
                r"^\d+[lI]\d+$",
            )
        )

    @staticmethod
    def extract_sentences(text: str) -> list[str]:
        """Extract sentences from text."""
        sentences = re.split(r"[.!?]+\s+", text)
        return [
            sentence.strip()
            for sentence in sentences
            if len(sentence.strip()) > MIN_TEXT_LENGTH
        ]
