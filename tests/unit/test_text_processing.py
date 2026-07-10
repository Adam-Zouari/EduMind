from __future__ import annotations

from edumind.ocr.processors.math_extractor import MathExtractor
from edumind.ocr.processors.text_cleaner import TextCleaner


def test_text_cleaner_normalizes_text_and_fixes_common_ocr_errors() -> None:
    raw_text = "Page 1\n\ntlie   wlth 2O23 ,,  \n\nconfidential\n"

    cleaned = TextCleaner.clean(raw_text, aggressive_ocr_fix=True)

    assert "Page 1" not in cleaned
    assert "confidential" not in cleaned.lower()
    assert "the with 2023," in cleaned
    assert "  " not in cleaned


def test_text_cleaner_can_strip_latex_when_requested() -> None:
    cleaned = TextCleaner.clean("Keep $x^2$ out", preserve_latex=False)

    assert "$x^2$" not in cleaned


def test_math_extractor_preserve_and_restore_round_trip() -> None:
    source = "Energy is $E=mc^2$ and $$a^2+b^2=c^2$$."

    preserved, placeholders = MathExtractor.preserve_math(source)
    restored = MathExtractor.restore_math(preserved, placeholders)

    assert "__INLINEMATH_" in preserved
    assert "__DISPLAYMATH_" in preserved
    assert restored == source
    assert MathExtractor.extract_latex(source)["inline"] == ["E=mc^2"]
