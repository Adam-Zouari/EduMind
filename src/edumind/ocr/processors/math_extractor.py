"""Extract and preserve mathematical notation."""

from __future__ import annotations

import re

from ..utils.logger import get_logger

logger = get_logger(__name__)


class MathExtractor:
    """Extract and preserve LaTeX-style mathematical notation."""

    @staticmethod
    def extract_latex(text: str) -> dict[str, list[str]]:
        """Extract inline and display LaTeX expressions from text."""
        inline_math = re.findall(r"(?<!\$)\$([^\$]+)\$(?!\$)", text)
        display_math = re.findall(r"\$\$([^\$]+)\$\$", text)
        equation_math = re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", text, re.DOTALL)
        align_math = re.findall(r"\\begin\{align\}(.*?)\\end\{align\}", text, re.DOTALL)

        return {
            "inline": inline_math,
            "display": display_math + equation_math + align_math,
        }

    @staticmethod
    def preserve_math(text: str) -> tuple[str, dict[str, str]]:
        """Replace math expressions with placeholders during text cleanup."""
        math_dict: dict[str, str] = {}
        counter = 0
        math_patterns = [
            (r"\$\$.*?\$\$", "DISPLAYMATH"),
            (r"\$.*?\$", "INLINEMATH"),
            (r"\\begin\{equation\}.*?\\end\{equation\}", "EQUATION"),
            (r"\\begin\{align\}.*?\\end\{align\}", "ALIGN"),
        ]

        for pattern, prefix in math_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                placeholder = f"__{prefix}_{counter}__"
                math_dict[placeholder] = match.group(0)
                text = text.replace(match.group(0), placeholder, 1)
                counter += 1

        return text, math_dict

    @staticmethod
    def restore_math(text: str, math_dict: dict[str, str]) -> str:
        """Restore math expressions after text cleanup."""
        for placeholder, expression in math_dict.items():
            text = text.replace(placeholder, expression)
        return text
