"""Layout analysis utilities for experimental document structure preservation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.types import OCRTokenPayload
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TextBlock:
    """Represents a text block with position and classification metadata."""

    x: int
    y: int
    width: int
    height: int
    text: str
    confidence: float
    block_type: str


class LayoutAnalyzer:
    """Analyze OCR word boxes and reconstruct a lightweight reading order."""

    @staticmethod
    def analyze_layout(image: np.ndarray, ocr_data: OCRTokenPayload) -> list[TextBlock]:
        """Convert OCR token data into sorted text blocks."""
        blocks: list[TextBlock] = []

        for index in range(len(ocr_data["text"])):
            if int(ocr_data["conf"][index]) < 0:
                continue

            text = str(ocr_data["text"][index]).strip()
            if not text:
                continue

            x = int(ocr_data["left"][index])
            y = int(ocr_data["top"][index])
            width = int(ocr_data["width"][index])
            height = int(ocr_data["height"][index])
            confidence = float(ocr_data["conf"][index])
            block_type = LayoutAnalyzer._classify_block_type(text, x, y, width, height, image.shape)
            blocks.append(TextBlock(x, y, width, height, text, confidence, block_type))

        return LayoutAnalyzer._sort_reading_order(blocks)

    @staticmethod
    def _classify_block_type(
        text: str,
        x: int,
        y: int,
        width: int,
        height: int,
        image_shape: tuple[int, ...],
    ) -> str:
        """Classify a token block using simple page-position heuristics."""
        img_height = image_shape[0]

        if y < img_height * 0.2 and height > 30:
            return "title"

        if text.startswith(("\u2022", "-", "*", "1.", "2.", "3.")):
            return "list"

        if height < 15 and (y > img_height * 0.8 or "Figure" in text or "Table" in text):
            return "caption"

        return "paragraph"

    @staticmethod
    def _sort_reading_order(blocks: list[TextBlock]) -> list[TextBlock]:
        """Sort blocks in a simple top-to-bottom, left-to-right order."""
        return sorted(blocks, key=lambda block: (block.y // 20, block.x))

    @staticmethod
    def reconstruct_text_with_structure(blocks: list[TextBlock]) -> str:
        """Render a text representation that roughly preserves layout semantics."""
        output = []
        current_type = None

        for block in blocks:
            if current_type and current_type != block.block_type:
                output.append("\n")

            if block.block_type == "title":
                output.append(f"\n# {block.text}\n")
            elif block.block_type == "list":
                output.append(f"  {block.text}\n")
            elif block.block_type == "caption":
                output.append(f"\n*{block.text}*\n")
            else:
                output.append(f"{block.text} ")

            current_type = block.block_type

        return "".join(output).strip()

    @staticmethod
    def detect_columns(blocks: list[TextBlock], image_width: int) -> int:
        """Estimate the number of page columns from block positions."""
        if not blocks:
            return 1

        x_positions = sorted(set(block.x for block in blocks))
        gaps = []
        for index in range(len(x_positions) - 1):
            gap = x_positions[index + 1] - x_positions[index]
            if gap > image_width * 0.1:
                gaps.append(gap)

        return len(gaps) + 1
