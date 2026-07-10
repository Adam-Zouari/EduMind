from __future__ import annotations

import numpy as np

from edumind.ocr.processors.form_recognizer import FormRecognizer
from edumind.ocr.processors.layout_analyzer import LayoutAnalyzer


def test_form_recognizer_extracts_structured_fields_and_checkboxes() -> None:
    text = "\n".join(
        [
            "Name: Alice Example",
            "Email: alice@example.com",
            "[X] Weekly Quiz",
            "\u2610 Dormitory",
        ]
    )

    fields = FormRecognizer.extract_form_fields(text)
    structured = FormRecognizer.to_structured_dict(fields)

    assert structured["email"]["type"] == "email"
    assert any(field.field_type == "checkbox" and field.value == "checked" for field in fields)
    assert any(field.field_type == "checkbox" and field.value == "unchecked" for field in fields)


def test_layout_analyzer_classifies_blocks_and_reconstructs_structure() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    ocr_data = {
        "text": ["Document Title", "- item", "Body text", "Figure 1"],
        "conf": [95, 88, 91, 80],
        "left": [20, 25, 20, 30],
        "top": [10, 70, 110, 185],
        "width": [120, 60, 140, 80],
        "height": [40, 20, 22, 10],
    }

    blocks = LayoutAnalyzer.analyze_layout(image, ocr_data)
    rendered = LayoutAnalyzer.reconstruct_text_with_structure(blocks)

    assert [block.block_type for block in blocks] == ["title", "list", "paragraph", "caption"]
    assert "# Document Title" in rendered
    assert "  - item" in rendered
    assert "*Figure 1*" in rendered


def test_layout_analyzer_detects_column_gaps() -> None:
    blocks = LayoutAnalyzer.analyze_layout(
        np.zeros((100, 400, 3), dtype=np.uint8),
        {
            "text": ["A", "B"],
            "conf": [90, 90],
            "left": [20, 260],
            "top": [20, 20],
            "width": [20, 20],
            "height": [20, 20],
        },
    )

    assert LayoutAnalyzer.detect_columns(blocks, image_width=400) == 2
