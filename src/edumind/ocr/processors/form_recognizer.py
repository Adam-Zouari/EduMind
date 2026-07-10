"""Structured field extraction utilities for OCR form-like documents."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FormField:
    """Represents a form field with label and extracted value."""

    label: str
    value: str
    confidence: float
    field_type: str


class FormRecognizer:
    """Recognize structured fields from OCR text using lightweight heuristics."""

    FIELD_PATTERNS = {
        "name": r"(?:name|nome|nom)\s*:?\s*([A-Za-z\s]+)",
        "email": r"(?:email|e-mail|correo)\s*:?\s*([\w\.-]+@[\w\.-]+\.\w+)",
        "phone": r"(?:phone|tel|telephone|telefono)\s*:?\s*([\d\s\-\(\)]+)",
        "date": r"(?:date|fecha|data)\s*:?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
        "address": r"(?:address|direccion|adresse)\s*:?\s*([A-Za-z0-9\s,\.]+)",
        "id": r"(?:id|identification|dni|passport)\s*:?\s*([A-Z0-9]+)",
        "amount": r"(?:amount|total|suma)\s*:?\s*\$?\s*([\d,\.]+)",
    }

    @staticmethod
    def extract_form_fields(text: str) -> list[FormField]:
        """Extract structured fields from OCR text."""
        fields = []

        for field_type, pattern in FormRecognizer.FIELD_PATTERNS.items():
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                label = match.group(0).split(":")[0].strip()
                value = match.group(1).strip()
                validated_type, confidence = FormRecognizer._validate_field(value, field_type)

                if confidence > 0.5:
                    fields.append(
                        FormField(
                            label=label,
                            value=value,
                            confidence=confidence,
                            field_type=validated_type,
                        )
                    )

        fields.extend(FormRecognizer._detect_checkboxes(text))
        fields.extend(FormRecognizer._detect_key_value_pairs(text))
        return fields

    @staticmethod
    def _validate_field(value: str, expected_type: str) -> tuple[str, float]:
        """Validate a field value and return a refined type plus confidence."""
        value = value.strip()

        if expected_type == "email":
            if re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", value):
                return "email", 0.95
            return "text", 0.3

        if expected_type == "phone":
            digits = re.sub(r"[^\d]", "", value)
            if 7 <= len(digits) <= 15:
                return "phone", 0.9
            return "text", 0.4

        if expected_type == "date":
            if re.match(r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}", value):
                return "date", 0.9
            return "text", 0.3

        if expected_type == "amount":
            if re.match(r"^\$?\s*[\d,\.]+$", value):
                return "number", 0.9
            return "text", 0.4

        if expected_type == "id":
            if len(value) >= 5 and any(char.isdigit() for char in value):
                return "id", 0.85
            return "text", 0.5

        return expected_type, 0.7

    @staticmethod
    def _detect_checkboxes(text: str) -> list[FormField]:
        """Detect checkbox fields using ASCII and unicode checkbox markers."""
        checkbox_patterns = [
            (r"\[([X\u2713\u2717])\]\s*([A-Za-z\s]+)", True),
            (r"\[\s\]\s*([A-Za-z\s]+)", False),
            (r"\u2611\s*([A-Za-z\s]+)", True),
            (r"\u2610\s*([A-Za-z\s]+)", False),
        ]

        fields = []
        for pattern, is_checked in checkbox_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                label = match.group(1 if len(match.groups()) == 1 else 2).strip()
                value = "checked" if is_checked else "unchecked"
                fields.append(
                    FormField(
                        label=label,
                        value=value,
                        confidence=0.85,
                        field_type="checkbox",
                    )
                )

        return fields

    @staticmethod
    def _detect_key_value_pairs(text: str) -> list[FormField]:
        """Detect generic `Label: Value` pairs."""
        matches = re.finditer(r"^([A-Za-z\s]+):\s*(.+)$", text, re.MULTILINE)
        fields = []

        for match in matches:
            label = match.group(1).strip()
            value = match.group(2).strip()
            if len(label) > 50 or len(value) > 200:
                continue

            fields.append(
                FormField(
                    label=label,
                    value=value,
                    confidence=0.7,
                    field_type="text",
                )
            )

        return fields

    @staticmethod
    def to_structured_dict(fields: list[FormField]) -> dict[str, dict[str, str | float]]:
        """Convert extracted fields to a dictionary keyed by normalized labels."""
        result: dict[str, dict[str, str | float]] = {}
        for field in fields:
            key = field.label.lower().replace(" ", "_")
            existing = result.get(key)
            candidate = {
                "value": field.value,
                "type": field.field_type,
                "confidence": field.confidence,
            }
            if existing is None or float(candidate["confidence"]) >= float(existing["confidence"]):
                result[key] = candidate
        return result
