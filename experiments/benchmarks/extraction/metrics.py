"""Text, structure, table, formula, and timing metrics for extraction."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from edumind.extraction import ExtractedDocument, SegmentKind

from experiments.benchmarks.common.metrics import (
    character_error_rate,
    levenshtein,
    normalized_tokens,
    word_error_rate,
)


def base_scores(reference: str, hypothesis: str) -> dict[str, float]:
    reference = canonicalize_for_evaluation(reference)
    hypothesis = canonicalize_for_evaluation(hypothesis)
    scores = content_scores(reference, hypothesis)
    scores.update(block_scores(reference, hypothesis))
    scores.update(
        {
            "character_error_rate": character_error_rate(reference, hypothesis),
            "word_error_rate": word_error_rate(reference, hypothesis),
            "reading_order_accuracy": reading_order_accuracy(reference, hypothesis),
            "empty_output_rate": float(not hypothesis.strip()),
            "word_accuracy": max(
                0.0, 1.0 - min(1.0, word_error_rate(reference, hypothesis))
            ),
            "duplicate_text_rate": duplicate_line_rate(hypothesis),
        }
    )
    return scores


def canonicalize_for_evaluation(text: str) -> str:
    """Standardize encoding and line endings without repairing extracted content."""

    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")


def page_scores(item: Mapping[str, object], document: ExtractedDocument) -> dict[str, float]:
    raw_references = item.get("reference_page_texts")
    if isinstance(raw_references, Sequence) and not isinstance(raw_references, (str, bytes)):
        references = [str(value) for value in raw_references]
    elif item.get("kind") == "image":
        references = [str(item.get("reference", ""))]
    else:
        expected = max(1, int(item.get("reference_pages", 1)))
        pages = {
            segment.page_number
            for segment in document.segments
            if segment.page_number is not None and segment.text.strip()
        }
        return {"page_coverage": min(1.0, len(pages) / expected)}

    predicted = {
        page: "\n".join(
            segment.text
            for segment in document.segments
            if segment.page_number == page and segment.text.strip()
        )
        for page in range(1, len(references) + 1)
    }
    populated = [value for value in predicted.values() if value.strip()]
    correct_attribution = 0
    same_page_scores: list[float] = []
    for page, reference in enumerate(references, 1):
        hypothesis = predicted[page]
        same_page_scores.append(content_scores(reference, hypothesis)["content_f1"])
        if not hypothesis.strip():
            continue
        similarities = [content_scores(value, hypothesis)["content_f1"] for value in references]
        if int(np.argmax(similarities)) + 1 == page:
            correct_attribution += 1
    normalized = [" ".join(value.casefold().split()) for value in populated]
    duplicate_rate = (
        (len(normalized) - len(set(normalized))) / len(normalized) if normalized else 0.0
    )
    coverage = len(populated) / len(references)
    return {
        "page_coverage": coverage,
        "page_content_f1": float(np.mean(same_page_scores)),
        "page_attribution_accuracy": correct_attribution / len(references),
        "missing_page_rate": 1.0 - coverage,
        "duplicate_page_rate": duplicate_rate,
    }


def content_scores(reference: str, hypothesis: str) -> dict[str, float]:
    reference_tokens = Counter(normalized_tokens(reference))
    hypothesis_tokens = Counter(normalized_tokens(hypothesis))
    overlap = sum((reference_tokens & hypothesis_tokens).values())
    reference_total = sum(reference_tokens.values())
    hypothesis_total = sum(hypothesis_tokens.values())
    precision = overlap / hypothesis_total if hypothesis_total else 0.0
    recall = overlap / reference_total if reference_total else float(not hypothesis_total)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    missing = sum((reference_tokens - hypothesis_tokens).values()) / max(1, reference_total)
    hallucinated = sum((hypothesis_tokens - reference_tokens).values()) / max(1, hypothesis_total)
    return {
        "content_precision": precision,
        "content_recall": recall,
        "content_f1": f1,
        "missing_text_rate": missing,
        "hallucinated_text_rate": hallucinated,
    }


def reading_order_accuracy(reference: str, hypothesis: str) -> float:
    """Pairwise order accuracy over tokens occurring once in both texts."""
    reference_tokens = normalized_tokens(reference)
    hypothesis_tokens = normalized_tokens(hypothesis)
    reference_counts, hypothesis_counts = Counter(reference_tokens), Counter(hypothesis_tokens)
    shared = {
        token
        for token in reference_counts
        if reference_counts[token] == 1 and hypothesis_counts[token] == 1
    }
    ordered = [token for token in reference_tokens if token in shared]
    if len(ordered) < 2:
        return float(reference_tokens == hypothesis_tokens)
    positions = {token: index for index, token in enumerate(hypothesis_tokens) if token in shared}
    correct = 0
    pairs = 0
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            pairs += 1
            correct += positions[ordered[left]] < positions[ordered[right]]
    return correct / pairs


def block_scores(reference: str, hypothesis: str) -> dict[str, float]:
    reference_blocks = Counter(_blocks(reference))
    hypothesis_blocks = Counter(_blocks(hypothesis))
    overlap = sum((reference_blocks & hypothesis_blocks).values())
    precision = overlap / sum(hypothesis_blocks.values()) if hypothesis_blocks else 0.0
    recall = overlap / sum(reference_blocks.values()) if reference_blocks else float(not hypothesis_blocks)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "block_precision": precision,
        "block_recall": recall,
        "block_structure_f1": f1,
    }


def timestamp_mae(reference: Sequence[float], hypothesis: Sequence[float]) -> float:
    if len(reference) != len(hypothesis) or not reference:
        raise ValueError("Timestamp arrays must be non-empty and have equal length")
    return sum(abs(left - right) for left, right in zip(reference, hypothesis)) / len(reference)


def duplicate_line_rate(text: str) -> float:
    lines = _blocks(text)
    return (len(lines) - len(set(lines))) / len(lines) if lines else 0.0


def _blocks(text: str) -> list[str]:
    return [" ".join(line.casefold().split()) for line in text.splitlines() if line.strip()]


def structured_document_scores(
    reference_elements: object, document: ExtractedDocument
) -> dict[str, float]:
    """Score only element types explicitly annotated for this sample.

    Manifest elements use ``kind: table`` with ``rows`` and ``kind: formula``
    with ``latex``.  Metrics are omitted when that element type has no reference
    annotation, so plain-text pages cannot inflate structural quality.
    """
    if not isinstance(reference_elements, Sequence) or isinstance(
        reference_elements, (str, bytes)
    ):
        return {}
    references = [item for item in reference_elements if isinstance(item, Mapping)]
    result: dict[str, float] = {}
    tables = [item for item in references if item.get("kind") == "table"]
    formulas = [item for item in references if item.get("kind") == "formula"]
    if tables:
        predicted_tables = [
            segment for segment in document.segments if segment.kind is SegmentKind.TABLE
        ]
        table_matches = _greedy_matches(
            tables,
            predicted_tables,
            lambda reference, prediction: _table_content_f1(
                _reference_rows(reference), _prediction_rows(prediction.structured_content)
            ),
        )
        similarities = [score for _, _, score in table_matches]
        detected = sum(score >= 0.5 for score in similarities)
        precision, recall, f1 = _prf(detected, len(predicted_tables), len(tables))
        result.update(
            {
                "table_detection_precision": precision,
                "table_detection_recall": recall,
                "table_detection_f1": f1,
                "table_content_f1": sum(similarities) / len(tables),
                "table_structure_f1": sum(
                    _table_relation_f1(
                        _reference_rows(tables[reference_index]),
                        _prediction_rows(predicted_tables[prediction_index].structured_content),
                    )
                    for reference_index, prediction_index, _ in table_matches
                )
                / len(tables),
            }
        )
    if formulas:
        predicted_formulas = [
            segment for segment in document.segments if segment.kind is SegmentKind.FORMULA
        ]
        formula_matches = _greedy_matches(
            formulas,
            predicted_formulas,
            lambda reference, prediction: _latex_similarity(
                str(reference.get("latex", "")),
                str(prediction.structured_content.get("latex", prediction.text)),
            ),
        )
        similarities = [score for _, _, score in formula_matches]
        detected = sum(score >= 0.5 for score in similarities)
        precision, recall, f1 = _prf(detected, len(predicted_formulas), len(formulas))
        result.update(
            {
                "formula_detection_precision": precision,
                "formula_detection_recall": recall,
                "formula_detection_f1": f1,
                "formula_latex_similarity": sum(similarities) / len(formulas),
                "formula_exact_match": sum(score == 1.0 for score in similarities)
                / len(formulas),
            }
        )
    return result


def _greedy_matches(references, predictions, similarity):
    candidates = sorted(
        (
            (float(similarity(reference, prediction)), reference_index, prediction_index)
            for reference_index, reference in enumerate(references)
            for prediction_index, prediction in enumerate(predictions)
        ),
        reverse=True,
    )
    used_references: set[int] = set()
    used_predictions: set[int] = set()
    result: list[tuple[int, int, float]] = []
    for score, reference_index, prediction_index in candidates:
        if reference_index in used_references or prediction_index in used_predictions:
            continue
        used_references.add(reference_index)
        used_predictions.add(prediction_index)
        result.append((reference_index, prediction_index, score))
    return result


def _reference_rows(reference: Mapping[str, object]) -> list[list[str]]:
    rows = reference.get("rows", [])
    return _rows(rows)


def _prediction_rows(content: Mapping[str, object]) -> list[list[str]]:
    return _rows(content.get("rows", []))


def _rows(value: object) -> list[list[str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [
        [str(cell) for cell in row]
        for row in value
        if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
    ]


def _table_content_f1(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    reference_tokens = Counter(
        token for row in reference for cell in row for token in normalized_tokens(cell)
    )
    hypothesis_tokens = Counter(
        token for row in hypothesis for cell in row for token in normalized_tokens(cell)
    )
    overlap = sum((reference_tokens & hypothesis_tokens).values())
    precision = overlap / sum(hypothesis_tokens.values()) if hypothesis_tokens else 0.0
    recall = overlap / sum(reference_tokens.values()) if reference_tokens else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _table_relation_f1(reference: list[list[str]], hypothesis: list[list[str]]) -> float:
    """Compare row/column adjacency relations after cell-text canonicalization."""
    reference_relations = _relations(reference)
    hypothesis_relations = _relations(hypothesis)
    overlap = len(reference_relations & hypothesis_relations)
    precision = overlap / len(hypothesis_relations) if hypothesis_relations else 0.0
    recall = overlap / len(reference_relations) if reference_relations else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _relations(rows: list[list[str]]) -> set[tuple[str, str, str]]:
    normalized = [[" ".join(normalized_tokens(cell)) for cell in row] for row in rows]
    relations: set[tuple[str, str, str]] = set()
    for row in normalized:
        relations.update(("row", left, right) for left, right in zip(row, row[1:]))
    width = max((len(row) for row in normalized), default=0)
    for column in range(width):
        values = [row[column] for row in normalized if column < len(row)]
        relations.update(("column", top, bottom) for top, bottom in zip(values, values[1:]))
    return relations


def _latex_similarity(reference: str, hypothesis: str) -> float:
    left, right = _normalize_latex(reference), _normalize_latex(hypothesis)
    if not left and not right:
        return 1.0
    return 1.0 - levenshtein(list(left), list(right)) / max(1, len(left), len(right))


def _normalize_latex(value: str) -> str:
    value = value.strip()
    if (value.startswith("$$") and value.endswith("$$")) or (
        value.startswith(r"\[") and value.endswith(r"\]")
    ):
        value = value[2:-2]
    return re.sub(r"\s+", "", value)


def _prf(true_positive: int, predicted: int, reference: int) -> tuple[float, float, float]:
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
