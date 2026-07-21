"""Transparent table and formula metrics for complete-document extraction."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence

from edumind.extraction import ExtractedDocument, SegmentKind

from .metrics import levenshtein, normalized_tokens


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
    """Compare row/column adjacency relations after cell-text normalization."""
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
