"""Document-extraction metric contract used by the executable benchmark."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from edumind.common.artifacts import stable_hash
from edumind.common.paths import PROJECT_ROOT
from edumind.extraction import ExtractedDocument, ExtractedSegment, SegmentKind
from experiments.benchmarks.common.metrics import (
    character_error_rate,
    normalize_prose,
    normalized_tokens,
    precision_recall_f1,
    word_error_rate,
)
from .official_metrics import score_official_metrics, validate_official_runtime

METRIC_DIRECTIONS = {
    "text.content_precision": "max",
    "text.content_recall": "max",
    "text.content_f1": "max",
    "text.character_error_rate": "min",
    "text.word_error_rate": "min",
    "text.reading_order_accuracy": "max",
    "pages.page_coverage": "max",
    "pages.page_content_f1": "max",
    "pages.page_attribution_accuracy": "max",
    "pages.duplicate_page_rate": "min",
    "layout.element_precision": "max",
    "layout.element_recall": "max",
    "layout.element_f1": "max",
    "layout.element_type_accuracy": "max",
    "layout.hierarchy_accuracy": "max",
    "layout.mean_bounding_box_iou": "max",
    "tables.detection_precision": "max",
    "tables.detection_recall": "max",
    "tables.detection_f1": "max",
    "tables.content_precision": "max",
    "tables.content_recall": "max",
    "tables.content_f1": "max",
    "tables.teds": "max",
    "tables.teds_s": "max",
    "formulas.detection_precision": "max",
    "formulas.detection_recall": "max",
    "formulas.detection_f1": "max",
    "formulas.recognition_similarity": "max",
    "formulas.exact_match": "max",
    "reliability.empty_output_rate": "min",
    "reliability.duplicate_content_rate": "min",
    "reliability.structured_output_determinism": "max",
    "reliability.candidate_failure_rate": "min",
    "operational.first_item_latency_seconds": "min",
    "operational.p50_warm_latency_per_page_seconds": "min",
    "operational.p95_warm_latency_per_page_seconds": "min",
    "operational.p50_complete_document_latency_seconds": "min",
    "operational.p95_complete_document_latency_seconds": "min",
    "operational.batch_pages_per_minute": "max",
    "operational.peak_process_tree_ram_mb": "min",
    "operational.peak_vram_mb": "min",
    "operational.peak_temporary_disk_mb": "min",
}

VISUAL_KINDS = {"image", "pdf"}
REFERENCE_CAPABILITIES = frozenset(
    {
        "text",
        "pages",
        "reading_order",
        "layout_boxes",
        "element_types",
        "hierarchy",
        "tables",
        "formulas",
    }
)
LAYOUT_KINDS = {
    SegmentKind.TEXT,
    SegmentKind.TITLE,
    SegmentKind.HEADING,
    SegmentKind.LIST_ITEM,
    SegmentKind.CAPTION,
    SegmentKind.FIGURE,
    SegmentKind.CODE,
    SegmentKind.PAGE_HEADER,
    SegmentKind.PAGE_FOOTER,
}


@dataclass(frozen=True)
class ReferenceElement:
    element_id: str
    kind: SegmentKind
    text: str = ""
    order: int = 0
    page_number: int | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    parent_id: str | None = None
    hierarchy_level: int | None = None
    html: str | None = None
    latex: str | None = None


@dataclass(frozen=True)
class ReferenceDocument:
    text: str
    pages: Mapping[int, str]
    elements: tuple[ReferenceElement, ...]
    capabilities: frozenset[str]


@dataclass
class DocumentEvaluation:
    groups: tuple[str, ...]
    metrics: dict[str, float] = field(default_factory=dict)
    counts: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    table_content_scores: list[tuple[float, float, float]] = field(default_factory=list)
    table_pairs: list[tuple[str, str]] = field(default_factory=list)
    table_scores: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    formula_pairs: list[tuple[str, str]] = field(default_factory=list)
    formula_scores: list[float] = field(default_factory=list)


def load_reference(item: Mapping[str, object]) -> ReferenceDocument:
    return load_reference_data(item)[1]


def load_reference_data(
    item: Mapping[str, object],
) -> tuple[Mapping[str, object], ReferenceDocument]:
    payload = _reference_payload(item)
    return payload, _reference_from_mapping(payload)


def validate_reference(
    item: Mapping[str, object],
    *,
    authoritative: bool,
    payload: Mapping[str, object] | None = None,
    reference: ReferenceDocument | None = None,
) -> None:
    if payload is None:
        payload = _reference_payload(item)
    if reference is None:
        reference = _reference_from_mapping(payload)
    raw_capabilities = payload.get("reference_capabilities")
    if authoritative and raw_capabilities is None:
        raise ValueError(
            f"Authoritative document sample {item.get('id')} requires "
            "reference_capabilities"
        )
    if raw_capabilities is not None:
        _validate_capability_list(raw_capabilities, item.get("id"))
    if "text" in reference.capabilities and not reference.text.strip():
        raise ValueError(f"Document sample {item.get('id')} has no verified reference text")
    if authoritative and not item.get("reference_path"):
        raise ValueError(
            f"Authoritative document sample {item.get('id')} requires reference_path"
        )
    kind = str(item.get("kind"))
    if "pages" in reference.capabilities and not reference.pages:
        raise ValueError(f"Visual sample {item.get('id')} requires verified page text")
    layout_capabilities = {
        "reading_order", "layout_boxes", "element_types", "hierarchy"
    }
    if reference.capabilities & layout_capabilities and not reference.elements:
        raise ValueError(
            f"Document sample {item.get('id')} claims layout capabilities without elements"
        )
    if "tables" in reference.capabilities:
        table_presence = _presence(payload, item, "has_table")
        if table_presence is None and authoritative:
            raise ValueError(
                f"Document sample {item.get('id')} with table annotations must set has_table"
            )
        table_elements = [
            element for element in reference.elements if element.kind is SegmentKind.TABLE
        ]
        if table_presence and not table_elements:
            raise ValueError(
                f"Document sample {item.get('id')} sets has_table:true without table references"
            )
        if table_presence is False and table_elements:
            raise ValueError(
                f"Document sample {item.get('id')} sets has_table:false with table references"
            )
        missing_table_html = [
            element.element_id
            for element in table_elements
            if not element.html
        ]
        if missing_table_html:
            raise ValueError(
                f"Document sample {item.get('id')} lacks official table references: "
                + ", ".join(missing_table_html[:10])
            )
    if "formulas" in reference.capabilities:
        formula_presence = _presence(payload, item, "has_formula")
        if formula_presence is None and authoritative:
            raise ValueError(
                f"Document sample {item.get('id')} with formula annotations must set has_formula"
            )
        formula_elements = [
            element for element in reference.elements if element.kind is SegmentKind.FORMULA
        ]
        if formula_presence and not formula_elements:
            raise ValueError(
                f"Document sample {item.get('id')} sets has_formula:true without formula references"
            )
        if formula_presence is False and formula_elements:
            raise ValueError(
                f"Document sample {item.get('id')} sets has_formula:false with formula references"
            )
        missing_formula_latex = [
            element.element_id
            for element in formula_elements
            if not element.latex
        ]
        if missing_formula_latex:
            raise ValueError(
                f"Document sample {item.get('id')} lacks official formula references: "
                + ", ".join(missing_formula_latex[:10])
            )
    if "hierarchy" in reference.capabilities and not any(
        element.parent_id is not None or element.hierarchy_level is not None
        for element in reference.elements
    ):
        raise ValueError(
            f"Document sample {item.get('id')} claims hierarchy without hierarchy annotations"
        )
    if "layout_boxes" in reference.capabilities:
        missing_boxes = [
            element.element_id
            for element in reference.elements
            if element.kind in LAYOUT_KINDS
            if element.bounding_box is None
        ]
        if missing_boxes:
            raise ValueError(
                f"Visual sample {item.get('id')} has elements without normalized boxes: "
                + ", ".join(missing_boxes[:10])
            )
    page_matched_kinds = set(LAYOUT_KINDS)
    if "tables" in reference.capabilities:
        page_matched_kinds.add(SegmentKind.TABLE)
    if "formulas" in reference.capabilities:
        page_matched_kinds.add(SegmentKind.FORMULA)
    if kind in VISUAL_KINDS and any(
        element.page_number is None
        for element in reference.elements
        if element.kind in page_matched_kinds
    ):
        raise ValueError(
            f"Visual sample {item.get('id')} has matched elements without page numbers"
        )


def validate_official_evaluators(references: Sequence[ReferenceDocument]) -> bool:
    tables = any(
        "tables" in reference.capabilities
        and any(element.kind is SegmentKind.TABLE for element in reference.elements)
        for reference in references
    )
    formulas = any(
        "formulas" in reference.capabilities
        and any(element.kind is SegmentKind.FORMULA for element in reference.elements)
        for reference in references
    )
    validate_official_runtime(tables=tables, formulas=formulas)
    return tables or formulas


def apply_official_metrics(records: Sequence[DocumentEvaluation]) -> None:
    """Batch all official table/formula scoring into one Docker invocation."""

    table_pairs = [pair for record in records for pair in record.table_pairs]
    formula_pairs = [pair for record in records for pair in record.formula_pairs]
    table_results, formula_results = score_official_metrics(table_pairs, formula_pairs)
    table_offset = 0
    formula_offset = 0
    for record in records:
        table_count = len(record.table_pairs)
        if table_count:
            official = table_results[table_offset : table_offset + table_count]
            record.table_scores = [
                (*content, teds, teds_s)
                for content, (teds, teds_s) in zip(record.table_content_scores, official)
            ]
            record.metrics["tables.teds"] = float(
                np.mean([value[3] for value in record.table_scores])
            )
            record.metrics["tables.teds_s"] = float(
                np.mean([value[4] for value in record.table_scores])
            )
            table_offset += table_count
        formula_count = len(record.formula_pairs)
        if formula_count:
            record.formula_scores = formula_results[
                formula_offset : formula_offset + formula_count
            ]
            record.metrics["formulas.recognition_similarity"] = float(
                np.mean(record.formula_scores)
            )
            record.metrics["formulas.exact_match"] = float(
                np.mean([value == 1.0 for value in record.formula_scores])
            )
            formula_offset += formula_count


def score_document(
    item: Mapping[str, object],
    document: ExtractedDocument | None,
    *,
    reference: ReferenceDocument | None = None,
    repeated_documents: Sequence[ExtractedDocument] = (),
    failed: bool = False,
) -> DocumentEvaluation:
    reference = reference or load_reference(item)
    kind = str(item["kind"])
    groups = _document_groups(item)
    result = DocumentEvaluation(groups)
    hypothesis = document.text if document else ""
    if "text" in reference.capabilities:
        result.metrics.update(_text_metrics(reference, hypothesis))
    predicted = tuple(document.segments) if document else ()

    if "pages" in reference.capabilities:
        result.metrics.update(_page_metrics(reference, predicted))
        # Page attribution follows content matches and then checks the page label.
        # Matching by box alone could count unrelated text at the same coordinates.
        page_matches = _match_elements(reference.elements, predicted, visual=False)
        attributed = [
            (reference.elements[left], predicted[right])
            for left, right, _ in page_matches
            if reference.elements[left].page_number is not None
            and predicted[right].page_number is not None
        ]
        if attributed:
            result.metrics["pages.page_attribution_accuracy"] = sum(
                expected.page_number == observed.page_number
                for expected, observed in attributed
            ) / len(attributed)

    layout_references = tuple(
        element for element in reference.elements if element.kind in LAYOUT_KINDS
    )
    layout_predictions = tuple(
        segment
        for segment in predicted
        if segment.kind in LAYOUT_KINDS
        and (segment.text.strip() or segment.bounding_box is not None)
    )
    layout_enabled = bool(
        reference.capabilities
        & {"reading_order", "layout_boxes", "element_types", "hierarchy"}
    )
    if layout_enabled:
        matches = _match_elements(
            layout_references,
            layout_predictions,
            visual=kind in VISUAL_KINDS,
            use_boxes="layout_boxes" in reference.capabilities,
        )
        result.counts["layout"] = (
            len(matches),
            len(layout_predictions) - len(matches),
            len(layout_references) - len(matches),
        )
        result.metrics.update(
            _layout_metrics(
                layout_references,
                layout_predictions,
                matches,
                capabilities=reference.capabilities,
            )
        )
        if "reading_order" in reference.capabilities:
            reading_order = _reading_order(matches, layout_references, layout_predictions)
            if reading_order is not None:
                result.metrics["text.reading_order_accuracy"] = reading_order

    if "tables" in reference.capabilities:
        _score_structured_kind(result, reference, predicted, SegmentKind.TABLE, kind)
    if "formulas" in reference.capabilities:
        _score_structured_kind(result, reference, predicted, SegmentKind.FORMULA, kind)
    result.metrics["reliability.empty_output_rate"] = float(not hypothesis.strip())
    if "text" in reference.capabilities:
        duplicate_rate = _duplicate_content_rate(reference.text, hypothesis)
        if duplicate_rate is not None:
            result.metrics["reliability.duplicate_content_rate"] = duplicate_rate
    result.metrics["reliability.candidate_failure_rate"] = float(failed)
    if failed:
        result.metrics["reliability.structured_output_determinism"] = 0.0
    elif repeated_documents:
        fingerprints = {_document_fingerprint(value) for value in repeated_documents}
        result.metrics["reliability.structured_output_determinism"] = float(
            len(fingerprints) == 1
        )
    return result


def aggregate_evaluations(
    records: Sequence[DocumentEvaluation],
    *,
    resamples: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    metrics: dict[str, float] = {}
    intervals: dict[str, dict[str, float]] = {}
    group_names = sorted(set().union(*(record.groups for record in records)))
    for group in (None, *group_names):
        selected = [record for record in records if group is None or group in record.groups]
        if not selected:
            continue
        estimates = _aggregate_group(selected)
        prefix = "" if group is None else f"{group}."
        for name, value in estimates.items():
            qualified = _grouped_name(name, prefix)
            metrics[qualified] = value
            metrics[f"{qualified}.sample_count"] = float(
                sum(_contributes(record, name) for record in selected)
            )
        if resamples and len(selected) >= 2:
            rng = np.random.default_rng(seed)
            draws: dict[str, list[float]] = {name: [] for name in estimates}
            for _ in range(resamples):
                sample = [selected[index] for index in rng.integers(0, len(selected), len(selected))]
                values = _aggregate_group(sample)
                for name in estimates:
                    if name in values:
                        draws[name].append(values[name])
            for name, values in draws.items():
                if len(values) < 2:
                    continue
                qualified = _grouped_name(name, prefix)
                intervals[qualified] = {
                    "estimate": metrics[qualified],
                    "lower": float(np.quantile(values, 0.025)),
                    "upper": float(np.quantile(values, 0.975)),
                    "confidence": 0.95,
                    "defined_resamples": float(len(values)),
                }
    return metrics, intervals


def _reference_from_mapping(payload: Mapping[str, object]) -> ReferenceDocument:
    text = _canonical(str(payload.get("text", payload.get("reference", ""))))
    raw_pages = payload.get("pages", payload.get("reference_page_texts", []))
    pages: dict[int, str] = {}
    if isinstance(raw_pages, Mapping):
        pages = {int(key): _canonical(str(value)) for key, value in raw_pages.items()}
    elif isinstance(raw_pages, Sequence) and not isinstance(raw_pages, (str, bytes)):
        for index, value in enumerate(raw_pages, 1):
            if isinstance(value, Mapping):
                pages[int(value.get("page_number", index))] = _canonical(
                    str(value.get("text", ""))
                )
            else:
                pages[index] = _canonical(str(value))
    raw_elements = payload.get("elements", payload.get("reference_elements", []))
    elements: list[ReferenceElement] = []
    if isinstance(raw_elements, Sequence) and not isinstance(raw_elements, (str, bytes)):
        for index, value in enumerate(raw_elements):
            if not isinstance(value, Mapping):
                continue
            kind = _kind(value.get("kind", "text"))
            elements.append(
                ReferenceElement(
                    element_id=str(value.get("id", value.get("element_id", f"ref-{index}"))),
                    kind=kind,
                    text=_canonical(str(value.get("text", ""))),
                    order=int(value.get("order", index)),
                    page_number=_optional_int(value.get("page_number")),
                    bounding_box=_box(value.get("bounding_box")),
                    parent_id=_optional_string(value.get("parent_id")),
                    hierarchy_level=_optional_int(value.get("hierarchy_level")),
                    html=_optional_string(value.get("html")),
                    latex=_optional_string(value.get("latex")),
                )
            )
    if not pages and payload.get("kind") == "image":
        pages[1] = text
    raw_capabilities = payload.get("reference_capabilities")
    capabilities = (
        frozenset(str(value) for value in raw_capabilities)
        if isinstance(raw_capabilities, Sequence)
        and not isinstance(raw_capabilities, (str, bytes))
        else _infer_capabilities(payload, text, pages, elements)
    )
    return ReferenceDocument(text, pages, tuple(elements), capabilities)


def _text_metrics(reference: ReferenceDocument, hypothesis: str) -> dict[str, float]:
    expected = normalize_prose(_canonical(reference.text))
    observed = normalize_prose(_canonical(hypothesis))
    precision, recall, f1 = _content_scores(expected, observed)
    return {
        "text.content_precision": precision,
        "text.content_recall": recall,
        "text.content_f1": f1,
        "text.character_error_rate": character_error_rate(expected, observed),
        "text.word_error_rate": word_error_rate(expected, observed),
    }


def _page_metrics(
    reference: ReferenceDocument, predicted: Sequence[ExtractedSegment]
) -> dict[str, float]:
    predicted_pages: dict[int, str] = {}
    for segment in predicted:
        if segment.page_number is None or not segment.text.strip():
            continue
        predicted_pages.setdefault(segment.page_number, "")
        predicted_pages[segment.page_number] += ("\n" if predicted_pages[segment.page_number] else "") + segment.text
    page_ids = sorted(set(reference.pages) | set(predicted_pages))
    page_f1 = {
        page: _content_f1(reference.pages.get(page, ""), predicted_pages.get(page, ""))
        for page in page_ids
    }
    coverage = sum(page_f1.get(page, 0.0) > 0.0 for page in reference.pages) / len(reference.pages)
    predicted_texts = [value for value in predicted_pages.values() if value.strip()]
    reference_texts = [value for value in reference.pages.values() if value.strip()]
    duplicates = _unsupported_near_duplicates(predicted_texts, reference_texts)
    result = {
        "pages.page_coverage": coverage,
        "pages.page_content_f1": float(np.mean(list(page_f1.values()))) if page_f1 else 0.0,
    }
    if predicted_pages:
        result["pages.duplicate_page_rate"] = duplicates / len(predicted_pages)
    return result


def _unsupported_near_duplicates(predicted: Sequence[str], reference: Sequence[str]) -> int:
    groups: list[list[str]] = []
    for text in predicted:
        for group in groups:
            if _content_f1(group[0], text) >= 0.95:
                group.append(text)
                break
        else:
            groups.append([text])
    return sum(
        max(
            0,
            len(group)
            - max(
                1,
                sum(
                    _content_f1(group[0], expected) >= 0.95
                    for expected in reference
                ),
            ),
        )
        for group in groups
    )


def _layout_metrics(
    references, predictions, matches, *, capabilities: frozenset[str]
) -> dict[str, float]:
    tp, fp, fn = len(matches), len(predictions) - len(matches), len(references) - len(matches)
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    result = {
        "layout.element_precision": precision,
        "layout.element_recall": recall,
        "layout.element_f1": f1,
    }
    if matches and "element_types" in capabilities:
        result["layout.element_type_accuracy"] = sum(
            references[left].kind is predictions[right].kind for left, right, _ in matches
        ) / len(matches)
    if matches and "hierarchy" in capabilities:
        hierarchy = []
        reference_to_prediction = {
            references[left].element_id: predictions[right].element_id
            for left, right, _ in matches
        }
        for left, right, _ in matches:
            reference = references[left]
            prediction = predictions[right]
            if reference.parent_id is None and reference.hierarchy_level is None:
                continue
            predicted_level = _optional_int(prediction.metadata.get("hierarchy_level"))
            parent_correct = (
                reference.parent_id is None
                or reference_to_prediction.get(reference.parent_id) == prediction.parent_id
            )
            level_correct = (
                reference.hierarchy_level is None
                or reference.hierarchy_level == predicted_level
            )
            hierarchy.append(float(parent_correct and level_correct))
        if hierarchy:
            result["layout.hierarchy_accuracy"] = float(np.mean(hierarchy))
    if matches and "layout_boxes" in capabilities:
        boxes = [score for left, right, score in matches if references[left].bounding_box and predictions[right].bounding_box]
        if boxes:
            result["layout.mean_bounding_box_iou"] = float(np.mean(boxes))
    return result


def _score_structured_kind(result, reference, predicted, target, source_kind) -> None:
    references = tuple(element for element in reference.elements if element.kind is target)
    predictions = tuple(segment for segment in predicted if segment.kind is target)
    annotation_key = "tables" if target is SegmentKind.TABLE else "formulas"
    # This function is reached only when the explicit capability is present.
    # Keep a count row even for a verified negative stored in a reference file so
    # false positives contribute to pooled detection metrics.
    result.counts[annotation_key] = (
        0, len(predictions), len(references)
    )
    if not references and not predictions:
        return
    matches = _match_elements(references, predictions, visual=source_kind in VISUAL_KINDS)
    result.counts[annotation_key] = (
        len(matches), len(predictions) - len(matches), len(references) - len(matches)
    )
    precision, recall, f1 = precision_recall_f1(*result.counts[annotation_key])
    result.metrics.update(
        {
            f"{annotation_key}.detection_precision": precision,
            f"{annotation_key}.detection_recall": recall,
            f"{annotation_key}.detection_f1": f1,
        }
    )
    by_reference = {left: right for left, right, _ in matches}
    if target is SegmentKind.TABLE and references:
        for index, item in enumerate(references):
            prediction = predictions[by_reference[index]] if index in by_reference else None
            content = _content_scores(item.text, prediction.text if prediction else "")
            result.table_content_scores.append(content)
            result.table_pairs.append((item.html or "", _table_html(prediction)))
        for name, index in (
            ("tables.content_precision", 0),
            ("tables.content_recall", 1),
            ("tables.content_f1", 2),
        ):
            result.metrics[name] = float(
                np.mean([value[index] for value in result.table_content_scores])
            )
    if target is SegmentKind.FORMULA and references:
        for index, item in enumerate(references):
            prediction = predictions[by_reference[index]] if index in by_reference else None
            result.formula_pairs.append(
                (item.latex or item.text, _formula_latex(prediction))
            )


def _aggregate_group(records: Sequence[DocumentEvaluation]) -> dict[str, float]:
    names = sorted(set().union(*(record.metrics for record in records)))
    values = {
        name: float(np.mean([record.metrics[name] for record in records if name in record.metrics]))
        for name in names
    }
    for category in ("layout", "tables", "formulas"):
        counts = [record.counts[category] for record in records if category in record.counts]
        if not counts:
            continue
        tp, fp, fn = (sum(value[index] for value in counts) for index in range(3))
        if tp + fp + fn == 0:
            continue
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        values[f"{category}.detection_precision" if category != "layout" else "layout.element_precision"] = precision
        values[f"{category}.detection_recall" if category != "layout" else "layout.element_recall"] = recall
        values[f"{category}.detection_f1" if category != "layout" else "layout.element_f1"] = f1
    table_scores = [score for record in records for score in record.table_scores]
    if table_scores:
        for name, index in (
            ("tables.content_precision", 0),
            ("tables.content_recall", 1),
            ("tables.content_f1", 2),
            ("tables.teds", 3),
            ("tables.teds_s", 4),
        ):
            values[name] = float(np.mean([value[index] for value in table_scores]))
    formula_scores = [score for record in records for score in record.formula_scores]
    if formula_scores:
        values["formulas.recognition_similarity"] = float(np.mean(formula_scores))
        values["formulas.exact_match"] = float(np.mean([value == 1.0 for value in formula_scores]))
    return values


def _match_elements(
    references, predictions, *, visual: bool, use_boxes: bool | None = None
):
    if not references or not predictions:
        return []
    scores = np.zeros((len(references), len(predictions)), dtype=np.float64)
    for left, reference in enumerate(references):
        for right, prediction in enumerate(predictions):
            if visual and reference.page_number != prediction.page_number:
                continue
            if visual and use_boxes is not False and reference.bounding_box:
                scores[left, right] = (
                    _iou(reference.bounding_box, prediction.bounding_box)
                    if prediction.bounding_box
                    else 0.0
                )
            else:
                scores[left, right] = _content_f1(reference.text, prediction.text)
    try:
        from scipy.optimize import linear_sum_assignment
    except ModuleNotFoundError as exc:
        raise RuntimeError("scipy is required for one-to-one document-element matching") from exc
    rows, columns = linear_sum_assignment(-scores)
    threshold = 0.5
    return [
        (int(left), int(right), float(scores[left, right]))
        for left, right in zip(rows, columns)
        if scores[left, right] >= threshold
    ]


def _reading_order(matches, references, predictions) -> float | None:
    if len(matches) < 2:
        return None
    ordered = sorted(matches, key=lambda value: references[value[0]].order)
    correct = 0
    total = 0
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            total += 1
            correct += (predictions[ordered[left][1]].order or 0) < (predictions[ordered[right][1]].order or 0)
    return correct / total


def _document_groups(item: Mapping[str, object]) -> tuple[str, ...]:
    broad = str(item["kind"])
    detailed = str(item.get("document_group") or item.get("document_family") or "")
    detailed = detailed.strip().casefold().replace("-", "_").replace(" ", "_")
    if detailed and not detailed.startswith(f"{broad}_") and detailed != broad:
        detailed = f"{broad}_{detailed}"
    return tuple(dict.fromkeys(value for value in (broad, detailed) if value))


def _grouped_name(name: str, prefix: str) -> str:
    if not prefix:
        return name
    category, metric = name.split(".", 1)
    return f"{category}.{prefix}{metric}"


def _contributes(record: DocumentEvaluation, name: str) -> bool:
    if name in record.metrics:
        return True
    for prefix, category in (
        ("layout.element_", "layout"),
        ("tables.detection_", "tables"),
        ("formulas.detection_", "formulas"),
    ):
        if name.startswith(prefix):
            return category in record.counts
    return False


def _content_f1(reference: str, prediction: str) -> float:
    return _content_scores(_canonical(reference), _canonical(prediction))[2]


def _content_scores(reference: str, prediction: str) -> tuple[float, float, float]:
    left = Counter(normalized_tokens(reference))
    right = Counter(normalized_tokens(prediction))
    overlap = sum((left & right).values())
    return precision_recall_f1(
        overlap,
        sum(right.values()) - overlap,
        sum(left.values()) - overlap,
    )


def _duplicate_content_rate(reference: str, prediction: str) -> float | None:
    predicted = normalized_tokens(prediction)
    if not predicted:
        return None
    predicted_counts = Counter(predicted)
    extra = predicted_counts - Counter(normalized_tokens(reference))
    repeated = sum(count for token, count in extra.items() if predicted_counts[token] > 1)
    return repeated / len(predicted)


def _document_fingerprint(document: ExtractedDocument) -> str:
    return stable_hash(
        {
            "text": document.text,
            "segments": [
                {
                    "text": segment.text,
                    "id": segment.element_id,
                    "parent": segment.parent_id,
                    "order": segment.order,
                    "page": segment.page_number,
                    "box": segment.bounding_box,
                    "kind": segment.kind.value,
                    "structured": segment.structured_content,
                }
                for segment in document.segments
            ],
        }
    )


def _table_html(segment: ExtractedSegment | None) -> str:
    return str(segment.structured_content.get("html", "")) if segment else ""


def _formula_latex(segment: ExtractedSegment | None) -> str:
    return str(segment.structured_content.get("latex", segment.text)) if segment else ""


def _iou(left, right) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _canonical(value: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


def _kind(value: object) -> SegmentKind:
    try:
        return SegmentKind(str(value))
    except ValueError as exc:
        raise ValueError(f"Unknown reference element kind: {value}") from exc


def _box(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        return None
    result = tuple(float(item) for item in value)
    if not all(0.0 <= item <= 1.0 for item in result):
        raise ValueError("Reference boxes must use normalized coordinates")
    return result  # type: ignore[return-value]


def _optional_int(value: object) -> int | None:
    return None if value in (None, "") else int(value)


def _optional_string(value: object) -> str | None:
    return None if value in (None, "") else str(value)


def _reference_payload(item: Mapping[str, object]) -> Mapping[str, object]:
    reference_path = item.get("reference_path")
    if not reference_path:
        return item
    path = PROJECT_ROOT / str(reference_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Reference document must be a JSON object: {path}")
    return payload


def _validate_capability_list(value: object, sample_id: object) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or not all(isinstance(item, str) for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(
            f"Document sample {sample_id} reference_capabilities must be a non-empty "
            "unique string list"
        )
    unknown = sorted(set(value) - REFERENCE_CAPABILITIES)
    if unknown:
        raise ValueError(
            f"Document sample {sample_id} has unknown reference capabilities: "
            + ", ".join(unknown)
        )


def _infer_capabilities(payload, text, pages, elements) -> frozenset[str]:
    capabilities: set[str] = set()
    if text.strip():
        capabilities.add("text")
    if pages:
        capabilities.add("pages")
    if elements:
        capabilities.add("element_types")
    if len(elements) >= 2:
        capabilities.add("reading_order")
    if any(element.bounding_box is not None for element in elements):
        capabilities.add("layout_boxes")
    if any(
        element.parent_id is not None or element.hierarchy_level is not None
        for element in elements
    ):
        capabilities.add("hierarchy")
    if any(element.kind is SegmentKind.TABLE for element in elements) or "has_table" in payload:
        capabilities.add("tables")
    if any(element.kind is SegmentKind.FORMULA for element in elements) or "has_formula" in payload:
        capabilities.add("formulas")
    return frozenset(capabilities)


def _presence(
    payload: Mapping[str, object], item: Mapping[str, object], key: str
) -> bool | None:
    value = payload.get(key, item.get(key))
    return value if isinstance(value, bool) else None
