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
    levenshtein,
    normalize_prose,
    normalized_tokens,
    precision_recall_f1,
    word_error_rate,
)
from experiments.benchmarks.extraction.scoring import (
    attempt_rates,
    bootstrap_sources,
    source_id,
)

from .official_metrics import score_official_metrics, validate_official_runtime

METRIC_DIRECTIONS = {
    "text.content_precision": "max",
    "text.content_recall": "max",
    "text.content_f1": "max",
    "text.character_error_rate": "min",
    "text.word_error_rate": "min",
    "text.reading_order_ned": "min",
    "pages.page_coverage": "max",
    "pages.page_content_f1": "max",
    "pages.page_attribution_recall": "max",
    "pages.duplicate_page_rate": "min",
    "layout.element_precision": "max",
    "layout.element_recall": "max",
    "layout.element_f1": "max",
    "layout.element_type_recall": "max",
    "layout.hierarchy_preservation_rate": "max",
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
    "reliability.unexpected_empty_output_rate": "min",
    "reliability.duplicate_content_rate": "min",
    "reliability.repeatability_success_rate": "max",
    "reliability.attempt_failure_rate": "min",
    "operational.cold_model_load_seconds": "min",
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
    parent_annotated: bool = False
    level_annotated: bool = False
    identity: str | None = None
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
    metrics: dict[str, float | None] = field(default_factory=dict)
    statuses: dict[str, dict[str, str | None]] = field(default_factory=dict)
    sample_id: str = ""
    source_group_id: str = ""
    counts: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    table_content_scores: list[tuple[float, float, float]] = field(default_factory=list)
    table_pairs: list[tuple[str, str]] = field(default_factory=list)
    table_scores: list[tuple[float, float, float, float, float]] = field(
        default_factory=list
    )
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
    if authoritative:
        raw = payload.get("elements", payload.get("reference_elements", []))
        for element in raw:
            if (
                not isinstance(element, Mapping)
                or not isinstance(element.get("id", element.get("element_id")), str)
                or not element.get("id", element.get("element_id"))
            ):
                raise ValueError(
                    "Authoritative reference elements require stable nonempty IDs"
                )
            if (
                "reading_order" in reference.capabilities
                and type(element.get("order")) is not int
            ):
                raise ValueError(
                    "Authoritative reading order requires explicit integer positions"
                )
    if "text" in reference.capabilities and not isinstance(
        payload.get("text", payload.get("reference")), str
    ):
        raise ValueError(
            f"Document sample {item.get('id')} has no verified reference text"
        )
    if authoritative and not item.get("reference_path"):
        raise ValueError(
            f"Authoritative document sample {item.get('id')} requires reference_path"
        )
    kind = str(item.get("kind"))
    if "pages" in reference.capabilities and not reference.pages:
        raise ValueError(f"Visual sample {item.get('id')} requires verified page text")
    layout_capabilities = {
        "reading_order",
        "layout_boxes",
        "element_types",
        "hierarchy",
    }
    raw_elements = payload.get("elements", payload.get("reference_elements"))
    if reference.capabilities & layout_capabilities and not isinstance(
        raw_elements, (list, tuple)
    ):
        raise ValueError(
            f"Document sample {item.get('id')} claims layout capabilities without an element annotation list"
        )
    identifiers = [element.element_id for element in reference.elements]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Reference element IDs must be unique")
    if "reading_order" in reference.capabilities:
        orders = [element.order for element in reference.elements]
        if len(orders) != len(set(orders)):
            raise ValueError("Reference reading-order positions must be unique")
    for element in reference.elements:
        if (
            element.level_annotated
            and "hierarchy" in reference.capabilities
            and (element.hierarchy_level is None or element.hierarchy_level < 0)
        ):
            raise ValueError("Claimed hierarchy levels must be nonnegative integers")
        if element.page_number is not None and element.page_number < 1:
            raise ValueError("Reference page numbers must be positive")
        if (
            "pages" in reference.capabilities
            and element.page_number is not None
            and element.page_number not in reference.pages
        ):
            raise ValueError("Reference element page is absent from verified pages")
        if (
            "hierarchy" in reference.capabilities
            and element.parent_id is not None
            and (
                element.parent_id not in identifiers
                or element.parent_id == element.element_id
            )
        ):
            raise ValueError(
                "Reference parent IDs must refer to another annotated element"
            )
    parents = {element.element_id: element.parent_id for element in reference.elements}
    if "hierarchy" in reference.capabilities:
        for identifier in identifiers:
            visited = set()
            current = identifier
            while current is not None:
                if current in visited:
                    raise ValueError("Reference hierarchy contains a cycle")
                visited.add(current)
                current = parents.get(current)
    if "tables" in reference.capabilities:
        table_presence = _presence(payload, item, "has_table")
        if table_presence is None and authoritative:
            raise ValueError(
                f"Document sample {item.get('id')} with table annotations must set has_table"
            )
        table_elements = [
            element
            for element in reference.elements
            if element.kind is SegmentKind.TABLE
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
            element.element_id for element in table_elements if not element.html
        ]
        if missing_table_html:
            raise ValueError(
                f"Document sample {item.get('id')} lacks official table references: "
                + ", ".join(missing_table_html[:10])
            )
        from .adapters import _parse_table_html

        if any(not _parse_table_html(element.html)[1] for element in table_elements):
            raise ValueError(
                "Document table references must contain parseable HTML cells"
            )
    if "formulas" in reference.capabilities:
        formula_presence = _presence(payload, item, "has_formula")
        if formula_presence is None and authoritative:
            raise ValueError(
                f"Document sample {item.get('id')} with formula annotations must set has_formula"
            )
        formula_elements = [
            element
            for element in reference.elements
            if element.kind is SegmentKind.FORMULA
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
            if not (element.latex or "").strip()
        ]
        if missing_formula_latex:
            raise ValueError(
                f"Document sample {item.get('id')} lacks official formula references: "
                + ", ".join(missing_formula_latex[:10])
            )
    if (
        "hierarchy" in reference.capabilities
        and reference.elements
        and not any(
            element.parent_annotated or element.level_annotated
            for element in reference.elements
        )
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


def validate_official_evaluators(
    references: Sequence[ReferenceDocument], *, timeout_seconds: int
) -> bool:
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
    validate_official_runtime(
        tables=tables, formulas=formulas, timeout_seconds=timeout_seconds
    )
    return tables or formulas


def apply_official_metrics(
    records: Sequence[DocumentEvaluation], *, timeout_seconds: int
) -> None:
    """Batch all official table/formula scoring into one Docker invocation."""

    table_pairs = [pair for record in records for pair in record.table_pairs]
    formula_pairs = [pair for record in records for pair in record.formula_pairs]
    table_results, formula_results = score_official_metrics(
        table_pairs, formula_pairs, timeout_seconds=timeout_seconds
    )
    if len(table_results) != len(table_pairs) or len(formula_results) != len(
        formula_pairs
    ):
        raise RuntimeError("Official evaluator returned an incomplete result")
    values = [value for pair in table_results for value in pair] + formula_results
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in values):
        raise RuntimeError("Official evaluator returned invalid scores")
    table_offset = 0
    formula_offset = 0
    for record in records:
        table_count = len(record.table_pairs)
        if table_count:
            official = table_results[table_offset : table_offset + table_count]
            record.table_scores = [
                (*content, teds, teds_s)
                for content, (teds, teds_s) in zip(
                    record.table_content_scores, official, strict=True
                )
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
        for name, value in record.metrics.items():
            if value is not None:
                record.statuses[name] = {"status": "scored", "reason": None}


def score_document(
    item: Mapping[str, object],
    document: ExtractedDocument | None,
    *,
    reference: ReferenceDocument | None = None,
    repeated_documents: Sequence[ExtractedDocument | None] = (),
    failed: bool = False,
    element_matching_threshold: float,
    duplicate_content_threshold: float,
) -> DocumentEvaluation:
    reference = reference or load_reference(item)
    result = DocumentEvaluation(
        _document_groups(item),
        sample_id=str(item["id"]),
        source_group_id=source_id(item),
    )
    predicted = tuple(document.segments) if document is not None else ()
    applicable = _eligible_metrics(reference)
    result.metrics = {
        name: None for name in METRIC_DIRECTIONS if not name.startswith("operational.")
    }
    if document is not None and not failed:
        if "text" in reference.capabilities:
            result.metrics.update(_text_metrics(reference, document.text))
            result.metrics["reliability.duplicate_content_rate"] = (
                _duplicate_content_rate(reference.text, document.text)
            )
        if "pages" in reference.capabilities:
            result.metrics.update(
                _page_metrics(
                    reference,
                    predicted,
                    duplicate_content_threshold,
                    element_matching_threshold,
                )
            )
            expected = tuple(
                e
                for e in reference.elements
                if e.page_number is not None and _element_tokens(e)
            )
            matches = _match_elements(
                expected, predicted, visual=False, threshold=element_matching_threshold
            )
            if expected:
                result.metrics["pages.page_attribution_recall"] = sum(
                    expected[left].page_number == predicted[right].page_number
                    for left, right, _ in matches
                ) / len(expected)
        layout_references = tuple(
            e for e in reference.elements if e.kind in LAYOUT_KINDS
        )
        layout_predictions = tuple(e for e in predicted if e.kind in LAYOUT_KINDS)
        if reference.capabilities & {
            "reading_order",
            "layout_boxes",
            "element_types",
            "hierarchy",
        }:
            matches = _match_elements(
                layout_references,
                layout_predictions,
                visual=str(item["kind"]) in VISUAL_KINDS,
                use_boxes="layout_boxes" in reference.capabilities,
                threshold=element_matching_threshold,
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
        if reference.capabilities & {"reading_order", "hierarchy"}:
            matches = _match_elements(
                reference.elements,
                predicted,
                visual=str(item["kind"]) in VISUAL_KINDS,
                use_boxes="layout_boxes" in reference.capabilities,
                threshold=element_matching_threshold,
            )
            if "reading_order" in reference.capabilities:
                result.metrics["text.reading_order_ned"] = _reading_order(
                    matches, reference.elements, predicted
                )
            if "hierarchy" in reference.capabilities:
                result.metrics["layout.hierarchy_preservation_rate"] = (
                    _hierarchy_preservation(reference.elements, predicted, matches)
                )
        for capability, target in (
            ("tables", SegmentKind.TABLE),
            ("formulas", SegmentKind.FORMULA),
        ):
            if capability in reference.capabilities:
                _score_structured_kind(
                    result,
                    reference,
                    predicted,
                    target,
                    str(item["kind"]),
                    element_matching_threshold,
                )
    expected_content = bool(
        normalized_tokens(reference.text)
        or any(_usable_element(e) for e in reference.elements)
    )
    observed_content = bool(
        document is not None
        and (
            normalized_tokens(document.text)
            or any(_usable_element(e) for e in predicted)
        )
    )
    result.metrics["reliability.unexpected_empty_output_rate"] = float(
        document is not None
        and not failed
        and expected_content
        and not observed_content
    )
    fingerprints = [
        _document_fingerprint(value) if value is not None else None
        for value in repeated_documents
    ]
    if not fingerprints:
        fingerprints = [
            _document_fingerprint(document)
            if document is not None and not failed
            else None
        ]
    for name, value in attempt_rates(fingerprints).items():
        result.metrics[f"reliability.{name}"] = value
    for name, value in result.metrics.items():
        if value is not None:
            status, reason = "scored", None
        elif (
            name == "reliability.repeatability_success_rate" and len(fingerprints) == 1
        ):
            status, reason = "inapplicable", "repeatability_not_measured"
        elif name not in applicable:
            status, reason = "inapplicable", "no_reference_task"
        else:
            status, reason = (
                "unavailable",
                "first_attempt_failed"
                if failed or document is None
                else "evaluation_pending",
            )
        result.statuses[name] = {"status": status, "reason": reason}
    return result


def _eligible_metrics(reference: ReferenceDocument) -> set[str]:
    result = {
        name
        for name in METRIC_DIRECTIONS
        if name.startswith("reliability.")
        and name != "reliability.duplicate_content_rate"
    }
    capabilities = reference.capabilities
    if "text" in capabilities:
        result.update(
            name
            for name in METRIC_DIRECTIONS
            if name.startswith("text.") and name != "text.reading_order_ned"
        )
        result.add("reliability.duplicate_content_rate")
    if "reading_order" in capabilities:
        result.add("text.reading_order_ned")
    if "pages" in capabilities:
        result.update(
            {
                "pages.page_coverage",
                "pages.page_content_f1",
                "pages.duplicate_page_rate",
            }
        )
        if any(
            e.page_number is not None and _element_tokens(e) for e in reference.elements
        ):
            result.add("pages.page_attribution_recall")
    layout = [e for e in reference.elements if e.kind in LAYOUT_KINDS]
    if capabilities & {"reading_order", "layout_boxes", "element_types", "hierarchy"}:
        result.update(
            {"layout.element_precision", "layout.element_recall", "layout.element_f1"}
        )
    if layout and "element_types" in capabilities:
        result.add("layout.element_type_recall")
    if layout and "layout_boxes" in capabilities:
        result.add("layout.mean_bounding_box_iou")
    if "hierarchy" in capabilities and any(
        e.parent_annotated or e.level_annotated for e in reference.elements
    ):
        result.add("layout.hierarchy_preservation_rate")
    for capability, kind in (
        ("tables", SegmentKind.TABLE),
        ("formulas", SegmentKind.FORMULA),
    ):
        if capability in capabilities:
            result.update(
                name
                for name in METRIC_DIRECTIONS
                if name.startswith(f"{capability}.detection_")
            )
            if any(e.kind is kind for e in reference.elements):
                result.update(
                    name
                    for name in METRIC_DIRECTIONS
                    if name.startswith(f"{capability}.")
                )
    return result


def _usable_element(element) -> bool:
    return bool(
        _element_tokens(element)
        or _element_identity(element)
        or element.kind is SegmentKind.FIGURE
        and _valid_box(element.bounding_box)
    )


def aggregate_evaluations(
    records, *, resamples, seed, confidence, minimum_sources=None
):
    metrics, intervals = {}, {}
    group_names = sorted({group for record in records for group in record.groups})
    for group in (None, *group_names):
        selected = [
            record for record in records if group is None or group in record.groups
        ]
        if not selected:
            continue
        prefix = "" if group is None else f"{group}."
        estimates = _aggregate_group(selected)
        for name, value in estimates.items():
            qualified = _grouped_name(name, prefix)
            metrics[qualified] = value
            contributors = [
                record for record in selected if record.metrics.get(name) is not None
            ]
            eligible = [
                record
                for record in selected
                if record.statuses.get(name, {}).get("status") != "inapplicable"
            ]
            metrics.update(
                {
                    f"{qualified}.sample_count": float(len(contributors)),
                    f"{qualified}.scheduled_count": float(len(selected)),
                    f"{qualified}.eligible_count": float(len(eligible)),
                    f"{qualified}.unavailable_count": float(
                        len(eligible) - len(contributors)
                    ),
                    f"{qualified}.independent_source_count": float(
                        len({record.source_group_id for record in contributors})
                    ),
                    f"{qualified}.failed_first_attempt_count": float(
                        sum(
                            record.statuses.get(name, {}).get("reason")
                            == "first_attempt_failed"
                            for record in selected
                        )
                    ),
                }
            )
        rows = [
            {"source_group_id": record.source_group_id or str(index), "record": record}
            for index, record in enumerate(selected)
        ]
        bounds = bootstrap_sources(
            rows,
            lambda values: _aggregate_group([row["record"] for row in values]),
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            minimum_sources=minimum_sources,
        )
        intervals.update(
            {_grouped_name(name, prefix): value for name, value in bounds.items()}
        )
    return metrics, intervals


def _reference_from_mapping(payload: Mapping[str, object]) -> ReferenceDocument:
    raw_capabilities = payload.get("reference_capabilities")
    if raw_capabilities is not None:
        _validate_capability_list(raw_capabilities, payload.get("id"))
    text = _canonical(str(payload.get("text", payload.get("reference", ""))))
    raw_pages = payload.get("pages", payload.get("reference_page_texts", []))
    if raw_capabilities is not None and "pages" not in raw_capabilities:
        raw_pages = []
    pages: dict[int, str] = {}
    if isinstance(raw_pages, Mapping):
        for key, value in raw_pages.items():
            page = _optional_int(key)
            if page is None or page < 1 or page in pages or not isinstance(value, str):
                raise ValueError(
                    "Reference pages require unique positive IDs and explicit text"
                )
            pages[page] = _canonical(value)
    elif isinstance(raw_pages, Sequence) and not isinstance(raw_pages, (str, bytes)):
        for index, value in enumerate(raw_pages, 1):
            if isinstance(value, Mapping):
                page = _optional_int(value.get("page_number", index))
                if (
                    page is None
                    or page < 1
                    or page in pages
                    or not isinstance(value.get("text"), str)
                ):
                    raise ValueError(
                        "Reference pages require unique positive IDs and explicit text"
                    )
                pages[page] = _canonical(value["text"])
            else:
                if not isinstance(value, str):
                    raise ValueError(
                        "Reference page text must be explicit, including verified blanks"
                    )
                pages[index] = _canonical(value)
    raw_elements = payload.get("elements", payload.get("reference_elements", []))
    elements: list[ReferenceElement] = []
    if isinstance(raw_elements, Sequence) and not isinstance(
        raw_elements, (str, bytes)
    ):
        for index, value in enumerate(raw_elements):
            if not isinstance(value, Mapping):
                raise ValueError("Reference elements must be objects")
            kind = _kind(value.get("kind", "text"))
            elements.append(
                ReferenceElement(
                    element_id=str(
                        value.get("id", value.get("element_id", f"ref-{index}"))
                    ),
                    kind=kind,
                    text=_canonical(str(value.get("text", ""))),
                    order=int(value.get("order", index)),
                    page_number=_optional_int(value.get("page_number")),
                    bounding_box=_box(value.get("bounding_box"))
                    if raw_capabilities is None or "layout_boxes" in raw_capabilities
                    else None,
                    parent_id=_optional_string(value.get("parent_id")),
                    hierarchy_level=_optional_int(value.get("hierarchy_level")),
                    parent_annotated="parent_id" in value,
                    level_annotated="hierarchy_level" in value,
                    identity=_optional_string(value.get("identity")),
                    html=_optional_string(value.get("html")),
                    latex=_optional_string(value.get("latex")),
                )
            )
    if not pages and payload.get("kind") == "image":
        pages[1] = text
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
    reference, predicted, duplicate_content_threshold, element_matching_threshold
):
    predicted_pages: dict[int, str] = {}
    for segment in predicted:
        if segment.page_number is not None:
            predicted_pages.setdefault(segment.page_number, "")
            predicted_pages[segment.page_number] += "\n" + segment.text
    expected_pages = set(reference.pages)
    page_ids = expected_pages | set(predicted_pages)
    f1_values = [
        _content_f1(reference.pages[page], predicted_pages.get(page, ""))
        if page in expected_pages
        else 0.0
        for page in sorted(page_ids)
    ]
    content_pages = {
        page for page, text in reference.pages.items() if normalized_tokens(text)
    } | {
        e.page_number
        for e in reference.elements
        if e.page_number is not None and _usable_element(e)
    }
    covered = set()
    for page in content_pages:
        expected = [
            e
            for e in reference.elements
            if e.page_number == page and _usable_element(e)
        ]
        if not expected and normalized_tokens(reference.pages.get(page, "")):
            expected = [
                ReferenceElement(
                    str(page),
                    SegmentKind.TEXT,
                    text=reference.pages[page],
                    page_number=page,
                )
            ]
        observed = [e for e in predicted if e.page_number == page]
        if _match_elements(
            expected,
            observed,
            visual=True,
            use_boxes="layout_boxes" in reference.capabilities,
            threshold=element_matching_threshold,
        ):
            covered.add(page)
    observed_usable = any(_usable_element(e) for e in predicted)
    expected_texts = [
        text for text in reference.pages.values() if normalized_tokens(text)
    ]
    observed_texts = [
        text for text in predicted_pages.values() if normalized_tokens(text)
    ]
    return {
        "pages.page_coverage": len(covered) / len(content_pages)
        if content_pages
        else float(not observed_usable),
        "pages.page_content_f1": float(np.mean(f1_values)),
        "pages.duplicate_page_rate": _unsupported_near_duplicates(
            observed_texts, expected_texts, duplicate_content_threshold
        )
        / len(observed_texts)
        if observed_texts
        else 0.0,
    }


def _unsupported_near_duplicates(predicted, reference, threshold):
    expected = [
        ReferenceElement(str(index), SegmentKind.TEXT, text=text)
        for index, text in enumerate(reference)
    ]
    observed = [
        ReferenceElement(str(index), SegmentKind.TEXT, text=text)
        for index, text in enumerate(predicted)
    ]
    supported = {
        right
        for _, right, _ in _match_elements(
            expected, observed, visual=False, threshold=threshold
        )
    }
    representatives = [predicted[index] for index in sorted(supported)]
    duplicates = 0
    for index, text in enumerate(predicted):
        if index in supported:
            continue
        if any(_content_f1(text, value) >= threshold for value in representatives):
            duplicates += 1
        else:
            representatives.append(text)
    return duplicates


def _layout_metrics(references, predictions, matches, *, capabilities):
    result = dict(
        zip(
            ("layout.element_precision", "layout.element_recall", "layout.element_f1"),
            _detection_scores(
                len(matches),
                len(predictions) - len(matches),
                len(references) - len(matches),
            ),
            strict=True,
        )
    )
    if references and "element_types" in capabilities:
        result["layout.element_type_recall"] = sum(
            references[left].kind is predictions[right].kind
            for left, right, _ in matches
        ) / len(references)
    if references and "layout_boxes" in capabilities:
        result["layout.mean_bounding_box_iou"] = sum(
            _iou(references[left].bounding_box, predictions[right].bounding_box)
            for left, right, _ in matches
            if references[left].bounding_box and predictions[right].bounding_box
        ) / len(references)
    return result


def _hierarchy_preservation(references, predictions, matches):
    by_reference = {left: right for left, right, _ in matches}
    eligible = [
        i for i, e in enumerate(references) if e.parent_annotated or e.level_annotated
    ]
    id_mapping = {
        references[left].element_id: predictions[right].element_id
        for left, right, _ in matches
    }
    correct = 0
    for index in eligible:
        if index not in by_reference:
            continue
        expected, observed = references[index], predictions[by_reference[index]]
        parent_correct = not expected.parent_annotated or (
            observed.parent_id is None
            and observed.metadata.get("parent_annotated") is True
            if expected.parent_id is None
            else observed.metadata.get("parent_annotated") is True
            and expected.parent_id in id_mapping
            and id_mapping[expected.parent_id] is not None
            and id_mapping[expected.parent_id] == observed.parent_id
        )
        level_correct = (
            not expected.level_annotated
            or expected.hierarchy_level == observed.metadata.get("hierarchy_level")
        )
        correct += bool(parent_correct and level_correct)
    return correct / len(eligible) if eligible else None


def _score_structured_kind(
    result, reference, predicted, target, source_kind, matching_threshold
) -> None:
    references = tuple(
        element for element in reference.elements if element.kind is target
    )
    predictions = tuple(segment for segment in predicted if segment.kind is target)
    annotation_key = "tables" if target is SegmentKind.TABLE else "formulas"
    # This function is reached only when the explicit capability is present.
    # Keep a count row even for a verified negative stored in a reference file so
    # verified negatives participate in document-macro detection.
    result.counts[annotation_key] = (0, len(predictions), len(references))
    matches = _match_elements(
        references,
        predictions,
        visual=source_kind in VISUAL_KINDS,
        threshold=matching_threshold,
    )
    result.counts[annotation_key] = (
        len(matches),
        len(predictions) - len(matches),
        len(references) - len(matches),
    )
    precision, recall, f1 = _detection_scores(*result.counts[annotation_key])
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
            prediction = (
                predictions[by_reference[index]] if index in by_reference else None
            )
            content = (
                _content_scores(_table_text(item), _table_text(prediction))
                if prediction is not None
                else (0.0, 0.0, 0.0)
            )
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
            prediction = (
                predictions[by_reference[index]] if index in by_reference else None
            )
            result.formula_pairs.append(
                (item.latex or item.text, _formula_latex(prediction))
            )


def _aggregate_group(records):
    names = sorted({name for record in records for name in record.metrics})
    result = {}
    for name in names:
        values = [
            record.metrics[name]
            for record in records
            if record.metrics.get(name) is not None
        ]
        result[name] = float(np.mean(values)) if values else None
    return result


def _match_elements(references, predictions, *, visual, use_boxes=None, threshold):
    if not references or not predictions:
        return []
    from scipy.optimize import linear_sum_assignment

    scores = np.full(
        (len(references) + len(predictions), len(references) + len(predictions)), 0.0
    )
    admissible = np.zeros((len(references), len(predictions)), dtype=bool)
    for left, reference in enumerate(references):
        for right, prediction in enumerate(predictions):
            if visual and reference.page_number != prediction.page_number:
                continue
            if visual and use_boxes is not False and reference.bounding_box is not None:
                score = (
                    _iou(reference.bounding_box, prediction.bounding_box)
                    if _valid_box(prediction.bounding_box)
                    else 0.0
                )
            else:
                expected, observed = (
                    _element_tokens(reference),
                    _element_tokens(prediction),
                )
                if expected and observed:
                    overlap = sum((Counter(expected) & Counter(observed)).values())
                    score = precision_recall_f1(
                        overlap, len(observed) - overlap, len(expected) - overlap
                    )[2]
                else:
                    expected_id = _element_identity(reference)
                    observed_id = _element_identity(prediction)
                    score = float(bool(expected_id) and expected_id == observed_id)
            if score > 0 and score >= threshold:
                scores[left, right] = score
                admissible[left, right] = True
    rows, columns = linear_sum_assignment(-scores)
    return [
        (int(left), int(right), float(scores[left, right]))
        for left, right in zip(rows, columns, strict=True)
        if left < len(references)
        and right < len(predictions)
        and admissible[left, right]
    ]


def _element_tokens(element):
    text = getattr(element, "text", "")
    if element.kind is SegmentKind.TABLE:
        text = _table_text(element)
    if element.kind is SegmentKind.FORMULA:
        text = (
            getattr(element, "latex", None)
            or getattr(element, "structured_content", {}).get("latex")
            or text
        )
    if element.kind in {SegmentKind.CODE, SegmentKind.FORMULA}:
        import re

        return re.findall(r"\w+|[^\w\s]", _canonical(text))
    return normalized_tokens(text)


def _table_text(element):
    """Cell contents, never HTML tags or Markdown separators, are lexical units."""
    structured = getattr(element, "structured_content", {})
    cells = structured.get("cells")
    if isinstance(cells, (list, tuple)):
        return " ".join(str(cell["text"]) for cell in cells)
    html = getattr(element, "html", None) or structured.get("html")
    if html:
        from .adapters import _parse_table_html

        return " ".join(str(cell["text"]) for cell in _parse_table_html(html)[1])
    return element.text


def _element_identity(element):
    structured = getattr(element, "structured_content", {})
    identity = getattr(element, "identity", None) or structured.get("identity")
    if identity:
        return ("identity", identity)
    html = getattr(element, "html", None) or structured.get("html")
    if element.kind is SegmentKind.TABLE and html:
        from .adapters import _parse_table_html

        rows, cells = _parse_table_html(html)
        if cells:
            return ("table", stable_hash({"rows": rows, "cells": cells}))
    return None


def _valid_box(box):
    return (
        box is not None
        and len(box) == 4
        and all(np.isfinite(x) and 0 <= x <= 1 for x in box)
        and box[0] < box[2]
        and box[1] < box[3]
    )


def _reading_order(matches, references, predictions):
    observed_ids = {right: ("reference", left) for left, right, _ in matches}
    expected = [
        ("reference", index)
        for index in sorted(range(len(references)), key=lambda i: references[i].order)
    ]
    observed = [
        observed_ids.get(index, ("extra", index))
        for index in sorted(
            range(len(predictions)),
            key=lambda i: (
                predictions[i].order if predictions[i].order is not None else i
            ),
        )
    ]
    return (
        levenshtein(expected, observed) / max(len(expected), len(observed))
        if expected or observed
        else 0.0
    )


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


def _content_f1(reference: str, prediction: str) -> float:
    return _content_scores(_canonical(reference), _canonical(prediction))[2]


def _content_scores(reference: str, prediction: str) -> tuple[float, float, float]:
    left = Counter(normalized_tokens(reference))
    right = Counter(normalized_tokens(prediction))
    overlap = sum((left & right).values())
    return _detection_scores(
        overlap,
        sum(right.values()) - overlap,
        sum(left.values()) - overlap,
    )


def _duplicate_content_rate(reference: str, prediction: str) -> float | None:
    predicted = normalized_tokens(prediction)
    if not predicted:
        return 0.0
    predicted_counts = Counter(predicted)
    extra = predicted_counts - Counter(normalized_tokens(reference))
    repeated = sum(
        count for token, count in extra.items() if predicted_counts[token] > 1
    )
    return repeated / len(predicted)


def _document_fingerprint(document: ExtractedDocument) -> str:
    identifiers = {
        segment.element_id: str(index)
        for index, segment in enumerate(document.segments)
        if segment.element_id is not None
    }
    return stable_hash(
        {
            "text": _canonical(document.text),
            "segments": [
                {
                    "text": segment.text,
                    "id": identifiers.get(segment.element_id),
                    "parent": identifiers.get(segment.parent_id, "unresolved-parent")
                    if segment.parent_id is not None
                    else None,
                    "order": segment.order,
                    "page": segment.page_number,
                    "box": segment.bounding_box,
                    "kind": segment.kind.value,
                    "structured": segment.structured_content,
                    "hierarchy_level": segment.metadata.get("hierarchy_level"),
                    "parent_annotated": segment.metadata.get("parent_annotated", False),
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
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        return None
    result = tuple(float(item) for item in value)
    if not _valid_box(result):
        raise ValueError("Reference boxes must use normalized coordinates")
    return result  # type: ignore[return-value]


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(
            "Reference integer annotations must not be fractional or boolean"
        )
    return int(value)


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
    if "text" in payload or "reference" in payload:
        capabilities.add("text")
    if pages:
        capabilities.add("pages")
    if elements:
        capabilities.add("element_types")
    if elements and all(
        "order" in value
        for value in payload.get("elements", payload.get("reference_elements", []))
    ):
        capabilities.add("reading_order")
    if any(element.bounding_box is not None for element in elements):
        capabilities.add("layout_boxes")
    if any(element.parent_annotated or element.level_annotated for element in elements):
        capabilities.add("hierarchy")
    if (
        any(element.kind is SegmentKind.TABLE for element in elements)
        or "has_table" in payload
    ):
        capabilities.add("tables")
    if (
        any(element.kind is SegmentKind.FORMULA for element in elements)
        or "has_formula" in payload
    ):
        capabilities.add("formulas")
    return frozenset(capabilities)


def _presence(
    payload: Mapping[str, object], item: Mapping[str, object], key: str
) -> bool | None:
    value = payload.get(key, item.get(key))
    return value if isinstance(value, bool) else None


def _detection_scores(tp, fp, fn):
    return (1.0, 1.0, 1.0) if tp + fp + fn == 0 else precision_recall_f1(tp, fp, fn)
