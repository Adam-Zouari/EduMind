"""Document-extraction metric contract used by the executable benchmark."""

from __future__ import annotations

import importlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from edumind.common.artifacts import stable_hash
from edumind.common.paths import PROJECT_ROOT
from edumind.extraction import ExtractedDocument, ExtractedSegment, SegmentKind
from experiments.benchmarks.common.metrics import (
    character_error_rate,
    normalized_tokens,
    word_error_rate,
)
from experiments.benchmarks.preparation.evaluators import OMNIDOCBENCH_REVISION

VISUAL_KINDS = {"image", "pdf"}
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


@dataclass
class DocumentEvaluation:
    groups: tuple[str, ...]
    metrics: dict[str, float] = field(default_factory=dict)
    counts: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    table_scores: list[tuple[float, float]] = field(default_factory=list)
    formula_scores: list[float] = field(default_factory=list)


def load_reference(item: Mapping[str, object]) -> ReferenceDocument:
    reference_path = item.get("reference_path")
    if reference_path:
        path = PROJECT_ROOT / str(reference_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError(f"Reference document must be a JSON object: {path}")
        return _reference_from_mapping(payload)
    return _reference_from_mapping(item)


def validate_reference(item: Mapping[str, object], *, authoritative: bool) -> None:
    reference = load_reference(item)
    if not reference.text.strip():
        raise ValueError(f"Document sample {item.get('id')} has no verified reference text")
    if authoritative and not item.get("reference_path"):
        raise ValueError(
            f"Authoritative document sample {item.get('id')} requires reference_path"
        )
    kind = str(item.get("kind"))
    if kind in VISUAL_KINDS and not reference.pages:
        raise ValueError(f"Visual sample {item.get('id')} requires verified page text")
    if authoritative and not reference.elements:
        raise ValueError(f"Document sample {item.get('id')} requires element annotations")
    if authoritative and kind in VISUAL_KINDS:
        missing_boxes = [
            element.element_id
            for element in reference.elements
            if element.bounding_box is None
        ]
        if missing_boxes:
            raise ValueError(
                f"Visual sample {item.get('id')} has elements without normalized boxes: "
                + ", ".join(missing_boxes[:10])
            )


def validate_official_evaluators(items: Sequence[Mapping[str, object]]) -> None:
    kinds = {
        element.kind
        for item in items
        for element in load_reference(item).elements
    }
    if SegmentKind.TABLE in kinds:
        _official_module("table_metric")
    if SegmentKind.FORMULA in kinds:
        missing = [
            name
            for name in ("pdflatex", "kpsewhich", "magick")
            if shutil.which(name) is None
        ]
        if missing:
            raise RuntimeError(
                "Official CDM formula scoring requires system commands: "
                + ", ".join(missing)
            )
        if _cdm("x", "x") != 1.0:
            raise RuntimeError("Official CDM evaluator failed its identity preflight")


def score_document(
    item: Mapping[str, object],
    document: ExtractedDocument | None,
    *,
    repeated_documents: Sequence[ExtractedDocument] = (),
    failed: bool = False,
) -> DocumentEvaluation:
    reference = load_reference(item)
    kind = str(item["kind"])
    groups = _document_groups(item)
    result = DocumentEvaluation(groups)
    hypothesis = document.text if document else ""
    result.metrics.update(_text_metrics(reference, hypothesis))
    predicted = tuple(document.segments) if document else ()

    if kind in VISUAL_KINDS:
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
    if layout_references or layout_predictions:
        matches = _match_elements(layout_references, layout_predictions, visual=kind in VISUAL_KINDS)
        result.counts["layout"] = (
            len(matches),
            len(layout_predictions) - len(matches),
            len(layout_references) - len(matches),
        )
        result.metrics.update(_layout_metrics(layout_references, layout_predictions, matches))
        reading_order = _reading_order(matches, layout_references, layout_predictions)
        if reading_order is not None:
            result.metrics["text.reading_order_accuracy"] = reading_order

    _score_structured_kind(result, reference, predicted, SegmentKind.TABLE, kind, item)
    _score_structured_kind(result, reference, predicted, SegmentKind.FORMULA, kind, item)
    result.metrics["reliability.empty_output_rate"] = float(not hypothesis.strip())
    duplicate_rate = _duplicate_content_rate(reference.text, hypothesis)
    if duplicate_rate is not None:
        result.metrics["reliability.duplicate_content_rate"] = duplicate_rate
    result.metrics["reliability.candidate_failure_rate"] = float(failed)
    if repeated_documents:
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
    return ReferenceDocument(text, pages, tuple(elements))


def _text_metrics(reference: ReferenceDocument, hypothesis: str) -> dict[str, float]:
    expected = _canonical(reference.text)
    observed = _canonical(hypothesis)
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
            - sum(_content_f1(group[0], expected) >= 0.95 for expected in reference),
        )
        for group in groups
    )


def _layout_metrics(references, predictions, matches) -> dict[str, float]:
    tp, fp, fn = len(matches), len(predictions) - len(matches), len(references) - len(matches)
    precision, recall, f1 = _prf(tp, fp, fn)
    result = {
        "layout.element_precision": precision,
        "layout.element_recall": recall,
        "layout.element_f1": f1,
    }
    if matches:
        result["layout.element_type_accuracy"] = sum(
            references[left].kind is predictions[right].kind for left, right, _ in matches
        ) / len(matches)
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
        boxes = [score for left, right, score in matches if references[left].bounding_box and predictions[right].bounding_box]
        if boxes:
            result["layout.mean_bounding_box_iou"] = float(np.mean(boxes))
    return result


def _score_structured_kind(
    result, reference, predicted, target, source_kind, item
) -> None:
    references = tuple(element for element in reference.elements if element.kind is target)
    predictions = tuple(segment for segment in predicted if segment.kind is target)
    annotation_key = "tables" if target is SegmentKind.TABLE else "formulas"
    presence_key = "has_table" if target is SegmentKind.TABLE else "has_formula"
    annotated = bool(references) or bool(predictions) or presence_key in item
    if not annotated:
        return
    result.counts[annotation_key] = (
        0, len(predictions), len(references)
    )
    if not references and not predictions:
        return
    matches = _match_elements(references, predictions, visual=source_kind in VISUAL_KINDS)
    result.counts[annotation_key] = (
        len(matches), len(predictions) - len(matches), len(references) - len(matches)
    )
    precision, recall, f1 = _prf(*result.counts[annotation_key])
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
            content = _content_f1(item.text, prediction.text if prediction else "")
            structure = _teds_structure(item.html or "", _table_html(prediction)) if prediction else 0.0
            result.table_scores.append((content, structure))
        result.metrics["tables.content_f1"] = float(np.mean([value[0] for value in result.table_scores]))
        result.metrics["tables.structure_score"] = float(np.mean([value[1] for value in result.table_scores]))
    if target is SegmentKind.FORMULA and references:
        for index, item in enumerate(references):
            prediction = predictions[by_reference[index]] if index in by_reference else None
            result.formula_scores.append(
                _cdm(item.latex or item.text, _formula_latex(prediction)) if prediction else 0.0
            )
        result.metrics["formulas.recognition_similarity"] = float(np.mean(result.formula_scores))
        result.metrics["formulas.exact_match"] = float(
            np.mean([value == 1.0 for value in result.formula_scores])
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
        precision, recall, f1 = _prf(tp, fp, fn)
        values[f"{category}.detection_precision" if category != "layout" else "layout.element_precision"] = precision
        values[f"{category}.detection_recall" if category != "layout" else "layout.element_recall"] = recall
        values[f"{category}.detection_f1" if category != "layout" else "layout.element_f1"] = f1
    table_scores = [score for record in records for score in record.table_scores]
    if table_scores:
        values["tables.content_f1"] = float(np.mean([value[0] for value in table_scores]))
        values["tables.structure_score"] = float(np.mean([value[1] for value in table_scores]))
    formula_scores = [score for record in records for score in record.formula_scores]
    if formula_scores:
        values["formulas.recognition_similarity"] = float(np.mean(formula_scores))
        values["formulas.exact_match"] = float(np.mean([value == 1.0 for value in formula_scores]))
    return values


def _match_elements(references, predictions, *, visual: bool):
    if not references or not predictions:
        return []
    scores = np.zeros((len(references), len(predictions)), dtype=np.float64)
    for left, reference in enumerate(references):
        for right, prediction in enumerate(predictions):
            if visual and reference.bounding_box:
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
    precision = overlap / sum(right.values()) if right else 0.0
    recall = overlap / sum(left.values()) if left else float(not right)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


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


def _teds_structure(reference_html: str, prediction_html: str) -> float:
    if not reference_html or not prediction_html:
        return 0.0
    module = _official_module("table_metric")
    return float(module.TEDS(structure_only=True).evaluate(_html(reference_html), _html(prediction_html)))


def _cdm(reference_latex: str, prediction_latex: str) -> float:
    if not reference_latex or not prediction_latex:
        return 0.0
    module = _official_module("cdm_metric")
    evaluator = module.CDM(output_root=str(Path(os.environ.get("TEMP", ".")) / "cdm"))
    previous = os.environ.get("CDM_SAVE_VIS")
    os.environ["CDM_SAVE_VIS"] = "0"
    try:
        metrics = evaluator.evaluate(
            reference_latex,
            prediction_latex,
            stable_hash([reference_latex, prediction_latex])[:16],
        )
    finally:
        if previous is None:
            os.environ.pop("CDM_SAVE_VIS", None)
        else:
            os.environ["CDM_SAVE_VIS"] = previous
    if metrics.get("cdm_eval_error"):
        raise RuntimeError(f"Official CDM evaluation failed: {metrics['cdm_eval_error']}")
    return float(metrics["F1_score"])


def _official_module(name: str):
    root = Path(os.getenv("EDUMIND_OMNIDOCBENCH_PATH", PROJECT_ROOT / "data/benchmarks/evaluators/OmniDocBench"))
    revision_file = root / ".edumind-revision"
    if not root.is_dir() or not revision_file.is_file():
        raise RuntimeError(
            "Official OmniDocBench evaluators are missing; run "
            "python experiments/benchmarks/prepare.py evaluators"
        )
    if revision_file.read_text(encoding="utf-8").strip() != OMNIDOCBENCH_REVISION:
        raise RuntimeError("OmniDocBench evaluator revision does not match the pinned metric contract")
    source_root = str(root / "src")
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    module = importlib.import_module(f"metrics.{name}")
    module_path = Path(module.__file__).resolve()
    if root.resolve() not in module_path.parents:
        raise RuntimeError(f"Conflicting metrics package loaded from {module_path}")
    if name == "cdm_metric" and os.name == "nt":
        _enable_windows_cdm_process_adapter()
    return module


def _enable_windows_cdm_process_adapter() -> None:
    """Keep the official CDM algorithm while replacing its POSIX shell launcher."""

    module_name = "metrics.cdm.modules.latex2bbox_color"
    module = sys.modules.get(module_name)
    if module is None:
        return

    def run_cmd(command: str, timeout_sec: float = 30) -> int:
        arguments = shlex.split(command.replace(">/dev/null", ""), posix=True)
        try:
            completed = subprocess.run(
                arguments,
                timeout=timeout_sec,
                env=module.build_tex_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return int(completed.returncode)
        except (OSError, subprocess.TimeoutExpired):
            return -1

    module.run_cmd = run_cmd


def _table_html(segment: ExtractedSegment | None) -> str:
    return str(segment.structured_content.get("html", "")) if segment else ""


def _formula_latex(segment: ExtractedSegment | None) -> str:
    return str(segment.structured_content.get("latex", segment.text)) if segment else ""


def _html(value: str) -> str:
    lowered = value.casefold()
    return value if "<body" in lowered else f"<html><body>{value}</body></html>"


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


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
    except ValueError:
        return SegmentKind.TEXT


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
