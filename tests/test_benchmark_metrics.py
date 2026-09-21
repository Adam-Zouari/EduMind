from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.benchmarks.common.metrics import (
    average_precision_at_k,
    balanced_accuracy,
    balanced_accuracy_interval,
    character_error_rate,
    citation_scores,
    context_precision_at_k,
    context_recall,
    exact_match,
    hit_rate_at_k,
    interval_overlap,
    merge_intervals,
    ndcg_at_k,
    normalize_prose,
    paired_bootstrap_interval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_grades,
    rouge_l,
    token_f1,
    word_error_rate,
)
from experiments.benchmarks.extraction.document.metrics import (
    METRIC_DIRECTIONS,
    aggregate_evaluations,
    load_reference,
    score_document as _score_document,
    validate_reference,
)
from experiments.benchmarks.extraction.document import cli as document_cli
from experiments.benchmarks.extraction.document.protocol import (
    load_protocol as load_document_protocol,
)
from experiments.benchmarks.extraction.document.cli import _document_candidates
from experiments.benchmarks.extraction.document.adapters import _paddle_blocks
from experiments.benchmarks.extraction.document.protocol import load_protocol as default_document_protocol
from edumind.extraction import (
    ExtractedDocument,
    ExtractedSegment,
    ExtractionProfile,
    ExtractionRequest,
    SegmentKind,
    SourceKind,
)
from edumind.extraction.structured import build_structured_document
from experiments.benchmarks.rag.chunking_embedding.strategies import (
    build_chunking_strategy,
)
from experiments.benchmarks.rag.chunking_embedding.protocol import load_protocol as default_chunking_protocol


def score_document(*args, **kwargs):
    protocol = default_document_protocol()
    kwargs.setdefault(
        "element_matching_threshold", protocol.element_matching_threshold
    )
    kwargs.setdefault(
        "duplicate_content_threshold", protocol.duplicate_content_threshold
    )
    return _score_document(*args, **kwargs)


def test_official_metric_worker_bootstraps_both_omnidocbench_import_roots(
    tmp_path, monkeypatch
) -> None:
    from experiments.benchmarks.extraction.document import omnidocbench_worker

    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    input_path.write_text(
        json.dumps({"tables": [], "formulas": []}), encoding="utf-8"
    )
    monkeypatch.setattr(omnidocbench_worker.sys, "path", list(sys.path))
    omnidocbench_worker.main(input_path, output_path)
    assert omnidocbench_worker.sys.path[:2] == [
        "/opt/omnidocbench/src",
        "/opt/omnidocbench",
    ]


def test_interval_overlap_is_clamped_and_union_avoids_double_counting() -> None:
    assert interval_overlap((0, 5), (10, 20)) == 0
    assert interval_overlap((0, 10), (5, 15)) == 5
    assert merge_intervals([(0, 8), (5, 10), (12, 14)]) == [(0, 10), (12, 14)]
    assert context_recall([(0, 10)], [(0, 7), (5, 10)]) == 1.0


def test_rank_metrics_have_known_values() -> None:
    relevance = [0.0, 1.0, 0.0, 1.0]
    assert context_precision_at_k(relevance, 4) == pytest.approx((1 / 2 + 2 / 4) / 2)
    assert average_precision_at_k(relevance, 2, 4) == pytest.approx(0.5)
    assert 0 < ndcg_at_k(relevance, 4) < 1
    assert ndcg_at_k([0.5], 1, [1.0, 0.5]) < 1.0


def test_error_and_answer_metrics() -> None:
    assert character_error_rate("cat", "cut") == pytest.approx(1 / 3)
    assert word_error_rate("the cat", "the dog") == pytest.approx(0.5)
    assert token_f1("red blue", "red green") == pytest.approx(0.5)
    assert normalize_prose("  CAFÉ—Test!\n") == "café test"
    assert normalize_prose("algo-\nrithm") == "algo rithm"


def test_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap_interval([1, 2, 3], resamples=500, seed=42)
    second = paired_bootstrap_interval([1, 2, 3], resamples=500, seed=42)
    assert first == second


def test_metric_edge_cases_and_directions() -> None:
    assert relevance_grades([(0, 4)], [(2, 6)]) == [0.5]
    assert precision_at_k([1.0, 0.0], 2) == 0.5
    with pytest.raises(ValueError, match="positive"):
        precision_at_k([], 0)
    assert recall_at_k([1.0], 0, 1) == 0.0
    assert hit_rate_at_k([0.0, 1.0], 2) == 1.0
    assert reciprocal_rank([0.0, 1.0]) == 0.5
    assert exact_match("The Answer!", "the answer") == 1.0
    assert rouge_l("a b c", "a c") > 0.0
    assert citation_scores("No citations", set())["citation_recall"] == 1.0
    assert balanced_accuracy([False, True], [False, True]) == 1.0
    with pytest.raises(ValueError, match="equal length"):
        balanced_accuracy([True], [])
    assert balanced_accuracy([], []) == 0.0
    balanced = balanced_accuracy_interval(
        [True, True, True, False],
        [True, True, True, True],
        resamples=500,
        seed=42,
    )
    assert balanced.estimate == 0.5
    assert balanced.lower == balanced.upper == 0.5
    paired = paired_bootstrap_interval([2, 4], [1, 1], resamples=100)
    assert paired.estimate == 2.0
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_interval([])


def test_extraction_text_metrics_have_known_ranges_and_directions() -> None:
    document = _document(
        "alpha extra",
        (ExtractedSegment("alpha extra", 0, 11, element_id="p", order=0),),
    )
    result = score_document(
        {
            "id": "text",
            "kind": "docx",
            "reference": "alpha beta",
            "reference_elements": [
                {"id": "p", "kind": "text", "text": "alpha beta", "order": 0}
            ],
        },
        document,
        repeated_documents=(document, document),
    )
    assert result.metrics["text.content_precision"] == 0.5
    assert result.metrics["text.content_recall"] == 0.5
    assert result.metrics["reliability.structured_output_determinism"] == 1.0

    equivalent = score_document(
        {"id": "projection", "kind": "docx", "reference": "CAFÉ—based"},
        _document(
            "café based",
            (ExtractedSegment("café based", 0, 10),),
        ),
    )
    assert equivalent.metrics["text.character_error_rate"] == 0.0
    assert equivalent.metrics["text.word_error_rate"] == 0.0


def test_document_metrics_use_element_order_and_grouped_aggregates() -> None:
    text = "Heading\n\nParagraph"
    document = _document(
        text,
        (
            ExtractedSegment(
                "Heading", 0, 7, element_id="h", order=1, kind=SegmentKind.HEADING
            ),
            ExtractedSegment("Paragraph", 9, 18, element_id="p", order=0),
        ),
    )
    result = score_document(
        {
            "id": "layout",
            "kind": "docx",
            "document_family": "native",
            "reference": text,
            "reference_elements": [
                {"id": "h", "kind": "heading", "text": "Heading", "order": 0},
                {"id": "p", "kind": "text", "text": "Paragraph", "order": 1},
            ],
        },
        document,
        repeated_documents=(document, document),
    )
    assert result.metrics["layout.element_f1"] == 1.0
    assert result.metrics["text.reading_order_accuracy"] == 0.0
    metrics, intervals = aggregate_evaluations(
        [result, result], resamples=50, seed=42, confidence=0.95
    )
    assert metrics["text.content_f1"] == 1.0
    assert metrics["text.docx.content_f1"] == 1.0
    assert metrics["text.docx_native.content_f1"] == 1.0
    assert intervals["text.content_f1"]["lower"] == 1.0


def test_table_metrics_separate_detection_content_and_tree_similarity(monkeypatch) -> None:
    from experiments.benchmarks.extraction.document import metrics as document_metrics

    monkeypatch.setattr(
        document_metrics,
        "score_official_metrics",
            lambda tables, formulas, **_kwargs: ([(0.8, 0.9) for _ in tables], [1.0 for _ in formulas]),
    )
    document = _document(
        "A extra",
        (
            ExtractedSegment(
                "A extra",
                0,
                7,
                element_id="table",
                order=0,
                kind=SegmentKind.TABLE,
                structured_content={
                    "rows": [["A", "extra"]],
                    "html": "<table><tr><td>A</td><td>extra</td></tr></table>",
                },
            ),
        ),
    )
    result = score_document(
        {
            "id": "table",
            "kind": "docx",
            "reference": "A missing",
            "reference_elements": [
                {
                    "id": "table",
                    "kind": "table",
                    "text": "A missing",
                    "order": 0,
                    "html": "<table><tr><td>A</td><td>missing</td></tr></table>",
                }
            ],
        },
        document,
    )
    document_metrics.apply_official_metrics([result], timeout_seconds=3600)
    assert result.metrics["tables.detection_f1"] == 1.0
    assert result.metrics["tables.content_precision"] == 0.5
    assert result.metrics["tables.content_recall"] == 0.5
    assert result.metrics["tables.content_f1"] == 0.5
    assert result.metrics["tables.teds"] == 0.8
    assert result.metrics["tables.teds_s"] == 0.9


def test_section_and_structure_chunkers_return_exact_source_spans() -> None:
    class CharacterTokenizer:
        name = "characters"

        @staticmethod
        def spans(text):
            return [(index, index + 1) for index, value in enumerate(text) if not value.isspace()]

        @classmethod
        def count(cls, text):
            return len(cls.spans(text))

    text = "# Section\n\nText\n\n| H | V |\n|---|---|\n| a | 1 |\n\n$$x^2$$"
    for name in ("section-aware-512-64", "structure-aware-512-64"):
        spans = build_chunking_strategy(
            name,
            tokenizer=CharacterTokenizer(),
            protocol=default_chunking_protocol(),
        ).split(text)
        assert spans
        assert all(text[start:end] and 0 <= start < end <= len(text) for start, end, _ in spans)


def test_page_metrics_detect_wrong_page_attribution() -> None:
    text = "beta\nalpha"
    document = _document(
        text,
        (
            ExtractedSegment("beta", 0, 4, page_number=1),
            ExtractedSegment("alpha", 5, 10, page_number=2),
        ),
        kind=SourceKind.PDF,
    )
    scores = score_document(
        {
            "id": "pages",
            "kind": "pdf",
            "reference": "alpha\nbeta",
            "reference_page_texts": ["alpha", "beta"],
            "reference_elements": [
                {"id": "a", "kind": "text", "text": "alpha", "page_number": 1},
                {"id": "b", "kind": "text", "text": "beta", "page_number": 2},
            ],
        },
        document,
    )
    assert scores.metrics["pages.page_coverage"] == 0.0
    assert scores.metrics["pages.page_attribution_accuracy"] == 0.0
    assert scores.metrics["pages.page_content_f1"] == 0.0


def test_page_metrics_detect_an_unsupported_repeated_page() -> None:
    document = _document(
        "alpha\nbeta\nbeta",
        (
            ExtractedSegment("alpha", 0, 5, page_number=1),
            ExtractedSegment("beta", 6, 10, page_number=2),
            ExtractedSegment("beta", 11, 15, page_number=3),
        ),
        kind=SourceKind.PDF,
    )
    scores = score_document(
        {
            "id": "duplicate-page",
            "kind": "pdf",
            "reference": "alpha\nbeta",
            "reference_page_texts": ["alpha", "beta"],
        },
        document,
    )
    assert scores.metrics["pages.duplicate_page_rate"] == 1 / 3


def test_page_metrics_do_not_call_one_unique_extra_page_a_duplicate() -> None:
    prediction = _document(
        "alpha\ngamma",
        (
            ExtractedSegment("alpha", 0, 5, page_number=1),
            ExtractedSegment("gamma", 6, 11, page_number=2),
        ),
        kind=SourceKind.PDF,
    )
    scores = score_document(
        {
            "id": "unique-extra-page",
            "kind": "pdf",
            "reference": "alpha",
            "reference_page_texts": ["alpha"],
        },
        prediction,
    )
    assert scores.metrics["pages.duplicate_page_rate"] == 0.0


def test_paddle_native_json_is_converted_without_markdown_inference() -> None:
    blocks = _paddle_blocks(
        {
            "res": {
                "page_index": 0,
                "width": 200,
                "height": 100,
                "parsing_res_list": [
                    {
                        "block_id": 7,
                        "block_label": "table",
                        "block_content": "<table><tr><td>A</td></tr></table>",
                        "block_bbox": [20, 10, 180, 90],
                    }
                ],
            }
        },
    )
    assert blocks[0]["kind"] == "table"
    assert blocks[0]["bounding_box"] == [0.1, 0.1, 0.9, 0.9]


def test_paddle_image_tables_formulas_and_recoverable_html_are_canonical() -> None:
    from edumind.extraction import ExtractionWarning

    warnings: list[ExtractionWarning] = []
    blocks = _paddle_blocks(
        {
            "page_index": None,
            "parsing_res_list": [
                {
                    "block_label": "table",
                    "block_content": (
                        "<table><tr><th>A</th><th>B</th></tr>"
                        "<tr><td>1</td><td>2</td></tr></table>"
                    ),
                },
                {"block_label": "formula", "block_content": r"\frac{x}{y}"},
                {"block_label": "table", "block_content": "<table><tr>broken"},
            ],
        },
        warnings=warnings,
    )
    assert all(block["page_number"] == 1 for block in blocks)
    assert blocks[0]["text"] == "A\tB\n1\t2"
    assert blocks[0]["structured_content"]["rows"] == [["A", "B"], ["1", "2"]]
    assert len(blocks[0]["structured_content"]["cells"]) == 4
    assert blocks[1]["text"] == r"\frac{x}{y}"
    assert blocks[1]["structured_content"]["latex"] == r"\frac{x}{y}"
    assert blocks[2]["structured_content"]["rows"] == [["broken"]]
    assert warnings[0].code == "paddle_table_html_malformed"


def test_paddle_malformed_block_is_skipped_with_a_conversion_warning() -> None:
    from edumind.extraction import ExtractionWarning

    warnings: list[ExtractionWarning] = []
    blocks = _paddle_blocks(
        {
            "page_index": None,
            "parsing_res_list": [
                "not-an-object",
                {"block_label": "text", "block_content": "kept"},
            ],
        },
        warnings=warnings,
    )
    assert [block["text"] for block in blocks] == ["kept"]
    assert [warning.code for warning in warnings] == [
        "paddle_block_conversion_failed"
    ]


def test_visual_layout_table_and_formula_never_match_across_pages() -> None:
    text = "same\n\nsame\n\nsame"
    document = _document(
        text,
        (
            ExtractedSegment(
                "same", 0, 4, page_number=2, bounding_box=(0.1, 0.1, 0.5, 0.5)
            ),
            ExtractedSegment(
                "same",
                6,
                10,
                page_number=2,
                bounding_box=(0.1, 0.1, 0.5, 0.5),
                kind=SegmentKind.TABLE,
                structured_content={"rows": [["same"]], "html": "<table><tr><td>same</td></tr></table>"},
            ),
            ExtractedSegment(
                "same",
                12,
                16,
                page_number=2,
                bounding_box=(0.1, 0.1, 0.5, 0.5),
                kind=SegmentKind.FORMULA,
                structured_content={"latex": "same"},
            ),
        ),
        kind=SourceKind.PDF,
    )
    item = {
        "id": "cross-page",
        "kind": "pdf",
        "reference": "same",
        "reference_capabilities": [
            "text", "pages", "layout_boxes", "element_types", "tables", "formulas"
        ],
        "reference_page_texts": ["same"],
        "has_table": True,
        "has_formula": True,
        "reference_elements": [
            {"id": "text", "kind": "text", "text": "same", "page_number": 1, "bounding_box": [0.1, 0.1, 0.5, 0.5]},
            {"id": "table", "kind": "table", "text": "same", "html": "<table><tr><td>same</td></tr></table>", "page_number": 1, "bounding_box": [0.1, 0.1, 0.5, 0.5]},
            {"id": "formula", "kind": "formula", "text": "same", "latex": "same", "page_number": 1, "bounding_box": [0.1, 0.1, 0.5, 0.5]},
        ],
    }
    result = score_document(item, document)
    assert result.metrics["layout.element_f1"] == 0.0
    assert result.metrics["tables.detection_f1"] == 0.0
    assert result.metrics["formulas.detection_f1"] == 0.0
    # Page attribution remains content-first and therefore still observes the wrong page.
    assert result.metrics["pages.page_attribution_accuracy"] == 0.0


def test_unclaimed_layout_boxes_do_not_change_content_matching() -> None:
    document = _document(
        "same",
        (
            ExtractedSegment(
                "same",
                0,
                4,
                page_number=1,
                bounding_box=(0.6, 0.6, 0.9, 0.9),
            ),
        ),
        kind=SourceKind.PDF,
    )
    result = score_document(
        {
            "id": "types-with-unclaimed-boxes",
            "kind": "pdf",
            "reference_capabilities": ["element_types"],
            "reference_elements": [
                {
                    "id": "text",
                    "kind": "text",
                    "text": "same",
                    "page_number": 1,
                    "bounding_box": [0.1, 0.1, 0.4, 0.4],
                }
            ],
        },
        document,
    )
    assert result.metrics["layout.element_f1"] == 1.0
    assert result.metrics["layout.element_type_accuracy"] == 1.0
    assert "layout.mean_bounding_box_iou" not in result.metrics


def test_authoritative_reference_capabilities_are_explicit_and_conditional(tmp_path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(
        '{"text":"verified","reference_capabilities":["text","tables"],'
        '"has_table":false,"elements":[]}',
        encoding="utf-8",
    )
    validate_reference(
        {"id": "negative", "kind": "docx", "reference_path": str(path)},
        authoritative=True,
    )
    missing = tmp_path / "missing-capabilities.json"
    missing.write_text('{"text":"verified"}', encoding="utf-8")
    with pytest.raises(ValueError, match="reference_capabilities"):
        validate_reference(
            {"id": "missing", "kind": "docx", "reference_path": str(missing)},
            authoritative=True,
        )


def test_authoritative_reference_rejects_contradictory_structured_negatives(tmp_path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(
        '{"reference_capabilities":["tables"],"has_table":false,'
        '"elements":[{"kind":"table","text":"A","html":"<table><tr><td>A</td></tr></table>"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="has_table:false"):
        validate_reference(
            {"id": "contradiction", "kind": "docx", "reference_path": str(path)},
            authoritative=True,
        )


def test_document_directions_omit_unclaimed_reference_metrics() -> None:
    from experiments.benchmarks.extraction.document.runner import directions_for

    selected = directions_for(
        [
            load_reference(
                {
                    "id": "pages-only",
                    "kind": "pdf",
                    "reference_capabilities": ["pages"],
                    "reference_page_texts": ["page"],
                }
            )
        ],
        METRIC_DIRECTIONS,
    )
    assert "pages.page_content_f1" in selected
    assert "text.content_f1" not in selected
    assert "layout.mean_bounding_box_iou" not in selected


def test_docling_candidate_requires_every_behavior_component() -> None:
    from experiments.benchmarks.extraction.document.runner import (
        validate_prepared_components,
    )

    candidate = (
        "docling-standard|ocr=tesseract|mode=full_page|table=accurate|formula=on"
    )
    with pytest.raises(RuntimeError, match="code_formula, tesseract-cli"):
        validate_prepared_components(
            candidate,
            {"prepared_components": ["layout", "tableformer"]},
            load_document_protocol(),
        )
    validate_prepared_components(
        candidate,
        {
            "prepared_components": [
                "layout",
                "tableformer",
                "code_formula",
                "tesseract-cli",
            ]
        },
        load_document_protocol(),
    )


def test_document_runner_keeps_all_attempts_and_empties_partial_failure(monkeypatch) -> None:
    from experiments.benchmarks.common.contracts import BenchmarkPlan
    from experiments.benchmarks.extraction.document import runner

    first_document = _document(
        "alpha", (ExtractedSegment("alpha", 0, 5, page_number=1),),
        kind=SourceKind.PDF,
    )
    second_document = _document(
        "alpha\nbeta",
        (
            ExtractedSegment("alpha", 0, 5, page_number=1),
            ExtractedSegment("beta", 6, 10, page_number=2),
        ),
        kind=SourceKind.PDF,
    )
    outcomes = iter(
        [(first_document, 0.1), RuntimeError("second failed"), (second_document, 0.2)]
    )

    def extract_once(*_args):
        value = next(outcomes)
        if isinstance(value, Exception):
            raise value
        document, latency = value
        return document, latency

    monkeypatch.setattr(runner, "_cold_latency", lambda *_args: 0.01)
    clock = iter((0.0, 1.0, 1.1, 2.0))
    monkeypatch.setattr(runner.time, "perf_counter", lambda: next(clock))
    samples, _operational, aggregate, _parameters, _intervals, artifacts = (
        runner.evaluate_candidate(
            "docling-standard-native",
            [{"id": "sample", "kind": "pdf", "reference": "alpha"}],
            BenchmarkPlan(
                "extraction", "document-test", "development", "fixture", ("candidate",),
                seed=default_document_protocol().meta.seed,
                repetitions=3,
                warmups=0,
                bootstrap_resamples=0,
            ),
            {},
            {"device": "cpu"},
            {"sample": load_reference({"reference": "alpha"})},
            None,
            extract_once,
            default_document_protocol(),
        )
    )
    assert len(artifacts["timings"]) == 3
    assert [row["success"] for row in artifacts["timings"]] == [True, False, True]
    assert samples[0].metrics["text.content_f1"] == 0.0
    assert samples[0].metrics["reliability.candidate_failure_rate"] == 1.0
    assert samples[0].metrics["reliability.structured_output_determinism"] == 0.0
    assert aggregate["reliability.candidate_failure_rate"] == 1.0
    assert _operational["batch_pages_per_minute"] == pytest.approx(450.0)
    assert _operational["p50_warm_latency_per_page_seconds"] == pytest.approx(0.1)


def test_document_configuration_matrix_has_no_duplicate_image_modes() -> None:
    arguments = SimpleNamespace(profile="development")
    protocol = default_document_protocol()
    pdf, _ = _document_candidates("pdf", arguments, protocol)
    image, _ = _document_candidates("image", arguments, protocol)
    docx, _ = _document_candidates("docx", arguments, protocol)
    assert len(pdf) == 24
    assert len(image) == len(set(image)) == 12
    assert all("mode=full_page" in candidate for candidate in image)
    assert docx == ("docling-standard-native",)


def test_document_architecture_uses_development_before_validation(monkeypatch) -> None:
    configuration = "docling-standard|ocr=rapidocr|mode=full_page|table=fast|formula=off"
    calls = []

    def selected(_path, expected_stage, **_limits):
        calls.append(expected_stage)
        if expected_stage == "document-configuration-pdf":
            return (configuration,)
        return (configuration, "docling-vlm-granite-258m")

    monkeypatch.setattr(document_cli, "_document_selection", selected)
    development_arguments = SimpleNamespace(
        profile="development",
        comparison="architecture",
        pdf_selection=Path("configuration-decision.json"),
        image_selection=None,
    )
    development, _ = _document_candidates(
        "pdf",
        development_arguments,
        default_document_protocol(),
    )
    assert development == (
        configuration,
        "docling-vlm-granite-258m",
        "paddleocr-vl-1.6",
    )

    validation_arguments = SimpleNamespace(
        profile="validation",
        comparison="architecture",
        pdf_selection=Path("architecture-decision.json"),
        image_selection=None,
    )
    validation, _ = _document_candidates(
        "pdf",
        validation_arguments,
        default_document_protocol(),
    )
    assert validation == (configuration, "docling-vlm-granite-258m")
    assert calls == [
        "document-configuration-pdf",
        "document-architecture-development-pdf",
    ]


def test_canonical_document_preserves_exact_offsets_and_structure() -> None:
    profile = ExtractionProfile("fixture", "fixture", "1")
    document = build_structured_document(
        ExtractionRequest(Path("fixture.pdf"), "checksum", profile=profile),
        SourceKind.PDF,
        profile,
        [
            {
                "text": "Heading",
                "element_id": "h",
                "order": 0,
                "page_number": 1,
                "kind": "heading",
            },
            {
                "text": "Body",
                "element_id": "p",
                "parent_id": "h",
                "order": 1,
                "page_number": 1,
                "kind": "text",
            },
        ],
    )
    assert document.text == "Heading\n\nBody"
    assert [(segment.start, segment.end) for segment in document.segments] == [
        (0, 7),
        (9, 13),
    ]
    assert document.segments[1].parent_id == "h"


def _document(
    text: str,
    segments: tuple[ExtractedSegment, ...],
    *,
    kind: SourceKind = SourceKind.DOCX,
) -> ExtractedDocument:
    return ExtractedDocument(
        "fixture",
        "fixture",
        kind,
        "checksum",
        None,
        text,
        segments,
        ExtractionProfile("fixture", "fixture", "1"),
    )
