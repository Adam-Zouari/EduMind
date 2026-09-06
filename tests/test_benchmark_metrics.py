from __future__ import annotations

import pytest
from types import SimpleNamespace
from pathlib import Path

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
    aggregate_evaluations,
    score_document,
)
from experiments.benchmarks.extraction.document.adapters import _paddle_blocks
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
from experiments.benchmarks.extraction.run_stage import _document_candidates


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
    metrics, intervals = aggregate_evaluations([result, result], resamples=50, seed=42)
    assert metrics["text.content_f1"] == 1.0
    assert metrics["text.docx.content_f1"] == 1.0
    assert metrics["text.docx_native.content_f1"] == 1.0
    assert intervals["text.content_f1"]["lower"] == 1.0


def test_table_metrics_separate_detection_content_and_tree_similarity(monkeypatch) -> None:
    from experiments.benchmarks.extraction.document import metrics as document_metrics

    monkeypatch.setattr(
        document_metrics,
        "_teds",
        lambda _reference, _prediction, *, structure_only=False: (
            0.8 if not structure_only else 0.9
        ),
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
        spans = build_chunking_strategy(name, tokenizer=CharacterTokenizer()).split(text)
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


def test_document_configuration_matrix_has_no_duplicate_image_modes() -> None:
    arguments = SimpleNamespace(profile="standard")
    path = Path("experiments/benchmarks/extraction/document/candidates.yaml")
    pdf, _ = _document_candidates("pdf", arguments, path)
    image, _ = _document_candidates("image", arguments, path)
    docx, _ = _document_candidates("docx", arguments, path)
    assert len(pdf) == 24
    assert len(image) == len(set(image)) == 12
    assert all("mode=full_page" in candidate for candidate in image)
    assert docx == ("docling-standard-native",)


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
