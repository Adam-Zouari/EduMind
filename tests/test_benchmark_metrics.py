from __future__ import annotations

import pytest

from experiments.benchmarks.common.metrics import (
    average_precision_at_k,
    balanced_accuracy,
    character_error_rate,
    citation_scores,
    context_precision_at_k,
    context_recall,
    exact_match,
    hit_rate_at_k,
    interval_overlap,
    merge_intervals,
    ndcg_at_k,
    paired_bootstrap_interval,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_grades,
    rouge_l,
    token_f1,
    word_error_rate,
)
from experiments.benchmarks.extraction.metrics import (
    content_scores,
    reading_order_accuracy,
    structured_document_scores,
)
from edumind.extraction import (
    ExtractedDocument,
    ExtractedSegment,
    ExtractionProfile,
    SegmentKind,
    SourceKind,
)
from experiments.benchmarks.rag.chunking_embedding.strategies import (
    build_chunking_strategy,
)
from edumind.extraction.structured import markdown_segments


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
    paired = paired_bootstrap_interval([2, 4], [1, 1], resamples=100)
    assert paired.estimate == 2.0
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_interval([])


def test_extraction_text_metrics_have_known_ranges_and_directions() -> None:
    scores = content_scores("alpha beta", "alpha extra")
    assert scores["content_precision"] == 0.5
    assert scores["content_recall"] == 0.5
    assert scores["missing_text_rate"] == 0.5
    assert scores["hallucinated_text_rate"] == 0.5
    assert reading_order_accuracy("alpha beta gamma", "alpha gamma beta") < 1.0


def test_structured_metrics_score_verified_tables_and_formulas() -> None:
    table, formula = "| H | V |\n|---|---|\n| a | 1 |", "$$x^2$$"
    text = f"{table}\n{formula}"
    document = ExtractedDocument(
        "fixture.md",
        "fixture.md",
        SourceKind.PDF,
        "checksum",
        "application/pdf",
        text,
        (
            ExtractedSegment(
                table,
                0,
                len(table),
                kind=SegmentKind.TABLE,
                structured_content={"rows": [["H", "V"], ["a", "1"]]},
            ),
            ExtractedSegment(
                formula,
                len(table) + 1,
                len(text),
                kind=SegmentKind.FORMULA,
                structured_content={"latex": "x^2"},
            ),
        ),
        ExtractionProfile("fixture", "fixture", "1"),
    )
    scores = structured_document_scores(
        [
            {"kind": "table", "rows": [["H", "V"], ["a", "1"]]},
            {"kind": "formula", "latex": "x^2"},
        ],
        document,
    )
    assert scores["table_content_f1"] == 1.0
    assert scores["table_structure_f1"] == 1.0
    assert scores["formula_exact_match"] == 1.0
    assert structured_document_scores([], document) == {}

    compact_markdown = "# Results\n| H | V |\n|---|---|\n| a | 1 |\n$$x^2$$"
    parsed_kinds = [kind for _, _, kind, _ in markdown_segments(compact_markdown)]
    assert SegmentKind.HEADING in parsed_kinds
    assert SegmentKind.TABLE in parsed_kinds
    assert SegmentKind.FORMULA in parsed_kinds


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
    from experiments.benchmarks.extraction.metrics import page_scores

    text = "beta\nalpha"
    document = ExtractedDocument(
        "fixture.pdf",
        "fixture.pdf",
        SourceKind.PDF,
        "checksum",
        "application/pdf",
        text,
        (
            ExtractedSegment("beta", 0, 4, page_number=1),
            ExtractedSegment("alpha", 5, 10, page_number=2),
        ),
        ExtractionProfile("fixture", "fixture", "1"),
    )
    scores = page_scores(
        {"kind": "pdf", "reference_page_texts": ["alpha", "beta"]}, document
    )
    assert scores["page_coverage"] == 1.0
    assert scores["page_attribution_accuracy"] == 0.0
    assert scores["page_content_f1"] == 0.0
