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
    holm_adjust,
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
from experiments.benchmarks.common.text_metrics import content_scores, reading_order_accuracy


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


def test_bootstrap_and_holm_are_deterministic() -> None:
    first = paired_bootstrap_interval([1, 2, 3], resamples=500, seed=42)
    second = paired_bootstrap_interval([1, 2, 3], resamples=500, seed=42)
    assert first == second
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted["a"] <= adjusted["c"] <= adjusted["b"]


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
