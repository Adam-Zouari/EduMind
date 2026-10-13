"""Video scoring regressions using saved outputs only; no models or GPU work."""

from collections import Counter
from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from experiments.benchmarks.common.metrics import normalized_tokens
from experiments.benchmarks.extraction.scoring import attempt_rates, bootstrap_sources
from experiments.benchmarks.extraction.video.metrics import (
    QUALITY_DIRECTIONS,
    _occurrence_matches,
    aggregate_quality,
    aggregate_quality_statuses,
    bootstrap_quality,
    quality_statuses,
    quality_support_counts,
    score_video,
)

MATCHING = {"content_f1_threshold": 0.5, "frame_timestamp_tolerance_seconds": 0.0}
NOT_EXECUTED = object()


def _occurrence(identifier="first", text="Alpha", start=0, end=10):
    return {"id": identifier, "text": text, "start": start, "end": end}


def _item(sample_id="video", reference="Alpha", occurrences=None, duration=10):
    return {
        "id": sample_id,
        "duration_seconds": duration,
        "reference_visual_text": reference,
        "visual_occurrences": [_occurrence()] if occurrences is None else occurrences,
    }


def _prediction(text="Alpha", timestamp=1):
    return {"text": text, "timestamp": timestamp}


def _row(sample_id="video", predictions=None, *, outcomes=("A", "A", "A"), group=None):
    first = outcomes[0] if outcomes else NOT_EXECUTED
    row = score_video(
        _item(sample_id),
        ([_prediction()] if predictions is None else predictions)
        if isinstance(first, str)
        else None,
        MATCHING,
    )
    # Deliberately retain surviving-pair rates on incomplete inputs: the video
    # helpers must reject those favorable partial reliability observations.
    row.update(
        attempt_rates([value if isinstance(value, str) else None for value in outcomes])
        if outcomes
        else {}
    )
    row.update(
        source_group_id=group or sample_id,
        scheduled_attempt_count=1 if len(outcomes) == 1 else 3,
        executed_attempt_count=sum(value is not NOT_EXECUTED for value in outcomes),
        failed_attempt_count=sum(value is None for value in outcomes),
        unexecuted_attempt_count=sum(value is NOT_EXECUTED for value in outcomes),
        first_attempt_executed=first is not NOT_EXECUTED,
        first_attempt_success=isinstance(first, str)
        if first is not NOT_EXECUTED
        else None,
        _timings=[
            {
                "sample_id": sample_id,
                "repetition": index,
                "executed": value is not NOT_EXECUTED,
                "success": isinstance(value, str)
                if value is not NOT_EXECUTED
                else None,
                "latency_seconds": 2 if value is not NOT_EXECUTED else None,
                "duration_seconds": 10,
                "selected_frame_count": 1 if value is not NOT_EXECUTED else None,
            }
            for index, value in enumerate(outcomes, 1)
        ],
    )
    return row


@pytest.mark.parametrize(
    "reference,prediction,expected",
    [("", "", 1), ("!", "...", 1), ("", "invented", 0), ("Alpha", "", 0)],
)
def test_valid_empty_conventions(reference, prediction, expected):
    row = score_video(
        _item(
            reference=reference,
            occurrences=[] if not normalized_tokens(reference) else None,
        ),
        [_prediction(prediction)],
        MATCHING,
    )
    for name in (
        "visual_content_precision",
        "visual_content_recall",
        "visual_content_f1",
    ):
        assert row[name] == expected
        assert quality_statuses(row)[name] == {"status": "scored", "reason": None}


def test_content_projects_each_line_symmetrically_and_counts_distinct_lines():
    row = score_video(
        _item(reference=["CAFÉ—Plan", "Cafe\u0301: Plan\n...", "Beta"]),
        [_prediction("café / plan\nCAFÉ: PLAN\nBeta\n!!!"), _prediction("beta", 2)],
        MATCHING,
    )
    assert row["reference_visual_unit_count"] == row["predicted_visual_unit_count"] == 2
    assert row["matched_visual_unit_count"] == 2
    assert row["predicted_visual_occurrence_count"] == 4
    assert row["visual_content_f1"] == 1


def test_exact_line_content_is_separate_from_approximate_timed_eligibility():
    row = score_video(
        _item(reference="alpha beta", occurrences=[_occurrence(text="alpha beta")]),
        [_prediction("alpha\nbeta")],
        MATCHING,
    )
    assert row["visual_content_f1"] == 0
    assert row["timed_visual_occurrence_coverage"] == 1
    assert row["assigned_visual_detection_count"] == 2
    assert row["duplicate_visual_text_rate"] == 0


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_content_f1_threshold_is_inclusive_despite_harmonic_roundoff(reverse):
    reference = " ".join(f"w{index}" for index in range(8))
    prediction = " ".join(
        [*(f"w{index}" for index in range(7)), *(f"x{index}" for index in range(13))]
    )
    if reverse:
        reference, prediction = prediction, reference
    row = score_video(
        _item(reference=reference, occurrences=[_occurrence(text=reference)]),
        [_prediction(prediction)],
        MATCHING,
    )
    assert row["visual_content_f1"] == 0
    assert row["timed_visual_occurrence_coverage"] == 1
    assert row["assigned_visual_detection_count"] == 1


@pytest.mark.parametrize(
    "timestamp,coverage", [(0.999, 0), (1, 1), (1.999, 1), (2, 0), (2.001, 0)]
)
def test_half_open_intervals_have_zero_tolerance(timestamp, coverage):
    row = score_video(
        _item(occurrences=[_occurrence(start=1, end=2)]),
        [_prediction(timestamp=timestamp)],
        MATCHING,
    )
    assert (
        row["visual_content_f1"] == 1
    )  # Useful content need not be recovered while visible.
    assert row["timed_visual_occurrence_coverage"] == coverage
    assert row["mean_visual_first_detection_delay_seconds"] == (
        timestamp - 1 if coverage else None
    )


def test_one_detection_cannot_cover_two_occurrences_and_units_stay_separate():
    item = _item(occurrences=[_occurrence("a"), _occurrence("b")])
    one = score_video(item, [_prediction()], MATCHING)
    two = score_video(item, [_prediction("Alpha\nAlpha")], MATCHING)
    assert one["matched_occurrence_count"] == 1
    assert one["timed_visual_occurrence_coverage"] == 0.5
    assert two["matched_occurrence_count"] == 2
    assert two["visual_content_f1"] == 1


def test_maximum_cardinality_wins_before_delay_or_individual_best_similarity():
    row = score_video(
        _item(
            reference=["alpha beta", "beta"],
            occurrences=[
                _occurrence("a", "alpha beta", 0, 10),
                _occurrence("b", "beta", 0, 1),
            ],
        ),
        [_prediction("beta", 0), _prediction("alpha", 9)],
        MATCHING,
    )
    assert row["matched_occurrence_count"] == 2
    assert row["mean_visual_first_detection_delay_seconds"] == 4.5


def test_delay_is_raw_seconds_and_tiny_real_improvements_are_not_rounded_away():
    row = score_video(
        _item(occurrences=[_occurrence("a", start=0), _occurrence("b", start=1e-16)]),
        [_prediction(timestamp=1)],
        MATCHING,
    )
    assert row["mean_visual_first_detection_delay_seconds"] == 1 - 1e-16
    assert row["timed_visual_occurrence_coverage"] == 0.5


def test_ties_use_stable_ids_then_chronological_frame_and_line_order():
    occurrences = [
        _occurrence("c", "beta"),
        _occurrence("b", "alpha"),
        _occurrence("a", "alpha beta"),
    ]
    detections = [
        {**_prediction("beta", 2), "index": 0},
        {**_prediction("unsupported", 0), "index": 1},
        {**_prediction("alpha", 1), "index": 2},
    ]
    matches = _occurrence_matches(occurrences, detections, MATCHING)
    assert matches == ((2, 2), (0, 0))
    tied = [
        {**_prediction("Alpha", 1), "index": 5},
        {**_prediction("Alpha", 1), "index": 2},
    ]
    assert _occurrence_matches([_occurrence()], tied, MATCHING) == ((0, 1),)


def test_fractional_timestamps_do_not_break_real_assignment_ties():
    occurrences = [
        _occurrence("d", "alpha alpha beta", 0, 1.1),
        _occurrence("b", "alpha beta", 0.1, 1.1),
        _occurrence("a", "gamma", 0.3, 1.1),
        _occurrence("c", "beta gamma", 0.4, 1.1),
    ]
    detections = [
        {**_prediction("alpha", 0.3), "index": 3},
        {**_prediction("gamma", 0.5), "index": 1},
        {**_prediction("gamma", 0.8), "index": 0},
        {**_prediction("alpha alpha beta", 1.1), "index": 2},
    ]
    assert _occurrence_matches(occurrences, detections, MATCHING) == (
        (2, 1),
        (1, 0),
        (3, 2),
    )


def _brute_matches(occurrences, detections):
    """Independent exhaustive oracle: cardinality, raw delay, stable tie key."""
    refs = sorted(range(len(occurrences)), key=lambda index: occurrences[index]["id"])
    dets = sorted(
        range(len(detections)),
        key=lambda index: (
            detections[index]["timestamp"],
            detections[index]["index"],
            index,
        ),
    )
    eligible = {}
    for left in refs:
        occurrence = occurrences[left]
        reference = Counter(normalized_tokens(occurrence["text"]))
        for right in dets:
            detection = detections[right]
            prediction = Counter(normalized_tokens(detection["text"]))
            overlap = sum((reference & prediction).values())
            similarity = (
                2 * overlap / (sum(reference.values()) + sum(prediction.values()))
            )
            if (
                occurrence["start"] <= detection["timestamp"] < occurrence["end"]
                and similarity >= 0.5
            ):
                eligible[left, right] = Fraction(detection["timestamp"]) - Fraction(
                    occurrence["start"]
                )
    candidates = []
    for assignment in product([*dets, None], repeat=len(refs)):
        real = [right for right in assignment if right is not None]
        if len(real) != len(set(real)):
            continue
        matches = tuple(
            (left, right)
            for left, right in zip(refs, assignment, strict=True)
            if right is not None
        )
        if any(match not in eligible for match in matches):
            continue
        key = (
            -len(matches),
            sum(eligible[match] for match in matches),
            tuple(
                dets.index(right) if right is not None else len(dets)
                for right in assignment
            ),
        )
        candidates.append((key, matches))
    return min(candidates)[1]


def test_matching_agrees_with_independent_exhaustive_oracle():
    rng = np.random.default_rng(42)
    texts = ["alpha", "beta", "alpha beta", "alpha alpha beta", "unsupported"]
    for _ in range(100):
        occurrences = [
            _occurrence(
                identifier, str(rng.choice(texts)), start=int(rng.integers(0, 3)), end=4
            )
            for identifier in ("c", "a", "b")
        ]
        detections = [
            {
                **_prediction(str(rng.choice(texts)), int(rng.integers(0, 5))),
                "index": index,
            }
            for index in (2, 0, 1)
        ]
        assert _occurrence_matches(occurrences, detections, MATCHING) == _brute_matches(
            occurrences, detections
        )


def test_duplicate_copies_are_exact_supported_and_scoped_to_real_reappearance():
    row = score_video(
        _item(
            reference="Chapter 1",
            duration=30,
            occurrences=[
                _occurrence("first", "Chapter 1", 10, 20),
                _occurrence("again", "Chapter 1", 20, 30),
            ],
        ),
        [
            _prediction("Chapter 1", 22),
            _prediction("Chapter", 10),
            _prediction("CHAPTER—1", 14),
            _prediction("Chapter 1", 12),
            _prediction("invented", 15),
        ],
        MATCHING,
    )
    assert row["duplicate_visual_unit_count"] == 1
    assert row["assigned_visual_detection_count"] == 4
    assert row["unassigned_visual_detection_count"] == 1
    assert row["duplicate_visual_text_rate"] == 0.25
    assert row["timed_visual_occurrence_coverage"] == 1
    assert row["mean_visual_first_detection_delay_seconds"] == 1


def test_duplicate_assignment_prefers_similarity_before_earliest_start_and_id():
    occurrences = [
        _occurrence("a", "alpha beta", 0, 10),
        _occurrence("b", "alpha", 1, 10),
    ]
    row = score_video(
        _item(occurrences=occurrences),
        [_prediction("alpha", 2), _prediction("alpha", 0)],
        MATCHING,
    )
    assert (
        row["duplicate_visual_text_rate"] == 0
    )  # Separate assignments, not two copies in a.
    tied = [
        _occurrence("z", "alpha beta", 1, 10),
        _occurrence("a", "alpha beta", 0, 10),
    ]
    row = score_video(
        _item(occurrences=tied),
        [_prediction("alpha", 0), _prediction("alpha", 2)],
        MATCHING,
    )
    assert row["duplicate_visual_text_rate"] == 0.5


@pytest.mark.parametrize("predictions", [[], [_prediction("unsupported")]])
def test_unmatched_output_is_zero_coverage_but_undefined_delay_and_duplication(
    predictions,
):
    row = score_video(_item(), predictions, MATCHING)
    statuses = quality_statuses(row)
    assert row["timed_visual_occurrence_coverage"] == 0
    assert statuses["mean_visual_first_detection_delay_seconds"] == {
        "status": "unavailable",
        "reason": "no_covered_occurrences",
    }
    assert statuses["duplicate_visual_text_rate"] == {
        "status": "unavailable",
        "reason": "no_assigned_detections",
    }


@pytest.mark.parametrize("predictions", [[], None, [_prediction("unsupported")]])
def test_text_free_reference_has_no_timing_or_supported_duplication_task(predictions):
    row = score_video(_item(reference=[], occurrences=[]), predictions, MATCHING)
    for name in (
        "timed_visual_occurrence_coverage",
        "mean_visual_first_detection_delay_seconds",
        "duplicate_visual_text_rate",
    ):
        assert row[name] is None
        assert quality_statuses(row)[name] == {
            "status": "inapplicable",
            "reason": "no_reference_occurrences",
        }


@pytest.mark.parametrize("outcomes", [(None, "A", "A"), ("A", None, None)])
def test_only_first_attempt_supplies_quality_and_planned_support_survives_failure(
    outcomes,
):
    row = _row(outcomes=outcomes)
    first_failed = outcomes[0] is None
    result = aggregate_quality([row])
    assert result["visual_content_f1"] == (None if first_failed else 1)
    assert (
        row["planned_reference_visual_unit_count"]
        == row["reference_visual_unit_count"]
        == 1
    )
    assert (
        row["planned_reference_occurrence_count"]
        == row["reference_occurrence_count"]
        == 1
    )
    assert quality_statuses(row)["visual_content_f1"] == {
        "status": "unavailable" if first_failed else "scored",
        "reason": "first_attempt_failed" if first_failed else None,
    }


@pytest.mark.parametrize(
    "outcomes,repeat,failure",
    [
        (("A", "A", "A"), 1, 0),
        (("A", "A", "B"), 1 / 3, 0),
        (("A", "B", "C"), 0, 0),
        (("A", "A", None), 1 / 3, 1 / 3),
        ((None, "A", "A"), 1 / 3, 1 / 3),
        (("A", None, None), 0, 2 / 3),
        ((None, None, None), 0, 1),
    ],
)
def test_all_three_scheduled_pairs_and_failure_outcomes(outcomes, repeat, failure):
    row = _row(predictions=[], outcomes=outcomes)
    values = aggregate_quality([row])
    assert values["repeatability_success_rate"] == repeat
    assert values["attempt_failure_rate"] == failure
    for name in ("repeatability_success_rate", "attempt_failure_rate"):
        assert quality_statuses(row)[name] == {"status": "scored", "reason": None}


@pytest.mark.parametrize(
    "outcomes", [(NOT_EXECUTED, "A", "A"), ("A", NOT_EXECUTED, "A"), ("A", "A")]
)
def test_incomplete_attempts_do_not_supply_favorable_reliability_or_replace_first(
    outcomes,
):
    row = _row(outcomes=outcomes)
    values = aggregate_quality([row, _row("complete")])
    for name in ("repeatability_success_rate", "attempt_failure_rate"):
        assert values[name] is None
        assert quality_statuses(row)[name] == {
            "status": "incomplete",
            "reason": "measurement_attempts_missing",
        }
    if outcomes[0] is NOT_EXECUTED:
        assert quality_statuses(row)["visual_content_f1"] == {
            "status": "incomplete",
            "reason": "first_attempt_not_executed",
        }
    else:
        assert row["visual_content_f1"] == 1


def test_duplicate_or_wrong_attempt_ids_are_incomplete_even_when_counts_match():
    row = _row()
    row["_timings"][2]["repetition"] = 2
    assert aggregate_quality([row])["attempt_failure_rate"] is None
    row["_timings"][2]["repetition"] = 4
    assert quality_statuses(row)["repeatability_success_rate"]["status"] == "incomplete"


def test_smoke_repeatability_is_inapplicable_not_missing_measurement():
    row = _row(outcomes=("A",))
    assert aggregate_quality([row])["repeatability_success_rate"] is None
    assert quality_statuses(row)["repeatability_success_rate"] == {
        "status": "inapplicable",
        "reason": "repeatability_not_measured",
    }
    assert aggregate_quality([row])["attempt_failure_rate"] == 0


def test_video_macro_f1_delay_and_duplication_are_not_pooled_or_rederived():
    first = _row("first")
    second = _row("second")
    first.update(
        visual_content_precision=1,
        visual_content_recall=0.5,
        visual_content_f1=2 / 3,
        mean_visual_first_detection_delay_seconds=1,
        matched_occurrence_count=100,
        duplicate_visual_text_rate=0,
        assigned_visual_detection_count=100,
    )
    second.update(
        visual_content_precision=0.5,
        visual_content_recall=1,
        visual_content_f1=2 / 3,
        mean_visual_first_detection_delay_seconds=9,
        matched_occurrence_count=1,
        duplicate_visual_text_rate=1,
        assigned_visual_detection_count=1,
    )
    result = aggregate_quality(
        [first, second, _row("failed", outcomes=(None, None, None))]
    )
    assert result["visual_content_precision"] == result["visual_content_recall"] == 0.75
    assert result["visual_content_f1"] == pytest.approx(2 / 3)
    assert result["mean_visual_first_detection_delay_seconds"] == 5
    assert result["duplicate_visual_text_rate"] == 0.5


def test_support_counts_preserve_planned_tasks_statuses_and_independent_groups():
    rows = [
        _row("a", group="source"),
        _row("b", outcomes=(None, "A", "A"), group="source"),
        _row("c", predictions=[]),
        _row("d", outcomes=(NOT_EXECUTED, "A", "A")),
    ]
    counts = quality_support_counts(rows)
    prefix = "visual_content_f1."
    expected = {
        "scheduled_count": 4,
        "eligible_count": 4,
        "contributing_count": 2,
        "independent_source_count": 3,
        "contributing_source_count": 2,
        "sample_count": 2,
        "failed_count": 1,
        "failed_first_attempt_count": 1,
        "unavailable_count": 1,
        "incomplete_count": 1,
        "planned_reference_visual_unit_count": 4,
        "contributing_reference_visual_unit_count": 2,
    }
    for name, value in expected.items():
        assert counts[prefix + name] == value
    assert counts["timed_visual_occurrence_coverage.contributing_count"] == 2
    assert counts["mean_visual_first_detection_delay_seconds.contributing_count"] == 1
    assert counts["mean_visual_first_detection_delay_seconds.sample_count"] == 1
    assert counts["attempt_failure_rate.sample_count"] == 0  # Aggregate is incomplete.


@pytest.mark.parametrize("duration", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_is_rejected_even_for_empty_or_failed_output(duration):
    for predictions in ([], None):
        with pytest.raises(ValueError, match="duration"):
            score_video(
                _item(reference=[], occurrences=[], duration=duration),
                predictions,
                MATCHING,
            )


@pytest.mark.parametrize("timestamp", [-1, 11, float("nan"), float("inf")])
def test_invalid_detection_time_is_rejected_even_for_projected_empty_text(timestamp):
    with pytest.raises(ValueError, match="timestamp"):
        score_video(_item(), [_prediction("!!!", timestamp)], MATCHING)


def test_missing_and_invalid_reference_is_not_reviewed_empty_data():
    for field in ("reference_visual_text", "visual_occurrences"):
        item = _item()
        del item[field]
        with pytest.raises(ValueError):
            score_video(item, [], MATCHING)
    with pytest.raises(ValueError, match="unique"):
        score_video(_item(occurrences=[_occurrence(), _occurrence()]), [], MATCHING)
    with pytest.raises(ValueError):
        score_video(_item(occurrences=[_occurrence(text="!!!")]), [], MATCHING)
    for text in (None, 123, ["alpha"]):
        with pytest.raises(ValueError, match="malformed"):
            score_video(_item(occurrences=[_occurrence(text=text)]), [], MATCHING)


def _bootstrap(rows, *, resamples=100, minimum_sources=1):
    return bootstrap_quality(
        rows,
        resamples=resamples,
        seed=42,
        confidence=0.95,
        minimum_sources=minimum_sources,
    )


def test_smoke_has_no_authoritative_intervals_and_unfrozen_support_has_no_draws():
    assert _bootstrap([_row()], resamples=0) == {}
    intervals = _bootstrap([_row()], minimum_sources=None)
    for interval in intervals.values():
        assert interval["lower"] is interval["upper"] is None
        assert interval["reason"] == "ci_support_not_frozen"
        assert interval["resampling_performed"] is False
        assert interval["defined_resamples"] == interval["undefined_resamples"] == 0


def test_bootstrap_preserves_all_excerpts_attempts_and_failures_inside_source_group():
    rows = [
        _row("a", group="source"),
        _row("b", predictions=[], group="source"),
        _row("c", outcomes=(None, None, None), group="source"),
    ]
    intervals = _bootstrap(rows)
    for name in QUALITY_DIRECTIONS:
        interval = intervals[name]
        assert (
            interval["independent_source_count"]
            == interval["contributing_source_count"]
            == 1
        )
        assert interval["defined_resamples"] == 100
        assert interval["undefined_resamples"] == 0
        assert interval["lower"] == interval["upper"] == interval["estimate"]
        assert interval["degenerate"] is True
    assert intervals["visual_content_f1"]["estimate"] == 0.5
    assert intervals["attempt_failure_rate"]["estimate"] == pytest.approx(1 / 3)


def test_bootstrap_does_not_prefilter_failed_unmatched_or_task_free_videos():
    rows = [
        _row("matched"),
        _row("unmatched", predictions=[]),
        _row("failed", outcomes=(None, None, None)),
    ]
    blank = _row("blank")
    blank.update(
        score_video(_item("blank", reference=[], occurrences=[]), [], MATCHING)
    )
    rows.append(blank)
    intervals = _bootstrap(rows)
    rng = np.random.default_rng(42)
    expected_draws = {name: [] for name in QUALITY_DIRECTIONS}
    for _ in range(100):
        selected = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        for name, value in aggregate_quality(selected).items():
            if value is not None:
                expected_draws[name].append(value)
    for name, draws in expected_draws.items():
        interval = intervals[name]
        assert interval["defined_resamples"] == len(draws)
        assert interval["undefined_resamples"] == 100 - len(draws)
        assert interval["independent_source_count"] == 4
        np.testing.assert_allclose(
            [interval["lower"], interval["upper"]], np.quantile(draws, [0.025, 0.975])
        )
    assert (
        intervals["mean_visual_first_detection_delay_seconds"]["undefined_resamples"]
        > 0
    )
    assert intervals["timed_visual_occurrence_coverage"]["estimate"] == 0.5
    assert intervals["mean_visual_first_detection_delay_seconds"]["estimate"] == 1


def test_ci_support_is_metric_specific_not_total_scheduled_videos():
    intervals = _bootstrap(
        [_row("a"), _row("b", predictions=[]), _row("c", outcomes=(None, None, None))],
        minimum_sources=2,
    )
    coverage = intervals["timed_visual_occurrence_coverage"]
    delay = intervals["mean_visual_first_detection_delay_seconds"]
    assert coverage["contributing_source_count"] == 2
    assert coverage["lower"] is not None
    assert delay["contributing_source_count"] == 1
    assert delay["estimate"] == 1
    assert delay["lower"] is delay["upper"] is None
    assert delay["reason"] == "insufficient_independent_sources"
    assert delay["defined_resamples"] + delay["undefined_resamples"] == 100


def test_undefined_point_has_no_bounds_or_fabricated_zero_draws():
    intervals = _bootstrap([_row("a", predictions=[]), _row("b", predictions=[])])
    delay = intervals["mean_visual_first_detection_delay_seconds"]
    assert delay["estimate"] is delay["lower"] is delay["upper"] is None
    assert delay["contributing_source_count"] == 0
    assert delay["defined_resamples"] == 0
    assert delay["undefined_resamples"] == 100
    assert intervals["timed_visual_occurrence_coverage"]["estimate"] == 0


def test_bootstrap_honors_declared_independence_and_falls_back_from_null_group():
    rows = [_row("a"), _row("b")]
    for row in rows:
        row.update(independent_source_id="original", source_group_id=None)
    assert _bootstrap(rows)["visual_content_f1"]["independent_source_count"] == 1
    for row in rows:
        row.pop("independent_source_id")
    assert _bootstrap(rows)["visual_content_f1"]["independent_source_count"] == 2


def test_empty_aggregate_is_not_an_observed_empty_video():
    with pytest.raises(ValueError, match="scheduled samples"):
        aggregate_quality([])


def test_aggregate_statuses_distinguish_no_task_failure_and_incomplete_execution():
    failed = _row(outcomes=(None, None, None))
    missing = _row(outcomes=(NOT_EXECUTED, "A", "A"))
    assert aggregate_quality_statuses([failed])["visual_content_f1"] == {
        "status": "unavailable",
        "reason": "first_attempt_failed",
    }
    assert aggregate_quality_statuses([missing])["visual_content_f1"] == {
        "status": "incomplete",
        "reason": "first_attempt_not_executed",
    }
    assert aggregate_quality_statuses([missing, _row("complete")])[
        "attempt_failure_rate"
    ] == {
        "status": "incomplete",
        "reason": "measurement_attempts_missing",
    }
    assert aggregate_quality_statuses([missing, _row("complete")])[
        "visual_content_f1"
    ] == {
        "status": "scored",
        "reason": None,
    }
    assert aggregate_quality_statuses([_row(outcomes=("A",))])[
        "repeatability_success_rate"
    ] == {
        "status": "inapplicable",
        "reason": "repeatability_not_measured",
    }


def test_incomplete_reliability_ci_reason_is_not_no_contributing_sources():
    intervals = _bootstrap(
        [_row("incomplete", outcomes=("A", NOT_EXECUTED, "A")), _row("complete")]
    )
    for name in ("repeatability_success_rate", "attempt_failure_rate"):
        interval = intervals[name]
        assert interval["estimate"] is interval["lower"] is interval["upper"] is None
        assert interval["reason"] == "measurement_attempts_missing"
        assert interval["contributing_source_count"] == 1
        assert 0 < interval["defined_resamples"] < 100
        assert interval["defined_resamples"] + interval["undefined_resamples"] == 100


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_saved_metric_is_unavailable_not_a_numeric_quality_or_ci(value):
    row = _row()
    row["visual_content_f1"] = value
    assert aggregate_quality([row])["visual_content_f1"] is None
    assert quality_statuses(row)["visual_content_f1"] == {
        "status": "unavailable",
        "reason": "non_finite_metric_value",
    }
    interval = _bootstrap([row])["visual_content_f1"]
    assert interval["lower"] is interval["upper"] is None
    assert interval["defined_resamples"] == 0


def test_operational_ratios_video_median_percentiles_and_first_selection_counts():
    # Main owns this pure worker helper; this test only consumes saved timings.
    from experiments.benchmarks.extraction.video.visual_worker import (
        _operational_statistics,
    )

    first = _row("first", outcomes=(None, "A", "A"))
    second = _row("second")
    first["selected_frame_count"] = 12  # Completed selection survives parser failure.
    second["selected_frame_count"] = 3
    for timing, latency in zip(first["_timings"], (100, 4, 8), strict=True):
        timing["latency_seconds"] = latency
    for timing, latency in zip(second["_timings"], (10, 12, 14), strict=True):
        timing.update(latency_seconds=latency, duration_seconds=30)
    values = _operational_statistics([first, second])
    assert values["visual_real_time_factor"] == pytest.approx(48 / 110)
    assert values["p50_warm_visual_latency_seconds"] == 9
    assert values["p95_warm_visual_latency_seconds"] == pytest.approx(11.7)
    assert values["mean_selected_frames_per_video"] == 7.5


def test_failed_speed_is_not_successful_speed_and_known_zero_selection_is_retained():
    from experiments.benchmarks.extraction.video.visual_worker import (
        _operational_statistics,
    )

    failed = _row(outcomes=(None, None, None))
    failed["selected_frame_count"] = 0  # Observed mandatory-frame-zero violation.
    values = _operational_statistics([failed])
    for name in (
        "visual_real_time_factor",
        "p50_warm_visual_latency_seconds",
        "p95_warm_visual_latency_seconds",
    ):
        assert values[name] is None
    assert values["mean_selected_frames_per_video"] == 0
    failed["selected_frame_count"] = None
    failed["_timings"][1]["selected_frame_count"] = 12
    assert _operational_statistics([failed])["mean_selected_frames_per_video"] is None


def test_operational_bootstrap_preserves_failed_first_selection_and_every_repetition():
    from experiments.benchmarks.extraction.video.visual_worker import (
        _operational_statistics,
    )

    rows = [
        _row("first", outcomes=(None, "A", "A"), group="source"),
        _row("second", group="source"),
    ]
    for row in rows:
        row["selected_frame_count"] = 12
    intervals = bootstrap_sources(
        rows,
        _operational_statistics,
        resamples=100,
        seed=42,
        confidence=0.95,
        minimum_sources=1,
    )
    for interval in intervals.values():
        assert interval["independent_source_count"] == 1
        assert interval["lower"] == interval["upper"] == interval["estimate"]
        assert interval["defined_resamples"] == 100
        assert interval["undefined_resamples"] == 0
