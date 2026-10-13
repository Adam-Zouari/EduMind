"""Pure ASR scoring regressions: no runtimes, model loading, or inference."""

from __future__ import annotations

import json
from copy import deepcopy
from itertools import pairwise, product

import numpy as np
import pytest

from experiments.benchmarks.extraction.audio import evaluate
from experiments.benchmarks.extraction.scoring import attempt_rates


def _segment(text, start=0.0, end=1.0):
    return {"text": text, "start": start, "end": end}


def _speech(sample_id, reference="go now", prediction="go now", *, group=None):
    row = evaluate.score_speech(
        {
            "id": sample_id,
            "reference": reference,
            "reference_segments": [_segment(reference)] if reference else [],
            "duration_seconds": 2.0,
        },
        prediction,
        [_segment(prediction)] if prediction else [],
        quality_latency_seconds=0.5,
        alignment_threshold=0.5,
        timestamp_tolerance_seconds=0.1,
    )
    row.update(
        source_group_id=group or sample_id,
        attempt_failure_rate=0.0,
        transcript_repeatability_success_rate=1.0,
    )
    return row


def _failed(sample_id, *, sample_type="speech", executed=True):
    row = {
        "sample_id": sample_id,
        "source_group_id": sample_id,
        "sample_type": sample_type,
        "duration_seconds": 2.0,
        "first_attempt_success": False if executed else None,
        "first_attempt_executed": executed,
        "metric_status": "unavailable" if executed else "incomplete",
        "reason": "first_attempt_failed" if executed else "first_attempt_not_executed",
        "reference_word_count": None,
        "reference_character_count": None,
        "reference_timed_segment_count": None,
        "planned_reference_word_count": 2 if sample_type == "speech" else 0,
        "planned_reference_character_count": 6 if sample_type == "speech" else 0,
        "planned_reference_timed_segment_count": 1 if sample_type == "speech" else 0,
        "scheduled_attempt_count": 3 if sample_type == "speech" else 1,
        "attempt_failure_rate": 1.0 if executed else None,
        "transcript_repeatability_success_rate": 0.0
        if executed and sample_type == "speech"
        else None,
    }
    return row


def _timing(
    row, latency=0.5, *, repetition=1, executed=True, success=True, labeled=True
):
    timing = {
        "sample_id": row["sample_id"],
        "repetition": repetition,
        "executed": executed,
        "success": success if executed else None,
        "latency_seconds": latency if executed else None,
        "duration_seconds": row["duration_seconds"],
    }
    if labeled:
        timing["sample_type"] = row["sample_type"]
    return timing


def _aggregate(rows, timings, **kwargs):
    return evaluate.aggregate(
        rows,
        timings,
        cold_model_load_seconds=None,
        peak_process_tree_ram_mb=None,
        peak_vram_mb=None,
        resamples=kwargs.pop("resamples", 0),
        seed=42,
        confidence=0.95,
        minimum_sources=kwargs.pop("minimum_sources", 1),
        minimum_latency_sources=kwargs.pop("minimum_latency_sources", 1),
        **kwargs,
    )


@pytest.mark.parametrize(
    "reference,prediction,wer,cer,substitution,deletion,insertion",
    [
        ("", "", 0, 0, 0, 0, 0),
        ("!", "...", 0, 0, 0, 0, 0),
        ("", "go now", 2, 6, 0, 0, 2),
        ("go now", "", 1, 1, 0, 1, 0),
        ("go", "go now now", 2, 4, 0, 0, 2),
    ],
)
def test_completed_empty_reference_conventions(
    reference, prediction, wer, cer, substitution, deletion, insertion
):
    row = _speech("clip", reference, prediction)
    values, statuses = evaluate.sample_metrics(row)
    pooled = evaluate.aggregate_rows([row], [])
    expected = {
        "word_error_rate": wer,
        "character_error_rate": cer,
        "word_substitution_rate": substitution,
        "word_deletion_rate": deletion,
        "word_insertion_rate": insertion,
    }
    for name, value in expected.items():
        assert values[name] == pooled[name] == pytest.approx(value)
        assert statuses[name] == {"status": "scored", "reason": None}
    assert sum(
        values[name]
        for name in (
            "word_substitution_rate",
            "word_deletion_rate",
            "word_insertion_rate",
        )
    ) == pytest.approx(wer)


def test_pool_counts_before_division_and_exclude_failed_first_and_controls():
    rows = [
        _speech("long", "go now please", "go now please"),
        _speech("short", "go", "no"),
        _speech("empty-reference", "", "go now"),
        _failed("failure"),
        evaluate.score_nonspeech(
            {"id": "control", "duration_seconds": 2.0},
            "invented many words",
            latency_seconds=0.5,
        ),
    ]
    result = evaluate.aggregate_rows(rows, [])
    assert result["word_error_rate"] == pytest.approx(3 / 4)
    assert result["word_substitution_rate"] == pytest.approx(1 / 4)
    assert result["word_insertion_rate"] == pytest.approx(2 / 4)
    assert result["word_deletion_rate"] == 0
    assert result["character_error_rate"] == pytest.approx(7 / 15)
    assert result["unexpected_empty_transcript_rate"] == 0
    assert result["nonspeech_false_transcription_rate"] == 1


@pytest.mark.parametrize(
    "executed,expected_status", [(True, "unavailable"), (False, "incomplete")]
)
def test_failed_first_keeps_planned_eligibility_and_null_quality(
    executed, expected_status
):
    row = _failed("failed", executed=executed)
    values, statuses = evaluate.sample_metrics(row)
    for name in (
        "word_error_rate",
        "character_error_rate",
        "word_substitution_rate",
        "word_deletion_rate",
        "word_insertion_rate",
        "timestamp_boundary_mae_seconds",
        "timestamp_alignment_coverage",
        "unexpected_empty_transcript_rate",
    ):
        assert values[name] is None
        assert statuses[name]["status"] == expected_status
        assert statuses[name]["reason"] == row["reason"]
    result = evaluate.aggregate_rows([row], [])
    assert result["word_error_rate"] is None
    assert result["timestamp_alignment_coverage"] is None
    assert result["unexpected_empty_transcript_rate"] is None


def test_timestamp_and_empty_event_denominators_are_completed_first_only():
    aligned = _speech("aligned")
    empty = _speech("empty", prediction="")
    failed = _failed("failed")
    timings = [_timing(aligned), _timing(empty)] + [
        _timing(failed, success=False, repetition=i) for i in range(1, 4)
    ]
    metrics, _ = _aggregate([aligned, empty, failed], timings)
    assert metrics["word_error_rate"] == 0.5
    assert metrics["unexpected_empty_transcript_rate"] == 0.5
    assert metrics["timestamp_boundary_mae_seconds"] == 0
    assert metrics["timestamp_alignment_coverage"] == 0.5
    assert metrics["timestamp_alignment_coverage.scheduled_count"] == 3
    assert metrics["timestamp_alignment_coverage.contributing_count"] == 2
    assert metrics["timestamp_boundary_mae_seconds.contributing_count"] == 1
    assert metrics["timestamp_alignment_coverage.failed_first_attempt_count"] == 1
    assert (
        metrics["timestamp_alignment_coverage.planned_reference_timed_segment_count"]
        == 3
    )
    assert (
        metrics[
            "timestamp_alignment_coverage.contributing_reference_timed_segment_count"
        ]
        == 2
    )
    assert metrics["word_error_rate.planned_reference_word_count"] == 6
    assert metrics["word_error_rate.contributing_reference_word_count"] == 4
    assert (
        empty["metric_statuses"]["timestamp_boundary_mae_seconds"]["reason"]
        == "no_aligned_boundaries"
    )
    assert "prediction" not in aligned and "transcript" not in aligned


def test_nonspeech_completed_event_denominator_excludes_failures():
    rows = [
        evaluate.score_nonspeech(
            {"id": f"control-{i}", "duration_seconds": 2.0}, text, latency_seconds=0.5
        )
        for i, text in enumerate(("", "invented"))
    ]
    failed = _failed("failed-control", sample_type="nonspeech")
    rows.append(failed)
    timings = [_timing(row, success=row["first_attempt_success"]) for row in rows]
    result = evaluate.aggregate_rows(rows, timings)
    assert result["nonspeech_false_transcription_rate"] == 0.5
    assert result["attempt_failure_rate"] == pytest.approx(1 / 3)
    assert result["word_error_rate"] is None
    assert result["real_time_factor"] is None
    assert (
        evaluate.aggregate_rows([failed], [timings[-1]])[
            "nonspeech_false_transcription_rate"
        ]
        is None
    )


def test_smoke_repeatability_has_documented_inapplicability_reason():
    row = _speech("smoke")
    row["transcript_repeatability_success_rate"] = None
    values, statuses = evaluate.sample_metrics(row, timing_rows=[_timing(row)])
    assert values["transcript_repeatability_success_rate"] is None
    assert statuses["transcript_repeatability_success_rate"] == {
        "status": "inapplicable",
        "reason": "repeatability_not_measured",
    }


def test_all_recorded_failures_have_zero_repeatability_and_one_failure_rate():
    row = _failed("failed")
    timings = [_timing(row, success=False, repetition=i) for i in range(1, 4)]
    result = evaluate.aggregate_rows([row], timings)
    assert result["transcript_repeatability_success_rate"] == 0
    assert result["attempt_failure_rate"] == 1
    assert result["word_error_rate"] is None
    assert result["real_time_factor"] is None


@pytest.mark.parametrize(
    "outcomes,expected",
    [
        (("A", "A", "A"), 1),
        (("A", "A", "B"), 1 / 3),
        (("A", "B", "C"), 0),
        (("A", "A", None), 1 / 3),
        (("A", "B", None), 0),
        (("A", None, None), 0),
        ((None, None, None), 0),
        (("", "", ""), 1),
    ],
)
def test_scheduled_repeat_pairs_credit_only_valid_agreement(outcomes, expected):
    projected = [
        evaluate.normalize_transcript(text) if text is not None else None
        for text in outcomes
    ]
    rates = attempt_rates(projected)
    row = (
        _speech("repeat", prediction=outcomes[0])
        if outcomes[0] is not None
        else _failed("repeat")
    )
    row.update(
        scheduled_attempt_count=3,
        transcript_repeatability_success_rate=rates["repeatability_success_rate"],
    )
    timings = [
        _timing(row, repetition=index, success=text is not None)
        for index, text in enumerate(outcomes, 1)
    ]
    result = evaluate.aggregate_rows([row], timings)
    assert result["transcript_repeatability_success_rate"] == pytest.approx(expected)
    assert result["attempt_failure_rate"] == pytest.approx(
        sum(text is None for text in outcomes) / 3
    )


def test_legitimately_empty_reference_has_no_timing_or_empty_event_task():
    row = _speech("empty-reference", "", "")
    values, statuses = evaluate.sample_metrics(row)
    for name in (
        "timestamp_boundary_mae_seconds",
        "timestamp_alignment_coverage",
        "unexpected_empty_transcript_rate",
    ):
        assert values[name] is None
        assert statuses[name] == {
            "status": "inapplicable",
            "reason": "no_reference_task",
        }


@pytest.mark.parametrize("unexecuted_type", ["speech", "nonspeech"])
def test_unexecuted_attempt_is_not_a_failure_and_does_not_erase_completed_quality(
    unexecuted_type,
):
    completed = _speech("completed")
    incomplete = _failed("unexecuted", sample_type=unexecuted_type, executed=False)
    incomplete["unexecuted_attempt_count"] = incomplete["scheduled_attempt_count"]
    timings = [_timing(completed)] + [
        _timing(incomplete, executed=False, repetition=i)
        for i in range(1, incomplete["scheduled_attempt_count"] + 1)
    ]
    metrics, _ = _aggregate([completed, incomplete], timings)
    assert metrics["word_error_rate"] == 0
    assert metrics["timestamp_alignment_coverage"] == 1
    assert metrics["attempt_failure_rate"] is None
    assert metrics["transcript_repeatability_success_rate"] == (
        None if unexecuted_type == "speech" else 1
    )
    assert metrics["real_time_factor"] == 0.25
    assert metrics["p50_warm_clip_latency_seconds"] == 0.5
    assert metrics["word_error_rate.failed_first_attempt_count"] == 0
    assert (
        incomplete["metric_statuses"]["attempt_failure_rate"]["status"] == "incomplete"
    )


def test_later_missing_attempts_do_not_repair_or_erase_first_attempt():
    row = _speech("partial")
    row.update(
        scheduled_attempt_count=3,
        executed_attempt_count=2,
        unexecuted_attempt_count=1,
        transcript_repeatability_success_rate=None,
    )
    timings = [
        _timing(row, repetition=1),
        _timing(row, success=False, repetition=2),
        _timing(row, executed=False, repetition=3),
    ]
    metrics, _ = _aggregate([row], timings)
    assert metrics["word_error_rate"] == 0
    for name in ("attempt_failure_rate", "transcript_repeatability_success_rate"):
        assert metrics[name] is None
        assert row["metric_statuses"][name] == {
            "status": "incomplete",
            "reason": "measurement_attempts_missing",
        }


@pytest.mark.parametrize("repetitions", [(1, 2), (1, 2, 2), (1, 2, 4)])
def test_missing_or_duplicated_required_attempt_records_are_incomplete(repetitions):
    row = _speech("inventory")
    row["scheduled_attempt_count"] = 3
    timings = [_timing(row, repetition=index) for index in repetitions]
    metrics, _ = _aggregate([row], timings)
    assert metrics["word_error_rate"] == 0
    for name in ("attempt_failure_rate", "transcript_repeatability_success_rate"):
        assert metrics[name] is None
        assert row["metric_statuses"][name]["status"] == "incomplete"


def test_successful_later_attempt_is_timing_only_after_failed_first():
    row = _failed("failed-first")
    row.update(attempt_failure_rate=1 / 3, transcript_repeatability_success_rate=1 / 3)
    timings = [
        _timing(row, success=False),
        _timing(row, latency=1, repetition=2),
        _timing(row, latency=3, repetition=3),
    ]
    metrics, _ = _aggregate([row], timings)
    assert metrics["word_error_rate"] is None
    assert metrics["timestamp_alignment_coverage"] is None
    assert metrics["attempt_failure_rate"] == pytest.approx(1 / 3)
    assert metrics["transcript_repeatability_success_rate"] == pytest.approx(1 / 3)
    assert metrics["real_time_factor"] == 1
    assert metrics["p50_warm_clip_latency_seconds"] == 2


@pytest.mark.parametrize(
    "latency,duration",
    [
        (None, 2),
        (float("nan"), 2),
        (float("inf"), 2),
        (-1, 2),
        (1, 0),
        (1, float("nan")),
    ],
)
def test_invalid_completed_timing_is_unavailable_without_fabricated_speed(
    latency, duration
):
    row = _speech("timing")
    timing = _timing(row, latency=latency)
    timing["duration_seconds"] = duration
    result = evaluate.aggregate_rows([row], [timing])
    assert result["attempt_failure_rate"] == 0
    assert all(result[name] is None for name in evaluate.TIMING_METRICS)


def test_speed_pools_completed_durations_and_takes_per_clip_medians():
    one, two = _speech("one"), _speech("two")
    timings = [
        _timing(one, latency=value, repetition=i)
        for i, value in enumerate((1, 3, 5), 1)
    ] + [
        _timing(two, latency=9),
        _timing(two, latency=100, success=False, repetition=2),
    ]
    control = evaluate.score_nonspeech(
        {"id": "control", "duration_seconds": 2}, "", latency_seconds=100
    )
    result = evaluate.aggregate_rows(
        [one, two, control], timings + [_timing(control, latency=100)]
    )
    assert result["real_time_factor"] == 18 / 8
    assert result["p50_warm_clip_latency_seconds"] == 6
    assert result["p95_warm_clip_latency_seconds"] == pytest.approx(8.7)


@pytest.mark.parametrize("labeled", [False, True])
def test_bootstrap_timing_estimates_equal_points_with_duplicated_clip_copies(labeled):
    rows = [_speech(str(i)) for i in range(3)]
    timings = [
        _timing(row, latency=latency, labeled=labeled)
        for row, latency in zip(rows, (1, 2, 100))
    ]
    # Old, ordinary pure-scoring fixtures need not carry execution metadata.
    if not labeled:
        for timing in timings:
            timing.pop("executed")
            timing.pop("success")
    metrics, intervals = _aggregate(rows, timings, resamples=200)
    expected = []
    rng = np.random.default_rng(42)
    for _ in range(200):
        draw = rng.integers(0, 3, 3)
        expected.append(float(np.median(np.asarray((1, 2, 100))[draw])))
    bound = intervals["p50_warm_clip_latency_seconds"]
    assert bound["estimate"] == metrics["p50_warm_clip_latency_seconds"] == 2
    assert (bound["lower"], bound["upper"]) == tuple(
        np.quantile(expected, (0.025, 0.975))
    )
    assert bound["defined_resamples"] == 200
    assert intervals["real_time_factor"]["estimate"] == metrics["real_time_factor"]
    assert [row["sample_id"] for row in rows] == ["0", "1", "2"]


def test_bootstrap_resamples_whole_sources_with_failures_and_separate_controls():
    rows = [
        _speech("a1", group="a"),
        _speech("a2", prediction="", group="a"),
        _failed("b"),
    ]
    controls = [
        evaluate.score_nonspeech(
            {"id": name, "duration_seconds": 2}, text, latency_seconds=0.5
        )
        for name, text in (("c", ""), ("d", "invented"))
    ]
    rows.extend(controls)
    timings = []
    for row in rows:
        for i in range(1, 4 if row["sample_type"] == "speech" else 2):
            timings.append(
                _timing(row, repetition=i, success=row["first_attempt_success"])
            )
    metrics, intervals = _aggregate(rows, timings, resamples=200)
    rng = np.random.default_rng(42)
    defined, expected_failure, expected_controls = 0, [], []
    for _ in range(200):
        speech_draw = rng.integers(0, 2, 2)
        control_draw = rng.integers(0, 2, 2)
        a_count = sum(speech_draw == 0)
        b_count = 2 - a_count
        defined += bool(a_count)
        expected_failure.append(3 * b_count / (6 * a_count + 3 * b_count + 2))
        expected_controls.append(float(np.mean(control_draw)))
    for name in (
        "word_error_rate",
        "timestamp_alignment_coverage",
        "unexpected_empty_transcript_rate",
    ):
        assert intervals[name]["defined_resamples"] == defined
        assert intervals[name]["undefined_resamples"] == 200 - defined
        assert intervals[name]["lower"] == intervals[name]["upper"] == 0.5
    assert intervals["attempt_failure_rate"]["defined_resamples"] == 200
    assert (
        intervals["attempt_failure_rate"]["lower"],
        intervals["attempt_failure_rate"]["upper"],
    ) == tuple(np.quantile(expected_failure, (0.025, 0.975)))
    assert (
        intervals["nonspeech_false_transcription_rate"]["lower"],
        intervals["nonspeech_false_transcription_rate"]["upper"],
    ) == tuple(np.quantile(expected_controls, (0.025, 0.975)))
    assert intervals["word_error_rate"]["stratum_source_counts"] == {
        "speech": 2,
        "nonspeech": 2,
    }
    assert metrics["word_error_rate.scheduled_source_count"] == 4
    assert metrics["word_error_rate.contributing_source_count"] == 1


def test_bootstrap_zero_reference_draws_use_insertion_counts():
    rows = [_speech("empty", "", "go now"), _speech("words")]
    _, intervals = _aggregate(rows, [_timing(row) for row in rows], resamples=200)
    assert intervals["word_error_rate"]["upper"] == 4
    assert intervals["character_error_rate"]["upper"] == 12
    assert intervals["word_substitution_rate"]["upper"] == 0
    assert intervals["word_deletion_rate"]["upper"] == 0
    assert intervals["word_insertion_rate"]["upper"] == 4


def test_bootstrap_support_is_metric_specific_and_latency_threshold_is_separate():
    rows = [_speech("success"), _failed("failure")]
    timings = [_timing(rows[0])] + [
        _timing(rows[1], success=False, repetition=i) for i in range(1, 4)
    ]
    _, intervals = _aggregate(
        rows, timings, resamples=50, minimum_sources=2, minimum_latency_sources=1
    )
    assert intervals["word_error_rate"]["lower"] is None
    assert intervals["word_error_rate"]["reason"] == "insufficient_independent_sources"
    assert intervals["word_error_rate"]["contributing_source_count"] == 1
    assert intervals["attempt_failure_rate"]["lower"] is not None
    assert intervals["p50_warm_clip_latency_seconds"]["lower"] == 0.5
    assert intervals["p50_warm_clip_latency_seconds"]["undefined_resamples"] > 0


@pytest.mark.parametrize(
    "segment",
    [
        {"start": 0, "end": 1},
        _segment(None),
        _segment(42),
        _segment(""),
        _segment("!", float("nan"), 1),
        _segment("!", 0, None),
        _segment("!", 0, 3),
    ],
)
def test_prediction_validation_does_not_hide_invalid_nonlexical_units(segment):
    with pytest.raises(ValueError, match="timestamp"):
        evaluate.validate_prediction("", [segment], 2.0, 0.1)


@pytest.mark.parametrize("label", ["reference", "prediction"])
@pytest.mark.parametrize(
    "start,end", [(False, 1), (True, 2), (0, True), ("0", 1), (0, "1"), ("0", "1")]
)
def test_timestamp_schema_rejects_boolean_and_string_boundaries(label, start, end):
    with pytest.raises(ValueError, match=f"ASR {label} timestamp segment is invalid"):
        evaluate._segments([_segment("go", start, end)], label)


@pytest.mark.parametrize("start,end", [(0, 1), (0.0, 1.0), (0, 1.5), (0.5, 2)])
def test_timestamp_schema_accepts_actual_integer_and_float_boundaries(start, end):
    assert evaluate._segments([_segment("go", start, end)], "prediction") == [
        _segment("go", float(start), float(end))
    ]


@pytest.mark.parametrize(
    "text,segments",
    [
        ("go", []),
        ("", [_segment("go")]),
        ("go now", [_segment("go")]),
        ("go now", [_segment("go", 0.5, 1), _segment("now", 0.1, 0.4)]),
    ],
)
def test_prediction_validation_rejects_missing_contradictory_or_unordered_timestamps(
    text, segments
):
    with pytest.raises(ValueError):
        evaluate.validate_prediction(text, segments, 2.0, 0.1)


def test_prediction_projection_and_boundary_tolerance_are_not_output_repairs():
    assert (
        evaluate.normalize_transcript("  CAFÉ—Straße! e\u0301  42 ")
        == "café strasse é 42"
    )
    assert evaluate.validate_prediction("GO!", [_segment("go", 0, 2.1)], 2.0, 0.1) == [
        _segment("go", 0, 2.1)
    ]
    assert evaluate.validate_prediction("!", [_segment("...", 0, 1)], 2.0, 0.1) == []


def _span_objective(matches):
    return (
        sum(match[3] for match in matches),
        len(matches),
        tuple((match[1], match[2]) for match in matches),
    )


def _brute_matches(reference, predicted, reference_index=0, first_unit=0):
    if reference_index == len(reference):
        yield ()
        return
    yield from _brute_matches(reference, predicted, reference_index + 1, first_unit)
    expected = evaluate.normalize_transcript(reference[reference_index]["text"]).split()
    for start in range(first_unit, len(predicted)):
        for end in range(start, len(predicted)):
            observed = evaluate.normalize_transcript(
                " ".join(segment["text"] for segment in predicted[start : end + 1])
            ).split()
            # Independent multiset-F1 oracle (not the implementation's helper).
            overlap = sum(
                min(expected.count(token), observed.count(token))
                for token in set(expected)
            )
            similarity = (
                2 * overlap / (len(expected) + len(observed))
                if expected or observed
                else 0
            )
            if similarity >= 0.5:
                for tail in _brute_matches(
                    reference, predicted, reference_index + 1, end + 1
                ):
                    yield ((reference_index, start, end, similarity), *tail)


def test_ordered_span_dp_agrees_with_exhaustive_objective_and_never_reuses_units():
    for refs in product(("a", "b", "a b"), repeat=2):
        for predictions in product(("a", "b", "a b"), repeat=3):
            reference = [_segment(text) for text in refs]
            predicted = [_segment(text) for text in predictions]
            options = list(_brute_matches(reference, predicted))
            best = min(
                options,
                key=lambda matches: (
                    -_span_objective(matches)[0],
                    -len(matches),
                    _span_objective(matches)[2],
                ),
            )
            actual = evaluate._ordered_span_matches(reference, predicted, threshold=0.5)
            actual_score, count, positions = _span_objective(actual)
            best_score, best_count, best_positions = _span_objective(best)
            assert actual_score == pytest.approx(best_score)
            assert count == best_count
            assert positions == best_positions
            assert all(
                left[0] < right[0] and left[2] < right[1]
                for left, right in pairwise(actual)
            )
            assert actual == evaluate._ordered_span_matches(
                reference, predicted, threshold=0.5
            )


def test_timestamp_threshold_envelope_and_no_broad_segment_reuse():
    references = [_segment("a b", 0, 1), _segment("c", 1, 2)]
    predictions = [
        _segment("a", 0.1, 1.8),
        _segment("b", 0.2, 0.8),
        _segment("c", 1.1, 1.9),
    ]
    totals = evaluate._timestamp_totals(
        references, predictions, alignment_threshold=0.5
    )
    assert totals["aligned_timed_segment_count"] == 2
    assert totals["timestamp_boundary_count"] == 4
    assert totals["timestamp_boundary_error_seconds"] == pytest.approx(1.1)
    broad = evaluate._timestamp_totals(
        [_segment("a"), _segment("b")], [_segment("a b")], alignment_threshold=0.5
    )
    assert broad["aligned_timed_segment_count"] == 1
    assert evaluate._ordered_span_matches(
        [_segment("a b c")], [_segment("a")], threshold=0.5
    )
    assert not evaluate._ordered_span_matches(
        [_segment("a b c d")], [_segment("a")], threshold=0.5
    )


def test_aggregate_status_records_are_complete_read_only_and_do_not_rescore(
    monkeypatch,
):
    row = _speech("smoke")
    row.update(scheduled_attempt_count=1, transcript_repeatability_success_rate=None)
    rows, timings = [row], [_timing(row)]
    metrics, _ = _aggregate(rows, timings)
    metrics.update(cold_model_load_seconds=1.0, peak_process_tree_ram_mb=100.0)
    metrics["word_error_rate"] = 0.37  # Reporting must preserve an existing estimate.
    original = deepcopy((metrics, rows, timings))

    def no_rescore(*_args, **_kwargs):
        pytest.fail("Status reporting must not recompute scores or inventory")

    for name in (
        "aggregate_rows",
        "sample_metrics",
        "process_words",
        "process_characters",
        "_ordered_span_matches",
        "_attempts_incomplete",
    ):
        monkeypatch.setattr(evaluate, name, no_rescore)
    records = evaluate.metric_statuses(metrics, rows, timings, "cpu")
    assert set(records) == set(evaluate.METRIC_DIRECTIONS)
    assert (metrics, rows, timings) == original
    for name, record in records.items():
        assert record["value"] == metrics[name]
        assert record["comparison_status"] == "complete"
        assert record["comparison_reason"] is None
        for key, value in metrics.items():
            if key.startswith(name + "."):
                assert record["support"][key[len(name) + 1 :]] == value
    assert all(records[name]["status"] == "scored" for name in evaluate.PRIMARY_METRICS)
    assert records["transcript_repeatability_success_rate"]["status"] == "inapplicable"
    assert (
        records["transcript_repeatability_success_rate"]["reason"]
        == "repeatability_not_measured"
    )
    assert records["peak_vram_mb"]["status"] == "inapplicable"
    assert records["peak_vram_mb"]["reason"] == "cpu_profile"
    json.dumps(records, allow_nan=False)


def test_aggregate_statuses_keep_all_recorded_failures_complete_but_quality_unavailable():
    speech, control = _failed("speech"), _failed("control", sample_type="nonspeech")
    rows = [speech, control]
    timings = [_timing(speech, repetition=i, success=False) for i in range(1, 4)] + [
        _timing(control, success=False)
    ]
    metrics, _ = _aggregate(rows, timings)
    records = evaluate.metric_statuses(metrics, rows, timings, "cpu")
    for name in (
        "word_error_rate",
        "character_error_rate",
        "word_substitution_rate",
        "word_deletion_rate",
        "word_insertion_rate",
        "timestamp_boundary_mae_seconds",
        "timestamp_alignment_coverage",
        "unexpected_empty_transcript_rate",
        "nonspeech_false_transcription_rate",
    ):
        assert records[name]["value"] is None
        assert records[name]["status"] == "unavailable"
        assert records[name]["reason"] == "no_successful_first_outputs"
        assert records[name]["comparison_status"] == "complete"
    assert records["attempt_failure_rate"]["value"] == 1
    assert records["attempt_failure_rate"]["status"] == "scored"
    assert records["transcript_repeatability_success_rate"]["value"] == 0
    assert records["transcript_repeatability_success_rate"]["status"] == "scored"
    assert records["real_time_factor"]["status"] == "unavailable"
    assert records["real_time_factor"]["reason"] == "no_successful_timed_attempts"
    support = records["attempt_failure_rate"]["support"]
    assert support["comparison_scheduled_attempt_count"] == 4
    assert support["comparison_executed_attempt_count"] == 4
    assert support["comparison_failed_attempt_count"] == 4
    assert support["comparison_unexecuted_attempt_count"] == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_aggregate_statuses_unexecuted_requests_and_failed_load_remain_distinct(device):
    row = _failed("unexecuted", executed=False)
    timings = [_timing(row, repetition=i, executed=False) for i in range(1, 4)]
    metrics, _ = _aggregate([row], timings)
    metrics["peak_process_tree_ram_mb"] = 17.0
    error = {"phase": "load", "error": "private native error details"}
    records = evaluate.metric_statuses(metrics, [row], timings, device, error)
    for name in (
        "word_error_rate",
        "timestamp_alignment_coverage",
        "timestamp_boundary_mae_seconds",
        "attempt_failure_rate",
        "transcript_repeatability_success_rate",
        "real_time_factor",
    ):
        assert records[name]["status"] == "incomplete"
        assert records[name]["value"] is None
    assert records["word_error_rate"]["reason"] == "first_attempt_not_executed"
    assert records["attempt_failure_rate"]["reason"] == "measurement_attempts_missing"
    assert records["cold_model_load_seconds"]["status"] == "unavailable"
    assert records["cold_model_load_seconds"]["reason"] == "cold_load_failed"
    assert records["peak_vram_mb"]["status"] == (
        "inapplicable" if device == "cpu" else "unavailable"
    )
    assert records["peak_process_tree_ram_mb"]["status"] == "scored"
    assert all(
        record["comparison_status"] == "incomplete" for record in records.values()
    )
    assert all(record["execution_error_phase"] == "load" for record in records.values())
    assert (
        records["attempt_failure_rate"]["support"]["comparison_failed_attempt_count"]
        == 0
    )
    assert (
        records["attempt_failure_rate"]["support"][
            "comparison_unexecuted_attempt_count"
        ]
        == 3
    )
    assert error["error"] not in json.dumps(records)


@pytest.mark.parametrize("prediction", ["wrong", ""])
def test_aggregate_statuses_no_matches_do_not_invent_boundary_mae(prediction):
    row = _speech("unaligned", prediction=prediction)
    metrics, _ = _aggregate([row], [_timing(row)])
    records = evaluate.metric_statuses(metrics, [row], [_timing(row)], "cpu")
    assert records["timestamp_boundary_mae_seconds"]["value"] is None
    assert records["timestamp_boundary_mae_seconds"]["status"] == "unavailable"
    assert (
        records["timestamp_boundary_mae_seconds"]["reason"] == "no_aligned_boundaries"
    )
    assert records["timestamp_alignment_coverage"]["value"] == 0
    assert records["timestamp_alignment_coverage"]["status"] == "scored"


def test_aggregate_statuses_legitimately_empty_tasks_are_inapplicable_not_failed():
    row = _speech("empty-reference", "", "")
    metrics, _ = _aggregate([row], [_timing(row)])
    records = evaluate.metric_statuses(metrics, [row], [_timing(row)], "cpu")
    assert records["word_error_rate"]["status"] == "scored"
    assert records["word_error_rate"]["value"] == 0
    for name in (
        "timestamp_boundary_mae_seconds",
        "timestamp_alignment_coverage",
        "unexpected_empty_transcript_rate",
        "nonspeech_false_transcription_rate",
    ):
        assert records[name]["status"] == "inapplicable"
        assert records[name]["reason"] == "no_reference_task"


@pytest.mark.parametrize(
    "phase", ["close", "placement_after", "telemetry", "parameters"]
)
def test_aggregate_statuses_scored_values_survive_later_lifecycle_failure(phase):
    row = _speech("completed")
    metrics, _ = _aggregate([row], [_timing(row)])
    metrics.update(
        cold_model_load_seconds=1.0, peak_process_tree_ram_mb=100.0, peak_vram_mb=128.0
    )
    original = deepcopy(metrics)
    records = evaluate.metric_statuses(
        metrics,
        [row],
        [_timing(row)],
        "cuda",
        {"phase": phase, "error": "later failure"},
    )
    for name in (
        *evaluate.PRIMARY_METRICS,
        "cold_model_load_seconds",
        "peak_process_tree_ram_mb",
        "peak_vram_mb",
    ):
        assert records[name]["status"] == "scored"
        assert records[name]["reason"] is None
        assert records[name]["value"] == original[name]
        assert records[name]["comparison_status"] == "incomplete"
        assert records[name]["comparison_reason"] == "execution_error"
        assert records[name]["execution_error_phase"] == phase
    assert metrics == original


def test_aggregate_statuses_completed_first_survives_unexecuted_later_attempts():
    row = _speech("partial")
    row.update(
        scheduled_attempt_count=3,
        executed_attempt_count=1,
        unexecuted_attempt_count=2,
        transcript_repeatability_success_rate=None,
    )
    timings = [_timing(row)] + [
        _timing(row, repetition=i, executed=False) for i in (2, 3)
    ]
    metrics, _ = _aggregate([row], timings)
    records = evaluate.metric_statuses(metrics, [row], timings, "cpu")
    for name in evaluate.PRIMARY_METRICS:
        assert records[name]["status"] == "scored"
        assert records[name]["value"] == metrics[name]
        assert records[name]["comparison_status"] == "incomplete"
    for name in ("attempt_failure_rate", "transcript_repeatability_success_rate"):
        assert records[name]["status"] == "incomplete"
        assert records[name]["reason"] == "measurement_attempts_missing"


def test_aggregate_statuses_missing_control_does_not_make_failed_speech_quality_incomplete():
    speech = _failed("speech")
    control = _failed("control", sample_type="nonspeech", executed=False)
    timings = [_timing(speech, repetition=i, success=False) for i in range(1, 4)] + [
        _timing(control, executed=False)
    ]
    metrics, _ = _aggregate([speech, control], timings)
    records = evaluate.metric_statuses(metrics, [speech, control], timings, "cpu")
    assert records["word_error_rate"]["status"] == "unavailable"
    assert records["word_error_rate"]["comparison_status"] == "incomplete"
    assert records["real_time_factor"]["status"] == "unavailable"
    assert records["transcript_repeatability_success_rate"]["status"] == "scored"
    assert records["attempt_failure_rate"]["status"] == "incomplete"
    assert records["nonspeech_false_transcription_rate"]["status"] == "incomplete"
