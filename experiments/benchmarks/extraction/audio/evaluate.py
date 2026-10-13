"""ASR-specific scoring, pooled aggregation, and clip bootstrap intervals."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
from jiwer import process_characters, process_words

from experiments.benchmarks.common.metrics import normalize_prose, precision_recall_f1
from experiments.benchmarks.common.protocol import number
from experiments.benchmarks.extraction.scoring import bootstrap_sources

METRIC_DIRECTIONS = {
    "word_error_rate": "min",
    "character_error_rate": "min",
    "word_substitution_rate": "min",
    "word_deletion_rate": "min",
    "word_insertion_rate": "min",
    "timestamp_boundary_mae_seconds": "min",
    "timestamp_alignment_coverage": "max",
    "unexpected_empty_transcript_rate": "min",
    "nonspeech_false_transcription_rate": "min",
    "transcript_repeatability_success_rate": "max",
    "attempt_failure_rate": "min",
    "real_time_factor": "min",
    "p50_warm_clip_latency_seconds": "min",
    "p95_warm_clip_latency_seconds": "min",
    "cold_model_load_seconds": "min",
    "peak_process_tree_ram_mb": "min",
    "peak_vram_mb": "min",
}
PRIMARY_METRICS = (
    "word_error_rate",
    "timestamp_boundary_mae_seconds",
    "timestamp_alignment_coverage",
    "real_time_factor",
)
INTERVAL_METRICS = tuple(
    name
    for name in METRIC_DIRECTIONS
    if name
    not in {
        "cold_model_load_seconds",
        "peak_process_tree_ram_mb",
        "peak_vram_mb",
    }
)
TIMING_METRICS = {
    "real_time_factor",
    "p50_warm_clip_latency_seconds",
    "p95_warm_clip_latency_seconds",
}


def normalize_transcript(text: str) -> str:
    """Apply only the frozen evaluator normalization."""
    return normalize_prose(text)


def score_speech(
    item: Mapping[str, object],
    prediction: str,
    predicted_segments: Sequence[Mapping[str, object]],
    *,
    quality_latency_seconds: float,
    warnings: Sequence[str] = (),
    alignment_threshold: float,
    timestamp_tolerance_seconds: float,
    allow_untimed_reference: bool = False,
) -> dict[str, object]:
    reference = normalize_transcript(str(item["reference"]))
    hypothesis = normalize_transcript(prediction)
    reference_words = reference.split()
    predicted_words = hypothesis.split()
    word_alignment = process_words(reference, hypothesis)
    character_alignment = process_characters(reference, hypothesis)
    references = _segments(
        item.get("reference_segments"),
        "reference",
        allow_empty=not reference_words or allow_untimed_reference,
    )
    references = [
        segment for segment in references if normalize_transcript(segment["text"])
    ]
    predictions = validate_prediction(
        prediction,
        predicted_segments,
        float(item["duration_seconds"]),
        timestamp_tolerance_seconds,
    )
    timestamp = _timestamp_totals(
        references, predictions, alignment_threshold=alignment_threshold
    )
    return {
        "sample_id": str(item["id"]),
        "sample_type": "speech",
        "conditions": list(item.get("conditions", [])),
        "duration_seconds": float(item["duration_seconds"]),
        "reference_word_count": len(reference_words),
        "word_substitutions": word_alignment.substitutions,
        "word_deletions": word_alignment.deletions,
        "word_insertions": word_alignment.insertions,
        "reference_character_count": len(reference),
        "character_substitutions": character_alignment.substitutions,
        "character_deletions": character_alignment.deletions,
        "character_insertions": character_alignment.insertions,
        **timestamp,
        "unexpected_empty_transcript": int(
            bool(reference_words) and not predicted_words
        ),
        "empty_transcript_eligible": bool(reference_words),
        "first_attempt_success": True,
        "nonspeech_false_transcription": None,
        "quality_latency_seconds": quality_latency_seconds,
        "warnings": list(warnings),
    }


def validate_prediction(prediction, predicted_segments, duration, tolerance_seconds):
    if not isinstance(prediction, str):
        raise ValueError("ASR prediction transcript must be a string")
    if not isinstance(predicted_segments, (list, tuple)) or any(
        not isinstance(segment, Mapping) for segment in predicted_segments
    ):
        raise ValueError(
            "ASR prediction timestamp segments must be a sequence of objects"
        )
    hypothesis = normalize_transcript(prediction)
    # Validate every supplied unit before ignoring nonlexical content. Otherwise
    # malformed units can disappear and turn an invalid request into a success.
    predictions = _segments(predicted_segments, "prediction", allow_empty=True)
    _validate_predicted_timeline(
        predictions, duration, tolerance_seconds=tolerance_seconds
    )
    predictions = [
        segment for segment in predictions if normalize_transcript(segment["text"])
    ]
    if hypothesis and not predictions:
        raise ValueError("ASR prediction timestamp segments are missing")
    if not hypothesis and predictions:
        raise ValueError(
            "ASR empty transcript has non-empty lexical timestamp segments"
        )
    if (
        normalize_transcript(" ".join(str(segment["text"]) for segment in predictions))
        != hypothesis
    ):
        raise ValueError("ASR transcript contradicts its lexical timestamp segments")
    return predictions


def score_nonspeech(
    item: Mapping[str, object],
    prediction: str,
    *,
    latency_seconds: float,
    warnings: Sequence[str] = (),
) -> dict[str, object]:
    return {
        "sample_id": str(item["id"]),
        "sample_type": "nonspeech",
        "conditions": [str(item.get("nonspeech_kind", "unspecified"))],
        "duration_seconds": float(item["duration_seconds"]),
        "reference_word_count": None,
        "word_substitutions": None,
        "word_deletions": None,
        "word_insertions": None,
        "reference_character_count": None,
        "character_substitutions": None,
        "character_deletions": None,
        "character_insertions": None,
        "reference_timed_segment_count": None,
        "aligned_timed_segment_count": None,
        "timestamp_boundary_error_seconds": None,
        "timestamp_boundary_count": None,
        "unexpected_empty_transcript": None,
        "empty_transcript_eligible": False,
        "nonspeech_false_transcription": int(bool(normalize_transcript(prediction))),
        "transcript_repeatability_success_rate": None,
        "first_attempt_success": True,
        "quality_latency_seconds": latency_seconds,
        "warnings": list(warnings),
    }


def aggregate(
    sample_rows,
    timing_rows,
    *,
    cold_model_load_seconds,
    peak_process_tree_ram_mb,
    peak_vram_mb,
    resamples,
    seed,
    confidence,
    minimum_sources=None,
    minimum_latency_sources=None,
):
    metrics = aggregate_rows(sample_rows, timing_rows)
    metrics.update(
        {
            "cold_model_load_seconds": cold_model_load_seconds,
            "peak_process_tree_ram_mb": peak_process_tree_ram_mb,
            "peak_vram_mb": peak_vram_mb,
        }
    )
    timings_by_sample = _timings_by_sample(timing_rows)
    rows = [
        {
            **row,
            "source_group_id": _source_group_id(row),
            "_timings": timings_by_sample.get(str(row["sample_id"]), []),
        }
        for row in sample_rows
    ]

    def statistic(draw):
        samples, timings = [], []
        for index, row in enumerate(draw):
            sample_id = f"{index}:{row['sample_id']}"
            samples.append({**row, "sample_id": sample_id})
            timings.extend(
                {**timing, "sample_id": sample_id, "sample_type": row["sample_type"]}
                for timing in row["_timings"]
            )
        return aggregate_rows(samples, timings)

    intervals = bootstrap_sources(
        rows,
        statistic,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        minimum_sources=minimum_sources,
        stratum_field="sample_type",
    )
    if minimum_latency_sources != minimum_sources:
        latency_bounds = bootstrap_sources(
            rows,
            statistic,
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            minimum_sources=minimum_latency_sources,
            stratum_field="sample_type",
        )
        for name in ("p50_warm_clip_latency_seconds", "p95_warm_clip_latency_seconds"):
            if name in latency_bounds:
                intervals[name] = latency_bounds[name]
    for row in sample_rows:
        values, statuses = sample_metrics(
            row, timing_rows=timings_by_sample.get(str(row["sample_id"]), [])
        )
        row["metric_values"], row["metric_statuses"] = values, statuses
    for name in INTERVAL_METRICS:
        if name in TIMING_METRICS:
            eligible = [row for row in sample_rows if row["sample_type"] == "speech"]
            contributing = [
                row
                for row in eligible
                if any(
                    _valid_completed_timing(timing)
                    for timing in timings_by_sample.get(str(row["sample_id"]), [])
                )
            ]
        else:
            eligible = [
                row
                for row in sample_rows
                if row["metric_statuses"][name]["status"] != "inapplicable"
            ]
            contributing = [
                row for row in eligible if row["metric_values"][name] is not None
            ]
        metrics[name + ".scheduled_count"] = float(len(sample_rows))
        metrics[name + ".eligible_count"] = float(len(eligible))
        metrics[name + ".contributing_count"] = float(len(contributing))
        metrics[name + ".scheduled_source_count"] = float(
            len({_source_group_id(row) for row in sample_rows})
        )
        metrics[name + ".contributing_source_count"] = float(
            len({_source_group_id(row) for row in contributing})
        )
        metrics[name + ".eligible_source_count"] = float(
            len({_source_group_id(row) for row in eligible})
        )
        metrics[name + ".failed_first_attempt_count"] = float(
            sum(
                not _first_completed(row) and not _first_incomplete(row)
                for row in eligible
            )
        )
        metrics[name + ".unavailable_count"] = float(len(eligible) - len(contributing))
        if name not in TIMING_METRICS:
            metrics[name + ".incomplete_count"] = float(
                sum(
                    row["metric_statuses"][name]["status"] == "incomplete"
                    for row in eligible
                )
            )
        if name in {
            "word_error_rate",
            "character_error_rate",
            "word_substitution_rate",
            "word_deletion_rate",
            "word_insertion_rate",
            "timestamp_boundary_mae_seconds",
            "timestamp_alignment_coverage",
        }:
            for count in ("word", "character", "timed_segment"):
                field = f"reference_{count}_count"
                metrics[name + f".planned_{field}"] = float(
                    sum(_reference_count(row, field) for row in eligible)
                )
                metrics[name + f".contributing_{field}"] = _sum(contributing, field)
    return metrics, intervals


def metric_statuses(metrics, sample_rows, timing_rows, device, execution_error=None):
    """Describe recorded aggregates without recomputing any scored value.

    Metric availability and comparison completeness are separate: a conditional
    estimate can be scored even when another request or lifecycle phase is missing.
    Only the error phase is copied here; detailed errors belong in execution evidence.
    """
    states = {name: set() for name in METRIC_DIRECTIONS}
    supports = {name: {} for name in METRIC_DIRECTIONS}
    for key, value in metrics.items():
        name, separator, field = key.partition(".")
        if separator and name in supports:
            supports[name][field] = value
    for row in sample_rows:
        for name, recorded in row["metric_statuses"].items():
            if name in states:
                states[name].add((recorded["status"], recorded["reason"]))
    attempt_counts = {
        "comparison_scheduled_attempt_count": len(timing_rows),
        "comparison_executed_attempt_count": 0,
        "comparison_failed_attempt_count": 0,
        "comparison_unexecuted_attempt_count": 0,
    }
    speech_unexecuted = False
    for row in timing_rows:
        executed = _attempt_executed(row)
        attempt_counts["comparison_executed_attempt_count"] += bool(executed)
        attempt_counts["comparison_failed_attempt_count"] += bool(
            executed and row.get("success") is False
        )
        attempt_counts["comparison_unexecuted_attempt_count"] += not executed
        speech_unexecuted |= (
            not executed and row.get("sample_type", "speech") == "speech"
        )
    comparison_reason = (
        "execution_error"
        if execution_error is not None
        else "measurement_attempts_missing"
        if attempt_counts["comparison_unexecuted_attempt_count"]
        or any(
            status == "incomplete"
            for recorded in states.values()
            for status, _ in recorded
        )
        else None
    )
    phase = execution_error.get("phase") if execution_error else None
    records = {}
    for name in METRIC_DIRECTIONS:
        support = supports[name]
        support.update(attempt_counts)
        value = metrics[name]
        status, reason = "scored", None
        if name == "peak_vram_mb" and device == "cpu":
            status, reason = "inapplicable", "cpu_profile"
        elif value is None:
            if name == "cold_model_load_seconds":
                status, reason = (
                    "unavailable",
                    "cold_load_failed" if phase == "load" else "cold_load_not_measured",
                )
            elif name in {"peak_process_tree_ram_mb", "peak_vram_mb"}:
                status, reason = "unavailable", "resource_measurement_unavailable"
            elif support.get("incomplete_count", 0):
                status, reason = (
                    "incomplete",
                    "measurement_attempts_missing"
                    if name
                    in {"attempt_failure_rate", "transcript_repeatability_success_rate"}
                    else "first_attempt_not_executed",
                )
            elif not support["eligible_count"]:
                status, reason = (
                    "inapplicable",
                    "repeatability_not_measured"
                    if name == "transcript_repeatability_success_rate"
                    else "no_reference_task",
                )
            elif name in TIMING_METRICS:
                status, reason = (
                    ("incomplete", "measurement_attempts_missing")
                    if speech_unexecuted
                    else ("unavailable", "no_successful_timed_attempts")
                )
            elif name in {
                "attempt_failure_rate",
                "transcript_repeatability_success_rate",
            }:
                status, reason = "incomplete", "measurement_attempts_missing"
            elif ("unavailable", "no_aligned_boundaries") in states[name]:
                status, reason = "unavailable", "no_aligned_boundaries"
            else:
                status, reason = "unavailable", "no_successful_first_outputs"
        records[name] = {
            "value": value,
            "status": status,
            "reason": reason,
            "support": support,
            "comparison_status": "incomplete" if comparison_reason else "complete",
            "comparison_reason": comparison_reason,
            "execution_error_phase": phase,
        }
    return records


def sample_metrics(row, *, timing_rows=None):
    """Per-request values and task statuses, independent of aggregate denominators."""
    success = _first_completed(row)
    speech = row["sample_type"] == "speech"
    words = _reference_count(row, "reference_word_count")
    timed = _reference_count(row, "reference_timed_segment_count")
    recognition = _recognition_metrics([row]) if success and speech else {}
    values, statuses = {}, {}
    for name in INTERVAL_METRICS:
        eligible = speech
        value = None
        incomplete = not success and _first_incomplete(row)
        reason = "first_attempt_not_executed" if incomplete else "first_attempt_failed"
        if name in TIMING_METRICS:
            continue
        if name == "attempt_failure_rate":
            eligible, value = True, row.get(name)
            incomplete = _attempts_incomplete(row, timing_rows)
            reason = "measurement_attempts_missing"
            if incomplete:
                value = None
            elif timing_rows is not None:
                value = sum(
                    not timing.get("success", True) for timing in timing_rows
                ) / len(timing_rows)
        elif name == "transcript_repeatability_success_rate":
            value = row.get(name) if speech else None
            incomplete = speech and _attempts_incomplete(row, timing_rows)
            scheduled = row.get(
                "scheduled_attempt_count",
                len(timing_rows) if timing_rows is not None else 1,
            )
            eligible = speech and (scheduled > 1 or value is not None)
            reason = "repeatability_not_measured"
            if eligible and (incomplete or value is None):
                incomplete, value, reason = True, None, "measurement_attempts_missing"
        elif name == "nonspeech_false_transcription_rate":
            eligible = not speech
            value = row.get("nonspeech_false_transcription") if success else None
        elif name == "unexpected_empty_transcript_rate":
            eligible = speech and words > 0
            value = row.get("unexpected_empty_transcript") if success else None
        elif name.startswith("timestamp_"):
            eligible = speech and timed > 0
            if success and eligible:
                boundaries = row["timestamp_boundary_count"]
                value = (
                    row["timestamp_boundary_error_seconds"] / boundaries
                    if boundaries
                    else None
                )
                if name == "timestamp_alignment_coverage":
                    value = row["aligned_timed_segment_count"] / timed
                reason = "no_aligned_boundaries"
        elif success and speech:
            value = recognition[name]
        values[name] = value if eligible else None
        statuses[name] = {
            "status": "inapplicable"
            if not eligible
            else "incomplete"
            if incomplete
            else "scored"
            if value is not None
            else "unavailable",
            "reason": reason
            if name == "transcript_repeatability_success_rate"
            and speech
            and not eligible
            else "no_reference_task"
            if not eligible
            else None
            if value is not None
            else reason,
        }
    return values, statuses


def aggregate_rows(sample_rows, timing_rows):
    speech = [
        row
        for row in sample_rows
        if row["sample_type"] == "speech" and _first_completed(row)
    ]
    nonspeech = [
        row
        for row in sample_rows
        if row["sample_type"] == "nonspeech" and _first_completed(row)
    ]
    result = _recognition_metrics(speech)
    boundaries = _sum(speech, "timestamp_boundary_count")
    references = _sum(speech, "reference_timed_segment_count")
    result["timestamp_boundary_mae_seconds"] = (
        _sum(speech, "timestamp_boundary_error_seconds") / boundaries
        if boundaries
        else None
    )
    result["timestamp_alignment_coverage"] = (
        _sum(speech, "aligned_timed_segment_count") / references if references else None
    )
    empty_eligible = [
        row
        for row in speech
        if row.get("empty_transcript_eligible", row.get("reference_word_count", 0) > 0)
    ]
    result["unexpected_empty_transcript_rate"] = (
        _sum(empty_eligible, "unexpected_empty_transcript") / len(empty_eligible)
        if empty_eligible
        else None
    )
    result["nonspeech_false_transcription_rate"] = (
        _sum(nonspeech, "nonspeech_false_transcription") / len(nonspeech)
        if nonspeech
        else None
    )
    repeats = [
        row["transcript_repeatability_success_rate"]
        for row in sample_rows
        if row["sample_type"] == "speech"
        and row.get("transcript_repeatability_success_rate") is not None
    ]
    timings_by_sample = _timings_by_sample(timing_rows)
    speech_rows = [row for row in sample_rows if row["sample_type"] == "speech"]
    repeat_incomplete = any(
        _attempts_incomplete(row, timings_by_sample.get(str(row["sample_id"]), []))
        or (
            row.get(
                "scheduled_attempt_count",
                len(timings_by_sample.get(str(row["sample_id"]), [])),
            )
            > 1
            and row.get("transcript_repeatability_success_rate") is None
        )
        for row in speech_rows
    )
    result["transcript_repeatability_success_rate"] = (
        float(np.mean(repeats)) if repeats and not repeat_incomplete else None
    )
    attempts_incomplete = any(
        _attempts_incomplete(row, timings_by_sample.get(str(row["sample_id"]), []))
        for row in sample_rows
    ) or any(not _attempt_executed(row) for row in timing_rows)
    result["attempt_failure_rate"] = (
        sum(not row.get("success", True) for row in timing_rows) / len(timing_rows)
        if timing_rows and not attempts_incomplete
        else None
    )
    speech_ids = {
        str(row["sample_id"]) for row in sample_rows if row["sample_type"] == "speech"
    }
    completed = [
        row
        for row in timing_rows
        if row.get("sample_type", "speech") == "speech"
        and _attempt_executed(row)
        and row.get("success", True)
        and (row.get("sample_type") == "speech" or str(row["sample_id"]) in speech_ids)
    ]
    if any(not _valid_completed_timing(row) for row in completed):
        result.update(dict.fromkeys(TIMING_METRICS))
        return result
    seconds = sum(float(row["duration_seconds"]) for row in completed)
    result["real_time_factor"] = (
        sum(float(row["latency_seconds"]) for row in completed) / seconds
        if seconds
        else None
    )
    per_clip = {}
    for row in completed:
        per_clip.setdefault(str(row["sample_id"]), []).append(
            float(row["latency_seconds"])
        )
    medians = [float(np.median(values)) for values in per_clip.values()]
    for percentile in (50, 95):
        result[f"p{percentile}_warm_clip_latency_seconds"] = (
            float(np.quantile(medians, percentile / 100)) if medians else None
        )
    return result


def _recognition_metrics(rows):
    """Use the same pooled edit arithmetic for individual clips and corpus draws."""
    result = {}
    for prefix in ("word", "character"):
        denominator = _sum(rows, f"reference_{prefix}_count")
        counts = [
            _sum(rows, f"{prefix}_{edit}")
            for edit in ("substitutions", "deletions", "insertions")
        ]
        edits = {f"{prefix}_error_rate": sum(counts)}
        if prefix == "word":
            edits.update(
                zip(
                    (
                        "word_substitution_rate",
                        "word_deletion_rate",
                        "word_insertion_rate",
                    ),
                    counts,
                )
            )
        for name, count in edits.items():
            result[name] = (
                count / denominator if denominator else count if rows else None
            )
    return result


def _source_group_id(row):
    return str(row.get("source_group_id") or row["sample_id"])


def _reference_count(row, field):
    planned = row.get("planned_" + field)
    return planned if planned is not None else row.get(field) or 0


def _first_completed(row):
    return row.get("first_attempt_executed", True) and row.get(
        "first_attempt_success", True
    )


def _first_incomplete(row):
    return (
        row.get("first_attempt_executed") is False
        or row.get("metric_status") == "incomplete"
    )


def _attempt_executed(row):
    return row.get("executed", True) and row.get("success", True) is not None


def _attempts_incomplete(row, timings=None):
    scheduled = row.get("scheduled_attempt_count")
    executed = row.get("executed_attempt_count")
    return (
        row.get("first_attempt_executed") is False
        or bool(row.get("unexecuted_attempt_count", 0))
        or (scheduled is not None and executed is not None and scheduled != executed)
        or (
            timings is not None
            and (
                not timings
                or any(not _attempt_executed(timing) for timing in timings)
                or (scheduled is not None and len(timings) != scheduled)
                or (
                    scheduled is not None
                    and any("repetition" in timing for timing in timings)
                    and {timing.get("repetition") for timing in timings}
                    != set(range(1, scheduled + 1))
                )
            )
        )
    )


def _timings_by_sample(timings):
    result = {}
    for timing in timings:
        result.setdefault(str(timing["sample_id"]), []).append(timing)
    return result


def _valid_completed_timing(row):
    if not _attempt_executed(row) or not row.get("success", True):
        return False
    try:
        latency, duration = (
            float(row["latency_seconds"]),
            float(row["duration_seconds"]),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return (
        np.isfinite(latency) and latency >= 0 and np.isfinite(duration) and duration > 0
    )


def _timestamp_totals(
    reference_segments, predicted_segments, *, alignment_threshold: float
) -> dict[str, int | float]:
    matches = _ordered_span_matches(
        reference_segments,
        predicted_segments,
        threshold=alignment_threshold,
    )
    error = 0.0
    for reference_index, start_index, end_index, _ in matches:
        reference = reference_segments[reference_index]
        predicted = predicted_segments[start_index : end_index + 1]
        predicted_start = min(float(segment["start"]) for segment in predicted)
        predicted_end = max(float(segment["end"]) for segment in predicted)
        error += abs(float(reference["start"]) - predicted_start)
        error += abs(float(reference["end"]) - predicted_end)
    return {
        "reference_timed_segment_count": len(reference_segments),
        "aligned_timed_segment_count": len(matches),
        "timestamp_boundary_error_seconds": error,
        "timestamp_boundary_count": len(matches) * 2,
    }


def _segments(
    value: object, label: str, *, allow_empty: bool = False
) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"ASR {label} timestamp segments are missing")
    if not value:
        if allow_empty:
            return []
        raise ValueError(f"ASR {label} timestamp segments are missing")
    segments = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError(f"ASR {label} timestamp segment is malformed")
        text = raw.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"ASR {label} timestamp segment is invalid")
        text = text.strip()
        try:
            start = number(raw.get("start"), f"ASR {label} timestamp start", minimum=0)
            end = number(
                raw.get("end"),
                f"ASR {label} timestamp end",
                minimum=start,
                minimum_exclusive=True,
            )
        except ValueError as exc:
            raise ValueError(f"ASR {label} timestamp segment is invalid") from exc
        segments.append({"text": text, "start": start, "end": end})
    return segments


def _ordered_span_matches(reference_segments, predicted_segments, *, threshold: float):
    """Choose one-to-one ordered spans with deterministic documented tie-breaks."""

    if not reference_segments or not predicted_segments:
        return ()
    predicted = [
        normalize_transcript(str(segment["text"])).split()
        for segment in predicted_segments
    ]
    following = [(0.0, 0, ())] * (len(predicted) + 1)
    # Bottom-up DP avoids a recursion limit on long video transcripts. Each
    # eligible span is considered once, rather than once per preceding unit.
    for reference_index in range(len(reference_segments) - 1, -1, -1):
        expected = normalize_transcript(
            str(reference_segments[reference_index]["text"])
        ).split()
        current = list(following)
        for start in range(len(predicted) - 1, -1, -1):
            best = following[start]  # Omit this reference segment.
            if _better_span_solution(current[start + 1], best):
                best = current[start + 1]  # Skip this predicted unit.
            observed: list[str] = []
            for end in range(start, len(predicted)):
                observed.extend(predicted[end])
                # Even perfect overlap cannot rescue a span this much longer
                # than the reference; extending it only lowers this upper bound.
                if (
                    observed
                    and threshold > 0
                    and 2 * len(expected) / (len(expected) + len(observed)) < threshold
                ):
                    break
                similarity = _token_content_f1(expected, observed)
                if similarity < threshold:
                    continue
                remaining = following[end + 1]
                candidate = (
                    similarity + remaining[0],
                    1 + remaining[1],
                    ((reference_index, start, end, similarity), *remaining[2]),
                )
                if _better_span_solution(candidate, best):
                    best = candidate
            current[start] = best
        following = current
    return following[0][2]


def _better_span_solution(candidate, incumbent) -> bool:
    if abs(candidate[0] - incumbent[0]) > 1e-12:
        return candidate[0] > incumbent[0]
    if candidate[1] != incumbent[1]:
        return candidate[1] > incumbent[1]
    candidate_positions = tuple((value[1], value[2]) for value in candidate[2])
    incumbent_positions = tuple((value[1], value[2]) for value in incumbent[2])
    return candidate_positions < incumbent_positions


def _token_content_f1(reference: Sequence[str], prediction: Sequence[str]) -> float:
    expected = Counter(reference)
    observed = Counter(prediction)
    overlap = sum((expected & observed).values())
    return precision_recall_f1(
        overlap,
        sum(observed.values()) - overlap,
        sum(expected.values()) - overlap,
    )[2]


def _validate_predicted_timeline(
    segments, duration: float, *, tolerance_seconds: float
) -> None:
    previous_start = -1.0
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        if start < previous_start or end > duration + tolerance_seconds:
            raise ValueError("ASR prediction contains invalid or unordered timestamps")
        previous_start = start


def _sum(rows, name: str) -> float:
    return sum(float(row[name]) for row in rows if row.get(name) is not None)
