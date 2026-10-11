"""ASR-specific scoring, pooled aggregation, and clip bootstrap intervals."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import cache

import numpy as np
from jiwer import process_characters, process_words

from experiments.benchmarks.common.metrics import normalize_prose, precision_recall_f1
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
) -> dict[str, object]:
    reference = normalize_transcript(str(item["reference"]))
    hypothesis = normalize_transcript(prediction)
    reference_words = reference.split()
    predicted_words = hypothesis.split()
    word_alignment = process_words(reference, hypothesis)
    character_alignment = process_characters(reference, hypothesis)
    references = _segments(
        item.get("reference_segments"), "reference", allow_empty=not reference_words
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
    predicted_segments = [
        segment
        for segment in predicted_segments
        if normalize_transcript(str(segment.get("text", "")))
    ]
    if hypothesis and not predicted_segments:
        raise ValueError("ASR prediction timestamp segments are missing")
    if not hypothesis and predicted_segments:
        raise ValueError(
            "ASR empty transcript has non-empty lexical timestamp segments"
        )
    predictions = _segments(
        predicted_segments, "prediction", allow_empty=not hypothesis
    )
    _validate_predicted_timeline(
        predictions,
        duration,
        tolerance_seconds=tolerance_seconds,
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
        "empty_transcript": None,
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
    rows = [
        {
            **row,
            "source_group_id": row.get("source_group_id", row["sample_id"]),
            "_timings": [
                timing
                for timing in timing_rows
                if timing["sample_id"] == row["sample_id"]
            ],
        }
        for row in sample_rows
    ]

    def statistic(draw):
        timings = []
        for index, row in enumerate(draw):
            timings.extend(
                {**timing, "sample_id": f"{index}:{row['sample_id']}"}
                for timing in row["_timings"]
            )
        return aggregate_rows(draw, timings)

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
        values, statuses = sample_metrics(row)
        row["metric_values"], row["metric_statuses"] = values, statuses
    for name in INTERVAL_METRICS:
        if name in {
            "real_time_factor",
            "p50_warm_clip_latency_seconds",
            "p95_warm_clip_latency_seconds",
        }:
            eligible = [row for row in sample_rows if row["sample_type"] == "speech"]
            contributing = [
                row
                for row in eligible
                if any(
                    timing.get("success", True)
                    for timing in timing_rows
                    if timing["sample_id"] == row["sample_id"]
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
    return metrics, intervals


def sample_metrics(row):
    """Per-request values and task statuses, independent of aggregate denominators."""
    success = row.get("first_attempt_success", True)
    speech = row["sample_type"] == "speech"
    words = (
        row.get("reference_word_count", row.get("planned_reference_word_count", 0)) or 0
    )
    timed = (
        row.get(
            "reference_timed_segment_count",
            row.get("planned_reference_timed_segment_count", 0),
        )
        or 0
    )
    values, statuses = {}, {}
    for name in INTERVAL_METRICS:
        eligible = speech
        value = None
        reason = "first_attempt_failed"
        if name in {
            "real_time_factor",
            "p50_warm_clip_latency_seconds",
            "p95_warm_clip_latency_seconds",
        }:
            continue
        if name == "attempt_failure_rate":
            eligible, value = True, row.get(name)
        elif name == "transcript_repeatability_success_rate":
            value = row.get(name) if speech else None
            eligible = speech and value is not None
            reason = "repeatability_not_measured"
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
            char_count = row["reference_character_count"]
            edits = {
                "word_substitution_rate": row["word_substitutions"],
                "word_deletion_rate": row["word_deletions"],
                "word_insertion_rate": row["word_insertions"],
                "word_error_rate": sum(
                    row[key]
                    for key in (
                        "word_substitutions",
                        "word_deletions",
                        "word_insertions",
                    )
                ),
                "character_error_rate": sum(
                    row[key]
                    for key in (
                        "character_substitutions",
                        "character_deletions",
                        "character_insertions",
                    )
                ),
            }
            denominator = char_count if name == "character_error_rate" else words
            value = edits[name] / denominator if denominator else float(edits[name])
        values[name] = value if eligible else None
        statuses[name] = {
            "status": "inapplicable"
            if not eligible
            else "scored"
            if value is not None
            else "unavailable",
            "reason": "no_reference_task"
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
        if row["sample_type"] == "speech" and row.get("first_attempt_success", True)
    ]
    nonspeech = [
        row
        for row in sample_rows
        if row["sample_type"] == "nonspeech" and row.get("first_attempt_success", True)
    ]
    word_count = _sum(speech, "reference_word_count")
    char_count = _sum(speech, "reference_character_count")
    result = {}
    for metric, count in (
        ("word_substitution_rate", "word_substitutions"),
        ("word_deletion_rate", "word_deletions"),
        ("word_insertion_rate", "word_insertions"),
    ):
        total = _sum(speech, count)
        result[metric] = total / word_count if word_count else total if speech else None
    word_errors = sum(
        _sum(speech, name)
        for name in ("word_substitutions", "word_deletions", "word_insertions")
    )
    char_errors = sum(
        _sum(speech, name)
        for name in (
            "character_substitutions",
            "character_deletions",
            "character_insertions",
        )
    )
    result["word_error_rate"] = (
        word_errors / word_count if word_count else word_errors if speech else None
    )
    result["character_error_rate"] = (
        char_errors / char_count if char_count else char_errors if speech else None
    )
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
    result["transcript_repeatability_success_rate"] = (
        float(np.mean(repeats)) if repeats else None
    )
    result["attempt_failure_rate"] = (
        sum(not row.get("success", True) for row in timing_rows) / len(timing_rows)
        if timing_rows
        else None
    )
    speech_ids = {
        str(row["sample_id"]) for row in sample_rows if row["sample_type"] == "speech"
    }
    completed = [
        row
        for row in timing_rows
        if row.get("sample_type", "speech") == "speech"
        and row.get("success", True)
        and (row.get("sample_type") == "speech" or str(row["sample_id"]) in speech_ids)
    ]
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
        text = str(raw.get("text", "")).strip()
        start, end = float(raw.get("start", -1)), float(raw.get("end", -1))
        if (
            not text
            or not np.isfinite(start)
            or not np.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise ValueError(f"ASR {label} timestamp segment is invalid")
        segments.append({"text": text, "start": start, "end": end})
    return segments


def _ordered_span_matches(reference_segments, predicted_segments, *, threshold: float):
    """Choose one-to-one ordered spans with deterministic documented tie-breaks."""

    if not reference_segments or not predicted_segments:
        return ()
    eligible: dict[int, list[tuple[int, int, float]]] = {}
    for reference_index, reference in enumerate(reference_segments):
        expected = normalize_transcript(str(reference["text"])).split()
        spans: list[tuple[int, int, float]] = []
        for start in range(len(predicted_segments)):
            observed: list[str] = []
            for end in range(start, len(predicted_segments)):
                observed.extend(
                    normalize_transcript(str(predicted_segments[end]["text"])).split()
                )
                similarity = _token_content_f1(expected, observed)
                if similarity >= threshold:
                    spans.append((start, end, similarity))
        eligible[reference_index] = spans

    @cache
    def solve(reference_index: int, minimum_prediction: int):
        if reference_index >= len(reference_segments):
            return (0.0, 0, ())
        best = solve(reference_index + 1, minimum_prediction)
        for start, end, similarity in eligible[reference_index]:
            if start < minimum_prediction:
                continue
            remaining = solve(reference_index + 1, end + 1)
            candidate = (
                similarity + remaining[0],
                1 + remaining[1],
                ((reference_index, start, end, similarity), *remaining[2]),
            )
            if _better_span_solution(candidate, best):
                best = candidate
        return best

    return solve(0, 0)[2]


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
