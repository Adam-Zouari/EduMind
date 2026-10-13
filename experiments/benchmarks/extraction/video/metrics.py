"""Frozen visible-text metrics for the dedicated video benchmark."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import fsum

import numpy as np

from experiments.benchmarks.common.metrics import (
    normalize_prose,
    normalized_tokens,
    precision_recall_f1,
)
from experiments.benchmarks.extraction.scoring import bootstrap_sources

QUALITY_DIRECTIONS = {
    "visual_content_precision": "max",
    "visual_content_recall": "max",
    "visual_content_f1": "max",
    "mean_visual_first_detection_delay_seconds": "min",
    "timed_visual_occurrence_coverage": "max",
    "duplicate_visual_text_rate": "min",
    "repeatability_success_rate": "max",
    "attempt_failure_rate": "min",
}
OPERATIONAL_DIRECTIONS = {
    "visual_real_time_factor": "min",
    "p50_warm_visual_latency_seconds": "min",
    "p95_warm_visual_latency_seconds": "min",
    "cold_visual_pipeline_load_seconds": "min",
    "peak_visual_process_tree_ram_mb": "min",
    "peak_visual_vram_mb": "min",
    "mean_selected_frames_per_video": "descriptive",
}
METRIC_DIRECTIONS = {**QUALITY_DIRECTIONS, **OPERATIONAL_DIRECTIONS}


def score_video(
    item: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]] | None,
    occurrence_matching: Mapping[str, object],
) -> dict[str, object]:
    """Score only the designated first output; None is failure, not empty text."""
    duration = float(item["duration_seconds"])
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Video duration must be positive and finite")
    reference_units = _reference_units(item.get("reference_visual_text"))
    expected = set(reference_units)
    occurrences = _occurrences(item.get("visual_occurrences"), duration=duration)
    row = {
        "sample_id": str(item["id"]),
        "duration_seconds": duration,
        "first_attempt_executed": True,
        "first_attempt_success": predictions is not None,
        "planned_reference_visual_unit_count": len(expected),
        "planned_reference_occurrence_count": len(occurrences),
        "reference_visual_unit_count": len(expected),
        "reference_occurrence_count": len(occurrences),
    }
    if predictions is None:
        return {
            **row,
            **{name: None for name in QUALITY_DIRECTIONS},
            **{
                name: None
                for name in (
                    "predicted_visual_unit_count",
                    "matched_visual_unit_count",
                    "predicted_visual_occurrence_count",
                    "duplicate_visual_unit_count",
                    "assigned_visual_detection_count",
                    "unassigned_visual_detection_count",
                    "matched_occurrence_count",
                    "first_detection_delay_total_seconds",
                )
            },
        }
    # Keep each normalized line detection, including repeated copies. Only the
    # content comparison below collapses them into distinct sets.
    detections = []
    for index, prediction in enumerate(predictions):
        timestamp = float(prediction["timestamp"])
        if not np.isfinite(timestamp) or not 0 <= timestamp <= duration:
            raise ValueError("Video detection timestamp is outside the video")
        detections.extend(
            {"text": unit, "timestamp": timestamp, "index": index}
            for unit in _normalized_lines(str(prediction.get("text", "")))
        )
    detections.sort(key=lambda detection: (detection["timestamp"], detection["index"]))
    observed = {str(detection["text"]) for detection in detections}
    overlap = len(expected & observed)
    if not expected and not observed:
        precision = recall = f1 = 1.0
    else:
        precision, recall, f1 = precision_recall_f1(
            overlap, len(observed) - overlap, len(expected) - overlap
        )
    matches = _occurrence_matches(occurrences, detections, occurrence_matching)
    duplicate_count, assigned_count = _duplicates_within_occurrences(
        occurrences, detections, occurrence_matching
    )
    duplicate = duplicate_count / assigned_count if assigned_count else None
    delays = [
        float(detections[right]["timestamp"]) - float(occurrences[left]["start"])
        for left, right in matches
    ]
    return {
        **row,
        "predicted_visual_unit_count": len(observed),
        "matched_visual_unit_count": overlap,
        "predicted_visual_occurrence_count": len(detections),
        "duplicate_visual_unit_count": duplicate_count,
        "assigned_visual_detection_count": assigned_count,
        "unassigned_visual_detection_count": len(detections) - assigned_count,
        "matched_occurrence_count": len(matches),
        "first_detection_delay_total_seconds": fsum(delays),
        "visual_content_precision": precision,
        "visual_content_recall": recall,
        "visual_content_f1": f1,
        "mean_visual_first_detection_delay_seconds": (
            float(np.mean(delays)) if delays else None
        ),
        "timed_visual_occurrence_coverage": (
            len(matches) / len(occurrences) if occurrences else None
        ),
        "duplicate_visual_text_rate": duplicate,
    }


def aggregate_quality(rows):
    if not rows:
        raise ValueError("Video quality aggregation requires scheduled samples")
    result = {}
    for name in QUALITY_DIRECTIONS:
        observations = [_quality_result(row, name) for row in rows]
        values = [float(value) for value, _, _ in observations if value is not None]
        incomplete = name in {
            "repeatability_success_rate",
            "attempt_failure_rate",
        } and any(status == "incomplete" for _, status, _ in observations)
        result[name] = float(np.mean(values)) if values and not incomplete else None
    return result


def quality_statuses(row, *, has_occurrences=None):
    return {
        name: {"status": status, "reason": reason}
        for name in QUALITY_DIRECTIONS
        for _, status, reason in [_quality_result(row, name, has_occurrences)]
    }


def _quality_result(row, name, has_occurrences=None):
    """One eligibility/status rule for points, support counts, and CI draws."""
    if has_occurrences is None:
        count = row.get(
            "planned_reference_occurrence_count", row.get("reference_occurrence_count")
        )
        has_occurrences = count != 0  # Unknown annotation is not a text-free task.
    if (
        name
        in {
            "timed_visual_occurrence_coverage",
            "mean_visual_first_detection_delay_seconds",
            "duplicate_visual_text_rate",
        }
        and not has_occurrences
    ):
        return None, "inapplicable", "no_reference_occurrences"
    if name in {"repeatability_success_rate", "attempt_failure_rate"}:
        scheduled = row.get("scheduled_attempt_count")
        timings = row.get("_timings")
        if scheduled is None and timings is not None:
            scheduled = len(timings)
        if name == "repeatability_success_rate" and scheduled == 1:
            return None, "inapplicable", "repeatability_not_measured"
        incomplete = row.get("unexecuted_attempt_count", 0) > 0 or (
            scheduled is not None
            and row.get("executed_attempt_count", scheduled) != scheduled
        )
        if timings is not None:
            incomplete |= (
                scheduled is None
                or len(timings) != scheduled
                or {timing.get("repetition") for timing in timings}
                != set(range(1, scheduled + 1))
                or any(
                    timing.get("executed", True) is not True
                    or type(timing.get("success")) is not bool
                    for timing in timings
                )
            )
        if incomplete or row.get(name) is None:
            return None, "incomplete", "measurement_attempts_missing"
    else:
        if row.get("first_attempt_executed", True) is not True:
            return None, "incomplete", "first_attempt_not_executed"
        if row.get("first_attempt_success", True) is not True:
            return None, "unavailable", "first_attempt_failed"
    value = row.get(name)
    if value is not None:
        if not np.isfinite(float(value)):
            return None, "unavailable", "non_finite_metric_value"
        return value, "scored", None
    reason = {
        "duplicate_visual_text_rate": "no_assigned_detections",
        "mean_visual_first_detection_delay_seconds": "no_covered_occurrences",
    }.get(name, "first_attempt_output_unavailable")
    return None, "unavailable", reason


def aggregate_quality_statuses(rows):
    """Aggregate status beside a conditional point, without hiding missing rows."""
    estimates = aggregate_quality(rows)
    result = {}
    for name, value in estimates.items():
        observations = [_quality_result(row, name) for row in rows]
        eligible = [entry for entry in observations if entry[1] != "inapplicable"]
        if value is not None:
            status, reason = "scored", None
        elif any(status == "incomplete" for _, status, _ in eligible):
            status = "incomplete"
            reason = (
                "measurement_attempts_missing"
                if name in {"repeatability_success_rate", "attempt_failure_rate"}
                else "first_attempt_not_executed"
            )
        else:
            status = "unavailable" if eligible else "inapplicable"
            reasons = {reason for _, _, reason in eligible or observations}
            reason = (
                next(iter(reasons)) if len(reasons) == 1 else "no_contributing_outputs"
            )
        result[name] = {"status": status, "reason": reason}
    return result


def quality_support_counts(rows):
    """Flat scalar support keys for main; independent samples are source groups."""
    estimates = aggregate_quality(rows)
    result = {}
    for name in QUALITY_DIRECTIONS:
        observations = [_quality_result(row, name) for row in rows]
        eligible = [
            row
            for row, (_, status, _) in zip(rows, observations, strict=True)
            if status != "inapplicable"
        ]
        contributing = [
            row
            for row, (value, _, _) in zip(rows, observations, strict=True)
            if value is not None
        ]
        counts = {
            "scheduled_count": len(rows),
            "eligible_count": len(eligible),
            "contributing_count": len(contributing),
            "independent_source_count": len({_source_group(row) for row in rows}),
            "eligible_source_count": len({_source_group(row) for row in eligible}),
            "contributing_source_count": len(
                {_source_group(row) for row in contributing}
            ),
            "sample_count": len({_source_group(row) for row in contributing})
            if estimates[name] is not None
            else 0,
            "failed_first_attempt_count": sum(
                row.get("first_attempt_executed", True) is True
                and row.get("first_attempt_success", True) is False
                for row in eligible
            ),
        }
        for status in ("unavailable", "incomplete", "inapplicable"):
            counts[status + "_count"] = sum(
                observed_status == status for _, observed_status, _ in observations
            )
        counts["failed_count"] = (
            sum(row.get("failed_attempt_count", 0) > 0 for row in eligible)
            if name in {"repeatability_success_rate", "attempt_failure_rate"}
            else counts["failed_first_attempt_count"]
        )
        reference = (
            "visual_unit"
            if name.startswith("visual_content_")
            else "occurrence"
            if name
            in {
                "timed_visual_occurrence_coverage",
                "mean_visual_first_detection_delay_seconds",
                "duplicate_visual_text_rate",
            }
            else None
        )
        if reference is not None:
            field = "reference_" + reference + "_count"
            counts["planned_" + field] = sum(
                row.get("planned_" + field, row.get(field)) or 0 for row in eligible
            )
            counts["contributing_" + field] = sum(
                row.get(field, row.get("planned_" + field)) or 0 for row in contributing
            )
        result.update({name + "." + key: float(value) for key, value in counts.items()})
    return result


def _source_group(row):
    return str(
        row.get("independent_source_id")
        or row.get("source_group_id")
        or row["sample_id"]
    )


def bootstrap_quality(rows, *, resamples, seed, confidence, minimum_sources=None):
    intervals = bootstrap_sources(
        [{**row, "source_group_id": _source_group(row)} for row in rows],
        aggregate_quality,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        minimum_sources=minimum_sources,
    )
    if intervals:
        for name, status in aggregate_quality_statuses(rows).items():
            if intervals[name]["estimate"] is None:
                intervals[name]["reason"] = status["reason"]
    return intervals


def _duplicates_within_occurrences(occurrences, detections, settings):
    assigned, duplicate = 0, 0
    seen = {}
    for detection in detections:
        eligible = []
        for index, occurrence in enumerate(occurrences):
            if (
                float(occurrence["start"])
                <= float(detection["timestamp"])
                < float(occurrence["end"])
            ):
                similarity = _content_f1(
                    str(occurrence["text"]), str(detection["text"])
                )
                if similarity >= float(settings["content_f1_threshold"]):
                    eligible.append(
                        (
                            -similarity,
                            float(occurrence["start"]),
                            str(occurrence["id"]),
                            index,
                        )
                    )
        if not eligible:
            continue
        index = min(eligible)[-1]
        assigned += 1
        text = str(detection["text"])
        repeated = seen.setdefault(index, set())
        duplicate += int(text in repeated)
        repeated.add(text)
    return duplicate, assigned


def _occurrence_matches(occurrences, detections, settings):
    if not occurrences or not detections:
        return ()
    # Stable identities/frame order, not incidental annotation list order, resolve ties.
    reference_order = sorted(
        range(len(occurrences)), key=lambda index: str(occurrences[index]["id"])
    )
    detection_order = sorted(
        range(len(detections)),
        key=lambda index: (
            detections[index]["timestamp"],
            detections[index]["index"],
            index,
        ),
    )
    occurrences = [occurrences[index] for index in reference_order]
    detections = [detections[index] for index in detection_order]
    threshold = float(settings["content_f1_threshold"])
    tolerance = float(settings["frame_timestamp_tolerance_seconds"])
    eligible = np.zeros((len(occurrences), len(detections)), dtype=bool)
    delays = np.zeros(eligible.shape, dtype=np.float64)
    for left, occurrence in enumerate(occurrences):
        for right, detection in enumerate(detections):
            timestamp = float(detection["timestamp"])
            if not (
                float(occurrence["start"]) - tolerance
                <= timestamp
                < float(occurrence["end"]) + tolerance
            ):
                continue
            similarity = _content_f1(str(occurrence["text"]), str(detection["text"]))
            if similarity < threshold:
                continue
            eligible[left, right] = True
            delays[left, right] = timestamp - float(occurrence["start"])

    from scipy.optimize import linear_sum_assignment

    # Solve cardinality before delay. A large additive match reward would round
    # away small, real delay differences. Exactly N-K dummy columns force K real
    # matches without changing their raw delay costs or allocating an M*M matrix.
    rows, columns = linear_sum_assignment(-eligible.astype(int))
    cardinality = int(eligible[rows, columns].sum())
    if not cardinality:
        return ()
    costs = np.column_stack(
        (
            np.where(eligible, delays, np.inf),
            np.zeros((len(occurrences), len(occurrences) - cardinality)),
        )
    )
    rows, columns = linear_sum_assignment(costs)

    def total_delay(assignment):
        # Summing rounded pair differences can break genuine assignment ties.
        # Exact endpoint arithmetic preserves the recorded float precision,
        # without a tolerance that would erase real small improvements.
        return sum(
            Fraction(float(detections[right]["timestamp"]))
            - Fraction(float(occurrences[left]["start"]))
            for left, right in enumerate(assignment)
            if right < len(detections)
        )

    optimum = total_delay(columns)
    # SciPy is deterministic, but sorted inputs alone do not enforce the frozen
    # ID/frame tie rule. Fix the earliest feasible detection for each stable ID,
    # preserving the optimal cardinality and delay exactly (no epsilon penalty).
    used = set()
    for left in range(len(occurrences)):
        for right in range(min(int(columns[left]), len(detections))):
            if right in used or not eligible[left, right]:
                continue
            available = [
                column
                for column in range(costs.shape[1])
                if column not in used and column != right
            ]
            try:
                _, remaining_columns = linear_sum_assignment(
                    costs[left + 1 :, available]
                )
            except ValueError:  # This prefix cannot complete the maximum matching.
                continue
            trial = [
                *columns[:left],
                right,
                *(available[int(column)] for column in remaining_columns),
            ]
            if total_delay(trial) == optimum:
                columns = np.asarray(trial, dtype=int)
                break
        used.add(int(columns[left]))
    return tuple(
        (reference_order[int(left)], detection_order[int(right)])
        for left, right in zip(rows, columns, strict=True)
        if right < len(detections)
    )


def _reference_units(value: object) -> list[str]:
    if isinstance(value, str):
        return _normalized_lines(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if any(not isinstance(item, str) for item in value):
            raise ValueError("Video visible-text reference units must be strings")
        return [unit for item in value for unit in _normalized_lines(item)]
    raise ValueError("Video sample requires reference_visual_text")


def _occurrences(value: object, *, duration: float) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Video sample requires visual_occurrences")
    result = []
    for raw in value:
        if (
            not isinstance(raw, Mapping)
            or not isinstance(raw.get("text"), str)
            or not raw["text"].strip()
        ):
            raise ValueError("Video visual occurrence is malformed")
        start, end = float(raw.get("start", -1)), float(raw.get("end", -1))
        if (
            not np.isfinite(start)
            or not np.isfinite(end)
            or start < 0
            or end <= start
            or end > duration
        ):
            raise ValueError("Video visual occurrence has invalid boundaries")
        if (
            not isinstance(raw.get("id"), str)
            or not raw["id"]
            or not _normalized_lines(str(raw["text"]))
        ):
            raise ValueError("Video visual occurrence has invalid boundaries")
        result.append(
            {"id": raw["id"], "text": str(raw["text"]), "start": start, "end": end}
        )
    if len({value["id"] for value in result}) != len(result):
        raise ValueError("Video occurrence IDs must be unique")
    return result


def _normalized_lines(value: str) -> list[str]:
    return [
        normalized
        for line in value.splitlines()
        if (normalized := normalize_prose(line))
    ]


def _content_f1(reference: str, prediction: str) -> float:
    expected = Counter(normalized_tokens(reference))
    observed = Counter(normalized_tokens(prediction))
    overlap = sum((expected & observed).values())
    total = sum(expected.values()) + sum(observed.values())
    # The equivalent harmonic-mean calculation can round an exact 0.5 below
    # the frozen eligibility boundary (e.g. lengths 8/20 with 7 shared tokens).
    return 2 * overlap / total if total else 0.0
