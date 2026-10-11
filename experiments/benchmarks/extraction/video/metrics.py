"""Frozen visible-text metrics for the dedicated video benchmark."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

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
    predictions: Sequence[Mapping[str, object]],
    occurrence_matching: Mapping[str, object],
) -> dict[str, object]:
    reference_units = _reference_units(item.get("reference_visual_text"))
    predicted_units = [
        unit
        for prediction in predictions
        for unit in _normalized_lines(str(prediction.get("text", "")))
    ]
    expected, observed = set(reference_units), set(predicted_units)
    overlap = len(expected & observed)
    if not expected and not observed:
        precision = recall = f1 = 1.0
    else:
        precision, recall, f1 = precision_recall_f1(
            overlap, len(observed) - overlap, len(expected) - overlap
        )
    occurrences = _occurrences(
        item.get("visual_occurrences"), duration=float(item["duration_seconds"])
    )
    detections = [
        {"text": unit, "timestamp": float(prediction["timestamp"]), "index": index}
        for index, prediction in enumerate(predictions)
        for unit in _normalized_lines(str(prediction.get("text", "")))
    ]
    for detection in detections:
        if not np.isfinite(detection["timestamp"]) or not 0 <= detection[
            "timestamp"
        ] <= float(item["duration_seconds"]):
            raise ValueError("Video detection timestamp is outside the video")
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
        "sample_id": str(item["id"]),
        "duration_seconds": float(item["duration_seconds"]),
        "reference_visual_unit_count": len(expected),
        "predicted_visual_unit_count": len(observed),
        "matched_visual_unit_count": overlap,
        "predicted_visual_occurrence_count": len(predicted_units),
        "duplicate_visual_unit_count": duplicate_count,
        "assigned_visual_detection_count": assigned_count,
        "reference_occurrence_count": len(occurrences),
        "matched_occurrence_count": len(matches),
        "first_detection_delay_total_seconds": sum(delays),
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
        values = [float(row[name]) for row in rows if row.get(name) is not None]
        result[name] = float(np.mean(values)) if values else None
    return result


def quality_statuses(row, *, has_occurrences):
    statuses = {}
    for name in QUALITY_DIRECTIONS:
        value = row.get(name)
        inapplicable = (name == "repeatability_success_rate" and value is None) or (
            name
            in {
                "timed_visual_occurrence_coverage",
                "mean_visual_first_detection_delay_seconds",
            }
            and not has_occurrences
        )
        reason = "no_reference_occurrences" if inapplicable else "first_attempt_failed"
        if name == "repeatability_success_rate" and inapplicable:
            reason = "repeatability_not_measured"
        elif row.get("first_attempt_success", True):
            if name == "duplicate_visual_text_rate":
                reason = "no_assigned_detections"
            elif (
                name == "mean_visual_first_detection_delay_seconds" and not inapplicable
            ):
                reason = "no_covered_occurrences"
        statuses[name] = {
            "status": "inapplicable"
            if inapplicable
            else "scored"
            if value is not None
            else "unavailable",
            "reason": reason if value is None else None,
        }
    return statuses


def bootstrap_quality(rows, *, resamples, seed, confidence, minimum_sources=None):
    return bootstrap_sources(
        [
            {**row, "source_group_id": row.get("source_group_id", row["sample_id"])}
            for row in rows
        ],
        aggregate_quality,
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        minimum_sources=minimum_sources,
    )


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
        range(len(detections)), key=lambda index: (detections[index]["index"], index)
    )
    occurrences = [occurrences[index] for index in reference_order]
    detections = [detections[index] for index in detection_order]
    threshold = float(settings["content_f1_threshold"])
    tolerance = float(settings["frame_timestamp_tolerance_seconds"])
    scores = np.zeros((len(occurrences), len(detections)), dtype=np.float64)
    eligible = np.zeros_like(scores, dtype=bool)
    delays = np.zeros_like(scores, dtype=np.float64)
    maximum_delay = 0.0
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
            delay = max(0.0, timestamp - float(occurrence["start"]))
            delays[left, right] = delay
            maximum_delay = max(maximum_delay, delay)

    # One additional match must outweigh every possible delay improvement in all
    # other matches. Among maximum-cardinality assignments, maximizing this score
    # therefore minimizes the documented raw delay in seconds.
    match_bound = min(len(occurrences), len(detections))
    cardinality_weight = (match_bound + 1) * (maximum_delay + 1.0)
    scores[eligible] = cardinality_weight - delays[eligible]
    from scipy.optimize import linear_sum_assignment

    rows, columns = linear_sum_assignment(-scores)
    return tuple(
        (reference_order[int(left)], detection_order[int(right)])
        for left, right in zip(rows, columns, strict=True)
        if eligible[left, right]
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
        if not isinstance(raw, Mapping) or not str(raw.get("text", "")).strip():
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
    return precision_recall_f1(
        overlap,
        sum(observed.values()) - overlap,
        sum(expected.values()) - overlap,
    )[2]
