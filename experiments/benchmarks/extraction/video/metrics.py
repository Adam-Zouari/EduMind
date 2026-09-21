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


QUALITY_DIRECTIONS = {
    "visual_content_precision": "max",
    "visual_content_recall": "max",
    "visual_content_f1": "max",
    "mean_visual_first_detection_delay_seconds": "min",
    "timed_visual_occurrence_coverage": "max",
    "duplicate_visual_text_rate": "min",
}
OPERATIONAL_DIRECTIONS = {
    "visual_real_time_factor": "min",
    "p50_warm_visual_latency_seconds": "min",
    "p95_warm_visual_latency_seconds": "min",
    "cold_visual_pipeline_load_seconds": "min",
    "peak_visual_process_tree_ram_mb": "min",
    "peak_visual_vram_mb": "min",
    "mean_selected_frames_per_video": "min",
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
        precision = recall = f1 = None
    else:
        precision, recall, f1 = precision_recall_f1(
            overlap, len(observed) - overlap, len(expected) - overlap
        )
    duplicate = (
        (len(predicted_units) - len(observed)) / len(predicted_units)
        if predicted_units
        else None
    )
    occurrences = _occurrences(
        item.get("visual_occurrences"), duration=float(item["duration_seconds"])
    )
    detections = [
        {"text": unit, "timestamp": float(prediction["timestamp"]), "index": index}
        for index, prediction in enumerate(predictions)
        for unit in sorted(set(_normalized_lines(str(prediction.get("text", "")))))
    ]
    matches = _occurrence_matches(occurrences, detections, occurrence_matching)
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
        "duplicate_visual_unit_count": len(predicted_units) - len(observed),
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


def aggregate_quality(rows: Sequence[Mapping[str, object]]) -> dict[str, float | None]:
    if not rows:
        raise ValueError("Video quality aggregation requires samples")
    reference_units = sum(int(row["reference_visual_unit_count"]) for row in rows)
    predicted_units = sum(int(row["predicted_visual_unit_count"]) for row in rows)
    matched_units = sum(int(row["matched_visual_unit_count"]) for row in rows)
    if not reference_units and not predicted_units:
        precision = recall = f1 = None
    else:
        precision, recall, f1 = precision_recall_f1(
            matched_units,
            predicted_units - matched_units,
            reference_units - matched_units,
        )
    reference_occurrences = sum(int(row["reference_occurrence_count"]) for row in rows)
    match_count = sum(int(row["matched_occurrence_count"]) for row in rows)
    predicted_occurrences = sum(
        int(row["predicted_visual_occurrence_count"]) for row in rows
    )
    duplicate_units = sum(int(row["duplicate_visual_unit_count"]) for row in rows)
    return {
        "visual_content_precision": precision,
        "visual_content_recall": recall,
        "visual_content_f1": f1,
        "mean_visual_first_detection_delay_seconds": (
            sum(float(row["first_detection_delay_total_seconds"]) for row in rows)
            / match_count
            if match_count
            else None
        ),
        "timed_visual_occurrence_coverage": (
            match_count / reference_occurrences if reference_occurrences else None
        ),
        "duplicate_visual_text_rate": (
            duplicate_units / predicted_occurrences if predicted_occurrences else None
        ),
    }


def bootstrap_quality(rows, *, resamples: int, seed: int, confidence: float):
    if not resamples or len(rows) < 2:
        return {}
    rng = np.random.default_rng(seed)
    draws = {name: [] for name in QUALITY_DIRECTIONS}
    for _ in range(resamples):
        sampled = [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        values = aggregate_quality(sampled)
        for name, value in values.items():
            if value is not None:
                draws[name].append(value)
    estimates = aggregate_quality(rows)
    alpha = (1.0 - confidence) / 2.0
    return {
        name: {
            "estimate": estimates[name],
            "lower": float(np.quantile(values, alpha)),
            "upper": float(np.quantile(values, 1.0 - alpha)),
            "confidence": confidence,
            "resamples": len(values),
        }
        for name, values in draws.items()
        if values
    }


def stitch_text(previous: str, current: str, *, maximum_overlap_tokens: int) -> str:
    """Remove the longest normalized suffix/prefix overlap exactly once."""

    left_raw = previous.split()
    right_raw = current.split()
    left = [normalize_prose(token) for token in left_raw]
    right = [normalize_prose(token) for token in right_raw]
    limit = min(len(left), len(right), maximum_overlap_tokens)
    overlap = next(
        (width for width in range(limit, 0, -1) if left[-width:] == right[:width]),
        0,
    )
    return " ".join([*left_raw, *right_raw[overlap:]]).strip()


def _occurrence_matches(occurrences, detections, settings):
    if not occurrences or not detections:
        return ()
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
                <= float(occurrence["end"]) + tolerance
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
        (int(left), int(right))
        for left, right in zip(rows, columns)
        if eligible[left, right]
    )


def _reference_units(value: object) -> list[str]:
    if isinstance(value, str):
        return _normalized_lines(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [unit for item in value for unit in _normalized_lines(str(item))]
    raise ValueError("Video sample requires reference_visual_text")


def _occurrences(value: object, *, duration: float) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Video sample requires visual_occurrences")
    result = []
    for raw in value:
        if not isinstance(raw, Mapping) or not str(raw.get("text", "")).strip():
            raise ValueError("Video visual occurrence is malformed")
        start, end = float(raw.get("start", -1)), float(raw.get("end", -1))
        if start < 0 or end < start or end > duration + 1e-6:
            raise ValueError("Video visual occurrence has invalid boundaries")
        result.append({"text": str(raw["text"]), "start": start, "end": end})
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
