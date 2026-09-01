"""ASR-specific scoring, pooled aggregation, and clip bootstrap intervals."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

METRIC_DIRECTIONS = {
    "word_error_rate": "min",
    "character_error_rate": "min",
    "word_substitution_rate": "min",
    "word_deletion_rate": "min",
    "word_insertion_rate": "min",
    "timestamp_boundary_mae_seconds": "min",
    "timestamp_alignment_coverage": "max",
    "empty_transcript_rate": "min",
    "nonspeech_false_transcription_rate": "min",
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


@dataclass(frozen=True)
class Alignment:
    substitutions: int
    deletions: int
    insertions: int
    exact_matches: tuple[tuple[int, int], ...]


def normalize_transcript(text: str) -> str:
    """Apply only the frozen evaluator normalization."""

    normalized = unicodedata.normalize("NFC", text).casefold()
    without_punctuation = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(without_punctuation.split())


def align_sequences(reference: Sequence[str], prediction: Sequence[str]) -> Alignment:
    """Return deterministic Levenshtein counts and exact token index matches."""

    rows, columns = len(reference) + 1, len(prediction) + 1
    distance = [[0] * columns for _ in range(rows)]
    for index in range(rows):
        distance[index][0] = index
    for index in range(columns):
        distance[0][index] = index
    for left in range(1, rows):
        for right in range(1, columns):
            substitution_cost = int(reference[left - 1] != prediction[right - 1])
            distance[left][right] = min(
                distance[left - 1][right] + 1,
                distance[left][right - 1] + 1,
                distance[left - 1][right - 1] + substitution_cost,
            )

    substitutions = deletions = insertions = 0
    matches: list[tuple[int, int]] = []
    left, right = len(reference), len(prediction)
    while left or right:
        if (
            left
            and right
            and reference[left - 1] == prediction[right - 1]
            and distance[left][right] == distance[left - 1][right - 1]
        ):
            matches.append((left - 1, right - 1))
            left -= 1
            right -= 1
        elif left and right and distance[left][right] == distance[left - 1][right - 1] + 1:
            substitutions += 1
            left -= 1
            right -= 1
        elif left and distance[left][right] == distance[left - 1][right] + 1:
            deletions += 1
            left -= 1
        else:
            insertions += 1
            right -= 1
    matches.reverse()
    return Alignment(substitutions, deletions, insertions, tuple(matches))


def score_speech(
    item: Mapping[str, object],
    prediction: str,
    predicted_segments: Sequence[Mapping[str, object]],
    *,
    quality_latency_seconds: float,
    warnings: Sequence[str] = (),
) -> dict[str, object]:
    reference = normalize_transcript(str(item["reference"]))
    hypothesis = normalize_transcript(prediction)
    reference_words = reference.split()
    predicted_words = hypothesis.split()
    word_alignment = align_sequences(reference_words, predicted_words)
    character_alignment = align_sequences(tuple(reference), tuple(hypothesis))
    references = _segments(item.get("reference_segments"), "reference")
    predictions = _segments(predicted_segments, "prediction")
    _validate_predicted_timeline(predictions, float(item["duration_seconds"]))
    timestamp = _timestamp_totals(references, predictions)
    return {
        "sample_id": str(item["id"]),
        "sample_type": "speech",
        "condition": str(item.get("condition", "unspecified")),
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
        "empty_transcript": int(not predicted_words),
        "nonspeech_false_transcription": None,
        "quality_latency_seconds": quality_latency_seconds,
        "warnings": list(warnings),
    }


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
        "condition": str(item.get("nonspeech_kind", "unspecified")),
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
        "quality_latency_seconds": latency_seconds,
        "warnings": list(warnings),
    }


def aggregate(
    sample_rows: Sequence[Mapping[str, object]],
    timing_rows: Sequence[Mapping[str, object]],
    *,
    cold_model_load_seconds: float,
    peak_process_tree_ram_mb: float,
    peak_vram_mb: float,
    resamples: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    speech = [row for row in sample_rows if row["sample_type"] == "speech"]
    nonspeech = [row for row in sample_rows if row["sample_type"] == "nonspeech"]
    metrics = _aggregate_rows(speech, nonspeech, timing_rows)
    metrics.update(
        {
            "cold_model_load_seconds": cold_model_load_seconds,
            "peak_process_tree_ram_mb": peak_process_tree_ram_mb,
            "peak_vram_mb": peak_vram_mb,
        }
    )
    missing = sorted(set(METRIC_DIRECTIONS) - set(metrics))
    if missing:
        raise ValueError("ASR candidate did not produce required metrics: " + ", ".join(missing))
    intervals = (
        _bootstrap(speech, nonspeech, timing_rows, resamples=resamples, seed=seed)
        if resamples
        else {}
    )
    for name, interval in intervals.items():
        interval["estimate"] = metrics[name]
    return metrics, intervals


def _aggregate_rows(
    speech, nonspeech, timing_rows, *, allow_undefined_timestamp_mae: bool = False
) -> dict[str, float | None]:
    if not speech:
        raise ValueError("ASR aggregation requires speech samples")
    if not nonspeech:
        raise ValueError("ASR aggregation requires nonspeech controls")
    reference_words = _sum(speech, "reference_word_count")
    reference_characters = _sum(speech, "reference_character_count")
    if not reference_words or not reference_characters:
        raise ValueError("ASR references must contain words and characters")
    substitutions = _sum(speech, "word_substitutions")
    deletions = _sum(speech, "word_deletions")
    insertions = _sum(speech, "word_insertions")
    character_errors = sum(
        _sum(speech, name)
        for name in (
            "character_substitutions",
            "character_deletions",
            "character_insertions",
        )
    )
    aligned_segments = _sum(speech, "aligned_timed_segment_count")
    reference_segments = _sum(speech, "reference_timed_segment_count")
    boundary_count = _sum(speech, "timestamp_boundary_count")
    if not reference_segments:
        raise ValueError("ASR references must contain timed segments")
    if (not aligned_segments or not boundary_count) and not allow_undefined_timestamp_mae:
        raise ValueError("No reference timestamp segment could be aligned")
    if not timing_rows:
        raise ValueError("ASR aggregation requires measured timing rows")
    per_clip: dict[str, list[float]] = {}
    for row in timing_rows:
        per_clip.setdefault(str(row["sample_id"]), []).append(float(row["latency_seconds"]))
    medians = [float(np.median(values)) for values in per_clip.values()]
    measured_seconds = sum(float(row["latency_seconds"]) for row in timing_rows)
    measured_audio_seconds = sum(float(row["duration_seconds"]) for row in timing_rows)
    return {
        "word_error_rate": (substitutions + deletions + insertions) / reference_words,
        "character_error_rate": character_errors / reference_characters,
        "word_substitution_rate": substitutions / reference_words,
        "word_deletion_rate": deletions / reference_words,
        "word_insertion_rate": insertions / reference_words,
        "timestamp_boundary_mae_seconds": (
            _sum(speech, "timestamp_boundary_error_seconds") / boundary_count
            if boundary_count
            else None
        ),
        "timestamp_alignment_coverage": aligned_segments / reference_segments,
        "empty_transcript_rate": _sum(speech, "empty_transcript") / len(speech),
        "nonspeech_false_transcription_rate": _sum(
            nonspeech, "nonspeech_false_transcription"
        )
        / len(nonspeech),
        "real_time_factor": measured_seconds / measured_audio_seconds,
        "p50_warm_clip_latency_seconds": float(np.quantile(medians, 0.50)),
        "p95_warm_clip_latency_seconds": float(np.quantile(medians, 0.95)),
    }


def _bootstrap(speech, nonspeech, timing_rows, *, resamples: int, seed: int):
    timings_by_sample: dict[str, list[Mapping[str, object]]] = {}
    for row in timing_rows:
        timings_by_sample.setdefault(str(row["sample_id"]), []).append(row)
    rng = np.random.default_rng(seed)
    estimates = {name: [] for name in INTERVAL_METRICS}
    for _ in range(resamples):
        sampled_speech = [speech[index] for index in rng.integers(0, len(speech), len(speech))]
        sampled_nonspeech = [
            nonspeech[index] for index in rng.integers(0, len(nonspeech), len(nonspeech))
        ]
        sampled_timings = []
        sampled_speech_with_ids = []
        for draw, row in enumerate(sampled_speech):
            bootstrap_id = f"{row['sample_id']}#{draw}"
            sampled_speech_with_ids.append({**row, "sample_id": bootstrap_id})
            sampled_timings.extend(
                {**timing, "sample_id": bootstrap_id}
                for timing in timings_by_sample[str(row["sample_id"])]
            )
        values = _aggregate_rows(
            sampled_speech_with_ids,
            sampled_nonspeech,
            sampled_timings,
            allow_undefined_timestamp_mae=True,
        )
        for name in estimates:
            value = values[name]
            if value is not None:
                estimates[name].append(value)
    result = {}
    for name, values in estimates.items():
        if not values:
            raise ValueError(f"Bootstrap produced no valid estimates for {name}")
        result[name] = {
            "estimate": float(np.mean(values)),
            "lower": float(np.quantile(values, 0.025)),
            "upper": float(np.quantile(values, 0.975)),
            "confidence": 0.95,
            "resamples": len(values),
        }
    return result


def _timestamp_totals(reference_segments, predicted_segments) -> dict[str, int | float]:
    reference_tokens, reference_owners = _segment_tokens(reference_segments)
    predicted_tokens, predicted_owners = _segment_tokens(predicted_segments)
    alignment = align_sequences(reference_tokens, predicted_tokens)
    matched: dict[int, list[int]] = {}
    for reference_index, predicted_index in alignment.exact_matches:
        matched.setdefault(reference_owners[reference_index], []).append(
            predicted_owners[predicted_index]
        )
    error = 0.0
    aligned = 0
    for reference_index, predicted_indices in matched.items():
        reference = reference_segments[reference_index]
        predicted = [predicted_segments[index] for index in predicted_indices]
        predicted_start = min(float(segment["start"]) for segment in predicted)
        predicted_end = max(float(segment["end"]) for segment in predicted)
        error += abs(float(reference["start"]) - predicted_start)
        error += abs(float(reference["end"]) - predicted_end)
        aligned += 1
    return {
        "reference_timed_segment_count": len(reference_segments),
        "aligned_timed_segment_count": aligned,
        "timestamp_boundary_error_seconds": error,
        "timestamp_boundary_count": aligned * 2,
    }


def _segments(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError(f"ASR {label} timestamp segments are missing")
    segments = []
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError(f"ASR {label} timestamp segment is malformed")
        text = str(raw.get("text", "")).strip()
        start, end = float(raw.get("start", -1)), float(raw.get("end", -1))
        if not text or start < 0 or end <= start:
            raise ValueError(f"ASR {label} timestamp segment is invalid")
        segments.append({"text": text, "start": start, "end": end})
    return segments


def _segment_tokens(segments):
    tokens: list[str] = []
    owners: list[int] = []
    for index, segment in enumerate(segments):
        current = normalize_transcript(str(segment["text"])).split()
        tokens.extend(current)
        owners.extend([index] * len(current))
    return tokens, owners


def _validate_predicted_timeline(segments, duration: float) -> None:
    previous_start = -1.0
    for segment in segments:
        start, end = float(segment["start"]), float(segment["end"])
        if start < previous_start or end > duration + 0.1:
            raise ValueError("ASR prediction contains invalid or unordered timestamps")
        previous_start = start


def _sum(rows, name: str) -> float:
    return sum(float(row[name]) for row in rows if row.get(name) is not None)
