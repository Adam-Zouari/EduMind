"""Quality contract for audio transcription."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from experiments.benchmarks.extraction.metrics import base_scores, timestamp_mae


def metrics(reference: str, hypothesis: str, item: Mapping[str, object], document):
    scores = base_scores(reference, hypothesis)
    reference_timestamps = item.get("reference_timestamps")
    if isinstance(reference_timestamps, Sequence) and not isinstance(reference_timestamps, str):
        predicted = [
            segment.timestamp_start
            for segment in document.segments
            if segment.timestamp_start is not None
        ]
        if reference_timestamps and len(reference_timestamps) == len(predicted):
            scores["timestamp_mae_seconds"] = timestamp_mae(
                [float(value) for value in reference_timestamps], predicted
            )
        scores["timestamp_alignment_coverage"] = min(
            len(reference_timestamps), len(predicted)
        ) / max(1, len(reference_timestamps), len(predicted))
    reference_segments = item.get("reference_segments")
    if isinstance(reference_segments, Sequence) and not isinstance(reference_segments, str):
        expected = [
            float(segment[key])
            for segment in reference_segments
            if isinstance(segment, Mapping)
            for key in ("start", "end")
            if segment.get(key) is not None
        ]
        predicted = [
            value
            for segment in document.segments
            for value in (segment.timestamp_start, segment.timestamp_end)
            if value is not None
        ]
        if expected and len(expected) == len(predicted):
            scores["segment_boundary_mae_seconds"] = timestamp_mae(expected, predicted)
    return scores


def directions() -> dict[str, str]:
    return {
        "character_error_rate": "min",
        "word_error_rate": "min",
        "content_f1": "max",
        "reading_order_accuracy": "max",
        "missing_text_rate": "min",
        "hallucinated_text_rate": "min",
        "timestamp_mae_seconds": "min",
        "segment_boundary_mae_seconds": "min",
        "timestamp_alignment_coverage": "max",
        "operational.p95_latency_seconds": "min",
        "operational.peak_ram_mb": "min",
        "operational.real_time_factor": "min",
    }

