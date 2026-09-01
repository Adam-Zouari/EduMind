"""Quality contract for combined audio and visual video extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from experiments.benchmarks.common.metrics import word_error_rate
from experiments.benchmarks.extraction.video.metrics import (
    content_scores,
    duplicate_line_rate,
    timestamp_mae,
)


def metrics(reference: str, hypothesis: str, item: Mapping[str, object], document):
    scores: dict[str, float] = {}
    audio_count = int(document.metadata.get("audio_segment_count", 0))
    audio_text = "\n".join(segment.text for segment in document.segments[:audio_count])
    visual_segments = document.segments[audio_count:]
    visual_text = "\n".join(segment.text for segment in visual_segments)
    transcript_reference = str(item.get("reference_transcript", reference))
    scores["transcript_word_error_rate"] = word_error_rate(transcript_reference, audio_text)
    scores["complete_content_recall"] = content_scores(reference, hypothesis)[
        "content_recall"
    ]
    if "reference_visual_text" in item:
        visual_reference = item["reference_visual_text"]
        if isinstance(visual_reference, Sequence) and not isinstance(visual_reference, str):
            visual_reference = "\n".join(str(value) for value in visual_reference)
        visual_scores = content_scores(str(visual_reference), visual_text)
        scores.update(
            {
                "visual_text_precision": visual_scores["content_precision"],
                "visual_text_recall": visual_scores["content_recall"],
                "visual_text_f1": visual_scores["content_f1"],
                "duplicate_visual_text_rate": duplicate_line_rate(visual_text),
            }
        )
    visual_timestamps = item.get("reference_visual_timestamps")
    predicted_visual_timestamps = [
        segment.timestamp_start
        for segment in visual_segments
        if segment.timestamp_start is not None
    ]
    if (
        isinstance(visual_timestamps, Sequence)
        and not isinstance(visual_timestamps, str)
        and visual_timestamps
        and len(visual_timestamps) == len(predicted_visual_timestamps)
    ):
        scores["audio_visual_alignment_mae_seconds"] = timestamp_mae(
            [float(value) for value in visual_timestamps], predicted_visual_timestamps
        )
    return scores


def directions() -> dict[str, str]:
    return {
        "transcript_word_error_rate": "min",
        "visual_text_precision": "max",
        "visual_text_recall": "max",
        "visual_text_f1": "max",
        "audio_visual_alignment_mae_seconds": "min",
        "complete_content_recall": "max",
        "operational.p95_latency_seconds": "min",
        "operational.peak_process_tree_ram_mb": "min",
        "operational.real_time_factor": "min",
    }
