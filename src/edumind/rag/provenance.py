"""Project exact source-segment locations onto indexed/displayed spans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def project_locations(
    segments: Sequence[Mapping[str, object]], start: int, end: int
) -> list[dict[str, object]]:
    locations = []
    for segment in segments:
        left, right = int(segment["start"]), int(segment["end"])
        overlap_start, overlap_end = max(start, left), min(end, right)
        if overlap_start >= overlap_end:
            continue
        location = {
            **segment,
            "segment_start": segment.get("segment_start", left),
            "segment_end": segment.get("segment_end", right),
            "start": overlap_start,
            "end": overlap_end,
        }
        if isinstance(text := segment.get("text"), str):
            location["text"] = text[overlap_start - left : overlap_end - left]
        locations.append(location)
    return locations
