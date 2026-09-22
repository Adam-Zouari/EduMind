"""Candidate identities and deterministic FFmpeg frame selectors."""

from __future__ import annotations

from dataclasses import dataclass

from edumind.extraction.video_policy import KeyframePolicy
from edumind.extraction.video_policy import frame_command as policy_command
from experiments.benchmarks.extraction.video.protocol import VideoProtocol


@dataclass(frozen=True)
class VideoCandidate:
    candidate: str
    strategy: str
    include_frame_zero: bool
    frame_sync: str
    interval_seconds: int | None = None
    scene_threshold: float | None = None
    maximum_gap_seconds: int | None = None

    @property
    def ffmpeg_filter(self) -> str:
        return self.policy.ffmpeg_filter

    @property
    def policy(self) -> KeyframePolicy:
        return KeyframePolicy(
            self.strategy,
            interval_seconds=self.interval_seconds,
            scene_threshold=self.scene_threshold,
            maximum_gap_seconds=self.maximum_gap_seconds,
            include_frame_zero=self.include_frame_zero,
            frame_sync=self.frame_sync,
        )


def fixed_candidates(protocol: VideoProtocol) -> tuple[str, ...]:
    return tuple(f"video-fixed-{seconds}s" for seconds in protocol.fixed_intervals)


def scene_candidates(protocol: VideoProtocol) -> tuple[str, ...]:
    return tuple(
        f"video-scene-{threshold:.2f}" for threshold in protocol.scene_thresholds
    )


def hybrid_candidates(
    protocol: VideoProtocol, scene_threshold: float
) -> tuple[str, ...]:
    return tuple(
        f"video-hybrid-{scene_threshold:.2f}-{seconds}s"
        for seconds in protocol.hybrid_gaps
    )


def all_candidates(protocol: VideoProtocol, scene_threshold: float) -> tuple[str, ...]:
    return (
        *fixed_candidates(protocol),
        *scene_candidates(protocol),
        *hybrid_candidates(protocol, scene_threshold),
    )


def parse_candidate(value: str, protocol: VideoProtocol) -> VideoCandidate:
    parts = value.split("-")
    try:
        if len(parts) == 3 and parts[:2] == ["video", "fixed"]:
            seconds = int(parts[2].removesuffix("s"))
            if seconds in protocol.fixed_intervals:
                return VideoCandidate(
                    value,
                    "fixed",
                    protocol.include_frame_zero,
                    protocol.frame_sync,
                    interval_seconds=seconds,
                )
        if len(parts) == 3 and parts[:2] == ["video", "scene"]:
            threshold = float(parts[2])
            if (
                threshold in protocol.scene_thresholds
                and parts[2] == f"{threshold:.2f}"
            ):
                return VideoCandidate(
                    value,
                    "scene",
                    protocol.include_frame_zero,
                    protocol.frame_sync,
                    scene_threshold=threshold,
                )
        if len(parts) == 4 and parts[:2] == ["video", "hybrid"]:
            threshold = float(parts[2])
            gap = int(parts[3].removesuffix("s"))
            if threshold in protocol.scene_thresholds and gap in protocol.hybrid_gaps:
                canonical = f"video-hybrid-{threshold:.2f}-{gap}s"
                if value == canonical:
                    return VideoCandidate(
                        value,
                        "hybrid",
                        protocol.include_frame_zero,
                        protocol.frame_sync,
                        scene_threshold=threshold,
                        maximum_gap_seconds=gap,
                    )
    except ValueError:
        pass
    raise ValueError(f"Unknown video benchmark candidate: {value}")


def frame_command(candidate: VideoCandidate, source: str, pattern: str) -> list[str]:
    return policy_command(candidate.policy, source, pattern)
