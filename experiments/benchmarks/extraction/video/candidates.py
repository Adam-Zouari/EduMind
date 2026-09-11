"""Candidate identities and deterministic FFmpeg frame selectors."""

from __future__ import annotations

from dataclasses import dataclass

FIXED_CANDIDATES = tuple(f"video-fixed-{seconds}s" for seconds in (5, 10, 20))
SCENE_CANDIDATES = tuple(
    f"video-scene-{threshold:.2f}" for threshold in (0.30, 0.40, 0.50)
)


@dataclass(frozen=True)
class VideoCandidate:
    candidate: str
    strategy: str
    interval_seconds: int | None = None
    scene_threshold: float | None = None
    maximum_gap_seconds: int | None = None

    @property
    def ffmpeg_filter(self) -> str:
        first = "eq(n,0)"
        if self.strategy == "fixed":
            return f"select='{first}+gte(t-prev_selected_t,{self.interval_seconds})'"
        if self.strategy == "scene":
            return f"select='{first}+gt(scene,{self.scene_threshold:.2f})'"
        return (
            f"select='{first}+gt(scene,{self.scene_threshold:.2f})+"
            f"isnan(prev_selected_t)+gte(t-prev_selected_t,{self.maximum_gap_seconds})'"
        )


def hybrid_candidates(scene_threshold: float) -> tuple[str, ...]:
    return tuple(
        f"video-hybrid-{scene_threshold:.2f}-{seconds}s" for seconds in (5, 10, 20)
    )


def all_candidates(scene_threshold: float) -> tuple[str, ...]:
    return (*FIXED_CANDIDATES, *SCENE_CANDIDATES, *hybrid_candidates(scene_threshold))


def parse_candidate(value: str) -> VideoCandidate:
    parts = value.split("-")
    try:
        if len(parts) == 3 and parts[:2] == ["video", "fixed"]:
            seconds = int(parts[2].removesuffix("s"))
            if seconds in {5, 10, 20}:
                return VideoCandidate(value, "fixed", interval_seconds=seconds)
        if len(parts) == 3 and parts[:2] == ["video", "scene"]:
            threshold = float(parts[2])
            if threshold in {0.30, 0.40, 0.50} and parts[2] == f"{threshold:.2f}":
                return VideoCandidate(value, "scene", scene_threshold=threshold)
        if len(parts) == 4 and parts[:2] == ["video", "hybrid"]:
            threshold = float(parts[2])
            gap = int(parts[3].removesuffix("s"))
            if 0 <= threshold <= 1 and gap in {5, 10, 20}:
                canonical = f"video-hybrid-{threshold:.2f}-{gap}s"
                if value == canonical:
                    return VideoCandidate(
                        value,
                        "hybrid",
                        scene_threshold=threshold,
                        maximum_gap_seconds=gap,
                    )
    except ValueError:
        pass
    raise ValueError(f"Unknown video benchmark candidate: {value}")


def frame_command(candidate: VideoCandidate, source: str, pattern: str) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-y",
        "-i",
        source,
        "-vf",
        f"{candidate.ffmpeg_filter},showinfo",
        "-fps_mode",
        "vfr",
        pattern,
    ]
