"""Pure, shared video keyframe policy and FFmpeg selector construction."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeyframePolicy:
    strategy: str
    interval_seconds: float | None = None
    scene_threshold: float | None = None
    maximum_gap_seconds: float | None = None
    include_frame_zero: bool = True
    frame_sync: str = "vfr"

    def __post_init__(self) -> None:
        if self.strategy not in {"fixed", "scene", "hybrid"}:
            raise ValueError(f"Unknown video keyframe strategy: {self.strategy}")
        if not self.include_frame_zero:
            raise ValueError("Video keyframe policy must include frame zero")
        if self.frame_sync != "vfr":
            raise ValueError("Video keyframe policy must use VFR output")
        if self.strategy == "fixed" and not _positive(self.interval_seconds):
            raise ValueError("Fixed video keyframes require a positive interval")
        if self.strategy in {"scene", "hybrid"} and not _probability(
            self.scene_threshold
        ):
            raise ValueError("Scene video keyframes require a threshold in [0, 1]")
        if self.strategy == "hybrid" and not _positive(self.maximum_gap_seconds):
            raise ValueError("Hybrid video keyframes require a positive maximum gap")

    @property
    def ffmpeg_filter(self) -> str:
        first = "eq(n,0)"
        if self.strategy == "fixed":
            return (
                f"select='{first}+gte(t-prev_selected_t,"
                f"{_format(self.interval_seconds)})'"
            )
        scene = _format(self.scene_threshold, decimals=2)
        if self.strategy == "scene":
            return f"select='{first}+gt(scene,{scene})'"
        return (
            f"select='{first}+gt(scene,{scene})+isnan(prev_selected_t)+"
            f"gte(t-prev_selected_t,{_format(self.maximum_gap_seconds)})'"
        )


def frame_command(policy: KeyframePolicy, source: str, pattern: str) -> list[str]:
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "info",
        "-y",
        "-i",
        source,
        "-vf",
        f"{policy.ffmpeg_filter},showinfo",
        "-fps_mode",
        policy.frame_sync,
        pattern,
    ]


def _positive(value: float | None) -> bool:
    return value is not None and value > 0


def _probability(value: float | None) -> bool:
    return value is not None and 0 <= value <= 1


def _format(value: float | None, *, decimals: int | None = None) -> str:
    assert value is not None
    return f"{value:.{decimals}f}" if decimals is not None else f"{value:g}"
