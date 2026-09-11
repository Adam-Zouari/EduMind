"""Exact metric contract for visual video candidates."""

from __future__ import annotations

from experiments.benchmarks.extraction.video.metrics import QUALITY_DIRECTIONS

OPERATIONAL_DIRECTIONS = {
    "visual_real_time_factor": "min",
    "p50_warm_visual_latency_seconds": "min",
    "p95_warm_visual_latency_seconds": "min",
    "cold_visual_pipeline_load_seconds": "min",
    "peak_visual_process_tree_ram_mb": "min",
    "peak_visual_vram_mb": "min",
    "mean_selected_frames_per_video": "min",
}


def directions() -> dict[str, str]:
    return {**QUALITY_DIRECTIONS, **OPERATIONAL_DIRECTIONS}
