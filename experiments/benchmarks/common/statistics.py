"""Aggregate per-sample metrics and confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .contracts import SampleResult
from .metrics import paired_bootstrap_interval


def aggregate_samples(
    samples: Sequence[SampleResult],
    *,
    resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[dict[str, float | None], dict[str, dict[str, object]]]:
    metric_names = (
        sorted(set().union(*(sample.metrics.keys() for sample in samples)))
        if samples
        else []
    )
    metrics: dict[str, float | None] = {}
    intervals: dict[str, dict[str, object]] = {}
    for name in metric_names:
        observed = [sample.metrics.get(name) for sample in samples]
        values = [value for value in observed if value is not None]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(
                f"Metric {name!r} contains a non-finite numeric observation"
            )
        metrics[name] = sum(values) / len(values) if values else None
        if not resamples:
            continue
        if len(values) < 2:
            intervals[name] = {
                "estimate": metrics[name],
                "lower": None,
                "upper": None,
                "confidence": confidence,
                "sample_count": len(values),
                "reason": "insufficient_independent_support",
            }
            continue
        interval = paired_bootstrap_interval(
            values,
            resamples=resamples,
            seed=seed,
            confidence=confidence,
        )
        intervals[name] = {
            "estimate": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
            "confidence": interval.confidence,
            "sample_count": len(values),
        }
    return metrics, intervals
