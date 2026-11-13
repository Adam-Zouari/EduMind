"""Diagnostic sample means; authoritative estimators belong to their benchmark."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .contracts import SampleResult


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
        # This legacy tuple boundary does not declare source grouping, estimator,
        # or reviewed support. A row count must never fabricate those contracts.
        intervals[name] = {
            "estimate": metrics[name],
            "lower": None,
            "upper": None,
            "confidence": confidence,
            "sample_count": len(values),
            "eligible_count": len(observed),
            "unavailable_count": len(observed) - len(values),
            "requested_resamples": resamples,
            "defined_resamples": 0,
            "undefined_resamples": 0,
            "resampling_performed": False,
            "degenerate": False,
            "reason": "benchmark_estimator_required",
        }
    return metrics, intervals
