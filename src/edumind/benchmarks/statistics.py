"""Aggregation, confidence intervals, gates, and Pareto promotion."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .contracts import SampleResult
from .metrics import paired_bootstrap_interval, pareto_front


def aggregate_samples(
    samples: Sequence[SampleResult], *, resamples: int = 10_000, seed: int = 42
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    metric_names = (
        sorted(set().union(*(sample.metrics.keys() for sample in samples))) if samples else []
    )
    metrics: dict[str, float] = {}
    intervals: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = [sample.metrics[name] for sample in samples if name in sample.metrics]
        interval = paired_bootstrap_interval(values, resamples=resamples, seed=seed)
        metrics[name] = interval.estimate
        intervals[name] = {
            "estimate": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
            "confidence": interval.confidence,
        }
    return metrics, intervals


def select_pareto(
    candidate_metrics: Mapping[str, Mapping[str, float]],
    directions: Mapping[str, str],
) -> list[str]:
    names = list(candidate_metrics)
    rows = [candidate_metrics[name] for name in names]
    return [names[index] for index in pareto_front(rows, directions)]
