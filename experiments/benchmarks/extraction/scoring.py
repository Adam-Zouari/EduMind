"""Shared extraction attempt accounting and source-group uncertainty.

Units and quality formulas remain owned by each extraction benchmark.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations

import numpy as np


def source_id(item: Mapping[str, object]) -> str:
    return str(
        item.get("independent_source_id") or item.get("source_group_id") or item["id"]
    )


def attempt_rates(fingerprints: Sequence[str | None]) -> dict[str, float | None]:
    if not fingerprints:
        raise ValueError("Measured attempts must be recorded")
    pairs = list(combinations(fingerprints, 2))
    return {
        "attempt_failure_rate": sum(value is None for value in fingerprints)
        / len(fingerprints),
        "repeatability_success_rate": (
            sum(left is not None and left == right for left, right in pairs)
            / len(pairs)
            if pairs
            else None
        ),
    }


def bootstrap_sources(
    rows: Sequence[Mapping[str, object]],
    aggregate: Callable[[Sequence[Mapping[str, object]]], Mapping[str, float | None]],
    *,
    resamples: int,
    seed: int,
    confidence: float,
    minimum_sources: int | None = None,
    stratum_field: str | None = None,
) -> dict[str, dict[str, object]]:
    """Resample complete source groups, retaining failures and dependent records."""
    if not resamples:
        return {}
    groups: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        groups.setdefault(str(row["source_group_id"]), []).append(row)
    strata: dict[str, list[list[Mapping[str, object]]]] = {}
    for group in groups.values():
        labels = (
            {str(row[stratum_field]) for row in group} if stratum_field else {"all"}
        )
        if len(labels) != 1:
            raise ValueError(
                "An independent source group cannot span separately resampled strata"
            )
        strata.setdefault(next(iter(labels)), []).append(group)
    stratum_counts = {name: len(units) for name, units in strata.items()}
    estimates = aggregate(rows)
    group_estimates = [aggregate(group) for group in groups.values()]
    support = {
        name: sum(group.get(name) is not None for group in group_estimates)
        for name in estimates
    }
    reasons = {
        name: "no_contributing_sources"
        if value is None
        else "ci_support_not_frozen"
        if minimum_sources is None
        else "insufficient_independent_sources"
        if support[name] < minimum_sources
        else None
        for name, value in estimates.items()
    }
    if not groups or all(reasons.values()):
        return {
            name: {
                "estimate": value,
                "lower": None,
                "upper": None,
                "confidence": confidence,
                "reason": reasons[name],
                "independent_source_count": len(groups),
                "stratum_source_counts": stratum_counts,
                "contributing_source_count": support[name],
                "requested_resamples": resamples,
                "defined_resamples": 0,
                "undefined_resamples": 0,
                "resampling_performed": False,
            }
            for name, value in estimates.items()
        }
    draws: dict[str, list[float]] = {name: [] for name in estimates}
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        sample = [
            row
            for units in strata.values()
            for index in rng.integers(0, len(units), len(units))
            for row in units[index]
        ]
        for name, value in aggregate(sample).items():
            if value is not None:
                draws[name].append(float(value))
    tail = (1 - confidence) / 2
    result = {}
    for name, values in draws.items():
        lower, upper = np.quantile(values, [tail, 1 - tail]) if values else (None, None)
        result[name] = {
            "estimate": estimates[name],
            "lower": float(lower)
            if lower is not None and reasons[name] is None
            else None,
            "upper": float(upper)
            if upper is not None and reasons[name] is None
            else None,
            "confidence": confidence,
            "defined_resamples": len(values),
            "undefined_resamples": resamples - len(values),
            "independent_source_count": len(groups),
            "stratum_source_counts": stratum_counts,
            "contributing_source_count": support[name],
            "requested_resamples": resamples,
            "resampling_performed": True,
            "degenerate": bool(values and lower == upper),
            "reason": reasons[name] or (None if values else "no_defined_resamples"),
        }
    return result
