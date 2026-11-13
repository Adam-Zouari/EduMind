"""Condition-local estimators and reviewed-unit bootstrap, never generic means."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def percentile(values, q):
    return float(np.quantile(values, q)) if values else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def repeatability(attempts, scheduled):
    if scheduled == 1:
        return {"value": None, "status": "inapplicable", "pairs": []}
    complete = len(attempts) == scheduled and all(
        row.get("status") != "not_executed" for row in attempts
    )
    pairs = [
        {
            "left": left + 1,
            "right": right + 1,
            "value": int(
                attempts[left]["status"] == attempts[right]["status"] == "success"
                and not attempts[left].get("filter_violation")
                and not attempts[right].get("filter_violation")
                and attempts[left].get("ids") == attempts[right].get("ids")
            )
            if complete
            else None,
        }
        for left, right in combinations(range(scheduled), 2)
    ]
    return {
        "value": mean([row["value"] for row in pairs]) if complete else None,
        "status": "scored" if complete else "incomplete",
        "pairs": pairs,
    }


def interval(
    groups, statistic, *, protocol, profile, metric, unit, blocked_reason=None
):
    estimate = statistic(groups)
    contributing = sum(statistic([group]) is not None for group in groups)
    policy_key = (
        metric if metric in protocol.ci_support else metric.split(".condition.", 1)[0]
    )
    policy = protocol.ci_support.get(policy_key)
    reason = blocked_reason or (
        "smoke"
        if profile == "smoke"
        else "ci_support_not_frozen"
        if not policy
        else "insufficient_independent_sources"
        if contributing < policy["minimum_groups"]
        else "no_contributing_observations"
        if estimate is None
        else None
    )
    result = {
        "estimate": estimate,
        "lower": None,
        "upper": None,
        "independent_unit": unit,
        "independent_source_count": len(groups),
        "contributing_source_count": contributing,
        "reason": reason,
        "defined_resamples": 0,
        "undefined_resamples": 0,
        "degenerate": False,
        "requested_resamples": protocol.profile(profile).bootstrap_resamples,
        "resampling_performed": False,
        "support_policy_key": policy_key if policy else None,
    }
    if reason:
        return result
    rng = np.random.default_rng(protocol.meta.seed)
    result["resampling_performed"] = True
    draws = []
    for _ in range(result["requested_resamples"]):
        value = statistic(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        if value is not None:
            draws.append(value)
    result.update(
        defined_resamples=len(draws),
        undefined_resamples=result["requested_resamples"] - len(draws),
    )
    if draws:
        lower, upper = np.quantile(
            draws,
            [(1 - protocol.confidence_level) / 2, (1 + protocol.confidence_level) / 2],
        )
        result.update(
            lower=float(lower), upper=float(upper), degenerate=bool(lower == upper)
        )
    else:
        result["reason"] = "no_defined_resamples"
    return result


def aggregate_condition(query_rows, requests, windows, condition, protocol, profile):
    """Keep null first outcomes in group inventories and condition identities."""
    cid = condition["id"]
    metrics, intervals, statuses = {}, {}, {}
    groups = defaultdict(list)
    for row in query_rows:
        groups[row["source_group"]].append(row)
    grouped = list(groups.values())
    for name in {key for row in query_rows for key in row["metrics"]}:
        key = name + ".condition." + cid

        def statistic(draw, name=name):
            return mean([row["metrics"].get(name) for group in draw for row in group])

        values = [row["metrics"].get(name) for row in query_rows]
        bounds = interval(
            grouped,
            statistic,
            protocol=protocol,
            profile=profile,
            metric=key,
            unit="query_source",
        )
        metrics[key] = bounds["estimate"]
        metrics[key + ".sample_count"] = sum(value is not None for value in values)
        metrics[key + ".independent_source_count"] = bounds["contributing_source_count"]
        intervals[key] = bounds
        statuses[key] = {
            "status": "incomplete"
            if any(
                row["metric_statuses"].get(name) in {"unavailable", "incomplete"}
                for row in query_rows
            )
            else "scored"
            if any(value is not None for value in values)
            else "inapplicable",
            "scheduled": len(values),
            "contributing": sum(value is not None for value in values),
            "ci": bounds,
        }
    submitted = [row for row in requests if row["status"] != "not_executed"]
    successes = [row for row in submitted if row["status"] == "success"]
    checkable = [row for row in successes if condition["filters"]]
    complete = len(submitted) == len(requests)
    ops_groups = defaultdict(list)
    design = condition.get("independent_window_groups")
    for row in requests:
        ops_groups[
            (design[row["window"]] if design else "unreviewed-window-dependence")
            if condition["mode"] == "load"
            else row["source_group"]
        ].append(row)
    request_groups = list(ops_groups.values())

    def flattened(draw):
        return [row for group in draw for row in group]

    estimators = {
        "diagnostic.request_error_rate.search": lambda draw: ratio(
            sum(row["status"] == "failed" for row in flattened(draw)),
            sum(row["status"] != "not_executed" for row in flattened(draw)),
        ),
        "diagnostic.filter_violation_rate": lambda draw: mean(
            [
                int(row["filter_violation"])
                for row in flattened(draw)
                if row["status"] == "success" and condition["filters"]
            ]
        ),
        **{
            f"operational.database_search_latency_ms_p{int(q * 100)}": lambda draw, q=q: (
                percentile(
                    [
                        row["latency_seconds"] * 1000
                        for row in flattened(draw)
                        if row["status"] == "success"
                    ],
                    q,
                )
            )
            for q in (0.5, 0.95)
        },
    }
    for name, statistic in estimators.items():
        key = name + ".condition." + cid
        bounds = interval(
            request_groups,
            statistic,
            protocol=protocol,
            profile=profile,
            metric=key,
            unit="independent_load_window"
            if condition["mode"] == "load"
            else "query_source",
            blocked_reason="load_window_dependence_design_not_frozen"
            if condition["mode"] == "load" and not design
            else "required_execution_incomplete"
            if not complete
            else None,
        )
        metrics[key], intervals[key] = bounds["estimate"], bounds
        metrics[key + ".sample_count"] = (
            len(checkable)
            if name.endswith("filter_violation_rate")
            else len(submitted)
            if name.endswith("rate.search")
            else len(successes)
        )
        metrics[key + ".independent_source_count"] = bounds["contributing_source_count"]
        statuses[key] = {
            "status": "scored"
            if metrics[key] is not None
            else "inapplicable"
            if name.endswith("filter_violation_rate") and not condition["filters"]
            else "unavailable",
            "submitted": len(submitted),
            "successful": len(successes),
            "checkable": len(checkable),
            "execution_complete": complete,
            "ci": bounds,
        }
    if windows and condition["mode"] == "load":
        name = "operational.qps"
        key = name + ".condition." + cid
        window_groups = defaultdict(list)
        for row in windows:
            window_groups[
                design[row["window"]] if design else "unreviewed-window-dependence"
            ].append(row)
        bounds = interval(
            list(window_groups.values()),
            lambda draw: ratio(
                sum(row["successful"] for group in draw for row in group),
                sum(row["wall_seconds"] for group in draw for row in group),
            ),
            protocol=protocol,
            profile=profile,
            metric=key,
            unit="independent_load_window",
            blocked_reason="load_window_dependence_design_not_frozen"
            if not design
            else "required_execution_incomplete"
            if not complete or len(windows) != condition["windows"]
            else None,
        )
        metrics[key], intervals[key] = bounds["estimate"], bounds
        metrics[key + ".sample_count"] = len(windows)
        metrics[key + ".independent_source_count"] = bounds["contributing_source_count"]
        statuses[key] = {
            "status": "scored" if metrics[key] is not None else "unavailable",
            "ci": bounds,
        }
    for label, value in {
        "submitted_count": len(submitted),
        "successful_count": len(successes),
        "failed_count": len(submitted) - len(successes),
        "not_executed_count": len(requests) - len(submitted),
    }.items():
        metrics[f"validity.search.{label}.condition.{cid}"] = value
    return metrics, intervals, statuses


def aggregate_operations(requests, operations, protocol, profile):
    from .comparisons import operation_condition, request_statistic

    specs = defaultdict(list)
    for spec in operations:
        specs[operation_condition(spec)].append(spec)
    metrics, intervals, statuses, problems = {}, {}, {}, []
    for cid, declared in specs.items():
        for kind in ("insert", "update", "delete"):
            rows = [
                row
                for row in requests
                if row["condition"] == cid and row["operation"] == kind
            ]
            submitted = [row for row in rows if row["status"] != "not_executed"]
            expected = {
                (spec["id"], attempt)
                for spec in declared
                for attempt in range(1, protocol.profile(profile).repetitions + 1)
            }
            actual = [(row.get("operation_id", cid), row["attempt"]) for row in rows]
            if len(set(actual)) != len(actual):
                raise ValueError("Duplicate logical operation ledger identity")
            complete = set(actual) == expected and len(submitted) == len(rows)
            groups = defaultdict(list)
            for row in rows:
                groups[
                    row.get("source_group") or "unreviewed-operation-dependence"
                ].append(row)
            reviewed = all(row.get("source_group") for row in rows)
            for base in (
                f"diagnostic.request_error_rate.{kind}",
                f"operational.{kind}_latency_ms_p50",
                f"operational.{kind}_latency_ms_p95",
            ):
                key = base + ".condition." + cid
                bounds = interval(
                    list(groups.values()),
                    lambda draw, base=base: request_statistic(
                        [row for group in draw for row in group], base
                    ),
                    protocol=protocol,
                    profile=profile,
                    metric=key,
                    unit="document_operation_source",
                    blocked_reason="required_execution_incomplete"
                    if not complete
                    else "operation_source_design_not_frozen"
                    if not reviewed
                    else None,
                )
                metrics[key], intervals[key] = bounds["estimate"], bounds
                metrics[key + ".sample_count"] = (
                    len(submitted)
                    if base.startswith("diagnostic.")
                    else sum(row["status"] == "success" for row in rows)
                )
                metrics[key + ".independent_source_count"] = bounds[
                    "contributing_source_count"
                ]
                statuses[key] = {
                    "status": "incomplete"
                    if not complete
                    else "scored"
                    if metrics[key] is not None
                    else "unavailable",
                    "execution_complete": complete,
                    "submitted": len(submitted),
                    "failed": sum(row["status"] == "failed" for row in submitted),
                    "ci": bounds,
                }
            for label, value in {
                "submitted_count": len(submitted),
                "successful_count": sum(
                    row["status"] == "success" for row in submitted
                ),
                "failed_count": sum(row["status"] == "failed" for row in submitted),
                "not_executed_count": len(expected) - len(submitted),
            }.items():
                metrics[f"validity.{kind}.{label}.condition.{cid}"] = value
            if not complete:
                problems.append(f"Incomplete document operations: {cid}/{kind}")
    return metrics, intervals, statuses, problems
