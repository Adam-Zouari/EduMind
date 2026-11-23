"""Condition-local estimators and reviewed-unit bootstrap, never generic means."""

from __future__ import annotations

import math
from collections import defaultdict
from itertools import combinations

import numpy as np

from benchmarks.common.metrics import cluster_bootstrap

from .workload import ALL_SESSION, SELECTED_FILES


def mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def percentile(values, q):
    known = [value for value in values if value is not None]
    return float(np.quantile(known, q)) if known else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def repeatability(attempts, scheduled):
    if scheduled == 1:
        return {"value": None, "status": "inapplicable", "pairs": []}
    by_attempt = {row["attempt"]: row for row in attempts}
    complete = (
        len(attempts) == scheduled
        and set(by_attempt) == set(range(1, scheduled + 1))
        and all(row.get("status") in {"success", "failed"} for row in attempts)
    )
    pairs = []
    for left, right in combinations(range(1, scheduled + 1), 2):
        a, b = by_attempt.get(left), by_attempt.get(right)
        pair_known = all(
            row and row["status"] in {"success", "failed"} for row in (a, b)
        )
        valid = pair_known and all(
            row["status"] == "success"
            and row.get("ids") is not None
            and not row.get("filter_violation")
            and not row.get("session_isolation_violation")
            for row in (a, b)
        )
        pairs.append(
            {
                "left": left,
                "right": right,
                "value": int(valid and a["ids"] == b["ids"]) if pair_known else None,
            }
        )
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
    result.update(
        cluster_bootstrap(
            groups,
            statistic,
            resamples=result["requested_resamples"],
            seed=protocol.meta.seed,
            confidence=protocol.confidence_level,
        )
    )
    return result


SESSION_UNIT = "reviewed_session_group"
SEARCH_LATENCY = "operational.database_search_latency_ms"
SEARCH_ERROR = "diagnostic.request_error_rate.search"
FILTER_VIOLATION = "diagnostic.filter_violation_rate"
REPEATABILITY = "diagnostic.ranking_repeatability"
ANN_KEYS = {
    ALL_SESSION: "quality.ann_recall_at_20",
    SELECTED_FILES: "quality.filtered_ann_recall_at_20",
}


def usable_duration(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def search_counts(rows):
    completed = [row for row in rows if row["status"] == "success"]
    timed = sum(usable_duration(row["latency_seconds"]) for row in completed)
    failed = sum(row["status"] == "failed" for row in rows)
    return {
        "scheduled": len(rows),
        "submitted": len(completed) + failed,
        "completed": len(completed),
        "timed": timed,
        "failed": failed,
        "not_executed": len(rows) - len(completed) - failed,
        "missing_timing": len(completed) - timed,
        "checkable": sum(
            type(row.get("filter_violation")) is bool for row in completed
        ),
    }


def session_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["session_group_id"]].append(row)
    return [grouped[key] for key in sorted(grouped)]


def session_macro(draw, metric):
    """Each repeated draw copy retains each of its workspace means once."""
    session_values = []
    for group in draw:
        sessions = defaultdict(list)
        for row in group:
            sessions[row["session_id"]].append(row["metrics"].get(metric))
        session_values.extend(mean(values) for values in sessions.values())
    return mean(session_values)


def request_statistic(requests, metric):
    successful = [row for row in requests if row["status"] == "success"]
    if metric.startswith("diagnostic.request_error_rate."):
        return ratio(
            sum(row["status"] == "failed" for row in requests),
            sum(row["status"] != "not_executed" for row in requests),
        )
    if metric == FILTER_VIOLATION:
        return mean(
            int(row["filter_violation"])
            for row in successful
            if type(row.get("filter_violation")) is bool
        )
    return percentile(
        [
            row["latency_seconds"] * 1000
            for row in successful
            if usable_duration(row["latency_seconds"])
        ],
        0.95 if metric.endswith("p95") else 0.5,
    )


def aggregate_session_scope(query_rows, requests, scope, protocol, profile):
    metrics, intervals, statuses, per_session = {}, {}, {}, []
    for session_id in sorted({row["session_id"] for row in query_rows}):
        cohort = [row for row in query_rows if row["session_id"] == session_id]
        per_session.append(
            {
                "session_id": session_id,
                "session_group_id": cohort[0]["session_group_id"],
                "condition": scope,
                "eligible_questions": len(cohort),
                **{
                    name: mean(row["metrics"].get(name) for row in cohort)
                    for name in (ANN_KEYS[scope], REPEATABILITY)
                },
                "ann_contributing_questions": sum(
                    row["metrics"].get(ANN_KEYS[scope]) is not None for row in cohort
                ),
                "ann_unavailable_questions": sum(
                    row["metric_statuses"].get(ANN_KEYS[scope]) == "unavailable"
                    for row in cohort
                ),
                "repeatability_contributing_questions": sum(
                    row["metrics"].get(REPEATABILITY) is not None for row in cohort
                ),
            }
        )
    for name in (ANN_KEYS[scope], REPEATABILITY):
        key = name + ".condition." + scope
        states = [row["metric_statuses"][name] for row in query_rows]
        eligible = [
            row for row in query_rows if row["metric_statuses"][name] != "inapplicable"
        ]
        scored = [row for row in eligible if row["metrics"][name] is not None]
        unavailable = any(state in {"unavailable", "incomplete"} for state in states)
        counts = {
            "eligible_questions": len(eligible),
            "contributing_questions": len(scored),
            "unavailable_questions": len(eligible) - len(scored),
            "failed_first_questions": sum(
                row.get("first_attempt_status") == "failed" for row in eligible
            ),
            "eligible_sessions": len({row["session_id"] for row in eligible}),
            "contributing_sessions": len({row["session_id"] for row in scored}),
        }

        def statistic(draw, name=name):
            return session_macro(draw, name)

        bounds = interval(
            session_groups(eligible),
            statistic,
            protocol=protocol,
            profile=profile,
            metric=key,
            unit=SESSION_UNIT,
            blocked_reason="required_quality_coverage_incomplete"
            if unavailable
            else None,
        )
        metrics[key] = bounds["estimate"] if not unavailable else None
        metrics[key + ".sample_count"] = counts["contributing_sessions"]
        metrics[key + ".independent_source_count"] = bounds["contributing_source_count"]
        intervals[key] = bounds
        statuses[key] = {
            **counts,
            "status": "incomplete"
            if unavailable
            else "scored"
            if scored
            else "inapplicable",
            "ci": bounds,
        }
    counts = search_counts(requests)
    request_groups = session_groups(requests)
    for name in (
        SEARCH_ERROR,
        FILTER_VIOLATION,
        SEARCH_LATENCY + "_p50",
        SEARCH_LATENCY + "_p95",
    ):
        key = name + ".condition." + scope
        applicable = bool(requests) and (
            name != FILTER_VIOLATION or scope == SELECTED_FILES
        )
        if not applicable:
            bounds = interval(
                [],
                lambda _: None,
                protocol=protocol,
                profile=profile,
                metric=key,
                unit=SESSION_UNIT,
                blocked_reason="inapplicable",
            )
        else:
            bounds = interval(
                request_groups,
                lambda draw, name=name: request_statistic(
                    [row for group in draw for row in group], name
                ),
                protocol=protocol,
                profile=profile,
                metric=key,
                unit=SESSION_UNIT,
                blocked_reason="required_execution_incomplete"
                if counts["not_executed"]
                else None,
            )
        metrics[key], intervals[key] = bounds["estimate"], bounds
        metrics[key + ".sample_count"] = (
            counts["checkable"]
            if name == FILTER_VIOLATION
            else counts["submitted"]
            if name == SEARCH_ERROR
            else counts["timed"]
        )
        metrics[key + ".independent_source_count"] = bounds["contributing_source_count"]
        statuses[key] = {
            **counts,
            "execution_complete": not counts["not_executed"],
            "status": "inapplicable"
            if not applicable
            else "incomplete"
            if counts["not_executed"]
            else "scored"
            if metrics[key] is not None
            else "unavailable",
            "ci": bounds,
        }
    for label, value in counts.items():
        metrics[f"validity.search.{label}_count.condition.{scope}"] = value
    return metrics, intervals, statuses, per_session


def paired_search_summary(query_rows, requests):
    paired_ids = {
        (row["session_id"], row["query_id"]) for row in query_rows if row["paired"]
    }
    summary = {
        "paired_sessions": len({session for session, _ in paired_ids}),
        "paired_questions": len(paired_ids),
    }
    for condition, prefix in (
        (ALL_SESSION, "unfiltered"),
        (SELECTED_FILES, "filtered"),
    ):
        rows = [
            row for row in query_rows if row["condition"] == condition and row["paired"]
        ]
        scheduled = [
            row for row in requests if row["condition"] == condition and row["paired"]
        ]
        key = ANN_KEYS[condition]
        unavailable = any(row["metric_statuses"][key] == "unavailable" for row in rows)
        summary.update(
            {
                f"{prefix}_{name}": count
                for name, count in search_counts(scheduled).items()
            }
        )
        summary[f"{prefix}_recall_at_20"] = (
            session_macro(session_groups(rows), key) if not unavailable else None
        )
        summary[f"{prefix}_recall_status"] = (
            "inapplicable" if not rows else "incomplete" if unavailable else "scored"
        )
        for p in ("p50", "p95"):
            summary[f"{prefix}_search_latency_ms_{p}"] = request_statistic(
                scheduled, SEARCH_LATENCY + "_" + p
            )
    return summary


def aggregate_operations(requests, operations, protocol, profile):
    from .comparisons import operation_condition

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
