"""Paired operational differences with complete, aligned dependence units."""

from __future__ import annotations

from collections import defaultdict

from edumind.common.artifacts import stable_hash

from .metrics import (
    FILTER_VIOLATION,
    SEARCH_ERROR,
    SEARCH_LATENCY,
    SESSION_UNIT,
    interval,
    mean,
    request_statistic,
    search_counts,
    session_macro,
)
from .workload import SCOPES, SELECTED_FILES


def operation_condition(spec):
    return spec.get("condition_id", spec["id"])


def _indexed(rows, key):
    result = {}
    for row in rows:
        identifier = key(row)
        if identifier in result:
            raise ValueError(
                f"Paired comparison has duplicate logical observation: {identifier}"
            )
        result[identifier] = row
    return result


def paired_operational(left, right, workloads, protocol, profile):
    """Return left-minus-right differences; never pair only surviving requests."""
    result = []
    for scope in SCOPES:
        names = [SEARCH_ERROR, SEARCH_LATENCY + "_p50", SEARCH_LATENCY + "_p95"]
        if scope == SELECTED_FILES:
            names.append(FILTER_VIOLATION)
        expected = {
            (session["id"], row["query_id"], row["attempt"]): session[
                "session_group_id"
            ]
            for session in workloads["sessions"]
            for row in session["schedule"]
            if row["condition"] == scope
        }
        if not expected:
            continue
        sides = [
            _indexed(
                (
                    row
                    for row in side["requests"]
                    if row["condition"] == scope and row["operation"] == "search"
                ),
                lambda row: (row["session_id"], row["query_id"], row["attempt"]),
            )
            for side in (left, right)
        ]
        complete = all(set(side) == set(expected) for side in sides) and all(
            row["status"] != "not_executed" for side in sides for row in side.values()
        )
        grouped = defaultdict(lambda: [[], []])
        for identifier, group in sorted(expected.items()):
            for ordinal, side in enumerate(sides):
                row = side.get(identifier)
                if row is not None:
                    if row["session_group_id"] != group:
                        raise ValueError(
                            "Paired search session group differs from sealed schedule"
                        )
                    grouped[group][ordinal].append(row)
        result.extend(
            _differences(
                left,
                right,
                scope,
                "search",
                names,
                list(grouped.values()),
                complete,
                protocol,
                profile,
                unit=SESSION_UNIT,
            )
        )
    specs = defaultdict(list)
    for spec in workloads["operations"]:
        specs[operation_condition(spec)].append(spec)
    for cid, operations in specs.items():
        for kind in ("insert", "update", "delete"):
            names = [
                f"diagnostic.request_error_rate.{kind}",
                f"operational.{kind}_latency_ms_p50",
                f"operational.{kind}_latency_ms_p95",
            ]
            rows = [
                [
                    row
                    for row in side["requests"]
                    if row["condition"] == cid and row["operation"] == kind
                ]
                for side in (left, right)
            ]
            indexed = [
                _indexed(
                    side,
                    lambda row, cid=cid: (row.get("operation_id", cid), row["attempt"]),
                )
                for side in rows
            ]
            expected = {
                (spec["id"], attempt)
                for spec in operations
                for attempt in range(1, protocol.profile(profile).repetitions + 1)
            }
            complete = all(set(side) == expected for side in indexed) and all(
                row["status"] != "not_executed" for side in rows for row in side
            )
            grouped = defaultdict(lambda: [[], []])
            reviewed = True
            for identifier in sorted(expected):
                pair = [side.get(identifier) for side in indexed]
                group = next(
                    (
                        row.get("source_group")
                        for row in pair
                        if row and row.get("source_group")
                    ),
                    None,
                )
                reviewed &= group is not None
                if all(pair) and pair[0].get("source_group") != pair[1].get(
                    "source_group"
                ):
                    raise ValueError("Paired operation source groups differ")
                for side, row in enumerate(pair):
                    if row:
                        grouped[group or "unreviewed-operation-dependence"][
                            side
                        ].append(row)
            units = list(grouped.values())
            result.extend(
                _differences(
                    left,
                    right,
                    cid,
                    kind,
                    names,
                    units,
                    complete,
                    protocol,
                    profile,
                    unit="document_operation_source",
                    reason=None if reviewed else "operation_source_design_not_frozen",
                )
            )
    return result


def _differences(
    left,
    right,
    condition,
    operation,
    names,
    units,
    complete,
    protocol,
    profile,
    *,
    unit,
    reason=None,
):
    results = []
    for metric in names:

        def values(draw, metric=metric):
            return [
                request_statistic(
                    [row for group in draw for row in group[side]],
                    metric,
                )
                for side in (0, 1)
            ]

        def delta(draw):
            pair = values(draw)
            return (
                pair[0] - pair[1] if all(value is not None for value in pair) else None
            )

        blocked = "required_execution_incomplete" if not complete else reason
        bounds = interval(
            units,
            delta,
            protocol=protocol,
            profile=profile,
            metric=metric + ".condition." + condition,
            unit=unit,
            blocked_reason=blocked,
        )
        actual = values(units)
        rows = [[row for group in units for row in group[side]] for side in (0, 1)]
        results.append(
            {
                "left": left["candidate"],
                "right": right["candidate"],
                "condition": condition,
                "operation": operation,
                "metric": metric,
                "left_value": actual[0],
                "right_value": actual[1],
                "difference": bounds.pop("estimate"),
                "status": "incomplete"
                if not complete
                else "scored"
                if all(value is not None for value in actual)
                else "unavailable",
                "alignment_checksum": stable_hash(
                    [
                        [
                            [
                                (
                                    row.get("session_id"),
                                    row.get("query_id"),
                                    row.get("operation_id"),
                                    row["attempt"],
                                    row.get("schedule_position"),
                                )
                                for row in group[side]
                            ]
                            for side in (0, 1)
                        ]
                        for group in units
                    ]
                ),
                **{
                    f"{label}_{kind}_count": value
                    for side, label in enumerate(("left", "right"))
                    for kind, value in search_counts(rows[side]).items()
                },
                **bounds,
            }
        )
    return results


def paired_quality(left, right, protocol, profile):
    """Aligned draws retain the weighting declared by each metric."""
    names = {
        name
        for side in (left, right)
        for sample in side.samples
        for name in sample.metrics
    }
    results = []
    for key in sorted(names):
        if ".condition." not in key:
            continue
        base, condition = key.split(".condition.", 1)
        records = [
            {
                sample.sample_id: sample
                for sample in side.samples
                if key in sample.metrics
            }
            for side in (left, right)
        ]
        session_weighted = condition in SCOPES
        group_key = "session_group_id" if session_weighted else "independent_source_id"
        grouped = defaultdict(list)
        complete = set(records[0]) == set(records[1])
        for identifier in sorted(set(records[0]) | set(records[1])):
            pair = [side.get(identifier) for side in records]
            if not all(pair):
                complete = False
                continue
            groups = [sample.metadata.get(group_key) for sample in pair]
            if pair[0].metadata != pair[1].metadata:
                raise ValueError("Paired quality independent-source identity mismatch")
            grouped[groups[0]].append(pair)
        units = list(grouped.values())

        def values(draw, *, key=key, session_weighted=session_weighted):
            if session_weighted:
                return [
                    session_macro(
                        [
                            [
                                {**pair[side].metadata, "metrics": pair[side].metrics}
                                for pair in group
                            ]
                            for group in draw
                        ],
                        key,
                    )
                    for side in (0, 1)
                ]
            return [
                mean(pair[side].metrics[key] for group in draw for pair in group)
                for side in (0, 1)
            ]

        def delta(draw):
            pair = values(draw)
            return (
                pair[0] - pair[1] if all(value is not None for value in pair) else None
            )

        missing = any(
            pair[side].metrics[key] is None
            for group in units
            for pair in group
            for side in (0, 1)
        )
        points = values(units)
        blocked = (
            "required_quality_coverage_incomplete"
            if not complete or missing and any(value is not None for value in points)
            else "unknown_independent_source_grouping"
            if None in grouped
            else None
        )
        bounds = interval(
            units,
            delta,
            protocol=protocol,
            profile=profile,
            metric=key,
            unit=SESSION_UNIT if session_weighted else "reviewed_source_group",
            blocked_reason=blocked,
        )
        results.append(
            {
                "left": left.candidate,
                "right": right.candidate,
                "condition": condition,
                "operation": "query",
                "metric": base,
                "left_value": points[0],
                "right_value": points[1],
                "difference": bounds.pop("estimate"),
                "status": "incomplete"
                if blocked == "required_quality_coverage_incomplete"
                else "scored"
                if all(value is not None for value in points)
                else "inapplicable",
                "alignment_checksum": stable_hash(
                    sorted(set(records[0]) | set(records[1]))
                ),
                **bounds,
            }
        )
    return results
