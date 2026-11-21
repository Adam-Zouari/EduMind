"""Paired operational differences with complete, aligned dependence units."""

from __future__ import annotations

from collections import defaultdict

from edumind.common.artifacts import stable_hash

from .metrics import interval, mean, percentile, ratio


def operation_condition(spec):
    return spec.get("condition_id", spec["id"])


def request_statistic(requests, metric, *, windows=(), filtered=False):
    submitted = [row for row in requests if row["status"] != "not_executed"]
    success = [row for row in submitted if row["status"] == "success"]
    if metric == "operational.qps":
        return ratio(
            sum(row["successful"] for row in windows),
            sum(row["wall_seconds"] for row in windows),
        )
    if metric.startswith("diagnostic.request_error_rate."):
        return ratio(
            sum(row["status"] == "failed" for row in submitted), len(submitted)
        )
    if metric == "diagnostic.filter_violation_rate":
        return (
            mean(
                [
                    int(row["filter_violation"])
                    for row in success
                    if row.get("filter_violation") is not None
                ]
            )
            if filtered
            else None
        )
    return percentile(
        [row["latency_seconds"] * 1000 for row in success],
        0.95 if metric.endswith("p95") else 0.5,
    )


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
    for condition in workloads["conditions"]:
        cid = condition["id"]
        names = [
            "diagnostic.request_error_rate.search",
            "diagnostic.filter_violation_rate",
            "operational.database_search_latency_ms_p50",
            "operational.database_search_latency_ms_p95",
        ]
        if condition["mode"] == "load":
            names.append("operational.qps")
        rows = [
            [
                row
                for row in side["requests"]
                if row["condition"] == cid and row["operation"] == "search"
            ]
            for side in (left, right)
        ]
        key = lambda row: (row["window"], row["query_id"], row["attempt"])
        indexed = [_indexed(side, key) for side in rows]
        expected = {
            (window, query, attempt)
            for window in range(condition["windows"])
            for query in condition["query_ids"]
            for attempt in range(1, protocol.profile(profile).repetitions + 1)
        }
        windows = [
            _indexed(
                [row for row in side.get("windows", []) if row["condition"] == cid],
                lambda row: row["window"],
            )
            for side in (left, right)
        ]
        complete = all(set(side) == expected for side in indexed) and all(
            row["status"] != "not_executed" for side in rows for row in side
        )
        if condition["mode"] == "load":
            complete &= all(
                set(side) == set(range(condition["windows"]))
                and all(row.get("complete") for row in side.values())
                for side in windows
            )
        groups = defaultdict(lambda: [[], []])
        design = condition.get("independent_window_groups")
        for identifier in sorted(expected):
            present = [side.get(identifier) for side in indexed]
            if condition["mode"] == "load":
                group = (
                    design[identifier[0]] if design else "unreviewed-window-dependence"
                )
            else:
                group = next(
                    (row["source_group"] for row in present if row), identifier[1]
                )
                if (
                    all(present)
                    and present[0]["source_group"] != present[1]["source_group"]
                ):
                    raise ValueError("Paired query source groups differ")
            for side, row in enumerate(present):
                if row:
                    groups[group][side].append(row)
        units = []
        for values in groups.values():
            unit_windows = [
                [
                    windows[side][value]
                    for value in sorted({row["window"] for row in values[side]})
                    if value in windows[side]
                ]
                for side in (0, 1)
            ]
            units.append({"requests": values, "windows": unit_windows})
        reason = (
            "load_window_dependence_design_not_frozen"
            if condition["mode"] == "load" and not design
            else None
        )
        result.extend(
            _differences(
                left,
                right,
                cid,
                "search",
                names,
                units,
                complete,
                protocol,
                profile,
                unit="independent_load_window_group"
                if condition["mode"] == "load"
                else "query_source",
                filtered=bool(condition["filters"]),
                reason=reason,
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
            units = [
                {"requests": values, "windows": [[], []]} for values in grouped.values()
            ]
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
    filtered=False,
    reason=None,
):
    results = []
    for metric in names:

        def values(draw, metric=metric):
            return [
                request_statistic(
                    [row for group in draw for row in group["requests"][side]],
                    metric,
                    windows=[row for group in draw for row in group["windows"][side]],
                    filtered=filtered,
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
        rows = [
            [row for group in units for row in group["requests"][side]]
            for side in (0, 1)
        ]
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
                else "inapplicable"
                if metric.endswith("filter_violation_rate") and not filtered
                else "unavailable",
                "alignment_checksum": stable_hash(units),
                **{
                    f"{label}_{kind}_count": sum(predicate(row) for row in rows[side])
                    for side, label in enumerate(("left", "right"))
                    for kind, predicate in (
                        ("submitted", lambda row: row["status"] != "not_executed"),
                        ("successful", lambda row: row["status"] == "success"),
                        ("failed", lambda row: row["status"] == "failed"),
                    )
                },
                **bounds,
            }
        )
    return results


def paired_quality(left, right, protocol, profile):
    """Aligned source draws for query means and document-macro evidence quality."""
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
        grouped = defaultdict(list)
        complete = set(records[0]) == set(records[1])
        for identifier in sorted(set(records[0]) | set(records[1])):
            pair = [side.get(identifier) for side in records]
            if not all(pair):
                complete = False
                continue
            groups = [sample.metadata.get("source_group") for sample in pair]
            if not all(groups) or groups[0] != groups[1]:
                raise ValueError("Paired quality independent-source identity mismatch")
            grouped[groups[0]].append(pair)
        units = list(grouped.values())
        document_macro = "evidence_" in base or "ndcg_" in base

        def values(draw, *, key=key, document_macro=document_macro):
            values = []
            for side in (0, 1):
                if document_macro:
                    documents = []
                    for group in draw:
                        by_document = defaultdict(list)
                        for pair in group:
                            sample = pair[side]
                            by_document[sample.metadata["document_id"]].append(
                                sample.metrics[key]
                            )
                        documents.extend(
                            mean(scores) for scores in by_document.values()
                        )
                    values.append(mean(documents))
                else:
                    values.append(
                        mean(
                            [
                                pair[side].metrics[key]
                                for group in draw
                                for pair in group
                            ]
                        )
                    )
            return values

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
            else None
        )
        bounds = interval(
            units,
            delta,
            protocol=protocol,
            profile=profile,
            metric=key,
            unit="reviewed_query_source",
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
                if blocked
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
