"""Question-weighted estimators over complete linked-source draws."""

from collections import defaultdict
from statistics import mean

from benchmarks.common.metrics import cluster_bootstrap

SOURCE_KEYS = ("source_document_ids", "independent_source_id")
SOURCE_SCOPES = ("single_source", "multi_source")
QUALITY_SCOPES = ("overall", "text", "table", "formula", "mixed", *SOURCE_SCOPES)


def source_scope(metadata):
    if not metadata.get("answerable", True):
        return None
    count = len(metadata["source_document_ids"])
    return "multi_source" if count > 1 else "single_source" if count == 1 else None


def in_quality_scope(metadata, scope):
    if scope == "overall":
        return True
    if scope in SOURCE_SCOPES:
        return source_scope(metadata) == scope
    return metadata["evidence_type"] == scope


def question_documents(metadata):
    return {document for row in metadata for document in row["source_document_ids"]}


def source_ids(metadata):
    return {row["independent_source_id"] for row in metadata} - {None}


def source_groups(rows, metadata=lambda row: row.metadata):
    grouped = defaultdict(list)
    for row in rows:
        grouped[metadata(row)["independent_source_id"]].append(row)
    return [grouped[key] for key in sorted(grouped, key=lambda key: (key is None, key))]


def question_mean(values):
    known = [value for value in values if value is not None]
    return float(mean(known)) if known else None


def question_macro(groups, metric):
    return question_mean(row.metrics.get(metric) for group in groups for row in group)


def require_aligned_questions(left, right):
    for key in (*SOURCE_KEYS, "evidence_type"):
        if left.metadata.get(key) != right.metadata.get(key):
            raise ValueError(f"Paired question {left.sample_id} has inconsistent {key}")


def cluster_interval(
    groups,
    statistic,
    *,
    resamples,
    seed,
    confidence,
    minimum_sources,
    group_known=lambda group: True,
):
    """Keep every eligible group (including failed work); never deduplicate draws."""
    estimate = statistic(groups)
    known = [group for group in groups if group_known(group)]
    contributing = sum(statistic([group]) is not None for group in known)
    reason = (
        "no_contributing_observations"
        if estimate is None
        else "smoke"
        if not resamples
        else "unknown_independent_source_grouping"
        if len(known) != len(groups)
        else "ci_support_not_frozen"
        if minimum_sources is None
        else "insufficient_independent_sources"
        if contributing < max(2, minimum_sources)
        else None
    )
    result = {
        "estimate": estimate,
        "lower": None,
        "upper": None,
        "confidence": confidence,
        "independent_source_count": len(known),
        "contributing_source_count": contributing,
        "requested_resamples": resamples,
        "defined_resamples": 0,
        "undefined_resamples": 0,
        "resampling_performed": False,
        "degenerate": False,
        "reason": reason,
    }
    if reason:
        return result
    result.update(
        cluster_bootstrap(
            groups,
            statistic,
            resamples=resamples,
            seed=seed,
            confidence=confidence,
        )
    )
    return result


def source_interval(
    groups,
    statistic,
    *,
    resamples,
    seed,
    confidence,
    minimum_sources,
    metadata=lambda row: row.metadata,
):
    rows = [row for group in groups for row in group]
    contributing = [row for row in rows if statistic([[row]]) is not None]
    # For paired conditional percentiles, the two sides can contribute via
    # different questions. Their source contribution is measured by the statistic.
    return {
        **cluster_interval(
            groups,
            statistic,
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            minimum_sources=minimum_sources,
            group_known=lambda group: (
                metadata(group[0])["independent_source_id"] is not None
            ),
        ),
        "independent_unit": "reviewed_source_group",
        "eligible_document_count": len(question_documents(map(metadata, rows))),
        "contributing_document_count": len(
            question_documents(map(metadata, contributing))
        ),
        "eligible_question_count": len(rows),
        "contributing_question_count": len(contributing),
    }
