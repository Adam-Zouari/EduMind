"""Deterministic scoring of saved responses, outcomes and semantic labels."""

import json
import math
import random
from collections import Counter, defaultdict
from itertools import combinations
from statistics import mean, median

from .judge import validate_pair_labels, validate_response_labels

REFUSAL = "The provided evidence is insufficient to answer the question."
QUALITY = (
    "faithfulness",
    "factual_correctness_precision",
    "factual_correctness_recall",
    "factual_correctness_f1",
    "answer_relevancy",
    "citation_precision",
    "citation_recall",
    "citation_f1",
)
PRIMARY_METRICS = tuple(
    "quality." + name
    for name in (
        "faithfulness",
        "factual_correctness_f1",
        "answer_relevancy",
        "citation_f1",
    )
)
RATE_KEYS = (
    "validity.response_validity_rate",
    "validity.refusal_validity_rate",
    "reliability.runtime_error_rate",
    "reliability.timeout_rate",
    "reliability.context_limit_reached_rate",
)
REPEAT_KEYS = tuple(
    "repeatability." + name + "_agreement"
    for name in ("status", "citation", "semantic")
)
WORKLOAD = (
    "prompt_tokens",
    "context_tokens",
    "reasoning_tokens",
    "visible_answer_tokens",
    "total_output_tokens",
)
BASE_KEYS = (
    tuple("quality." + name for name in QUALITY)
    + RATE_KEYS
    + ("validity.malformed_output_rate",)
    + REPEAT_KEYS
)
OPERATIONAL_KEYS = tuple(
    f"operational.{name}_p{p}_seconds" for name in ("ttft", "latency") for p in (50, 95)
) + ("operational.decode_tokens_per_second",)
GENERATION_DIRECTIONS = {
    **{key: "max" for key in BASE_KEYS},
    **{
        key: "min"
        for key in BASE_KEYS
        if key.startswith("reliability.") or key == "validity.malformed_output_rate"
    },
    **{key: "min" for key in OPERATIONAL_KEYS},
    "operational.decode_tokens_per_second": "max",
    **{
        f"operational.{key}": "min"
        for key in ("cold_model_load_seconds", "peak_ram_mib", "peak_vram_mib")
    },
}


def cell(value=None, status="unavailable", reason=None):
    return {
        "value": value,
        "status": "scored" if value is not None else status,
        "reason": reason,
    }


def f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def parse_response(text):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    try:
        value = json.loads(
            text, object_pairs_hook=unique, parse_constant=lambda _: None
        )
        if not isinstance(value, dict) or set(value) != {
            "status",
            "answer",
            "citations",
        }:
            raise ValueError("Response must contain exactly status, answer, citations")
        if value["status"] not in (
            "answered",
            "insufficient_evidence",
        ) or not isinstance(value["answer"], str):
            raise ValueError("Invalid status/answer type")
        if not isinstance(value["citations"], list) or any(
            not isinstance(item, str) for item in value["citations"]
        ):
            raise ValueError("Citations must be an ordered string array")
    except (ValueError, TypeError) as exc:
        return None, str(exc)
    return value, None


def score_attempt(question, attempt, judgment=None):
    """All semantic values depend on usable labels, never lexical substitutes."""
    outcome = attempt["outcome"]
    if outcome not in {
        "completed",
        "runtime_error",
        "timeout",
        "context_limit",
        "not_executed",
        "missing_outcome",
    }:
        raise ValueError(f"Unknown generation outcome: {outcome}")
    response, schema_error = (
        parse_response(attempt.get("visible_text", ""))
        if outcome == "completed"
        else (None, None)
    )
    executed = outcome not in {"not_executed", "missing_outcome"}
    missing = cell(status="incomplete", reason=outcome)
    unavailable = cell(
        reason="output_unavailable" if outcome != "completed" else "malformed_output"
    )
    metrics = {
        "quality." + name: cell(status="inapplicable", reason="unanswerable_question")
        if not question["answerable"]
        else dict(unavailable)
        for name in QUALITY
    }
    target = (
        "validity.response_validity_rate"
        if question["answerable"]
        else "validity.refusal_validity_rate"
    )
    metrics[RATE_KEYS[0]] = cell(status="inapplicable", reason="unanswerable_question")
    metrics[RATE_KEYS[1]] = cell(status="inapplicable", reason="answerable_question")
    metrics[target] = dict(missing) if not executed else cell(0.0)
    for name, event in (
        ("runtime_error", "runtime_error"),
        ("timeout", "timeout"),
        ("context_limit_reached", "context_limit"),
    ):
        metrics[f"reliability.{name}_rate"] = (
            cell(float(outcome == event)) if executed else dict(missing)
        )
    metrics["validity.malformed_output_rate"] = (
        cell(float(response is None))
        if outcome == "completed"
        else cell(status="inapplicable", reason="no_completed_output")
        if executed
        else dict(missing)
    )
    invalid_ids, counts, labels = [], {}, None
    if response is not None:
        produced = set(response["citations"])
        supplied = {block["id"]: block for block in question["blocks"]}
        invalid_ids = sorted(produced - supplied.keys())
        counts["citation_count"] = len(produced)
        if question["answerable"]:
            correct = {
                cid
                for cid in produced
                if cid in supplied and supplied[cid]["gold_unit_ids"]
            }
            covered = {uid for cid in correct for uid in supplied[cid]["gold_unit_ids"]}
            precision = len(correct) / len(produced) if produced else 0.0
            recall = len(covered & set(question["gold_unit_ids"])) / len(
                question["gold_unit_ids"]
            )
            for key, value in (
                ("precision", precision),
                ("recall", recall),
                ("f1", f1(precision, recall)),
            ):
                metrics["quality.citation_" + key] = cell(value)
        if judgment and judgment.get("status") == "evaluated":
            try:
                labels = validate_response_labels(
                    judgment["labels"], question, response
                )
            except (ValueError, TypeError, KeyError) as exc:
                judgment = {
                    **judgment,
                    "status": "unavailable",
                    "reason": f"invalid_judge_labels: {exc}",
                }
        if question["answerable"]:
            mechanical = (
                response["status"] == "answered"
                and bool(response["answer"].strip())
                and bool(produced)
                and not invalid_ids
            )
            metrics[target] = (
                cell(float(bool(labels["claims"])))
                if mechanical and labels is not None
                else cell(reason="judge_unavailable")
                if mechanical
                else cell(0.0)
            )
            for name in QUALITY[:5]:
                metrics["quality." + name] = cell(reason="judge_unavailable")
            if labels is not None:
                claims = labels["claims"]
                counts["generated_claim_count"] = len(claims)
                matched = {
                    match["id"] for claim in claims for match in claim["gold_matches"]
                }
                recall = len(matched) / len(question["gold_claims"])
                precision = (
                    sum(claim["correct"] for claim in claims) / len(claims)
                    if claims
                    else None
                )
                metrics["quality.faithfulness"] = (
                    cell(sum(claim["supported"] for claim in claims) / len(claims))
                    if claims
                    else cell(status="inapplicable", reason="no_generated_claims")
                )
                metrics["quality.factual_correctness_precision"] = (
                    cell(precision)
                    if claims
                    else cell(status="inapplicable", reason="no_generated_claims")
                )
                metrics["quality.factual_correctness_recall"] = cell(recall)
                metrics["quality.factual_correctness_f1"] = cell(
                    f1(precision, recall) if claims else 0.0
                )
                metrics["quality.answer_relevancy"] = cell(
                    labels["relevancy"]["level"] / 2
                )
        else:
            metrics[target] = cell(
                float(
                    response
                    == {
                        "status": "insufficient_evidence",
                        "answer": REFUSAL,
                        "citations": [],
                    }
                )
            )
    return {
        **attempt,
        "response": response,
        "schema_valid": response is not None if outcome == "completed" else None,
        "schema_error": schema_error,
        "invalid_citation_ids": invalid_ids,
        "metrics": metrics,
        "counts": counts,
        "judgment": judgment,
    }


def score_pairs(question, attempts, judge=None):
    result = []
    for left, right in combinations(attempts, 2):
        values, judgment = {}, None
        missing = bool(
            {"not_executed", "missing_outcome"} & {left["outcome"], right["outcome"]}
        )
        valid = left["response"] is not None and right["response"] is not None
        for key in REPEAT_KEYS:
            values[key] = (
                cell(status="incomplete", reason="not_executed")
                if missing
                else cell(0.0)
            )
        if valid:
            a, b = left["response"], right["response"]
            same_status = a["status"] == b["status"]
            values[REPEAT_KEYS[0]] = cell(float(same_status))
            union = set(a["citations"]) | set(b["citations"])
            values[REPEAT_KEYS[1]] = cell(
                len(set(a["citations"]) & set(b["citations"])) / len(union)
                if union
                else 1.0
            )
            if same_status:
                if judge is not None:
                    try:
                        judgment = judge.pair(question, a, b)
                        if judgment.get("status") == "evaluated":
                            labels = validate_pair_labels(judgment["labels"])
                            values[REPEAT_KEYS[2]] = cell(float(labels["equivalent"]))
                        else:
                            values[REPEAT_KEYS[2]] = cell(reason="judge_unavailable")
                    except Exception as exc:  # noqa: BLE001 - preserve pair evaluator failures
                        judgment = {
                            "status": "unavailable",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        values[REPEAT_KEYS[2]] = cell(reason="judge_unavailable")
                else:
                    values[REPEAT_KEYS[2]] = cell(reason="judge_unavailable")
        if not question["answerable"]:
            values[REPEAT_KEYS[1]] = cell(
                status="inapplicable", reason="unanswerable_question"
            )
        result.append(
            {
                "seeds": [left["seed"], right["seed"]],
                "metrics": values,
                "judgment": judgment,
            }
        )
    return result


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * p
    left = int(position)
    return values[left] + (values[min(left + 1, len(values) - 1)] - values[left]) * (
        position - left
    )


def _valid_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _measurement_coverage(attempt):
    """Known conditional exclusions differ from missing required instruments."""
    timing, work = attempt.get("timing", {}), attempt.get("workload", {})
    events, count = attempt.get("token_event_seconds"), work.get("total_output_tokens")
    completed = attempt["outcome"] == "completed"
    inventory = (
        isinstance(count, int)
        and not isinstance(count, bool)
        and count >= 0
        and isinstance(events, list)
        and len(events) == count
        and all(_valid_number(value) for value in events)
        and events == sorted(events)
    )
    first_observed = isinstance(events, list) and bool(events)
    ttft = (
        _valid_number(timing.get("ttft_seconds"))
        and _valid_number(timing.get("generation_call_offset_seconds", 0))
        and math.isclose(
            timing["ttft_seconds"],
            events[0] - timing.get("generation_call_offset_seconds", 0),
        )
        if first_observed
        else inventory or not completed
    )
    decode = (
        not completed
        or inventory
        and (
            count < 2
            or (
                _valid_number(timing.get("first_token_seconds"))
                and _valid_number(timing.get("last_token_seconds"))
                and timing["first_token_seconds"] == events[0]
                and timing["last_token_seconds"] == events[-1]
            )
        )
    )
    latency = attempt.get("response") is None or _valid_number(
        timing.get("latency_seconds")
    )
    executed = attempt["outcome"] not in {"not_executed", "missing_outcome"}
    return {
        key: executed
        and (
            decode
            if key.endswith("decode_tokens_per_second")
            else ttft
            if ".ttft_" in key
            else latency
        )
        for key in OPERATIONAL_KEYS
    }


def question_statistics(record):
    attempts, pairs = record["attempts"], record["pairs"]
    stats, support = {}, {}
    for key in BASE_KEYS:
        cells = (
            [pair["metrics"][key] for pair in pairs]
            if key in REPEAT_KEYS
            else [attempt["metrics"][key] for attempt in attempts]
        )
        eligible = [
            c
            for c in cells
            if c["status"] != "inapplicable" or c["reason"] == "no_generated_claims"
        ]
        defined = [c["value"] for c in cells if c["value"] is not None]
        incomplete = any(c["status"] == "incomplete" for c in cells)
        unavailable = any(c["status"] == "unavailable" for c in eligible)
        # Scheduled rates/pairs cannot shrink their denominator to surviving events.
        stats[key] = (
            mean(defined)
            if defined
            and not (key in (*RATE_KEYS, *REPEAT_KEYS) and (incomplete or unavailable))
            else None
        )
        support[key] = {
            "eligible": bool(eligible)
            or key == "validity.malformed_output_rate"
            and bool(attempts),
            "complete": not incomplete and not unavailable,
            "eligible_observations": len(eligible),
            "contributing_observations": len(defined),
            "claim_free": sum(c["reason"] == "no_generated_claims" for c in cells),
            "unavailable": sum(
                c["status"] in {"unavailable", "incomplete"} for c in eligible
            ),
        }
    for key in WORKLOAD:
        values = [a.get("workload", {}).get(key) for a in attempts]
        stats["workload." + key] = (
            mean([v for v in values if _valid_number(v)])
            if any(_valid_number(v) for v in values)
            else None
        )
    ttfts = [a.get("timing", {}).get("ttft_seconds") for a in attempts]
    latencies = [
        a.get("timing", {}).get("latency_seconds")
        for a in attempts
        if a["response"] is not None
    ]
    stats["ttft"] = (
        median([v for v in ttfts if _valid_number(v)])
        if any(_valid_number(v) for v in ttfts)
        else None
    )
    stats["latency"] = (
        median([v for v in latencies if _valid_number(v)])
        if any(_valid_number(v) for v in latencies)
        else None
    )
    streams = []
    for a in attempts:
        n = a.get("workload", {}).get("total_output_tokens")
        timing = a.get("timing", {})
        first, last = (
            timing.get("first_token_seconds"),
            timing.get("last_token_seconds"),
        )
        if (
            a["outcome"] == "completed"
            and _valid_number(n)
            and n >= 2
            and _valid_number(first)
            and _valid_number(last)
            and last > first
        ):
            streams.append((n - 1, last - first))
    stats["decode_work"] = sum(pair[0] for pair in streams)
    stats["decode_time"] = sum(pair[1] for pair in streams)
    coverage = [_measurement_coverage(attempt) for attempt in attempts]
    for key in OPERATIONAL_KEYS:
        support[key] = {"complete": all(row[key] for row in coverage)}
    return stats, support


def _estimate(key, records):
    if key in OPERATIONAL_KEYS:
        if key == "operational.decode_tokens_per_second":
            duration = sum(r["statistics"]["decode_time"] for r in records)
            return (
                sum(r["statistics"]["decode_work"] for r in records) / duration
                if duration
                else None
            )
        kind = "ttft" if ".ttft_" in key else "latency"
        return percentile(
            [
                r["statistics"][kind]
                for r in records
                if r["statistics"][kind] is not None
            ],
            0.95 if "p95" in key else 0.5,
        )
    by_document = defaultdict(list)
    for record in records:
        value = record["statistics"].get(key)
        if value is not None:
            # A bootstrap copy remains a separate document, never deduplicated by ID.
            by_document[
                (record.get("draw_copy", 0), record["question"]["document_id"])
            ].append(value)
    return (
        mean([mean(values) for values in by_document.values()]) if by_document else None
    )


def aggregate(
    records,
    *,
    profile,
    resamples=10000,
    seed=42,
    confidence=0.95,
    minimum_ci_sources=None,
):
    """Recompute document macros, conditional percentiles and pooled work/time.

    Bootstrap copies carry all questions, attempts, failures and pair labels.
    Numeric support requirements are supplied by reviewed protocol inputs.
    """
    minimum_ci_sources = minimum_ci_sources or {}
    records = [
        {
            **r,
            "statistics": question_statistics(r)[0],
            "support": question_statistics(r)[1],
        }
        for r in records
    ]
    results = {}
    slices = {
        "": records,
        **{
            f"slice.{name}.": [
                r for r in records if r["question"]["answerable"] == answerable
            ]
            for name, answerable in (("answerable", True), ("unanswerable", False))
        },
        **{
            f"slice.{name}.": [
                r for r in records if r["question"]["evidence_type"] == name
            ]
            for name in ("text", "table", "formula", "mixed")
        },
    }
    for prefix, subset in slices.items():
        for key in (
            *BASE_KEYS,
            *OPERATIONAL_KEYS,
            *("workload." + name for name in WORKLOAD),
        ):
            name = prefix + key
            if key in BASE_KEYS:
                eligible = [r for r in subset if r["support"][key]["eligible"]]
                complete = all(r["support"][key]["complete"] for r in eligible)
            else:
                eligible = subset
                complete = all(
                    a["outcome"] not in {"not_executed", "missing_outcome"}
                    for r in eligible
                    for a in r["attempts"]
                )
                if key in OPERATIONAL_KEYS:
                    complete = complete and all(
                        r["support"][key]["complete"] for r in eligible
                    )
            value = _estimate(key, subset)
            contributing = [r for r in eligible if _estimate(key, [r]) is not None]
            sources = {r["question"]["source_id"] for r in eligible}
            contributing_sources = {r["question"]["source_id"] for r in contributing}
            claim_free = (
                sum(r["support"][key]["claim_free"] for r in eligible)
                if key in BASE_KEYS
                else 0
            )
            reason = (
                None
                if value is not None
                else "no_generated_claims"
                if claim_free and complete
                else "no_eligible_task"
                if not eligible
                else "no_contributing_observations"
            )
            if key == "validity.malformed_output_rate" and eligible and value is None:
                reason = "no_completed_outputs"
            result = {
                **cell(
                    value,
                    "inapplicable"
                    if not eligible or (claim_free and complete)
                    else "unavailable",
                    reason,
                ),
                "evidence_complete": complete,
                "partial": not complete,
                "eligible_questions": len(eligible),
                "contributing_questions": len(contributing),
                "eligible_sources": len(sources),
                "contributing_sources": len(contributing_sources),
                "scheduled_attempts": sum(len(r["attempts"]) for r in eligible),
                "executed_attempts": sum(
                    a["outcome"] not in {"not_executed", "missing_outcome"}
                    for r in eligible
                    for a in r["attempts"]
                ),
                "failed_attempts": sum(
                    a["outcome"] not in {"completed", "not_executed", "missing_outcome"}
                    for r in eligible
                    for a in r["attempts"]
                ),
                "evaluated_observations": sum(
                    r["support"][key]["contributing_observations"]
                    + r["support"][key]["claim_free"]
                    for r in eligible
                )
                if key in BASE_KEYS
                else len(contributing),
                "unavailable_observations": sum(
                    r["support"][key]["unavailable"] for r in eligible
                )
                if key in BASE_KEYS
                else None,
                "claim_free_observations": claim_free,
            }
            ci = {
                "estimate": value,
                "lower": None,
                "upper": None,
                "confidence": confidence,
                "defined_resamples": 0,
                "undefined_resamples": 0,
                "requested_resamples": resamples,
                "eligible_sources": len(sources),
                "contributing_sources": len(contributing_sources),
                "degenerate": False,
            }
            minimum = minimum_ci_sources.get(name)
            ci["reason"] = (
                "descriptor_no_interval"
                if key.startswith("workload.")
                else "smoke_no_interval"
                if profile == "smoke"
                else "point_undefined"
                if value is None
                else "incomplete_evidence"
                if not complete
                else "ci_support_not_frozen"
                if minimum is None
                else "insufficient_independent_sources"
                if min(len(sources), len(contributing_sources)) < minimum
                else "no_bootstrap_resampling"
                if not resamples
                else None
            )
            if ci["reason"] is None:
                groups = {
                    source: [
                        r for r in eligible if r["question"]["source_id"] == source
                    ]
                    for source in sorted(sources)
                }
                rng, draws = random.Random(seed), []
                inventory = list(groups)
                for _ in range(resamples):
                    drawn = [
                        {**r, "draw_copy": copy}
                        for copy in range(len(inventory))
                        for r in groups[rng.choice(inventory)]
                    ]
                    estimate = _estimate(key, drawn)
                    if estimate is not None:
                        draws.append(estimate)
                ci.update(
                    defined_resamples=len(draws),
                    undefined_resamples=resamples - len(draws),
                    lower=percentile(draws, (1 - confidence) / 2),
                    upper=percentile(draws, (1 + confidence) / 2),
                )
                ci["degenerate"] = bool(draws and ci["lower"] == ci["upper"])
                if not draws:
                    ci["reason"] = "no_defined_bootstrap_draws"
            result["interval"] = ci
            results[name] = result
    return results


def finish_inventory(records):
    reasons = Counter(
        a["finish_reason"]
        for r in records
        for a in r["attempts"]
        if a.get("finish_reason") is not None
    )
    total = sum(reasons.values())
    return {
        "scheduled": sum(len(r["attempts"]) for r in records),
        "observed": total,
        "reasons": {
            key: {"count": count, "share": count / total}
            for key, count in reasons.items()
        },
    }
