"""Blinded human-review export/import with strict rubric validation."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import pandas as pd

from edumind.common.artifacts import atomic_write_json, stable_hash

RUBRIC_FIELDS = (
    "human_faithfulness",
    "human_answer_correctness",
    "human_completeness",
    "human_citation_accuracy",
    "answerability_correct",
)


def export_review(
    summary_path: Path,
    output_path: Path,
    *,
    finalist_count: int = 3,
    question_count: int = 20,
    seed: int = 42,
) -> Path:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    successful = [
        candidate
        for candidate in payload.get("candidates", [])
        if candidate.get("status") == "success"
    ]
    pareto = list(payload.get("pareto_candidates", []))
    ordered = sorted(
        successful,
        key=lambda item: (
            0 if item.get("candidate") in pareto else 1,
            -float(item.get("metrics", {}).get("citation_f1", 0)),
            float(item.get("operational", {}).get("p95_latency_seconds", float("inf"))),
        ),
    )[:finalist_count]
    if len(ordered) != finalist_count:
        raise ValueError(f"Human review requires exactly {finalist_count} successful finalists")
    for candidate in ordered:
        if not candidate.get("samples"):
            candidate["samples"] = _load_samples(summary_path.parent, str(candidate["candidate"]))
    selected_ids = _stratified_ids(ordered[0].get("samples", []), question_count, seed)
    if len(selected_ids) != question_count:
        raise ValueError(f"Human review requires {question_count} distinct questions")
    rows: list[dict[str, object]] = []
    identity_map: dict[str, dict[str, str]] = {}
    for candidate in ordered:
        samples = {str(sample["sample_id"]): sample for sample in candidate.get("samples", [])}
        for sample_id in selected_ids:
            sample = samples[sample_id]
            metadata = sample.get("metadata", {})
            item_id = stable_hash(
                {
                    "run": payload.get("run_id"),
                    "candidate": candidate["candidate"],
                    "sample": sample_id,
                }
            )[:16]
            identity_map[item_id] = {
                "candidate": str(candidate["candidate"]),
                "sample_id": sample_id,
            }
            rows.append(
                {
                    "item_id": item_id,
                    "question": metadata.get("question", ""),
                    "answer": metadata.get("generated_answer", ""),
                    "reference_answer": metadata.get("reference_answer", ""),
                    "evidence": metadata.get("frozen_context", ""),
                    **{field: "" for field in RUBRIC_FIELDS},
                    "notes": "",
                }
            )
    random.Random(seed).shuffle(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_write_json(
        output_path.with_suffix(".identity.json"),
        {
            "run_id": payload.get("run_id"),
            "items": identity_map,
            "expected_judgments": finalist_count * question_count,
            "finalist_count": finalist_count,
            "question_count": question_count,
        },
    )
    return output_path


def import_review(review_path: Path, identity_path: Path | None = None) -> dict[str, object]:
    identity_file = identity_path or review_path.with_suffix(".identity.json")
    identity = json.loads(identity_file.read_text(encoding="utf-8"))
    known = identity.get("items", {})
    with review_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = int(identity.get("expected_judgments", 60))
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} blinded judgments, received {len(rows)}")
    if len({row.get("item_id") for row in rows}) != len(rows):
        raise ValueError("Human review contains duplicate item IDs")
    aggregates: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        item_id = str(row.get("item_id", ""))
        if item_id not in known:
            raise ValueError(f"Unknown blinded review item: {item_id}")
        candidate = str(known[item_id]["candidate"])
        candidate_values = aggregates.setdefault(candidate, {field: [] for field in RUBRIC_FIELDS})
        for field in RUBRIC_FIELDS:
            try:
                value = int(row.get(field, ""))
            except ValueError as exc:
                raise ValueError(
                    f"Review field {field} is missing or non-numeric for {item_id}"
                ) from exc
            maximum = 1 if field == "answerability_correct" else 2
            if value < 0 or value > maximum:
                raise ValueError(f"Review field {field} must be in [0, {maximum}] for {item_id}")
            candidate_values[field].append(float(value))
    summary = {
        "run_id": identity.get("run_id"),
        "judgment_count": len(rows),
        "candidates": {
            candidate: {field: sum(values) / len(values) for field, values in fields.items()}
            for candidate, fields in aggregates.items()
        },
        "complete": True,
    }
    atomic_write_json(review_path.with_suffix(".results.json"), summary)
    return summary


def _stratified_ids(samples: list[dict[str, object]], count: int, seed: int) -> list[str]:
    answerable = [str(item["sample_id"]) for item in samples if _sample_answerable(item)]
    unanswerable = [str(item["sample_id"]) for item in samples if not _sample_answerable(item)]
    random_state = random.Random(seed)
    random_state.shuffle(answerable)
    random_state.shuffle(unanswerable)
    unanswerable_count = min(len(unanswerable), max(1, count // 3))
    return answerable[: count - unanswerable_count] + unanswerable[:unanswerable_count]


def _sample_answerable(sample: dict[str, object]) -> bool:
    metadata = sample.get("metadata")
    return bool(metadata.get("answerable")) if isinstance(metadata, dict) else False


def _load_samples(run_directory: Path, candidate: str) -> list[dict[str, object]]:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "_" for character in candidate
    )
    path = run_directory / "samples" / f"{safe}.parquet"
    if not path.is_file():
        raise ValueError(f"Missing per-sample artifact for human review: {path}")
    rows = pd.read_parquet(path).to_dict(orient="records")
    return [
        {
            "sample_id": str(row["sample_id"]),
            "metadata": json.loads(str(row.get("metadata", "{}"))),
        }
        for row in rows
    ]
