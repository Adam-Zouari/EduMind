from __future__ import annotations

import csv
import json

from edumind.benchmarks.review import RUBRIC_FIELDS, export_review, import_review


def test_blinded_60_item_review_roundtrip(tmp_path) -> None:
    candidates = []
    for candidate_index in range(3):
        samples = []
        for index in range(24):
            samples.append(
                {
                    "sample_id": f"q{index}",
                    "metrics": {},
                    "latency_seconds": 0.1,
                    "metadata": {
                        "answerable": index % 3 != 0,
                        "question": f"Question {index}",
                        "generated_answer": f"Answer {candidate_index} [1]",
                        "frozen_context": "Evidence",
                    },
                }
            )
        candidates.append(
            {
                "candidate": f"system-{candidate_index}",
                "status": "success",
                "metrics": {"citation_f1": 1.0},
                "operational": {"p95_latency_seconds": candidate_index + 1},
                "samples": samples,
            }
        )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps({"run_id": "run", "pareto_candidates": [], "candidates": candidates})
    )
    review = export_review(summary, tmp_path / "review.csv")
    with review.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 60
    for row in rows:
        for field in RUBRIC_FIELDS:
            row[field] = "1"
    with review.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = import_review(review)
    assert result["judgment_count"] == 60
    assert result["complete"] is True
