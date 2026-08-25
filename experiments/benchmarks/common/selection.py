"""Read the approved benchmark shortlist from its single evidence table."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from edumind.common.paths import PROJECT_ROOT

SELECTION_PATH = PROJECT_ROOT / "experiments/benchmarks/selection_evidence.csv"
EXPECTED_COLUMNS = (
    "component",
    "candidate",
    "purpose",
    "decision",
    "approx_params_b",
    "public_benchmark",
    "public_metric",
    "public_score",
    "benchmark_source_url",
    "benchmark_source_revision",
    "candidate_source_url",
    "candidate_revision",
    "license",
    "reviewed_date",
    "reason",
)


@dataclass(frozen=True)
class SelectionEntry:
    component: str
    candidate: str
    purpose: str
    revision: str
    source_url: str


def selection_entries(path: Path = SELECTION_PATH) -> tuple[SelectionEntry, ...]:
    """Return included rows after validating the evidence-table contract."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
            raise ValueError(f"Unexpected model-selection columns in {path}")
        entries: list[SelectionEntry] = []
        seen: set[tuple[str, str]] = set()
        for row in reader:
            component = row["component"].strip()
            candidate = row["candidate"].strip()
            key = (component, candidate)
            if key in seen:
                raise ValueError(f"Duplicate selection row: {component}/{candidate}")
            seen.add(key)
            if row["decision"].strip() != "include":
                continue
            revision = row["candidate_revision"].strip()
            if not revision:
                raise ValueError(f"Included candidate has no pinned revision: {candidate}")
            entries.append(
                SelectionEntry(
                    component=component,
                    candidate=candidate,
                    purpose=row["purpose"].strip(),
                    revision=revision,
                    source_url=row["candidate_source_url"].strip(),
                )
            )
    return tuple(entries)


def included_candidates(component: str) -> tuple[str, ...]:
    return tuple(
        entry.candidate for entry in selection_entries() if entry.component == component
    )


def included_revisions(*components: str) -> dict[str, str]:
    allowed = set(components)
    return {
        entry.candidate: entry.revision
        for entry in selection_entries()
        if not allowed or entry.component in allowed
    }


def require_included(component: str, candidate: str) -> SelectionEntry:
    for entry in selection_entries():
        if entry.component == component and entry.candidate == candidate:
            return entry
    raise ValueError(f"Candidate is not approved for {component}: {candidate}")
