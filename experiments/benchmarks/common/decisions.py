"""Validate engineer-authored decisions between benchmark stages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class EngineerDecision:
    """A deliberate selection made from one complete benchmark run."""

    source_summary: Path
    source_run_id: str
    selected_candidates: tuple[str, ...]
    selected_by: str
    selected_date: str
    reason: str


def load_engineer_decision(
    path: Path,
    *,
    minimum: int = 1,
    maximum: int | None = None,
    exact: int | None = None,
) -> EngineerDecision:
    """Load a decision and prove that it references complete comparable evidence."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"{path} must use engineer-decision schema_version 1")

    raw_summary = payload.get("source_summary")
    if not isinstance(raw_summary, str) or not raw_summary.strip():
        raise ValueError(f"{path} must name its source_summary")
    source_summary = Path(raw_summary)
    if not source_summary.is_absolute():
        source_summary = (path.parent / source_summary).resolve()
    if not source_summary.is_file():
        raise ValueError(f"Decision source summary does not exist: {source_summary}")

    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    if summary.get("complete") is not True:
        raise ValueError(
            f"{source_summary} is incomplete; an engineer may select only from a complete run"
        )
    profile = summary.get("plan", {}).get("profile")
    if profile not in {"standard", "full"}:
        raise ValueError(
            f"{source_summary} uses profile {profile!r}; smoke runs cannot support selection"
        )

    source_run_id = payload.get("source_run_id")
    if not isinstance(source_run_id, str) or source_run_id != summary.get("run_id"):
        raise ValueError(f"{path} source_run_id does not match {source_summary}")

    selected = payload.get("selected_candidates")
    if (
        not isinstance(selected, list)
        or not all(isinstance(value, str) and value for value in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError(f"{path} selected_candidates must be a non-empty unique string list")
    if exact is not None and len(selected) != exact:
        raise ValueError(f"{path} must select exactly {exact} candidate(s)")
    if len(selected) < minimum:
        raise ValueError(f"{path} must select at least {minimum} candidate(s)")
    if maximum is not None and len(selected) > maximum:
        raise ValueError(f"{path} must select at most {maximum} candidate(s)")

    successful = {
        str(candidate.get("candidate"))
        for candidate in summary.get("candidates", [])
        if candidate.get("status") == "success"
    }
    unknown = sorted(set(selected) - successful)
    if unknown:
        raise ValueError(
            f"{path} selects candidates absent from the completed run: {', '.join(unknown)}"
        )

    selected_by = payload.get("selected_by")
    reason = payload.get("reason")
    selected_date = payload.get("selected_date")
    if not isinstance(selected_by, str) or not selected_by.strip():
        raise ValueError(f"{path} must record selected_by")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError(f"{path} must record the engineer's reason")
    if not isinstance(selected_date, str):
        raise ValueError(f"{path} must record selected_date as YYYY-MM-DD")
    try:
        date.fromisoformat(selected_date)
    except ValueError as exc:
        raise ValueError(f"{path} selected_date must use YYYY-MM-DD") from exc

    return EngineerDecision(
        source_summary,
        source_run_id,
        tuple(selected),
        selected_by.strip(),
        selected_date,
        reason.strip(),
    )
