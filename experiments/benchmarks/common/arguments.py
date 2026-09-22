"""Small command-line helpers shared by direct experiment scripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .decisions import load_engineer_decision


def parser(
    description: str,
    *,
    shortlist: bool = True,
    profiles: Sequence[str] = ("smoke", "development", "validation"),
) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--profile", choices=tuple(profiles), default="smoke")
    result.add_argument("--manifest", type=Path)
    if shortlist:
        result.add_argument(
            "--shortlist",
            type=Path,
            help="engineer decision JSON selecting candidates from a completed run",
        )
    result.add_argument(
        "--no-mlflow", action="store_true", help="Debug without MLflow logging"
    )
    return result


def resolved_candidates(
    declared: Sequence[str],
    profile: str,
    shortlist: Path | None,
    *,
    expected_source: tuple[str, str, str] | None = None,
    minimum: int = 1,
    maximum: int | None = None,
    exact: int | None = None,
) -> tuple[str, ...]:
    if shortlist is None:
        if profile == "validation":
            raise ValueError(
                "Validation profiles run engineer-selected finalists only; provide "
                "--shortlist DECISION_JSON"
            )
        return tuple(declared)
    selected = load_engineer_decision(
        shortlist,
        expected_source=expected_source,
        minimum=minimum,
        maximum=maximum,
        exact=exact,
    ).selected_candidates
    unknown = sorted(set(selected) - set(declared))
    if unknown:
        raise ValueError(
            "Decision selects unsupported candidates: " + ", ".join(unknown)
        )
    return selected
