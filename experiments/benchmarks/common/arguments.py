"""Small command-line helpers shared by direct experiment scripts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path

import yaml

from .decisions import load_engineer_decision


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--profile", choices=("smoke", "standard", "full"), default="smoke")
    result.add_argument("--manifest", type=Path)
    result.add_argument(
        "--shortlist",
        type=Path,
        help="engineer decision JSON whose selected candidates replace candidates.yaml",
    )
    result.add_argument("--no-mlflow", action="store_true", help="Debug without MLflow logging")
    return result


def load_candidates(path: Path, profile: str) -> tuple[str, ...]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = payload.get(profile) if isinstance(payload, dict) else None
    if isinstance(values, list) and values and all(isinstance(value, str) for value in values):
        return tuple(values)
    if isinstance(values, Mapping):
        factors = values.get("matrix")
        if isinstance(factors, Mapping) and factors:
            choices = list(factors.values())
            if all(_string_list(choice) for choice in choices):
                return tuple("|".join(items) for items in product(*choices))
    raise ValueError(
        f"{path} must define a non-empty string list or string-list matrix "
        f"for profile '{profile}'"
    )


def _string_list(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and bool(value)
        and all(isinstance(item, str) for item in value)
    )


def resolved_candidates(path: Path, profile: str, shortlist: Path | None) -> tuple[str, ...]:
    if shortlist is None:
        if profile == "full":
            raise ValueError(
                "Full profiles run engineer-selected finalists only; provide --shortlist DECISION_JSON"
            )
        return load_candidates(path, profile)
    return load_engineer_decision(shortlist).selected_candidates
