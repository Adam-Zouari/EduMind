"""Small command-line helpers shared by direct experiment scripts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path

import yaml


def parser(description: str) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--profile", choices=("smoke", "standard", "full"), default="smoke")
    result.add_argument("--manifest", type=Path)
    result.add_argument(
        "--shortlist",
        type=Path,
        help="summary.json whose Pareto candidates replace candidates.yaml (normally for full)",
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
                "Full profiles run explicitly approved finalists only; provide --shortlist SUMMARY_JSON"
            )
        return load_candidates(path, profile)
    payload = json.loads(shortlist.read_text(encoding="utf-8"))
    values = payload.get("pareto_candidates")
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{shortlist} has no non-empty pareto_candidates list")
    return tuple(values)
