"""Small command-line helpers shared by direct experiment scripts."""

from __future__ import annotations

import argparse
import json
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
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{path} must define a non-empty string list for profile '{profile}'")
    return tuple(values)


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
