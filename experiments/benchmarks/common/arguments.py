"""Small command-line helpers shared by direct experiment scripts."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from itertools import product
from pathlib import Path

import yaml

from .decisions import load_engineer_decision


_CANDIDATE_IDENTITY_FIELDS = {"profiles", "model_id", "backend", "loader"}


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
            help="engineer decision JSON whose selected candidates replace candidates.yaml",
        )
    result.add_argument("--no-mlflow", action="store_true", help="Debug without MLflow logging")
    return result


def load_candidates(path: Path, profile: str) -> tuple[str, ...]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    structured_matrix = _structured_matrix(payload, profile, path)
    if structured_matrix is not None:
        return structured_matrix
    if isinstance(payload, Mapping) and isinstance(payload.get("candidates"), Mapping):
        unknown_sections = set(payload) - {"candidates"}
        if unknown_sections:
            raise ValueError(
                f"{path} candidate registry has unknown sections: "
                + ", ".join(sorted(unknown_sections))
            )
        result = []
        for alias, raw in payload["candidates"].items():
            if not isinstance(alias, str) or not isinstance(raw, Mapping):
                raise ValueError(f"{path} has a malformed structured candidate registry")
            _validate_candidate_identity(raw, path, alias)
            profiles = raw.get("profiles")
            if not _string_list(profiles):
                raise ValueError(f"{path} candidate {alias!r} requires non-empty profiles")
            if profile in profiles:
                result.append(alias)
        if result:
            return tuple(result)
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


def _structured_matrix(
    payload: object, profile: str, path: Path
) -> tuple[str, ...] | None:
    if not isinstance(payload, Mapping) or "matrix" not in payload:
        return None
    matrix = payload.get("matrix")
    if not isinstance(matrix, Mapping) or set(matrix) != {"axes", "separator"}:
        raise ValueError(f"{path} has a malformed structured candidate matrix")
    axes = matrix.get("axes")
    separator = matrix.get("separator")
    if not _string_list(axes) or not isinstance(separator, str) or not separator:
        raise ValueError(f"{path} candidate matrix requires axes and a separator")
    unknown = set(payload) - {"matrix", *axes}
    if unknown:
        raise ValueError(
            f"{path} candidate matrix has unknown sections: {', '.join(sorted(unknown))}"
        )
    choices: list[list[str]] = []
    for axis in axes:
        registry = payload.get(axis)
        if not isinstance(registry, Mapping) or not registry:
            raise ValueError(f"{path} candidate matrix axis {axis!r} is empty")
        selected = []
        for alias, raw in registry.items():
            if not isinstance(alias, str) or not isinstance(raw, Mapping):
                raise ValueError(f"{path} has a malformed {axis!r} candidate")
            _validate_candidate_identity(raw, path, alias)
            profiles = raw.get("profiles")
            if not _string_list(profiles):
                raise ValueError(
                    f"{path} candidate {alias!r} requires non-empty profiles"
                )
            if profile in profiles:
                selected.append(alias)
        if not selected:
            raise ValueError(
                f"{path} has no {axis!r} candidates for profile {profile!r}"
            )
        choices.append(selected)
    return tuple(separator.join(items) for items in product(*choices))


def _string_list(value: object) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
        and len(set(value)) == len(value)
    )


def _validate_candidate_identity(
    value: Mapping[object, object], path: Path, alias: str
) -> None:
    unknown = set(value) - _CANDIDATE_IDENTITY_FIELDS
    if unknown:
        raise ValueError(
            f"{path} candidate {alias!r} has non-identity fields: "
            + ", ".join(sorted(str(name) for name in unknown))
        )


def resolved_candidates(
    path: Path,
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
        return load_candidates(path, profile)
    return load_engineer_decision(
        shortlist,
        expected_source=expected_source,
        minimum=minimum,
        maximum=maximum,
        exact=exact,
    ).selected_candidates
