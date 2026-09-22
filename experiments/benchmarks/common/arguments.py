"""Small command-line helpers shared by direct experiment scripts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from edumind.common.paths import PROJECT_ROOT

from .decisions import load_engineer_decision

LIFECYCLE_PROFILES = ("smoke", "preflight", "development", "validation", "locked")


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


def execution_devices(
    profile: str,
    requested: str | None,
    *,
    smoke_devices: Sequence[str],
    authoritative_device: str,
    allow_preflight: bool = True,
) -> tuple[str, ...]:
    """Resolve device runs without weakening authoritative profile contracts."""

    if requested == "both" and profile != "smoke":
        raise ValueError("--device both is valid only with --profile smoke")
    if profile == "smoke":
        available = tuple(smoke_devices) or (authoritative_device,)
        if requested is None or requested == "both":
            return available
        if requested not in available:
            raise ValueError(
                f"Smoke device {requested!r} is not declared by the protocol"
            )
        return (requested,)
    if profile == "preflight":
        if not allow_preflight:
            raise ValueError("This benchmark does not define GPU preflight")
        if requested not in {None, "cuda"}:
            raise ValueError("GPU preflight requires CUDA")
        return ("cuda",)
    if requested not in {None, authoritative_device}:
        raise ValueError(
            f"{profile} requires the protocol device {authoritative_device}"
        )
    return (authoritative_device,)


def default_decision_path(benchmark: str, profile: str) -> Path | None:
    """Return the project-owned transition decision for a profile."""

    suffix = {"validation": "validation", "locked": "locked"}.get(profile)
    if suffix is None:
        return None
    return PROJECT_ROOT / "data/benchmarks/decisions" / f"{benchmark}-{suffix}.json"


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
