"""Resolve immutable production model snapshots from the generated local lock."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


class ModelPreparationError(RuntimeError):
    """A required pinned model entry or local snapshot is unavailable."""


@dataclass(frozen=True)
class ModelSnapshot:
    name: str
    revision: str
    path: Path
    values: Mapping[str, object]


def load_model_lock(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise ModelPreparationError(
            f"Missing model lock {path}; run `python "
            "-m experiments.benchmarks.prepare app-models`."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPreparationError(f"Cannot read model lock {path}: {exc}") from exc
    models = payload.get("models") if isinstance(payload, Mapping) else None
    if not isinstance(models, Mapping):
        raise ModelPreparationError(f"Malformed model lock: {path}")
    return {
        str(name): dict(entry)
        for name, entry in models.items()
        if isinstance(entry, Mapping)
    }


def require_model(lock: Mapping[str, Mapping[str, object]], name: str) -> ModelSnapshot:
    entry = lock.get(name)
    if not isinstance(entry, Mapping):
        raise ModelPreparationError(
            f"Model {name} is absent from selected.json; run `python "
            "-m experiments.benchmarks.prepare app-models`."
        )
    revision = entry.get("revision")
    raw_path = entry.get("model_path")
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or not isinstance(raw_path, str)
        or not raw_path.strip()
        or not (path := Path(raw_path).expanduser()).is_dir()
    ):
        raise ModelPreparationError(
            f"Prepared model {name} is incomplete; run `python "
            "-m experiments.benchmarks.prepare app-models`."
        )
    return ModelSnapshot(name, revision.strip(), path.resolve(), dict(entry))
