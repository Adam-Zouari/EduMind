"""Minimal MLflow integration; experiment calculations remain ordinary Python."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


class Tracker(Protocol):
    @contextmanager
    def run(self, name: str, *, nested: bool = False) -> Iterator[str]: ...

    def parameters(self, values: Mapping[str, object]) -> None: ...
    def metrics(self, values: Mapping[str, float]) -> None: ...
    def artifact(self, path: Path) -> None: ...


class NoTracking:
    @contextmanager
    def run(self, name: str, *, nested: bool = False) -> Iterator[str]:
        del nested
        yield f"no-mlflow:{name}"

    def parameters(self, values: Mapping[str, object]) -> None:
        del values

    def metrics(self, values: Mapping[str, float]) -> None:
        del values

    def artifact(self, path: Path) -> None:
        del path


class MLflowTracking:
    def __init__(self, uri: str, experiment: str) -> None:
        try:
            import mlflow
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "MLflow is required by default; install requirements/benchmarks.lock "
                "or pass --no-mlflow for debugging"
            ) from exc
        self.mlflow = mlflow
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, name: str, *, nested: bool = False) -> Iterator[str]:
        with self.mlflow.start_run(run_name=name, nested=nested) as active:
            yield str(active.info.run_id)

    def parameters(self, values: Mapping[str, object]) -> None:
        self.mlflow.log_params({key: str(value)[:500] for key, value in values.items()})

    def metrics(self, values: Mapping[str, float]) -> None:
        self.mlflow.log_metrics({key: float(value) for key, value in values.items()})

    def artifact(self, path: Path) -> None:
        self.mlflow.log_artifact(str(path))


def tracker(*, disabled: bool, experiment: str) -> Tracker:
    if disabled:
        return NoTracking()
    return MLflowTracking(
        os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        experiment,
    )
