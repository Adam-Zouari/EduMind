"""Optional MLflow adapter; benchmark logic never depends on MLflow."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol


class Tracker(Protocol):
    @contextmanager
    def run(self, name: str, *, nested: bool = False) -> Iterator[str]: ...

    def log_parameters(self, values: Mapping[str, object]) -> None: ...
    def log_metrics(self, values: Mapping[str, float]) -> None: ...
    def log_artifact(self, path: Path) -> None: ...


class NullTracker:
    @contextmanager
    def run(self, name: str, *, nested: bool = False) -> Iterator[str]:
        del nested
        yield f"local:{name}"

    def log_parameters(self, values: Mapping[str, object]) -> None:
        del values

    def log_metrics(self, values: Mapping[str, float]) -> None:
        del values

    def log_artifact(self, path: Path) -> None:
        del path


class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment: str = "EduMind Benchmarks") -> None:
        try:
            import mlflow
        except ModuleNotFoundError as exc:
            raise RuntimeError("MLflow tracking requested but MLflow is not installed") from exc
        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

    @contextmanager
    def run(self, name: str, *, nested: bool = False) -> Iterator[str]:
        with self.mlflow.start_run(run_name=name, nested=nested) as active:
            yield str(active.info.run_id)

    def log_parameters(self, values: Mapping[str, object]) -> None:
        self.mlflow.log_params({key: str(value)[:500] for key, value in values.items()})

    def log_metrics(self, values: Mapping[str, float]) -> None:
        self.mlflow.log_metrics(dict(values))

    def log_artifact(self, path: Path) -> None:
        self.mlflow.log_artifact(str(path))


def build_tracker(uri: str | None) -> Tracker:
    return MLflowTracker(uri) if uri else NullTracker()
