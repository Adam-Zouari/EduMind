"""MLflow logging helpers for maintained experiments."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from experiments.mlflow.mlflow_config import configure_mlflow, ensure_experiment

logger = logging.getLogger(__name__)


def set_experiment(experiment_name: str) -> str:
    """Set the active MLflow experiment and ensure its artifact root exists."""
    configure_mlflow(verbose=False)
    experiment_id = ensure_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None:
        logger.info(
            "Active MLflow experiment: %s (ID: %s)",
            experiment_name,
            experiment.experiment_id,
        )
        return experiment.experiment_id
    return experiment_id


def start_run(run_name: str | None = None, tags: dict[str, str] | None = None) -> mlflow.ActiveRun:
    """Start a new MLflow run."""
    run = mlflow.start_run(run_name=run_name, tags=tags)
    logger.info("Started MLflow run: %s (ID: %s)", run.info.run_name, run.info.run_id)
    return run


def log_params(params: dict[str, Any]) -> None:
    """Log multiple parameters to MLflow."""
    for key, value in params.items():
        mlflow.log_param(key, value if isinstance(value, (str, int, float, bool)) else str(value))


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log multiple numeric metrics to MLflow."""
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.number)):
            mlflow.log_metric(key, float(value), step=step)


def log_dict_as_json(data: dict[str, Any], filename: str) -> None:
    """Save a dictionary as JSON and log it as an artifact."""
    resolved_filename = filename if filename.endswith(".json") else f"{filename}.json"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / resolved_filename
        temp_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(temp_path))


def log_text_as_artifact(text: str, filename: str) -> None:
    """Save text and log it as an artifact."""
    resolved_filename = filename if "." in filename else f"{filename}.txt"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / resolved_filename
        temp_path.write_text(text, encoding="utf-8")
        mlflow.log_artifact(str(temp_path))


def log_numpy_array(array: np.ndarray, filename: str) -> None:
    """Save a NumPy array and log it as an artifact."""
    resolved_filename = filename if filename.endswith(".npy") else f"{filename}.npy"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / resolved_filename
        np.save(temp_path, array)
        mlflow.log_artifact(str(temp_path))


def log_figure(fig: plt.Figure, filename: str) -> None:
    """Save a Matplotlib figure and log it as an artifact."""
    resolved_filename = filename if any(
        filename.endswith(extension) for extension in [".png", ".jpg", ".pdf"]
    ) else f"{filename}.png"
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / resolved_filename
        fig.savefig(temp_path, dpi=300, bbox_inches="tight")
        mlflow.log_artifact(str(temp_path))
    plt.close(fig)


def log_experiment_results(
    params: dict[str, Any],
    metrics: dict[str, float],
    artifacts: dict[str, Any] | None = None,
) -> None:
    """Log parameters, metrics, and optional artifacts with one helper call."""
    log_params(params)
    log_metrics(metrics)
    if artifacts:
        for filename, content in artifacts.items():
            if isinstance(content, dict):
                log_dict_as_json(content, filename)
            elif isinstance(content, (list, tuple)):
                log_text_as_artifact(
                    json.dumps(content, indent=2, ensure_ascii=False, default=str),
                    filename if "." in filename else f"{filename}.json",
                )
            elif isinstance(content, str):
                log_text_as_artifact(content, filename)
            elif isinstance(content, np.ndarray):
                log_numpy_array(content, filename)
            elif isinstance(content, plt.Figure):
                log_figure(content, filename)
            else:
                logger.warning("Unknown artifact type for %s: %s", filename, type(content))


def create_comparison_plot(
    data: dict[str, list[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str = "comparison.png",
) -> plt.Figure:
    """Create a simple comparison plot for experiment summaries."""
    del filename
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = list(data.keys())
    values = list(data.values())
    if values and isinstance(values[0], list):
        means = [np.mean(value) for value in values]
        stds = [np.std(value) for value in values]
        ax.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)
    else:
        ax.bar(labels, values, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def end_run(status: str = "FINISHED") -> None:
    """End the current MLflow run."""
    mlflow.end_run(status=status)


class MLflowExperiment:
    """Context manager for maintained MLflow experiment runs."""

    def __init__(self, experiment_name: str, run_name: str | None = None) -> None:
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run: mlflow.ActiveRun | None = None

    def __enter__(self) -> MLflowExperiment:
        set_experiment(self.experiment_name)
        self.run = start_run(run_name=self.run_name)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        end_run(status="FAILED" if exc_type is not None else "FINISHED")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters for the active run."""
        log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log metrics for the active run."""
        log_metrics(metrics)

    def log_artifact(self, filename: str, content: Any) -> None:
        """Log a single artifact for the active run."""
        log_experiment_results({}, {}, {filename: content})
