"""
MLflow Metrics Logger

Provides consistent helpers for logging parameters, metrics, and artifacts to MLflow.
Simplifies experiment tracking across all experiment scripts.
"""

import mlflow
import json
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_experiment(experiment_name: str) -> str:
    """
    Set the active MLflow experiment.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Experiment ID
    """
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    logger.info(f"Active MLflow experiment: {experiment_name} (ID: {experiment.experiment_id})")
    return experiment.experiment_id


def start_run(run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
    """
    Start a new MLflow run.
    
    Args:
        run_name: Optional name for the run
        tags: Optional tags for the run
        
    Returns:
        MLflow run object
    """
    run = mlflow.start_run(run_name=run_name, tags=tags)
    logger.info(f"Started MLflow run: {run.info.run_name} (ID: {run.info.run_id})")
    return run


def log_params(params: Dict[str, Any]):
    """
    Log multiple parameters to MLflow.
    
    Args:
        params: Dictionary of parameters to log
    """
    for key, value in params.items():
        # Convert non-string values to strings
        if not isinstance(value, (str, int, float, bool)):
            value = str(value)
        mlflow.log_param(key, value)
    
    logger.debug(f"Logged {len(params)} parameters to MLflow")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Log multiple metrics to MLflow.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Optional step number for time-series metrics
    """
    for key, value in metrics.items():
        # Ensure value is numeric
        if isinstance(value, (int, float, np.number)):
            mlflow.log_metric(key, float(value), step=step)
    
    logger.debug(f"Logged {len(metrics)} metrics to MLflow")


def log_dict_as_json(data: Dict[str, Any], filename: str):
    """
    Save a dictionary as JSON and log as artifact.
    
    Args:
        data: Dictionary to save
        filename: Name of the JSON file
    """
    # Ensure .json extension
    if not filename.endswith('.json'):
        filename += '.json'
    
    # Save to temporary file
    temp_path = Path(filename)
    with open(temp_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    # Log as artifact
    mlflow.log_artifact(str(temp_path))
    
    # Clean up
    temp_path.unlink()
    
    logger.debug(f"Logged JSON artifact: {filename}")


def log_text_as_artifact(text: str, filename: str):
    """
    Save text as file and log as artifact.
    
    Args:
        text: Text content to save
        filename: Name of the text file
    """
    # Ensure .txt extension if not specified
    if '.' not in filename:
        filename += '.txt'
    
    # Save to temporary file
    temp_path = Path(filename)
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Log as artifact
    mlflow.log_artifact(str(temp_path))
    
    # Clean up
    temp_path.unlink()
    
    logger.debug(f"Logged text artifact: {filename}")


def log_numpy_array(array: np.ndarray, filename: str):
    """
    Save numpy array and log as artifact.
    
    Args:
        array: Numpy array to save
        filename: Name of the .npy file
    """
    # Ensure .npy extension
    if not filename.endswith('.npy'):
        filename += '.npy'
    
    # Save to temporary file
    temp_path = Path(filename)
    np.save(temp_path, array)
    
    # Log as artifact
    mlflow.log_artifact(str(temp_path))
    
    # Clean up
    temp_path.unlink()
    
    logger.debug(f"Logged numpy artifact: {filename}")


def log_figure(fig: plt.Figure, filename: str):
    """
    Save matplotlib figure and log as artifact.
    
    Args:
        fig: Matplotlib figure
        filename: Name of the image file
    """
    # Ensure image extension
    if not any(filename.endswith(ext) for ext in ['.png', '.jpg', '.pdf']):
        filename += '.png'
    
    # Save to temporary file
    temp_path = Path(filename)
    fig.savefig(temp_path, dpi=300, bbox_inches='tight')
    
    # Log as artifact
    mlflow.log_artifact(str(temp_path))
    
    # Clean up
    temp_path.unlink()
    plt.close(fig)
    
    logger.debug(f"Logged figure artifact: {filename}")


def log_experiment_results(
    params: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Optional[Dict[str, Any]] = None
):
    """
    Comprehensive logging of parameters, metrics, and artifacts.
    
    Args:
        params: Parameters to log
        metrics: Metrics to log
        artifacts: Optional dictionary of artifacts to log
                  Format: {'filename': content} where content can be:
                  - dict (saved as JSON)
                  - str (saved as text)
                  - np.ndarray (saved as .npy)
                  - plt.Figure (saved as image)
    """
    # Log parameters
    log_params(params)
    
    # Log metrics
    log_metrics(metrics)
    
    # Log artifacts
    if artifacts:
        for filename, content in artifacts.items():
            if isinstance(content, dict):
                log_dict_as_json(content, filename)
            elif isinstance(content, str):
                log_text_as_artifact(content, filename)
            elif isinstance(content, np.ndarray):
                log_numpy_array(content, filename)
            elif isinstance(content, plt.Figure):
                log_figure(content, filename)
            else:
                logger.warning(f"Unknown artifact type for {filename}: {type(content)}")


def create_comparison_plot(
    data: Dict[str, List[float]],
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str = "comparison.png"
) -> plt.Figure:
    """
    Create a bar plot comparing multiple metrics.
    
    Args:
        data: Dictionary mapping labels to values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        filename: Filename to save
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(data.keys())
    values = list(data.values())
    
    # Handle both single values and lists
    if isinstance(values[0], list):
        # Multiple values per label - use bar plot with error bars
        means = [np.mean(v) for v in values]
        stds = [np.std(v) for v in values]
        ax.bar(labels, means, yerr=stds, capsize=5, alpha=0.7)
    else:
        # Single values - simple bar plot
        ax.bar(labels, values, alpha=0.7)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    return fig


def end_run(status: str = "FINISHED"):
    """
    End the current MLflow run.
    
    Args:
        status: Run status (FINISHED, FAILED, KILLED)
    """
    mlflow.end_run(status=status)
    logger.info(f"Ended MLflow run with status: {status}")


class MLflowExperiment:
    """
    Context manager for MLflow experiments.
    
    Usage:
        with MLflowExperiment("experiment_name", "run_name") as exp:
            exp.log_params({"param1": "value1"})
            exp.log_metrics({"metric1": 0.95})
            exp.log_artifact("results.json", {"key": "value"})
    """
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.run = None
    
    def __enter__(self):
        set_experiment(self.experiment_name)
        self.run = start_run(run_name=self.run_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error during MLflow run: {exc_val}")
            end_run(status="FAILED")
        else:
            end_run(status="FINISHED")
    
    def log_params(self, params: Dict[str, Any]):
        log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float]):
        log_metrics(metrics)
    
    def log_artifact(self, filename: str, content: Any):
        """Log a single artifact."""
        log_experiment_results({}, {}, {filename: content})


if __name__ == "__main__":
    # Test MLflow logging utilities
    print("=== MLflow Logging Utilities Test ===\n")
    
    # Test experiment setup
    with MLflowExperiment("test_experiment", "test_run") as exp:
        # Test parameter logging
        exp.log_params({
            "model": "test-model",
            "batch_size": 32,
            "learning_rate": 0.001
        })
        print("✓ Logged parameters")
        
        # Test metric logging
        exp.log_metrics({
            "accuracy": 0.95,
            "loss": 0.15,
            "f1_score": 0.93
        })
        print("✓ Logged metrics")
        
        # Test artifact logging
        exp.log_artifact("test_results.json", {"test": "data", "score": 0.95})
        exp.log_artifact("test_text.txt", "This is a test text artifact")
        exp.log_artifact("test_array.npy", np.array([1, 2, 3, 4, 5]))
        print("✓ Logged artifacts")
        
        # Test plot creation
        fig = create_comparison_plot(
            {"Model A": 0.85, "Model B": 0.90, "Model C": 0.88},
            "Model Comparison",
            "Model",
            "Accuracy"
        )
        exp.log_artifact("comparison.png", fig)
        print("✓ Logged comparison plot")
    
    print("\n✓ All MLflow logging utilities working correctly")
    print("\nRun 'mlflow ui' and navigate to http://localhost:5000 to view the test run")
