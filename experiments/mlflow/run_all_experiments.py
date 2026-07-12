"""Main runner for the maintained MLflow experiment suite."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from dataclasses import dataclass

from edumind.common.paths import PROJECT_ROOT
from experiments.mlflow.mlflow_config import configure_mlflow, get_tracking_uri

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerSpec:
    """One maintained experiment runner."""

    name: str
    module: str
    description: str


RUNNERS = (
    RunnerSpec(
        name="chunking",
        module="experiments.mlflow.chunking_experiments.run_experiments",
        description="Chunking strategy comparisons",
    ),
    RunnerSpec(
        name="embedding",
        module="experiments.mlflow.embedding_experiments.run_experiments",
        description="Embedding model comparisons",
    ),
    RunnerSpec(
        name="retrieval",
        module="experiments.mlflow.retrieval_experiments.run_experiments",
        description="Dense and hybrid retrieval comparisons",
    ),
    RunnerSpec(
        name="llm",
        module="experiments.mlflow.llm_experiments.run_experiments",
        description="Ollama answer-generation comparisons",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the maintained experiment CLI parser."""
    parser = argparse.ArgumentParser(description="Run maintained EduMind MLflow experiments.")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run smaller, faster experiment passes.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[runner.name for runner in RUNNERS],
        help="Run only selected experiment groups.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip Ollama-based LLM experiments.",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Start the MLflow UI after the experiment run finishes.",
    )
    parser.add_argument("--full", action="store_true", help=argparse.SUPPRESS)
    return parser


def select_runners(args: argparse.Namespace) -> list[RunnerSpec]:
    """Select the maintained runners based on CLI arguments."""
    selected_names = set(args.only or [runner.name for runner in RUNNERS])
    selected_runners = [runner for runner in RUNNERS if runner.name in selected_names]
    if args.skip_llm:
        selected_runners = [runner for runner in selected_runners if runner.name != "llm"]
    return selected_runners


def run_experiment_runner(spec: RunnerSpec, *, test_mode: bool) -> int:
    """Run one maintained experiment module in a subprocess."""
    command = [sys.executable, "-m", spec.module]
    if test_mode:
        command.append("--test-mode")

    logger.info("Running %s experiments", spec.name)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return int(completed.returncode)


def maybe_start_mlflow_ui() -> subprocess.Popen[str]:
    """Start the MLflow UI for the maintained tracking database."""
    logger.info("Starting MLflow UI for %s", get_tracking_uri())
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            get_tracking_uri(),
        ],
        cwd=PROJECT_ROOT,
        text=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the selected maintained experiment suite."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    configure_mlflow(verbose=True)
    selected_runners = select_runners(args)
    if not selected_runners:
        logger.error("No experiment runners selected.")
        return 1

    failures: list[str] = []
    for runner in selected_runners:
        exit_code = run_experiment_runner(runner, test_mode=args.test_mode)
        if exit_code != 0:
            failures.append(runner.name)

    if args.ui:
        maybe_start_mlflow_ui()

    if failures:
        logger.error("Experiment runners failed: %s", ", ".join(failures))
        return 1

    logger.info("Experiment suite completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
