"""Main runner for the staged English-only MLflow experiment suite."""

from __future__ import annotations

import argparse
import importlib
import logging
import subprocess
import sys
from dataclasses import dataclass

from edumind.common.paths import PROJECT_ROOT
from experiments.mlflow.mlflow_config import configure_mlflow, get_tracking_uri

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunnerSpec:
    """One maintained staged experiment runner."""

    name: str
    module: str
    description: str


RUNNERS = (
    RunnerSpec("chunking", "experiments.mlflow.chunking_experiments.run_experiments", "Chunking strategy comparisons"),
    RunnerSpec("embedding", "experiments.mlflow.embedding_experiments.run_experiments", "Embedding model comparisons"),
    RunnerSpec("vectordb", "experiments.mlflow.vectordb_experiments.run_experiments", "Vector database comparisons"),
    RunnerSpec("retrieval", "experiments.mlflow.retrieval_experiments.run_experiments", "Retrieval strategy comparisons"),
    RunnerSpec("llm", "experiments.mlflow.llm_experiments.run_experiments", "SLM answer-generation comparisons"),
    RunnerSpec("final", "experiments.mlflow.final_experiments.run_experiments", "Final full-stack bakeoff"),
)

SUITES = ("smoke", "chunking", "embedding", "vectordb", "retrieval", "llm", "final", "all")


def build_parser() -> argparse.ArgumentParser:
    """Build the staged experiment CLI parser."""
    parser = argparse.ArgumentParser(description="Run staged EduMind MLflow experiments.")
    parser.add_argument("--suite", choices=SUITES, default="all")
    parser.add_argument(
        "--dataset",
        choices=("synthetic_regression", "student_benchmark", "challenge_benchmark"),
        default="student_benchmark",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse completed cached candidate results.")
    parser.add_argument("--force", action="store_true", help="Ignore resume cache and rerun candidates.")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top candidates to promote.")
    parser.add_argument("--stage-limit", type=int, default=None, help="Limit how many candidates run per stage.")
    parser.add_argument("--ui", action="store_true", help="Start the MLflow UI after the suite finishes.")
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run a tiny synthetic smoke-oriented version of the suite.",
    )
    return parser


def select_runners(args: argparse.Namespace) -> list[RunnerSpec]:
    """Select stage runners based on the CLI suite arguments."""
    if args.suite == "all":
        selected = list(RUNNERS)
    elif args.suite == "smoke":
        selected = list(RUNNERS)
    else:
        selected = [runner for runner in RUNNERS if runner.name == args.suite]
    return selected


def run_experiment_runner(
    spec: RunnerSpec,
    *,
    dataset: str,
    resume: bool,
    force: bool,
    top_n: int,
    stage_limit: int | None,
    test_mode: bool,
) -> int:
    """Run one stage module by importing it and calling its public runner."""
    module = importlib.import_module(spec.module)
    run_all = getattr(module, "run_all_experiments")
    kwargs = {
        "dataset_name": dataset,
        "resume": resume,
        "force": force,
        "top_n": top_n,
        "stage_limit": stage_limit,
        "test_mode": test_mode,
    }
    logger.info("Running %s suite with %s", spec.name, kwargs)
    return int(run_all(**kwargs))


def maybe_start_mlflow_ui() -> subprocess.Popen[str]:
    """Start MLflow UI for the maintained tracking database."""
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
    """Run the selected staged experiment suite."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    configure_mlflow(verbose=True)
    selected_runners = select_runners(args)
    if not selected_runners:
        logger.error("No experiment runners selected.")
        return 1

    dataset = "synthetic_regression" if args.test_mode or args.suite == "smoke" else args.dataset
    stage_limit = args.stage_limit
    if args.suite == "smoke":
        stage_limit = stage_limit or 1

    failures: list[str] = []
    for runner in selected_runners:
        exit_code = run_experiment_runner(
            runner,
            dataset=dataset,
            resume=args.resume,
            force=args.force,
            top_n=args.top_n,
            stage_limit=stage_limit,
            test_mode=args.test_mode or args.suite == "smoke",
        )
        if exit_code != 0:
            failures.append(runner.name)
            if args.suite == "smoke":
                break

    if args.ui:
        maybe_start_mlflow_ui()

    if failures:
        logger.error("Experiment runners failed: %s", ", ".join(failures))
        return 1

    logger.info("Experiment suite completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
