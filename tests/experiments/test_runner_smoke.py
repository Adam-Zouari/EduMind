from __future__ import annotations

import argparse
import importlib

from experiments.mlflow.run_all_experiments import RUNNERS, build_parser, select_runners


import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "experiments.mlflow.run_all_experiments",
        "experiments.mlflow.chunking_experiments.run_experiments",
        "experiments.mlflow.embedding_experiments.run_experiments",
        "experiments.mlflow.vectordb_experiments.run_experiments",
        "experiments.mlflow.retrieval_experiments.run_experiments",
        "experiments.mlflow.llm_experiments.run_experiments",
        "experiments.mlflow.final_experiments.run_experiments",
    ],
)
def test_maintained_runner_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_runner_selection_supports_all_suite() -> None:
    args = argparse.Namespace(suite="all")
    selected = select_runners(args)

    assert [runner.name for runner in selected] == [runner.name for runner in RUNNERS]


def test_runner_selection_supports_smoke_suite() -> None:
    args = argparse.Namespace(suite="smoke")
    selected = select_runners(args)

    assert [runner.name for runner in selected] == [runner.name for runner in RUNNERS]


def test_parser_accepts_new_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["--suite", "retrieval", "--dataset", "student_benchmark", "--resume", "--top-n", "3"]
    )

    assert args.suite == "retrieval"
    assert args.dataset == "student_benchmark"
    assert args.resume is True
    assert args.top_n == 3
