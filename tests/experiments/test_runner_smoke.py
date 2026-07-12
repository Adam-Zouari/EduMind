from __future__ import annotations

import argparse
import importlib

import pytest

pytest.importorskip("mlflow")

from experiments.mlflow.run_all_experiments import RUNNERS, build_parser, select_runners


@pytest.mark.parametrize(
    "module_name",
    [
        "experiments.mlflow.run_all_experiments",
        "experiments.mlflow.chunking_experiments.run_experiments",
        "experiments.mlflow.embedding_experiments.run_experiments",
        "experiments.mlflow.retrieval_experiments.run_experiments",
        "experiments.mlflow.llm_experiments.run_experiments",
    ],
)
def test_maintained_runner_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None


def test_runner_selection_honors_skip_llm() -> None:
    args = argparse.Namespace(only=None, skip_llm=True)
    selected = select_runners(args)

    assert all(runner.name != "llm" for runner in selected)
    assert len(selected) == len(RUNNERS) - 1


def test_parser_accepts_new_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--test-mode", "--skip-llm", "--only", "retrieval", "embedding"])

    assert args.test_mode is True
    assert args.skip_llm is True
    assert args.only == ["retrieval", "embedding"]
