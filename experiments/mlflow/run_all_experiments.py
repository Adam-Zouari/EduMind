"""Run the full experiment suite."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.mlflow.mlflow_config import DB_PATH


class Colors:
    HEADER = "\033[95m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


ROOT = Path(__file__).resolve().parent


def print_header(text: str) -> None:
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 70}{Colors.ENDC}\n")


def print_success(text: str) -> None:
    print(f"{Colors.OKGREEN}OK {text}{Colors.ENDC}")


def print_error(text: str) -> None:
    print(f"{Colors.FAIL}ERR {text}{Colors.ENDC}")


def print_info(text: str) -> None:
    print(f"{Colors.OKCYAN}INFO {text}{Colors.ENDC}")


def check_dependencies() -> None:
    required_packages = ["mlflow", "sentence_transformers", "rank_bm25", "numpy", "matplotlib", "sklearn"]
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    if missing:
        print_error(f"Missing packages: {', '.join(missing)}")
    else:
        print_success("All dependencies appear to be installed")


def check_ollama_for_llm() -> bool:
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except Exception:
        print_info("Ollama not detected; LLM experiments will be skipped")
        return False


def run_experiment(name: str, module: str, args: list[str] | None = None, skip: bool = False) -> bool:
    if skip:
        print_info(f"Skipping {name}")
        return True

    print(f"\n{Colors.BOLD}Running: {name}{Colors.ENDC}")
    print("-" * 70)

    command = [sys.executable, "-m", module]
    if args:
        command.extend(args)

    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT.parent.parent / "src"), str(ROOT.parent.parent), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(command, cwd=ROOT.parent.parent, capture_output=False, text=True, env=env)
    elapsed = time.time() - start_time

    if result.returncode == 0:
        print_success(f"{name} completed in {elapsed:.1f}s")
        return True

    print_error(f"{name} failed with return code {result.returncode}")
    return False


def start_mlflow_ui() -> None:
    print_header("Starting MLflow UI")
    print_info("MLflow UI will be available at http://localhost:5000")
    db_uri = f"sqlite:///{DB_PATH.as_posix()}"
    subprocess.run([sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", db_uri])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all MLflow experiments")
    parser.add_argument("--full", action="store_true", help="Run the full experiment suite")
    parser.add_argument("--ui", action="store_true", help="Start the MLflow UI after experiments")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM experiments")
    args = parser.parse_args()

    print_header("MLflow Experiments")
    check_dependencies()

    ollama_available = check_ollama_for_llm()
    skip_llm = args.skip_llm or not ollama_available
    test_args = [] if args.full else ["--test-mode"]

    experiments = [
        ("Embedding Model Experiments", "experiments.mlflow.embedding_experiments.run_experiments", test_args, False),
        ("Retrieval Strategy Experiments", "experiments.mlflow.retrieval_experiments.run_experiments", test_args, False),
        ("Chunking Strategy Experiments", "experiments.mlflow.chunking_experiments.run_experiments", test_args, False),
        ("LLM Model Experiments", "experiments.mlflow.llm_experiments.run_experiments", test_args + ["--num-queries", "3"], skip_llm),
    ]

    results = {}
    start_time = time.time()
    for name, module, exp_args, skip in experiments:
        results[name] = run_experiment(name, module, exp_args, skip)

    total_time = time.time() - start_time
    print_header("Summary")
    for name, success in results.items():
        if success:
            print_success(name)
        else:
            print_error(name)
    print_info(f"Total elapsed time: {total_time:.1f}s")

    if args.ui:
        start_mlflow_ui()


if __name__ == "__main__":
    main()
