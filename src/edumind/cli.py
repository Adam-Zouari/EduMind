"""Installed EduMind command-line interface."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from edumind.common.config import load_settings


def _repo_root() -> Path:
    package = Path(__file__).resolve().parent
    candidate = package.parents[1]
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd()


def _run(command: list[str]) -> int:
    return subprocess.call(command, cwd=_repo_root())


def run_ui() -> int:
    app = _repo_root() / "apps" / "streamlit_app.py"
    if not app.is_file():
        raise RuntimeError(
            "The Streamlit source app is not installed; run `edumind ui` from the repository."
        )
    return _run([sys.executable, "-m", "streamlit", "run", str(app)])


def run_extraction_api() -> int:
    settings = load_settings()
    return _run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "services.extraction_service:app",
            "--app-dir",
            str(_repo_root()),
            "--host",
            settings.service.host,
            "--port",
            str(settings.service.extraction_port),
        ]
    )


def run_rag_api() -> int:
    settings = load_settings()
    return _run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "services.rag_service:app",
            "--app-dir",
            str(_repo_root()),
            "--host",
            settings.service.host,
            "--port",
            str(settings.service.rag_port),
        ]
    )


def run_benchmark(argv: list[str] | None = None) -> int:
    from edumind.benchmarks.cli import main as benchmark_main

    return benchmark_main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edumind", description="Local EduMind extraction and RAG")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ui", help="Launch the local Streamlit app")
    subparsers.add_parser("extraction-api", help="Launch the local extraction API")
    subparsers.add_parser("rag-api", help="Launch the local RAG API")
    benchmark = subparsers.add_parser("benchmark", help="Run reproducible benchmarks")
    benchmark.add_argument("benchmark_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    # Benchmark owns a rich nested parser; forward its arguments without interpreting them here.
    if resolved_argv and resolved_argv[0] == "benchmark":
        return run_benchmark(resolved_argv[1:])
    args = build_parser().parse_args(resolved_argv)
    if args.command == "ui":
        return run_ui()
    if args.command == "extraction-api":
        return run_extraction_api()
    if args.command == "rag-api":
        return run_rag_api()
    return run_benchmark(args.benchmark_args)


if __name__ == "__main__":
    raise SystemExit(main())
