"""Project CLI entrypoints."""

from __future__ import annotations

import argparse
import subprocess
import sys

from edumind.common.paths import PROJECT_ROOT


def _run(command: list[str]) -> int:
    return subprocess.call(command, cwd=PROJECT_ROOT)


def run_ui() -> int:
    return _run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(PROJECT_ROOT / "apps" / "streamlit_app.py"),
        ]
    )


def run_ocr_api() -> int:
    return _run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "services.ocr_service:app",
            "--app-dir",
            str(PROJECT_ROOT),
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
    )


def run_rag_api() -> int:
    return _run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "services.rag_service:app",
            "--app-dir",
            str(PROJECT_ROOT),
            "--host",
            "0.0.0.0",
            "--port",
            "8001",
        ]
    )


def run_experiments() -> int:
    return _run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "mlflow" / "run_all_experiments.py"),
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="EduMind-AI developer CLI")
    parser.add_argument(
        "target",
        choices=["ui", "ocr-api", "rag-api", "experiments"],
        help="Runtime target to launch.",
    )
    args = parser.parse_args()

    mapping = {
        "ui": run_ui,
        "ocr-api": run_ocr_api,
        "rag-api": run_rag_api,
        "experiments": run_experiments,
    }
    return mapping[args.target]()


if __name__ == "__main__":
    raise SystemExit(main())
