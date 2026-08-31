"""Prepare the pinned official document-metric implementation."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

OMNIDOCBENCH_URL = "https://github.com/opendatalab/OmniDocBench.git"
OMNIDOCBENCH_REVISION = "193627ae9e97d89188468ed1ee3b7a856ff76044"


def prepare_evaluators(root: Path, *, dry_run: bool = False) -> list[Path]:
    destination = root / "data/benchmarks/evaluators/OmniDocBench"
    if dry_run:
        print(f"clone {OMNIDOCBENCH_URL}@{OMNIDOCBENCH_REVISION} -> {destination}")
        return [destination]
    revision_file = destination / ".edumind-revision"
    if revision_file.is_file() and revision_file.read_text(encoding="utf-8").strip() == OMNIDOCBENCH_REVISION:
        return [destination]
    temporary = destination.with_name(destination.name + ".partial")
    shutil.rmtree(temporary, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", OMNIDOCBENCH_URL, str(temporary)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(temporary), "checkout", "--detach", OMNIDOCBENCH_REVISION],
            check=True,
        )
        revision_file = temporary / ".edumind-revision"
        revision_file.write_text(OMNIDOCBENCH_REVISION + "\n", encoding="utf-8")
        shutil.rmtree(temporary / ".git", ignore_errors=True)
        if destination.exists():
            shutil.rmtree(destination)
        temporary.replace(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return [destination]
