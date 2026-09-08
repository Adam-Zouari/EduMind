"""Prepare the pinned official document-metric implementation."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from edumind.common.artifacts import atomic_write_json

OMNIDOCBENCH_URL = "https://github.com/opendatalab/OmniDocBench.git"
OMNIDOCBENCH_REVISION = "193627ae9e97d89188468ed1ee3b7a856ff76044"
OMNIDOCBENCH_IMAGE = "ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204"
OMNIDOCBENCH_IMAGE_LOCK = ".edumind-image.json"


def prepare_evaluators(root: Path, *, dry_run: bool = False) -> list[Path]:
    destination = root / "data/benchmarks/evaluators/OmniDocBench"
    image_lock = destination / OMNIDOCBENCH_IMAGE_LOCK
    if dry_run:
        print(f"clone {OMNIDOCBENCH_URL}@{OMNIDOCBENCH_REVISION} -> {destination}")
        print(f"pull {OMNIDOCBENCH_IMAGE} and record its immutable digest -> {image_lock}")
        return [destination, image_lock]
    revision_file = destination / ".edumind-revision"
    source_ready = (
        revision_file.is_file()
        and revision_file.read_text(encoding="utf-8").strip() == OMNIDOCBENCH_REVISION
    )
    if not source_ready:
        temporary = destination.with_name(destination.name + ".partial")
        shutil.rmtree(temporary, ignore_errors=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    OMNIDOCBENCH_URL,
                    str(temporary),
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(temporary), "checkout", "--detach", OMNIDOCBENCH_REVISION],
                check=True,
            )
            (temporary / ".edumind-revision").write_text(
                OMNIDOCBENCH_REVISION + "\n", encoding="utf-8"
            )
            shutil.rmtree(temporary / ".git", ignore_errors=True)
            if destination.exists():
                shutil.rmtree(destination)
            temporary.replace(destination)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    if not _image_available(image_lock):
        try:
            subprocess.run(["docker", "pull", OMNIDOCBENCH_IMAGE], check=True)
            inspected = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    OMNIDOCBENCH_IMAGE,
                    "--format",
                    "{{index .RepoDigests 0}}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Docker is required to prepare OmniDocBench scoring") from exc
        digest = inspected.stdout.strip()
        if "@sha256:" not in digest:
            raise RuntimeError("Docker did not return an immutable OmniDocBench image digest")
        atomic_write_json(
            image_lock,
            {
                "tag": OMNIDOCBENCH_IMAGE,
                "digest": digest,
                "source_revision": OMNIDOCBENCH_REVISION,
            },
        )
    return [destination, image_lock]


def _image_available(image_lock: Path) -> bool:
    if not image_lock.is_file():
        return False
    try:
        payload = json.loads(image_lock.read_text(encoding="utf-8"))
        digest = str(payload["digest"])
        if (
            payload.get("tag") != OMNIDOCBENCH_IMAGE
            or payload.get("source_revision") != OMNIDOCBENCH_REVISION
        ):
            return False
        subprocess.run(
            ["docker", "image", "inspect", digest],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, KeyError, json.JSONDecodeError, subprocess.CalledProcessError):
        return False
    return "@sha256:" in digest
