"""Prepare the pinned official document-metric implementation."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from edumind.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    stable_hash,
)

OMNIDOCBENCH_URL = "https://github.com/opendatalab/OmniDocBench.git"
OMNIDOCBENCH_REVISION = "193627ae9e97d89188468ed1ee3b7a856ff76044"
OMNIDOCBENCH_IMAGE = "ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204"
OMNIDOCBENCH_IMAGE_LOCK = ".edumind-image.json"
OMNIDOCBENCH_SOURCE_MANIFEST = ".edumind-source.json"


def _source_inventory(source: Path) -> dict[str, str]:
    excluded = {
        ".edumind-revision",
        OMNIDOCBENCH_IMAGE_LOCK,
        OMNIDOCBENCH_SOURCE_MANIFEST,
    }
    paths = list(source.rglob("*"))
    if any(
        path.is_symlink()
        for path in paths
        if ".git" not in path.relative_to(source).parts
    ):
        raise RuntimeError(
            "Evaluator source cannot contain external/unverifiable symlinks"
        )
    return {
        path.relative_to(source).as_posix(): sha256_file(path)
        for path in sorted(paths)
        if path.is_file()
        and ".git" not in path.relative_to(source).parts
        and path.relative_to(source).as_posix() not in excluded
    }


def verify_evaluator_source(source: Path) -> str:
    """Authenticate every bind-mounted source/dependency against prepared content."""
    try:
        manifest = json.loads(
            (source / OMNIDOCBENCH_SOURCE_MANIFEST).read_text(encoding="utf-8")
        )
        marker = (source / ".edumind-revision").read_text(encoding="utf-8").strip()
        files = manifest["files"]
        if (
            type(manifest.get("schema_version")) is not int
            or manifest["schema_version"] != 1
            or manifest["source_revision"] != OMNIDOCBENCH_REVISION
            or marker != OMNIDOCBENCH_REVISION
            or not isinstance(files, dict)
            or not files
        ):
            raise ValueError("invalid source manifest")
        actual = _source_inventory(source)
        if actual != files or stable_hash(actual) != manifest["source_manifest_sha256"]:
            raise ValueError(
                "source-content manifest mismatch (changed/deleted/added file)"
            )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "OmniDocBench source integrity failed; retain the existing source and reprepare evaluators into a clean destination"
        ) from exc
    return stable_hash(actual)


def evaluator_identity(source: Path) -> dict[str, str]:
    source_checksum = verify_evaluator_source(source)
    try:
        payload = json.loads(
            (source / OMNIDOCBENCH_IMAGE_LOCK).read_text(encoding="utf-8")
        )
        digest = payload["digest"]
        if (
            payload.get("source_revision") != OMNIDOCBENCH_REVISION
            or payload.get("tag") != OMNIDOCBENCH_IMAGE
            or not isinstance(digest, str)
            or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", digest)
        ):
            raise ValueError("invalid immutable image identity")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "The OmniDocBench Docker image lock is missing or invalid; reprepare evaluators"
        ) from exc
    return {
        "source_revision": OMNIDOCBENCH_REVISION,
        "source_manifest_sha256": source_checksum,
        "image_digest": digest,
        "image_lock_sha256": sha256_file(source / OMNIDOCBENCH_IMAGE_LOCK),
    }


def prepare_evaluators(root: Path, *, dry_run: bool = False) -> list[Path]:
    destination = root / "data/benchmarks/evaluators/OmniDocBench"
    image_lock = destination / OMNIDOCBENCH_IMAGE_LOCK
    if dry_run:
        print(f"clone {OMNIDOCBENCH_URL}@{OMNIDOCBENCH_REVISION} -> {destination}")
        print(
            f"pull {OMNIDOCBENCH_IMAGE} and record its immutable digest -> {image_lock}"
        )
        return [destination, image_lock]
    source_ready = destination.exists()
    if source_ready:
        verify_evaluator_source(destination)
    if not source_ready:
        temporary = destination.with_name(destination.name + ".partial")
        if temporary.exists():
            raise RuntimeError(
                f"Retain interrupted evaluator preparation and use a clean destination: {temporary}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Interrupted checkouts remain intact, not an implicit destructive repair.
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
            [
                "git",
                "-C",
                str(temporary),
                "checkout",
                "--detach",
                OMNIDOCBENCH_REVISION,
            ],
            check=True,
        )
        head = subprocess.run(
            ["git", "-C", str(temporary), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        clean = subprocess.run(
            [
                "git",
                "-C",
                str(temporary),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        if head.stdout.strip() != OMNIDOCBENCH_REVISION or clean.stdout.strip():
            raise RuntimeError("Evaluator checkout is not the clean pinned source")
        inventory = _source_inventory(temporary)
        atomic_write_json(
            temporary / OMNIDOCBENCH_SOURCE_MANIFEST,
            {
                "schema_version": 1,
                "source_revision": OMNIDOCBENCH_REVISION,
                "files": inventory,
                "source_manifest_sha256": stable_hash(inventory),
            },
        )
        atomic_write_text(temporary / ".edumind-revision", OMNIDOCBENCH_REVISION + "\n")
        shutil.rmtree(temporary / ".git", ignore_errors=True)
        temporary.replace(destination)
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
            raise RuntimeError(
                "Docker is required to prepare OmniDocBench scoring"
            ) from exc
        digest = inspected.stdout.strip()
        if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", digest):
            raise RuntimeError(
                "Docker did not return an immutable OmniDocBench image digest"
            )
        atomic_write_json(
            image_lock,
            {
                "tag": OMNIDOCBENCH_IMAGE,
                "digest": digest,
                "source_revision": OMNIDOCBENCH_REVISION,
            },
        )
    evaluator_identity(destination)
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
            or not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", digest)
        ):
            return False
        subprocess.run(
            ["docker", "image", "inspect", digest],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (
        FileNotFoundError,
        KeyError,
        json.JSONDecodeError,
        TypeError,
        subprocess.CalledProcessError,
    ):
        return False
    return True
