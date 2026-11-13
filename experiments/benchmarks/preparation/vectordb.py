"""Prepare pinned vector-server images and client provenance."""

from __future__ import annotations

import re
import subprocess
from importlib.metadata import version
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, atomic_write_text
from edumind.common.paths import PROJECT_ROOT

VECTOR_IMAGES = {
    "chroma": "chromadb/chroma:1.5.9",
    "qdrant": "qdrant/qdrant:v1.17.0",
    "weaviate": "cr.weaviate.io/semitechnologies/weaviate:1.38.2",
    "pgvector": "pgvector/pgvector:0.8.2-pg17-bookworm",
    "inspector": "alpine:3.21",
}


def prepare_vectordb(output_path: Path) -> Path:
    """Pull pinned tags, resolve immutable digests, and write Compose overrides."""
    environment = PROJECT_ROOT / "experiments/benchmarks/vectordb/.env"
    destinations = (output_path.expanduser().resolve(), environment.resolve())
    if not destinations[1].is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError("Compose environment must remain in the owned project")
    if destinations[0] == destinations[1]:
        raise ValueError(
            "Vector lock destination must differ from the owned Compose environment"
        )
    for destination in destinations:
        if destination.exists() and not destination.is_file():
            raise ValueError(f"Preparation output must be a file: {destination}")
        for parent in destination.parents:
            if parent.exists() and not parent.is_dir():
                raise ValueError(
                    f"Preparation output parent is not a directory: {parent}"
                )
    packages = {
        package: version(package)
        for package in (
            "chromadb-client",
            "qdrant-client",
            "weaviate-client",
            "psycopg",
            "psycopg-pool",
        )
    }
    resolved: dict[str, str] = {}
    for name, image in VECTOR_IMAGES.items():
        subprocess.run(["docker", "pull", image], check=True)
        process = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                "{{index .RepoDigests 0}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        digest_image = process.stdout.strip()
        if not re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", digest_image):
            raise RuntimeError(f"Docker did not report an immutable digest for {image}")
        resolved[name] = digest_image
    atomic_write_json(
        destinations[0], {"schema_version": 1, "images": resolved, "clients": packages}
    )
    atomic_write_text(
        destinations[1],
        "\n".join(f"{name.upper()}_IMAGE={image}" for name, image in resolved.items())
        + "\n",
    )
    return output_path
