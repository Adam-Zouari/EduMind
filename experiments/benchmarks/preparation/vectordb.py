"""Prepare pinned vector-server images and client provenance."""

from __future__ import annotations

import subprocess
from importlib.metadata import version
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, atomic_write_text

VECTOR_IMAGES = {
    "chroma": "chromadb/chroma:1.5.9",
    "qdrant": "qdrant/qdrant:v1.17.0",
    "weaviate": "cr.weaviate.io/semitechnologies/weaviate:1.38.2",
    "pgvector": "pgvector/pgvector:0.8.2-pg17-bookworm",
    "inspector": "alpine:3.21",
}


def prepare_vectordb(output_path: Path) -> Path:
    """Pull pinned tags, resolve immutable digests, and write Compose overrides."""
    resolved: dict[str, str] = {}
    for name, image in VECTOR_IMAGES.items():
        subprocess.run(["docker", "pull", image], check=True)
        process = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{index .RepoDigests 0}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        digest_image = process.stdout.strip()
        if "@sha256:" not in digest_image:
            raise RuntimeError(f"Docker did not report an immutable digest for {image}")
        resolved[name] = digest_image
    packages = {
        package: version(package)
        for package in (
            "chromadb",
            "qdrant-client",
            "weaviate-client",
            "psycopg",
            "psycopg-pool",
        )
    }
    atomic_write_json(output_path, {"schema_version": 1, "images": resolved, "clients": packages})
    environment = output_path.parents[3] / "experiments/benchmarks/vectordb/.env"
    atomic_write_text(
        environment,
        "\n".join(f"{name.upper()}_IMAGE={image}" for name, image in resolved.items()) + "\n",
    )
    return output_path

