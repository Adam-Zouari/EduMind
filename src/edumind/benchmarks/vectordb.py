"""Identical-vector conformance and performance benchmark for local vector systems."""

from __future__ import annotations

import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from edumind.common.artifacts import stable_hash
from edumind.common.config import load_settings

from .contracts import BenchmarkPlan, BenchmarkResult, CandidateResult, SampleResult
from .harness import BenchmarkHarness


class VectorBackend(Protocol):
    def build(self, vectors: np.ndarray, metadata: Sequence[Mapping[str, object]]) -> None: ...
    def query(
        self, vector: np.ndarray, k: int, filters: Mapping[str, object] | None = None
    ) -> list[int]: ...
    def close(self) -> None: ...


class NumpyBackend:
    def build(self, vectors: np.ndarray, metadata: Sequence[Mapping[str, object]]) -> None:
        self.vectors = vectors
        self.metadata = list(metadata)

    def query(
        self, vector: np.ndarray, k: int, filters: Mapping[str, object] | None = None
    ) -> list[int]:
        indices = [
            index
            for index, item in enumerate(self.metadata)
            if not filters or all(item.get(key) == value for key, value in filters.items())
        ]
        return sorted(indices, key=lambda index: float(self.vectors[index] @ vector), reverse=True)[
            :k
        ]

    def close(self) -> None:
        return None


class ChromaBackend:
    def __init__(self, path: Path) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ModuleNotFoundError as exc:
            raise RuntimeError("Chroma is missing; install .[benchmarks]") from exc
        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            "benchmark", metadata={"hnsw:space": "cosine"}
        )

    def build(self, vectors: np.ndarray, metadata: Sequence[Mapping[str, object]]) -> None:
        self.collection.upsert(
            ids=[str(index) for index in range(len(vectors))],
            embeddings=vectors.tolist(),
            metadatas=cast(Any, [dict(item) for item in metadata]),
        )

    def query(
        self, vector: np.ndarray, k: int, filters: Mapping[str, object] | None = None
    ) -> list[int]:
        clauses = [{key: value} for key, value in sorted((filters or {}).items())]
        where = None if not clauses else clauses[0] if len(clauses) == 1 else {"$and": clauses}
        result = self.collection.query(
            query_embeddings=[vector.tolist()], n_results=k, where=cast(Any, where)
        )
        return [int(value) for value in result["ids"][0]]

    def close(self) -> None:
        return None


class QdrantBackend:
    def __init__(self, path: Path, dimension: int) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ModuleNotFoundError as exc:
            raise RuntimeError("Qdrant client is missing; install .[benchmarks]") from exc
        self.models = __import__("qdrant_client.models", fromlist=["models"])
        self.client = QdrantClient(path=str(path))
        self.client.create_collection(
            "benchmark", vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
        )

    def build(self, vectors: np.ndarray, metadata: Sequence[Mapping[str, object]]) -> None:
        points = [
            self.models.PointStruct(id=index, vector=vector.tolist(), payload=dict(metadata[index]))
            for index, vector in enumerate(vectors)
        ]
        self.client.upsert("benchmark", points=points, wait=True)

    def query(
        self, vector: np.ndarray, k: int, filters: Mapping[str, object] | None = None
    ) -> list[int]:
        query_filter = None
        if filters:
            query_filter = self.models.Filter(
                must=[
                    self.models.FieldCondition(key=key, match=self.models.MatchValue(value=value))
                    for key, value in filters.items()
                ]
            )
        response = self.client.query_points(
            "benchmark", query=vector.tolist(), limit=k, query_filter=query_filter
        )
        return [int(point.id) for point in response.points]

    def close(self) -> None:
        self.client.close()


class LanceBackend:
    def __init__(self, path: Path) -> None:
        try:
            import lancedb
        except ModuleNotFoundError as exc:
            raise RuntimeError("LanceDB is missing; install .[benchmarks]") from exc
        self.database = lancedb.connect(path)
        self.table = None

    def build(self, vectors: np.ndarray, metadata: Sequence[Mapping[str, object]]) -> None:
        rows = [
            {"id": index, "vector": vector.tolist(), **dict(metadata[index])}
            for index, vector in enumerate(vectors)
        ]
        self.table = self.database.create_table("benchmark", data=rows, mode="overwrite")

    def query(
        self, vector: np.ndarray, k: int, filters: Mapping[str, object] | None = None
    ) -> list[int]:
        if self.table is None:
            raise RuntimeError("LanceDB benchmark table is not built")
        query = self.table.search(vector.tolist())
        if filters:
            clauses = [
                f"{key} = '{value}'" if isinstance(value, str) else f"{key} = {value}"
                for key, value in filters.items()
            ]
            query = query.where(" AND ".join(clauses), prefilter=True)
        return [int(row["id"]) for row in query.limit(k).to_list()]

    def close(self) -> None:
        self.table = None


def run_vectordb(profile: str, *, artifact_root: Path | None = None) -> BenchmarkResult:
    settings = load_settings()
    candidates = (
        ("numpy-exact-smoke",)
        if profile == "smoke"
        else ("chroma", "qdrant-local", "lancedb-local")
    )
    count = 500 if profile == "smoke" else 10_000
    queries = 20 if profile == "smoke" else 200
    dimension = 64 if profile == "smoke" else 384
    random_state = np.random.default_rng(42)
    vectors = random_state.normal(size=(count, dimension)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    query_vectors = vectors[random_state.choice(count, size=queries, replace=False)]
    metadata = [
        {"parity": "even" if index % 2 == 0 else "odd", "bucket": index % 5}
        for index in range(count)
    ]
    dataset_checksum = stable_hash(
        {"seed": 42, "count": count, "queries": queries, "dimension": dimension}
    )
    plan = BenchmarkPlan(
        "systems",
        "vectordb",
        profile,
        "seeded-vector-conformance-v1",
        candidates,
        bootstrap_resamples=min(500, settings.benchmark.bootstrap_resamples)
        if profile == "smoke"
        else settings.benchmark.bootstrap_resamples,
    )
    harness = BenchmarkHarness(
        artifact_root or settings.benchmark.artifact_directory,
        tracking_uri=settings.benchmark.tracking_uri,
    )

    def evaluate(candidate: str):
        with tempfile.TemporaryDirectory(prefix="edumind-vectordb-") as temporary:
            path = Path(temporary)
            backend = _backend(candidate, path, dimension)
            build_started = time.perf_counter()
            backend.build(vectors, metadata)
            build_seconds = time.perf_counter() - build_started
            samples: list[SampleResult] = []
            latencies: list[float] = []
            for index, vector in enumerate(query_vectors):
                exact = _exact(vectors, vector, 10)
                filters = {"parity": "even", "bucket": 0}
                filtered_exact = _exact(vectors, vector, 10, metadata, filters)
                started = time.perf_counter()
                result = backend.query(vector, 10)
                latency = time.perf_counter() - started
                filtered_started = time.perf_counter()
                filtered = backend.query(vector, 10, filters)
                filtered_latency = time.perf_counter() - filtered_started
                latencies.append(latency)
                filter_correct = all(
                    metadata[item]["parity"] == "even" and metadata[item]["bucket"] == 0
                    for item in filtered
                )
                metrics = {
                    **{f"ann_recall_at_{k}": _recall(result[:k], exact[:k]) for k in (1, 3, 5, 10)},
                    **{
                        f"filtered_ann_recall_at_{k}": _recall(filtered[:k], filtered_exact[:k])
                        for k in (1, 3, 5, 10)
                    },
                    "filter_correctness": float(filter_correct),
                }
                samples.append(
                    SampleResult(
                        str(index), metrics, latency, {"filtered_latency_seconds": filtered_latency}
                    )
                )
            disk = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            backend.close()
            return samples, {
                "build_seconds": build_seconds,
                "build_vectors_per_second": count / max(build_seconds, 1e-9),
                "p50_latency_seconds": float(np.median(latencies)),
                "p95_latency_seconds": float(np.quantile(latencies, 0.95)),
                "disk_bytes": float(disk),
            }

    return harness.run(
        plan,
        evaluate,
        dataset_checksum=dataset_checksum,
        directions={
            "ann_recall_at_10": "max",
            "filtered_ann_recall_at_10": "max",
            "filter_correctness": "max",
            "operational.p95_latency_seconds": "min",
            "operational.disk_bytes": "min",
        },
        hard_gates={
            "ann_recall_at_10": ("max", 0.99),
            "filtered_ann_recall_at_10": ("max", 0.99),
            "filter_correctness": ("max", 1.0),
        },
    )


def _backend(candidate: str, path: Path, dimension: int) -> VectorBackend:
    if candidate == "numpy-exact-smoke":
        return NumpyBackend()
    if candidate == "chroma":
        return ChromaBackend(path)
    if candidate == "qdrant-local":
        return QdrantBackend(path, dimension)
    if candidate == "lancedb-local":
        return LanceBackend(path)
    raise ValueError(f"Unknown vector backend: {candidate}")


def _exact(
    vectors: np.ndarray,
    query: np.ndarray,
    k: int,
    metadata: Sequence[Mapping[str, object]] | None = None,
    filters: Mapping[str, object] | None = None,
) -> list[int]:
    if filters is not None and metadata is None:
        raise ValueError("Filtered exact search requires metadata")
    indices = [
        index
        for index in range(len(vectors))
        if not filters
        or all(
            cast(Sequence[Mapping[str, object]], metadata)[index].get(key) == value
            for key, value in filters.items()
        )
    ]
    return sorted(indices, key=lambda index: float(vectors[index] @ query), reverse=True)[:k]


def _recall(retrieved: Sequence[int], expected: Sequence[int]) -> float:
    return len(set(retrieved) & set(expected)) / len(expected) if expected else 1.0


def recommend_vector_backend(candidates: Sequence[CandidateResult]) -> str:
    """Apply the documented correctness and 20% migration rule."""
    successful = {item.candidate: item for item in candidates if item.status == "success"}
    chroma = successful.get("chroma")
    if chroma is None:
        return "chroma"
    chroma_p95 = chroma.operational.get("p95_latency_seconds", float("inf"))
    chroma_disk = chroma.operational.get("disk_bytes", float("inf"))
    eligible = []
    for name in ("qdrant-local", "lancedb-local"):
        candidate = successful.get(name)
        if candidate is None:
            continue
        correct = (
            candidate.metrics.get("ann_recall_at_10", 0.0) >= 0.99
            and candidate.metrics.get("filtered_ann_recall_at_10", 0.0) >= 0.99
            and candidate.metrics.get("filter_correctness", 0.0) >= 1.0
        )
        improved = (
            candidate.operational.get("p95_latency_seconds", float("inf")) <= 0.8 * chroma_p95
            or candidate.operational.get("disk_bytes", float("inf")) <= 0.8 * chroma_disk
        )
        if correct and improved:
            eligible.append(candidate)
    if not eligible:
        return "chroma"
    return min(
        eligible,
        key=lambda item: (
            item.operational.get("p95_latency_seconds", float("inf")),
            item.operational.get("disk_bytes", float("inf")),
        ),
    ).candidate
