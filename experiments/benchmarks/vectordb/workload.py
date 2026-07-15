"""Identical deterministic corpora and exact NumPy oracle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from edumind.common.artifacts import stable_hash

from .adapters import Hit, Record


@dataclass(frozen=True)
class Corpus:
    vectors: np.ndarray
    queries: np.ndarray
    query_indices: np.ndarray
    metadata: tuple[Mapping[str, str], ...]
    fingerprint: str


def clustered(size: int, dimension: int, queries: int, seed: int = 42) -> Corpus:
    random = np.random.default_rng(seed + size + dimension)
    centroids = random.normal(size=(128, dimension)).astype(np.float32)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    assignments = random.integers(0, len(centroids), size=size)
    vectors = centroids[assignments] + random.normal(0, 0.08, size=(size, dimension)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    duplicate_count = size // 20
    if duplicate_count:
        vectors[-duplicate_count:] = vectors[:duplicate_count] + random.normal(
            0, 0.001, size=(duplicate_count, dimension)
        ).astype(np.float32)
        vectors[-duplicate_count:] /= np.linalg.norm(vectors[-duplicate_count:], axis=1, keepdims=True)
    query_indices = random.choice(size, size=min(queries, size), replace=False)
    query_vectors = vectors[query_indices] + random.normal(
        0, 0.01, size=(len(query_indices), dimension)
    ).astype(np.float32)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    metadata = tuple(
        {
            "source_id": f"doc-{index // 5}",
            "scope_50": str(index % 2),
            "scope_10": str(index % 10),
            "scope_1": str(index % 100),
            "scope_01": str(index % 1000),
        }
        for index in range(size)
    )
    return Corpus(
        vectors,
        query_vectors,
        query_indices,
        metadata,
        stable_hash(
            {
                "generator": "clustered-v1",
                "size": size,
                "dimension": dimension,
                "queries": queries,
                "seed": seed,
                "centroids": 128,
                "noise": 0.08,
                "near_duplicate_fraction": 0.05,
            }
        ),
    )


def records(corpus: Corpus) -> list[Record]:
    return [
        Record(str(index), vector, f"benchmark record {index}", corpus.metadata[index])
        for index, vector in enumerate(corpus.vectors)
    ]


def exact_ids(
    corpus: Corpus,
    query: np.ndarray,
    limit: int,
    filters=None,
    *,
    candidate_indices: np.ndarray | None = None,
) -> list[str]:
    indices = (
        candidate_indices
        if candidate_indices is not None
        else np.arange(len(corpus.vectors))
    )
    if filters and candidate_indices is None:
        indices = np.asarray(
            [
                index
                for index in indices
                if all(corpus.metadata[int(index)].get(key) == str(value) for key, value in filters.items())
            ]
        )
    if not len(indices):
        return []
    scores = corpus.vectors[indices] @ query
    count = min(limit, len(indices))
    local = np.argpartition(-scores, count - 1)[:count]
    local = local[np.lexsort((indices[local], -scores[local]))]
    order = indices[local]
    return [str(value) for value in order]


def recall(hits: Sequence[Hit], expected: Sequence[str], limit: int) -> float:
    expected_set = set(expected[:limit])
    if not expected_set:
        return float(not hits)
    return len({hit.identifier for hit in hits[:limit]} & expected_set) / len(expected_set)
