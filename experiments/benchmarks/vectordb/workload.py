"""Identical deterministic corpora and exact NumPy oracle."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from edumind.common.artifacts import stable_hash

from .adapters import Hit, Record
from .protocol import SyntheticSettings


@dataclass(frozen=True)
class Corpus:
    vectors: np.ndarray
    queries: np.ndarray
    query_indices: np.ndarray
    metadata: tuple[Mapping[str, str], ...]
    fingerprint: str
    identifiers: tuple[str, ...] = ()
    texts: tuple[str, ...] = ()
    query_ids: tuple[str, ...] = ()
    source_groups: tuple[str, ...] = ()
    provenance: Mapping[str, object] | None = None

    @property
    def ids(self):
        return self.identifiers or tuple(
            str(index) for index in range(len(self.vectors))
        )

    @property
    def content_checksum(self):
        return stable_hash(
            {
                "vectors": _array_identity(self.vectors),
                "queries": _array_identity(self.queries),
                "query_indices": _array_identity(self.query_indices),
                "ids": self.ids,
                "texts": self.texts,
                "metadata": self.metadata,
                "query_ids": self.query_ids,
                "source_groups": self.source_groups,
                "provenance": self.provenance,
            }
        )


def _array_identity(value):
    array = np.ascontiguousarray(value)
    return {
        "shape": array.shape,
        "dtype": array.dtype.str,
        "sha256": sha256(array.tobytes()).hexdigest(),
    }


def clustered(
    size: int,
    dimension: int,
    queries: int,
    seed: int,
    *,
    settings: SyntheticSettings,
) -> Corpus:
    random = np.random.default_rng(seed + size + dimension)
    centroids = random.normal(size=(settings.centroid_count, dimension)).astype(
        np.float32
    )
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    assignments = random.integers(0, len(centroids), size=size)
    vectors = centroids[assignments] + random.normal(
        0, settings.cluster_noise, size=(size, dimension)
    ).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    duplicate_count = int(size * settings.near_duplicate_fraction)
    if duplicate_count:
        vectors[-duplicate_count:] = vectors[:duplicate_count] + random.normal(
            0, settings.near_duplicate_noise, size=(duplicate_count, dimension)
        ).astype(np.float32)
        vectors[-duplicate_count:] /= np.linalg.norm(
            vectors[-duplicate_count:], axis=1, keepdims=True
        )
    query_indices = random.choice(size, size=min(queries, size), replace=False)
    query_vectors = vectors[query_indices] + random.normal(
        0, settings.query_noise, size=(len(query_indices), dimension)
    ).astype(np.float32)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)
    metadata = tuple(
        {
            "source_id": f"doc-{index // settings.records_per_document}",
            **{
                name: str(index % divisor)
                for name, divisor in settings.metadata_divisors.items()
            },
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
                "centroids": settings.centroid_count,
                "records_per_document": settings.records_per_document,
                "noise": settings.cluster_noise,
                "near_duplicate_fraction": settings.near_duplicate_fraction,
                "near_duplicate_noise": settings.near_duplicate_noise,
                "query_noise": settings.query_noise,
                "metadata_divisors": dict(settings.metadata_divisors),
            }
        ),
    )


def records(corpus: Corpus) -> list[Record]:
    return [
        Record(
            corpus.ids[index],
            vector,
            corpus.texts[index] if corpus.texts else f"benchmark record {index}",
            corpus.metadata[index],
        )
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
    if limit <= 0:
        return []
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
                if all(
                    corpus.metadata[int(index)].get(key) == value
                    for key, value in filters.items()
                )
            ]
        )
    if not len(indices):
        return []
    scores = cosine_scores(corpus.vectors[indices], query)
    count = min(limit, len(indices))
    boundary = np.partition(scores, len(scores) - count)[len(scores) - count]
    local = np.flatnonzero(scores >= boundary)
    local = local[np.lexsort((indices[local], -scores[local]))][:count]
    order = indices[local]
    return [corpus.ids[int(value)] for value in order]


def recall(hits: Sequence[Hit], expected: Sequence[str], limit: int) -> float | None:
    expected_set = set(expected[:limit])
    if not expected_set:
        return None
    return len({hit.identifier for hit in hits[:limit]} & expected_set) / len(
        expected_set
    )


def cosine_scores(vectors, query):
    """Canonical float64 cosine evaluator; backend rounded scores are not an oracle."""
    vectors = np.asarray(vectors, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    if (
        not np.isfinite(vectors).all()
        or not np.isfinite(query).all()
        or np.any(norms <= 0)
    ):
        raise ValueError("Exact-distance inputs must be finite and nonzero")
    return (vectors @ query) / norms


def validate_hits(hits, trusted, limit, filters=None):
    """Execution validity and filter correctness are deliberately separate."""
    if not isinstance(hits, (list, tuple)) or len(hits) > limit:
        raise ValueError("Malformed search response or excess hits")
    seen = set()
    for hit in hits:
        if (
            not isinstance(hit, Hit)
            or hit.identifier not in trusted
            or hit.identifier in seen
        ):
            raise ValueError("Search contains duplicate/unknown IDs or malformed hits")
        if (
            isinstance(hit.score, bool)
            or not isinstance(hit.score, (int, float, np.number))
            or not math.isfinite(hit.score)
        ):
            raise ValueError("Search requires finite scores")
        seen.add(hit.identifier)
    return bool(filters) and any(
        any(trusted[hit.identifier].get(key) != value for key, value in filters.items())
        for hit in hits
    )


def tie_reference(corpus, query, limit=20, filters=None):
    indices = [
        i
        for i, row in enumerate(corpus.metadata)
        if all(row.get(key) == value for key, value in (filters or {}).items())
    ]
    count = min(limit, len(indices))
    if not count:
        return {
            "effective_count": 0,
            "cutoff_distance": None,
            "strict_ids": [],
            "boundary_ids": [],
            "boundary_slots": 0,
            "tolerance": 0.0,
        }
    distances = 1.0 - cosine_scores(corpus.vectors[indices], query)
    cutoff = float(np.partition(distances, count - 1)[count - 1])
    strict = [corpus.ids[indices[i]] for i in np.flatnonzero(distances < cutoff)]
    boundary = [corpus.ids[indices[i]] for i in np.flatnonzero(distances == cutoff)]
    return {
        "effective_count": count,
        "cutoff_distance": cutoff,
        "strict_ids": strict,
        "boundary_ids": boundary,
        "boundary_slots": count - len(strict),
        "tolerance": 0.0,
    }


def tie_recall(hits, reference, limit=20):
    count = reference["effective_count"]
    if not count:
        return {"value": None, "status": "inapplicable", **reference}
    returned = {hit.identifier for hit in hits[:limit]}
    strict = len(returned & set(reference["strict_ids"]))
    boundary = min(
        reference["boundary_slots"], len(returned & set(reference["boundary_ids"]))
    )
    return {
        **reference,
        "strict_credit": strict,
        "boundary_credit": boundary,
        "value": (strict + boundary) / count,
        "status": "scored",
    }
