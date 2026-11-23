"""Identical deterministic corpora and exact NumPy oracle."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from edumind.common.artifacts import stable_hash

from .adapters import Hit, Record
from .adapters.base import filter_values
from .protocol import SyntheticSettings

ALL_SESSION = "all-session"
SELECTED_FILES = "selected-files"
SCOPES = (ALL_SESSION, SELECTED_FILES)
SEARCH_LIMIT = 20


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
    ids = corpus.ids
    return [
        Record(
            ids[index],
            vector,
            corpus.texts[index] if corpus.texts else f"benchmark record {index}",
            corpus.metadata[index],
        )
        for index, vector in enumerate(corpus.vectors)
    ]


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


def validate_response(hits, trusted, limit):
    """Check the response envelope against the globally trusted ID inventory."""
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


def matches_filters(metadata, filters):
    return all(
        metadata.get(key) in filter_values(key, value)
        for key, value in (filters or {}).items()
    )


def validate_hits(hits, trusted, limit, filters=None):
    """Used by standalone correctness probes; sessions use both inventories."""
    validate_response(hits, trusted, limit)
    return bool(filters) and any(
        not matches_filters(trusted[hit.identifier], filters) for hit in hits
    )


def allowed_ids(corpus, filters=None):
    return [
        identifier
        for identifier, metadata in zip(corpus.ids, corpus.metadata, strict=True)
        if matches_filters(metadata, filters)
    ]


def classify_hits(
    hits, global_ids, session_ids, allowed, *, filtered, limit=SEARCH_LIMIT
):
    validate_response(hits, global_ids, limit)
    ids = [hit.identifier for hit in hits]
    outside_session = [
        identifier for identifier in ids if identifier not in session_ids
    ]
    outside_selection = [identifier for identifier in ids if identifier not in allowed]
    return {
        "outside_session_ids": outside_session,
        "outside_selection_ids": outside_selection if filtered else [],
        "session_isolation_violation": bool(outside_session),
        "filter_violation": bool(outside_selection) if filtered else None,
    }


def global_inventory(corpora):
    inventory = {}
    for corpus in corpora.values():
        for identifier, text, metadata, vector in zip(
            corpus.ids, corpus.texts, corpus.metadata, corpus.vectors, strict=True
        ):
            identity = (text, metadata, _array_identity(vector))
            if inventory.setdefault(identifier, identity) != identity:
                raise ValueError(
                    f"Conflicting canonical chunk identity across sessions: {identifier}"
                )
    return inventory


def query_filters(session, query_id, condition):
    if condition == ALL_SESSION:
        return None
    if condition != SELECTED_FILES:
        raise ValueError(f"Unknown search condition: {condition}")
    sources = session["selected_sources"][query_id]
    return {"source_id": sources[0] if len(sources) == 1 else tuple(sources)}


def freeze_session_search(session, corpus, *, repetitions, pair_offset):
    """Prepare serial order and independent exact references; no backend runs."""
    selected = session["selected_sources"]
    ordinal = {
        qid: pair_offset + i
        for i, qid in enumerate(qid for qid in corpus.query_ids if qid in selected)
    }
    schedule = []
    for attempt in range(1, repetitions + 1):
        for qid in corpus.query_ids:
            scopes = SCOPES if qid in selected else (ALL_SESSION,)
            if qid in selected and (ordinal[qid] + attempt - 1) % 2:
                scopes = tuple(reversed(scopes))
            schedule.extend(
                {"query_id": qid, "attempt": attempt, "condition": scope}
                for scope in scopes
            )
    references = {
        qid: {
            scope: tie_reference(
                corpus, vector, SEARCH_LIMIT, query_filters(session, qid, scope)
            )
            for scope in (SCOPES if qid in selected else (ALL_SESSION,))
        }
        for qid, vector in zip(corpus.query_ids, corpus.queries, strict=True)
    }
    return schedule, references


def paired_search_schedule(sessions):
    """Return the frozen logical attempts shared by both sides of the comparison."""
    sides = {scope: set() for scope in SCOPES}
    for session in sessions:
        for row in session["schedule"]:
            qid = row["query_id"]
            if qid not in session["selected_sources"]:
                continue
            identity = (session["id"], qid, row["attempt"])
            side = sides[row["condition"]]
            if identity in side:
                raise ValueError(
                    f"Duplicate paired search schedule identity: {identity}"
                )
            side.add(identity)
    if sides[ALL_SESSION] != sides[SELECTED_FILES]:
        raise ValueError("Frozen paired search schedule differs between sides")
    return sides[ALL_SESSION]


def search_rows(session, corpus):
    """Common frozen identities used by online dispatch and offline replay."""
    scopes = {}
    reference_checksums = {}
    paired = session["selected_sources"]
    for position, request in enumerate(session["schedule"], 1):
        qid, scope = request["query_id"], request["condition"]
        selection = tuple(paired[qid]) if scope == SELECTED_FILES else ()
        if selection not in scopes:
            filters = query_filters(session, qid, scope)
            ids = frozenset(corpus.ids if not filters else allowed_ids(corpus, filters))
            scopes[selection] = filters, ids, stable_hash(sorted(ids))
        filters, ids, allowed_checksum = scopes[selection]
        if (qid, scope) not in reference_checksums:
            reference_checksums[qid, scope] = stable_hash(
                session["exact_references"][qid][scope]
            )
        yield (
            {
                **request,
                "selected_source_ids": list(selection) if selection else None,
                "allowed_set_checksum": allowed_checksum,
                "eligible_chunk_count": len(ids),
                "exact_reference_checksum": reference_checksums[qid, scope],
                "session_id": session["id"],
                "session_group_id": session["session_group_id"],
                "schedule_position": position,
                "paired": qid in paired,
                "operation": "search",
                "quality_designated": request["attempt"] == 1,
                "corpus_checksum": corpus.fingerprint,
            },
            filters,
            ids,
        )


def tie_reference(corpus, query, limit=20, filters=None):
    indices = [
        i for i, row in enumerate(corpus.metadata) if matches_filters(row, filters)
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
    ids = corpus.ids
    strict = [ids[indices[i]] for i in np.flatnonzero(distances < cutoff)]
    boundary = [ids[indices[i]] for i in np.flatnonzero(distances == cutoff)]
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
