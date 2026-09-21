"""Frozen retrieval/reranker candidate contracts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


_CANDIDATE_PATH = Path(__file__).with_name("candidates.yaml")


def _candidate_identities() -> tuple[tuple[str, ...], dict[str, str]]:
    payload = yaml.safe_load(_CANDIDATE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Retrieval candidate registry must be an object")
    retrievers = payload.get("retrievers")
    rerankers = payload.get("rerankers")
    if not isinstance(retrievers, Mapping) or not isinstance(rerankers, Mapping):
        raise ValueError("Retrieval candidate registry requires retrievers and rerankers")
    expected_retrievers = {
        "dense": "exact-cosine",
        "bm25": "rank-bm25",
        "rrf": "reciprocal-rank-fusion",
    }
    if set(retrievers) != set(expected_retrievers):
        raise ValueError("Retrieval candidate registry must define dense, bm25, and rrf")
    for alias, backend in expected_retrievers.items():
        value = retrievers[alias]
        if not isinstance(value, Mapping) or value.get("backend") != backend:
            raise ValueError(f"Retrieval candidate {alias!r} has an invalid backend")
    if "none" not in rerankers:
        raise ValueError("Retrieval candidate registry requires the no-reranker control")
    models: dict[str, str] = {}
    for alias, value in rerankers.items():
        if alias == "none":
            continue
        if not isinstance(alias, str) or not isinstance(value, Mapping):
            raise ValueError("Retrieval reranker registry is malformed")
        model_id = value.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"Reranker {alias!r} requires a model_id")
        models[alias] = model_id
    return tuple(expected_retrievers), models


RETRIEVERS, RERANKER_MODELS = _candidate_identities()
RERANKERS = ("none", *RERANKER_MODELS)


@dataclass(frozen=True)
class RetrievalCandidate:
    retriever: str
    reranker: str

    @property
    def identifier(self) -> str:
        return f"{self.retriever}|{self.reranker}"

    @property
    def owner_identifier(self) -> str:
        return f"{self.retriever}|none"

    @property
    def reranker_model(self) -> str | None:
        return RERANKER_MODELS.get(self.reranker)

    @property
    def model_backed(self) -> bool:
        return self.retriever in {"dense", "rrf"} or self.reranker_model is not None


def parse_candidate(value: str) -> RetrievalCandidate:
    parts = value.split("|")
    if len(parts) != 2 or parts[0] not in RETRIEVERS or parts[1] not in RERANKERS:
        raise ValueError(f"Malformed retrieval/reranker candidate: {value}")
    return RetrievalCandidate(parts[0], parts[1])


def owner_first(candidates: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(candidates, key=lambda value: parse_candidate(value).reranker != "none")
    )


def validation_candidates(
    finalists: tuple[str, ...], declared: tuple[str, ...]
) -> tuple[str, ...]:
    """Add no-reranker controls while preserving a deterministic owner-first order."""

    parsed = tuple(parse_candidate(candidate) for candidate in finalists)
    requested = set(finalists) | {
        candidate.owner_identifier
        for candidate in parsed
        if candidate.reranker != "none"
    }
    return owner_first(tuple(candidate for candidate in declared if candidate in requested))


def required_reranker_models(candidates: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                model
                for value in candidates
                if (model := parse_candidate(value).reranker_model) is not None
            }
        )
    )
