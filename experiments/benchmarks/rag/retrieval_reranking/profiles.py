"""Frozen retrieval/reranker candidate contracts."""

from __future__ import annotations

from dataclasses import dataclass


RETRIEVERS = ("dense", "bm25", "rrf")
RERANKER_MODELS = {
    "gte-modernbert": "Alibaba-NLP/gte-reranker-modernbert-base",
    "ettin-150m": "cross-encoder/ettin-reranker-150m-v1",
    "ettin-400m": "cross-encoder/ettin-reranker-400m-v1",
    "ettin-1b": "cross-encoder/ettin-reranker-1b-v1",
}
RERANKERS = ("none", *RERANKER_MODELS)


def development_candidates() -> tuple[str, ...]:
    return tuple(f"{retriever}|{reranker}" for retriever in RETRIEVERS for reranker in RERANKERS)


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
