"""Dense, BM25, reciprocal-rank fusion, and cross-encoder retrieval."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from .types import RetrievalHit
from .vector_store import VectorStore

RERANKER_MODELS = {
    "rrf-minilm-reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "rrf-qwen3-reranker": "Qwen/Qwen3-Reranker-0.6B",
}


class BM25Ranker:
    """In-memory BM25 used by exact benchmark and non-persistent retrieval paths."""

    def __init__(self, documents: Sequence[str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ModuleNotFoundError as exc:
            raise RuntimeError("rank_bm25 is required for BM25 retrieval") from exc
        self._documents = tuple(documents)
        self._model = BM25Okapi([_tokenize(document) for document in self._documents])

    def rank(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores = self._model.get_scores(_tokenize(query))
        return sorted(
            enumerate(float(score) for score in scores),
            key=lambda item: (-item[1], item[0]),
        )[:limit]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalHit]], *, limit: int, rrf_k: int = 60
) -> list[RetrievalHit]:
    scores: dict[str, float] = {}
    representatives: dict[str, RetrievalHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (rrf_k + rank)
            representatives.setdefault(hit.id, hit)
    ordered = sorted(scores, key=lambda item: (-scores[item], item))[:limit]
    return [
        replace(representatives[doc_id], score=scores[doc_id], rank=rank, retrieval_method="rrf")
        for rank, doc_id in enumerate(ordered, start=1)
    ]


class StoreRetrieval:
    def __init__(self, store: VectorStore, strategy: str = "rrf", rrf_k: int = 60) -> None:
        if strategy not in {"dense", "bm25", "rrf"}:
            raise ValueError(f"Unsupported retrieval strategy: {strategy}")
        self.store = store
        self.name = strategy
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        query_embedding: Sequence[float],
        limit: int,
        filters: Mapping[str, object] | None = None,
    ) -> list[RetrievalHit]:
        if self.name == "dense":
            return self.store.query_dense(query_embedding, top_k=limit, filter_metadata=filters)
        if self.name == "bm25":
            return self.store.query_lexical(query, top_k=limit, filter_metadata=filters)
        return reciprocal_rank_fusion(
            [
                self.store.query_dense(query_embedding, top_k=limit, filter_metadata=filters),
                self.store.query_lexical(query, top_k=limit, filter_metadata=filters),
            ],
            limit=limit,
            rrf_k=self.rrf_k,
        )


class CrossEncoderReranker:
    def __init__(self, model_name: str, *, device: str = "cpu", revision: str = "main") -> None:
        self.model_name = model_name
        self.device = device
        self.revision = revision
        self.name = f"cross-encoder:{model_name}"
        self._model = None

    def rerank(self, query: str, hits: Sequence[RetrievalHit], limit: int) -> list[RetrievalHit]:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ModuleNotFoundError as exc:
                raise RuntimeError("sentence-transformers is required for reranking") from exc
            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
                revision=self.revision,
                local_files_only=True,
            )
        assert self._model is not None
        scores = self._model.predict([(query, hit.document) for hit in hits])
        ordered = sorted(
            zip(hits, scores, strict=True), key=lambda item: float(item[1]), reverse=True
        )[:limit]
        return [
            replace(hit, score=float(score), rank=rank, retrieval_method=self.name)
            for rank, (hit, score) in enumerate(ordered, start=1)
        ]


def base_retrieval_strategy(strategy: str) -> str:
    """Return the first-stage retriever used by a named production stack."""
    return "rrf" if strategy in RERANKER_MODELS else strategy


def build_reranker(
    strategy: str, *, revision: str | None, device: str = "cpu"
) -> CrossEncoderReranker | None:
    """Build a lazy, local-only reranker for a registered retrieval stack."""
    model_name = RERANKER_MODELS.get(strategy)
    if model_name is None:
        return None
    if not revision or revision in {"main", "unpinned"}:
        raise ValueError(f"{strategy} requires an immutable reranker revision")
    return CrossEncoderReranker(model_name, device=device, revision=revision)


def _tokenize(text: str) -> list[str]:
    return [token for token in text.casefold().split() if token]
