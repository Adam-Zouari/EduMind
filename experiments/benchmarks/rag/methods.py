"""Experiment-only BM25, rank fusion, and reranking implementations."""

from __future__ import annotations

import os
import pickle
import re
from collections.abc import Sequence

RERANKER_MAX_LENGTH = {
    "Alibaba-NLP/gte-reranker-modernbert-base": 8192,
    "cross-encoder/ettin-reranker-150m-v1": 7999,
    "cross-encoder/ettin-reranker-400m-v1": 7999,
    "cross-encoder/ettin-reranker-1b-v1": 7999,
}


class BM25:
    def __init__(self, documents: Sequence[str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install requirements/benchmarks.lock for BM25") from exc
        self.model = BM25Okapi([_tokens(text) for text in documents])

    @property
    def storage_bytes(self) -> int:
        return len(pickle.dumps(self.model, protocol=pickle.HIGHEST_PROTOCOL))

    def rank(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores = self.model.get_scores(_tokens(query))
        return sorted(enumerate(map(float, scores)), key=lambda row: (-row[1], row[0]))[:limit]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], limit: int, rrf_k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:limit]


class Reranker:
    def __init__(self, model: str, revision: str, model_path: str) -> None:
        self.model_name = model
        self.revision = revision
        self.model_path = model_path
        self.model = None

    def rank(self, query: str, documents: Sequence[str]) -> list[int]:
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                self.model_path,
                device=os.environ.get("EDUMIND_BENCHMARK_RERANKER_DEVICE", "cpu"),
                local_files_only=True,
                trust_remote_code=False,
                max_length=RERANKER_MAX_LENGTH[self.model_name],
            )
        scores = self.model.predict([(query, document) for document in documents])
        return sorted(range(len(documents)), key=lambda index: (-float(scores[index]), index))


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())
