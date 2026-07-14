from __future__ import annotations

from edumind.rag.retrieval import reciprocal_rank_fusion
from edumind.rag.types import RetrievalHit


def _hit(identifier: str, rank: int, method: str) -> RetrievalHit:
    return RetrievalHit(identifier, identifier, {}, 100.0 - rank, rank, method)


def test_rrf_uses_ranks_not_incompatible_raw_scores() -> None:
    dense = [_hit("a", 1, "dense"), _hit("b", 2, "dense")]
    lexical = [_hit("b", 1, "bm25"), _hit("c", 2, "bm25")]
    result = reciprocal_rank_fusion([dense, lexical], limit=3, rrf_k=60)
    assert result[0].id == "b"
    assert result[0].retrieval_method == "rrf"
    assert result[0].score < 1
