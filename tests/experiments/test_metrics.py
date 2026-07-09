import pytest

np = pytest.importorskip("numpy")

from experiments.mlflow.utils.evaluation import (  # noqa: E402
    compute_hit_rate_at_k,
    compute_map,
    compute_ndcg_at_k,
    compute_precision_at_k,
)


def test_retrieval_metric_examples() -> None:
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = ["doc2", "doc5", "doc7"]

    assert compute_precision_at_k(retrieved, relevant, 5) == pytest.approx(0.4)
    assert compute_hit_rate_at_k(retrieved, relevant, 5) == pytest.approx(1.0)
    assert compute_map(retrieved, relevant) > 0
    assert 0 <= compute_ndcg_at_k(retrieved, relevant, 5) <= 1
