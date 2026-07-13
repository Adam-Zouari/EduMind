from __future__ import annotations

from experiments.mlflow.harness import (
    StageCandidateResult,
    build_candidate_hash,
    load_cached_candidate_result,
    save_cached_candidate_result,
)


def test_stage_candidate_result_round_trip() -> None:
    result = StageCandidateResult(
        stage="chunking",
        dataset_name="synthetic_regression",
        dataset_version="0.1.0",
        split="default",
        candidate_name="token_256_32",
        candidate_config={"name": "token_256_32"},
        metrics={"chunk_recall_at_5": 0.5},
    )

    saved_path = save_cached_candidate_result(result)
    loaded_result = load_cached_candidate_result(
        stage="chunking",
        dataset_name="synthetic_regression",
        dataset_version="0.1.0",
        split="default",
        candidate_config={"name": "token_256_32"},
    )

    assert saved_path.exists()
    assert loaded_result is not None
    assert loaded_result.candidate_name == result.candidate_name
    assert loaded_result.metrics == result.metrics


def test_candidate_hash_changes_with_config() -> None:
    left = build_candidate_hash(
        stage="embedding",
        dataset_name="student_benchmark",
        dataset_version="0.1.0",
        split="dev",
        candidate_config={"model_name": "a"},
    )
    right = build_candidate_hash(
        stage="embedding",
        dataset_name="student_benchmark",
        dataset_version="0.1.0",
        split="dev",
        candidate_config={"model_name": "b"},
    )

    assert left != right
