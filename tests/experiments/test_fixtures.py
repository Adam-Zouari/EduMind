from edumind.common.paths import DATA_DIR


def test_evaluation_fixtures_exist() -> None:
    assert (DATA_DIR / "evaluation" / "eval_queries.json").exists()
    assert (DATA_DIR / "evaluation" / "ground_truth.json").exists()
