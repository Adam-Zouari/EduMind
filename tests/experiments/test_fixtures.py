from edumind.common.paths import DATA_DIR
from experiments.mlflow.utils import load_evaluation_dataset


def test_evaluation_fixtures_exist() -> None:
    assert (DATA_DIR / "evaluation" / "eval_queries.json").exists()
    assert (DATA_DIR / "evaluation" / "ground_truth.json").exists()


def test_evaluation_dataset_loads() -> None:
    queries, documents = load_evaluation_dataset()

    assert queries
    assert documents
    assert all(query.query for query in queries)
    assert all(document.text for document in documents)
