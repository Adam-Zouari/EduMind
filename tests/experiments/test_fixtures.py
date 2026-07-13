from edumind.common.paths import DATA_DIR
from experiments.mlflow.benchmark import load_benchmark_dataset, prepare_benchmark_dataset


def test_native_benchmark_split_files_exist() -> None:
    assert (DATA_DIR / "evaluation" / "synthetic_regression" / "default.json").exists()
    assert (DATA_DIR / "evaluation" / "student_benchmark" / "dev.json").exists()
    assert (DATA_DIR / "evaluation" / "student_benchmark" / "holdout.json").exists()
    assert (DATA_DIR / "evaluation" / "challenge_benchmark" / "default.json").exists()


def test_benchmark_manifests_exist() -> None:
    assert (DATA_DIR / "evaluation" / "synthetic_regression" / "manifest.json").exists()
    assert (DATA_DIR / "evaluation" / "student_benchmark" / "manifest.json").exists()
    assert (DATA_DIR / "evaluation" / "challenge_benchmark" / "manifest.json").exists()


def test_flat_evaluation_files_are_removed() -> None:
    assert not (DATA_DIR / "evaluation" / "eval_queries.json").exists()
    assert not (DATA_DIR / "evaluation" / "ground_truth.json").exists()
    assert not (DATA_DIR / "evaluation" / "ocr_extraction_result.json").exists()


def test_native_benchmark_dataset_loads() -> None:
    dataset = load_benchmark_dataset("student_benchmark", split="dev")

    assert dataset.questions
    assert dataset.snapshots
    assert dataset.assets
    assert all(question.gold_answer for question in dataset.questions)
    assert all(question.relevant_source_ids for question in dataset.questions)
    assert dataset.metadata["split_file"] == "dev.json"


def test_prepare_benchmark_dataset_reports_summary() -> None:
    report = prepare_benchmark_dataset("synthetic_regression", split="default")

    assert report["dataset_name"] == "synthetic_regression"
    assert report["num_questions"] > 0
    assert "pdf" in report["modalities"]
