# Experiment Workflow

This repository keeps experiment code in `experiments/mlflow/` and routes local MLflow state into `artifacts/mlflow/`.

## What belongs here

- benchmark runners for retrieval, embeddings, chunking, and LLMs
- evaluation utilities and metrics helpers
- curated notes that explain experiment outcomes

## Main entrypoint

Quick run:

```bash
python -m edumind.cli experiments
```

Advanced flags:

```bash
python experiments/mlflow/run_all_experiments.py --full --ui
python experiments/mlflow/run_all_experiments.py --skip-llm
```

## Data and outputs

- evaluation fixtures: `data/evaluation/`
- MLflow SQLite database: `artifacts/mlflow/mlflow.db`
- MLflow artifacts: `artifacts/mlflow/mlartifacts/`

## Notes for contributors

- keep generated runs and large artifacts out of Git
- prefer curated markdown summaries over raw binary outputs
- make experiment scripts runnable from the repo root
- avoid coupling experiment code to Streamlit or service entrypoints
