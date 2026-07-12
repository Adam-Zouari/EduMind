# Experiment Workflow

This repository keeps experiment code in `experiments/mlflow/` and routes local MLflow state into
`artifacts/experiments/mlflow/`.

## What belongs here

- maintained runners for retrieval, embeddings, chunking, and LLMs
- evaluation, fixture, GPU, and MLflow helper utilities
- one local implementation guide at `experiments/mlflow/README.md`

## Main entrypoint

Quick run:

```bash
python -m edumind.cli experiments
```

Advanced flags:

```bash
python experiments/mlflow/run_all_experiments.py --test-mode
python experiments/mlflow/run_all_experiments.py --skip-llm
python experiments/mlflow/run_all_experiments.py --only retrieval llm --ui
```

## Data and outputs

- evaluation fixtures: `data/evaluation/`
- MLflow SQLite database: `artifacts/experiments/mlflow/mlflow.db`
- MLflow artifacts: `artifacts/experiments/mlflow/mlartifacts/`
- experiment vector-store state: `artifacts/experiments/mlflow/vector_store/`

## Notes for contributors

- keep generated runs and large artifacts out of Git
- keep pseudo-tests and one-off diagnostics out of `experiments/mlflow/`
- make experiment scripts runnable from the repo root
- avoid coupling experiment code to Streamlit or service entrypoints
