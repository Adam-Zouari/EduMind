## MLflow Experiments

This folder now contains only maintained experiment code.

### What stays here

- `run_all_experiments.py`: main local runner
- `mlflow_config.py`: shared MLflow path and experiment setup
- `chunking_experiments/`: chunking strategy comparisons
- `embedding_experiments/`: embedding model comparisons
- `retrieval_experiments/`: dense vs hybrid retrieval comparisons
- `llm_experiments/`: Ollama answer-generation comparisons
- `utils/`: shared evaluation, fixture, GPU, and MLflow logging helpers

### What no longer stays here

- generated MLflow state
- backup files
- pseudo-tests and manual diagnostics
- historical summaries and one-off setup notes

Generated runtime state lives under `artifacts/experiments/mlflow/`.

### Run the maintained suite

```bash
edumind-experiments --test-mode
edumind-experiments --skip-llm
edumind-experiments --only retrieval llm --ui
```

Equivalent direct runner:

```bash
python experiments/mlflow/run_all_experiments.py --test-mode
```

### Output locations

- MLflow SQLite database: `artifacts/experiments/mlflow/mlflow.db`
- MLflow artifacts: `artifacts/experiments/mlflow/mlartifacts/`
- experiment-local vector-store state: `artifacts/experiments/mlflow/vector_store/`

### Notes

- Evaluation fixtures stay in `data/evaluation/`.
- The LLM runner requires a reachable Ollama instance with the target models installed.
- No maintained runner should generate files inside `experiments/mlflow/`.
