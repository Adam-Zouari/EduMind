# MLflow Experiments

This folder contains the maintained English-only staged benchmark system for EduMind.

## What lives here

- `run_all_experiments.py`
  Main suite runner for `smoke`, `chunking`, `embedding`, `vectordb`, `retrieval`, `llm`, `final`, and `all`.
- `benchmark.py`
  Shared benchmark contracts and dataset loading.
- `harness.py`
  Resume cache, leaderboard persistence, and stage artifact helpers.
- `stage_specs.py`
  Canonical candidate definitions for chunkers, embeddings, vector DBs, retrieval modes, and SLMs.
- `stage_utils.py`
  Shared chunk-building, retrieval evaluation, and answer-evaluation helpers.
- `chunking_experiments/`
  Stage 1 chunking sweep.
- `embedding_experiments/`
  Stage 2 embedding sweep.
- `vectordb_experiments/`
  Stage 3 vector database sweep.
- `retrieval_experiments/`
  Stage 4 retrieval strategy sweep.
- `llm_experiments/`
  Stage 5 SLM sweep.
- `final_experiments/`
  Stage 6 holdout and challenge confirmation bakeoff.

## Dataset layout

Benchmark manifests live under `data/evaluation/`:

- `synthetic_regression/`
  Fast smoke and regression benchmark.
- `student_benchmark/`
  Main English benchmark with `dev` and `holdout` splits.
- `challenge_benchmark/`
  Harder benchmark slice used for final confirmation.

Each dataset folder contains:

- `manifest.json`
  Dataset-level metadata and split-to-file mapping.
- one JSON file per split such as `dev.json` or `default.json`
  Native benchmark payloads with assets, snapshots, and questions.

## How the staged system works

1. `chunking`
   Holds embeddings/vector DB/retrieval fixed and compares chunkers.
2. `embedding`
   Holds chunking fixed and compares embedding models.
3. `vectordb`
   Holds chunking and embeddings fixed and compares vector backends.
4. `retrieval`
   Holds chunking, embeddings, and vector DB fixed and compares dense/BM25/hybrid retrieval.
5. `llm`
   Holds the promoted retrieval stacks fixed and compares local answer models.
6. `final`
   Re-evaluates the promoted full stacks on holdout plus challenge data.

The key rule is:

- change one variable at a time
- keep the rest of the pipeline fixed
- then validate the best combinations together

## Commands

Quick smoke run:

```bash
edumind-experiments --suite smoke
```

Run the full staged benchmark:

```bash
edumind-experiments --suite all --dataset student_benchmark --resume
```

Run one stage only:

```bash
edumind-experiments --suite retrieval --dataset student_benchmark --resume
```

Run a tiny synthetic pass:

```bash
python experiments/mlflow/run_all_experiments.py --test-mode
```

Useful flags:

- `--suite`
- `--dataset`
- `--resume`
- `--force`
- `--top-n`
- `--stage-limit`
- `--ui`

## Outputs

Tracked runtime state stays under `artifacts/experiments/mlflow/`:

- `mlflow.db`
  Local MLflow SQLite tracking database.
- `mlartifacts/`
  MLflow logged artifacts.
- `vector_store/`
  Experiment-local vector store persistence.
- `staged_results/`
  Resume cache, stage leaderboards, and promoted candidate summaries.

Each stage writes:

- candidate result cache JSON
- `leaderboard.json`
- `leaderboard.csv`
- `best_candidates.json`
- `stage_summary.md`

## Notes

- The experiment suite is local-first.
- `Chroma` works immediately through the maintained RAG runtime.
- `Qdrant` and `LanceDB` are scaffolded as experiment candidates and are marked `skipped` when their runtime is not available locally.
- The LLM stages require a reachable local Ollama instance and installed target models.
- No maintained runner should write generated files back into `experiments/mlflow/`.
