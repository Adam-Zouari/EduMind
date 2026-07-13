# EduMind Project Summary

## What This Project Is

EduMind-AI is a local-first study assistant for students.

Core product flow:

1. The user uploads study material such as PDF, DOCX, images, audio, or video.
2. The OCR/extraction layer normalizes those files into text.
3. The RAG layer chunks, embeds, indexes, retrieves, and answers questions from that content.
4. The main user-facing entrypoint is a Streamlit app.
5. Experiments are tracked with MLflow to choose the best RAG stack.

The repo has been modernized into a package-first monorepo centered around `src/edumind/`.

## Main Architecture

Top-level structure:

- `src/edumind/common`
  Shared paths, schemas, config helpers, and package-wide utilities.
- `src/edumind/ocr`
  OCR and extraction subsystem for PDF, DOCX, images, web, audio, and video.
- `src/edumind/rag`
  Chunking, embeddings, vector-store logic, retrieval, and Ollama-based answer generation.
- `src/edumind/pipeline`
  OCR-to-RAG orchestration layer used by the UI and services.
- `apps`
  Main Streamlit app. The intended maintained UI is `apps/streamlit_app.py`.
- `services`
  FastAPI boundaries for OCR and RAG.
- `experiments/mlflow`
  Maintained staged experiment system for chunking, embeddings, vector DBs, retrieval, LLMs, and final bakeoff.
- `tests`
  Unit, integration, and experiment tests.
- `artifacts`
  Local generated state such as caches, MLflow outputs, vector stores, and benchmark results.

## Important Design Decisions

- The repo is one monorepo, not split into OCR/RAG subrepos.
- The project is package-first and meant to be installed with editable mode:
  - `pip install -e .[dev,ui,api,rag,experiments,ocr]`
- The main supported UI path is the direct Streamlit app, not alternate Streamlit variants.
- Microservices mode still exists through FastAPI services, but it is optional.
- OCR is an upstream producer of normalized content.
- RAG is the main retrieval and answering subsystem.
- Ollama is the local generation backend.
- Chroma is the currently integrated vector-store baseline in the product path.

## Current Status By Subsystem

### OCR

The OCR package has already gone through major cleanup and stabilization work.

Current state:

- package-first imports
- optional dependency guards
- safer media extraction paths
- caching support
- improved config usage
- tests and docs added
- OCR doc exists at `src/edumind/ocr/doc.md`

### RAG

The RAG package has been refactored and cleaned up without over-modularizing it.

Current state:

- typed ingest/query/answer contracts
- deterministic ids for sources and chunks
- cleaner vector-store behavior
- import-safe service boundaries
- `edumind.rag` remains the main public RAG package surface
- RAG doc exists at `src/edumind/rag/doc.md`

### Pipeline

The pipeline/orchestrator layer has also been cleaned up.

Current state:

- orchestrates OCR -> RAG flow
- intended shared application boundary for UI and services
- pipeline doc exists at `src/edumind/pipeline/doc.md`

### Apps

The apps folder was refined toward one maintained Streamlit app.

Current state:

- `apps/streamlit_app.py` is the main product UI
- deprecated parallel Streamlit product paths were removed
- app-related tests and docs were added

### Experiments

This is the subsystem currently most recently changed.

The experiment system is being redesigned into a staged English-only benchmark workflow:

1. `chunking`
2. `embedding`
3. `vectordb`
4. `retrieval`
5. `llm`
6. `final`

Goal:

- choose the best chunking strategy
- choose the best embedding model
- choose the best local vector DB
- choose the best retrieval strategy
- choose the best small local model
- then validate the best full stacks together

Important current experiment design:

- English only for now
- OCR is not meant to rerun during ordinary RAG sweeps
- MLflow outputs go under `artifacts/experiments/mlflow/`
- benchmark datasets now live under `data/evaluation/<dataset>/`

## Recent Experiment Refactor

The old experiment legacy path has been removed from code.

What changed:

- native benchmark loader now lives in `experiments/mlflow/benchmark.py`
- staged helper layer exists in:
  - `experiments/mlflow/harness.py`
  - `experiments/mlflow/stage_specs.py`
  - `experiments/mlflow/stage_utils.py`
  - `experiments/mlflow/vector_backends.py`
- stage runners now exist for:
  - `chunking_experiments`
  - `embedding_experiments`
  - `vectordb_experiments`
  - `retrieval_experiments`
  - `llm_experiments`
  - `final_experiments`
- the old flat fixture compatibility layer was removed:
  - deleted `experiments/mlflow/utils/fixtures.py`
  - deleted `data/evaluation/eval_queries.json`
  - deleted `data/evaluation/ground_truth.json`
  - deleted `data/evaluation/ocr_extraction_result.json`

Benchmark data now uses native split JSON files like:

- `data/evaluation/synthetic_regression/default.json`
- `data/evaluation/student_benchmark/dev.json`
- `data/evaluation/student_benchmark/holdout.json`
- `data/evaluation/challenge_benchmark/default.json`

## Important Product/Experiment Caveat

One key methodological point already discussed:

- Stage 1 chunking is evaluated with a fixed baseline embedding model and dense retrieval.
- That means the Stage 1 winner is not “best chunker in the abstract”.
- It is “best chunker under that fixed baseline setup”.
- The overall experiment design relies on later stages and the final bakeoff to validate whole-stack winners.

## Main Entry Commands

Install:

```bash
pip install -e .[dev,ui,api,rag,experiments,ocr]
```

Run UI:

```bash
python -m edumind.cli ui
```

Run OCR API:

```bash
python -m edumind.cli ocr-api
```

Run RAG API:

```bash
python -m edumind.cli rag-api
```

Run experiments:

```bash
python -m edumind.cli experiments
```

Examples:

```bash
edumind-experiments --suite smoke
edumind-experiments --suite all --dataset student_benchmark --resume
edumind-experiments --suite retrieval --dataset student_benchmark --resume
```

## Validation / Test Status

Recent experiment-side validation that passed:

- `python -m compileall experiments/mlflow tests/experiments`
- `PYTHONPATH=src python -m pytest -q tests/experiments tests/integration/test_module_smoke.py`

Result at last check:

- `21 passed, 2 skipped`

The two skips were optional smoke-test skips, not experiment failures.

## Current Git/Workspace State

There are still uncommitted changes in the workspace related mainly to the experiment refactor and benchmark migration.

Important current modified/untracked areas include:

- `experiments/mlflow/*`
- `data/evaluation/*`
- `tests/experiments/*`
- `README.md`
- `docs/experiments/README.md`
- `docs/setup/RUN_INSTRUCTIONS.md`
- `src/edumind/rag/vector_store.py`
- `pyproject.toml`

So if a new chat continues from here, it should assume the repo is mid-refactor but in a deliberate way, not in a broken random state.

## Good Next Questions For A New Chat

- review the experiment architecture and simplify any files that still feel too heavy
- validate that each experiment stage is methodologically sound
- inspect one stage runner in detail, especially chunking or retrieval
- help split the current git changes into clean commits
- run broader repo-wide checks and fix any remaining lint/type issues
- review benchmark data quality and whether the current English datasets are realistic enough

## Short Handoff Summary

This is a modernized Python monorepo for a student-study OCR + RAG application. OCR, RAG, pipeline, and app layers were cleaned up heavily. The current active area is the MLflow experiment system, which has been migrated from old flat fixtures to a staged native benchmark structure under `data/evaluation/` plus `experiments/mlflow/`. The repo is functional, but there are still uncommitted experiment-related changes that should be reviewed, committed cleanly, and possibly simplified further.
