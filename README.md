# EduMind

EduMind extracts local study material, indexes it in a Chroma HTTP server, and answers questions through Ollama with numbered evidence citations. The application defaults are provisional; alternative extractors, chunkers, embeddings, retrieval strategies, language models, and vector servers are selected only through experiments.

For every prerequisite, model, dataset location, server command, and recommended experiment order, use [guide.md](guide.md).

## Layout

```text
src/edumind/              production extraction, RAG, config, and pipeline
apps/                     complete Streamlit application
config/base.yaml          single production configuration
experiments/benchmarks/   direct experiments, metrics, candidates, and docs
infrastructure/chroma.yml provisional production Chroma server
requirements/             separate app and benchmark dependency pins
```

There is no application API layer and no benchmark package under `src`. Streamlit constructs the pipeline directly. Production uses Chroma server dense retrieval only; experiment-only BM25, RRF, rerankers, and alternative database clients do not become runtime dependencies automatically.

## Install once, edit directly

Use a fresh virtual environment. In Git Bash, activate it before invoking `python -m pip`; this avoids installing into the global Python environment.

```bash
python -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/app.lock
python -m pip install -r requirements/benchmarks.lock
python -m pip install -e . --no-deps
```

Editable installation means changes under `src`, `apps`, and `experiments` are visible immediately. There is no rebuild step and no Ruff, MyPy, coverage, or wheel gate.

System programs still required by the selected production path are Docker Desktop, Ollama, FFmpeg, and Tesseract 5.

## Run the provisional application

Prepare only the three provisional application models first, then start Chroma and Streamlit:

```bash
python experiments/benchmarks/prepare.py app-models
docker compose -f infrastructure/chroma.yml up -d
streamlit run apps/streamlit_app.py
```

The defaults in `config/base.yaml` are token chunks of 256 with 32 overlap, `all-MiniLM-L6-v2`, dense top-5 retrieval under a 2,048-token evidence budget, and `qwen3:1.7b`.

## Run experiments

MLflow is the run browser and defaults to local SQLite/artifacts:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
python experiments/benchmarks/prepare.py qasper
python experiments/benchmarks/rag/chunking_embedding/run.py --profile standard
python experiments/benchmarks/rag/retrieval/run.py --profile standard --embedding-summary EMBEDDING_SUMMARY_JSON
python experiments/benchmarks/rag/generation/run.py --profile standard
```

For the real four-server database benchmark:

```bash
python experiments/benchmarks/prepare.py vectordb
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml up -d
python experiments/benchmarks/vectordb/run.py --profile smoke
python experiments/benchmarks/vectordb/run.py --profile standard
```

The benchmark compares Chroma, Qdrant, Weaviate, and PostgreSQL/pgvector. Smoke only validates the real path. Standard/full results retain every query, confidence intervals, exact NumPy recall, filter and replacement conformance, concurrency measurements, resource measurements, and environment provenance.

Start with [the benchmark overview](experiments/benchmarks/doc.md) and then read the `doc.md` beside the experiment you plan to run.

## Minimal checks

Only metric and dataset invariants are automated:

```bash
python -m pytest tests/test_benchmark_metrics.py tests/test_benchmark_datasets.py
```

Real smoke scripts are the integration checks. A failed optional candidate remains a recorded failed run; it is never silently converted into a successful benchmark.
