# EduMind experiment program

Read `benchmark_manual.md` for the consolidated scientific and operational specification and `model_selection.md` for candidate evidence and include/exclude decisions.

## Boundary

All selection work lives in this directory. Production code under `src/edumind` provides the strategies that already exist in the application, while experiment-only alternatives such as BM25, RRF, rerankers, database adapters, datasets, statistics, and MLflow logging remain here. A benchmark never edits `config/base.yaml`; changing production requires a separate, explicit decision after reviewing evidence.

Each experiment directory contains `run.py`, `candidates.yaml`, and `doc.md`. The runners are ordinary Python scripts rather than an installed benchmark framework.

## Environment and preparation

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements/app.lock
python -m pip install -r requirements/benchmarks.lock
python -m pip install -e . --no-deps

python experiments/benchmarks/prepare.py smoke-fixtures
python experiments/benchmarks/prepare.py app-models
python experiments/benchmarks/prepare.py qasper
python experiments/benchmarks/prepare.py huggingface-models
python experiments/benchmarks/prepare.py extraction-models
python experiments/benchmarks/prepare.py ollama-models
python experiments/benchmarks/prepare.py vectordb
```

Every preparation step is explicit. Imports and application startup do not download models or start Docker.

`app-models` downloads only MiniLM, faster-whisper `base.en` int8 weights, and `qwen3:1.7b`. `huggingface-models` prepares seven embeddings, four rerankers, and HHEM. `extraction-models` prepares the selected OCR, complete-document, and ASR candidates. Both commands accept repeatable `--candidate` options and persist every completed model, so large downloads are resumable. `ollama-models` pulls every documented generation candidate. `vectordb` pulls and digest-locks the four server images. `qasper` creates the frozen text source manifests; `rag-selection` combines them with verified table/formula/mixed manifests. Large public extraction assets require a checksum/license plan passed to `prepare.py assets --plan PLAN.json`; standard extraction runs refuse assets without SHA-256 provenance.

## Runs and artifacts

MLflow is enabled by default at `sqlite:///mlflow.db`. Start its browser separately with:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

An invocation creates one parent run and one child per candidate. It logs the complete plan, dataset checksum, Git state, dependency-lock hashes, model revisions or Docker digests, seed, hardware, scalar metrics, 95% confidence bounds, `plan.json`, `provenance.json`, `summary.json`, candidate status files, and one `samples.parquet` per successful candidate. Candidate exceptions fail that child and remain visible. `--no-mlflow` is only a debugging option.

`smoke` uses tiny real paths and one repetition. It supports no quality or speed claim. `standard` runs all candidates on validation data with three measured repetitions and selects a Pareto set. `full` accepts a standard-run `summary.json` through `--shortlist`, runs finalists only, and provides stronger operational evidence. The final RAG locked test is run once only after blinded review.

## Scientific rules

Manifests freeze source, license, revision, split, preprocessing, sample IDs, and checksums. RAG splits are paper-isolated and evidence uses validated half-open offsets. Candidate and query order use seed 42. Standard/full retain per-sample observations and use 10,000 bootstrap resamples for 95% intervals. The reports make interval and Pareto comparisons; they do not perform an unused hypothesis-testing workflow.

Hard correctness/resource gates run before Pareto selection. There is no min-max normalization and no weighted overall score. When quality intervals overlap, prefer lower p95 latency, then memory, then storage. Smoke winners, import success, and automated generation metrics are not promotion evidence.

## Direct commands

```powershell
python experiments/benchmarks/extraction/image/run.py --profile standard
python experiments/benchmarks/rag/chunking_embedding/run.py --profile standard
python experiments/benchmarks/rag/retrieval/run.py --profile standard --embedding-summary EMBEDDING_SUMMARY_JSON
python experiments/benchmarks/rag/generation/run.py --profile standard
python experiments/benchmarks/rag/final/run.py --profile standard --retrieval-summary RETRIEVAL_SUMMARY_JSON --generation-summary GENERATION_SUMMARY_JSON
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml up -d
python experiments/benchmarks/vectordb/run.py --profile standard
```

See each experiment's `doc.md` for its hypothesis, formulas, candidates, promotion rules, limitations, and interpretation example.
