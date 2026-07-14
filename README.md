# EduMind

EduMind is a local, English-first extraction and retrieval-augmented generation system whose configurable production strategies are selected through reproducible benchmarks. It targets one user on an i7-12700H, 32 GB RAM, and RTX 3050 4 GB machine.

## What is included

- Image, PDF, DOCX, audio, and video text extraction through typed, revisioned contracts
- Offset-preserving chunking, contract-aware embeddings, dense/BM25/RRF retrieval, reranking, and cited Ollama generation
- A thin Streamlit application and local-only extraction/RAG APIs
- Technology-neutral extraction, RAG, vector-system, human-review, and reporting benchmarks
- Deterministic network-free smoke fixtures plus manifest-driven standard/full datasets

Web extraction, structured table extraction, formulas, and dedicated form parsing are intentionally excluded. Flattened text present in PDF/DOCX sources may survive extraction without structural guarantees.

## Repository map

```text
src/edumind/extraction/   production extraction contracts and adapters
src/edumind/rag/          production indexing, retrieval, and generation
src/edumind/pipeline/     typed end-to-end orchestration
src/edumind/app/          testable application actions and state
src/edumind/benchmarks/   benchmark engine, metrics, manifests, and reports
apps/                     thin Streamlit entry point
services/                 local FastAPI adapters
experiments/benchmarks/   benchmark documentation and source wrapper
data/benchmarks/          committed smoke fixtures and versioned manifests
```

## Installation

Python 3.10 or newer is supported; Python 3.11 is recommended for a fresh environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui,api,rag,extraction,asr,benchmarks]"
```

Dependencies are split so normal runtime paths do not require MLflow or every extraction engine. `requirements.lock` records the validated environment snapshot. External tools such as Tesseract, FFmpeg, Ollama, datasets, and model weights are checked by preflight and are never downloaded at import time.

## Public commands

```powershell
edumind ui
edumind extraction-api
edumind rag-api

edumind benchmark preflight
edumind benchmark prepare qasper
edumind benchmark prepare assets ASSET_PLAN_JSON
edumind benchmark prepare extraction-models
edumind benchmark prepare huggingface-models
edumind benchmark prepare ollama-models
edumind benchmark extraction image
edumind benchmark extraction all
edumind benchmark rag chunking-embedding
edumind benchmark rag retrieval
edumind benchmark rag generation
edumind benchmark rag final
edumind benchmark systems vectordb
edumind benchmark review export SUMMARY_JSON REVIEW_CSV
edumind benchmark review import REVIEW_CSV
edumind benchmark report SUMMARY_JSON
edumind benchmark all
```

Commands default to `smoke`. Put `--profile standard` or `--profile full` immediately after `benchmark` for authoritative local runs. Smoke uses deterministic fixtures/fakes where appropriate, makes no quality claim, and requires no network or Ollama.

The APIs bind to `127.0.0.1`. Their liveness and readiness endpoints are separate, uploads are streamed with a 100 MiB default limit, and destructive resets are serialized.

## Benchmark interpretation

Standard/full runs save per-sample results, complete fingerprints, 95% paired-bootstrap intervals, and Pareto candidates. They apply hard correctness/resource gates before promotion and never manufacture a min-max overall score. Overlapping quality intervals are resolved by p95 latency, memory, then storage.

Packaged defaults refer to a versioned recommendation manifest. It is deliberately non-authoritative until the required datasets/models are present, 60 blinded judgments are imported, and one selected system is evaluated once on the locked test. Chroma remains the runtime vector backend unless a conforming alternative reaches recall/filter gates and improves p95 latency or storage by at least 20%.

Start with [the benchmark overview](experiments/benchmarks/doc.md), then use each subsystem's `doc.md` for its exact formulas, controls, commands, artifacts, and limitations.

## Verification

```powershell
pytest
ruff check src apps services experiments tests
ruff format --check src apps services experiments tests
mypy src apps services experiments
python -m pip check
python -m compileall -q src apps services experiments
edumind benchmark preflight
edumind benchmark all
```

Standard benchmarks require explicit preparation and may take hours. Final recommendations must not be inferred from smoke output or from a model merely importing successfully.
