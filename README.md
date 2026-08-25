# EduMind

EduMind is a local, benchmark-driven system for turning educational material into
searchable evidence and citation-grounded answers. It accepts images, PDFs, DOCX,
audio, and video; preserves source provenance; retrieves the most relevant
content; and asks a local language model to answer only from that evidence.

The project has two equally important parts:

1. **A usable provisional application** for extracting, indexing, and querying
   study material today.
2. **A reproducible benchmark program** for deciding which extractors, chunkers,
   embedding models, retrieval strategies, vector servers, rerankers, and
   generators should power the application later.

Public leaderboards and model documentation help decide which candidates are worth testing. EduMind's own
datasets, quality metrics, latency/resource measurements, and human review decide
whether a candidate is actually suitable for this project.

## What EduMind does

```text
Image / PDF / DOCX / audio / video
                 |
                 v
       structured local extraction
      (text, pages, timestamps, tables,
       formulas, offsets, provenance)
                 |
                 v
       chunking and local embeddings
                 |
                 v
          Chroma HTTP server
                 |
                 v
     token-budget evidence retrieval
                 |
                 v
      local Hugging Face generation
                 |
                 v
        answer with [1], [2], ... citations
```

The application does not silently download models, start Docker, call a hosted
judge, or change its defaults after a benchmark. Downloads and production changes
are explicit actions.

## Current project status

The implemented application path uses deliberately conservative **provisional
controls**. They are baselines, not claims that these components are already the best:

| Stage | Current application default |
|---|---|
| Document extraction | Docling Standard; RapidOCR; PDF-aware OCR for PDFs; full-page OCR for images; TableFormer fast; formula enrichment off |
| Speech extraction | Whisper `small.en` |
| Chunking | Token chunks, 256 tokens with 32-token overlap |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Retrieval | Dense top-5 retrieval packed into a 2,048-token evidence budget |
| Vector server | Chroma over HTTP at `127.0.0.1:8001` |
| Generation | Pinned Hugging Face `Qwen/Qwen3-1.7B`, CPU, thinking disabled |

The benchmark program is used to challenge every one of these choices. A result
becomes a recommendation only after the relevant standard/full experiment and
review process are complete.

## Start here

Choose the path that matches what you are trying to do:

| Goal | Read this |
|---|---|
| Understand the project before installing anything | This README, then the [architecture overview](docs/architecture/overview.md) |
| Install every prerequisite, model, dataset, and server | [Complete installation and preparation guide](docs/setup/installation.md) |
| Start only the current application | [Application run instructions](docs/setup/running.md) |
| Understand how benchmark conclusions are produced | [Benchmark manual](docs/benchmarks/methodology.md) |
| Understand why specific models were shortlisted | [Model-selection rationale](docs/benchmarks/model-selection.md) |
| Find one specific subsystem or experiment | [Documentation map](docs/README.md) |
| Contribute or change the architecture | [Contributing guide](CONTRIBUTING.md) |

## Quick application start

The commands below are the shortest path. Use the [complete guide](docs/setup/installation.md) if
Python, Docker, FFmpeg, or model preparation is not already set up.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements/app.lock
python -m pip install -e . --no-deps
python experiments/benchmarks/prepare.py app-models
docker compose -f infrastructure/chroma.yml up -d
streamlit run src/edumind/ui/streamlit_app.py
```

`app-models` downloads exact pinned snapshots into the repository-controlled
`data/benchmarks/downloads/` directory and writes the generated model-location
file `data/benchmarks/models/selected.json`.

To stop the current vector server:

```powershell
docker compose -f infrastructure/chroma.yml down
```

## How component selection works

EduMind deliberately separates screening, experimentation, and deployment:

```text
public evidence
      |
      v
approved candidate shortlist
(model-selection.md + selection_evidence.csv)
      |
      v
component benchmarks on frozen local data
      |
      v
correctness gates + confidence intervals + Pareto set
      |
      v
final RAG comparison + blinded human review
      |
      v
explicitly reviewed production change
```

- `docs/benchmarks/model-selection.md` explains the shortlist to a reader.
- `experiments/benchmarks/selection_evidence.csv` records machine-readable include/exclude decisions and
  immutable revisions.
- Candidate YAML files define experiment settings, not model revisions.
- `selected.json` is generated during preparation and records local paths.
- MLflow and benchmark artifacts record what was actually executed.
- No benchmark edits `config/base.yaml` or promotes a winner automatically.

## Benchmark path

The experiments are designed as a sequence so that one uncontrolled component
does not contaminate another decision:

| Order | Experiment | Question answered | Documentation |
|---:|---|---|---|
| 1 | Document extraction | Which complete parser/configuration best preserves educational documents? | [Document extraction](docs/benchmarks/extraction/document.md) |
| 2 | Audio extraction | Which ASR profile best transcribes and timestamps educational recordings? | [Audio extraction](docs/benchmarks/extraction/audio.md) |
| 3 | Normalization | Which repairs remove extraction corruption without deleting content? | [Normalization](docs/benchmarks/extraction/normalization.md) |
| 4 | Video extraction | Which keyframe policy adds useful visual text to a frozen ASR transcript? | [Video extraction](docs/benchmarks/extraction/video.md) |
| 5 | Chunking + embedding | Which of the 64 deployable pairs retrieves verified evidence best? | [Chunking and embedding](docs/benchmarks/rag/chunking-embedding.md) |
| 6 | Retrieval + reranking | Do BM25, RRF, or rerankers improve the shortlisted pairs? | [Retrieval](docs/benchmarks/rag/retrieval.md) |
| 7 | Vector servers | Which server preserves recall/filter correctness and performs well under load? | [Vector databases](docs/benchmarks/systems/vector-databases.md) |
| 8 | Generation | Which local generator gives the best grounded, cited answer from frozen evidence? | [Generation](docs/benchmarks/rag/generation.md) |
| 9 | Final RAG | Which shortlisted complete system wins automated and blinded human evaluation? | [Final RAG](docs/benchmarks/rag/final-rag.md) |

`smoke` profiles verify that a real path runs; they are not performance evidence.
`standard` profiles compare the approved candidates. `full` profiles run explicit
finalists on larger or locked workloads.

## Repository map

```text
config/base.yaml            single production configuration
src/edumind/                complete application, including Streamlit UI
experiments/benchmarks/     experiment runners, candidates, metrics, and inputs
data/benchmarks/            committed fixtures/manifests and ignored downloads
requirements/               pinned application and benchmark environments
infrastructure/             provisional production Chroma Compose file
docs/                       all detailed reader documentation
artifacts/                   ignored benchmark and runtime output
```

Production code and experiment code are intentionally separate. Experiments reuse
real production strategies where that is necessary for a valid measurement, but
experiment-only search strategies and candidate adapters do not become application
defaults merely because they can run.

## Deliberate boundaries

EduMind is currently English-first, self-hosted, and local. The repository does
not currently provide:

- cloud inference or a paid model judge;
- automatic production promotion;
- a public API or multi-host service architecture;
- Kubernetes, distributed ingestion, or cloud vector databases;
- a claim that smoke-test results establish quality or speed.

These are scope decisions for the current evidence-gathering phase, not permanent
limitations of the project.

## First benchmark commands

Preview model preparation without downloading:

```powershell
python experiments/benchmarks/prepare.py --list
python experiments/benchmarks/prepare.py all-models --dry-run
```

After following the [preparation guide](docs/setup/installation.md), start with the benchmark
overview rather than running stages in an arbitrary order:

```powershell
python experiments/benchmarks/extraction/normalization/run.py --profile smoke --no-mlflow
python experiments/benchmarks/extraction/document/run.py --profile smoke --no-mlflow
python experiments/benchmarks/rag/chunking_embedding/run.py --profile smoke --no-mlflow
```

See the [benchmark overview](docs/benchmarks/overview.md) for the complete command
sequence, profiles, artifacts, and MLflow behavior.
