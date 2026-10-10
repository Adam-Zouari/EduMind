# EduMind documentation map

Start with the [project README](../README.md) for the project's purpose, current
status, and shortest application start. This page maps each detailed question to
one document so the same instructions are not maintained in several places.

## Setup and operation

| Task | Document |
|---|---|
| Install system tools, environments, dependencies, models, datasets, and servers | [Installation and preparation](setup/installation.md) |
| Start, stop, check, or troubleshoot the current application | [Running the application](setup/running.md) |

## Production architecture

| Question | Document |
|---|---|
| What are the main boundaries and data flows? | [Architecture overview](architecture/overview.md) |
| How does the end-to-end application orchestrator behave? | [Application pipeline](architecture/application.md) |
| How are documents, audio, and video extracted? | [Extraction subsystem](architecture/extraction.md) |
| How do chunking, embedding, retrieval, and generation work in production? | [RAG subsystem](architecture/rag.md) |
| How is Streamlit state separated from application logic? | [User interface](architecture/ui.md) |

## Experiments

EduMind benchmarks components before changing the provisional application
defaults. Runs record results and provenance in MLflow; an engineer reviews the
evidence and records which candidates advance in decision files.

The main dependencies are:

```text
document parser + audio ASR -> video extraction
chunking x embedding -> retrieval/reranking -> real vector-server retrieval
generation on frozen evidence (independent of retrieval)

selected components -> non-locked extraction-impact confirmation
                    -> one Final RAG locked report -> human review
```

Document, audio, chunking/embedding, synthetic vector-server checks, and
generation can begin independently. Each component has its own locked report;
Final RAG evaluates one already-selected complete system. Each benchmark has
its own methodology and metric reference under `benchmarks/extraction/`,
`benchmarks/rag/`, or `benchmarks/vectordb/`, matching the implementation layout.
The shared methodology explains the lifecycle and decision files; shared metric
conventions define cross-benchmark measurement and reporting rules. The runbook
records commands and current implementation limitations.

| Question | Document |
|---|---|
| What is the shared lifecycle, and where is each benchmark's evaluation procedure? | [Shared methodology and benchmark guides](benchmarks/methodology.md) |
| What are the shared measurement rules, and where are individual metric definitions? | [Shared conventions and benchmark metric references](benchmarks/metrics.md) |
| Why was each candidate included? | [Model-selection rationale](benchmarks/model-selection.md) |
| Which commands prepare and run experiments? | [Benchmark runbook](benchmarks/running.md) |
| How are benchmark datasets acquired and described? | [Benchmark dataset guide](benchmarks/datasets.md) |
| How are prepared inputs checked before execution, and how will validation reports be reused? | [Data-validation contract and planned commands](benchmarks/data-validation.md) |
| Which benchmark decisions remain blocked on inspecting downloaded data? | [Temporary data-review checklist](benchmarks/pending-data-review.md) |
| What are the machine-readable model decisions and revisions? | [`selection_evidence.csv`](../experiments/benchmarks/selection_evidence.csv) |

## Project maintenance

- [Contributing](../CONTRIBUTING.md) explains how to change code and documentation.
- [Changelog](../CHANGELOG.md) records notable changes.
- [License](../LICENSE) contains the PolyForm Strict License 1.0.0 terms.

## Machine-readable authorities

Human documentation explains the system; it does not override executable
inputs:

- `config/base.yaml` defines provisional production settings.
- `experiments/benchmarks/selection_evidence.csv` records screening include/exclude
  decisions, checkpoint identities/revisions, and supporting public evidence.
- Each benchmark's `protocol.yaml` defines its settings and, where needed, its
  candidate roster; fixed adapter support is defined in code.
- Frozen dataset manifests define samples, splits, checksums, and provenance.
- Generated data-validation reports identify checked inputs and requirements;
  their standalone/reuse interface is specified but not yet implemented.
- `data/benchmarks/models/selected.json` records model revisions, prepared local
  snapshot paths, and checksums.
- Engineer-reviewed files under `data/benchmarks/decisions/` record selections
  and the completed runs that support them.
- MLflow and run artifacts record what an experiment actually executed.
