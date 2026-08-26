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

| Question | Document |
|---|---|
| How does the benchmark program fit together? | [Benchmark overview](benchmarks/overview.md) |
| What runs in each experiment, in what order, on which data, and why? | [Experiment methodology](benchmarks/methodology.md) |
| What does each metric mean and how is it calculated? | [Metric reference](benchmarks/metrics.md) |
| Why was each candidate included? | [Model-selection rationale](benchmarks/model-selection.md) |
| Which commands prepare and run experiments? | [Benchmark runbook](benchmarks/running.md) |
| How are extraction datasets acquired and described? | [Extraction dataset guide](benchmarks/extraction/datasets.md) |
| What are the machine-readable model decisions and revisions? | [`selection_evidence.csv`](../experiments/benchmarks/selection_evidence.csv) |

## Project maintenance

- [Contributing](../CONTRIBUTING.md) explains how to change code and documentation.
- [Changelog](../CHANGELOG.md) records notable changes.
- [License](../LICENSE) contains the MIT terms.

## Machine-readable authorities

Human documentation explains the system; it does not override executable
inputs:

- `config/base.yaml` defines provisional production settings.
- `experiments/benchmarks/selection_evidence.csv` defines included model
  identities and immutable revisions.
- Each experiment's `candidates.yaml` defines its runtime matrix.
- Frozen dataset manifests define samples, splits, checksums, and provenance.
- `data/benchmarks/models/selected.json` records prepared local model paths.
- MLflow and run artifacts record what an experiment actually executed.
