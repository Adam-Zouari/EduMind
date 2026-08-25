# EduMind documentation map

This page is the index for the repository's documentation. Start with the
[project README](../README.md) if you do not yet know what EduMind does or how the
application and benchmark program relate.

## Choose documentation by task

| If you want to... | Read... |
|---|---|
| Install all system tools, Python dependencies, models, datasets, and servers | [Complete installation and preparation guide](setup/installation.md) |
| Start or stop the current Streamlit application | [Application run instructions](setup/running.md) |
| Understand component boundaries and data flow | [Technical architecture](architecture/overview.md) |
| Understand the Streamlit UI/controller | [Application documentation](architecture/ui.md) |
| Understand extraction types, routing, caching, and provenance | [Extraction implementation](architecture/extraction.md) |
| Understand chunking, embedding, retrieval, and generation defaults | [RAG implementation](architecture/rag.md) |
| Understand the end-to-end in-process pipeline | [Pipeline implementation](architecture/application.md) |
| Understand benchmark profiles, artifacts, and commands | [Benchmark overview](benchmarks/overview.md) |
| Understand statistical validity and promotion rules | [Benchmark manual](benchmarks/methodology.md) |
| Review why models and vector servers were included | [Model-selection rationale](benchmarks/model-selection.md) |
| Inspect the machine-readable selection decisions | [Selection evidence](../experiments/benchmarks/selection_evidence.csv) |
| Prepare extraction datasets and manifests | [Extraction dataset guide](benchmarks/extraction/datasets.md) |
| Make a contribution | [Contributing guide](../CONTRIBUTING.md) |
| Review notable repository changes | [Changelog](../CHANGELOG.md) |
| Review the project license | [MIT license](../LICENSE) |

## Experiment documentation

Experiment code and machine-readable inputs remain under `experiments/benchmarks/`.
The matching reader documentation is centralized here so it is easy to navigate.

| Area | Experiment document |
|---|---|
| Extraction | [Document](benchmarks/extraction/document.md), [audio](benchmarks/extraction/audio.md), [video](benchmarks/extraction/video.md), [normalization](benchmarks/extraction/normalization.md) |
| RAG | [Chunking and embedding](benchmarks/rag/chunking-embedding.md), [retrieval and reranking](benchmarks/rag/retrieval.md), [generation](benchmarks/rag/generation.md), [final RAG](benchmarks/rag/final-rag.md) |
| Systems | [Vector database servers](benchmarks/systems/vector-databases.md) |

## Which document is authoritative?

- `config/base.yaml` is the production runtime configuration.
- `benchmarks/model-selection.md` explains candidate-selection policy.
- `experiments/benchmarks/selection_evidence.csv` is authoritative for included model identities and
  immutable revisions.
- Each experiment's `candidates.yaml` is authoritative for that
  experiment's runtime matrix.
- Frozen dataset manifests are authoritative for samples, splits, checksums, and
  provenance.
- MLflow plus the generated benchmark artifacts are authoritative for what a run
  actually measured.

Documentation explains these sources; it does not override them.
