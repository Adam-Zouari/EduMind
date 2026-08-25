# EduMind architecture

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Run the application](../setup/running.md) ·
[Benchmark overview](../benchmarks/overview.md)

## Architectural intent

EduMind has one simple application path and a separate experiment area. The
application provides a stable baseline while experiments collect evidence for
future replacements.

```text
Production now
Streamlit -> EduMindPipeline -> extraction -> chunking/embedding
          -> Chroma HTTP -> retrieval/context packing -> local generation

Experiments
frozen datasets -> approved candidates -> per-sample metrics/resources
                -> confidence intervals/Pareto set -> human review
                -> explicit recommendation (never an automatic config change)
```

This separation prevents benchmark candidates, partially implemented strategies,
or provisional results from silently changing the user-facing application.

## Production data flow

```text
upload
  |
  v
source classification
  |
  +-- image/PDF/DOCX -> Docling document parser
  +-- audio          -> Whisper small.en
  +-- video          -> FFmpeg + frozen ASR/document-parser profiles
  |
  v
ExtractedDocument
(normalized text, exact offsets, page/timestamps, structured elements,
 source checksum, model/profile revision, warnings)
  |
  v
token 256/32 chunks -> normalized MiniLM vectors -> Chroma HTTP
  |
  v
dense candidates -> 2,048-token context pack -> numbered evidence
  |
  v
pinned Hugging Face Qwen3 1.7B -> cited answer
```

`EduMindPipeline` is the in-process boundary joining these stages. The Streamlit
controller handles UI concerns such as temporary uploads, duplicate prevention,
progress, and safe errors; it does not duplicate extraction or RAG logic.

## Ownership by directory

| Location | Responsibility |
|---|---|
| `src/edumind/ui/` | Streamlit presentation, session state, upload lifecycle, and calls into the application boundary |
| `src/edumind/extraction/` | Source contracts, detection, extraction adapters, normalization, provenance, and cache |
| `src/edumind/rag/` | Chunking, embedding contracts, Chroma access, context packing, and direct HF generation |
| `src/edumind/application.py` | Typed end-to-end application operations, timings, warnings, and progress events |
| `experiments/benchmarks/` | Candidate matrices, datasets, metrics, statistical selection, MLflow, and reports |
| `config/base.yaml` | The only production configuration source |
| `data/benchmarks/` | Frozen manifests/fixtures plus ignored prepared assets and model snapshots |

## Configuration and model provenance

`config/base.yaml` defines the current production profile. Environment variables
may override only the small set documented in `.env.example`.

Models are never downloaded at import or application startup. Explicit preparation
writes exact snapshots under `data/benchmarks/downloads/` and generates
`data/benchmarks/models/selected.json`. Runtime profiles include the model revision
and all behavior-changing options in their fingerprints, so incompatible cached
extraction or vector indexes are rejected instead of silently reused.

The candidate-selection package has a different role:

- `docs/benchmarks/model-selection.md` explains why a model or vector server was shortlisted.
- `experiments/benchmarks/selection_evidence.csv` provides exact included identities and revisions.
- experiment `candidates.yaml` files define only the settings/matrix to execute.

## Why Chroma and an in-process application are provisional

Chroma HTTP is the current control because it provides a complete server-backed
application path while vector-server experiments compare Chroma, Qdrant,
Weaviate, and PostgreSQL/pgvector. The application has no public API layer yet:
Streamlit constructs the pipeline directly because deployment and multi-host
requirements have not been established.

Docker is started and stopped explicitly. Imports, preparation, and UI startup do
not manage background services.

## Deliberate simplifications

The current architecture does not include a second application implementation,
embedded vector database, local production BM25 index, automatic benchmark
promotion, cloud inference, Kubernetes, or distributed ingestion coordination.
Those additions require a concrete deployment need and benchmarkable hypothesis.

## Detailed component documents

- [Extraction implementation](extraction.md)
- [RAG implementation](rag.md)
- [Application pipeline](application.md)
- [Streamlit application](ui.md)
- [Benchmark method](../benchmarks/methodology.md)
