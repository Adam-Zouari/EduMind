# EduMind benchmark program

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Preparation guide](../setup/installation.md) · [Benchmark manual](methodology.md) ·
[Candidate selection](model-selection.md)

This is the entry point for running experiments. The benchmark manual explains
scientific validity; each experiment document explains its own hypothesis,
dataset, candidates, metrics, and limitations.

## Boundary

All experiment code is under `experiments/benchmarks`. Production code supplies the real extraction, chunking, embedding, generation, and Chroma implementations, but a benchmark never changes `config/base.yaml` or promotes a winner automatically.

The [model-selection document](model-selection.md) explains candidate selection.
`experiments/benchmarks/selection_evidence.csv` is the machine-readable authority
for included model identities and revisions. Runnable YAML files contain experiment
settings only. `data/benchmarks/models/selected.json` is generated from that
evidence and records the exact local snapshots.

## Preparation

```powershell
python experiments/benchmarks/prepare.py --list
python experiments/benchmarks/prepare.py all-models --dry-run
python experiments/benchmarks/prepare.py app-models
python experiments/benchmarks/prepare.py rag-models
python experiments/benchmarks/prepare.py extraction-models
python experiments/benchmarks/prepare.py qasper
python experiments/benchmarks/prepare.py vectordb
```

`app-models` prepares only Docling Standard, Whisper `small.en`, MiniLM, and Hugging Face Qwen3 1.7B. The other model targets prepare only included rows. `vectordb` digest-locks Chroma, Qdrant, Weaviate, and PostgreSQL/pgvector images. Public extraction assets require an explicit checksum/license plan.

## Profiles and artifacts

- `smoke` uses tiny real paths and one repetition. It proves wiring only.
- `standard` runs the approved candidates on validation data with warmups and three repetitions.
- `full` accepts an explicit engineer-authored `--shortlist DECISION_JSON` and expands the workload for finalists.

MLflow is enabled by default. Each invocation creates one parent run and one child per candidate. The parent logs `plan.json`, `provenance.json`, `summary.json`, and completion counts. Each child logs its scalar metrics, confidence intervals, candidate JSON, and per-sample Parquet. Failures remain visible with their errors and partial artifacts.

Downstream parent runs also log the engineer decision files they consume. Imported blinded-review judgments and aggregate human metrics are attached to the original final-RAG parent run. `--no-mlflow` is only a debugging path.

Standard/full retain fixed seed 42, frozen manifests, document-level split isolation, checksums, randomized execution order, per-sample output, 10,000 paired bootstrap resamples, and 95% intervals. The runner verifies completeness and computes comparisons but never chooses a winner. Holm correction is used only for formal multiple-comparison claims.

## Recommended experiment order

| Order | Stage | Input from an earlier stage | Detailed procedure |
|---:|---|---|---|
| 1 | Document extraction configuration and architecture | none | [Document extraction](extraction/document.md) |
| 2 | Audio extraction | none | [Audio extraction](extraction/audio.md) |
| 3 | Normalization | none | [Normalization](extraction/normalization.md) |
| 4 | Video keyframes | one approved document parser and ASR | [Video extraction](extraction/video.md) |
| 5 | Chunking and embedding | frozen verified text/evidence | [Chunking and embedding](rag/chunking-embedding.md) |
| 6 | Retrieval and reranking | up to three approved chunking/embedding pairs | [Retrieval](rag/retrieval.md) |
| 7 | Vector servers | precomputed vectors; later one approved retrieval stack | [Vector databases](systems/vector-databases.md) |
| 8 | Generation | identical frozen evidence contexts | [Generation](rag/generation.md) |
| 9 | Final RAG and human review | approved retrieval and generator finalists | [Final RAG](rag/final-rag.md) |

Independent component stages can be prepared in parallel, but downstream stages
must receive engineer-authored decision files where shown. Passing a smoke run does not
support such a decision.

## Direct commands

```powershell
python experiments/benchmarks/extraction/document/run.py --profile standard --phase configuration
python experiments/benchmarks/extraction/audio/run.py --profile standard --device cuda
python experiments/benchmarks/extraction/video/run.py --profile standard --document-selection DOCUMENT_DECISION --audio-selection AUDIO_DECISION
python experiments/benchmarks/extraction/normalization/run.py --profile standard
python experiments/benchmarks/rag/chunking_embedding/run.py --profile standard
python experiments/benchmarks/rag/retrieval/run.py --profile standard --embedding-selection EMBEDDING_DECISION
python experiments/benchmarks/rag/generation/run.py --profile standard --device cuda
python experiments/benchmarks/rag/final/run.py --profile standard --retrieval-selection RETRIEVAL_DECISION --generation-selection GENERATION_DECISION --device cuda
python experiments/benchmarks/vectordb/run.py --profile standard
```

See the [benchmark manual](methodology.md) for the complete method and the stage
pages linked above for each experiment's exact contract.
