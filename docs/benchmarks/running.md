# Benchmark runbook

[Benchmark overview](overview.md) · [Experiment methodology](methodology.md) ·
[Metric definitions](metrics.md) · [Installation](../setup/installation.md)

This is the single command reference for preparing and running EduMind
benchmarks. The methodology explains the experiment design; this page focuses
on execution, inputs, outputs, and failure handling.

Run commands from the repository root in the prepared virtual environment.

## 1. Confirm preparation

Follow the [installation guide](../setup/installation.md) once. It is the only
authority for dependencies, model downloads, dataset creation, and server-image
preparation. This runbook assumes those steps and the required frozen manifests
are complete.

## 2. Start MLflow

MLflow logging is enabled by default. Start the local browser in a separate
terminal if you want to inspect runs while they execute:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Use `--no-mlflow` only for debugging. A normal invocation creates one parent run
and one child per candidate. Failures remain visible rather than being silently
skipped.

## 3. Understand profiles and decision files

- `smoke` uses tiny fixtures and checks only that the path executes.
- `standard` runs the candidate registry for the stage.
- `full` requires `--shortlist DECISION_JSON` and runs only candidates explicitly
  selected by an engineer.
- `--manifest PATH` overrides a stage's default dataset manifest.

A decision JSON names candidates selected after inspecting a completed upstream
run. It is an explicit input to the next stage, not an automatically generated
winner. The runner validates that the referenced candidate exists.

The methodology assigns development, validation, and locked data according to
the experiment phase. Pass the shown `--manifest` paths explicitly. In
particular, the shared extraction runner's current automatic profile-to-split
mapping does not represent the two-phase development/validation method by
itself.

## 4. Run extraction experiments

Document extraction first screens the Docling configuration matrix, then
compares selected configurations with the visual-parser architectures:

```powershell
python experiments/benchmarks/extraction/document/run.py --profile standard `
  --phase configuration `
  --manifest data/benchmarks/extraction/document-development.json

python experiments/benchmarks/extraction/document/run.py --profile full `
  --phase architecture `
  --shortlist DOCUMENT_CONFIG_DECISION `
  --manifest data/benchmarks/extraction/document-validation.json
```

Run audio independently:

```powershell
python experiments/benchmarks/extraction/audio/run.py --profile standard `
  --manifest data/benchmarks/extraction/audio-development.json `
  --device cuda
python experiments/benchmarks/extraction/audio/run.py --profile full `
  --manifest data/benchmarks/extraction/audio-validation.json `
  --shortlist AUDIO_DECISION `
  --device cuda
```

Video requires one selected document parser and one selected ASR profile:

```powershell
python experiments/benchmarks/extraction/video/run.py --profile standard `
  --manifest data/benchmarks/extraction/video-development.json `
  --document-selection DOCUMENT_DECISION `
  --audio-selection AUDIO_DECISION `
  --device cuda
```

Extraction candidates are scored without an additional cleanup profile. The
runner records the parser or ASR output and applies only the fixed evaluator
representation rules described in the methodology.

## 5. Run chunking, embedding, and retrieval

Run the chunker–embedding matrix first:

```powershell
python experiments/benchmarks/rag/chunking_embedding/run.py --profile standard
python experiments/benchmarks/rag/chunking_embedding/run.py --profile full --shortlist EMBEDDING_DECISION
```

Then give the retrieval experiment an engineer-selected chunker–embedding
decision:

```powershell
python experiments/benchmarks/rag/retrieval/run.py --profile standard `
  --embedding-selection EMBEDDING_DECISION

python experiments/benchmarks/rag/retrieval/run.py --profile full `
  --embedding-selection EMBEDDING_DECISION `
  --shortlist RETRIEVAL_DECISION
```

Exact NumPy dense search and local BM25 are experiment controls here; they do
not become production indexes.

## 6. Run vector-server experiments

Start the four real servers explicitly:

```powershell
docker compose -f experiments/benchmarks/vectordb/compose.yml up -d
docker compose -f experiments/benchmarks/vectordb/compose.yml ps
```

Run dense ANN and conformance measurements:

```powershell
python experiments/benchmarks/vectordb/run.py --profile smoke
python experiments/benchmarks/vectordb/run.py --profile standard
python experiments/benchmarks/vectordb/run.py --profile full `
  --shortlist DATABASE_DECISION `
  --embedding-selection EMBEDDING_DECISION
```

Then measure complete retrieval through a selected server:

```powershell
python experiments/benchmarks/vectordb/retrieval_run.py `
  --profile standard `
  --database-selection DATABASE_DECISION `
  --embedding-selection EMBEDDING_DECISION `
  --retrieval-selection RETRIEVAL_DECISION
```

Stop the benchmark servers when finished:

```powershell
docker compose -f experiments/benchmarks/vectordb/compose.yml down
```

The Compose file uses fixed loopback ports. Image preparation and digest locking
belong to the [installation guide](../setup/installation.md#5-vector-database-servers).

## 7. Run generation

Generation uses frozen evidence contexts so generator quality is not confused
with retrieval quality:

```powershell
python experiments/benchmarks/rag/generation/run.py --profile standard --device cuda
python experiments/benchmarks/rag/generation/run.py --profile full `
  --device cuda `
  --shortlist GENERATION_DECISION
```

All candidates in one invocation use the same requested whole-model device.

## 8. Run Final RAG and blinded review

Run validation systems from explicit retrieval and generation decisions:

```powershell
python experiments/benchmarks/rag/final/run.py --profile standard `
  --retrieval-selection RETRIEVAL_DECISION `
  --generation-selection GENERATION_DECISION `
  --device cuda
```

Export anonymous answers, enter judgments in the CSV, then import them:

```powershell
python experiments/benchmarks/review.py export FINAL_SELECTION REVIEW.csv
python experiments/benchmarks/review.py import REVIEW.csv
```

Import writes `REVIEW.results.json` beside the CSV and attaches the judgments to
the original MLflow run. The exact positional arguments and options are always
available through `python experiments/benchmarks/review.py --help`.

The one locked-test run requires reviewed judgments and explicit confirmation:

```powershell
python experiments/benchmarks/rag/final/run.py --profile full `
  --shortlist LOCKED_FINAL_DECISION `
  --review-results REVIEW.results.json `
  --confirm-locked-test `
  --device cuda
```

The intended Final RAG methodology includes the selected vector server. The
current Final RAG runner has no database-decision input and still evaluates the
experiment retrieval path directly. Until that is implemented, its results must
not be presented as confirmation of a complete server-backed system.

## 9. Confirm extraction impact

After choosing a complete system, compare verified reference text with extracted
text on separate non-locked documents:

```powershell
python experiments/benchmarks/rag/final/confirm_extraction.py `
  --reference-manifest REFERENCE_MANIFEST `
  --extracted-manifest EXTRACTED_MANIFEST `
  --candidate FINAL_CANDIDATE `
  --device cuda
```

This measures extraction-induced degradation; it is not another locked-test
tuning opportunity.

## 10. Read results

Each completed parent run records the plan, provenance, candidate completion
status, aggregate summary, applicable confidence intervals, and decision inputs.
Child runs record candidate parameters, scalar metrics, errors, and per-sample
Parquet artifacts. Local artifacts are also written atomically under the
configured artifact root.

A run is usable only when every planned candidate and required metric completed.
Smoke runs prove wiring only. Standard/full results remain evidence for an
engineer; the code does not calculate a universal winner or edit production
configuration.

If a run fails, inspect the failed MLflow child and its error artifact, correct
the missing model/data/server problem, and rerun. Do not reuse partial results as
an authoritative comparison.
