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

Use `--no-mlflow` only for debugging. A normal comparison creates one parent run
and one child per candidate. Document `--source all` deliberately launches three
comparisons and therefore creates separate PDF, image, and DOCX parents.
Failures remain visible rather than being silently skipped.

## 3. Understand profiles and decision files

- `smoke` uses tiny fixtures and checks only that the path executes.
- `standard` runs the candidate registry for the stage.
- `full` runs only candidates explicitly selected by an engineer. Most stages use
  `--shortlist`; document extraction uses separate `--pdf-selection` and
  `--image-selection` decisions because their valid configuration sets differ.
- `--manifest PATH` overrides a stage's default dataset manifest.

A decision JSON names candidates selected after inspecting a completed upstream
run. It is an explicit input to the next stage, not an automatically generated
winner. The runner validates that the referenced candidate exists.

```json
{
  "schema_version": 1,
  "source_summary": "../../artifacts/benchmarks/extraction/document-configuration-pdf/<run-id>/summary.json",
  "source_run_id": "<run-id>",
  "selected_candidates": ["<exact child-run candidate name>"],
  "selected_by": "<engineer name>",
  "selected_date": "YYYY-MM-DD",
  "reason": "<why this profile advances>"
}
```

Resolve `source_summary` relative to the decision file. Create separate PDF and
image decisions because they refer to different parent summaries.

The methodology assigns development, validation, and locked data according to
the experiment phase. The document runner maps `standard` to development and
`full` to validation; `--manifest` can still provide an explicit frozen path.

## 4. Run extraction experiments

Document extraction first screens the Docling configuration matrix, then
compares selected configurations with the visual-parser architectures:

```powershell
python experiments/benchmarks/extraction/document/run.py --profile standard `
  --manifest data/benchmarks/extraction/document-development.json

python experiments/benchmarks/extraction/document/run.py --profile full `
  --pdf-selection PDF_CONFIG_DECISION `
  --image-selection IMAGE_CONFIG_DECISION `
  --manifest data/benchmarks/extraction/document-validation.json
```

The default `--source all` creates the three parent runs described in the
methodology. Use `--source pdf`, `--source image`, or `--source docx` to run one
comparison independently. PDF configuration executes 24 profiles, image
configuration executes 12 unique full-page profiles, and DOCX executes native
Docling once.

Run audio independently:

```powershell
python experiments/benchmarks/extraction/audio/run.py --profile standard `
  --manifest data/benchmarks/extraction/audio-development.json `
  --device cuda
python experiments/benchmarks/extraction/audio/run.py --profile full `
  --manifest data/benchmarks/extraction/audio-validation.json `
  --shortlist AUDIO_DECISION `
  --device cuda
python experiments/benchmarks/extraction/audio/run.py --profile locked `
  --manifest data/benchmarks/extraction/audio-locked-test.json `
  --shortlist SELECTED_ASR_DECISION `
  --device cuda
```

Prepare all three frozen audio manifests before running any of these commands;
the runner validates their sample, checksum, and speaker/document-family isolation
as one dataset contract.

The runner reads the matching split from
`data/benchmarks/extraction/audio-reliability.json`. Use
`--reliability-manifest PATH` only when the frozen reliability manifest is stored
elsewhere. The locked profile rejects decisions containing more than one ASR
profile.

Video requires one selected document parser and one selected ASR profile:

```powershell
python experiments/benchmarks/extraction/video/run.py --profile standard `
  --manifest data/benchmarks/extraction/video-development.json `
  --document-selection DOCUMENT_DECISION `
  --audio-selection AUDIO_DECISION `
  --device cuda
```

Development is an ordered nine-configuration study:

1. compare fixed intervals of 5, 10, and 20 seconds;
2. compare FFmpeg scene thresholds of 0.30, 0.40, and 0.50;
3. record the selected scene threshold; and
4. compare hybrid maximum gaps of 5, 10, and 20 seconds using that threshold.

Every configuration includes the first frame. The three comparisons remain in
the same `EduMind / extraction` MLflow experiment, where their nine child runs
can be filtered and compared together. Validation runs only the
engineer-selected finalists; locked test runs one selected configuration once.
The current video runner must implement this sequence before a video result can
be treated as authoritative; the existing smoke command remains a wiring check.

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
