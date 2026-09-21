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

## 3. Understand execution profiles and decision files

- `smoke` uses tiny fixtures and checks only that the path executes. Its results
  are never selection evidence.
- `development` runs the stage's protocol-defined or code-generated candidate
  roster on development data;
  this is where comparison, tuning, and finalist selection happen.
- `validation` runs only candidates explicitly selected by an engineer on unseen
  validation data. Most stages use
  `--shortlist`; document extraction uses separate `--pdf-selection` and
  `--image-selection` decisions because their valid configuration sets differ.
- `locked` runs exactly one frozen selection on the locked-test split after
  validation and any required human review. It reports the final estimate and is
  never used for tuning.
- `--manifest PATH` overrides a stage's default dataset manifest.

Every command-line interface uses the semantic names directly. Run plans,
decision provenance, resolved protocol artifacts, and MLflow parameters record
the same profile name. The retired `standard` and `full` spellings are not
accepted.

Each runner loads the adjacent `protocol.yaml` by default. Use its `--protocol`
option only to execute another reviewed, committed protocol revision. Final RAG
and complete retrieval expose separate options for each composed protocol. A
run uploads every source YAML, writes a resolved `<name>_protocol.json`, and
records every protocol checksum in the parent fingerprint and child parameters.

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
  "reason": "<why these candidates advance>"
}
```

Resolve `source_summary` relative to the decision file. Create separate PDF and
image decisions because they refer to different parent summaries.

Keep reviewed decision files under `data/benchmarks/decisions/`, creating that
directory when the first real decision exists. Use descriptive names such as
`audio-validation.json`, `chunking-embedding-validation.json`, or
`retrieval-validation.json`. Do not commit placeholder decisions: the source
run ID, source summary, exact selected candidate names, reviewer, date, and
reason must come from an actual completed development run. The path is then
passed to the validation command through `--shortlist` or the stage-specific
selection option shown below.

The methodology assigns development, validation, and locked data according to
the execution profile. A parser architecture must be compared on development
before it can be named a validation finalist.

## 4. Run extraction experiments

Start Docker Desktop and prepare the official document scorer before any
document corpus containing table or formula annotations:

```powershell
docker version
python -m experiments.benchmarks.prepare evaluators
```

Document parsers and common metrics run in the project's Python 3.12
environment. Only TEDS, TEDS-S, and CDM run in the pinned OmniDocBench Python
3.10 container. A run fails clearly if that prepared scorer is unavailable.

Document extraction first screens the Docling configuration matrix on
development:

```powershell
python -m experiments.benchmarks.extraction.document.run --profile development `
  --manifest data/benchmarks/extraction/document-development.json

```

The default `--source all` creates the three parent runs described in the
methodology. Use `--source pdf`, `--source image`, or `--source docx` to run one
comparison independently. PDF configuration executes 24 profiles, image
configuration executes 12 unique full-page profiles, and DOCX executes native
Docling once. After those configuration decisions are recorded, the selected
Standard profiles, Granite Docling, and PaddleOCR-VL must be compared on the
development split. Only the architecture finalists recorded from that comparison
may run with `--profile validation` on the validation manifest.

Run the development architecture comparison with the configuration decisions:

```powershell
python -m experiments.benchmarks.extraction.document.run --profile development `
  --comparison architecture `
  --manifest data/benchmarks/extraction/document-development.json `
  --pdf-selection PDF_CONFIGURATION_DECISION.json `
  --image-selection IMAGE_CONFIGURATION_DECISION.json
```

After reviewing that comparison, run only its recorded finalists on validation:

```powershell
python -m experiments.benchmarks.extraction.document.run --profile validation `
  --comparison architecture `
  --manifest data/benchmarks/extraction/document-validation.json `
  --pdf-selection PDF_ARCHITECTURE_FINALISTS.json `
  --image-selection IMAGE_ARCHITECTURE_FINALISTS.json
```

For `development`, each selection file must choose one candidate profile from the matching
completed configuration parent. For `validation`, each file must choose one or more
finalists from the matching completed development architecture parent. The
runner rejects using configuration decisions directly on validation.

Run audio independently:

```powershell
python -m experiments.benchmarks.extraction.audio.run --profile development `
  --manifest data/benchmarks/extraction/audio-development.json `
  --device cuda
python -m experiments.benchmarks.extraction.audio.run --profile validation `
  --manifest data/benchmarks/extraction/audio-validation.json `
  --shortlist AUDIO_DECISION `
  --device cuda
python -m experiments.benchmarks.extraction.audio.run --profile locked `
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

Video is a two-step benchmark. First, decode and transcribe the phase's audio
once with the selected ASR. The runner composes the normal video and ASR
protocols and binds their checksums plus the manifest checksum into the frozen
artifact:

```powershell
python -m experiments.benchmarks.extraction.video.run --profile development --phase frozen-asr `
  --manifest data/benchmarks/extraction/video-development.json `
  --audio-selection AUDIO_DECISION `
  --frozen-asr artifacts/video-development-asr.json `
  --device cuda
```

Then run visual-only comparisons. Every child validates and references the same
frozen artifact and never invokes ASR. Visual runs also load the adjacent
document protocol (override it only with `--document-protocol PATH`) and record
its checksum beside the video and audio protocol identities:

```powershell
python -m experiments.benchmarks.extraction.video.run --profile development --phase fixed `
  --manifest data/benchmarks/extraction/video-development.json `
  --frozen-asr artifacts/video-development-asr.json `
  --document-selection DOCUMENT_DECISION `
  --device cuda

python -m experiments.benchmarks.extraction.video.run --profile development --phase scene `
  --manifest data/benchmarks/extraction/video-development.json `
  --frozen-asr artifacts/video-development-asr.json `
  --document-selection DOCUMENT_DECISION `
  --device cuda

python -m experiments.benchmarks.extraction.video.run --profile development --phase hybrid `
  --manifest data/benchmarks/extraction/video-development.json `
  --frozen-asr artifacts/video-development-asr.json `
  --document-selection DOCUMENT_DECISION `
  --device cuda
```

The initial video protocol deliberately has
`selected_scene_threshold: null` and
`selected_scene_source_run_id: null`, so fixed and scene phases run while
hybrid rejects execution. After reviewing the scene comparison, edit those two
fields in `experiments/benchmarks/extraction/video/protocol.yaml`, increment
`protocol_version`, and regenerate the frozen-ASR artifact. Its previous
checksum is intentionally incompatible. `--phase all` may then rerun the
resulting fixed, scene, and selected-threshold hybrid configurations in one
nine-child development parent. The validation finalist decision must reference that completed
`video-development` parent; the locked decision must reference the completed
`video-validation` parent.

Development is an ordered nine-configuration study:

1. compare fixed intervals of 5, 10, and 20 seconds;
2. compare FFmpeg scene thresholds of 0.30, 0.40, and 0.50;
3. record the selected scene threshold and source development run ID in
   `protocol.yaml`, bump its version, and regenerate frozen ASR; and
4. compare hybrid maximum gaps of 5, 10, and 20 seconds using that threshold.

Every configuration includes the first frame. The three comparisons remain in
the same `EduMind / extraction` MLflow experiment, where their nine child runs
can be filtered and compared together. Validation runs only the
engineer-selected finalists; locked test runs one selected configuration once.
`--profile validation` requires a shortlist of at most three development finalists;
`--profile locked` requires a decision containing exactly one validation
winner. Frozen-ASR reuse rejects any manifest, video-protocol, audio-protocol,
or sample-ID mismatch.

Run the validation finalists with a newly generated validation frozen-ASR
artifact:

```powershell
python -m experiments.benchmarks.extraction.video.run --profile validation --phase frozen-asr `
  --manifest data/benchmarks/extraction/video-validation.json `
  --audio-selection SELECTED_ASR_DECISION.json `
  --frozen-asr artifacts/video-validation-asr.json `
  --device cuda

python -m experiments.benchmarks.extraction.video.run --profile validation --phase all `
  --manifest data/benchmarks/extraction/video-validation.json `
  --frozen-asr artifacts/video-validation-asr.json `
  --document-selection DOCUMENT_DECISION.json `
  --shortlist VIDEO_FINALISTS_DECISION.json `
  --device cuda
```

After validation records one winner, generate the locked split's frozen ASR and
run that configuration once:

```powershell
python -m experiments.benchmarks.extraction.video.run --profile locked --phase frozen-asr `
  --manifest data/benchmarks/extraction/video-locked-test.json `
  --audio-selection SELECTED_ASR_DECISION.json `
  --frozen-asr artifacts/video-locked-asr.json `
  --device cuda

python -m experiments.benchmarks.extraction.video.run --profile locked --phase all `
  --manifest data/benchmarks/extraction/video-locked-test.json `
  --frozen-asr artifacts/video-locked-asr.json `
  --document-selection DOCUMENT_DECISION.json `
  --shortlist SELECTED_VIDEO_DECISION.json `
  --device cuda
```

Smoke uses the smoke settings in the committed video protocol:

```powershell
python -m experiments.benchmarks.extraction.video.run --profile smoke --phase frozen-asr `
  --audio-candidate whisper-small-en-control `
  --frozen-asr artifacts/video-smoke-asr.json

python -m experiments.benchmarks.extraction.video.run --profile smoke --phase all `
  --frozen-asr artifacts/video-smoke-asr.json `
  --image-candidate "docling-standard|ocr=rapidocr|mode=full_page|table=fast|formula=off"
```

Extraction candidates are scored without an additional cleanup profile. The
runner records the parser or ASR output and applies only the fixed evaluator
representation rules described in the methodology.

## 5. Run chunking, embedding, and retrieval

Run the chunker–embedding matrix first:

```powershell
python -m experiments.benchmarks.rag.chunking_embedding.run --profile development `
  --device cuda `
  --dtype float16
python -m experiments.benchmarks.rag.chunking_embedding.run --profile validation `
  --shortlist EMBEDDING_DECISION `
  --device cuda `
  --dtype float16
```

These authoritative commands use embedding batch size `1` and require peak
process VRAM no greater than 3,584 MiB on the 4,096 MiB RTX 3050. Reject a
candidate during engineer review if its recorded peak exceeds that gate. A CUDA
run fails if execution falls back to CPU or VRAM cannot be measured.

Then give the retrieval experiment an engineer-selected chunker–embedding
decision:

```powershell
python -m experiments.benchmarks.rag.retrieval.run --profile development `
  --embedding-selection EMBEDDING_DECISION

python -m experiments.benchmarks.rag.retrieval.run --profile validation `
  --embedding-selection EMBEDDING_DECISION `
  --shortlist RETRIEVAL_DECISION
```

Both commands load
`experiments/benchmarks/rag/retrieval/protocol.yaml` by default. Use
`--protocol PATH` only for another reviewed protocol revision. The file owns
pool depth, BM25/Dense/RRF settings, embedding and reranker batch sizes,
reranker input limits, quality cutoffs, confidence settings, finalist limit,
execution counts, hardware requirements, and the VRAM gate. Unknown or missing
fields are fatal.
The source file is logged as an input, and the resolved values plus checksum are
stored as `retrieval_protocol.json` under the parent run.

The development plan must contain exactly 15 direct candidate children: Dense,
BM25, and RRF, each crossed with no reranker, GTE ModernBERT, Ettin 150M, Ettin
400M, and Ettin 1B. The three `retriever|none` children own the checksummed
top-20 quality pools; reranker children reference the matching owner. Pools and
paired comparisons are artifacts under the same parent, not intermediate MLflow
runs. Retriever effects are written to authoritative
`retriever_comparisons.parquet` and its human-readable
`retriever_comparisons.csv` mirror. Reranker effects use
`reranker_comparisons.parquet` and `reranker_comparisons.csv`. Validation writes
the optional `finalist_comparisons.parquet` and `finalist_comparisons.csv` only
when cross-stack finalist comparisons are explicitly requested.

The retrieval/reranking smoke fixture must contain exactly 30 frozen canonical
chunks. It validates selection into a 20-chunk pool without imposing a fixed
chunk count on authoritative chunking candidates.

The validation profile runs the engineer-selected finalists plus any matching
`retriever|none` controls needed to measure their incremental effect. Do not use
a legacy eight-method or RRF-only plan as authoritative evidence.

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
python -m experiments.benchmarks.vectordb.run --profile smoke
python -m experiments.benchmarks.vectordb.run --profile development
python -m experiments.benchmarks.vectordb.run --profile validation `
  --shortlist DATABASE_DECISION `
  --embedding-selection EMBEDDING_DECISION
```

Then measure complete retrieval through a selected server:

```powershell
python -m experiments.benchmarks.vectordb.retrieval_run `
  --profile development `
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
python -m experiments.benchmarks.rag.generation.run --profile development --device cuda
python -m experiments.benchmarks.rag.generation.run --profile validation `
  --device cuda `
  --shortlist GENERATION_DECISION
```

Development runs the Falcon-H1-Tiny-R-90M control plus the Qwen3-0.6B,
Qwen3.5-0.8B, and MiniCPM5-1B candidates. Validation runs only the finalists named by
`GENERATION_DECISION`. Every executed profile uses reasoning mode, the same
whole-model CUDA device, `float16`, and batch size `1`; prompts and repetitions
are processed sequentially.

## 8. Run Final RAG and blinded review

Run the complete-system candidate grid on development data from explicit
retrieval and generation decisions:

```powershell
python -m experiments.benchmarks.rag.final.run --profile development `
  --manifest data/benchmarks/rag/rag-selection-dev.json `
  --retrieval-selection RETRIEVAL_DECISION `
  --generation-selection GENERATION_DECISION `
  --device cuda
```

After inspecting development, record exactly three complete-system finalists and
run them on validation:

```powershell
python -m experiments.benchmarks.rag.final.run --profile validation `
  --manifest data/benchmarks/rag/rag-selection-validation.json `
  --shortlist FINAL_RAG_FINALISTS_DECISION `
  --device cuda
```

Export anonymous answers, enter judgments in the CSV, then import them:

```powershell
python -m experiments.benchmarks.review export FINAL_RAG_VALIDATION REVIEW.csv
python -m experiments.benchmarks.review import REVIEW.csv
```

Import writes `REVIEW.results.json` beside the CSV and attaches the judgments to
the original MLflow run. The exact positional arguments and options are always
available through `python -m experiments.benchmarks.review --help`.

The one locked-test run requires reviewed judgments and explicit confirmation:

```powershell
python -m experiments.benchmarks.rag.final.run --profile locked `
  --manifest data/benchmarks/rag/rag-selection-locked-test.json `
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
python -m experiments.benchmarks.rag.final.confirm_extraction `
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
Smoke runs prove wiring only. Development results support finalist selection,
validation results support the final selection, and the one locked result is the
frozen final estimate. The code does
not calculate a universal winner or edit production configuration.

If a run fails, inspect the failed MLflow child and its error artifact, correct
the missing model/data/server problem, and rerun. Do not reuse partial results as
an authoritative comparison.
