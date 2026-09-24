# Benchmark runbook

[Benchmark overview](overview.md) · [Experiment methodology](methodology.md) ·
[Metric definitions](metrics.md) · [Installation](../setup/installation.md)

This is the command reference for running EduMind benchmarks. Run every command
from the repository root in the prepared Python 3.12 environment.

## 1. Prepare the environment

Follow the [installation guide](../setup/installation.md). It is the authority
for dependencies, model snapshots, data preparation, evaluator images, and
vector-server images. A benchmark does not download missing models while it is
running.

Start MLflow in a separate terminal:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

New runs are routed to one experiment per benchmark:

| Benchmark | MLflow experiment |
|---|---|
| Document extraction | `EduMind / Document` |
| ASR | `EduMind / ASR` |
| Video extraction | `EduMind / Video` |
| Chunking–embedding | `EduMind / Chunking–Embedding` |
| Retrieval–reranking | `EduMind / Retrieval–Reranking` |
| Vector database | `EduMind / Vector Database` |
| Generation | `EduMind / Generation` |
| Final RAG | `EduMind / Final RAG` |

Historical runs remain in their original experiments. Use `--no-mlflow` only
for local debugging.

## 2. Follow the lifecycle

Applicable model-backed benchmarks use this order:

```text
smoke-cpu + smoke-cuda -> preflight -> development -> validation -> locked
```

- `smoke` checks wiring on tiny fixtures. One command creates independent CPU
  and CUDA parents. `--device cpu` or `--device cuda` may narrow it for
  debugging; neither run may fall back to the other device. Per-device smoke
  dtypes come from the protocol, so CUDA checks use the authoritative FP16 path
  where the CPU path requires a different dtype.
- `preflight` qualifies every declared candidate on the target CUDA machine.
  It records placement and process-tree resources but produces no quality
  evidence.
- `development` automatically finds the exact matching preflight and runs only
  its qualified candidates.
- `validation` runs the finalists named in the reviewed development decision.
- `locked` runs exactly one validation winner and is reporting-only.

Vector Database is CPU-only and has no CUDA preflight. Generation and Final RAG
have CPU/CUDA smoke but no preflight until their benchmark designs are frozen.

Development, validation, and locked use the device, dtype, and batch size in
the benchmark protocol. Their current model-backed contracts require CUDA;
passing a CPU override is rejected.

## 3. Understand automatic inputs and overrides

Each benchmark owns a strict `protocol.yaml`. It defines the candidate roster,
execution profiles, and behavior-changing settings. The runner records the
source YAML, its resolved JSON, and its checksum.

A **manifest** fixes the exact samples, split, paths, checksums, annotations,
and source provenance. `--profile` selects the project manifest automatically.
Use `--manifest PATH` only to run an alternate reviewed manifest.

A **decision file** is an engineer-reviewed transition between stages. It names
the exact successful candidates selected from a completed upstream parent:

```json
{
  "schema_version": 1,
  "source_summary": "../../../artifacts/benchmarks/<suite>/<run-id>/summary.json",
  "source_run_id": "<completed-parent-run-id>",
  "selected_candidates": ["<exact-candidate-id>"],
  "selected_by": "<reviewer>",
  "selected_date": "YYYY-MM-DD",
  "reason": "<selection rationale>"
}
```

Store reviewed decisions under `data/benchmarks/decisions/`. Do not commit
placeholders. Validation automatically reads
`<benchmark>-validation.json`; locked reads `<benchmark>-locked.json`. Once a
decision has been consumed, preserve it and create a versioned replacement if
the selection changes. `--shortlist PATH` and stage-specific selection options
remain explicit overrides.

Development locates preflight by an exact SHA-256 fingerprint covering the
candidate roster, model revisions and checksums, protocols, software locks,
executable source tree and Git commit, GPU and driver, device, dtype, batch
size, exact stress manifest, and tested input envelope. It never chooses an
arbitrary latest run. Use `--preflight-run-id ID` to select an exact MLflow run,
or `--preflight-report PATH` for local no-MLflow debugging.

Definitive hardware exclusions (`vram_limit_exceeded`, `gpu_oom`, or
`offload_detected`) are omitted from development and remain in its provenance.
`measurement_unavailable`, unverifiable placement, and infrastructure failures
block development until preflight is rerun successfully.

Each candidate child contains a `preflight_candidate.json` evidence artifact.
The parent `preflight_report.json` records the complete roster and required
component groups. Video is ready only when frozen ASR and a visual policy are
qualified; Document additionally requires viable PDF, image, and DOCX routes.
Preflight warmups, repetitions, telemetry interval, polling interval, and
timeout are read from the benchmark protocol.

## 4. Document extraction

Prepare the official table/formula evaluator when the manifest contains those
annotations:

```powershell
docker version
python -m experiments.benchmarks.prepare evaluators
```

Run wiring and hardware qualification:

```powershell
python -m experiments.benchmarks.extraction.document.run --profile smoke
python -m experiments.benchmarks.extraction.document.run --profile preflight
```

Document development has two deliberate comparisons. First screen Docling
configuration candidates:

```powershell
python -m experiments.benchmarks.extraction.document.run --profile development
```

Review the PDF and image parents and create
`document-pdf-configuration.json` and `document-image-configuration.json`.
Then compare the chosen Standard configurations with Granite Docling and
PaddleOCR-VL:

```powershell
python -m experiments.benchmarks.extraction.document.run `
  --profile development --comparison architecture `
  --pdf-selection data/benchmarks/decisions/document-pdf-configuration.json `
  --image-selection data/benchmarks/decisions/document-image-configuration.json
```

After reviewing those architecture parents, create the PDF and image validation
decisions and run:

```powershell
python -m experiments.benchmarks.extraction.document.run `
  --profile validation --comparison architecture
```

After validation, record exactly one winner from each PDF and image parent in
`document-pdf-locked.json` and `document-image-locked.json`. Run the frozen
source routes once on the locked-test manifest:

```powershell
python -m experiments.benchmarks.extraction.document.run --profile locked
```

The locked profile selects the architecture comparison automatically, resolves
both decisions, runs the selected PDF and
image parser once per locked sample, and runs native Docling for DOCX. The same
`document-image-locked.json` decision is consumed by Video. Locked Document
results are reporting-only and cannot change either winner. A `--source`
override is rejected for locked execution so a partial source result cannot be
mistaken for the complete frozen routing policy.

Use `--source pdf|image|docx` to isolate one source type. The default `all`
creates separate parents because the valid candidate sets differ.

## 5. ASR

```powershell
python -m experiments.benchmarks.extraction.audio.run --profile smoke
python -m experiments.benchmarks.extraction.audio.run --profile preflight
python -m experiments.benchmarks.extraction.audio.run --profile development
python -m experiments.benchmarks.extraction.audio.run --profile validation
python -m experiments.benchmarks.extraction.audio.run --profile locked
```

After development, create `audio-validation.json`. After validation, create
`audio-locked.json` containing exactly one ASR profile. The three authoritative
audio manifests and their matching reliability-control splits are validated as
one leakage-free dataset contract.

## 6. Video extraction

Video freezes audio once per split, then reuses it for visual candidates.
Project defaults resolve the selected ASR, selected document image parser,
manifest, and frozen-ASR artifact path.

```powershell
python -m experiments.benchmarks.extraction.video.run --profile smoke
python -m experiments.benchmarks.extraction.video.run --profile preflight
python -m experiments.benchmarks.extraction.video.run --profile development --phase frozen-asr
python -m experiments.benchmarks.extraction.video.run --profile development --phase fixed
python -m experiments.benchmarks.extraction.video.run --profile development --phase scene
```

The default smoke command creates each device-specific frozen-ASR artifact
before running that device's visual candidates. Pass an explicit `--phase` only
to debug one half of that sequence.

Review the scene parent, set `selected_scene_threshold` and
`selected_scene_source_run_id` in the video protocol, increment its version,
rerun video preflight, and regenerate frozen ASR because the protocol checksum
changed. Then run:

```powershell
python -m experiments.benchmarks.extraction.video.run --profile preflight
python -m experiments.benchmarks.extraction.video.run --profile development --phase hybrid
```

Create `video-validation.json`, regenerate the validation frozen-ASR artifact,
and run the finalists. After validation, create `video-locked.json` with exactly
one winner and repeat on locked data:

```powershell
python -m experiments.benchmarks.extraction.video.run --profile validation --phase frozen-asr
python -m experiments.benchmarks.extraction.video.run --profile validation --phase all
python -m experiments.benchmarks.extraction.video.run --profile locked --phase frozen-asr
python -m experiments.benchmarks.extraction.video.run --profile locked --phase all
```

Frozen-ASR reuse rejects any manifest, video-protocol, audio-protocol, or sample
identity mismatch. Every visual candidate references the artifact and never
invokes ASR.

## 7. Chunking–embedding

```powershell
python -m experiments.benchmarks.rag.chunking_embedding.run --profile smoke
python -m experiments.benchmarks.rag.chunking_embedding.run --profile preflight
python -m experiments.benchmarks.rag.chunking_embedding.run --profile development
python -m experiments.benchmarks.rag.chunking_embedding.run --profile validation
python -m experiments.benchmarks.rag.chunking_embedding.run --profile locked
```

Development runs qualified members of the full matrix. Create
`chunking-embedding-validation.json` after development and
`chunking-embedding-locked.json` with exactly one pair after validation. The
locked result is reporting-only.

## 8. Retrieval–reranking

Retrieval uses the selected chunking–embedding pair from
`chunking-embedding-locked.json` by default:

```powershell
python -m experiments.benchmarks.rag.retrieval_reranking.run --profile smoke
python -m experiments.benchmarks.rag.retrieval_reranking.run --profile preflight
python -m experiments.benchmarks.rag.retrieval_reranking.run --profile development
python -m experiments.benchmarks.rag.retrieval_reranking.run --profile validation
python -m experiments.benchmarks.rag.retrieval_reranking.run --profile locked
```

Development crosses Dense, BM25, and RRF with no reranker and the four learned
rerankers. The three `retriever|none` candidates own reusable top-20 pools.
Validation and locked execute exactly the candidates in
`retrieval-reranking-validation.json` and
`retrieval-reranking-locked.json`; a required owner pool is prepared as an
input, not added as another finalist child.

Pool artifacts and paired-comparison Parquet/CSV files are attached to the
comparison parent. Exact NumPy dense search and local BM25 are benchmark
controls, not production indexes.

## 9. Vector database

Start the four servers, run the CPU-only profiles, and stop them afterward:

```powershell
docker compose -f experiments/benchmarks/vectordb/compose.yml up -d
python -m experiments.benchmarks.vectordb.run --profile smoke
python -m experiments.benchmarks.vectordb.run --profile development
python -m experiments.benchmarks.vectordb.run --profile validation
python -m experiments.benchmarks.vectordb.retrieval_run --profile validation
docker compose -f experiments/benchmarks/vectordb/compose.yml down
```

The server-finalist decision defaults to `vector-database-validation.json` and
selects one or more development-qualified servers for validation. Complete
retrieval consumes those finalists together with the locked chunking–embedding
and retrieval–reranking decisions. After reviewing all validation evidence,
record exactly one selected server in `vector-database-locked.json` for Final
RAG.

## 10. Generation and Final RAG

Generation uses frozen evidence contexts:

```powershell
python -m experiments.benchmarks.rag.generation.run --profile smoke
python -m experiments.benchmarks.rag.generation.run --profile development
python -m experiments.benchmarks.rag.generation.run --profile validation
```

Its smoke command runs CPU and CUDA independently. Authoritative runs use the
protocol CUDA contract. Validation reads `generation-validation.json`. After
reviewing validation, record the selected generator in `generation-locked.json`;
this is a transition decision consumed by Final RAG, not another generation
profile.

Final RAG composes the locked retrieval choice and selected generator by
default:

```powershell
python -m experiments.benchmarks.rag.final.run --profile smoke
python -m experiments.benchmarks.rag.final.run --profile development
python -m experiments.benchmarks.rag.final.run --profile validation
```

Export anonymous validation answers, enter judgments, and import them:

```powershell
python -m experiments.benchmarks.review export FINAL_RAG_VALIDATION REVIEW.csv
python -m experiments.benchmarks.review import REVIEW.csv
```

Run the selected complete system once after review:

```powershell
python -m experiments.benchmarks.rag.final.run --profile locked `
  --review-results REVIEW.results.json --confirm-locked-test
```

The complete server-backed Final RAG path remains pending until the Final RAG
runner accepts the selected vector-server decision. Do not present its current
in-process retrieval result as server-backed confirmation.

## 11. Interpret failures and artifacts

Every parent stores the plan, resolved protocols, input checksums, provenance,
qualification or decision inputs, completion state, aggregates, and confidence
intervals. Each candidate child stores executed parameters, metrics, errors,
and per-sample artifacts. Local artifacts are written atomically beneath the
configured artifact root.

A quality comparison is usable only when all planned candidate children and
required metrics complete. Smoke proves wiring only. Preflight proves hardware
eligibility only. Development supports finalist selection, validation supports
the final choice, and locked reports the frozen estimate. No benchmark runner
edits production configuration or chooses a winner automatically.
