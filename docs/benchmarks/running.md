# Benchmark runbook

[Benchmark program](../README.md#experiments) · [Experiment methodology](methodology.md) ·
[Metric definitions](metrics.md) · [Installation](../setup/installation.md)

This is the command reference for running EduMind benchmarks. Run every command
from the repository root in the prepared Python 3.12 environment.

## 1. Prepare the environment

Follow the [installation guide](../setup/installation.md). It is the authority
for dependencies, model snapshots, data preparation, evaluator images, and
vector-server images. A benchmark does not download missing models while it is
running.

The Vector Database locked stage, frozen vector-index finalist selection, and
locked-only Final RAG workflow below retain their implementation boundaries.
Chunking–embedding implements its separate load/build/query-component timings,
retrieved-context descriptors, diagnostic top-five pair score, and execution,
quality, and integrity accounting. CI support requirements remain subject to data
review; unfrozen support gives descriptive points and null bounds, not
authoritative advancement evidence. See its
[implementation boundary](rag/chunking_embedding/methodology.md#implementation-boundary).
Retrieval–reranking implements the corresponding complete-ranking diagnostic,
one authoritative warmup, single-unit alpha-nDCG eligibility, index ownership,
and reporting contract. Generation implements the seven model-mode runtime and
scoring interface; a frozen calibrated judge, reviewed source support, and
prepared model snapshots are required before authoritative comparison.
Document, ASR, and Video provide their extraction metric, failure, repeatability,
resource, and display-name paths. Their standalone data validators and
automatic sealed-report verification are available. Video uses grouped parents
with separate shared-audio preparation and visual children.
Verify the [required retained prediction evidence](metrics.md#prediction-evidence-and-rescoring)
before claiming independent rescoring from a run; summary counts/fingerprints
alone are insufficient. Historical outputs are not migrated to the current
protocol or made rescorable by adding a summary reference.

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

First prepare and review the inputs. The approved workflow has two steps:

- **Full data validation:** you run the standalone validator for a prepared
  dataset version. It checks the data rules and saves a report; rerun it after
  relevant input or validation-requirement changes.
- **Automatic validated-input verification:** each profile invocation checks
  that its current input-file contents and applicable requirements match a
  successful report. It does not repeat full validation. A mismatch stops the
  run before loading any candidate, rather than triggering silent revalidation.

Both steps are outside latency and candidate resource measurement. See the
[data-validation guide](data-validation.md#preparation-and-execution) for the
exact checks, example, and module layout. Full validation is available for the
three extraction benchmarks, chunking–embedding, retrieval–reranking, and
generation; the deferred suites retain their current input checks.
Human annotation review remains
necessary.

Available preparation commands:

```powershell
python -m experiments.benchmarks.validate audio --profile development
python -m experiments.benchmarks.validate audio --profile all
python -m experiments.benchmarks.validate document --profile all
python -m experiments.benchmarks.validate video --profile all
python -m experiments.benchmarks.validate chunking-embedding --profile all
python -m experiments.benchmarks.validate retrieval-reranking --profile all
python -m experiments.benchmarks.validate generation --profile all
python -m experiments.benchmarks.validate all --profile all
```

One domain validator serves its applicable profiles; the coordinator uses the
same domain checks across the implemented benchmarks. Missing required inputs make a validation batch
incomplete, not successful. `all --profile smoke` checks only applicable smoke
fixtures when authoritative data is unavailable. Preflight uses validated
development/stress inputs and never inspects held-out answers. Generated reports
live under `artifacts/benchmarks/data-validation/<benchmark>/<fingerprint>.json`.

Each profile uses its own inputs and matching report; validating development
does not certify the unseen validation or locked datasets. The benchmark
`validation` profile evaluates finalists on unseen data and is distinct from
both preparation checks and automatic validated-input verification.

After valid input preparation, applicable model-backed benchmarks use this order:

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

Vector Database is CPU-only and has no CUDA preflight, but has its own locked
report. Final RAG has only one locked evaluation of the already-selected system;
it does not run another smoke, preflight, development, or validation comparison.

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

A **component decision file** is an engineer-reviewed transition between stages.
It names exact candidates with complete, run-level-valid evidence selected from
a completed upstream parent. Sample-attempt failures remain visible; their
effect on candidate validity follows the suite's failure contract. A recorded
failed attempt is distinct from missing scheduled records:

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

A `*-locked.json` component decision records the winner chosen from validation;
it exists before the component's locked report. Downstream benchmarks consume
that frozen choice, not a selection made from locked-test scores.

Development locates preflight by an exact SHA-256 fingerprint covering the
candidate roster, model revisions and checksums, protocols, software locks,
executable source tree and Git commit, GPU and driver, device, dtype, batch
size, exact stress manifest, and tested input envelope. It never chooses an
arbitrary latest run. Use `--preflight-run-id ID` to select an exact MLflow run,
or `--preflight-report PATH` for local no-MLflow debugging.

Definitive hardware exclusions (`gpu_oom` or `offload_detected`) are omitted from
development and remain in its provenance. `vram_limit_exceeded` additionally
applies to capped non-extraction families; no extraction phase enforces a
3.5 GiB cutoff or threshold warning.
`measurement_unavailable`, unverifiable placement, and infrastructure failures
block development until preflight is rerun successfully.

Each candidate child contains a `preflight_candidate.json` evidence artifact.
The parent `preflight_report.json` records the complete roster and required
component groups. Video is ready only when frozen ASR and a visual policy are
qualified; Document additionally requires viable PDF, image, and DOCX routes.
Every measured model-backed `smoke`, `development`, `validation`, and `locked`
profile uses one warmup per fresh candidate worker and materially distinct execution path.
Preflight uses zero warmups: monitoring starts before loading and continues
through one retained demanding qualification inference so first-use allocation,
offloading, and peak VRAM remain visible. The preflight worker then exits and
cannot warm a later benchmark process.

Where Cold Model-Load Time is reported, a fresh worker times loading and device
placement, ends that timer, runs the one warmup, and only then begins measured
requests. Cold load, warmup, and measured latency are separate lifecycle phases;
neither cold load nor warmup is included in warm Time to First Token or
warm end-to-end request latency. First-item latency, where reported, separately
includes startup/loading and the first complete request. The operating system's
disk cache is not forcibly cleared.
Qualification repetitions, telemetry interval, polling interval, and timeout
are read from the benchmark protocol.

Before a CUDA comparison, close other GPU workloads and record the assigned
GPU's identity, idle memory use, and total/free capacity under the
[shared Peak Device VRAM contract](metrics.md#peak-device-vram).
Extraction reports the peak without an artificial
cap in every phase, including preflight and frozen video ASR. Its workers still
retain OOM, timeout, telemetry, and placement failure evidence. Capped RAG paths
stop workers at their protocol's `3,584 MiB` limit. Offload inspection remains a
separate check.
If unrelated GPU activity changes the baseline during a comparison, rerun the
affected measurement under controlled conditions. A preflight made under the
former process/delta policy cannot qualify this device-total contract.

## 4. Document extraction

Prepare the official table/formula evaluator when the manifest contains positive
reference tables or formulas requiring reconstruction scores. Verified-negative
presence annotations alone do not require an official reconstruction scorer:

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
`document-pdf-locked.json` and `document-image-locked.json`. Produce one frozen
source-routing report on the untouched locked-test manifest:

```powershell
python -m experiments.benchmarks.extraction.document.run --profile locked
```

The locked profile selects the architecture comparison automatically, resolves
both decisions, runs one frozen PDF/image profile per source with three measured
attempts per locked document, and runs native Docling for DOCX under the same
measurement contract. Video consumes the same `document-image-locked.json`
decision. Locked Document
results are reporting-only and cannot change either winner. A `--source`
override is rejected for locked execution so a partial source result cannot be
mistaken for the complete frozen routing policy.

Use `--source pdf|image|docx` to isolate one source type. The default `all`
creates separate parents because the valid candidate sets differ.

### Reading the document results

Development, validation, and locked load each candidate in a fresh worker,
measure cold loading, perform one complete warmup, and execute every document
three times with the same settings and inference seed. Smoke uses one measured
attempt per fixture; preflight has no warmup and does not measure repeatability.

- Attempt 1 is the only quality-scored output. Attempts 2 and 3 are validated,
  timed, and compared for repeatability, not quality-scored again.
- Never substitute a later success for a failed first attempt or erase valid
  first-attempt quality because another attempt failed.
- Repeatability Success Rate compares all three scheduled output pairs;
  Attempt Failure Rate counts failed attempts. `A, A, failure` gives `1/3` for
  both rates while keeping the first `A` as the quality output.
- Quality is document-macro averaged from valid completed first outputs, including
  missed expected objects in those outputs. Failed first attempts have unavailable
  quality; report them and contributing counts beside every aggregate. No reference
  tables/formulas means no reconstruction task; verified-negative detection applies.
- Apply the [shared CI procedure](metrics.md#confidence-intervals) to independent
  documents. Quality uses first-attempt scores; repeatability/failure intervals
  use each document's three-attempt values.

Inspect `samples.parquet` for first-attempt quality and each metric's status,
reason, eligibility, and coverage. Inspect `timings.parquet` and retained outputs
for all attempts, including failures and unexecuted requests. A valid empty
output, a failed extraction, an unavailable evaluator, and interrupted execution
are distinct outcomes. Resolve unavailable required evaluator results before
publishing a complete comparison; quality scoring may be rerun on saved outputs.
Missing telemetry is not zero memory. The full rules and score directions are
in the [document metric reference](extraction/document/metrics.md).

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

The [approved ASR metric contract](extraction/audio/metrics.md) uses only the
designated first measured output for quality, with no replacement after failure.
Development, validation, and locked use three measured speech attempts with the
same seed/settings and one measured attempt per control after the one warmup.
All speech attempts supply pairwise Transcript Repeatability Success Rate;
speech and control attempts supply Attempt Failure Rate. These are evidence
inside each candidate run, not separate candidate-selection rounds. Confidence
intervals resample independent sources, not the three attempts.

Unexpected Empty Transcript Rate and Nonspeech False-Transcription Rate use
valid completed eligible first outputs; crashes remain failure records, not
empty transcripts. WER/CER/components pool completed first-output edit counts;
their raw counts remain artifacts. Both timestamp metrics are unavailable after
first-attempt failure; valid empty timed outputs give coverage zero and unavailable
MAE. Keep planned/contributing reference counts and failures beside these values.
Null fields retain reasons and support counts in artifacts even when MLflow
omits their scalar keys.

## 6. Video extraction

Normal phase execution prepares all shared audio and then runs visual candidates
under one comparison parent. Project defaults resolve the selected ASR, selected image
parser, phase manifest, and frozen-artifact location.

Commands:

```powershell
python -m experiments.benchmarks.extraction.video.run --profile smoke
python -m experiments.benchmarks.extraction.video.run --profile preflight
python -m experiments.benchmarks.extraction.video.run --profile development --phase scene-selection
```

Smoke creates `smoke-cpu` and `smoke-cuda`, each with its own frozen-ASR
preparation child and visual smoke children. No device fallback is allowed.
The preliminary scene command creates `development-scene-selection`, with
shared audio and the qualified scene thresholds. Every unique video is decoded
once and each required ASR window transcribed once during measured preparation.
The separate warmup is excluded. "Once" does not mean one inference for the
entire split.

Review the preliminary scene parent. Set `selected_scene_threshold` and
`selected_scene_source_run_id` in the video protocol and increment its version.
Refresh preflight; the main comparison regenerates frozen ASR because the composed
checksum changed, then evaluates all qualified members of the final nine-policy
roster under that checksum:

```powershell
python -m experiments.benchmarks.extraction.video.run --profile preflight
python -m experiments.benchmarks.extraction.video.run --profile development --phase all
```

Create `data/benchmarks/decisions/video-validation.json` after reviewing the
complete main development comparison. Evaluate its finalists on validation,
then record one validation winner in `video-locked.json`:

```powershell
python -m experiments.benchmarks.extraction.video.run --profile validation --phase all
python -m experiments.benchmarks.extraction.video.run --profile locked --phase all
```

Each parent contains one phase-specific `frozen-asr — <selected candidate>`
preparation child and its visual candidate children. Locked has one visual
competitor, not a new ASR competition. A visual child covers every phase video;
per-video/window/repetition results are artifacts, not nested runs. Complete ASR
models are unloaded before fresh visual workers begin.

Audio artifact reuse checks the actual producer, selected-model decision,
model lock, manifest, composed protocols, and exact sample membership. Retain
per-video/window statuses: failed required windows cannot become complete
transcripts. Reused measurements retain their producing run and execution scope;
do not claim a new cold-load/latency/resource observation without new execution.
No visual child invokes ASR or copies its aggregate metrics.

Visual development/validation/locked use one warmup and three measured attempts
per video, with quality only from attempt 1. Failed first outputs have unavailable
quality; later success cannot replace them. Repeatability and failure rates use
scheduled outcomes. Warm visual timing includes the full selector/frame/parser
path but excludes audio, loading, warmup, input verification, and offline scoring.
Selected-frame counts record selection, not parser acceptance.

The shared audio child reports WER/CER, word-error components, applicable spoken
timestamps, empty/nonspeech behavior, window failures, audio RTF, p50/p95 complete
audio-processing latency, cold load, and RAM/VRAM. Each window is executed once,
so audio repeatability is not measured. Confidence intervals use supported
independent video/source groups, not windows or additional inference. Limited
p95 support retains a descriptive value with null bounds/reason. See
[video metrics](extraction/video/metrics.md) and
[the run hierarchy](extraction/video/methodology.md#mlflow-result-structure).

## 7. Chunking–embedding

Validate the inputs once before running either smoke device. Authoritative
reports require all three reviewed split manifests in the frozen inventory.

```powershell
python -m experiments.benchmarks.validate chunking-embedding --profile smoke
python -m experiments.benchmarks.rag.chunking_embedding.run --profile smoke
python -m experiments.benchmarks.rag.chunking_embedding.run --profile preflight
python -m experiments.benchmarks.rag.chunking_embedding.run --profile development
python -m experiments.benchmarks.rag.chunking_embedding.run --profile validation
python -m experiments.benchmarks.rag.chunking_embedding.run --profile locked
```

Use `python -m experiments.benchmarks.validate chunking-embedding --profile all`
after preparing the authoritative inputs. Every run verifies the matching report
before loading candidates; it does not repeat full annotation validation.

Development runs qualified members of the full matrix. Create
`chunking-embedding-validation.json` after development and
`chunking-embedding-locked.json` with exactly one pair after validation. The
locked result is reporting-only.

### Reading the chunking–embedding results

- Prepare/review canonical sources and evidence separately. The standalone
  chunking–embedding validator and sealed-report reuse are implemented. Each
  profile verifies its matching report before candidate loading or measurement.
- Execute only answerable questions. Pairs share canonical source documents and
  questions, but each builds its own chunk corpus. Shared-manifest unanswerable
  queries are not embedded, timed, or scored.
- Build the corpus once after loading, including semantic boundary work and
  matrix validation in Corpus-Build Time. Run one query warmup after the matrix
  exists, then one measured request per query in smoke or three in authoritative
  profiles with the same seed/settings.
- Under the approved breakdown, record Cold Model-Load Time before build;
  Chunking Time includes semantic-boundary inference, Chunk-Embedding Time
  includes final chunk-vector production, and the enclosing build includes
  assembly/validation. Do not count semantic inference in two components.
- Score only the first measured ranking at both `@3` and `@5`; retain top-20
  artifacts. First-request failure leaves unavailable quality even after later
  success. Later failure does not erase valid first-ranking quality.
- Construct complete-query and embedding/search-component p50/p95 from each
  query's own median completed duration under that timer's boundary,
  excluding failed elapsed times. Preserve all outcomes and failed elapsed times
  in `timings.parquet`. A query with no completed measured request has null
  complete-query latency, not zero; its completed earlier component may remain
  available. Component percentiles are not additive.
- Diagnostic Ranking Repeatability compares all three attempt pairs on ordered
  top-five IDs. Valid ordering changes are not an integrity failure. Keep all
  attempt rankings, not just agreement hashes or first-quality scores.
- Mean/p95 Retrieved Context Tokens@3/@5 use complete first-ranking prefix
  counts with `cl100k_base`, including repeated source occurrences. These describe
  returned volume, not corpus-wide chunk sizes, native limits, or a token budget.
- Read linear evidence-count nDCG with Recall and token Precision. Zero complete
  evidence in the candidate corpus gives nDCG/alpha-nDCG zero. Invalid empty
  rankings give unavailable quality. Alpha-nDCG includes single-unit questions
  and retains raw greedy-baseline exceedances as artifact diagnostics.
- Partial quality estimates stay labelled in artifacts and do not enter normal
  quality scalars, successful leaderboards, or paired selection comparisons.
  Inspect planned/scored/failed/unavailable support, not just surviving scores.
- Distinguish a complete outcome ledger from executed workload coverage and
  complete quality. Explicit unexecuted rows account for missing execution;
  they are neither successes nor observed model failures. Do not infer a blanket
  zero-request-error acceptance rule from legacy validity counters.
- A failed child can retain completed build measurements and observed resource
  peaks. `measurement.resource_scope=partial` labels interrupted windows on the
  existing child; missing CUDA telemetry is unavailable and CPU VRAM inapplicable.
  An unfinished matrix has no final matrix-byte value or partial-bytes metric.
- Apply the [shared CI procedure](metrics.md#confidence-intervals) to whole source
  documents with their dependent queries/attempts; review the suite's
  [CI applicability](rag/chunking_embedding/metrics.md#which-values-receive-an-interval).
  Ranking Repeatability and each query-latency family require frozen source
  support for intervals; single load/build-component observations and token
  descriptors do not receive invented intervals.

See [metric definitions](rag/chunking_embedding/metrics.md) and the
[artifact/MLflow contract](rag/chunking_embedding/methodology.md#mlflow-result-structure)
for status, measurement-boundary, and advancement rules.

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

The parent stores pool-collection metadata in `pool_collections.json` and
paired-comparison Parquet/CSV files. Evaluated pool-owner children store ranked
pool rows in `pool_collection.parquet`; when no owner child is evaluated, its
internally prepared input collection remains referenced by the parent metadata
without inventing an owner run. Exact NumPy dense search and local BM25 are benchmark
controls, not production indexes.

The [metric reference](rag/retrieval_reranking/metrics.md) explains each quality,
latency, pool-size diagnostic, token-workload, and storage value separately.
The following reporting layout is the approved contract; consult the
[implementation boundary](rag/retrieval_reranking/methodology.md#implementation-boundary)
before interpreting current runner output. Retrieval Warm Latency uses target
keys `operational.retrieval_latency_ms_p50/p95`; Full-Stack Warm Latency also
includes learned reranking when present. No-reranker latency/input fields are
inapplicable rather than measured zeros. Index build and storage are owned by
Dense and BM25; RRF references both indexes instead of constructing a third.

Every profile prepares fresh query pools for its own inputs. Ranking
Repeatability compares all measured-attempt pairs over the complete ordered
result and is diagnostic, while pool provenance, live retrieval consistency,
exact permutation, and final ordered-prefix checks remain enforced integrity
checks. No separate pool-match scalar is required.
Its supported Ranking Repeatability interval uses whole-source resampling of
saved pair records, not extra inference. A failed-query count is an observed
outcome count; required execution, quality coverage, and actual integrity gates
determine evidence completeness under the shared accounting rule.

Index owners use `storage.index_size_bytes` and
`operational.index_build_seconds`. RRF uses the same size key for its dependency
total plus `storage.dense_index_size_bytes` and `storage.bm25_index_size_bytes`;
its artifact references both build observations without a third build scalar.
Mean/p95 Retrieved Context Tokens@3/@5 describe the final ranking. Optional
Candidate-Pool Tokens@20 stays in per-query pool artifacts; native Reranker Input
Tokens describes all prepared query-passage pairs. Fixed inventory counts belong
to parameters/input artifacts. Local snapshot size belongs to the generated
model lock, not candidate-run metrics.

## 9. Vector databases

Start the four servers before available CPU-only runs and stop them afterward.
The command sequence below also shows the approved target for frozen-finalist
validation and a component locked report. Those final-stage contracts are not
yet fully implemented; follow the
[vector methodology's implementation status](vectordb/methodology.md#implementation-status)
before interpreting a current run as that report. In particular, validation
must not retune the index settings that it is supposed to confirm.

```powershell
docker compose -f experiments/benchmarks/vectordb/compose.yml up -d
python -m experiments.benchmarks.vectordb.run --profile smoke
python -m experiments.benchmarks.vectordb.run --profile development
python -m experiments.benchmarks.vectordb.run --profile validation
python -m experiments.benchmarks.vectordb.retrieval_run --profile validation
python -m experiments.benchmarks.vectordb.run --profile locked
docker compose -f experiments/benchmarks/vectordb/compose.yml down
```

The server-finalist decision defaults to `vector-database-validation.json` and
selects one or more complete server/index finalists from development, including
their exact tested HNSW settings. The grid search belongs to development;
validation rebuilds and compares only those recorded configurations without
retuning. Complete retrieval consumes those finalists together with the locked
chunking–embedding and retrieval–reranking decisions. After reviewing all
validation evidence, record exactly one selected server/index profile in
`vector-database-locked.json`. Its own locked run reports that frozen
configuration on the reviewed held-out workload. Final RAG consumes the same
choice without further server or index selection.

## 10. Generation and Final RAG

Generation uses frozen evidence contexts. Its implemented model-mode/seed/judge
contract and external readiness requirements are described in the
[generation implementation boundary](rag/generation/methodology.md#implementation-boundary)
and apply before authoritative execution.

```powershell
python -m experiments.benchmarks.rag.generation.run --profile smoke
python -m experiments.benchmarks.rag.generation.run --profile preflight
python -m experiments.benchmarks.rag.generation.run --profile development --judge-records path/to/development-judge-replies.json
python -m experiments.benchmarks.rag.generation.run --profile validation --judge-records path/to/validation-judge-replies.json
python -m experiments.benchmarks.rag.generation.run --profile locked --judge-records path/to/locked-judge-replies.json
```

Its smoke command runs every model-mode configuration independently on CPU and
CUDA. Preflight uses frozen demanding development/stress inputs and produces
hardware qualification only. Development runs every qualified configuration on
the complete frozen development question set; validation reads
`generation-validation.json` and compares only the recorded frozen finalists on
the complete validation set, without introducing configurations or retuning.
Authoritative runs use the protocol CUDA/FP16 contract, batch size `1`, one
warmup, and aligned measured seeds `42`, `43`, and `44`.

Before development, select and configure one semantic judge, freeze its exact
identity and rubric, and verify its human-calibration artifact. The judge runs
after generator timing and supplies
structured labels for Faithfulness, Factual Correctness, Answer Relevancy, and
Repeat Semantic Agreement. Its latency and resources are not attributed to the
generator. After reviewing validation, record exactly one successful model-mode
configuration in `generation-locked.json`. Generation reports that choice on
its own locked frozen-evidence test. Final RAG uses the same generator unchanged.

Set `judge.contract_path` in the generation protocol to the reviewed contract.
`--judge-records` replays replies already obtained under it; it does not call a
provider. A live integration supplies a transport through the Python judge
interface after judge selection. Replay misses or exhausted judge retries leave
evaluation incomplete, never a candidate-quality zero. Generation's tiny
committed fixture is `data/benchmarks/rag/generation-smoke.json`; it is separate
from retrieval's corpus fixture and supplies no selection evidence.

When reading a generation child, inspect `input_compatibility` and its
`input_compatibility.json` artifact first. A failed gate is visible incomplete
execution, not a row of fabricated zero metrics. The detailed
[input contract](data-validation.md#generation-references-and-model-input-compatibility)
explains native prompt checks and how live Final RAG inputs differ.

The [generation metric reference](rag/generation/metrics.md) defines the three
seed outputs' quality aggregation, completed-output Malformed Output Rate,
`reliability.runtime_error_rate`, and completed-stream Decode Throughput.
Check eligible/contributing support and evaluator status before treating a
quality value as complete. Inspect partial execution and monitoring coverage
alongside observed memory peaks; failed-attempt peaks are not discarded.
Citation/claim counts and finish-reason inventories are artifacts, not extra
summary metrics. Shared null, dependency, and CI interpretation belongs to the
[general metric guide](metrics.md#dependent-metrics-and-partial-observations).

Freeze one complete system in `final-rag-locked.json`, referencing the selected
component decisions and the chosen top-K, prompt, and context settings. Finish
non-locked integration and extraction-confirmation checks before locked
execution. Final RAG runs only this selected system:

```powershell
python -m experiments.benchmarks.rag.final.run --profile locked --confirm-locked-test
```

After locked execution, export the reporting-only human-review sample, enter
judgments, and import them:

```powershell
python -m experiments.benchmarks.review export FINAL_RAG_LOCKED REVIEW.csv
python -m experiments.benchmarks.review import REVIEW.csv
```

The review describes the frozen system; it does not select another winner or
authorize another locked run. If component reports share locked data with Final
RAG, freeze the entire system before the first component locked report. Otherwise
use a separate untouched Final RAG holdout.

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
