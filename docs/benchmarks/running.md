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

The separate Generation and Vector Database locked stages and the locked-only
Final RAG workflow below define the approved target interface. Runner alignment
for those stages, the revised generation scoring/timing rules and full-development
workload, frozen vector-index finalist selection, whole-device CUDA memory
measurement, linear evidence-count nDCG, and shared answerable-question
eligibility for alpha-nDCG remains pending. Documentation changes alone do not
implement these contracts or enable the new commands. The current retrieval
scorer still uses binary nDCG gains and narrower alpha-nDCG eligibility.
The revised extraction first-attempt failure/VRAM/display-name rules, ASR
empty/repeatability contract, video occurrence-aware duplication/reliability,
expanded frozen-ASR reporting, grouped video parents, standalone full data
validation, and automatic validated-input verification also require
implementation. Planned validator commands are identified below; existing
runners still use inline data checks.

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
exact checks, example, and planned module layout. Until implementation, retain
current inline checks and manual reference/split review rather than treating a
missing validator as certification.

Planned preparation commands (not yet executable):

```powershell
python -m experiments.benchmarks.validate audio --profile development
python -m experiments.benchmarks.validate audio --profile all
python -m experiments.benchmarks.validate document --profile all
python -m experiments.benchmarks.validate all --profile all
```

One domain validator serves its applicable profiles; the coordinator uses the
same checks for all benchmarks. Missing required inputs make a validation batch
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
a completed upstream parent. Sample-attempt failures remain visible in that
evidence rather than being confused with an incomplete candidate run:

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

Definitive hardware exclusions (`vram_limit_exceeded`, `gpu_oom`, or
`offload_detected`) are omitted from development and remain in its provenance.
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
GPU's identity, idle memory use, and total/free capacity. Monitor raw NVML
device `memory.used` from before model loading through the final inference;
record `nvml-device-total` and keep all samples. Do not subtract the idle baseline
or infer process bytes from it. For capped model-backed paths, stop the worker
if a sampled device total exceeds `3,584 MiB`; document/video visual paths report
the peak without this shared cap. Offload inspection remains a separate check.
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
- Confidence intervals are calculated after execution by resampling saved
  independent document results 10,000 times with seed `42`. Quality intervals
  use first-attempt scores; repeatability/failure intervals use each document's
  three-attempt values. Resampling does not run additional inference.

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
omits their scalar keys. Implement these approved changes before treating runs
as evidence under this revised contract.

## 6. Video extraction

**Approved grouped interface, pending runner alignment:** normal phase execution
prepares all shared audio and then runs visual candidates under one comparison
parent. Do not assume current independent `--phase frozen-asr` invocations already
create this hierarchy. Project defaults resolve the selected ASR, selected image
parser, phase manifest, and frozen-artifact location.

Target commands:

```powershell
python -m experiments.benchmarks.extraction.video.run --profile smoke
python -m experiments.benchmarks.extraction.video.run --profile preflight
python -m experiments.benchmarks.extraction.video.run --profile development --phase scene
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

## 9. Vector databases

Start the four servers, run the CPU-only profiles, and stop them afterward:

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

Generation uses frozen evidence contexts:

```powershell
python -m experiments.benchmarks.rag.generation.run --profile smoke
python -m experiments.benchmarks.rag.generation.run --profile preflight
python -m experiments.benchmarks.rag.generation.run --profile development
python -m experiments.benchmarks.rag.generation.run --profile validation
python -m experiments.benchmarks.rag.generation.run --profile locked
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
