# Shared benchmark methodology

[Benchmark program](../README.md#experiments) · [Metric definitions](metrics.md) ·
[Run commands](running.md) · [Candidate rationale](model-selection.md)

The experiments form a sequence. First compare extraction components. Separately,
compare chunker/embedding pairs, then retrieval methods and vector servers.
Generation is tested on fixed evidence so retrieval cannot influence it. Each
component benchmark reports its selected configuration on its own locked test.
Final RAG then reports one already-selected complete system, rather than running
another candidate comparison.

```text
1. Document extraction: configuration and architecture comparisons on development,
   finalists on validation, and one frozen route per source type on locked test
2. Audio extraction: development, validation, and one selected ASR on locked test
1 + 2 --> 3. Video keyframes with parser and ASR frozen

4. Chunking x embedding --> 5. Retrieval and reranking
5 --> 6. Real vector-server retrieval

7. Generation on fixed evidence (independent of retrieval)

Each component: development -> validation -> one frozen component locked report

selected extraction + chunking/embedding + retrieval + server + generator
--> extraction-to-RAG confirmation on separate non-locked data
--> one Final RAG locked report, followed by reporting-only human review
```

Document extraction, audio, chunking/embedding, the synthetic
vector-server workload, and generation can start independently. Video waits for
a document parser and ASR. Retrieval waits for chunking/embedding. Real server
retrieval waits for a retrieval stack. Final RAG waits for one selected server,
retrieval stack, and generator.

## Benchmark methodologies

The shared rules on this page apply alongside each benchmark's specific contract.

| Benchmark | Methodology | Metrics |
|---|---|---|
| Document extraction | [Methodology](extraction/document/methodology.md) | [Metrics](extraction/document/metrics.md) |
| Audio extraction | [Methodology](extraction/audio/methodology.md) | [Metrics](extraction/audio/metrics.md) |
| Video extraction | [Methodology](extraction/video/methodology.md) | [Metrics](extraction/video/metrics.md) |
| Chunking and embedding | [Methodology](rag/chunking_embedding/methodology.md) | [Metrics](rag/chunking_embedding/metrics.md) |
| Retrieval and reranking | [Methodology](rag/retrieval_reranking/methodology.md) | [Metrics](rag/retrieval_reranking/metrics.md) |
| Generation | [Methodology](rag/generation/methodology.md) | [Metrics](rag/generation/metrics.md) |
| Vector databases | [Methodology](vectordb/methodology.md) | [Metrics](vectordb/metrics.md) |
| Final RAG and human review | [Methodology](rag/final/methodology.md) | [Metrics](rag/final/metrics.md) |

## Benchmark execution profiles

Each profile consumes reviewed inputs. The shared
[data-validation guide](data-validation.md#preparation-and-execution) distinguishes
full data validation, run separately during preparation, from automatic
validated-input verification before each profile invocation. Full validation
checks the data rules and records the checked inputs; verification confirms that
the current file contents and applicable requirements still match that successful
report, without rerunning the full validator. Changed or unverified inputs stop
execution before candidates load. Each profile needs a report covering its own
inputs; a development report does not certify a different validation dataset.
Those checks are outside model loading, warmup, measured latency, and candidate
resource monitoring. Valid inputs do not make every metric applicable to every
row; task eligibility and prediction failures follow the metric contracts.
The standalone validator/report-reuse interface is approved but still requires
implementation; existing inline runner checks remain in place meanwhile.
The benchmark `validation` profile means evaluating finalists on unseen data,
not either of these data checks.

An execution profile controls **which candidates may run, which dataset split
they may see, and what decisions the result may support**. It is not a model
architecture, quality level, or hardware preset. Candidate/runtime profiles,
such as a Docling parser configuration or an ASR decoding configuration, are a
different concept and always run inside one of these execution profiles.

The component-benchmark lifecycle is one-way:

```text
smoke-cpu + smoke-cuda -> preflight -> development -> validation -> locked
```

`preflight` is a hardware-qualification phase, not a quality-evaluation
profile. The four evaluation profiles remain `smoke`, `development`,
`validation`, and `locked`.

Every independently evaluated component has its own locked report, including
Generation and Vector Database. Final RAG is different: its components and
complete-system settings are already selected, so it has only one locked
evaluation, not another smoke, development, or validation comparison.

All choices affecting a shared locked manifest must be frozen before its first
use by any benchmark. Component locked results cannot inform later system
selection. If further selection is needed after those results are inspected,
Final RAG requires a separate untouched holdout.

| Phase or profile | Data | Candidates | Purpose | Result may be used for |
|---|---|---|---|---|
| `smoke` | Tiny committed fixtures | The protocol's smoke roster | Run separate CPU and CUDA parents to catch loading, wiring, schema, scoring, artifact, and device-path errors cheaply. | Debugging only; never ranking, tuning, qualification, or selection. |
| `preflight` | Reviewed development/stress inputs | Every declared candidate | Qualify the exact model, protocol, software locks, CUDA device, dtype, batch size, and supported input envelope. | Hardware eligibility for authoritative execution while the qualification identity and input envelope still match; never quality selection. |
| `development` | Development manifest | Matching preflight's qualified candidates; declared CPU-only server configurations for Vector Database | Compare alternatives, inspect failures, and make all tuning or shortlist decisions. | An engineer-reviewed finalist decision for validation. |
| `validation` | Unseen validation manifest | Only finalists recorded from a completed development run | Test whether the development conclusion holds on unseen data without reopening the search. | An engineer-reviewed final component decision. |
| `locked` | Untouched locked-test manifest | Exactly one fully frozen selection | Produce the final unbiased estimate after every model, setting, and policy decision is fixed. | Reporting only; never further tuning or reselection. |

The profiles answer different questions, so a later profile does not merely mean
"a bigger run." `development` asks what should advance; `validation` asks
whether that choice generalizes; `locked` estimates the performance of the one
system that will be reported. If validation exposes a problem, the work returns
to development and starts a new benchmark version; a newly held-out validation
set is required before another validation claim. The observed validation result
must not be used to quietly tune and rerun the same evaluation. Any post-lock
change to data, models, settings, metrics, or protocol requires a new benchmark
version and a new untouched locked-test set.

For applicable model-backed suites, one `--profile smoke` command launches
independent `smoke-cpu` and `smoke-cuda` parents. Neither path may fall back to the other device. The
optional `--device cpu|cuda|both` argument only narrows smoke for debugging;
model-backed development, validation, and locked use their frozen backend-specific
device and dtype rules and reject overrides that violate that contract. When CPU and CUDA require different numeric
types, the smoke profile records an explicit dtype for each device; the current
embedding, retrieval, and generation CUDA smoke paths use the same
FP16 contract as their authoritative CUDA execution.

Vector Database is CPU-only: it has one smoke execution and no CUDA preflight.
Its server/configuration roster advances through conformance checks and reviewed
decisions. Document locks one routing policy containing a frozen route for each
source type, rather than forcing PDF, image, and native DOCX through one parser.
Final RAG has only its locked evaluation. These suite-specific contracts take
precedence over the generic lifecycle above.

## Preflight and measured-run lifecycles

Preflight answers one question: can the exact candidate execute the supported
workload on the target CUDA hardware without exceeding its resource contract or
moving model state to CPU, disk, or meta storage?

Each candidate receives a fresh process and follows this sequence:

```text
start process-tree and GPU monitoring before model construction
-> load the exact pinned model and runtime components
-> inspect device placement
-> run one frozen stress input near the upper supported workload boundary
-> synchronize CUDA and collect the final resource sample
-> inspect device placement again
-> stop monitoring, write the qualification report, and terminate the worker
```

The stress input is demanding but valid and comes only from reviewed development
or dedicated stress data. Preflight uses no warmup: the first inference is the
qualification event, so its initial allocations, offloading, peak VRAM, and OOM
behavior must remain visible. It produces hardware evidence rather than quality
or steady-state latency evidence. Because the worker terminates, none of its
runtime state carries into development.

Preflight has one parent and one child per candidate in the benchmark's normal
MLflow experiment. ASR, embedding, learned-reranker, generation, and video-ASR
workers are stopped when sampled total memory on the assigned GPU exceeds
3,584 MiB. Document and video visual parsers instead report VRAM without a shared
cap. Their loaded backend state is inspected
through available parameters and buffers, Paddle places, Hugging Face device
maps, Accelerate hooks, ONNX execution providers, and disk/meta placement. A
backend whose placement cannot be observed is blocked as
`placement_unverifiable`. CPU tokenization and media decoding are not offloading.

A measured model-backed `smoke`, `development`, `validation`, or `locked` worker
uses a different lifecycle:

```text
start a fresh worker and resource monitoring
-> start the cold-load timer
-> load and place the candidate
-> synchronize the device and stop the cold-load timer
-> run one complete representative warmup request
-> synchronize the device and validate the warmup output
-> execute the measured requests in the same worker
-> stop monitoring after the final measured request
```

Cold Model-Load Time covers only model and runtime preparation. "Cold" means that
the model is absent from the new process and device; the operating-system disk
cache is not forcibly cleared.

The warmup starts after cold loading has finished and ends after one complete
request, including preprocessing, inference, postprocessing, output validation,
and device synchronization. It prepares first-use inference state such as
kernels, reusable buffers, and runtime caches before steady-state timing.
Its output and latency are excluded from quality and warm-latency aggregates,
while any failure still invalidates the candidate execution. The worker remains
alive, so the measured requests reuse the state initialized by the warmup.

Every measured model-backed profile uses exactly one warmup per candidate,
device, fresh worker, and materially distinct execution path. Preflight always
uses zero. Vector-server startup, index readiness, and warm query measurement
follow the [Vector Database methodology](vectordb/methodology.md), not the model
constructor lifecycle.

`vram_limit_exceeded`, `gpu_oom`, and `offload_detected` are definitive hardware
exclusions. Missing measurement, unverifiable placement, and infrastructure or
execution failures block development; they never silently eliminate a
candidate. Development resolves an exact preflight fingerprint and runs only
its `qualified_candidates`. Validation and locked reuse that qualification only
when candidate identity, model lock, protocols, dependency locks, executable
source-tree hash, Git commit, GPU/driver, device, dtype, batch size, exact stress
manifest, and tested input envelope still match. Identical fingerprints are
expected only for the same qualification contract and inputs; the fingerprint
is an identity key, not a random run identifier.

Readiness is evaluated by each benchmark's required component groups. Video
requires its selected frozen ASR and at least one visual policy; Document
requires at least one viable route for PDF, image, and DOCX. Every preflight
child stores `preflight_candidate.json` with its placement, telemetry, input
validation, and failure evidence. The parent `preflight_report.json` stores the
complete roster and group-readiness decision.

Development, validation, and locked runs use protocol-frozen seeds, retain
per-sample results, and report 95% confidence intervals for eligible
sample-based aggregates. Deterministic suites use seed `42`; sampled generation
uses the aligned seed list `42`, `43`, and `44`. A stage's locked split is used
once for its engineer-selected winner or frozen source-routing policy.
Decision files are written after engineer review; runners validate those files
but never promote candidates automatically.

## How candidates advance between stages

Benchmark progression uses two different records that must not be confused:

- `preflight_report.json` is generated automatically. It records hardware
  qualification and determines which declared candidates are eligible to enter
  development. It does not rank quality or select a preferred candidate.
- An engineer-decision JSON is written manually after reviewing a complete
  development or validation comparison. It records which candidates with
  complete, valid run-level evidence advance and why. It never changes benchmark
  settings or application configuration.

The complete flow is:

```text
reviewed assets + references + frozen manifests
-> standalone full data validation and cross-split checks
-> successful versioned data-validation reports
-> automatic validated-input verification before each invocation

protocol + model lock + validated smoke manifest
-> smoke-cpu and smoke-cuda
-> wiring evidence only; no candidate advances from smoke

protocol + model lock + validated development/stress inputs
-> preflight
-> machine-generated preflight_report.json
-> exact qualified candidate roster

qualified candidates + development manifest
-> development comparison
-> complete development summary.json
-> engineer review
-> <benchmark>-validation.json

selected finalists + validation manifest
-> validation comparison
-> complete validation summary.json
-> engineer review
-> <benchmark>-locked.json

one frozen winner + locked-test manifest
-> locked report
-> no further tuning or reselection from locked results
```

Final RAG consumes the single selected configuration from each component's
validation-winner decision. The engineer freezes their composition, including
top-K, prompt, and context settings, before any applicable locked evaluation.
Offline annotation review is separate from candidate selection. Component locked
reports are evidence about those fixed choices, not inputs to another finalist search.

Project-owned decisions live under `data/benchmarks/decisions/` because they are
reviewed experimental inputs, not executable implementation. Validation
resolves its development-finalist decision automatically; locked resolves its
validation-winner decision automatically. CLI selection arguments may point to
an explicit alternative file, but the same provenance checks still apply. A
decision is treated as immutable after a run consumes it; a changed choice is a
new versioned decision, not an edit to the consumed file. Placeholder decisions
are not committed.

Component transition decisions use this contract:

```json
{
  "schema_version": 1,
  "source_summary": "path/to/completed/source/summary.json",
  "source_run_id": "matching-source-run-id",
  "selected_candidates": ["candidate-id"],
  "selected_by": "reviewer identity",
  "selected_date": "YYYY-MM-DD",
  "reason": "Evidence-based rationale for the selection"
}
```

The runner rejects a decision unless the referenced summary is complete, its
run ID and expected suite/stage/profile match, and every selected candidate is a
child of that run with complete required evidence and successful run-level
validation. A failed sample attempt is distinct from an incomplete or invalid
candidate run; measured failures remain in its quality and reliability evidence.
The protocol sets the permitted finalist count; locked decisions require exactly
one winner. Each consumed decision is uploaded as an MLflow input artifact, and
its content fingerprint is attached to the parent and candidate children.

Document extraction has source-specific routing rather than one parser that
accepts every format. Its development configuration decisions are
`document-pdf-configuration.json` and `document-image-configuration.json`;
architecture-development results produce `document-pdf-validation.json` and
`document-image-validation.json`; validation produces the exact-one
`document-pdf-locked.json` and `document-image-locked.json` decisions. Native
Docling is the fixed DOCX route. The Document locked command consumes both
winner files and evaluates the complete PDF/image/DOCX policy together.

Protocols remain the source of candidate rosters and execution settings. They
are not edited to remove a failed preflight candidate or advance a finalist.
Manifests remain the source of exact samples and split provenance; model locks
remain the source of revisions, local snapshots, and checksums. MLflow summaries
provide the evidence referenced by decisions. This separation prevents a
selection from silently changing the benchmark definition, data, or model
identity.

## Protocol ownership and reproducibility

Each executable benchmark owns one strict, versioned `protocol.yaml` beside its
implementation:

| Benchmark | Protocol |
|---|---|
| Document extraction | `experiments/benchmarks/extraction/document/protocol.yaml` |
| ASR extraction | `experiments/benchmarks/extraction/audio/protocol.yaml` |
| Video extraction | `experiments/benchmarks/extraction/video/protocol.yaml` |
| Chunking and embedding | `experiments/benchmarks/rag/chunking_embedding/protocol.yaml` |
| Retrieval and reranking | `experiments/benchmarks/rag/retrieval_reranking/protocol.yaml` |
| Generation | `experiments/benchmarks/rag/generation/protocol.yaml` |
| Final RAG | `experiments/benchmarks/rag/final/protocol.yaml` |
| Vector database | `experiments/benchmarks/vectordb/protocol.yaml` |

These files own the editable benchmark settings that can change outputs,
eligibility, latency, memory, or failure status: search ranges, parser and decoder
options, cutoffs,
the shared one-warmup rule for measured profiles, repetitions, batch sizes,
statistical settings, hardware gates, and preflight telemetry interval, polling
interval, qualification repetitions, and worker timeout. Preflight warmups are
always zero. Fixed metric formulas, schema checks, model-architecture facts,
numerical tolerances, and deterministic algorithm rules remain code invariants;
the metric contract and source provenance identify the definitions used.
Their schemas reject missing, unknown, contradictory, and non-finite values.
The resolved protocol has a stable checksum. Parent fingerprints include all
composed protocol checksums; workers verify the version, checksum, and resolved
payload before loading a model or processing data.

Every runner uses the same evaluation-profile names directly: `smoke`,
`development`, `validation`, and, where the benchmark has a final holdout stage,
`locked`. Applicable model-backed suites additionally accept the `preflight`
phase.
Legacy `standard` and `full` spellings are rejected so command lines, run plans,
decision provenance, protocol settings, and MLflow artifacts use one vocabulary.

Configuration files have separate responsibilities:

- benchmark `protocol.yaml` files own executable settings and any candidate
  roster that is genuinely configurable; code defines supported adapters and
  generated candidate combinations. There are no separate candidate registries.
- `data/benchmarks/models/selected.json` stores pinned revisions, snapshot
  locations, and checksums.
- dataset manifests store samples, splits, annotations, and data provenance.
- Generated data-validation reports under
  `artifacts/benchmarks/data-validation/` attest to the exact checked inputs and
  validation requirements; they are not selection or hardware decisions.
- `config/base.yaml` stores provisional application behavior. Promotion is a
  deliberate manual edit after review; no benchmark modifies it.

Every parent run uploads the source YAML and writes a resolved
`<name>_protocol.json`. Parent and child MLflow runs record protocol versions,
checksums, and the resolved settings they executed. Historical artifacts remain
valid evidence for their recorded checksum, but new runners do not reinterpret
old protocol formats.

Every comparison gives its candidates the same samples. MLflow stores the exact
settings, revisions, data checksum, hardware, aggregate metrics, confidence
intervals, and per-sample results. The engineer chooses what continues; the
runner never chooses a winner or changes the application configuration.

## Shared MLflow lifecycle

Each benchmark has its own MLflow experiment. A parent represents one fair
comparison or qualification invocation; candidate children represent complete
configurations. Video also has an explicitly typed shared input-preparation
child for frozen ASR; it is not a competing visual candidate. Runs are not nested by extraction/RAG category because that would
mix unrelated candidate sets and metric contracts in the same experiment.

```text
EduMind / <Benchmark>
|- parent: smoke-cpu
|  `- child per smoke candidate
|- parent: smoke-cuda
|  `- child per smoke candidate
|- parent: preflight
|  `- child per declared candidate
|- parent: development
|  `- child per qualified candidate
|- parent: validation
|  `- child per selected finalist
`- parent: locked
   `- one selected candidate child
```

Only applicable phases appear. Vector Database omits CUDA smoke and preflight
but retains its own locked report. Final RAG has only a locked parent with one
frozen complete-system child. Document uses source-specific parents: its locked
invocation reports the selected PDF route, selected image route, and fixed DOCX
route as one frozen routing policy, not three competing winners. See the
[Document run structure](extraction/document/methodology.md#mlflow-result-structure).
Video groups its frozen-ASR preparation child and visual candidates under one
comparison parent. The preliminary scene-selection comparison has its own
`development-scene-selection` parent; the main `development` parent compares the
final fixed/scene/hybrid roster. Document keeps source-format-specific and
configuration/parser comparison parents rather than an extra phase wrapper.

Human-readable run names omit timestamps and redundant experiment names.
MLflow run IDs, recorded execution timestamps, protocol checksums, and local
artifact identifiers still distinguish invocations. Repeated parent display
names are allowed; never resolve provenance by display name alone. Full candidate
IDs remain in parameters even when display labels are shortened.

Every parent and child is tagged with the benchmark, profile, concrete phase
(`smoke-cpu`, `smoke-cuda`, or the authoritative profile), run type, device,
dataset checksum, protocol checksums, exact qualification identity, and a
fingerprint of any engineer decisions. The parent stores the complete plan and
comparison artifacts; a child stores one candidate's executed settings,
metrics, resources, and per-sample evidence.

Until hardware qualification is complete, every model-backed benchmark
processes one inference input at a time. Runtimes that expose a batch setting
use batch size `1`; the remaining document and video paths execute samples
sequentially. Non-model storage or transport batching is not an inference
setting. A later protocol revision may increase a component's common batch size
only after every candidate in that comparison passes the same recorded hardware
gate.

Authoritative development, validation, and locked comparisons for ASR, embedding,
learned reranking, and generation use the laptop's RTX 3050 through CUDA. Each
stage freezes one supported 16-bit dtype, keeps the whole active model on that
GPU, and forbids CPU fallback, CPU/GPU offload, automatic device splitting, and
quantization. Peak Device VRAM must not exceed `3,584 MiB`. Smoke executes both
CPU and CUDA by default, but neither result is selection evidence.
Document and video parser backends follow their separately recorded lifecycle
and device contracts because not every parser runtime exposes the same backend.
Final RAG is intentionally outside preflight because its component candidates
have already been qualified separately; its ordinary runtime resource gates
still apply. Vector Database is CPU-only and has neither CUDA smoke nor
preflight.

CUDA memory is sampled as raw NVML `memory.used` for the assigned GPU, using
`nvml-device-total`. Monitoring starts before loading and includes cold load,
warmup, and measured inference; preflight instead includes loading and its
retained first stress inference. Report the largest observed device total, not
per-process bytes or an increase above baseline. Record GPU identity, idle
memory, total/free memory, and timestamped samples. The idle baseline is not
subtracted: driver and desktop allocations consume the same physical budget.
The 3,584 MiB device-total limit leaves a nominal 512 MiB margin on a 4,096 MiB
GPU at the observed samples, not a separate 3,584 MiB allowance for model weights.

Close other GPU workloads before running and keep the background baseline
stable across candidates. If unrelated GPU activity changes during a comparison,
the affected measurement must be rerun under a controlled environment rather
than attributed to the candidate. Whole-device VRAM does not prove model
placement; the independent pre/post-inference offload inspection still applies.
Document and video visual backends report this same measurement without an
artificial shared cap. Video's frozen ASR is measured separately and unloaded
before a fresh visual worker. Qualification evidence from a process-attributed
or baseline-delta memory policy is not reusable under the device-total contract.

When a stage declares paired candidate comparisons, they are analysis artifacts
rather than new metrics. They are calculated from aligned per-sample results;
bootstrap comparisons resample the same sample IDs for both candidates instead
of comparing unrelated aggregate values. The applicable pairs, stored fields,
and artifact placement are defined in that stage's methodology section.

## Shared uncertainty procedure

Every applicable development, validation, and locked metric uses the same
resampling procedure, with the independent unit appropriate to its benchmark:

| Benchmark | Independent resampling unit |
|---|---|
| Document | Source document, retaining all pages and capture variants. |
| ASR | Speech or control clip; related clips remain grouped if the reviewed manifest defines their common source as the independent unit. |
| Video | Video; related excerpts remain grouped by the reviewed original source when required. |
| Chunking–embedding, retrieval–reranking, generation, Final RAG, and real vector-server retrieval quality | Source document with all eligible questions, repeated outputs, and pair results. |
| Vector-database ANN workloads | Query with its aligned requests, within one declared workload cell. |

Resample complete units with replacement 10,000 times using bootstrap seed `42`,
recalculate the metric's existing aggregate on every draw, and take the 2.5th
and 97.5th percentiles. Preserve each benchmark's defined statistic: document
extraction uses document-macro quality, ASR uses corpus WER/CER, and other
benchmarks retain their own quality and latency definitions. Paired comparisons
resample aligned units together and recalculate both candidates on each draw.
Repeated attempts are retained inside their unit, not treated as additional
independent evidence. These resamples perform arithmetic on saved measurements,
not extra model inference. More resamples stabilize the numerical calculation;
only additional representative independent inputs add new source evidence.

The minimum independent support for each conditional or slice interval is
frozen after manifest review and before authoritative evaluation. An undefined
draw is excluded only from that metric's bounds, with defined/undefined draw
counts recorded. If support is insufficient, retain the point estimate and null
bounds with a reason in artifacts; omit only the MLflow scalar bounds. Smoke,
configuration counts, one cold-load observation, and observed resource peaks
do not receive authoritative intervals.

## Shared MLflow metric convention

The benchmark-specific methodology pages list only each metric's base key. CUDA keys named
`peak_vram_mb`, `peak_vram_mib`, or `peak_visual_vram_mb`, including their
`operational.` forms, report Peak Device VRAM under `nvml-device-total`.
Their recorded measurement method distinguishes them from historical process
or delta measurements. A sample-based aggregate uses one consistent convention:

```text
<metric_key>
<metric_key>.sample_count
<metric_key>.ci_lower
<metric_key>.ci_upper
```

For example, an eligible audio aggregate appears as:

```text
word_error_rate
word_error_rate.sample_count
word_error_rate.ci_lower
word_error_rate.ci_upper
```

`sample_count` records how many independent samples contributed to the point
estimate. The CI keys exist only when the metric qualifies for a 95% confidence
interval under [metrics.md](metrics.md). Their absence means that no interval
was calculated; it never means zero. A numerically degenerate calculated
interval is flagged in the artifact rather than interpreted as certainty.
Statuses, configuration values, checksums, and one-off measurements such as a
cold load or observed resource peak do not receive these suffixes.

Every automated benchmark value has one of six roles. **Primary** metrics answer
the experiment's central question. **Secondary** metrics explain or qualify a
primary result. **Diagnostic** metrics expose a specific behavior or failure
mode without deciding the winner. **Operational** metrics measure latency,
throughput, and resources. A **validity gate** must pass before a run may be used
as evidence; it is not a trade-off metric. A **descriptor** records workload or
storage needed to interpret other results but has no better direction. Evidence
types and document groups are reporting slices, not additional metric roles.
Blinded rubric scores are labeled separately as human judgments.

## Locked test

Every component benchmark reports its selected configuration on its own locked
test. After all selection and non-locked extraction confirmation are complete,
Final RAG runs the one frozen system exactly once on its locked-test manifest.
The selected parser, ASR, chunker, embedding, retrieval method, vector server, generator, prompt, and
context settings cannot change between confirmation and this run.

The locked result is the final unbiased estimate. It is not used for more tuning.
If the system changes after the result is inspected, a new benchmark and locked
dataset version are required.
