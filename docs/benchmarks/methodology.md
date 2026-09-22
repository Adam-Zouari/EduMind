# How the EduMind experiments work

[Benchmark overview](overview.md) · [Metric definitions](metrics.md) ·
[Run commands](running.md) · [Candidate rationale](model-selection.md)

The experiments form a sequence. First compare extraction components. Separately,
compare chunker/embedding pairs, then retrieval methods and vector servers.
Generation is tested on fixed evidence so retrieval cannot influence it. Finally,
combine the approved retrieval and generation profiles and review their answers.

```text
1. Document extraction: configuration on development, architectures on validation,
   and one frozen route per source type on locked test
2. Audio extraction: qualified candidates on development, finalists on validation
1 + 2 --> 3. Video keyframes with parser and ASR frozen

4. Chunking x embedding --> 5. Retrieval and reranking
5 --> 6. Real vector-server retrieval

7. Generation on fixed evidence (independent of retrieval)

approved server + retrieval finalists + generator finalists
--> 8. Final RAG systems on development, finalists on validation + blinded review
--> 9. Extraction-to-RAG confirmation on separate non-locked data
--> 10. Exactly one locked-test run
```

Document extraction, audio, chunking/embedding, the synthetic
vector-server workload, and generation can start independently. Video waits for
a document parser and ASR. Retrieval waits for chunking/embedding. Real server
retrieval waits for a retrieval stack. Final RAG waits for one selected server,
retrieval stack, and generator.

### Benchmark execution profiles

An execution profile controls **which candidates may run, which dataset split
they may see, and what decisions the result may support**. It is not a model
architecture, quality level, or hardware preset. Candidate/runtime profiles,
such as a Docling parser configuration or an ASR decoding configuration, are a
different concept and always run inside one of these execution profiles.

The benchmark lifecycle is one-way:

```text
smoke-cpu + smoke-cuda -> preflight -> development -> validation -> locked
```

`preflight` is a hardware-qualification phase, not a quality-evaluation
profile. The four evaluation profiles remain `smoke`, `development`,
`validation`, and `locked`.

| Phase or profile | Data | Candidates | Purpose | Result may be used for |
|---|---|---|---|---|
| `smoke` | Tiny committed fixtures | The protocol's smoke roster | Run separate CPU and CUDA parents to catch loading, wiring, schema, scoring, artifact, and device-path errors cheaply. | Debugging only; never ranking, tuning, qualification, or selection. |
| `preflight` | Reviewed development/stress inputs | Every declared candidate | Qualify the exact model, protocol, software locks, CUDA device, dtype, batch size, and supported input envelope. | Hardware eligibility for development only. |
| `development` | Development manifest | Candidates marked `qualified` by the matching preflight | Compare alternatives, inspect failures, and make all tuning or shortlist decisions. | An engineer-reviewed finalist decision for validation. |
| `validation` | Unseen validation manifest | Only finalists recorded from a completed development run | Test whether the development conclusion holds on unseen data without reopening the search. | An engineer-reviewed final component or complete-system decision. |
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

One `--profile smoke` command launches independent `smoke-cpu` and
`smoke-cuda` parents. Neither path may fall back to the other device. The
optional `--device cpu|cuda|both` argument only narrows smoke for debugging;
development, validation, and locked use the CUDA device and dtype frozen in the
protocol and reject CPU overrides. When CPU and CUDA require different numeric
types, the smoke profile records an explicit dtype for each device; the current
embedding, retrieval, generation, and Final RAG CUDA smoke paths use the same
FP16 contract as their authoritative CUDA execution.

Preflight has one parent and one child per candidate in the benchmark's normal
MLflow experiment. A fresh CUDA worker performs real inference while the parent
continuously samples its process tree. ASR, embedding, learned-reranker, and
video-ASR workers are stopped when sampled VRAM exceeds 3,584 MiB. Document and
video visual parsers instead use their backend-specific placement contracts and
report VRAM without a shared cap. Framework placement inspection checks all
available parameters and buffers, Hugging Face device maps, Accelerate hooks,
and disk/meta placement. CPU tokenization and media decoding are not offloading.

`vram_limit_exceeded`, `gpu_oom`, and `offload_detected` are definitive hardware
exclusions. Missing measurement, unverifiable placement, and infrastructure or
execution failures block development; they never silently eliminate a
candidate. Development resolves an exact preflight fingerprint and runs only
its `qualified_candidates`. Validation and locked reuse that qualification only
when candidate identity, model lock, protocols, dependency locks, GPU/driver,
device, dtype, batch size, and input envelope still match.

Development, validation, and locked runs use seed 42, retain per-sample results,
and report 95% confidence intervals for eligible sample-based aggregates. A
stage's locked split is used once for its engineer-selected winner or frozen
source-routing policy.
Decision files are written after engineer review; runners validate those files
but never promote candidates automatically.

### How candidates advance between stages

Benchmark progression uses two different records that must not be confused:

- `preflight_report.json` is generated automatically. It records hardware
  qualification and determines which declared candidates are eligible to enter
  development. It does not rank quality or select a preferred candidate.
- An engineer-decision JSON is written manually after reviewing a complete
  development or validation comparison. It records which successful candidates
  advance and why. It never changes benchmark settings or application
  configuration.

The complete flow is:

```text
protocol + model lock + smoke manifest
-> smoke-cpu and smoke-cuda
-> wiring evidence only; no candidate advances from smoke

protocol + model lock + reviewed development inputs
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
-> no further tuning, selection, or promotion
```

Project-owned decisions live under `data/benchmarks/decisions/` because they are
reviewed experimental inputs, not executable implementation. Validation
resolves its development-finalist decision automatically; locked resolves its
validation-winner decision automatically. CLI selection arguments may point to
an explicit alternative file, but the same provenance checks still apply. A
decision is treated as immutable after a run consumes it; a changed choice is a
new versioned decision, not an edit to the consumed file. Placeholder decisions
are not committed.

Every decision uses this contract:

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
successful child of that run. The protocol sets the permitted finalist count;
locked decisions require exactly one winner. Each consumed decision is uploaded
as an MLflow input artifact, and its content fingerprint is attached to the
parent and candidate children.

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

### Protocol ownership and reproducibility

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

These files own every setting that can change outputs, eligibility, latency,
memory, or failure status: search ranges, parser and decoder options, cutoffs,
warmups, repetitions, batch sizes, statistical settings, and hardware gates.
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
- benchmark protocols store executable experimental behavior.
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

### Shared MLflow lifecycle

Each benchmark has its own MLflow experiment. A parent represents one fair
comparison or qualification invocation; each direct child represents one
candidate. Runs are not nested by extraction/RAG category because that would
mix unrelated candidate sets and metric contracts in the same experiment.

```text
EduMind / <Benchmark>
|- parent: <benchmark>-smoke-cpu-<timestamp>
|  `- child per smoke candidate
|- parent: <benchmark>-smoke-cuda-<timestamp>
|  `- child per smoke candidate
|- parent: <benchmark>-preflight-<timestamp>
|  `- child per declared candidate
|- parent: <benchmark>-development-<timestamp>
|  `- child per qualified candidate
|- parent: <benchmark>-validation-<timestamp>
|  `- child per selected finalist
`- parent: <benchmark>-locked-<timestamp>
   `- one selected candidate child
```

Only applicable phases appear. Vector Database omits CUDA smoke and preflight;
Generation and Final RAG currently omit preflight. Video may create separate
frozen-ASR and visual-comparison parents within one phase.

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
quantization. Peak process VRAM must not exceed `3,584 MiB`. Smoke executes both
CPU and CUDA by default, but neither result is selection evidence.
Document and video parser backends follow their separately recorded lifecycle
and device contracts because not every parser runtime exposes the same backend.
Generation and Final RAG are intentionally outside preflight until their
benchmark designs are finalized; their ordinary runtime resource gates still
apply. Vector Database is CPU-only and has neither CUDA smoke nor preflight.

When a stage declares paired candidate comparisons, they are analysis artifacts
rather than new metrics. They are calculated from aligned per-sample results;
bootstrap comparisons resample the same sample IDs for both candidates instead
of comparing unrelated aggregate values. The applicable pairs, stored fields,
and artifact placement are defined in that stage's methodology section.

### Shared MLflow metric convention

The stage sections below list only each metric's base key. A sample-based
aggregate is stored using one consistent convention:

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
was calculated; it never means zero. Statuses, configuration values, checksums,
and one-off measurements such as a cold load or observed resource peak do not
receive these suffixes.

Every automated benchmark value has one of six roles. **Primary** metrics answer
the experiment's central question. **Secondary** metrics explain or qualify a
primary result. **Diagnostic** metrics expose a specific behavior or failure
mode without deciding the winner. **Operational** metrics measure latency,
throughput, and resources. A **validity gate** must pass before a run may be used
as evidence; it is not a trade-off metric. A **descriptor** records workload or
storage needed to interpret other results but has no better direction. Evidence
types and document groups are reporting slices, not additional metric roles.
Blinded rubric scores are labeled separately as human judgments.

## 1. Document extraction

Which complete parser profile best converts educational images, PDFs, and DOCX
files into accurate text and useful structure, including pages, reading order,
tables, and formulas?

OCR is not tested as an isolated product. It is tested inside the complete
document pipeline because OCR text and boxes influence layout, reading order,
page attribution, tables, and formulas.

### Terminology and unit of comparison

An **extraction profile** is one complete executable parser configuration. The
benchmark runner currently calls this value a `candidate`, but the word means a
configuration run, not an individual model. For example:

```text
Docling Standard
+ RapidOCR
+ PDF-aware OCR
+ TableFormer accurate
+ formula enrichment on
```

An extraction profile produces one canonical structured document for every
applicable input. Results use only two organizing concepts.

**Metric groups say what is measured:**

```text
text
pages
layout
tables
formulas
reliability
operational
```

These are the sections defined in [metrics.md](metrics.md). Content F1 is a text
metric, Page Coverage is a page metric, and TEDS is a table metric.

**Document groups say what kind of document was processed:**

```text
PDF
├── digital
├── scanned
├── mixed
└── broken text

image
├── scan
└── phone photo

DOCX
└── native document
```

Every sample belongs to one broad format group (`image`, `pdf`, or `docx`) and
one detailed group where applicable. Compound detailed names use underscores in
MLflow: `image_scan`, `image_phone_photo`, `pdf_digital`, `pdf_scanned`,
`pdf_mixed`, and `pdf_broken_text`. A phone photo therefore contributes to the
`image` aggregate and the more specific `image_phone_photo` aggregate.

An unqualified metric is the aggregate across every eligible sample in that
child run. Because PDF, image, and DOCX have separate parents, its scope is the
source being compared by that parent. Adding a document group gives the same
metric for that group only:

```text
text.content_f1                 -> every document with a text reference
text.image.content_f1           -> images only
text.pdf_digital.content_f1     -> digital PDFs only
text.pdf_scanned.content_f1     -> scanned PDFs only
text.docx.content_f1            -> DOCX only

tables.teds                     -> every reference table
tables.pdf_scanned.teds         -> reference tables in scanned PDFs only
```

Documents still carry annotations such as `has_table`, `has_formula`,
`multi_column`, and `layout_difficulty`. Those labels remain in the manifest and
per-sample artifact for investigation, but they do not create another required
MLflow namespace.

Metrics from different categories are never averaged together. `text.content_f1`
does not combine text, pages, layout, tables, formulas, latency, or memory. It is
only the total Content F1 across documents eligible for Content F1.

### Which metrics apply to which inputs

The following matrix defines applicability. `Yes` means the metric group is
expected for that input when its required reference annotation exists.
`Conditional` means only the relevant annotated subset is scored. `No` means
the metric would not represent the native input and must be omitted rather than
recorded as zero.

| Metric category | Images/photos | Digital PDF | Scanned PDF | Mixed PDF | Broken-text PDF | Native DOCX |
|---|---:|---:|---:|---:|---:|---:|
| Text content and recognition | Yes | Yes | Yes | Yes | Yes | Yes |
| Page metrics | Yes, as one page | Yes | Yes | Yes | Yes | No |
| Reading order | Yes | Yes | Yes | Yes | Yes | Yes |
| Visual layout and bounding boxes | Yes | Yes | Yes | Yes | Yes | No |
| Semantic element types and hierarchy | Conditional | Yes | Yes | Yes | Yes | Yes |
| Tables | Conditional | Conditional | Conditional | Conditional | Conditional | Conditional |
| Formulas | Conditional | Conditional | Conditional | Conditional | Conditional | Conditional |
| Reliability | Yes | Yes | Yes | Yes | Yes | Yes |
| Operational | Yes | Yes | Yes | Yes | Yes | Yes |

#### Text

Content Precision, Content Recall, Content F1, CER, and WER apply to every input
with verified reference text: images, every PDF family, and DOCX. Reading Order
Accuracy also applies to all of those formats when the reference contains enough
ordered elements to form comparable pairs.

#### Pages

Page Coverage, Page Content F1, Page Attribution Accuracy, and Duplicate Page
Rate apply to PDFs and to images treated as one-page documents. They do not
apply to native DOCX. DOCX page boundaries depend on a renderer, fonts, margins,
and page settings; native ingestion deliberately avoids inventing a fixed visual
pagination.

#### Layout

Visual layout detection and Mean Bounding-Box IoU apply to images and PDFs with
layout boxes. Native DOCX does not receive bounding-box metrics. Semantic
Element Type Accuracy and Hierarchy Accuracy can apply to DOCX because headings,
paragraphs, lists, captions, nesting, and parent-child relationships exist in
the authored structure without rendering. Reading order is kept in the text
group even though it uses matched document elements.

#### Tables

Table metrics can apply to images, every PDF family, and DOCX. Detection
precision, recall, and F1 use table-presence annotations, including verified
negative cases needed to expose false detections. Table Content F1, TEDS, and
TEDS-S apply only to reference tables that the parser is expected to recover. A
document with no reference table does not receive a zero table-structure score.

#### Formulas

Formula metrics can apply to images, every PDF family, and DOCX. Detection
metrics use annotated positive and verified-negative cases. Formula Recognition
Similarity and Formula Exact Match apply only to reference formulas. A
formula-free document does not receive a zero recognition score.

#### Difficult layouts

`layout_difficulty` is a per-sample diagnostic label, not another metric
category or required MLflow namespace. If the general layout result needs
investigation, the per-sample artifact can be filtered to documents with
columns, unusual reading order, dense pages, or overlapping elements.

#### Reliability and operations

Reliability metrics apply to every scheduled supported input, including inputs
that fail extraction. Complete-document latency, first-item latency, RAM, VRAM,
temporary disk, and appropriate throughput measurements apply to every processed
format. Per-page latency and pages per minute apply to PDFs and one-page images,
not to native DOCX with no fixed rendering.

In short:

```text
text                          -> images + all PDFs + DOCX
pages                         -> images + all PDFs
visual layout and boxes       -> images + all PDFs
semantic structure            -> annotated images + all PDFs + DOCX
tables                        -> table-evaluation inputs of any supported format
formulas                      -> formula-evaluation inputs of any supported format
reliability                   -> every scheduled supported input
operational                   -> every processed input; page rates exclude native DOCX
```

### Data

Smoke uses two committed images, two PDFs, and two DOCX files. The authoritative
corpus target is:

| Modality | Total | Development | Validation | Locked |
|---|---:|---:|---:|---:|
| Images/pages | 120 | 72 | 24 | 24 |
| PDFs | 60 | 36 | 12 | 12 |
| DOCX | 45 | 27 | 9 | 9 |

The reviewed source pool is OmniDocBench v1.6 for structured pages, OHR-Bench v2
for real multi-page PDFs, PureDocBench v1.0 for matched clean/degraded images,
DocPTBench for photographed documents, and EduMind-specific native DOCX and
held-out samples. The frozen manifests—not the source-pool
names—define the actual cases and record every selected ID, source revision,
license, checksum, document family, source type, and available annotations. An
authoritative run cannot be claimed until those manifests and references are
complete.

OmniDocBench is one annotated source within this combined corpus, not the whole
EduMind experiment. Its selected English pages provide verified text, reading
order, element, table, and formula references. EduMind then evaluates its own
Docling configuration matrix, exact parser revisions, canonical-output
integration, reliability, and local operational cost across OmniDocBench and the
other sources above. The public OmniDocBench leaderboard is candidate-screening
evidence; it is not reused as an EduMind result.

The corpus covers clean and degraded images; digital, scanned, mixed, and
broken-text PDFs; phone photos; low-resolution and skewed pages; multiple
columns; headings, lists, and captions; tables and formulas; and native DOCX
documents. Every sample has verified text. Page, layout, table, and formula
metrics are calculated only when the required annotations exist. Every
conditional aggregate records how many samples were eligible.

### Phase A: Docling Standard configuration screen

The development PDF set receives all 24 configurations:

```text
OCR engine:          RapidOCR, Tesseract, EasyOCR                 (3)
OCR mode:            PDF-aware regions, full page                 (2)
TableFormer mode:    fast, accurate                               (2)
Formula enrichment: off, on                                      (2)
                                                                  ───
Total:               3 × 2 × 2 × 2 = 24 configurations
```

Each factor answers a production question:

| Setting | Question answered by changing it |
|---|---|
| OCR engine | When Docling genuinely needs OCR, which backend gives the best final text, boxes, reading order, table content, speed, and resource use? |
| PDF-aware versus full-page OCR | When should EduMind preserve usable native PDF text, and when should it reconstruct the complete page through OCR? |
| TableFormer fast versus accurate | Does better row, column, header, merged-cell, and span reconstruction justify the additional execution cost? |
| Formula enrichment off versus on | Does CodeFormulaV2 improve mathematical-expression recovery enough to justify its model load, latency, memory, and false detections? |

The applicability rules prevent meaningless duplicate work:

| Source | Configurations executed | Reason |
|---|---|---|
| PDF | All 24 | Every factor can affect digital, scanned, mixed, or broken PDFs. |
| Image | 12 unique engine × table × formula profiles | Images always use full-page OCR, so the two PDF OCR modes would duplicate work. |
| DOCX | Native Docling ingestion once | OCR engine and PDF OCR mode do not apply to native DOCX parsing. |

Docling 2.117.0, English, OCR scale 3.0, table-cell matching enabled,
code enrichment disabled, canonical structured output, and native DOCX
ingestion remain fixed. They define the common evaluation environment rather
than useful strategy questions.

### Phase B: scoring and result slices

Every applicable execution is converted to the same canonical document
representation containing text, pages, ordered elements, types, hierarchy,
bounding boxes, tables, formulas, provenance, warnings, and timing. The
benchmark scores that representation without an EduMind cleanup profile. Raw
outputs remain unchanged. For prose comparison only, the evaluator applies the
same symmetric projection to reference and prediction: Unicode NFC,
case-folding, punctuation-to-space replacement, and whitespace collapse. It
does not dehyphenate words, correct spelling, rewrite numbers, remove headers,
or alter formulas, code, layout trees, or table trees.

Every authoritative reference declares a `reference_capabilities` list drawn
from `text`, `pages`, `reading_order`, `layout_boxes`, `element_types`,
`hierarchy`, `tables`, and `formulas`. Validation and metric eligibility follow
that declaration instead of assuming that every source has every annotation.
Smoke fixtures may infer capabilities from their inline annotations. Table and
formula capability declarations also carry explicit `has_table` and
`has_formula` booleans so verified negatives contribute false detections while
missing annotations remain inapplicable.

OmniDocBench annotations act as ground truth for every applicable metric on its
samples. EduMind calculates the common text, page, layout, detection,
reliability, and operational metrics so their definitions remain identical for
OHR-Bench, PureDocBench, DocPTBench, native DOCX, and EduMind-specific samples.
Only the specialized table-tree and formula scorers—TEDS, TEDS-S, and CDM—come
from the pinned official evaluator; ExpRate@CDM is derived from those CDM
results.

The execution boundary is explicit:

```text
Python 3.12: Docling/Granite/Paddle extraction -> canonical prediction
             -> common EduMind metrics
Python 3.10 Docker: all table HTML pairs -> TEDS and TEDS-S
                    all formula LaTeX pairs -> CDM
```

The table and formula pairs for one extraction profile are scored in one
container call after extraction finishes. Container startup and scoring time are
evaluation overhead, not extraction latency. MLflow provenance records both the
pinned OmniDocBench source revision and the immutable Docker image digest.

The result groups are:

| Group | What it establishes |
|---|---|
| Text | Whether required text was recovered accurately and in the correct order. |
| Pages | Whether content was recovered from, and attributed to, the correct pages without duplication. |
| Layout | Whether elements, semantic types, hierarchy, and locations were preserved. |
| Tables | Whether tables were detected and their content and structure reconstructed. |
| Formulas | Whether formulas were detected and recognized correctly. |
| Reliability | Whether the profile returns complete, non-duplicated, deterministic output without fatal failures. |
| Operations | First-request cost, warm latency, throughput, RAM, VRAM, and temporary disk. |

The exact formulas, eligibility rules, ranges, and directions are defined in
[metrics.md](metrics.md). An inapplicable conditional metric is absent rather
than zero. A real extraction failure remains visible and contributes to the
reliability result.

Each metric is aggregated across all its eligible documents and, separately,
across relevant document groups. This permits conclusions such as "high text
quality across all documents but weak table structure on scanned PDFs" instead
of hiding the weakness inside one mean.

### Phase C: parser architecture comparison

After reviewing the Standard-pipeline screen, the engineer records the selected
Docling profiles. Those complete profiles are compared with:

| Parser profile | Question answered |
|---|---|
| Selected Docling Standard profile | How well does the conventional OCR, layout, table, and formula pipeline perform? |
| Granite Docling 258M | Does Docling's compact full-page VLM improve complete document parsing? |
| PaddleOCR-VL-1.6 | Does an independent visual parser outperform the two Docling architectures? |

The common architecture comparison uses images and PDFs. DOCX is evaluated
through native Docling and reported as format coverage; it is not rasterized to
give visual parsers artificial DOCX support.

### Development, validation, and locked test

```text
smoke:
one small real profile → verify loading, extraction, scoring, artifacts, and MLflow

development:
Docling configuration screen → document-group breakdowns
→ engineer selects one PDF configuration and one image configuration
→ selected Standard configurations + Granite Docling + PaddleOCR-VL
  are compared on the same development split
→ engineer records architecture finalists

validation:
only the engineer-selected architecture finalists
on unseen image/PDF inputs; native Docling on DOCX
→ engineer selects the complete parser profiles without adding candidates

locked:
one PDF validation winner + one image validation winner + native DOCX
on untouched locked-test inputs -> reporting only, with no further selection
```

Within a development, validation, or locked comparison, every candidate profile
receives the same deterministically shuffled eligible samples, one cold
measurement, warmups, and three measured repetitions. Smoke uses one measured
repetition because it is only a wiring check.
All scheduled repetitions run even after an earlier repetition fails, and each
attempt has its own timing/error row. If any measured repetition fails, that
document's quality output is the empty prediction, Candidate Failure Rate is
one, and Structured-output Determinism is zero. Throughput counts pages from
successful attempts but divides by the elapsed time of every measured attempt.
Development determines both the Standard settings and the parser-architecture
finalists. Validation confirms only those finalists; it is not the first local
comparison of Granite Docling or PaddleOCR-VL. After validation, the engineer
records exactly one PDF winner and one image winner. Locked evaluation applies
those frozen routes to the untouched locked-test split; native DOCX remains the
only valid DOCX route. The locked result is reporting-only and cannot reopen
configuration or architecture selection.

### MLflow result structure

Document runs use `EduMind / Document`. CPU/CUDA smoke parents and one
all-candidate preflight parent precede the comparisons below. A document command with
`--source all` creates three independent **parent runs**, because PDF, image,
and DOCX execute different valid configuration sets. Each parent is one fair
comparison; it is not a parser result itself. The standard configuration tree
is:

```text
MLflow experiment: EduMind / Document
├── parent: extraction-document-configuration-pdf-<timestamp>
│   └── up to 24 children: one per qualified PDF extraction profile
├── parent: extraction-document-configuration-image-<timestamp>
│   └── up to 12 children: one per qualified full-page image profile
└── parent: extraction-document-configuration-docx-<timestamp>
    └── 1 child run: native Docling ingestion
```

`--source pdf`, `--source image`, or `--source docx` runs only that parent. After
the configuration screen, development architecture parents compare the selected
Standard profile with Granite Docling and PaddleOCR-VL. Validation parents then
contain only the architecture finalists recorded by the engineer. The DOCX
parent validates native Docling because the two visual parsers do not accept
native DOCX.

```text
MLflow experiment: EduMind / Document
├── parent: extraction-document-architecture-development-pdf-<timestamp>
│   ├── child: <selected PDF Docling Standard profile>
│   ├── child: docling-vlm-granite-258m
│   └── child: paddleocr-vl-1.6
├── parent: extraction-document-architecture-development-image-<timestamp>
│   ├── child: <selected image Docling Standard profile>
│   ├── child: docling-vlm-granite-258m
│   └── child: paddleocr-vl-1.6
└── parent: extraction-document-architecture-development-docx-<timestamp>
    └── child: docling-standard-native
```

The corresponding validation parents use
`extraction-document-architecture-validation-<source>-<timestamp>` and contain
only the recorded finalists for that source. Locked parents use
`extraction-document-architecture-locked-<source>-<timestamp>` and contain one
validation winner for PDF or image, or the fixed native-Docling route for DOCX.

The parent run stores:

- execution profile (`smoke`, `development`, `validation`, or `locked`), stage, dataset
  name and checksum;
- seed, required metric contract, run fingerprint, Git state, hardware, model
  revisions, dependency locks, and any engineer-decision file;
- `plan.json`, `provenance.json`, and the final `summary.json` artifacts;
- completion metrics: whether the invocation is complete and how many profiles
  succeeded or failed; and
- paired comparisons derived from aligned per-sample results in `summary.json`.

Paired comparisons are emitted for document-level scalar metrics. Pooled layout,
table, and formula detection precision/recall/F1 use their candidate-level
document-bootstrap intervals instead; averaging per-document F1 differences
would not reproduce the documented pooled-count calculation.

Each nested child run represents exactly one extraction profile. Its run name is
the complete configuration identifier, for example:

```text
docling-standard|ocr=rapidocr|mode=pdf_aware_layout_regions|table=accurate|formula=on
```

The child run stores:

- the complete resolved runtime profile, even for values shared by every child:
  profile identifier and factors, engine revision and local model path, device,
  language, fixed Docling options, normalization mode, seed, warmups,
  repetitions, and success/failure status;
- prepared-component identities and cache-manifest checksums for the primary
  model and every parser dependency, including Paddle layout components;
- aggregate quality metrics and their `ci_lower` and `ci_upper` values when the
  metric is eligible for an interval;
- `operational.*` latency, throughput, memory, VRAM, and disk metrics when
  available;
- a candidate-result JSON containing the resolved parameters, status,
  aggregates, intervals, operational values, fingerprint, and any error; and
- `samples.parquet` with one row per processed sample, including sample ID,
  latency, individual metrics, document-group and annotation labels, warnings,
  and other diagnostic metadata, plus `timings.parquet` with one row for every
  measured repetition and its success/error state.

The MLflow comparison page is used for compact aggregates. The Parquet artifact
is the detailed evidence: it supports document-group breakdowns, diagnostic
filtering, and paired inspection of the same document across profiles. Raw input
documents are not duplicated into each child run.

The authoritative layout logs **every applicable metric in
[metrics.md](metrics.md)**. The first name is the metric category. With no
document group in the name, the value is the total aggregate across every
eligible document processed by that child run:

```text
text.content_precision
text.content_recall
text.content_f1
text.character_error_rate
text.word_error_rate
text.reading_order_accuracy

pages.page_coverage
pages.page_content_f1
pages.page_attribution_accuracy
pages.duplicate_page_rate

layout.element_precision
layout.element_recall
layout.element_f1
layout.element_type_accuracy
layout.hierarchy_accuracy
layout.mean_bounding_box_iou

tables.detection_precision
tables.detection_recall
tables.detection_f1
tables.content_precision
tables.content_recall
tables.content_f1
tables.teds
tables.teds_s

formulas.detection_precision
formulas.detection_recall
formulas.detection_f1
formulas.recognition_similarity
formulas.exact_match
```

For example, `text.content_f1` is the total average Content F1 across documents
with verified text. `tables.teds` is the total average across reference tables.
These totals remain separate; Content F1 is never combined
with page, layout, table, formula, reliability, or operational metrics.

Inserting a document-group name gives the same metric for that group only:

```text
text.image.content_f1
text.image_scan.content_f1
text.image_phone_photo.content_f1
text.pdf.content_f1
text.pdf_digital.content_f1
text.pdf_scanned.content_f1
text.pdf_mixed.content_f1
text.pdf_broken_text.content_f1
text.docx.content_f1

pages.image.page_coverage
pages.pdf_digital.page_coverage
pages.pdf_scanned.page_coverage

layout.image.element_f1
layout.pdf_scanned.element_f1
layout.docx.hierarchy_accuracy

tables.image.teds
tables.pdf_digital.teds
tables.pdf_scanned.teds
tables.docx.teds

formulas.image.recognition_similarity
formulas.pdf_scanned.recognition_similarity
formulas.docx.recognition_similarity
```

A document group receives only metrics that apply to it. Native DOCX can receive
text, semantic hierarchy, table, formula, reliability, and document-level
operational metrics. It does not receive page or bounding-box metrics without a
fixed renderer.

Annotations such as `has_table`, `has_formula`, and `layout_difficulty` remain
in the manifest and per-sample Parquet artifact for diagnostic filtering. The
standard MLflow aggregates use only the metric category and optional document
group shown above.

Reliability and operational results use:

```text
reliability.empty_output_rate
reliability.duplicate_content_rate
reliability.structured_output_determinism
reliability.candidate_failure_rate

operational.first_item_latency_seconds
operational.p50_warm_latency_per_page_seconds
operational.p95_warm_latency_per_page_seconds
operational.p50_complete_document_latency_seconds
operational.p95_complete_document_latency_seconds
operational.batch_pages_per_minute
operational.peak_process_tree_ram_mb
operational.peak_vram_mb
operational.peak_temporary_disk_mb
```

Operational latency is additionally aggregated by document group, for example
`operational.pdf_scanned.p95_complete_document_latency_seconds`. Peak RAM, VRAM,
and temporary disk describe the complete profile execution and remain top-level
operational metrics rather than being misleadingly attributed to one slice.

Each development, validation, or locked sample-based quality or reliability base key follows the
shared suffix convention. This applies equally to a total such as
`tables.teds` and a document-group result such as
`tables.pdf_scanned.teds`. Its `sample_count` makes clear when, for
example, a table result is based on fewer annotated documents than a text
result.

Confidence intervals are not attached indiscriminately:

- text, page, layout, table, formula, and reliability aggregates receive 95%
  intervals when they are calculated from enough independent samples;
- p50 and p95 latency receive intervals when enough independent document or page
  observations support the estimate;
- smoke intervals, if emitted for debugging, are not authoritative;
- `sample_count`, run status, completion status, revisions, checksums, and other
  fixed values do not receive intervals; and
- one first-item measurement, one throughput batch, and one observed peak RAM,
  VRAM, or temporary-disk value do not receive intervals. Repeated independent
  measurements may support an interval, but the repetitions and aggregation
  unit must be recorded.

This rule avoids presenting statistical precision that the measurements do not
contain.

MLflow records evidence but does not choose a winner. After reviewing complete
parent and child runs, the engineer records finalists or a final extraction
policy in a separate decision file. The benchmark never modifies production
configuration automatically.

The approved document-parser profile is frozen before video and
downstream extraction are evaluated.

## 2. Audio extraction

Which English speech-to-text profile produces the most accurate educational
transcript with timestamps suitable for navigation and citations?

### Terminology and unit of comparison

An **ASR profile** is one complete executable transcription configuration. It
includes the model revision, decoder settings, device, numeric precision,
audio preprocessing, timestamp method, and any required aligner. The benchmark
runner calls it a `candidate`, but the result belongs to the complete profile,
not only to the model weights.

Each speech clip produces:

- one ordered transcript;
- timestamped segments in the common benchmark representation;
- warnings and timing information; and
- the counts required to reproduce Corpus WER and CER.

Audio already defines chronological order, so the benchmark evaluates the final
ordered transcript with WER. It does not split recognition into Content F1 and
Reading Order Accuracy as document extraction does for two-dimensional pages.

The independent quality sample is one audio clip. Three latency repetitions of
the same clip improve timing measurement but do not become three independent
quality samples.

### ASR profiles

| Model/profile | Why it is included |
|---|---|
| Whisper `small.en` | Established English control with timestamp output. |
| Canary 180M | Compact timestamp-capable challenger. |
| Parakeet TDT 0.6B v2 | Mid-size profile with word, segment, and character timestamps. |
| MOSS Transcribe-Diarize | Larger timestamp-capable transcription challenger. Diarization is not scored because speaker identification is not currently an EduMind requirement. |

### Data

Smoke uses two committed audio clips. The authoritative corpus contains 90
English clips split 54 development, 18 validation, and 18 locked. It includes:

- verified transcripts;
- complete audio duration;
- segment or word timestamps;
- clean and noisy speech;
- accents and multiple speakers;
- technical and educational vocabulary.

The [dataset guide](datasets.md) defines the LibriSpeech, M³AV, EdAcc, and AMI
source pools and the recommended allocation. The exact recordings are
fixed by the manifests, which record source, revision, selected clip interval,
license, checksum, duration, and speaker/document family. Public leaderboard
scores are screening context; they do not replace this frozen corpus.

A small fixed reliability set contains verified silence, music without lyrics,
background noise, and other nonspeech audio. These controls are separate from the
90 speech clips and are used only to measure false transcription on audio that has
no spoken reference. The reliability manifest labels its development,
validation, and locked-test controls so each phase uses only its own subset.

Every speech sample records at least:

```text
sample ID
source, license, revision, and checksum
split and document/speaker family
duration and audio-condition labels
verified transcript
verified timestamped reference segments
```

Each sample stores a `conditions` list. It contains exactly one acoustic label,
`clean` or `noisy`, and may also contain `accented` and/or `multi_speaker`.
Every authoritative split covers all four labels. They remain in the per-sample
artifact for diagnosis and do not create extra required MLflow metric namespaces
or a larger metric contract.

### Common input and output rules

Every candidate receives the same decoded audio waveform: mono, 16 kHz, with no
candidate-specific denoising, volume repair, prompting, or vocabulary hints.
Model-native feature extraction and the documented deterministic decoder remain
part of the ASR profile and are recorded. Candidate output receives only the
fixed evaluator normalization used by WER and CER; the evaluator does not repair
misspellings, remove repetitions, or rewrite transcripts.

All timestamp outputs are converted to ordered benchmark segments containing
text, start time, and end time. For each reference segment, the evaluator
considers contiguous predicted spans whose normalized token Content F1 is at
least `0.5`. Dynamic programming chooses ordered, non-overlapping one-to-one
matches by maximum total similarity, then match count, then earliest span.
Predicted timestamp units cannot be reused by another reference segment. This
supports word timestamps and broader segment timestamps without truncating
unequal arrays. Boundary MAE uses the matched span's minimum start and maximum
end; Alignment Coverage records how much of the timed reference aligned.

An empty transcript with no timestamp segments is a valid low-quality result:
all reference words and characters are deletions, timestamp coverage is zero,
Boundary MAE is null, and Empty Transcript Rate increments. Non-empty text with
no timestamps remains a fatal candidate error, as does an empty transcript with
non-empty lexical timestamp segments.

### Per-candidate execution

One child run executes one ASR profile in a fresh operating-system process. A
CPU process has CUDA hidden before any model runtime is imported; a CUDA process
must provide working NVML VRAM measurement. No profile may change device or use
CPU/GPU offloading silently.

The process performs:

```text
load the exact pinned model
→ record cold model-load time
→ run two warmups
→ transcribe every deterministically shuffled speech clip
→ run three measured warm repetitions per speech clip
→ process the corresponding nonspeech reliability controls
→ aggregate quality, timestamp, reliability, and operational results
→ unload the model and release resources
```

The quality result for a clip comes from one designated measured output.
Repeated executions preserve raw timing measurements but are not averaged into
additional quality samples. Their exact transcript agreement is summarized
separately by the diagnostic Repeat Transcript Agreement Rate.

Authoritative comparisons use the frozen CUDA hardware profile defined above:
batch size `1`, the stage's supported 16-bit dtype, one whole model on the GPU,
and no fallback, offload, device splitting, or quantization. Device, dtype,
decoder, timestamp path, and runtime versions are recorded. Smoke and debugging
may explicitly request CPU or CUDA but cannot support candidate selection. If
one selected profile cannot complete under the authoritative profile, that
parent comparison is incomplete; the engineer fixes the candidate plan and
reruns it instead of comparing partial results.

Behavior-changing settings are part of each child artifact: batch size one;
Whisper word timestamps and deterministic generation; Canary beam size one,
punctuation, capitalization, and timestamps; Parakeet greedy-batch decoding and
timestamp level; and MOSS `max_new_tokens=2048` with `do_sample=false`. Runtime
versions, model paths, revisions, cache-manifest checksums, dtype, and device are
recorded with them.

### Development, validation, and locked test

```text
smoke:
all runnable ASR paths on tiny committed speech and nonspeech fixtures
→ verify loading, transcription, timestamps, scoring, artifacts, and cleanup

development:
all hardware-qualified members of the four-profile ASR roster on 54 speech clips
and development reliability controls
→ engineer reviews MLflow and records finalists

validation:
engineer-selected finalists on 18 unseen speech clips and validation controls
→ engineer records exactly one selected ASR profile

locked test:
the selected profile once on 18 locked speech clips and locked controls
→ final unbiased ASR report; no further tuning in this benchmark version
```

Smoke validates wiring only. Development is where all candidates that passed
the exact matching preflight are compared; exclusions remain recorded in the
parent provenance.
Validation checks whether the chosen finalists retain their behavior on unseen
recordings. The locked split is used only after the engineer has selected one
profile. MLflow records evidence throughout but never advances a candidate or
changes application configuration.

### Metrics and why they are used

| Category | Metrics | Why they are needed |
|---|---|---|
| Recognition | **Corpus WER** (primary), Corpus CER | WER measures the complete ordered word transcript; CER exposes character-level spelling, name, and number errors. |
| WER diagnostics | Word Substitution Rate, Word Deletion Rate, Word Insertion Rate | Shows whether WER comes mainly from confused, omitted, or unsupported words. These explain WER but do not replace it. |
| Timestamps | **Timestamp Boundary MAE**, **Timestamp Alignment Coverage** | MAE measures the accuracy of aligned start/end boundaries; coverage prevents a candidate from looking accurate after aligning only easy segments. |
| Reliability | Empty Transcript Rate, Nonspeech False-Transcription Rate, Repeat Transcript Agreement Rate | Measures complete empty output on speech, invented lexical output on verified nonspeech controls, and transcript stability across repeated runs. |
| Operational | **Complete-Pipeline Real-Time Factor**, p50/p95 warm clip latency, cold model-load time, peak process-tree RAM, peak VRAM | Measures the complete transcription and alignment cost of the frozen CUDA profile. |

Content F1 and Reading Order Accuracy are not ASR metrics in this benchmark.
Audio already supplies chronological order, so Corpus WER evaluates the required
ordered transcript. Technical-Term Accuracy is also excluded because EduMind is
not restricted to a stable subject vocabulary. Diarization is not scored unless
speaker identification becomes a product requirement.

The exact calculations, examples, ranges, directions, and confidence-interval
rules are defined in [metrics.md](metrics.md). Corpus WER, CER, and the three
WER components pool edit counts across speech clips before division; they are
not averages of independently calculated clip error rates. Timestamp Boundary
MAE and Alignment Coverage are interpreted together. Reliability controls are
not included in WER because their references contain no speech.

### MLflow result structure

Audio uses its own MLflow experiment. Smoke creates separate CPU and CUDA
parents, and preflight creates one qualification parent before development:

```text
MLflow experiment: EduMind / ASR
├── parent: asr-smoke-cpu-<timestamp>
│   └── one child per smoke-tested ASR profile on CPU
├── parent: asr-smoke-cuda-<timestamp>
│   └── one child per smoke-tested ASR profile on CUDA
├── parent: audio-preflight-<timestamp>
│   └── one qualification child per declared ASR profile
├── parent: asr-development-<timestamp>
│   └── one child per hardware-qualified ASR profile
├── parent: asr-validation-<timestamp>
│   └── one child per engineer-selected finalist
└── parent: asr-locked-<timestamp>
    └── one child for the selected ASR profile
```

The parent is the comparison run. It stores the phase, profile, dataset and
reliability-manifest checksums, candidate order, seed, Git state, hardware,
dependency locks, model revisions, runtime plan, and any engineer-decision file.
Its artifacts are `plan.json`, `provenance.json`, the frozen manifests, and
`summary.json`. Its only direct metrics describe completion: whether the entire
comparison completed and how many candidates succeeded or failed.

Each child is one ASR profile. Its parameters contain the complete resolved
runtime profile even when some values repeat the parent plan: candidate and
submodel revisions and paths, device, dtype, language, decoder, timestamp
method, seed, FFmpeg version, canonical audio format, duration limit, warmups,
repetitions, data split, and manifest checksums. This makes a child interpretable
when viewed or exported alone. It has no child runs for individual clips,
repetitions, or metrics.

ASR child metrics use descriptive flat names because the run already has
`stage=audio` and no metric names collide inside it:

```text
word_error_rate
character_error_rate
word_substitution_rate
word_deletion_rate
word_insertion_rate

timestamp_boundary_mae_seconds
timestamp_alignment_coverage

empty_transcript_rate
nonspeech_false_transcription_rate
repeat_transcript_agreement_rate

real_time_factor
p50_warm_clip_latency_seconds
p95_warm_clip_latency_seconds
cold_model_load_seconds
peak_process_tree_ram_mb
peak_vram_mb
```

The recognition, timestamp, reliability, and operational labels remain useful
documentation categories, but they are not repeated as MLflow prefixes.
Applicable development, validation, and locked uncertainty bounds use the shared MLflow suffix
convention defined at the beginning of this document.

Corpus WER/CER and their components, timestamp metrics, reliability rates,
Repeat Transcript Agreement Rate, RTF, and sufficiently supported warm latency
estimates receive clip-bootstrap intervals. One cold-load observation and
observed peak RAM/VRAM do not receive fabricated intervals. Every bootstrap
draw contributes to every metric that is defined for that draw. A draw with no
aligned timestamp segment still
contributes zero Alignment Coverage and contributes normally to recognition,
reliability, and latency intervals; only its undefined Boundary MAE is omitted.
If the complete candidate has no valid timestamp alignment,
`timestamp_boundary_mae_seconds` is stored as null and the run remains
successful; `timestamp_alignment_coverage=0` makes the failure visible. No
confidence interval is emitted for the undefined MAE. Because MLflow's scalar
metric store does not accept null, the scalar key is absent there while
`candidate.json` and `summary.json` preserve the field as null. The interval
artifact records the number of contributing resamples.

Each successful child stores three artifacts:

| Artifact | Contents and purpose |
|---|---|
| `samples.parquet` | One row per speech or nonspeech sample with sample ID, condition labels, duration, word/character edit counts, reference lengths, timestamp alignment counts and error totals, reliability and repeat-agreement flags, warnings, and the designated quality-pass latency. It makes every aggregate traceable. |
| `timings.parquet` | One row per speech clip and measured repetition with latency, duration, RTF, and device. It preserves the observations used for p50, p95, and operational analysis. |
| `candidate.json` | Resolved runtime parameters, candidate status, fingerprint, aggregate metrics, confidence intervals, operational values, and artifact references. |

Fields that do not apply to a row are absent or null, not fabricated as zero.
Raw audio and candidate predictions are not uploaded to MLflow. The frozen
speech manifest is uploaded and contains the verified reference transcripts,
source identifiers, and checksums needed to reproduce scoring.

Every successful development, validation, or locked child must contain all 16 aggregate
metric fields. Timestamp Boundary MAE is the sole nullable field, under the rule
above. A CPU profile may report zero VRAM only when execution confirms that no
GPU process was used; unavailable instrumentation is not converted to zero.
CUDA children record `vram_measurement_method`. NVML process-tree bytes are
used directly when the driver exposes them. On Windows WDDM, where NVML can
identify the benchmark PID but returns no per-process byte count, the monitor
records `nvml-device-delta-wddm` and measures the peak increase from the
pre-run device-memory baseline while that PID is present. If neither method
captures a positive allocation, the CUDA child fails rather than reporting a
fabricated zero.
If a candidate crashes, lacks required timestamp output, or cannot produce the
required artifacts or aggregates, its child remains visible as failed and the
parent is incomplete. The engineer repairs the problem and reruns the complete
comparison rather than selecting from partial evidence.

The engineer reviews the completed child runs and per-sample artifacts. No
weighted overall score is calculated. Finalist and winner choices are written
to explicit engineer-decision files containing the source parent run and
selected child profiles. The selected ASR is then frozen for video extraction.

## 3. Video extraction

With the document parser and ASR fixed, which keyframe policy recovers useful
on-screen text without processing too many duplicate frames?

### Candidates

The experiment compares three frame-selection strategies, but it does not test
only one arbitrary setting for each strategy. Development produces nine
configurations:

| Strategy | Development configurations | Question answered |
|---|---|---|
| Fixed interval | One frame every 5, 10, or 20 seconds | How much visual coverage is gained by sampling more frequently, and what does that coverage cost? |
| Scene change | FFmpeg scene threshold 0.30, 0.40, or 0.50 | How sensitive should transition detection be before extra frames become mostly redundant? |
| Hybrid | The selected scene threshold plus a maximum gap of 5, 10, or 20 seconds | How frequently must the fallback sample gradual or static scenes that never produce a strong transition? |

Every configuration includes the first frame. This protects titles, opening
slides, and initial screen state even when no early scene transition occurs.

These are nine configurations of three strategies, not nine unrelated
strategies. The hybrid configurations use the scene threshold selected from the
scene-change comparison. Testing every scene threshold with every maximum gap
would produce 15 configurations and answer an additional interaction question
that is not required in the first benchmark.

### Data

Smoke uses two committed videos. The authoritative set contains 30 educational
videos split 18 development, 6 validation, and 6 locked. Every video has a
verified transcript, duration, visible text, and visual timestamps. The set
includes slides, screen recordings, presenter video, gradual text changes, and
repeated scenes. Exact sources, revisions, licenses, clip intervals, and checksums
are frozen in the manifests. The [dataset guide](datasets.md) defines the
SlideSpeech, AVLectures, and EduMind-owned allocation and explains why public
subtitles and OCR must be manually corrected rather than accepted as ground
truth.

### Execution

The dedicated runner supports smoke, development, validation, and locked
profiles. Its normal `protocol.yaml` freezes the candidate ranges, frame-zero
and VFR rules, ASR windows and overlap, deterministic normalized suffix/prefix
stitching, visible-text unitization, occurrence matching, dataset counts, and
execution profiles. Manifests remain ordinary data inputs whose checksums are
recorded in run provenance and in the frozen-ASR artifact; there is no separate
manifest-bound video lock.

The frozen-ASR phase composes the video and ASR protocols. Each visual phase
also composes the document protocol so the selected image parser receives the
same resolved parser options and candidate-factor validation used by the
document benchmark. The visual worker verifies both protocol payloads before
initializing the parser.

The authoritative hybrid threshold and its development source run ID begin as
`null`. Fixed and scene development comparisons can run in that state. After
reviewing the scene comparison, an engineer writes the selected declared
threshold and source run ID into `protocol.yaml` and increments the protocol
version. Hybrid execution rejects missing or out-of-range values. Because this
changes the protocol checksum, the development frozen-ASR artifact must be
regenerated before the hybrid or combined nine-configuration run. The runner
never writes the threshold or application configuration itself.

```text
video frozen-ASR phase
└─ FFmpeg mono 16 kHz windows → selected ASR once → FrozenASRArtifact

video visual phase
└─ candidate FFmpeg keyframes → frozen selected document parser
   → visible-text and timing metrics
   → reference the FrozenASRArtifact checksum; never invoke ASR
```

Only the keyframe configuration changes. The selected ASR is executed once per
video phase, and its timestamped per-video outputs and measurements are frozen
as an upstream artifact. Every keyframe child references that artifact by run ID
and checksum instead of retranscribing the videos. Reopening parser or ASR
selection here would make it unclear whether a difference came from frame
selection, visual parsing, or speech recognition.

Video audio may be longer than the ASR benchmark's 30-second single-clip limit.
The selected ASR therefore receives deterministic windows no longer than 30
seconds. Window-local timestamps are shifted back onto the video timeline and
overlapping text is stitched once. The exact overlap and stitching rule are
frozen before the video comparison. This qualifies the serving policy of the
already selected ASR; it does not reopen the four-profile ASR comparison.

Development proceeds in this order:

1. Run the fixed-interval configurations at 5, 10, and 20 seconds.
2. Run the scene-change configurations at thresholds 0.30, 0.40, and 0.50.
3. Review visual quality and processing cost, then record one scene threshold
   and its source run ID in `protocol.yaml`; increment the protocol version and
   regenerate the frozen-ASR artifact.
4. Combine that threshold with maximum gaps of 5, 10, and 20 seconds and run
   the three hybrid configurations.
5. Compare the resulting nine development configurations. Record finalists;
   do not calculate an automatic overall score.
6. Run only the engineer-selected finalists on validation. Run one selected
   configuration once on the locked test.

The numerical settings are initial development search points, not universal
constants. Five seconds is the high-coverage/high-cost interval, 20 seconds is
the low-cost/low-coverage interval, and 10 seconds is the midpoint. FFmpeg's
[scene-filter guidance](https://ffmpeg.org/pipermail/ffmpeg-cvslog/2012-June/051105.html)
defines the score on a 0-to-1 scale and identifies roughly 0.3 to 0.5 as a
practical range; 0.30, 0.40, and 0.50 sample that range without a large grid.
The hybrid gaps reuse the fixed-interval values so the fallback cost can be
compared directly with the fixed strategy. The selected values are valid only
for the recorded educational-video corpus.

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | Visual Content F1; Mean Visual First-Detection Delay with Timed Visual Occurrence Coverage | Measures whether useful on-screen text was recovered, whether it was captured while visible, and how quickly it was first captured. Delay and coverage must be interpreted together. |
| Secondary | Visual Content Precision/Recall | Explains whether a low F1 came from unsupported extracted text or missed visible text. |
| Diagnostic | Duplicate Visual Text Rate | Shows whether repeatedly selected unchanged frames duplicate the same content. |
| Diagnostic | Frozen-ASR Transcript WER, recorded once for the shared ASR output | Confirms the audio input to every policy; it is not used to compare keyframe policies because it is constant. |
| Operational | Visual Real-Time Factor, p50/p95 warm visual latency, cold visual-pipeline load time, peak visual process-tree RAM, peak visual VRAM | Measures the keyframe and visual-parser cost that differs between configurations. |
| Workload descriptor | Mean selected frames per video | Records how much visual input each policy sends to the parser without treating fewer frames as inherently better. |

Spoken and visible tokens remain separate. A video's transcript usually
contains far more words than its frames, so one combined recall value would be
dominated by audio and could hide a poor keyframe policy. The isolated video
comparison therefore scores visible content and its timing; the usefulness of
the combined audio-and-visual result is evaluated later in the end-to-end
retrieval experiment.

### MLflow result structure

Video uses `EduMind / Video`. Smoke creates separate CPU and CUDA parents, and
preflight qualifies the frozen-ASR stack and every visual policy before
development. The shared ASR input is recorded first. The nine development
configurations are then created by three ordered comparisons
so the hybrid run can consume the engineer-selected scene threshold:

```text
MLflow experiment: EduMind / Video
├── parents: video-frozen-asr-smoke-cpu/cuda-<timestamp>
├── parents: video-visual-smoke-cpu/cuda-<timestamp>
├── parent: video-preflight-<timestamp>
│   └── frozen-ASR qualification plus one child per visual policy
├── parent: video-frozen-asr-development-<timestamp>
│   └── child: <selected-asr-profile-across-all-development-videos>
├── parent: video-visual-development-fixed-<timestamp>
│   ├── child: video-fixed-5s
│   ├── child: video-fixed-10s
│   └── child: video-fixed-20s
├── parent: video-visual-development-scene-<timestamp>
│   ├── child: video-scene-0.30
│   ├── child: video-scene-0.40
│   └── child: video-scene-0.50
├── parent: video-visual-development-hybrid-<timestamp>
│   ├── child: video-hybrid-<selected-threshold>-5s
│   ├── child: video-hybrid-<selected-threshold>-10s
│   └── child: video-hybrid-<selected-threshold>-20s
├── parent: video-visual-validation-<timestamp>
│   └── one child per engineer-selected finalist
└── parent: video-visual-locked-<timestamp>
    └── one child for the selected configuration
```

Each child is one complete keyframe configuration evaluated on every video in
that phase. It is not split into child runs for individual videos or metrics.
Each child records its complete resolved runtime profile: strategy and numerical
settings, parser revision and settings, device, seed, FFmpeg version and command,
warmups and repetitions, dataset checksum, plus the frozen ASR run ID and
artifact checksum. Values shared with the parent are repeated deliberately so
the child remains interpretable when exported alone. Validation and
locked phases follow the same pattern: one phase-specific frozen-ASR input run,
then the visual-policy comparison run.
The child logs these aggregate metrics:

```text
visual_content_precision
visual_content_recall
visual_content_f1
mean_visual_first_detection_delay_seconds
timed_visual_occurrence_coverage
duplicate_visual_text_rate

visual_real_time_factor
p50_warm_visual_latency_seconds
p95_warm_visual_latency_seconds
cold_visual_pipeline_load_seconds
peak_visual_process_tree_ram_mb
peak_visual_vram_mb
mean_selected_frames_per_video
```

The frozen-ASR child logs `word_error_rate` and the ASR component's
actual per-video latency, RTF, RAM, and VRAM measurements. These values describe
shared upstream audio work and are not copied into every visual child. Each
visual child stores per-video
quality and timing rows in `samples.parquet`, per-repetition timings in
`timings.parquet`, and the complete aggregate result in `candidate.json`.
In development, validation, and locked runs, Visual Content Precision/Recall/F1, Timed
Visual Occurrence Coverage, Duplicate Visual Text Rate, Visual Real-Time Factor,
and Mean Selected Frames per Video receive video-bootstrap intervals. Mean
Visual First-Detection Delay receives an interval over videos with covered timed
occurrences. Warm p50
and p95 latency receive intervals when enough independent videos support the
percentiles. The frozen-ASR child's Transcript WER receives an interval from
the same videos. Cold load and observed RAM/VRAM peaks do not receive fabricated
intervals.

Frozen-ASR operational measurements and visual-child measurements remain
separate because adding aggregate percentiles or memory peaks would not recreate
a real pipeline measurement. After the keyframe policy is selected, one
integrated confirmation run measures actual end-to-end latency, RTF, RAM, and
VRAM for the complete video path.

For every audio or video decoding step, the run records the FFmpeg version and
the exact argument vector used. This belongs in the run plan/provenance
artifacts, not only in documentation, because installed codecs and command
options can change the decoded input.

The selected keyframe policy joins the selected parser and ASR as the provisional
video-extraction profile.

## 4. Chunking and embedding

Which complete chunker/embedding pair retrieves verified educational evidence
best?

### Terminology and unit of comparison

Chunking and embedding are evaluated together. The candidate is the complete
`chunker|embedding` pair: a chunker cannot be scored until its chunks are
represented and ranked, and an embedding cannot be judged independently of the
passages it embeds. This phase isolates that pair by using exact cosine search;
it does not test BM25, rank fusion, a reranker, a vector database, a generator,
or final answer quality.

Each pair produces:

- one chunk manifest with stable IDs and source offsets;
- one ranked retrieval list for every answerable question;
- evidence matches and per-question quality values; and
- timing, resource, warning, and validity records.

Quality is evaluated on answerable questions with verified evidence. The
answerability label remains in the manifest for downstream generation and final
RAG evaluation, but this phase does not ask a dense retriever to abstain. An
unanswerable question has no gold evidence and therefore cannot receive a
fabricated retrieval-quality score.

One answerable question is one scoring sample. Because questions from the same
paper are related, the source document—not the individual question—is the
independent resampling unit. Repeated query executions improve timing
measurement but do not create additional quality samples.

### Candidate pairs

The development comparison crosses every declared chunking strategy with every
declared embedding model. The result belongs to the complete pair.

#### Chunking strategies

| Strategy | What it tests |
|---|---|
| Recursive character 1000/200 | Cheap textual separators without tokenizer dependence. |
| Token 256/32 | Short, focused passages with modest overlap. |
| Token 384/64 | Middle fixed-token trade-off. |
| Token 512/64 | More local context per chunk. |
| Sentence 8/2 | Linguistic boundaries instead of fixed token windows. |
| Semantic | Topic-change boundaries based on adjacent-sentence embedding similarity. |
| Section-aware 512/64 | Preserves authored Markdown sections before splitting oversized sections. |
| Structure-aware 512/64 | Preserves headings, tables, formulas, and table-row boundaries where possible. |

#### Embedding models

| Model | Role in the comparison |
|---|---|
| GTE ModernBERT base | 149M lightweight production control with an 8,192-token input limit. |
| Snowflake Arctic Embed M v2 | Strong small-model retrieval candidate with a long input limit. |
| F2LLM v2 0.6B | Alternative 0.6B retrieval architecture. |
| Octen Embedding 0.6B | Strong 0.6B retrieval challenger. |
| Qwen3 Embedding 0.6B | Strong directly comparable 0.6B candidate using documented last-token pooling. |
| Nemotron 3 Embed 1B | Larger nearby-size retrieval candidate. |

Full public evidence and exact revisions are in
[model-selection.md](model-selection.md).

### Data

The RAG corpus uses pinned QASPER papers plus EduMind's additional verified
structured-evidence set. The counts below describe QASPER papers only:

| Split | QASPER papers | Used for |
|---|---:|---|
| Development | 100 | Development candidate comparison |
| Validation | 40 | Validation finalist comparison |
| Locked test | 40 | One final complete system only |

Each question stores answerability, accepted answers, evidence type, and exact
half-open evidence offsets. The structured supplement contains table, formula,
and mixed-evidence questions because QASPER is primarily text.

### Common input and output rules

Every pair receives the same frozen documents, answerable questions, and
evidence annotations. Candidate-specific text repair, query rewriting, reranking,
and approximate vector search are not allowed in this phase.

Each answerable question contains one or more independently useful **gold
evidence units**. A unit has a stable ID, one mutually exclusive evidence type,
and one or more exact half-open source intervals. Units must be atomic enough to
fit inside a valid candidate chunk. A table unit includes any row/column headers
needed to interpret its selected cells, and a formula unit includes the complete
expression. Evidence that genuinely requires different modalities or separated
support is represented by multiple units and the question is labelled `mixed`.

For the rank-aware and unit-recall metrics, chunk `j` covers evidence unit `i`
only when that single chunk contains every required source interval for the
unit. Partial overlap does not turn a fragment into complete evidence. This
rule makes boundary fragmentation visible and avoids candidate-specific fuzzy
matching. The metric roles, counting behavior, and edge cases are summarized in
[metrics.md](metrics.md).

Evidence-token Precision uses the frozen evaluation tokenizer
`tiktoken:cl100k_base` for every candidate. It never uses each embedding model's
tokenizer for scoring, because changing the counting unit between candidates
would make their precision values incomparable. Model-specific tokenizers are
used separately to prepare inference inputs and validate their exact lengths.
The fixed tokenizer is a protocol control, not another benchmark candidate. A
future multilingual protocol may evaluate tokenizer sensitivity separately
before freezing a replacement.

Every one of the 48 planned pairs must fit the embedding model's native input
contract, including query/document prefixes and required special tokens. The
preflight records native-token counts for every generated chunk and answerable
query. It never silently truncates, changes a strategy's size, or splits chunks
again for one model. An oversized input is reported as
`input_length_exceeded`; the child fails and the comparison is incomplete.

### Per-candidate execution

One child run executes one planned pair in a fresh operating-system process.
The requested device, dtype, model and tokenizer revisions, query/document
prefixes, pooling, normalization, seed, warmups, and repetitions are fixed and
recorded. Silent device fallback or unrecorded truncation invalidates the child.
Authoritative development and validation comparisons use the target RTX 3050 through
CUDA with `float16` and embedding batch size `1`. Peak process VRAM must remain
at or below 3,584 MiB, leaving a 512 MiB safety reserve on the 4,096 MiB device.
The same settings apply to every pair; a candidate cannot receive a smaller
batch or a different precision to avoid an out-of-memory result.

An eligible pair performs:

```text
split documents into chunks with exact source offsets
-> validate every chunk with the candidate model tokenizer
-> embed every valid chunk
-> embed each answerable question
-> rank with exact NumPy cosine search and deterministic tie-breaking
-> retain the top 20 as the auditable retrieval artifact
-> score the first three and first five against verified evidence units and source spans
```

Exact NumPy search removes vector-database approximation from this experiment.
For semantic chunking, the tested embedding also creates the boundaries; that
result intentionally represents the complete semantic-chunker/embedding pair.
The search scores the complete candidate corpus, retains the top 20 only for
near-miss diagnosis, and scores both `@3` and `@5`. These cutoffs match the two
downstream Final-RAG context counts; this phase adds no token budget or context
curve.

Corpus-build timing starts after model and tokenizer loading and includes
chunking, document embedding, and searchable-matrix/metadata assembly. Query
latency includes query tokenization, query embedding, exact cosine scoring, and
deterministic top-20 selection. Each query's median measured repetition is its
warm-latency observation. Model downloads, data downloads, and environment setup
are excluded.

A successful child must account for every expected document and answerable
query, produce the expected number and dimension of finite nonzero vectors, use
zero truncated inputs, and reproduce the same ordered top-five IDs for repeated
identical queries. An input-limit violation stops after preflight with its
complete validation report. It is a failed child, like a crash, out-of-memory
condition, or malformed output, and makes the comparison incomplete.

### Development, validation, and locked test

```text
smoke:
declared chunking and embedding paths on tiny committed fixtures
-> verify compatibility checks, embedding, ranking, scoring, and artifacts

development:
8 chunkers × 6 embeddings = 48 planned pair records on 100 papers
-> run and account for all 48 pairs
-> engineer selects up to three complete finalist pairs

validation:
selected finalists on 40 unseen papers
-> engineer records exactly one selected chunker/embedding pair

locked test:
the selected pair runs only inside the one frozen complete-system evaluation
on 40 locked papers -> no further component tuning
```

Smoke validates wiring only. Development compares the declared matrix, and
validation checks the finalists on unseen papers. A failed pair makes its parent
comparison incomplete. The locked split is reserved for the final complete
system and is not another chunking/embedding selection round.

### Metrics and why they are used

| Category | Metrics | Why they are needed |
|---|---|---|
| Retrieval quality | **nDCG@3/@5**, **Evidence-unit Recall@3/@5**, **Evidence-token Precision@3/@5**; alpha-nDCG@3/@5 diagnostic | Measures conventional relevance order, evidence completeness, and context concentration; alpha-nDCG diagnoses repeated evidence on eligible multi-evidence questions. |
| Evidence slices | Retrieval metrics repeated for text, table, formula, and mixed questions | Reveals a pair that performs well only on the majority evidence type. |
| Operational | Corpus-build time and source-token throughput, p50/p95 warm query latency, peak process-tree RAM, peak VRAM | Measures the observed preparation, query, and hardware cost of the complete pair. |
| Workload and storage | Corpus counts, source and indexed-token counts, chunk count and lengths, embedding dimension and dtype, matrix bytes | Explains how much work and storage each pair creates without treating those values as quality. |

The three bold metric families are primary and are reviewed separately at both
cutoffs. Alpha-nDCG is diagnostic, `alpha=0.5` is fixed, and no weighted overall
score is created. Alpha-nDCG is omitted, rather than scored as zero, when a
question has fewer than two distinct evidence units.
First-hit metrics and chunk-level precision/recall are omitted because they add
little coverage information or use denominators changed by the chunker. Exact
definitions, examples, directions, and confidence-interval rules are in
[metrics.md](metrics.md#chunking-and-embedding).

The four retrieval metric families are reported overall and for mutually
exclusive `text`, `table`, `formula`, and `mixed` slices. A mixed question
requires at least two evidence types. Each slice reports its contributing
question and document counts; alpha-nDCG also reports its smaller eligible
question and document counts. Unanswerable questions remain a workload count
and do not enter retrieval-quality aggregates.

### MLflow result structure

Chunking/embedding uses `EduMind / Chunking–Embedding` and one parent per fair
comparison. Independent CPU/CUDA smoke parents precede preflight; the final
reporting-only parent contains exactly one validation winner:

```text
MLflow experiment: EduMind / Chunking–Embedding
├── parent: rag-chunking-embedding-smoke-cpu-<timestamp>
│   └── one child per smoke-tested pair on CPU
├── parent: rag-chunking-embedding-smoke-cuda-<timestamp>
│   └── one child per smoke-tested pair on CUDA
├── parent: chunking-embedding-preflight-<timestamp>
│   └── one qualification child per declared pair
├── parent: rag-chunking-embedding-development-<timestamp>
│   └── up to 48 children: one per hardware-qualified planned pair
├── parent: rag-chunking-embedding-validation-<timestamp>
│   └── up to three child runs: one per engineer-selected finalist pair
└── parent: rag-chunking-embedding-locked-<timestamp>
    └── one child for the selected pair
```

Each parent stores the phase, dataset and checksum, candidate plan, seed,
repetitions, metric contract, model lock, Git and
dependency provenance, hardware, and any engineer-decision file. Its direct
metrics contain completion counts only: planned, successful, and failed pairs.
`plan.json`, `provenance.json`, `metric_contract.json`, `leaderboard.parquet`,
paired comparisons, and `summary.json` are parent artifacts. The chosen pair's
locked result remains in this component experiment and cannot reopen selection.

Each child is one planned pair. Its run name is the complete pair identifier,
`<chunker_id>|<embedding_id>`. Parameters contain the resolved chunker and
embedding contracts, model and tokenizer revisions and checksums, device, dtype,
prefixes, pooling, normalization, evaluation tokenizer, evidence rule,
`quality_cutoffs=[3,5]`, `artifact_top_k=20`, `alpha=0.5`, seed, warmups,
repetitions, split, and manifest checksum.
The child has no child runs for documents, questions, repetitions, or metrics.

A valid child logs:

```text
quality.overall.ndcg_at_3
quality.overall.ndcg_at_5
quality.overall.evidence_unit_recall_at_3
quality.overall.evidence_unit_recall_at_5
quality.overall.evidence_token_precision_at_3
quality.overall.evidence_token_precision_at_5
quality.overall.alpha_ndcg_at_3       # eligible multi-evidence questions only
quality.overall.alpha_ndcg_at_5       # eligible multi-evidence questions only

quality.text.<metric>
quality.table.<metric>
quality.formula.<metric>
quality.mixed.<metric>

operational.corpus_build_seconds
operational.corpus_build_source_tokens_per_second
operational.query_latency_ms_p50
operational.query_latency_ms_p95
operational.peak_process_tree_ram_mb
operational.peak_vram_mb

workload.source_tokens
workload.indexed_token_occurrences
workload.document_count
workload.answerable_query_count
workload.unanswerable_query_count
workload.chunk_count
workload.chunk_tokens_mean
workload.chunk_tokens_p95
workload.embedding_dimension
workload.embedding_matrix_bytes
```

Primary, diagnostic, operational, and descriptor labels are documentation
roles; they are not repeated as MLflow prefixes. The prefixes above describe
metric families and evidence slices. Eligible uncertainty bounds use the shared
`.ci_lower` and `.ci_upper` suffixes.
Validity counters such as expected/processed documents and queries, truncated
inputs, failed cases, nonfinite/zero-norm vectors, dimension mismatches, and
determinism mismatches are logged under `validity.*`. General retrieval
eligibility and the smaller alpha-nDCG-eligible question/document counts are
logged per evidence scope. The final decision is also stored as a
`benchmark.valid` tag and a `validation.status` tag whose value is `passed` or
`failed`. MLflow scalar values do not replace the
detailed `validation_report.json` artifact. Stored dtype is a parameter in the
resolved embedding contract rather than a numeric metric.

Each successful child stores:

| Artifact | Contents and purpose |
|---|---|
| `chunk_manifest.parquet` | One row per chunk with source document, exact offsets, strategy metadata, and token counts. |
| `query_metrics.parquet` | One row per answerable question with @3/@5 quality values and evidence slice; ineligible alpha-nDCG fields are absent. |
| `retrievals.parquet` | Ordered top-20 chunk IDs and scores for near-miss diagnosis; the first three and first five are scored. |
| `evidence_matches.parquet` | Trace from each question and evidence unit to the chunks that recovered it. |
| `timings.parquet` | One row per query and measured repetition with warm latency and success state. |
| `resources.parquet` | Timestamped process-tree RAM and VRAM samples. |
| `candidate.json` | Resolved contracts, status, fingerprint, aggregates, confidence intervals, operational values, and artifact references. |
| `validation_report.json` | Input-length and validity checks, counts, limits, checksums, and any errors. |

Raw documents, model weights, and a full embedding matrix are not duplicated
into every child; frozen inputs, offsets, revisions, and checksums make the rows
reproducible. Any failed child logs its reason and available validation evidence,
remains visible with MLflow status `FAILED`, and makes the parent comparison
incomplete.

Aggregation and uncertainty follow the
[chunking and embedding confidence-interval contract](metrics.md#chunking-and-embedding-confidence-intervals).
Evidence slices use the classifications above and report their contributing
question and document counts.

The engineer approves up to three complete chunker/embedding pairs. No separate
embedding winner or chunker winner is required. Advancement jointly reviews
nDCG, Evidence-unit Recall, and Evidence-token Precision at `@3` and `@5` as the
three primary quality dimensions. Diagnostic alpha-nDCG, uncertainty, evidence
slices, and operational feasibility explain or constrain that judgment; they
are not steps in an automatic lexicographic ranking. The decision and rationale
are stored in a versioned engineer-decision file that references the
parent/child MLflow run IDs and all governing checksums. The benchmark never
promotes a candidate automatically.

## 5. Retrieval and reranking

Which complete first-stage retriever and reranker gives the best evidence
ordering for the selected chunker/embedding pair, and is any quality improvement
worth its latency and resource cost?

### Candidate matrix

The selected chunker/embedding pair, corpus, questions, and evidence annotations
are frozen before this phase. A candidate is one complete
`<retriever>|<reranker>` stack. Every first-stage retriever is crossed with every
reranker option, producing exactly `3 × 5 = 15` peers in the development run.

| First-stage retriever | What it tests |
|---|---|
| `dense` | Exact cosine ranking from the selected embedding. |
| `bm25` | Lexical ranking for exact terminology, names, identifiers, and codes. |
| `rrf` | Reciprocal-rank fusion of the Dense and BM25 top-20 lists without mixing their raw scores. |

| Reranker | What it tests |
|---|---|
| `none` | The first-stage order without learned reranking. |
| `gte-modernbert` | Weaker long-context GTE ModernBERT cross-encoder control. |
| `ettin-150m` | Compact learned reranker. |
| `ettin-400m` | Mid-size learned reranker. |
| `ettin-1b` | Larger learned reranker. |

The no-reranker rows are full candidates, not administrative baselines. Dense
and BM25 remain independent when selected directly. RRF reads their ranked
top-20 source lists, combines them using the frozen fusion settings, and returns
one deterministic top-20 list.

### Frozen candidate pools and execution

The experiment uses the same frozen QASPER-plus-structured manifest, canonical
chunks, questions, evidence-unit IDs, and source intervals as the preceding
chunking/embedding decision.

The retrieval/reranking smoke fixture contains 30 frozen canonical chunks. This
leaves ten chunks outside each top-20 pool, so smoke execution can exercise pool
selection, exclusion, fusion, and permutation checks. The count is a smoke
fixture control; authoritative chunking strategies are not forced to produce
exactly 30 chunks because their natural chunk counts are benchmark outputs.

Quality comparisons use one checksummed top-20 pool per first-stage retriever.
The three no-reranker candidates own those artifacts:

| Pool owner | Shared by |
|---|---|
| `dense|none` | All five Dense candidates. |
| `bm25|none` | All five BM25 candidates. |
| `rrf|none` | All five RRF candidates. |

A reranker receives exactly the pool owned by its matching no-reranker child. It
may only permute those 20 chunk IDs: it cannot add, remove, duplicate, or retrieve
chunks. The reranker child records the owner's MLflow run ID and the pool
SHA-256. A mismatch or non-permutation invalidates the child and makes the parent
comparison incomplete.

```text
load the frozen selected chunker/embedding pair and corpus
→ build Dense and BM25 indexes
→ create the Dense, BM25, and RRF top-20 pools
→ checksum each pool under its no-reranker owner
→ score each no-reranker order at @3 and @5
→ rerank each matching frozen pool with each of the four learned rerankers
→ score every resulting order at @3 and @5
```

The top-20 pool supports reranking and diagnosis; it is not another quality
cutoff. No token-budget packing policy is applied in this phase. The first three
and first five canonical chunk texts are scored exactly as ranked.

Operational measurements do not reuse cached retrieval timings. Every reranked
candidate executes its first-stage method and reranker together so full-stack
latency and memory describe the deployable path. The frozen pool is reused only
to guarantee a fair quality comparison. The no-reranker child measures the same
live first stage without a learned reranker.

The versioned `experiments/benchmarks/rag/retrieval_reranking/protocol.yaml` file is the
single source of truth for benchmark hyperparameters. Its strict schema rejects
missing or unknown fields. Algorithm definitions and deterministic tie-breaking
remain tested code invariants rather than configurable alternatives. The runner
checks the resolved protocol against the plan, records its checksum in every
child, logs the source file as an input, and writes `retrieval_protocol.json`
under the parent so a run can be reproduced without reconstructing settings
from code defaults.

The frozen controls include BM25
tokenization, variant, `k1`, `b`, and tie breaking; Dense similarity,
normalization, query formatting, and tie breaking; RRF source depth, fusion
constant, union/truncation behavior, and tie breaking; and each reranker's model
ID, revision, cache checksum, input template, native tokenizer, maximum input
length, batch size, score interpretation, device, dtype, and tie breaking.
Truncation is forbidden.

The target-hardware profile uses CUDA `float16`. The protocol fixes both
embedding and reranker inference to batch size `1`; learned rerankers therefore
score the 20 query-passage pairs sequentially. Qualification includes
both components under the planned lifecycle and requires NVML-measured peak
process VRAM at or below 3,584 MiB, preserving a 512 MiB reserve on the 4 GiB
GPU. Per-candidate quantization, offload, fallback, or batch reduction is not an
allowed way to pass the gate.

The retrieval controls are fixed as follows. They are established, untuned
baselines rather than claims that one parameter set is optimal for every corpus:

- BM25 uses `BM25Okapi` with `k1=1.5`, `b=0.75`, and `epsilon=0.25`. The
  moderate `k1` gives repeated query terms diminishing returns, while `b=0.75`
  substantially normalizes variable chunk lengths without treating every
  length difference as irrelevant. In the selected implementation, `epsilon`
  floors negative IDF values for terms present in more than half the corpus.
  These are the explicit
  [`rank_bm25` defaults](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py),
  so freezing them avoids adding a development-set tuning advantage to the
  lexical baseline.
- Dense retrieval uses exact cosine similarity over L2-normalized document and
  query vectors. Normalization prevents vector magnitude from affecting rank,
  and exact search isolates semantic retrieval from approximate-index error.
  Approximate nearest-neighbour behavior is evaluated separately in the vector
  database phase.
- RRF reads the Dense and BM25 top-20 lists, gives each source weight `1.0`,
  uses fusion constant `60`, joins by stable chunk ID, and keeps the highest
  scoring 20 unique chunks. Equal weights avoid an unvalidated preference for
  lexical or semantic retrieval, while rank fusion avoids normalizing their
  incomparable raw scores. The constant `60` is the frozen value used by the
  [original RRF study](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).
  A missing source rank contributes zero. Ties resolve by best source rank, then
  combined source rank with missing ranks treated as `21`, then chunk ID.

Dense and BM25 own their respective searchable states. RRF builds no third
index: it references both checksummed indexes and creates only the fused ranked
pool. Its required storage is the unique sum of those dependencies, while its
incremental fusion-index storage is zero.

The ideal ranking used to normalize nDCG is derived from the complete frozen
chunk corpus for each question, never from a candidate's retrieved pool. The
alpha-nDCG ideal is constructed from the same complete corpus using its frozen
evidence-unit coverage. Consequently, every executed candidate is compared
against the same candidate-independent ideal for a given question.

### Execution profiles and selection

```text
smoke:
minimal deterministic fixtures → wiring and artifact checks only

development:
all hardware-qualified members of the 15-candidate roster on the development manifest
→ engineer records up to three complete-stack finalists

validation:
the finalists on the unseen validation manifest
→ engineer approves up to three retrieval stacks for complete-system testing

locked test:
exactly one validation winner on the locked component split
→ no further retrieval or reranker tuning
```

A complete stack decision includes the frozen chunker, embedding, first-stage
retriever, and reranker. When a selected reranker needs its no-reranker owner's
pool, the runner prepares and verifies that pool as an input; the owner is not
added as another validation or locked candidate. No weighted score or automatic
winner rule is used.

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | **nDCG@3/@5**, **Evidence-unit Recall@3/@5**, **Evidence-token Precision@3/@5** | Separates ranking quality, evidence completeness, and concentration of retrieved text. |
| Diagnostic | alpha-nDCG@3/@5, candidate-pool Evidence-unit Recall@20 | Diagnoses repeated evidence and first-stage pool limits without replacing the primary decision. |
| Validity gate | Ranking Agreement | Requires repeated inference to return the same complete ordering before the run may be used as evidence. |
| Operational | Full-stack warm p50/p95 latency, first-stage warm p50/p95 latency, reranker warm p50/p95 latency when applicable, cold initialization, peak process-tree RAM, peak VRAM, index-build time | Separates retriever cost, incremental reranker cost, startup, memory, and one-time index preparation. |
| Workload descriptor | Corpus/query/chunk/pool counts, retrieved tokens at @3/@5, reranker input tokens when applicable | Explains the amount of work behind quality and operational results. |
| Storage descriptor | Dense/BM25 index bytes, RRF required and incremental index bytes, reranker snapshot bytes | Describes the local searchable state and model storage each candidate requires. |

The three bold quality families are primary and are reviewed separately at both
cutoffs. Alpha-nDCG uses `alpha=0.5` and is reported only for questions with at
least two distinct evidence units. Candidate-pool Recall@20 is recorded once on
each no-reranker pool owner because its four learned-reranker children receive
the same pool. Reranker-only latency and input-token metrics are omitted, not set to zero,
for the no-reranker option.

Eligible quality metrics are also reported separately for text, table, formula,
and mixed questions. These evidence types are reporting slices, not additional
metric roles.

The evidence-unit definitions, `tiktoken:cl100k_base` evaluation tokenizer,
answerable-question eligibility, document-macro aggregation, confidence
intervals, and text/table/formula/mixed slices are the same as in the
chunking/embedding phase. Exact definitions and comparison rules are in
[metrics.md](metrics.md#retrieval-and-reranking).

Hit Rate, MRR, MAP, binary chunk Precision/Recall, and a context-budget metric
are not reported. They either repeat questions already answered by the primary
metrics, use a chunk-dependent denominator, or test a packing policy that this
phase does not have. Failures and truncation are validity conditions rather than
quality metrics.

### MLflow result structure

Retrieval/reranking uses `EduMind / Retrieval–Reranking` and one parent for each
fair comparison. Every planned
candidate is a direct child; the development parent therefore has up to 15
candidate children, with any definitive hardware exclusions retained in
qualification provenance. Pools and paired comparisons do not create
intermediate or nested runs.
Separate CPU/CUDA smoke parents and the preflight parent precede development.
Validation contains exactly the engineer-selected finalists; a required owner
pool is an input rather than an extra child. Locked contains exactly one
validation winner and is reporting-only.

```text
MLflow experiment: EduMind / Retrieval–Reranking
├── parent: rag-retrieval-reranking-smoke-cpu-<timestamp>
├── parent: rag-retrieval-reranking-smoke-cuda-<timestamp>
├── parent: retrieval-reranking-preflight-<timestamp>
│   └── one qualification child per declared stack
├── parent: rag-retrieval-reranking-development-<timestamp>
│   ├── up to 5 Dense children
│   ├── up to 5 BM25 children
│   └── up to 5 RRF children
├── parent: rag-retrieval-reranking-validation-<timestamp>
│   └── exactly the engineer-selected finalists
└── parent: rag-retrieval-reranking-locked-<timestamp>
    └── one selected stack
```

The parent parameters identify the phase, profile, manifest and checksum,
selected chunker/embedding decision and fingerprint, the qualified subset of
the declared 15-candidate plan,
metric contract, protocol version and checksum, seed, warmups, repetitions,
model lock, Git/dependency
provenance, and hardware. Its direct metrics are completion counts only. Parent
artifacts are `plan.json`, `provenance.json`, `metric_contract.json`,
`retrieval_protocol.json`, `pool_index.json`, `leaderboard.parquet`,
`retriever_comparisons.parquet`,
`retriever_comparisons.csv`, `reranker_comparisons.parquet`,
`reranker_comparisons.csv`, and `summary.json`. Validation additionally stores
`finalist_comparisons.parquet` and `finalist_comparisons.csv` when explicit
cross-stack finalist comparisons are requested.

Each child run name is its complete candidate ID. Its parameters contain the
resolved retrieval and reranking contracts, all model/configuration/checksum
fields, `requested_pool_size=20`, `quality_cutoffs=[3,5]`, evidence and tokenizer
rules, device, dtype, seed, warmups, repetitions, split, and manifest checksum.
A reranker child also stores `pool_owner_run_id` and `pool_checksum`.

A valid child logs the applicable values under these families:

```text
quality.overall.ndcg_at_3
quality.overall.ndcg_at_5
quality.overall.evidence_unit_recall_at_3
quality.overall.evidence_unit_recall_at_5
quality.overall.evidence_token_precision_at_3
quality.overall.evidence_token_precision_at_5
quality.overall.alpha_ndcg_at_3
quality.overall.alpha_ndcg_at_5
quality.<text|table|formula|mixed>.<metric>

diagnostic.pool_evidence_unit_recall_at_20   # pool owners only
validity.ranking_agreement

operational.full_stack_latency_ms_p50
operational.full_stack_latency_ms_p95
operational.first_stage_latency_ms_p50
operational.first_stage_latency_ms_p95
operational.reranker_latency_ms_p50          # reranked children only
operational.reranker_latency_ms_p95          # reranked children only
operational.cold_initialization_seconds
operational.peak_process_tree_ram_mb
operational.peak_vram_mb
operational.index_build_seconds              # pool owners only

storage.index_bytes                          # Dense and BM25 owners
storage.required_index_bytes                 # RRF owner: unique Dense + BM25 bytes
storage.incremental_index_bytes              # RRF owner: zero
storage.reranker_snapshot_bytes              # reranked children only

workload.corpus_document_count
workload.answerable_query_count
workload.chunk_count
workload.requested_pool_size
workload.actual_pool_size_mean
workload.actual_pool_size_min
workload.short_pool_query_count
workload.retrieved_tokens_at_3_mean
workload.retrieved_tokens_at_3_p95
workload.retrieved_tokens_at_5_mean
workload.retrieved_tokens_at_5_p95
workload.reranker_input_tokens_per_query_mean # reranked children only
workload.reranker_input_tokens_per_query_p95  # reranked children only
```

Eligibility counts and validity counters are logged under `validity.*`.
Required checks include expected/processed/failed queries, zero truncation,
finite scores, pool-checksum agreement, exact pool permutation, and deterministic
rank agreement. Every child stores `query_metrics.parquet`, `rankings.parquet`,
`evidence_matches.parquet`, `timings.parquet`, `resources.parquet`,
`candidate.json`, and `validation_report.json`. Pool-owner children additionally
store `candidate_pool.parquet` and `index_build.json`. Raw documents and model
weights are never copied into child artifacts.

### Paired comparisons and advancement

Paired comparisons are analysis rows under the parent, not MLflow runs. They
are separated by the intervention being studied:

- `reranker_comparisons.parquet` and `reranker_comparisons.csv` contain the 12
  reranker-effect comparisons, each learned reranker against the matching
  `<retriever>|none` candidate over the same checksummed pool;
- `retriever_comparisons.parquet` and `retriever_comparisons.csv` contain Dense
  versus BM25, Dense versus RRF, and BM25 versus RRF using the three
  no-reranker candidates; and
- validation uses `finalist_comparisons.parquet` and
  `finalist_comparisons.csv` only for explicitly declared cross-stack finalist
  comparisons.

Parquet is authoritative and typed; CSV is the human-readable mirror. Each pair
is generated from one in-memory typed table. The writer reads both files back
with their declared schemas, sorts them by stable row keys, and verifies column
and cell equality with explicit null and floating-point handling. Matching row
counts alone is insufficient. Each file has its own SHA-256 and the pair records
one shared logical-table SHA-256.

The benchmark does not generate all 105 possible candidate pairs. Each stored
row represents one comparison, metric, evidence slice, and cutoff. It identifies
the applicable retriever or complete stacks, both run IDs and pool checksums,
metric direction, both values, raw `candidate - baseline` difference, whether
the result favors the candidate, eligible question/document counts, and
paired-bootstrap confidence bounds when supported.

The reranker-comparison schema is:

```text
comparison_id, retriever,
baseline_candidate, candidate, baseline_run_id, candidate_run_id,
shared_pool_checksum, metric_name, evidence_slice, cutoff, direction,
baseline_value, candidate_value, candidate_minus_baseline, favors_candidate,
ci_lower, ci_upper, confidence_level, ci_status, eligible_questions,
eligible_documents, bootstrap_resamples, seed
```

The retriever-comparison schema replaces `retriever` and
`shared_pool_checksum` with `baseline_retriever`, `candidate_retriever`,
`baseline_pool_checksum`, and `candidate_pool_checksum`. Finalist rows identify
both complete stacks and both pool checksums.

`candidate_minus_baseline` always preserves the raw arithmetic difference.
`favors_candidate` is populated only when the metric has an unconditional
direction: positive quality differences and negative operational differences
favor the candidate. For descriptive workload or storage differences,
`direction=descriptive` and `favors_candidate` is null.

Quality rows require `cutoff` and an `evidence_slice` such as `overall`, `text`,
`table`, `formula`, or `mixed`; non-quality rows set both fields to null.
Confidence bounds, `confidence_level`, bootstrap resamples, and seed are
populated only when a paired interval is available. Otherwise they are null and
`ci_status` states why, such as `smoke`, `insufficient_documents`, or
`not_supported`.

Reranker-effect rows compare all primary quality metrics, eligible alpha-nDCG,
full-stack latency, cold initialization, RAM, VRAM, and retrieved-token workload.
They do not pretend that shared pool Recall@20, shared index build, first-stage
latency, or reranker-only fields are reranker improvements. Retriever-effect rows
may compare quality, pool Recall@20, full-stack and first-stage latency, cold
initialization, RAM, VRAM, retrieved tokens, index-build time, and index bytes.
Validity counters are gates, not winner metrics, and receive no comparison rows.

Quality differences use 10,000 paired bootstrap resamples of aligned source
documents with seed 42. One-off cold-load and peak-resource observations receive
point differences but no invented confidence interval. The engineer selects up
to three complete finalists after jointly reviewing primary quality, uncertainty,
evidence slices, diagnostics, and operational feasibility, then approves up to
three validation-qualified stacks for complete-system testing. Final RAG, not
this component benchmark, selects the one deployed retrieval stack. The
versioned decision file references the parent and child run IDs and all
governing checksums.

## 6. Vector database servers

Which networked vector server preserves nearest-neighbour and filter correctness
while providing the most useful latency, concurrency, ingestion, memory, and
storage trade-off?

### Servers

| Server | Why it is included |
|---|---|
| Chroma server | Current provisional production baseline. |
| Qdrant server | Purpose-built vector server with payload filtering. |
| Weaviate | Independent vector-server architecture with structured filters. |
| PostgreSQL + pgvector | Transactional relational alternative with JSON metadata and HNSW. |

All servers receive identical precomputed vectors and metadata. They never create
embeddings internally.

### Data and configurations

| Execution profile | Workload |
|---|---|
| Smoke | 1,000 vectors at dimension 384; 50 queries; concurrency 1 |
| Development | 100,000 vectors at dimensions 384 and 1,024; 500 queries; concurrency 1/8/32 |
| Validation | Selected real embeddings plus 1,000,000 clustered vectors; up to 1,000 queries; concurrency 1/8/32/64 |

Synthetic vectors contain clusters and 5% near-duplicates. Metadata creates
filters matching approximately 50%, 10%, 1%, and in validation 0.1% of records.

Development and validation test supported HNSW combinations of:

```text
m:                     16 or 32
construction breadth:  100 or 200
search breadth:         64 or 128
```

This makes sure one server is not compared with an unnecessarily weak default.
Unsupported settings are recorded rather than silently replaced.
Every supported configuration remains visible in MLflow; the runner does not
automatically choose the database winner.

### Execution

The experiment has three steps.

#### A. Conformance

The real server is checked for health, cosine behavior, wrong-dimension
rejection, compound filters, empty filters, duplicate-ID replacement, complete
document replacement, deletion, real ANN-index use, persistence after restart,
and index availability after restart. These are validity checks. A server that
fails one is reported as non-conformant rather than assigned a misleading
performance rank.

#### B. Dense ANN performance

```text
NumPy computes exact top neighbours
→ server returns approximate neighbours
→ compare returned IDs with the exact IDs
→ repeat unfiltered and filtered queries
→ repeat at each configured concurrency
```

#### C. Real retrieval

After the engineer chooses database finalists and one retrieval stack, every
finalist stores the same real chunks and vectors. The complete retrieval strategy
is rerun so database ANN behavior is connected to actual RAG quality. The
engineer then approves one server profile for Final RAG.

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Validity gate | Health, cosine behavior, dimension rejection, Filter Correctness, Empty-Filter Correctness, replacement, deletion, persistence, restart, and ANN-index verification | Determines whether results are trustworthy; these are not quality scores. |
| Primary | ANN Recall@3/@5/@10, Filtered ANN Recall@3/@5/@10 | Measures preservation of exact neighbours at application-relevant depths, with and without metadata filters. |
| Secondary | ANN and Filtered ANN Recall@1; complete-RAG nDCG, Evidence-unit Recall, and Evidence-token Precision at @3/@5 | Shows rank-one behavior and whether ANN results preserve real evidence retrieval. |
| Diagnostic | Unfiltered/filtered p50 latency, first query after restart | Explains typical and cold-query behavior. |
| Operational | Unfiltered/filtered p95/p99, throughput and error rate at each concurrency, build time and vectors/second, incremental upsert/delete throughput, restart readiness, peak server RAM, persistent storage | Measures tail latency, load handling, ingestion, restart, memory, and disk cost. |

The database report remains separate evidence. It shows which server should be
used by the final benchmark, but the benchmark never changes the current Chroma
production default automatically.

## 7. Generation

Which local Hugging Face generator produces the best grounded, cited answer when
every model receives exactly the same verified evidence?

### Models

| Model/profile | Why it is included |
|---|---|
| Falcon-H1-Tiny-R-90M, reasoning | Independent weak control that establishes a low-resource quality floor. |
| Qwen3 0.6B, reasoning enabled | Small established reasoning candidate. |
| Qwen3.5 0.8B, reasoning enabled | Newer intermediate candidate. |
| MiniCPM5 1B, reasoning enabled | Strongest candidate in the common public hardware-feasible screen. |

No trustworthy public benchmark compares all four under EduMind's grounded QA,
citation, refusal, faithfulness, and local-latency protocol. Public evidence made
the shortlist; this experiment makes them directly comparable.

### Data

Smoke uses committed wiring fixtures. Development uses 24 development questions
balanced across answerability, answer type, and evidence type as an initial
screen. Validation evaluates only engineer-selected generator finalists on the
complete frozen validation question set; it is not limited to 24 questions.
The selected generator is exercised on locked-test questions only as part of the
one frozen complete-system run.

- Answerable questions receive their verified numbered evidence blocks.
- Unanswerable questions receive text from their document that does not answer
  the question.
- Retrieval is not run in this stage.

Using frozen evidence prevents a good generator from being penalized by a poor
retriever.

### Execution

Every generator uses its exact pinned local snapshot, official chat template,
reasoning mode, temperature 0, seed 42, an 8,192-token context limit, and at
most 256 generated tokens. Authoritative development and validation runs use the same
CUDA device, `float16`, and batch size `1`: one complete question-and-evidence
prompt is generated at a time, and repetitions run sequentially. No model
receives hidden quantization, CPU/GPU offload, automatic device splitting, or a
candidate-specific batch size.

### Execution profiles and selection

```text
smoke:
all runnable generators on tiny committed fixtures
-> verify loading, output parsing, scoring, artifacts, and cleanup

development:
all four generator profiles on the 24-question development screen
-> engineer records up to three generator finalists

validation:
only the recorded finalists on the complete unseen validation question set
-> engineer records the generator profiles approved for complete-system testing

locked test:
the one selected generator runs only inside the frozen Final RAG system
-> no further generator tuning
```

### Per-candidate workflow

```text
unload previous generator
→ cold-load candidate
→ run two warmups
→ give every candidate the same question and numbered evidence
→ generate three times
→ separate hidden reasoning from visible answer
→ validate citations and compare with accepted answers
```

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | Citation Precision/Recall/F1 on answerable questions, Answerability Balanced Accuracy, Unsupported Answer Rate, Malformed Output Rate | Measures evidence use, answer/refusal decisions, substantive answers to unanswerable questions, and protocol failures without penalizing correct refusals for having no citations. |
| Secondary | Token F1 | Provides partial answer-correctness evidence before human review. |
| Diagnostic | Exact Match, ROUGE-L, Refusal Precision/Recall/F1, HHEM on substantive non-refusal answers, Repeat Output Agreement | Explains lexical similarity, refusal errors, automated support estimates, and stability. HHEM never replaces human Faithfulness. |
| Operational | Cold load, Time to First Token, generation time, total p50/p95 latency, tokens/second, peak RAM, peak VRAM | Separates startup, responsiveness, decoding speed, total latency, and memory. |
| Workload descriptor | Prompt, visible-answer, reasoning, and total generated token counts | Records how much native-tokenizer input and output produced the observed quality and latency. |

The engineer approves up to three generator profiles after inspecting automatic
quality, citation/refusal behavior, latency, and resources. An automatically
supported citation must identify a supplied evidence block that completely
covers at least one required gold evidence unit. Human review of broader claim
faithfulness happens only after retrieval and generation are combined.

## 8. Final RAG and human review

Which complete retrieval-and-generation system gives the best evidence-backed
answers when every component runs together?

### Execution profiles and systems tested

Development crosses the approved component finalists on development data:

```text
1 approved vector-server profile
× up to 3 retrieval stacks
× up to 3 generators
× top_k {3, 5}
= at most 18 complete systems
```

The engineer records exactly three successful complete-system finalists. Validation
runs only those three systems on unseen validation data and supplies their
answers for blinded review. After review, the engineer records exactly one
complete system. The locked profile runs only that frozen system on the locked-test data.

### Data and execution

The development Final RAG profile uses the development manifest, validation uses
the validation manifest, and locked uses the locked-test manifest. For every question, the
complete path runs:

```text
chunk document
→ embed chunks and question
→ store/query dense vectors through the approved server
→ apply BM25/RRF/reranking selected for the stack
→ select the ranked top 3 or top 5 chunks
→ number the evidence blocks
→ generate the answer and citations
```

For `top_k=3`, retrieval quality is reported at 3. For `top_k=5`, it is reported
at 3 and 5.

Final RAG reports nDCG, Evidence-unit Recall, and Evidence-token Precision at the
available cutoff, plus eligible alpha-nDCG as a diagnostic. It also reports the
primary and secondary generation metrics from Experiment 7 and retrieval,
generation, server-call, and complete end-to-end p50/p95 latency. RAM and VRAM
remain operational measurements.

### Human review

Validation evaluates the three successful complete systems selected after development.
The exporter selects 20 common validation questions and creates:

```text
20 questions × 3 anonymous systems = 60 anonymous answer items
```

One reviewer scores all 60 answer items while system identity remains hidden. The
reviewer sees the question, answer, accepted answer, and evidence. Each answer
receives:

| Human metric | Scale | What it measures |
|---|---:|---|
| Faithfulness | 0–2 | Whether every material claim is supported by the supplied evidence. |
| Answer Correctness | 0–2 | Whether the answer is correct for the question. |
| Completeness | 0–2 | Whether all essential parts are covered. |
| Citation Accuracy | 0–2 | Whether citations are attached to the right claims and evidence blocks. |
| Answerability Correctness | 0–1 | Whether the system correctly answered or refused. |

This is single-reviewer evidence, so the report does not claim inter-reviewer
reliability. After ratings are imported and validated, system identities are
revealed and the engineer selects exactly one complete system. A future
multi-reviewer study must define overlap, agreement, and adjudication separately.

## 9. Extraction-to-RAG confirmation

How much does real extraction reduce the quality of the selected RAG system?

### Execution

Two versions of the same documents and questions are compared:

```text
verified reference text → frozen selected RAG
selected extracted text → the same frozen selected RAG
```

Question IDs, document IDs, questions, model profiles, prompt, and retrieval
strategy remain identical. The selected parser, ASR, and vector server are part
of the extracted-text path. Each text version keeps its own evidence offsets
because extraction can change length and layout.

This comparison uses a separate frozen confirmation manifest derived without
locked-test questions. It may describe deployment risk, but it cannot reopen
component selection after the system has been frozen.

### Metrics

The experiment reports the paired extracted-minus-reference difference for:

- **Retrieval:** nDCG, Evidence-unit Recall, and Evidence-token Precision at the
  system's actual top-K, plus eligible alpha-nDCG as a diagnostic.
- **Generation:** Token F1, answerable-only Citation Precision/Recall/F1,
  Answerability Balanced Accuracy, Unsupported Answer Rate, and Malformed Output
  Rate.
- **Diagnostics:** refusal metrics, HHEM on substantive answers, and per-question
  error inspection.
- **Operational:** server-call, retrieval, generation, and complete p50/p95
  latency.

This experiment does not select the extractor again. It quantifies the downstream
cost of extraction after component selection.

## 10. Locked test

After Final RAG review and extraction confirmation are complete, the one frozen
system runs exactly once on the locked-test manifest. The selected parser, ASR,
chunker, embedding, retrieval method, vector server, generator, prompt, and
context settings cannot change between confirmation and this run.

The locked result is the final unbiased estimate. It is not used for more tuning.
If the system changes after the result is inspected, a new benchmark and locked
dataset version are required.
