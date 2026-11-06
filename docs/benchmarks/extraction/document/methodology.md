# Document extraction methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Purpose and scope](#purpose-and-scope)
- [Candidates and configurations](#candidates-and-configurations)
- [Data](#data)
- [Metrics and their purpose](#metrics-and-their-purpose)
- [Evaluation setup](#evaluation-setup)
- [Benchmark workflow and selection](#benchmark-workflow-and-selection)
- [Execution and measurements](#execution-and-measurements)
- [Results and MLflow structure](#results-and-mlflow-structure)

Which complete parser profile best converts educational images, PDFs, and DOCX
files into accurate text and useful structure, including pages, reading order,
tables, and formulas?

OCR is not tested as an isolated product. It is tested inside the complete
document pipeline because OCR text and boxes influence layout, reading order,
page attribution, tables, and formulas.

## Purpose and scope

An **extraction profile** is one complete executable parser configuration,
called a `candidate` by the runner. Results describe that complete configuration,
not an individual model. For example:

```text
Docling Standard
+ RapidOCR
+ PDF-aware OCR
+ TableFormer accurate
+ formula enrichment on
```

One source document is the scoring unit. Pages and reference objects are
observations within that document, not independent benchmark samples.
Native DOCX follows its own route; visual parsers are compared on images and PDFs.

## Candidates and configurations

### Docling Standard configuration screen

The declared PDF screen contains 24 configurations. Development runs every
configuration qualified by preflight on the same PDF set:

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

The source format determines which configurations apply:

| Source | Configurations executed | Reason |
|---|---|---|
| PDF | Up to 24 qualified configurations | Every factor can affect digital, scanned, mixed, or broken PDFs. |
| Image | Up to 12 qualified engine × table × formula profiles | Images always use full-page OCR, so the two PDF OCR modes would duplicate work. |
| DOCX | One qualified native Docling profile | OCR engine and PDF OCR mode do not apply to native DOCX parsing; one profile still receives all scheduled attempts. |

### Parser architectures

The architecture comparison uses the selected Standard profiles and the other
qualified architectures below; preflight exclusions remain visible:

| Parser profile | Question answered |
|---|---|
| Selected Docling Standard profile | How well does the conventional OCR, layout, table, and formula pipeline perform? |
| Granite Docling 258M | Does Docling's compact full-page VLM improve complete document parsing? |
| PaddleOCR-VL-1.6 | Does an independent visual parser outperform the two Docling architectures? |

The common architecture comparison uses images and PDFs. DOCX is evaluated
through native Docling and reported as format coverage; it is not rasterized to
give visual parsers artificial DOCX support.

## Data

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
held-out samples. The frozen manifests define the actual cases selected from
these pools. They record every selected ID, source revision, license, checksum,
document family, source type, and available annotations. Authoritative evaluation
requires complete manifests and references.

The committed smoke fixtures demonstrate the implemented workflow, not a
completed authoritative corpus or a selected parser. The protocol's
`minimum_ci_sources` and `minimum_latency_ci_sources` are currently null pending
review of independent-source support. Freeze them through the
[pending-data-review checklist](../../pending-data-review.md#document-extraction)
before authoritative evaluation. Until support is reviewed, intervals remain
unsupported; target split sizes alone do not establish independent-source
support for every annotation subset or latency tail. See
[insufficient confidence-interval support](../../metrics.md#insufficient-confidence-interval-support).

The [shared data-validation contract](../../data-validation.md) defines checks
for claimed capabilities, reviewed blanks/negative objects, complete page
inventories, geometry, hierarchy, and split isolation. Explicit empty text or
element sequences must not be rejected merely for being empty, nor inserted as
defaults for missing fields. These checks finish before candidate loading and
measurement; they do not change document metric eligibility or scoring.
Run `python -m experiments.benchmarks.validate document --profile <profile>`
after data preparation. Before candidate execution, the runner verifies the
matching sealed report against actual input bytes and applicable requirements;
it does not repeat full validation. Legitimate reviewed empty references remain
valid.

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
reference-eligible aggregate records how many documents were eligible and how
many independent sources support its interval. Verified blank sources and
table/formula negatives are labelled explicitly; they are not inferred from
missing reference fields.

## Metrics and their purpose

The result groups are:

| Category | Main metrics | Why they are needed |
|---|---|---|
| Text | Content F1, Reading Order NED; precision/recall and recognition diagnostics | Separates recovery of the right text from its correct ordering. |
| Pages | Page Content F1; coverage, attribution, and duplication | Checks recovery on the correct pages without duplication. |
| Layout | Layout Element F1, Element Type Recall, Hierarchy Preservation Rate; detection and localization diagnostics | Checks elements, semantic types, relationships, and locations separately. |
| Tables | Table Detection F1, Table Content F1, TEDS; precision/recall and TEDS-S | Separates finding tables from recovering their content and structure. |
| Formulas | Formula Detection F1, Formula Exact Match; detection diagnostics and CDM similarity | Separates finding mathematical expressions from recognizing them correctly. |
| Reliability | Unexpected Empty Output Rate, Duplicate Content Rate, Repeatability Success Rate, Attempt Failure Rate | Distinguishes output defects from execution failures and repeatable delivery. |
| Operational | Cold load, first-item and warm latencies, throughput, RAM, VRAM, temporary disk | Measures initialization, steady-state work, and resource requirements. |

Metric applicability depends on source format and explicit reference capabilities.
Images are one-page inputs; native DOCX has no fixed visual pagination or
bounding-box task. Tables and formulas distinguish detection on verified
negatives from reconstruction of actual reference objects. See the
[format applicability and detailed metric definitions](metrics.md#format-applicability).

## Evaluation setup

Docling 2.117.0, English, OCR scale 3.0, table-cell matching enabled,
code enrichment disabled, canonical structured output, and native DOCX
ingestion remain fixed. These settings define the common evaluation environment;
they are not factors in the configuration screen.

Document extraction runs as an isolated workload without an artificial VRAM
cap in any profile, including preflight. GPU-backed parser components must
remain on the requested GPU without framework offloading or silent CPU fallback;
native DOCX and Tesseract retain their explicit CPU routes. Peak VRAM remains
measured; actual OOM, placement violations, and missing required telemetry remain
failures. CPU applicability and CUDA measurement status follow
[shared Peak Device VRAM](../../metrics.md#peak-device-vram).

RapidOCR's CUDA placement check inspects the optimized ONNX Runtime graph at
initialization. CPU shape bookkeeping is permitted only when dimension-derived
integer values flow through verified shape operations into shape arguments,
not model feature computation. Learned computation stays on CUDA, and runtime
CPU retry is disabled. Graph verification is part of cold initialization,
requires no extra inference, and records its graph checksum and permitted CPU
nodes in placement evidence. Unverifiable graphs fail closed.

Every applicable execution is converted to the same canonical document
representation containing text, pages, ordered elements, types, hierarchy,
bounding boxes, tables, formulas, provenance, warnings, and timing. The
benchmark scores that representation without an EduMind cleanup profile. Raw
outputs remain unchanged. Prose comparison uses the
[shared symmetric projection and its exclusions](../../metrics.md#prose-scoring-projection),
not a parser-specific repair step; formulas, code, layout trees, and table trees
retain their own scoring representations.

Every authoritative reference declares a `reference_capabilities` list drawn
from `text`, `pages`, `reading_order`, `layout_boxes`, `element_types`,
`hierarchy`, `tables`, and `formulas`. Validation and metric eligibility follow
that declaration instead of assuming that every source has every annotation.
Full data validation currently checks reference self-comparisons in the pinned
official Docker evaluators. Self-comparison alone cannot establish that header
text and spans affect table scores. The approved
[table-scorer normalization contract](metrics.md#table-scorer-normalization)
requires symmetric pinned normalization and header-text/header-span perturbation
checks as evaluator acceptance evidence. Reference checks remain outside candidate
measurements; profile startup verifies the sealed report and current input identity.
Smoke fixtures may infer capabilities from their inline annotations. Table and
formula capability declarations also carry explicit `has_table` and
`has_formula` booleans so invented objects on verified negatives count as false
detections. Unclaimed tasks are inapplicable; missing annotations for a claimed
capability are invalid inputs and must be rejected before candidate execution.

OmniDocBench annotations act as ground truth for every applicable metric on its
samples. EduMind calculates the common text, page, layout, detection,
reliability, and operational metrics so their definitions remain identical for
OHR-Bench, PureDocBench, DocPTBench, native DOCX, and EduMind-specific samples.
Only the specialized table-tree and formula scorers—TEDS, TEDS-S, and CDM—come
from the pinned official evaluator. The approved
[ExpRate@CDM contract](metrics.md#formula-exact-match-expratecdm) derives exactness
from its raw matched/reference/predicted character counts, not rounded CDM F1.

**Implementation boundary:** the current project wrapper calls TEDS directly
after adding an HTML/body envelope; it bypasses the required official HTML
normalization. It also returns only rounded CDM F1, which the project currently
tests for equality to one. Symmetric table normalization, raw-count transport,
exact-count scoring, and their acceptance checks remain unimplemented changes.
Authoritative results under these approved definitions require the wrapper
migration and versioned evaluation provenance. The current scorer path remains
outside that contract, and historical results retain their original definitions.

The execution boundary is explicit:

```text
Python 3.12: Docling/Granite/Paddle extraction -> canonical prediction
             -> common EduMind metrics
Python 3.10 Docker (approved scorer contract):
    table HTML pairs -> symmetric pinned normalization -> TEDS and TEDS-S
    formula LaTeX pairs -> official CDM similarity + raw counts
Python 3.12: raw CDM counts -> exact-count ExpRate@CDM
```

The table and formula pairs for one extraction profile are scored in one
container call after extraction finishes. Container startup and scoring time are
evaluation overhead, not extraction latency. In addition to the pinned
OmniDocBench revision and immutable Docker image digest, the approved scorer contract
requires the normalization identity, project wrapper revision, and exact-count
policy version. Retain raw and scorer-facing table HTML and per-formula counts
with the existing prediction/evaluation artifacts; a scalar or checksum alone
cannot reconstruct those comparisons. The current wrapper does not yet retain
this added provenance; it is part of the pending migration.

## Benchmark workflow and selection

```text
smoke:
separate CPU and CUDA fixture runs
→ verify loading, extraction, scoring, artifacts, and MLflow; no selection evidence

preflight:
fresh CUDA worker per declared configuration → demanding valid development/stress input
→ hardware qualification report; no warmup or quality ranking

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

Development determines both the Standard settings and the parser-architecture
finalists. Validation confirms only those finalists; it is not the first local
comparison of Granite Docling or PaddleOCR-VL. After validation, the engineer
records exactly one PDF winner and one image winner. Locked evaluation applies
those frozen routes to the untouched locked-test split; native DOCX remains the
only valid DOCX route. The locked result is reporting-only and cannot reopen
configuration or architecture selection.

## Execution and measurements

Within a development, validation, or locked comparison, every candidate receives
the same deterministically shuffled reference-eligible documents, a fresh-worker
cold measurement, one complete representative warmup, and three measured attempts
for every document. Repeatability uses all documents in that profile's split,
not a candidate-dependent subset. Smoke uses one measured attempt per fixture
and does not estimate repeatability. Preflight uses no warmup and terminates its
worker; it does not warm a later comparison.

Document preflight currently selects the largest source file in each supported
format from the reviewed development manifest. File size is a deterministic
stress-selection proxy, not a proven page-count or image-resolution boundary.
Review its coverage against the supported workloads before authoritative use;
normal runtime OOM, placement, and telemetry gates still apply to every profile.

| Measured attempt | Evidence retained |
|---|---|
| 1 | The sole quality output, timing, validity/failure state, and canonical output for repeatability. |
| 2 and 3 | Output validation, timing, validity/failure state, and canonical outputs for repeatability; no additional quality scoring. |

All attempts use the same settings and inference seed. A failed first attempt
leaves quality unavailable; a later failure does not erase valid first-output
quality. See [first and later attempt failures](../../metrics.md#first-and-later-attempt-failures).
Recoverable failures do not cancel scheduled work; missing execution or records
leave the affected evidence incomplete. See
[execution accounting](../../metrics.md#execution-accounting-and-result-completeness)
and [document-specific output coverage](metrics.md#scoring-inputs-eligibility-and-result-status).

Repeatability Success Rate applies
[shared pairwise repeatability](../../metrics.md#pairwise-repeatability) to each
document's three scheduled pairs and its
[canonical structured output](metrics.md#repeatability-success-rate). Attempt Failure Rate records
the fraction of failed measured attempts. For `A, A, failure`, both rates are
`1/3` and quality comes from the first `A`. For `failure, A, A`, the rates are
unchanged but quality is null/unavailable. No cached previous
extraction may substitute for a measured request. All output and canonicalization
settings, including normalized-box precision and stable parent-ID mapping, are
frozen and recorded.

Warm latency uses successful measured attempts with completion coverage;
failure durations remain separate and enter measured batch time. Throughput
counts physical input pages of successful measured attempts and divides by the
actual measured batch interval, including failures. Loading, warmup, and offline
scoring are outside warm latency and batch throughput. Loading and warmup are
inside resource monitoring; offline scoring is outside extraction resource
measurements, which stop after the final measured request.

Apply [shared confidence intervals](../../metrics.md#confidence-intervals)
using reviewed independent-source groups, with every member document, page,
object, capture variant, and attempt kept together. A document is an independent
unit only when its reviewed source grouping establishes that. Quality uses first-attempt scores;
repeatability uses each document's three-attempt pair score; failure uses its
attempt-failure fraction. For example, 100 documents with three attempts remain
100 independent units if their source families are independent. The
[document CI applications](metrics.md#aggregation-and-confidence-intervals)
specify conditional cohorts, support, and operational eligibility.

### Scoring and result slices

The exact formulas, eligibility rules, ranges, and directions are defined in
[document metrics](metrics.md). Eligibility is determined from the reference, not
the candidate's recovered matches or supported features. Missing expected
elements contribute zero to attribute/reconstruction scores, while a genuinely
absent reference task is inapplicable. Valid empty comparisons use explicit
best/worst conventions under the
[shared empty-comparison rules](../../metrics.md#content-and-detection-empty-comparisons).
A failed or invalid complete output has unavailable dependent quality, not a
fabricated empty score: see [failed or invalid outputs](../../metrics.md#failed-or-invalid-outputs).
An evaluator failure also leaves its required evaluation unavailable; repair the
evaluator or rescore saved predictions under
[missing measurements or evaluations](../../metrics.md#missing-measurements-or-evaluations).
Quality failures do not automatically select or disqualify candidates.

Each quality metric is calculated from the first measured output and aggregated
as a document-macro average across its reference-eligible documents and,
separately, relevant document groups. Reference-object and page scores first
average within their document; detection F1 is calculated per document before
averaging. Do not pool detection counts across documents or average different
categories into an overall score. This permits conclusions such as "high text
quality across all documents but weak table structure on scanned PDFs" instead
of hiding the weakness inside one mean.

## Results and MLflow structure

### Metric and document groups

An extraction profile produces one canonical structured document for every
applicable input. Results are organized by metric group and document group.

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

These are the sections defined in [document metrics](metrics.md). Content F1 is a text
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

tables.teds                     -> document average after averaging each document's reference tables
tables.pdf_scanned.teds         -> the same document-macro score for scanned PDFs only
```

Documents still carry annotations such as `has_table`, `has_formula`,
`multi_column`, and `layout_difficulty`. Those labels remain in the manifest and
per-sample artifact for investigation, but they do not create another required
MLflow namespace.

Metrics from different categories are never averaged together. `text.content_f1`
does not combine text, pages, layout, tables, formulas, latency, or memory. It is
only the total Content F1 across documents eligible for Content F1.

### Run hierarchy and artifacts

Document runs use `EduMind / Document`. CPU/CUDA smoke parents and one
all-candidate preflight parent precede the comparisons below. Human-readable
parent names omit timestamps and redundant benchmark prefixes; run IDs and
recorded timestamps retain invocation identity. Smoke parents use
`smoke-cpu-<format>` and `smoke-cuda-<format>` for applicable paths. `<format>` is
`pdf`, `image`, or `docx`, not a dataset/provider. `config` identifies the Docling
Standard setting screen; `parsers` identifies the architecture comparison.
Both are development comparisons, not new profiles. The configuration screen
with `--source all` creates three independent **parent runs**, because PDF,
image, and DOCX execute different valid configuration sets. The architecture
comparison creates PDF and image parents; the fixed native-DOCX route reuses
its development evidence. Validation and locked again evaluate all three
formats on their own unseen splits. Each parent is one fair comparison, not
a parser result itself. The Standard configuration tree is:

```text
MLflow experiment: EduMind / Document
├── parent: development-config-pdf
│   └── up to 24 children: one per qualified PDF extraction profile
├── parent: development-config-image
│   └── up to 12 children: one per qualified full-page image profile
└── parent: development-docx
    └── 1 child run: native Docling ingestion
```

`--source pdf`, `--source image`, or `--source docx` restricts execution to that
format's applicable comparison. After
the configuration screen, development architecture parents compare the selected
Standard profile with Granite Docling and PaddleOCR-VL. Validation parents then
contain only the architecture finalists recorded by the engineer. The DOCX
parent validates native Docling because the two visual parsers do not accept
native DOCX. The architecture tree below shows the roster when all three
architectures qualify; hardware-excluded architectures do not create comparison
children.

```text
MLflow experiment: EduMind / Document
├── parent: development-parsers-pdf
│   ├── child: <selected PDF Docling Standard profile>
│   ├── child: docling-vlm-granite-258m
│   └── child: paddleocr-vl-1.6
├── parent: development-parsers-image
│   ├── child: <selected image Docling Standard profile>
│   ├── child: docling-vlm-granite-258m
│   └── child: paddleocr-vl-1.6
└── native DOCX: reuse development-docx evidence; no duplicate parser comparison
```

The corresponding validation parents use
`validation-<format>` and contain
only the recorded finalists for that source. Locked parents use
`locked-<format>` and contain one
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

Configuration, finalist, and winner decisions must reference evidence produced
under the current document protocol and matching hardware qualification.
Changing either requires fresh matching evidence before advancing candidates;
the runner does not reuse a stale decision merely because its candidate names
still exist.

Paired comparisons use aligned document-level scalar results, including layout,
table, and formula detection precision/recall/F1 under document-macro aggregation.
Reference eligibility stays fixed, but failed first outputs can make the actual
scored cohorts differ. Paired quality comparisons use aligned first-attempt scores
defined for both candidates and identify that common cohort,
its independent-source count, and all exclusions. Failure/repeatability evidence
retains scheduled outcomes. A conditional comparison is not a claim about the
entire planned workload; missing required evaluations cannot be silently dropped.

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
- first-attempt document-macro quality, all-attempt repeatability/failure rates,
  eligible/scored/failed/unavailable counts, and `ci_lower`/`ci_upper` values when
  supported, including degenerate-interval flags;
- `operational.*` latency, throughput, memory, VRAM, and disk metrics when
  available;
- a candidate-result JSON containing the resolved parameters, status,
  aggregates, intervals, operational values, fingerprint, and any error; and
- `samples.parquet` with one row per scheduled document, including its
  designated first-attempt ID/status, quality metric values and statuses/reasons,
  eligible object counts, repeatability/failure values, document-group and
  annotation labels, warnings, and independent source ID;
- `timings.parquet` with one row for every scheduled measured attempt, its
  executed/not-executed state, completion/error, elapsed time, verified input
  page count, and canonical output fingerprint; and
- retained outputs for every executed attempt under the
  [shared evidence-retention rule](../../metrics.md#prediction-evidence-and-rescoring),
  with durable prediction references and input/conversion identities. Official
  quality evaluation uses
  only the first measured output; later outputs are validated and fingerprinted,
  not rescored. Warmup and loading records are separate from measured attempts.

Loading, warmup, telemetry, or placement failures retain their lifecycle phase,
elapsed-to-failure, available outputs, and partial resource observations.
Scheduled but unexecuted attempts remain explicit null observations, not
fabricated extraction failures. Missing measured attempts make the invocation
incomplete; they do not yield a complete Attempt Failure Rate or repeatability
score.

The MLflow comparison page is used for compact aggregates. The Parquet artifact
is the detailed evidence: it supports document-group breakdowns, diagnostic
filtering, and paired inspection of the same document across profiles. Raw input
documents are not duplicated into each child run.

The authoritative layout logs **every applicable metric in
[document metrics](metrics.md)**. The first name is the metric category. With no
document group in the name, the value is the total aggregate across every
eligible document processed by that child run:

```text
text.content_precision
text.content_recall
text.content_f1
text.character_error_rate
text.word_error_rate
text.reading_order_ned

pages.page_coverage
pages.page_content_f1
pages.page_attribution_recall
pages.duplicate_page_rate

layout.element_precision
layout.element_recall
layout.element_f1
layout.element_type_recall
layout.hierarchy_preservation_rate
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
with verified text. `tables.teds` averages each document's reference-table
scores first, then averages eligible documents equally.
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
layout.docx.hierarchy_preservation_rate

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
reliability.unexpected_empty_output_rate
reliability.duplicate_content_rate
reliability.repeatability_success_rate
reliability.attempt_failure_rate

operational.cold_model_load_seconds
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

Each development, validation, or locked sample-based quality or reliability
base key follows the
[shared suffix convention](../../methodology.md#shared-mlflow-metric-convention).
This applies equally to a total such as
`tables.teds` and a document-group result such as
`tables.pdf_scanned.teds`. Its `sample_count` makes clear when, for
example, a table result is based on fewer annotated documents than a text
result.

Interval eligibility follows the
[document CI applicability table](metrics.md#which-values-receive-an-interval),
using the [shared procedure](../../metrics.md#confidence-intervals).
Pages and repeated attempts do not increase independent-source support.

MLflow records evidence but does not choose a winner. After reviewing complete
parent and child runs, the engineer records finalists or a final extraction
policy in a separate decision file. The benchmark never modifies production
configuration automatically.

The approved document-parser profile is frozen before video and downstream
extraction are evaluated.
