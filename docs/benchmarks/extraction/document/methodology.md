# Document extraction methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Terminology and unit of comparison](#terminology-and-unit-of-comparison)
- [Which metrics apply to which inputs](#which-metrics-apply-to-which-inputs)
- [Data](#data)
- [Phase A: Docling Standard configuration screen](#phase-a-docling-standard-configuration-screen)
- [Phase B: scoring and result slices](#phase-b-scoring-and-result-slices)
- [Phase C: parser architecture comparison](#phase-c-parser-architecture-comparison)
- [Development, validation, and locked test](#development-validation-and-locked-test)
- [MLflow result structure](#mlflow-result-structure)

Which complete parser profile best converts educational images, PDFs, and DOCX
files into accurate text and useful structure, including pages, reading order,
tables, and formulas?

OCR is not tested as an isolated product. It is tested inside the complete
document pipeline because OCR text and boxes influence layout, reading order,
page attribution, tables, and formulas.

## Terminology and unit of comparison

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

## Which metrics apply to which inputs

The following matrix defines applicability. `Yes` means the metric group is
expected for that input when its required reference annotation exists.
`Conditional` means only the relevant annotated subset is scored. `No` means
the metric would not represent the native input: retain an inapplicable status
and reason, and omit its numeric aggregate rather than recording zero.

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

### Text

Content Precision, Content Recall, Content F1, CER, and WER apply to every input
with verified reference text, including explicitly verified empty text: images,
every PDF family, and DOCX. Reading Order NED applies when complete element-order
annotations exist, including a single element or an explicitly empty sequence;
it does not require a pair of successfully matched elements.

### Pages

Page Coverage, Page Content F1, Page Attribution Recall, and Duplicate Page
Rate apply to PDFs and to images treated as one-page documents. They do not
apply to native DOCX. DOCX page boundaries depend on a renderer, fonts, margins,
and page settings; native ingestion deliberately avoids inventing a fixed visual
pagination.

### Layout

Visual layout detection and Mean Bounding-Box IoU apply to images and PDFs with
layout boxes. Native DOCX does not receive bounding-box metrics. Semantic
Element Type Recall and Hierarchy Preservation Rate can apply to DOCX because
headings, paragraphs, lists, captions, nesting, and parent-child relationships
exist in the authored structure without rendering. Reading order is kept in the
text group even though it evaluates complete element sequences after matching.

### Tables

Table metrics can apply to images, every PDF family, and DOCX. Detection
precision, recall, and F1 use table-presence annotations, including verified
negative cases needed to expose false detections. Table Content Precision/Recall/F1,
TEDS, and TEDS-S apply only to reference tables that the parser is expected to
recover. No reference tables makes those reconstruction tasks inapplicable;
verified-negative detection still distinguishes correct absence from invented
tables. A missed real table, including one with blank cells, receives zero.

### Formulas

Formula metrics can apply to images, every PDF family, and DOCX. Detection
metrics use annotated positive and verified-negative cases. Formula Recognition
Similarity and Formula Exact Match apply only to reference formulas. A
formula-free document does not receive a zero recognition score.

### Difficult layouts

`layout_difficulty` is a per-sample diagnostic label, not another metric
category or required MLflow namespace. If the general layout result needs
investigation, the per-sample artifact can be filtered to documents with
columns, unusual reading order, dense pages, or overlapping elements.

### Reliability and operations

Reliability records account for every scheduled supported input, including
extraction failures. Duplication diagnostics require a valid completed first
output; smoke does not measure repeatability. Complete-document latency,
first-item latency, RAM, VRAM, temporary disk, and appropriate throughput
measurements apply to every processed
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
held-out samples. The frozen manifests—not the source-pool
names—define the actual cases and record every selected ID, source revision,
license, checksum, document family, source type, and available annotations. An
authoritative run cannot be claimed until those manifests and references are
complete.

The [shared data-validation contract](../../data-validation.md) defines checks
for claimed capabilities, reviewed blanks/negative objects, complete page
inventories, geometry, hierarchy, and split isolation. Explicit empty text or
element sequences must not be rejected merely for being empty, nor inserted as
defaults for missing fields. These checks finish before candidate loading and
measurement; they do not change document metric eligibility or scoring.
Currently the runner checks manifests/assets/references inline before candidate
execution. The standalone validator and matching-report reuse are planned
changes, including alignment of those checks with legitimate empty references.

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

## Phase A: Docling Standard configuration screen

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

The applicability rules prevent meaningless duplicate work:

| Source | Configurations executed | Reason |
|---|---|---|
| PDF | Up to 24 qualified configurations | Every factor can affect digital, scanned, mixed, or broken PDFs. |
| Image | Up to 12 qualified engine × table × formula profiles | Images always use full-page OCR, so the two PDF OCR modes would duplicate work. |
| DOCX | One qualified native Docling profile | OCR engine and PDF OCR mode do not apply to native DOCX parsing; one profile still receives all scheduled attempts. |

Docling 2.117.0, English, OCR scale 3.0, table-cell matching enabled,
code enrichment disabled, canonical structured output, and native DOCX
ingestion remain fixed. They define the common evaluation environment rather
than useful strategy questions.

## Phase B: scoring and result slices

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
`has_formula` booleans so invented objects on verified negatives count as false
detections. Unclaimed tasks are inapplicable; missing annotations for a claimed
capability are invalid inputs and must be rejected before candidate execution.

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
| Reliability | Whether valid outputs are unexpectedly empty or duplicated, repeat successfully, and avoid attempt failures. |
| Operations | Cold loading, first-request cost, warm latency, throughput, RAM, VRAM, and temporary disk. |

The exact formulas, eligibility rules, ranges, and directions are defined in
[document metrics](metrics.md). Eligibility is determined from the reference, not
the candidate's recovered matches or supported features. Missing expected
elements contribute zero to attribute/reconstruction scores, while a genuinely
absent reference task is inapplicable. Valid empty comparisons use explicit
best/worst conventions; a crash is not a valid empty result. Scored failure
penalties, inapplicable tasks, unavailable evaluations, incomplete execution,
and invalid references have distinct statuses and reasons. Evaluator failures
are repaired or rescored from saved predictions rather than labelled poor
extractions. Quality failures do not automatically select or disqualify candidates.

Each quality metric is calculated from the first measured output and aggregated
as a document-macro average across its reference-eligible documents and,
separately, relevant document groups. Reference-object and page scores first
average within their document; detection F1 is calculated per document before
averaging. Do not pool detection counts across documents or average different
categories into an overall score. This permits conclusions such as "high text
quality across all documents but weak table structure on scanned PDFs" instead
of hiding the weakness inside one mean.

## Phase C: parser architecture comparison

After reviewing the Standard-pipeline screen, the engineer records the selected
Docling profiles. The architecture comparison uses those complete profiles and
the other qualified architectures below; preflight exclusions remain visible:

| Parser profile | Question answered |
|---|---|
| Selected Docling Standard profile | How well does the conventional OCR, layout, table, and formula pipeline perform? |
| Granite Docling 258M | Does Docling's compact full-page VLM improve complete document parsing? |
| PaddleOCR-VL-1.6 | Does an independent visual parser outperform the two Docling architectures? |

The common architecture comparison uses images and PDFs. DOCX is evaluated
through native Docling and reported as format coverage; it is not rasterized to
give visual parsers artificial DOCX support.

## Development, validation, and locked test

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

Within a development, validation, or locked comparison, every candidate receives
the same deterministically shuffled reference-eligible documents, a fresh-worker
cold measurement, one complete representative warmup, and three measured attempts
for every document. Repeatability uses all documents in that profile's split,
not a candidate-dependent subset. Smoke uses one measured attempt per fixture
and does not estimate repeatability. Preflight uses no warmup and terminates its
worker; it does not warm a later comparison.

| Measured attempt | Evidence retained |
|---|---|
| 1 | The sole quality output, timing, validity/failure state, and canonical output for repeatability. |
| 2 and 3 | Output validation, timing, validity/failure state, and canonical outputs for repeatability; no additional quality scoring. |

All attempts use the same settings and inference seed. Never replace the
predesignated first output with a later success or best result. A later failure
does not erase a valid first output. A failed first attempt receives the
applicable bounded quality penalties; unavailable CER/WER diagnostics explicitly
report completed-output coverage. Attempt all remaining requests after
recoverable failures. A fatal interruption that prevents required requests is
incomplete, not a set of fabricated observed failures. Loading/warmup failures
are setup failures, not failed observations for every unexecuted document.

Repeatability Success Rate compares the three scheduled pairs for each document,
counting only pairs of valid identical outputs. Attempt Failure Rate records
the fraction of failed measured attempts. For `A, A, failure`, both rates are
`1/3` and quality comes from the first `A`. For `failure, A, A`, the rates are
unchanged but quality receives first-attempt failure penalties. Matching error
messages never earn agreement credit. Identical incorrect outputs can repeat
successfully; their quality scores remain incorrect. No cached previous
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

Confidence intervals are computed afterward from saved document-level results.
Quality intervals use first-attempt scores; repeatability intervals use each
document's three-attempt pair score; failure intervals use its attempt-failure
fraction. Bootstrap complete independent source documents 10,000 times with
seed `42`, retaining pages, objects, capture variants, and attempts together.
The 2.5th/97.5th percentiles form the estimated 95% bounds. These are arithmetic
resamples, not extra inference runs; 100 documents with three attempts remain
100 independent units if their source families are independent. Paired
comparisons use the same sampled source IDs for both candidates. Support,
unavailable-value, and degenerate-interval rules are defined in [document metrics](metrics.md).

Development determines both the Standard settings and the parser-architecture
finalists. Validation confirms only those finalists; it is not the first local
comparison of Granite Docling or PaddleOCR-VL. After validation, the engineer
records exactly one PDF winner and one image winner. Locked evaluation applies
those frozen routes to the untouched locked-test split; native DOCX remains the
only valid DOCX route. The locked result is reporting-only and cannot reopen
configuration or architecture selection.

## MLflow result structure

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
native DOCX. The architecture tree below shows the roster when all three
architectures qualify; hardware-excluded architectures do not create comparison
children.

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

Paired comparisons use aligned document-level scalar results, including layout,
table, and formula detection precision/recall/F1 under document-macro aggregation.
Reference-fixed eligibility and first-attempt failure penalties preserve the
same comparison cohort for main quality metrics. Completed-output diagnostics
and successful-completion latency comparisons explicitly identify their common
observed cohort and exclusions; missing evaluations never become normal
authoritative comparisons by silently dropping documents.

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
- retained outputs for every executed attempt. Official quality evaluation uses
  only the first measured output; later outputs are validated and fingerprinted,
  not rescored. Warmup and loading records are separate from measured attempts.

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

Each development, validation, or locked sample-based quality or reliability base key follows the
shared suffix convention. This applies equally to a total such as
`tables.teds` and a document-group result such as
`tables.pdf_scanned.teds`. Its `sample_count` makes clear when, for
example, a table result is based on fewer annotated documents than a text
result.

Confidence intervals are not attached indiscriminately:

- text, page, layout, table, formula, and reliability aggregates receive 95%
  intervals when they are calculated from enough independent samples;
- p50 and p95 latency receive intervals when enough independent source documents
  support the estimate; pages and repeated attempts do not increase that count;
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
