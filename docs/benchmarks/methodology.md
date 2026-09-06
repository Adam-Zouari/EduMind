# How the EduMind experiments work

[Benchmark overview](overview.md) · [Metric definitions](metrics.md) ·
[Run commands](running.md) · [Candidate rationale](model-selection.md)

The experiments form a sequence. First compare extraction components. Separately,
compare chunker/embedding pairs, then retrieval methods and vector servers.
Generation is tested on fixed evidence so retrieval cannot influence it. Finally,
combine the approved retrieval and generation profiles and review their answers.

```text
1. Document extraction: configuration on development, architectures on validation
2. Audio extraction: all candidates on development, finalists on validation
1 + 2 --> 3. Video keyframes with parser and ASR frozen

4. Chunking x embedding --> 5. Retrieval and reranking
5 --> 6. Real vector-server retrieval

7. Generation on fixed evidence (independent of retrieval)

selected server + selected retrieval + selected generator
--> 8. Final RAG on validation + blinded human review
--> 9. Extraction-to-RAG confirmation on separate non-locked data
--> 10. Exactly one locked-test run
```

Document extraction, audio, chunking/embedding, the synthetic
vector-server workload, and generation can start independently. Video waits for
a document parser and ASR. Retrieval waits for chunking/embedding. Real server
retrieval waits for a retrieval stack. Final RAG waits for one selected server,
retrieval stack, and generator.

`smoke` checks that a small real path works and cannot support a selection.
`standard` compares every candidate on development data. `full` compares only
engineer-selected finalists on validation data. Standard, full, and locked runs
use seed 42, retain per-sample results, and report 95% confidence intervals for
eligible sample-based aggregates. A stage's locked split is used once for its one
engineer-selected final profile; it is never used to choose or tune candidates.

Every comparison gives its candidates the same samples. MLflow stores the exact
settings, revisions, data checksum, hardware, aggregate metrics, confidence
intervals, and per-sample results. The engineer chooses what continues; the
runner never chooses a winner or changes the application configuration.

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

Metrics are labeled **primary**, **secondary**, **diagnostic**, or
**operational**. Primary metrics answer the experiment's central question.
Secondary metrics explain or qualify a primary result. Diagnostic metrics expose
specific failure modes. Operational metrics measure latency, throughput, and
resources. These four role names are used consistently throughout this document.

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
negative cases needed to expose false detections. Table Content F1 and Table
Structure Score apply only to reference tables that the parser is expected to
recover. A document with no reference table does not receive a zero structure
score.

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

development / standard:
Docling configuration screen → document-group breakdowns
→ engineer selects one PDF configuration and one image configuration
→ selected Standard configurations + Granite Docling + PaddleOCR-VL
  are compared on the same development split
→ engineer records architecture finalists

validation / full:
only the engineer-selected architecture finalists
on unseen image/PDF inputs; native Docling on DOCX
→ engineer selects the complete parser profiles without adding candidates

future locked test, after the runtime routing policy is defined:
run the one frozen extraction policy once
```

Within a standard/full comparison, every profile receives the same
deterministically shuffled eligible samples, one cold measurement, warmups, and
three measured repetitions. Smoke uses one measured repetition because it is
only a wiring check.
Development determines both the Standard settings and the parser-architecture
finalists. Validation confirms only those finalists; it is not the first local
comparison of Granite Docling or PaddleOCR-VL. The locked test is not used for
tuning. Runtime routing and the resulting locked-test execution are deliberately
deferred until these parser results exist; neither is part of the current
configuration/architecture commands.

### MLflow result structure

MLflow uses one experiment named `EduMind / extraction`. A document command with
`--source all` creates three independent **parent runs**, because PDF, image,
and DOCX execute different valid configuration sets. Each parent is one fair
comparison; it is not a parser result itself. The standard configuration tree
is:

```text
MLflow experiment: EduMind / extraction
├── parent: extraction-document-configuration-pdf-<timestamp>
│   └── 24 child runs: one per PDF extraction profile
├── parent: extraction-document-configuration-image-<timestamp>
│   └── 12 child runs: one per unique full-page image profile
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
MLflow experiment: EduMind / extraction
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
only the recorded finalists for that source.

The parent run stores:

- profile (`smoke`, `standard`, or `full`), stage, dataset name and checksum;
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
- aggregate quality metrics and their `ci_lower` and `ci_upper` values when the
  metric is eligible for an interval;
- `operational.*` latency, throughput, memory, VRAM, and disk metrics when
  available;
- a candidate-result JSON containing status, aggregates, intervals, operational
  values, fingerprint, and any error; and
- a Parquet artifact with one row per processed sample, including sample ID,
  latency, individual metrics, document-group and annotation labels, warnings,
  and other diagnostic metadata.

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

Each standard, full, or locked sample-based quality or reliability base key follows the
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
Transcript Order Accuracy as document extraction does for two-dimensional
pages.

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
| Qwen3 ASR 1.7B + ForcedAligner 0.6B | Strong transcription candidate whose separate aligner provides timestamps. |

For Qwen, transcription runs first. Its ASR model is unloaded before the forced
aligner runs. The benchmark still measures transcription plus alignment as one
complete candidate.

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

Each authoritative split contains the canonical `clean`, `noisy`, `accented`,
and `multi_speaker` condition labels. They remain in the per-sample artifact for
diagnosis and do not create extra required
MLflow metric namespaces or a larger metric contract.

### Common input and output rules

Every candidate receives the same decoded audio waveform: mono, 16 kHz, with no
candidate-specific denoising, volume repair, prompting, or vocabulary hints.
Model-native feature extraction and the documented deterministic decoder remain
part of the ASR profile and are recorded. Candidate output receives only the
fixed evaluator normalization used by WER and CER; the evaluator does not repair
misspellings, remove repetitions, or rewrite transcripts.

All timestamp outputs are converted to ordered benchmark segments containing
text, start time, and end time. The evaluator uses one fixed transcript
alignment to associate predicted timed content with reference segments. It does
not require candidates to emit the same number of segments and does not truncate
unequal timestamp arrays. Boundary MAE is calculated only from valid alignments;
Alignment Coverage records how much of the timed reference aligned.

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
additional quality samples and do not constitute a determinism metric. Qwen's
complete execution includes both transcription and forced alignment.

Every profile uses the same explicitly requested CPU or CUDA device within one
comparison. Device, dtype, decoder, timestamp path, and runtime versions are
recorded. Silent CPU fallback invalidates the profile. If one selected profile
cannot complete on the requested device, that parent comparison is incomplete;
the engineer fixes the runtime plan and reruns it instead of comparing partial
results.

### Development, validation, and locked test

```text
smoke:
all runnable ASR paths on tiny committed speech and nonspeech fixtures
→ verify loading, transcription, timestamps, scoring, artifacts, and cleanup

development / standard:
all five ASR profiles on 54 speech clips and development reliability controls
→ engineer reviews MLflow and records finalists

validation / full:
engineer-selected finalists on 18 unseen speech clips and validation controls
→ engineer records exactly one selected ASR profile

locked test:
the selected profile once on 18 locked speech clips and locked controls
→ final unbiased ASR report; no further tuning in this benchmark version
```

Smoke validates wiring only. Development is where all candidates are compared.
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
| Reliability | Empty Transcript Rate, Nonspeech False-Transcription Rate | Measures complete empty output on speech and invented lexical output on verified nonspeech controls. |
| Operational | **Complete-Pipeline Real-Time Factor**, p50/p95 warm clip latency, cold model-load time, peak process-tree RAM, peak VRAM | Measures the complete transcription and alignment cost of the recorded CPU or GPU profile. |

Content F1 and Transcript Order Accuracy are not ASR metrics in this benchmark.
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

Audio uses the same MLflow experiment as the other extraction stages:

```text
MLflow experiment: EduMind / extraction
├── parent: extraction-audio-smoke-<timestamp>
│   └── one child per smoke-tested ASR profile
├── parent: extraction-audio-development-<timestamp>
│   ├── child: whisper-small-en-control
│   ├── child: canary-180m
│   ├── child: parakeet-tdt-0.6b-v2
│   ├── child: moss-transcribe-diarize
│   └── child: qwen3-asr-1.7b-aligned
├── parent: extraction-audio-validation-<timestamp>
│   └── one child per engineer-selected finalist
└── parent: extraction-audio-locked-test-<timestamp>
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
Applicable standard, full, and locked uncertainty bounds use the shared MLflow suffix
convention defined at the beginning of this document.

Corpus WER/CER and their components, timestamp metrics, reliability rates,
Repeat Transcript Agreement Rate, RTF, and sufficiently supported warm latency estimates receive clip-bootstrap
intervals. One cold-load observation and observed peak RAM/VRAM do not receive
fabricated intervals. Every bootstrap draw contributes to every metric that is
defined for that draw. A draw with no aligned timestamp segment still
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
| `candidate.json` | Candidate status, fingerprint, aggregate metrics, confidence intervals, operational values, and artifact references. |

Fields that do not apply to a row are absent or null, not fabricated as zero.
Raw audio and candidate predictions are not uploaded to MLflow. The frozen
speech manifest is uploaded and contains the verified reference transcripts,
source identifiers, and checksums needed to reproduce scoring.

Every successful standard, full, or locked child must contain all 16 aggregate
metric fields. Timestamp Boundary MAE is the sole nullable field, under the rule
above. A CPU profile may report zero VRAM only when execution confirms that no
GPU process was used; unavailable instrumentation is not converted to zero.
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

The current direct video command supports only the non-authoritative smoke
wiring check. Standard, full, and locked execution remains disabled until the
downloaded video annotations are inspected and the two pending rules in
[pending-data-review.md](pending-data-review.md)—ASR window stitching and timed
occurrence text matching—are frozen. The following is the approved design for
that runner, not a claim that authoritative video evidence can already be
produced.

```text
video
├─ FFmpeg extracts mono 16 kHz audio → frozen selected ASR
└─ FFmpeg extracts candidate keyframes → frozen selected document parser
        ↓
combine timestamped audio and visual segments
→ compare with transcript and visible-text references
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
already selected ASR; it does not reopen the five-model ASR comparison.

Development proceeds in this order:

1. Run the fixed-interval configurations at 5, 10, and 20 seconds.
2. Run the scene-change configurations at thresholds 0.30, 0.40, and 0.50.
3. Review visual quality and processing cost, then record one scene threshold.
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
| Operational | Visual Real-Time Factor, p50/p95 warm visual latency, cold visual-pipeline load time, peak visual process-tree RAM, peak visual VRAM, mean selected frames per video | Measures the keyframe and visual-parser cost that differs between configurations. |

Spoken and visible tokens remain separate. A video's transcript usually
contains far more words than its frames, so one combined recall value would be
dominated by audio and could hide a poor keyframe policy. The isolated video
comparison therefore scores visible content and its timing; the usefulness of
the combined audio-and-visual result is evaluated later in the end-to-end
retrieval experiment.

### MLflow result structure

Video uses `EduMind / extraction`. The shared ASR input is recorded first. The
nine development configurations are then created by three ordered comparisons
so the hybrid run can consume the engineer-selected scene threshold:

```text
MLflow experiment: EduMind / extraction
├── parent: extraction-video-input-asr-development-<timestamp>
│   └── child: <selected-asr-profile-across-all-development-videos>
├── parent: extraction-video-development-fixed-<timestamp>
│   ├── child: video-fixed-5s
│   ├── child: video-fixed-10s
│   └── child: video-fixed-20s
├── parent: extraction-video-development-scene-<timestamp>
│   ├── child: video-scene-0.30
│   ├── child: video-scene-0.40
│   └── child: video-scene-0.50
├── parent: extraction-video-development-hybrid-<timestamp>
│   ├── child: video-hybrid-<selected-threshold>-5s
│   ├── child: video-hybrid-<selected-threshold>-10s
│   └── child: video-hybrid-<selected-threshold>-20s
├── parent: extraction-video-validation-<timestamp>
│   └── one child per engineer-selected finalist
└── parent: extraction-video-locked-test-<timestamp>
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
In standard, full, and locked runs, Visual Content Precision/Recall/F1, Timed
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

Chunking and embedding are evaluated together. A chunker cannot be scored until
its chunks are represented and ranked, and an embedding cannot be judged without
the passages it embeds.

### Chunking strategies

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

### Embedding models

| Model | Role in the comparison |
|---|---|
| MiniLM L6 v2 | Lightweight production control. |
| Snowflake Arctic Embed M v2 | Strong small-model retrieval candidate with a long input limit. |
| F2LLM v2 0.6B | Alternative 0.6B retrieval architecture. |
| Octen Embedding 0.6B | Strong 0.6B retrieval challenger. |
| Qwen3 Embedding 0.6B | Strong directly comparable 0.6B candidate using documented last-token pooling. |
| Nemotron 3 Embed 1B | Larger nearby-size retrieval candidate. |
| Octen Embedding 4B | Upper-size challenger. |
| Qwen3 Embedding 4B | Upper-size quality candidate using the same documented Qwen retrieval recipe. |

Full public evidence and exact revisions are in
[model-selection.md](model-selection.md).

### Data

The RAG corpus uses pinned QASPER papers plus EduMind's verified structured
evidence set:

| Split | Papers | Used for |
|---|---:|---|
| Development | 100 | Standard component comparison |
| Validation | 40 | Full finalist comparison |
| Locked test | 40 | One final complete system only |

Each question stores answerability, accepted answers, evidence type, and exact
half-open evidence offsets. The structured supplement contains table, formula,
and mixed-evidence questions because QASPER is primarily text.

### Execution

Standard evaluates all:

```text
8 chunkers × 8 embeddings = 64 complete pairs
```

For every pair:

```text
split documents into chunks with exact source offsets
→ embed every chunk
→ embed each question
→ rank with exact NumPy cosine search
→ retain the top 20
→ compare retrieved spans with verified evidence spans
```

Exact NumPy search removes vector-database approximation from this experiment.
For semantic chunking, the tested embedding also creates the boundaries; that
result intentionally represents the complete semantic-chunker/embedding pair.

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | nDCG@3/@5, rank-aware Context Precision@3/@5, Context Recall@3/@5, Context Recall under 2,048 tokens | Measures early graded ranking, context cleanliness, evidence-span coverage, and coverage at equal context cost. |
| Secondary | nDCG@10, Hit Rate@5/@10, MRR | Shows deeper ranking and whether at least one useful passage appears. |
| Diagnostic | Precision/Recall@1/@3/@5/@10, MAP@3/@5/@10, Hit Rate@1/@3, Context Precision/Recall@1/@10 | Helps explain behavior, but binary chunk relevance depends on candidate-created boundaries. |
| Evidence slices | Primary metrics repeated for text, table, formula, and mixed questions | Reveals candidates that work well only on the majority evidence type. |
| Operational | Chunk count, mean/p95 chunk tokens, indexing time, p50/p95 query latency, vector storage, RAM, VRAM, determinism | Explains the storage and execution cost behind retrieval quality. |

Chunk-level Recall and MAP are comparable within one fixed chunker. Across
different chunkers, the number of relevant chunks changes with the boundaries,
so decisions use span-based Context Recall and graded nDCG instead.

The engineer approves up to three complete chunker/embedding pairs. No separate
embedding winner or chunker winner is required.

## 5. Retrieval and reranking

For the approved chunker/embedding pairs, do exact-term search, rank fusion, or a
learned reranker improve the ordering enough to justify their cost?

### Strategies

| Strategy | What runs | Why it is included |
|---|---|---|
| Dense | Cosine ranking from the selected embedding | Semantic baseline. |
| BM25 | Lexical ranking from word frequency and rarity | Finds names, exact terminology, identifiers, and codes. |
| RRF | Dense top 20 + BM25 top 20 combined by reciprocal ranks | Combines rankings without mixing incompatible raw scores. |
| RRF + reranker | RRF top 20 reordered by a query/passage model | Tests deeper relevance scoring before final context packing. |

RRF is tested with these rerankers:

- MiniLM cross-encoder control;
- Ettin 150M;
- Ettin 400M;
- Ettin 1B;
- Qwen3 Reranker 4B.

Together with Dense, BM25, and RRF, this produces eight retrieval methods for
each selected chunker/embedding pair.

### Data and execution

The experiment uses the same frozen QASPER-plus-structured corpus and evidence
offsets as the previous stage.

```text
use an approved chunker/embedding pair
→ retrieve dense top 20, BM25 top 20, or both
→ apply RRF when required
→ rerank the fused top 20 when required
→ pack results under 2,048 tokens
→ compare the final ranking with verified evidence
```

Dense-only does not secretly build BM25. BM25-only does not embed queries unless
the selected semantic chunker requires its embedding to create boundaries.

### Metrics and why they are used

The metric roles stay the same as in chunking/embedding so changing only the
retrieval strategy remains interpretable:

- **Primary:** nDCG@3/@5, Context Precision@3/@5, Context Recall@3/@5, and
  Context Recall under 2,048 tokens.
- **Secondary:** nDCG@10, Hit Rate@5/@10, and MRR.
- **Diagnostic:** conventional Precision/Recall/Hit Rate at the remaining
  cutoffs, MAP@3/@5/@10, Context Precision/Recall@1/@10, determinism, and the
  primary metrics split by evidence type.
- **Operational:** total retrieval p50/p95 latency, indexing time, retrieved
  tokens, chunk distribution, storage, RAM, and VRAM.

Reranker time is included in total retrieval latency. Binary chunk Recall and MAP
are not used to compare different chunkers because their denominators change with
chunk boundaries.

The engineer approves up to three complete retrieval stacks. A stack contains
the chunker, embedding, retrieval method, and reranker when applicable.

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

| Profile | Workload |
|---|---|
| Smoke | 1,000 vectors at dimension 384; 50 queries; concurrency 1 |
| Standard | 100,000 vectors at dimensions 384 and 1,024; 500 queries; concurrency 1/8/32 |
| Full | Selected real embeddings plus 1,000,000 clustered vectors; up to 1,000 queries; concurrency 1/8/32/64 |

Synthetic vectors contain clusters and 5% near-duplicates. Metadata creates
filters matching approximately 50%, 10%, 1%, and in full 0.1% of records.

Standard/full test supported HNSW combinations of:

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
| Validity checks | Health, cosine behavior, dimension rejection, compound/empty filters, replacement, deletion, persistence, restart, and ANN-index verification | Determines whether results are trustworthy; these are not quality scores. |
| Primary | ANN Recall@3/@5/@10, Filtered ANN Recall@3/@5/@10, Filter Correctness, Empty-Filter Correctness | Measures preservation of exact neighbours and metadata behavior at application-relevant depths. |
| Secondary | ANN and Filtered ANN Recall@1, complete RAG nDCG@3/@5 and Context Recall/Precision@3/@5 | Shows rank-one behavior and whether ANN results preserve real evidence retrieval. |
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
| Qwen3 1.7B, thinking disabled | Small production control. |
| MiniCPM5 1B, reasoning enabled | Compact reasoning candidate. |
| G9v3 3B, reasoning enabled | Middle-size general-capability candidate. |
| Qwen3.5 4B, reasoning enabled | Upper compact quality candidate. |

No trustworthy public benchmark compares all four under EduMind's grounded QA,
citation, refusal, faithfulness, and local-latency protocol. Public evidence made
the shortlist; this experiment makes them directly comparable.

### Data

Standard uses 24 development questions balanced across answerability, answer
type, and evidence type as an initial screen. Full evaluates only
engineer-selected generator finalists on the complete frozen validation question
set; it is not limited to 24 questions.

- Answerable questions receive their verified numbered evidence blocks.
- Unanswerable questions receive text from their document that does not answer
  the question.
- Retrieval is not run in this stage.

Using frozen evidence prevents a good generator from being penalized by a poor
retriever.

### Execution

Every generator uses its exact pinned local snapshot, official chat template,
the same CPU or CUDA device, native checkpoint dtype, temperature 0, seed 42,
8,192 context tokens, and at most 256 generated tokens. No model receives hidden
quantization or CPU/GPU offload.

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
| Primary | Citation Precision/Recall/F1 on answerable questions, Answerability Balanced Accuracy, Unsupported Answer Rate, Malformed Output Rate | Measures evidence use, answer/refusal decisions, unsupported answers, and protocol failures without penalizing correct refusals for having no citations. |
| Secondary | Token F1 | Provides partial answer-correctness evidence before human review. |
| Diagnostic | Exact Match, ROUGE-L, Refusal Precision/Recall/F1, HHEM on substantive non-refusal answers, determinism, prompt/answer/reasoning token counts | Explains lexical similarity, refusal errors, automated support estimates, stability, and verbosity. HHEM never replaces human Faithfulness. |
| Operational | Cold load, Time to First Token, generation time, total p50/p95 latency, tokens/second, peak RAM, peak VRAM | Separates startup, responsiveness, decoding speed, total latency, and memory. |

The engineer approves up to three generator profiles after inspecting automatic
quality, citation/refusal behavior, latency, and resources. Human review happens
only after retrieval and generation are combined.

## 8. Final RAG and human review

Which complete retrieval-and-generation system gives the best evidence-backed
answers when every component runs together?

### Systems tested

Standard crosses only approved finalists:

```text
1 approved vector-server profile
× up to 3 retrieval stacks
× up to 3 generators
× top_k {3, 5}
= at most 18 complete systems
```

### Data and execution

Final RAG uses the validation manifest. For every question, the complete path
runs:

```text
chunk document
→ embed chunks and question
→ store/query dense vectors through the approved server
→ apply BM25/RRF/reranking selected for the stack
→ pack top 3 or top 5 under 2,048 context tokens
→ number the evidence blocks
→ generate the answer and citations
```

For `top_k=3`, retrieval quality is reported at 3. For `top_k=5`, it is reported
at 3 and 5. Final RAG does not report @10 from a list that contains only three or
five passages; the dedicated retrieval experiment owns @10 conclusions.

Final RAG reports nDCG, Context Precision, Context Recall, and token-budget recall
at the available cutoff; the primary and secondary generation metrics from
Experiment 8; and retrieval, generation, server-call, and complete end-to-end
p50/p95 latency. RAM and VRAM remain operational measurements.

### Human review

The engineer chooses exactly three successful complete systems. The exporter
selects 20 common questions and creates:

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

- **Retrieval:** nDCG, Context Precision, and Context Recall at the system's
  actual top-K, plus Context Recall under 2,048 tokens.
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
