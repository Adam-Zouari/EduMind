# Document extraction metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Empty comparisons, eligibility, and result status](#empty-comparisons-eligibility-and-result-status)
- [Prose scoring projection](#prose-scoring-projection)
- [Text content and recognition](#text-content-and-recognition-1)
- [Pages](#pages-1)
- [Layout and document structure](#layout-and-document-structure-1)
- [Tables](#tables-1)
- [Mathematical formulas](#mathematical-formulas-1)
- [Reliability and failure behavior](#reliability-and-failure-behavior-1)
- [Operational performance](#operational-performance-1)
- [Worked candidate interpretation](#worked-candidate-interpretation)
- [Document-extraction confidence intervals](#document-extraction-confidence-intervals)

The document experiment evaluates complete image, PDF, and DOCX parsing. It does
not reduce a parser to one overall score. Each retained metric answers a distinct
question about text, pages, layout, tables, formulas, reliability, or execution
cost.

OmniDocBench has two separate roles. Its selected pages supply ground-truth text,
reading order, element types and boxes, tables, and formulas for all applicable
metrics below. Its scoring code is reused only for TEDS, TEDS-S, and CDM;
ExpRate@CDM is derived from CDM results. EduMind calculates the remaining metrics
from the same canonical reference/prediction pairs because those definitions
must also work unchanged on OHR-Bench, PureDocBench, DocPTBench, native DOCX, and
EduMind-specific samples.

EduMind calculates the common metrics in its Python 3.12 environment. TEDS,
TEDS-S, and CDM alone run from the pinned official evaluator inside its verified
Python 3.10 Docker image. This changes only where those calculations execute;
all metrics still belong to the same extraction-profile child run in MLflow.

Within each quality category, **primary metrics** summarize the category's main
outcomes. The remaining metrics are **secondary metrics**: they explain the
primary results or expose a narrower failure mode. Primary status does not assign
weights, combine categories into an overall score, or select a candidate
automatically.

Each metric has a self-contained explanation: question, calculation, example,
interpretation, valid range, and preferred direction. Category introductions
define shared rules such as the element-matching protocol.

## Metric summary

The following tables provide the complete metric list for readers who only need
to know what the document-extraction benchmark measures. The detailed contracts,
examples, and confidence-interval rules follow the summary.

### Text content and recognition

| Metric | Question answered |
|---|---|
| Content Precision | How much extracted content is supported by the reference? |
| Content Recall | How much required content was recovered? |
| Content F1 | Does the extractor balance correct output and complete output? |
| Character Error Rate (CER) | How severe are character-recognition errors? |
| Word Error Rate (WER) | How severe are complete-word errors? |
| Reading Order NED | How much editing is needed to recover the complete expected element sequence? |

### Pages

| Metric | Question answered |
|---|---|
| Page Coverage | Did every expected page produce relevant content? |
| Page Content F1 | Was the correct text recovered within the correct page? |
| Page Attribution Recall | How many expected content-bearing elements were recovered on the correct page? |
| Duplicate Page Rate | How often did the parser repeat a page? |

### Layout and document structure

| Metric | Question answered |
|---|---|
| Layout Element Precision | How many predicted document elements are real? |
| Layout Element Recall | How many required document elements were detected? |
| Layout Element F1 | Does layout detection balance false and missed elements? |
| Element Type Recall | How many expected elements were recovered with the correct type? |
| Hierarchy Preservation Rate | How many expected annotated elements retained their required parent and level? |
| Mean Bounding-Box IoU | How accurately were expected elements localized, including missed elements? |

### Tables

| Metric | Question answered |
|---|---|
| Table Detection Precision | How many predicted tables are real tables? |
| Table Detection Recall | How many reference tables were found? |
| Table Detection F1 | Does table detection balance extra and missed tables? |
| Table Content Precision | How much text inside expected tables is supported by the reference? |
| Table Content Recall | How much text inside expected tables was recovered? |
| Table Content F1 | Was the text inside expected tables recovered without unsupported additions? |
| TEDS | How similar is the complete predicted table tree, including structure and cell text, to the reference? |
| TEDS-S | Was the table's row/cell structure and its merged-cell spans reconstructed correctly, ignoring cell text? |

### Mathematical formulas

| Metric | Question answered |
|---|---|
| Formula Detection Precision | How many predicted formula regions are real formulas? |
| Formula Detection Recall | How many reference formulas were found? |
| Formula Detection F1 | Does formula detection balance extra and missed formulas? |
| Formula Recognition Similarity (CDM) | How visually and structurally close is each recognized formula to its reference? |
| Formula Exact Match (ExpRate@CDM) | How often was a formula reconstructed perfectly? |

### Reliability and failure behavior

| Metric | Question answered |
|---|---|
| Unexpected Empty Output Rate | How often does extraction complete empty when the source requires usable content? |
| Duplicate Content Rate | How much repeated token content did the parser add without source support? |
| Repeatability Success Rate | How often do two scheduled attempts both deliver the same valid structured output? |
| Attempt Failure Rate | How often does a measured extraction attempt fail to deliver a valid output? |

### Operational performance

| Metric | Question answered |
|---|---|
| Cold Model-Load Time | How long does the fresh worker take to initialize the candidate before warmup? |
| First-item Latency | How long does loading plus the first complete request take in a fresh worker? |
| p50/p95 Warm Latency per Page | What are the typical and slow-tail steady-state page speeds? |
| Complete Document Latency | How long does a user wait for a complete source? |
| Batch Pages per Minute | What sustained batch capacity does the parser provide? |
| Peak Process-Tree RAM | How much system memory does the complete extractor require? |
| Peak Device VRAM | How much total memory is occupied on the assigned GPU during extraction? |
| Peak Temporary Disk | How much additional working-disk space does extraction require? |

All candidates receive the same canonical reference and output conversion.
The [shared prose projection](../../metrics.md#shared-conventions) handles harmless representation differences;
it does not repair words, remove page content, deduplicate text, or otherwise
clean a candidate's extraction.
Each metric is reported across its eligible documents and separately for
applicable document groups such as `image`, `pdf_scanned`, and `docx`. Quality
metrics use only the predesignated first measured attempt for each document.
The other measured attempts supply repeatability, attempt-failure, and timing
evidence; their quality is not rescored or averaged with the first output.

Quality aggregates are document-macro averages. Calculate each document's
precision, recall, and F1 before averaging; do not pool detection counts across
documents or derive F1 from averaged precision and recall. For table content,
table structure, and formula recognition, first average the reference-object
scores within each document, including zero for missed objects, then average
eligible documents equally. Pages and annotated elements likewise produce one
document-level score before the document average. Reliability rates average
document-level values; latency, throughput, and resource peaks keep their own
operational definitions. Bootstrap draws preserve the complete independent
source document and all its dependent observations.

Eligibility comes from each authoritative reference's explicit
`reference_capabilities` list: `text`, `pages`, `reading_order`,
`layout_boxes`, `element_types`, `hierarchy`, `tables`, and `formulas`. Smoke
fixtures may infer this list from inline annotations. A missing capability marks
the metric inapplicable, retaining its reason rather than fabricating a zero.
Table/formula capabilities use explicit
`has_table:false` and `has_formula:false` negatives in detection counts.

## Empty comparisons, eligibility, and result status

Eligibility is fixed from the reference before candidate execution and is the
same for every candidate. A parser that does not support an expected feature
still receives the applicable missed-feature score; it cannot omit that metric.
Verified absence is different from absent annotation. Invalid references,
including contradictory presence labels, invalid boxes, or missing claimed
table/formula representations, must be repaired before authoritative execution.

For a valid, completed output, prose Content Precision/Recall/F1 and layout,
table, and formula Detection Precision/Recall/F1 use this explicit convention:

| Evaluated reference units | Evaluated predicted units | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Empty | Empty | 1 | 1 | 1 |
| Empty | Nonempty | 0 | 0 | 0 |
| Nonempty | Empty | 0 | 0 | 0 |
| Nonempty | Nonempty | Normal calculation | Normal calculation | Normal calculation; zero when both inputs to F1 are zero |

The empty-reference recall values are benchmark conventions, not divisions by
zero. Prose emptiness is assessed after the frozen symmetric projection. Two
projected empty strings therefore compare as equal even when their raw strings
differ; raw strings remain available. Formula expressions, code, and table
structure do not receive that prose projection.

Table Content Precision/Recall/F1, TEDS, and TEDS-S require reference tables.
CDM and ExpRate@CDM require reference formulas. No reference objects means an
inapplicable reconstruction task, not a perfect or failed reconstruction. A
real missed table or formula contributes zero. A recovered table with blank
cells can have perfect empty-text Content scores while its actual structure
still receives TEDS/TEDS-S. A missed blank-cell table receives zero for all
three Content scores and both structure scores, regardless of its empty text.
Verified-negative detection remains applicable and catches invented objects.

Attribute metrics require eligible reference annotations. No annotated type,
hierarchy, box, or content-bearing page element means no corresponding attribute
task. Missing or incorrect predictions for existing eligible reference elements
receive zero rather than becoming inapplicable. An explicitly annotated root
with no parent is meaningful; an absent parent annotation is not a verified root.

Every per-document metric value retains its result status and reason:

| Status | Meaning and treatment |
|---|---|
| `scored` | A defined numeric result under that metric's contract; output-dependent quality requires a valid completed first output. |
| `inapplicable` | No reference task exists; retain null and the eligibility reason. |
| `unavailable` | A measurement or evaluator could not produce the required value; retain null, reason, and coverage counts. |
| `incomplete` | Scheduled execution or required attempt records are missing; do not fabricate observations. |
| `invalid_reference` | Reference annotations violate the contract; repair before evaluation. |

A failed first attempt is not a successfully extracted empty document. Every
applicable output-dependent quality metric is null/unavailable, including
bounded content/layout/reconstruction scores and Reading Order NED. Genuinely
inapplicable reference tasks stay inapplicable. Quality aggregates describe
valid completed first outputs and retain scheduled, reference-eligible, scored,
failed, and unavailable counts. Attempt Failure Rate and repeatability retain
their own numeric rules; completed measurements can survive later failure.
Do not replace missing quality with a later successful output.

The revised failure contract is approved documentation and requires runner
alignment before new runs can claim compliance. A complete set of recorded
attempt failures is reliability evidence, not successful recovery of the corpus.

Known malformed predicted table/formula reconstructions receive zero
reconstruction credit where applicable. An official evaluator crash, timeout,
or missing dependency is an unavailable evaluation, not a candidate-quality
score of zero. Resolve missing required evaluator results before publishing a
complete authoritative comparison; saved first-attempt predictions may be
rescored without rerunning inference. Any provisional aggregate based on only
available required evaluations is labelled incomplete, not a normal comparable
result. MLflow omits unavailable numeric scalars, but artifacts retain every
document and its status. Report scheduled, eligible, scored, failed,
inapplicable, unavailable, and positive/verified-negative reference counts.

## Prose scoring projection

Before prose text is compared, both sides undergo the same four representation
steps: Unicode NFC, case-folding, punctuation replacement with spaces, and
whitespace collapse. This reduces the effect of case and punctuation differences
on prose scores.

```text
Reference:   CAFÉ—based   learning
Prediction:  café based learning
Compared:    café based learning
```

The transformation is intentionally not a cleanup system. For example,
`algo-\nrithm` becomes the two units `algo rithm`; it is not repaired into
`algorithm`. Raw reference and prediction text remain available for inspection.
Formula, code, layout, and table-tree metrics use their own representations and
do not receive this prose projection.

## Text content and recognition

Repeated token occurrences are counted; token sets are not used.

**Primary metrics:** Content F1 and Reading Order NED.

- **Content F1** is primary because it summarizes whether the parser recovered
  the required text without adding unsupported text. It balances content
  precision and recall in one category-level measure.
- **Reading Order NED** is also primary because correct words can still be
  unusable when headings, columns, paragraphs, or list items are returned in the
  wrong sequence. It measures a different outcome from Content F1.
- **Content Precision and Content Recall** are secondary metrics. They separate
  hallucinated or extra content from missing content and therefore explain why
  Content F1 changed.
- **CER and WER** are supporting error diagnostics. They reveal character-level
  and word-level recognition mistakes, but neither alone measures both content
  completeness and unsupported output as directly as Content F1.

### Content Precision

**Question:** How much extracted content is supported by the reference?

Repeated occurrences are matched only up to the smaller count in the reference
and prediction. For example, if a word appears twice in the reference and three
times in the prediction, only two occurrences are correct.

In plain language:

```text
correct extracted tokens
────────────────────────
 all extracted tokens
```

**Example:**

```text
Reference:  machine learning model
Prediction: machine learning system

Correct extracted tokens = 2
All extracted tokens     = 3
Content Precision        = 2 / 3 = 0.67
```

High precision means the extractor rarely adds incorrect text.

**Range and direction:** `[0, 1]`; higher is better. Valid empty/empty comparison
receives one; either one-sided empty comparison receives zero under the shared
document convention. An applicable failed first attempt is null/unavailable.

### Content Recall

**Question:** How much required content was recovered?

Repeated occurrences are matched only up to the smaller count in the reference
and prediction, using the same rule as Content Precision.

In plain language:

```text
correct extracted tokens
────────────────────────
 all reference tokens
```

**Example:**

```text
Reference:  machine learning improves education
Prediction: machine learning

Content Precision = 2 / 2 = 1.00
Content Recall    = 2 / 4 = 0.50
```

Everything extracted is correct, but half of the reference content is missing.

**Range and direction:** `[0, 1]`; higher is better. Valid empty/empty comparison
receives one; either one-sided empty comparison receives zero. An applicable
failed first attempt is null/unavailable.

### Content F1

**Question:** Does the extractor balance correct output and complete output?

Content F1 combines the Content Precision and Content Recall calculated above.
Repeated token occurrences still count only up to the smaller reference and
prediction count.

In plain language:

```text
2 × Content Precision × Content Recall
──────────────────────────────────────
   Content Precision + Content Recall
```

F1 is high only when precision and recall are both high. It summarizes their
balance, while the individual values reveal whether errors came mainly from
additional text or missing text.

**Example:** If precision is `1.00` and recall is `0.50`, then:

```text
Content F1 = (2 × 1.00 × 0.50) / (1.00 + 0.50) = 0.67
```

**Range and direction:** `[0, 1]`; higher is better. Valid empty/empty comparison
receives one. Either one-sided empty comparison receives zero, as does a
nonempty comparison with no overlap. An applicable failed first attempt is
null/unavailable.

### Character Error Rate (CER)

**Question:** How severe are character-recognition errors?

The evaluator finds the minimum character substitutions, deletions, and
insertions needed to transform the reference into the prediction.

In plain language:

```text
character substitutions + deletions + insertions
────────────────────────────────────────────────
         number of reference characters
```

**Example:**

```text
Reference:  model
Prediction: motel

One substituted character
CER = 1 / 5 = 0.20
```

CER exposes errors such as `0` instead of `O`, `rn` instead of `m`, and
misspelled technical terms. Punctuation differences are intentionally ignored
by the prose projection. A perfect result has `CER = 0`.

**Range and direction:** `[0, infinity)`; lower is better. CER can exceed `1`
when insertions outnumber reference characters. Use the [JiWER empty-reference
convention](https://jitsi.github.io/jiwer/#a-note-on-empty-references): a valid
empty/empty comparison is zero; an empty reference with nonempty prediction
returns the character insertion count; a nonempty reference with empty prediction
is one. A failed first attempt has unavailable CER, not an invented worst score;
its diagnostic aggregate reports completed-output coverage and failures.

### Word Error Rate (WER)

**Question:** How severe are complete-word errors?

The evaluator finds the minimum word substitutions, deletions, and insertions
needed to transform the reference into the prediction.

In plain language:

```text
word substitutions + deletions + insertions
───────────────────────────────────────────
       number of reference words
```

**Example:**

```text
Reference:  the model is accurate
Prediction: the system is accurate

One substituted word
WER = 1 / 4 = 0.25
```

CER and WER are not redundant: one incorrect character may be a small fraction
of the characters while still making an entire word incorrect.

**Range and direction:** `[0, infinity)`; lower is better. WER can exceed `1`
when insertions outnumber reference words. The same JiWER convention gives zero
for valid empty/empty strings, the word insertion count for an empty reference
with nonempty prediction, and one for a nonempty reference with empty prediction.
A failed first attempt has unavailable WER; report completed-output coverage
and failures with its diagnostic aggregate.

### Reading Order NED

**Question:** How much editing is needed to recover the complete expected
element sequence?

Build the complete reference and prediction sequences using the frozen eligible
element types and segmentation granularity. One-to-one matched predictions
inherit their reference identities. Unmatched predicted elements receive distinct
extra identities; unmatched reference elements remain in the reference sequence.
Match without using order correctness to optimize the pairing. Compare element
identities, not characters in their serialized IDs.

In plain language:

```text
minimum element insertions + deletions + substitutions
─────────────────────────────────────────────────────
          length of the longer element sequence
```

**Example:**

```text
Reference:  A → B → C
Prediction: A → C

One deletion / three reference elements = Reading Order NED 0.33
```

For `A → B → X → C`, one extra element gives `1 / 4 = 0.25`. For
`B → A → C`, two substitutions give `2 / 3 = 0.67`. The score captures complete
sequence recovery, including omissions, insertions, order, and segmentation
differences; it is not a pure ordering accuracy conditional on successful matches.
This complete-sequence contract is an EduMind adaptation, not a claim of exact
equivalence to OmniDocBench's matched-element reading-order evaluation.

**Range and direction:** `[0, 1]`; lower is better. Valid empty/empty sequences
receive zero; exactly one empty sequence receives one. One correct element with
no extras receives zero, but one match out of a longer reference is not perfect.
An unclaimed reading-order capability makes the task inapplicable; missing
claimed annotations are invalid data. An applicable failed first attempt is
null/unavailable.

## Pages

**Primary metric:** Page Content F1.

- **Page Content F1** is primary because it requires the parser to recover the
  correct text on the correct page. It captures more than merely producing
  some output for each page.
- **Page Coverage** is supporting because a page can count as covered after only
  a small amount of relevant content is recovered.
- **Page Attribution Recall** is supporting because it measures recovery of
  expected content-bearing elements on their correct page. Unlike Page Content
  F1, it counts elements rather than token occurrences and does not penalize
  extra output directly.
- **Duplicate Page Rate** is supporting because it isolates one narrow failure:
  repeating pages that should appear once.

### Page Coverage

**Question:** Did every expected page produce relevant content?

Each reference content-bearing page counts as covered when at least one expected
usable content unit is matched to output on that page. Usable page content can
include an annotated table, formula, or other required structure, not only prose.
Empty text by itself does not establish a match.

In plain language:

```text
reference pages containing matched content
───────────────────────────────────────────
        reference content-bearing pages
```

**Example:** For a ten-page PDF, if page 7 produces no matching content:

```text
Page Coverage = 9 / 10 = 0.90
```

Coverage only establishes that each page produced some valid content. It does
not establish that all content on those pages was recovered correctly.

**Range and direction:** `[0, 1]`; higher is better. If the source is verified
entirely blank, valid empty output receives one and unsupported usable content
receives zero. An unclaimed page capability makes the metric inapplicable;
missing claimed page annotations are invalid data. An applicable failed first
attempt is null/unavailable.

### Page Content F1

**Question:** Was the correct text recovered within the correct page?

Prose Content F1 is calculated separately for every expected page and then
averaged within the document, including any unexpected predicted page numbers
as additional zero-score observations. Missing output on a page with nonempty
projected reference text receives zero. Verified empty page text paired with
empty predicted page text receives one, whether or not the parser emits a blank
page record. Unexpected predicted page numbers receive zero even when their
page text is blank. This metric evaluates page text, not non-text structure;
Page Coverage and the table/formula metrics evaluate that recovery separately.

In plain language:

```text
sum of individual page Content F1 scores
────────────────────────────────────────
 number of reference or unexpected pages
```

**Example:**

```text
Page 1 Content F1 = 1.00
Page 2 Content F1 = 0.80
Page 3 Content F1 = 0.00  ← missing page

Page Content F1 = (1.00 + 0.80 + 0.00) / 3 = 0.60
```

An unexpected additional page also receives zero, so this metric penalizes
missing, incorrect, misplaced, and unexpected page content.

**Range and direction:** `[0, 1]`; higher is better.

### Page Attribution Recall

**Question:** How many expected content-bearing elements were recovered on the
correct page?

For this metric, Hungarian one-to-one content matching maximizes total Content
F1 over admissible pairs with `Content F1 >= 0.5`; page numbers are deliberately
ignored until after the content pairs are formed. Eligibility is fixed to
reference elements with page annotations and nonempty comparison text units.
Prose uses the shared projection; code and formula units retain case and symbols,
as in the common matching rules. Empty text or a non-text identity alone is not
eligible for this content-first diagnostic. Each eligible reference element
receives one only when it is recovered and its predicted page is correct.
Missing elements and missing or wrong predicted page labels receive zero.

In plain language:

```text
matched elements assigned to the correct page
─────────────────────────────────────────────
 eligible reference content-bearing elements
```

**Example:** An extractor may recover a paragraph correctly but assign it to
page 2 instead of page 3. The paragraph still contributes to text recovery, but
it receives no Page Attribution Recall credit. If eight of ten eligible
reference elements are recovered and seven have the right page, recall is
`7 / 10 = 0.70`, not `7 / 8`.

**Range and direction:** `[0, 1]`; higher is better. No eligible reference
content-bearing page elements makes the task inapplicable. Zero recovered
elements with a nonempty eligible reference receives zero. An applicable failed
first attempt is null/unavailable.

### Duplicate Page Rate

**Question:** How often did the parser repeat a page?

Page records with nonempty projected page text are near-duplicates when their
Content F1 is at least `0.95`. Empty projected text cannot establish page
identity, even when a page contains non-text structure; those records are not in
this text-duplication diagnostic's denominator.

First match predicted pages one-to-one to reference pages by Content F1 at the
same `0.95` threshold, ignoring page labels and using the frozen Hungarian
assignment/tie-break. These reference-supported copies are not duplicates.
Initialize representatives with those matched predictions, then visit unmatched
predicted records in emitted order. A record similar to an existing
representative counts as an unsupported duplicate; otherwise retain it as a
new representative. Similarity is always checked against a representative, not
transitively through previously counted duplicates.

In plain language:

```text
unsupported duplicated page-text records
────────────────────────────────────────
 predicted records with comparison text
```

**Example:**

```text
Reference pages:             1, 2, 3
Predicted pages:             1, 2, 3, 4
Page 4 repeats page 2:       unsupported duplicate
Unsupported duplicate pages: 1

Duplicate Page Rate = 1 / 4 = 0.25
```

Legitimate repetition already present in the source does not count as a parser
duplicate. Two reference copies can support two predicted copies; a third copy
is extra. For two copies of entirely invented page text, only the additional
copy counts as duplication; the unsupported first copy affects content quality.

**Range and direction:** `[0, 1]`; lower is better. If no predicted page has
comparison text, valid output receives zero. A failed first attempt has no
observed duplication rate; retain an unavailable value and its failure reason,
not a claim that it duplicated nothing.

## Layout and document structure

The layout set contains headings, paragraphs, list items, captions, figures,
code blocks, and other annotated non-table/non-formula elements. Tables and
formulas are evaluated in their own sections.

**Primary metrics:** Layout Element F1, Element Type Recall, and Hierarchy
Preservation Rate.

- **Layout Element F1** is primary because it summarizes whether the expected
  document elements were found without inventing extra elements.
- **Element Type Recall** is primary because detecting a region is not enough:
  the parser must distinguish headings, paragraphs, list items, captions, and
  other element types.
- **Hierarchy Preservation Rate** is primary because parent-child relationships,
  heading levels, and list nesting determine whether the recovered document
  structure is usable. Neither detection nor type classification measures these
  relations.
- **Layout Element Precision and Recall** are secondary metrics. They distinguish
  extra detected elements from missed elements and explain Layout Element F1.
- **Mean Bounding-Box IoU** is supporting because it measures localization with
  zero for missed expected regions. Detection applies a match threshold; IoU
  additionally distinguishes loose and precise matched boxes. Neither geometry
  measure establishes correct semantic types or hierarchy.

Reference and predicted elements use Hungarian one-to-one matching, maximizing
total similarity over admissible pairs and permitting unmatched elements. The
reference annotations and task fix the similarity representation before
execution. Visual matches use bounding-box Intersection over Union (IoU), with
`IoU >= 0.5` required. Missing predicted geometry never switches such a match
to text. Non-visual comparisons without reference boxes use
representation-specific element Content F1 and require `Content F1 >= 0.5`.
This includes native DOCX and text/semantic references from PDF/image sources
that do not provide boxes; it does not make visual-box metrics applicable.

Prose uses the shared projected token occurrences. Code and formula identity
representations retain case and symbols; they do not use the prose
punctuation-removal rule. Empty text alone cannot identify an object: a reference
object without comparison text requires a frozen non-text identity, such as
table structure or media identity. An absent claimed reference identity is
invalid reference data; a missing predicted identity cannot create a match.
Apply the threshold to admissible match edges and use the frozen deterministic
tie-break, without optimizing the attribute being scored.
Visual layout, table, and formula pairs must also have equal page numbers;
identical boxes on different pages never match. Page Attribution Recall is
the deliberate exception: it first matches content while ignoring page, then
scores whether the predicted page label is correct.

Matching establishes which reference and prediction describe the same element;
it does not establish that the predicted type, hierarchy, page, or order is
correct. Layout attribute matching does not use type, hierarchy, or order
correctness to optimize those attributes' own scores. Table/formula detection
still compares elements in their respective declared object family.
Their correctness is scored afterward over fixed eligible reference elements,
including zero for misses. Invalid reference boxes are data errors; missing or
invalid predicted boxes cannot create a visual match. No reference annotation
for an attribute means an inapplicable task, not a candidate-specific exemption.
These scores share omission penalties with detection and are complementary,
not independent measurements. Do not publish both matched-only and
reference-denominator versions as additional headline metrics.

### Layout Element Precision

**Question:** How many predicted document elements are real?

One-to-one matched elements count as correct; unmatched predicted elements count
as extra.

In plain language:

```text
correctly detected document elements
────────────────────────────────────
      all predicted elements
```

It reveals whether the parser invented blocks or split content into elements
that do not exist in the reference.

**Example:** If eight of nine predicted elements match the reference, precision
is `8 / 9 = 0.89`.

**Range and direction:** `[0, 1]`; higher is better.

### Layout Element Recall

**Question:** How many required document elements were detected?

One-to-one matched elements count as correct; unmatched reference elements count
as missed.

In plain language:

```text
correctly detected document elements
────────────────────────────────────
      all reference elements
```

It reveals whether headings, paragraphs, lists, captions, figures, or code
blocks were missed.

**Example:** If eight of ten reference elements are matched, recall is
`8 / 10 = 0.80`.

**Range and direction:** `[0, 1]`; higher is better.

### Layout Element F1

**Question:** Does layout detection balance false and missed elements?

Layout Element F1 combines Layout Element Precision and Recall calculated from
that document's matched, extra, and missed element counts. Calculate document
F1 first, then macro-average the document scores; do not pool counts across
documents.

In plain language:

```text
2 × Layout Precision × Layout Recall
────────────────────────────────────
    Layout Precision + Layout Recall
```

**Example:** Assume the reference has ten layout elements. The parser predicts
nine elements: eight match and one is extra.

```text
Layout Precision = 8 / 9  = 0.89
Layout Recall    = 8 / 10 = 0.80
Layout F1        ≈ 0.84
```

**Range and direction:** `[0, 1]`; higher is better. It is zero when precision
and recall are both zero.

### Element Type Recall

**Question:** How many expected elements were recovered with the correct type?

Every eligible reference element receives credit only when it is matched and
its predicted type equals its annotated type. Missing elements and wrong or
missing predicted types receive zero. Reference elements without a type
annotation are not in the denominator.

In plain language:

```text
matched elements assigned the correct type
──────────────────────────────────────────
  reference elements with type annotations
```

**Example:** Ten reference elements have type annotations. Eight are detected
and five have the right type. Layout Recall is `8 / 10 = 0.80`; Element Type
Recall is `5 / 10 = 0.50`. A detected heading labelled as a paragraph receives
no type credit. Predicting no elements receives zero, not an undefined score.

**Range and direction:** `[0, 1]`; higher is better. No type-annotated reference
elements makes the task inapplicable. An applicable failed first attempt is
null/unavailable.

### Hierarchy Preservation Rate

**Question:** How many expected annotated elements retained their required
parent and level?

Every reference element with explicit hierarchy annotations is in the
denominator. It receives credit only when recovered and every claimed parent
relationship and hierarchy level is correct. Compare parent identities through
the reference-to-prediction match map, not incidental raw IDs. A required parent
that is not recovered cannot establish a correct relationship. An explicitly
annotated root requires an explicit no-parent value in the prediction; an omitted
parent annotation is not a verified root. Missing reference attributes are
unclaimed; missing required prediction attributes or a missing element earn zero.

In plain language:

```text
recovered elements with correct claimed hierarchy
─────────────────────────────────────────────────
      reference elements with hierarchy labels
```

It checks relationships such as subsection-to-section attachment, list nesting,
and heading level.

**Example:** A detected level-two heading incorrectly promoted to level one
fails Hierarchy Preservation Rate even though the heading itself was found.
If three of ten hierarchy-annotated reference elements are recovered with all
claimed relationships correct, the rate is `3 / 10 = 0.30`.

**Range and direction:** `[0, 1]`; higher is better. No hierarchy-annotated
reference elements makes the task inapplicable. Missing all eligible elements
receives zero. An applicable failed first attempt is null/unavailable.

### Mean Bounding-Box Intersection over Union (IoU)

**Question:** How accurately were expected elements localized, including missed
elements?

In plain language, IoU for one matched element is:

```text
area shared by reference and predicted boxes
───────────────────────────────────────────
 area covered by either of the two boxes
```

For each reference element with a valid box, use its admissible one-to-one
matched IoU or zero when it has no qualifying match. Average these contributions
over the eligible reference elements. Missing or invalid predicted boxes earn
zero; zero-area, non-finite, or otherwise invalid reference boxes are rejected
before evaluation. This is not the matched-only mean IoU.

**Example:**

```text
IoU = 1.0  → identical boxes
IoU = 0.8  → boxes mostly agree
IoU = 0.0  → boxes do not overlap
```

**Example:** Ten reference elements have boxes. Two are matched at IoUs `0.8`
and `0.6`; the other eight receive zero. Mean Bounding-Box IoU is
`(0.8 + 0.6) / 10 = 0.14`, not the matched-only mean of `0.70`.

This combines recovery and localization. With identical eligibility it equals
box-detection recall multiplied by the matched-only mean IoU. It is related to,
but not interchangeable with, thresholded Layout Recall.

**Range and direction:** `[0, 1]`; higher is better. No valid reference boxes
makes the task inapplicable, including native DOCX without fixed visual geometry.
A nonempty box reference with no matches receives zero. An applicable failed
first attempt is null/unavailable.

## Tables

Table metrics are calculated only for samples with table annotations. Detection
uses one-to-one table-region matching at `IoU >= 0.5` when reference boxes are
provided. Missing predicted boxes do not trigger text fallback. For references
without boxes, one-to-one matching uses table Content F1 with the fixed `0.5`
threshold. When a reference table has empty projected cell
text, use exact equality of its canonical table structure as the identity
fallback: ordered table-section/row/cell tags and row/column spans, excluding
cell text and presentation-only attributes. Equality gives match similarity
one; otherwise there is no admissible fallback match. This can match a blank
reference table to a same-structure prediction containing unsupported cell text,
which then receives zero Content scores rather than false reconstruction credit.
The fallback does not run TEDS-S to optimize table-quality scores. Missing or
invalid predicted structure cannot establish a box-free blank-table match.
Results are document-macro averages, also reported by document group.

Table evaluation answers three separate questions: was the table found, was
its text recovered, and was its structure reconstructed?

**Primary metrics:** Table Detection F1, Table Content F1, and TEDS.

- **Table Detection F1** is primary because it summarizes whether tables were
  found without producing false table regions.
- **Table Content F1** is primary because finding a table does not show whether
  the text inside its cells was recovered correctly.
- **TEDS** is primary because it evaluates the complete table representation:
  both the structure and the text assigned to its cells.
- **Table Detection Precision and Recall** are secondary metrics. They reveal
  whether a Detection F1 result is limited by false table regions or missed
  tables.
- **Table Content Precision and Recall** are secondary metrics. They reveal
  whether Content F1 is limited by unsupported text or missing text.
- **TEDS-S** is secondary. It removes cell-text similarity from TEDS so a low
  TEDS result can be diagnosed as a structural rather than textual failure.

### Table Detection Precision

**Question:** How many predicted tables are real tables?

One-to-one matched table regions count as correct; unmatched predicted tables
count as extra.

In plain language:

```text
correctly detected tables
─────────────────────────
  all predicted tables
```

It reveals whether ordinary text or page regions were incorrectly labelled as
tables.

**Example:** If eight of ten predicted table regions match reference tables,
precision is `8 / 10 = 0.80`.

**Range and direction:** `[0, 1]`; higher is better.

### Table Detection Recall

**Question:** How many reference tables were found?

One-to-one matched table regions count as correct; unmatched reference tables
count as missed.

In plain language:

```text
correctly detected tables
─────────────────────────
  all reference tables
```

It reveals how many real tables were found.

**Example:** If eight of nine reference tables are detected, recall is
`8 / 9 = 0.89`.

**Range and direction:** `[0, 1]`; higher is better.

### Table Detection F1

**Question:** Does table detection balance extra and missed tables?

Table Detection F1 combines Table Detection Precision and Recall calculated
from each document's matched, extra, and missed table-region counts. Average
the document F1 values equally; do not pool counts across documents.

In plain language:

```text
2 × Table Detection Precision × Table Detection Recall
──────────────────────────────────────────────────────
     Table Detection Precision + Table Detection Recall
```

It summarizes whether the parser finds tables without inventing additional
ones.

**Example:** With precision `0.80` and recall `0.89`, Table Detection F1 is
approximately `0.84`.

**Range and direction:** `[0, 1]`; higher is better. It is zero when precision
and recall are both zero.

### Table Content Precision, Recall, and F1

**Question:** How much extracted table text is supported, how much reference
table text was recovered, and does the result balance both?

Each reference table is paired with its matched prediction. Repeated projected
cell-text token occurrences are counted. A missed table receives zero for all
three Content scores before any empty-text convention is applied. For a
recovered blank-cell table, valid empty/empty cell text receives one; one-sided
empty cell text receives zero. With no reference tables these metrics are
inapplicable, while table detection still evaluates verified negatives.

In plain language:

```text
Table Content Precision =
matched predicted table-text units
----------------------------------
 all predicted table-text units

Table Content Recall =
matched reference table-text units
----------------------------------
 all reference table-text units

Table Content F1 =
the balance between Table Content Precision and Table Content Recall

Document result =
sum of reference-table Content F1 scores
────────────────────────────────────────
  number of reference tables in that document
```

Repeat that within-document averaging for Precision and Recall, then
macro-average eligible documents for each metric. These scores evaluate the
content of expected tables; they do not include unmatched extra predicted
tables. Table Detection Precision exposes those false tables instead.
Precision exposes unsupported cell text in reference-paired tables, recall
exposes missing cell text, and F1 summarizes their balance.

**Example:** If a table prediction contains ten text units, eight match, and the
reference contains twelve units, precision is `8 / 10 = 0.80`, recall is
`8 / 12 = 0.67`, and F1 is approximately `0.73`. For per-table F1 values
`1.00`, `0.80`, and `0.00` for a missed table in one document, that document's
Table Content F1 is `0.60`.

**Range and direction:** All three values lie in `[0, 1]`; higher is better.

### TEDS

**Question:** How similar is the complete predicted table tree, including its
structure and cell text, to the reference?

TEDS represents the HTML table as a tree and converts the normalized edit
distance between the reference and prediction into a similarity score. Unlike
TEDS-S, edits to cell text affect the result.

```text
TEDS = 1.0       → identical structure and cell text
TEDS near 1.0    → small table-tree or cell-text differences
TEDS near 0.0    → severely incorrect complete table
```

**Example:** A table with the right words but the wrong column assignments loses
TEDS credit because the complete tree is wrong. A structurally perfect table
with incorrect cell values also loses credit.

TEDS is primary because it supplies one established end-to-end table score.
The separate Table Content scores and TEDS-S show whether a TEDS loss came
from missing or extra cell text, structure, or both. Average reference-table
TEDS scores within each document, including zero for misses, then macro-average
eligible documents. Unmatched extra tables are handled by detection precision.

**Range and direction:** `[0, 1]`; higher is better. A missed table receives
zero, including a real table with blank cells. No reference table makes the task
inapplicable. A known invalid predicted tree receives zero; an evaluator failure
is unavailable, not zero. Applicable failed first attempts are null/unavailable.

### TEDS-S

**Question:** Was the table's row/cell structure and its merged-cell spans
reconstructed correctly, ignoring cell text?

In plain language:

```text
TEDS-S = 1 − normalized table-tree edit distance
```

The evaluator compares the canonical HTML table trees, including row/cell tags
and row/column spans. It measures how many tree edits separate them, normalizes
that distance for table size, converts it to similarity, and then averages
reference-table scores within each document before averaging eligible documents
equally. Cell text is ignored, so this score focuses on structure rather than
header wording or cell-value placement:

```text
TEDS-S = 1.0       → identical structure
TEDS-S near 1.0    → small structural differences
TEDS-S near 0.0    → severely incorrect structure
```

**Example:** A parser may recover every table word but merge two separate cells
into one. Table Content F1 can remain high while TEDS-S exposes the incorrect
structure. Swapping words between otherwise unchanged cells does not lower
TEDS-S; TEDS detects that cell-content error. Conversely, high TEDS-S with low
Table Content Recall means the shape is right but text is missing.

**Range and direction:** `[0, 1]`; higher is better. A missed table receives
zero, including a real table with blank cells. No reference table makes the task
inapplicable. A known invalid predicted tree receives zero; an evaluator failure
is unavailable. Applicable failed first attempts are null/unavailable.

The benchmark uses the pinned official scorer rather than a custom
approximation. OmniDocBench documents TEDS and TEDS-S in its
[official evaluation repository](https://github.com/opendatalab/OmniDocBench).

## Mathematical formulas

Formula metrics are calculated only for samples with formula annotations and are
reported as a total and by document group. Detection uses one-to-one region
matching at `IoU >= 0.5` when reference boxes are provided; missing predicted
boxes do not trigger text fallback. For references without boxes, one-to-one
matching uses symbol-preserving formula Content F1 with a
`0.5` threshold. The reference conversion freezes the case-sensitive LaTeX or
symbol unitization; punctuation removal must not turn different expressions
into empty-string matches. Empty claimed reference expressions are invalid data.

**Primary metrics:** Formula Detection F1 and Formula Exact Match
(ExpRate@CDM).

- **Formula Detection F1** is primary because it summarizes whether formula
  regions were found without hallucinating extra formula regions.
- **ExpRate@CDM** is primary because mathematical meaning can change after one
  wrong symbol. It reports the proportion of reference formulas reconstructed
  perfectly under the CDM evaluator.
- **Formula Detection Precision and Recall** are secondary metrics. They expose
  whether Detection F1 is limited by false formula regions or missed formulas.
- **Formula Recognition Similarity (CDM)** is supporting because it shows how
  close imperfect recognitions are to the reference. It is valuable for error
  analysis, but a high average similarity can hide formulas with small,
  meaning-changing mistakes; ExpRate@CDM makes perfect reconstruction explicit.

### Formula Detection Precision

**Question:** How many predicted formula regions are real formulas?

Matched formula regions count as correct; unmatched predicted formula regions
count as extra.

In plain language:

```text
correctly detected formula regions
──────────────────────────────────
    all predicted formula regions
```

It reveals how often ordinary text or symbols were incorrectly labelled as
mathematical formulas.

**Example:** If nine of ten predicted regions match real formulas, precision is
`9 / 10 = 0.90`.

**Range and direction:** `[0, 1]`; higher is better.

### Formula Detection Recall

**Question:** How many reference formulas were found?

Matched formula regions count as correct; unmatched reference formula regions
count as missed.

In plain language:

```text
correctly detected formula regions
──────────────────────────────────
    all reference formula regions
```

It reveals how many real formulas were found.

**Example:** If nine of twelve reference formulas are detected, recall is
`9 / 12 = 0.75`.

**Range and direction:** `[0, 1]`; higher is better.

### Formula Detection F1

**Question:** Does formula detection balance extra and missed formulas?

Formula Detection F1 combines Formula Detection Precision and Recall calculated
from each document's matched, extra, and missed formula-region counts. Average
the document F1 values equally; do not pool counts across documents.

In plain language:

```text
2 × Formula Detection Precision × Formula Detection Recall
──────────────────────────────────────────────────────────
       Formula Detection Precision + Formula Detection Recall
```

It summarizes formula detection without hiding whether errors were false
detections or missed formulas.

**Example:** With precision `0.90` and recall `0.75`, Formula Detection F1 is
approximately `0.82`.

**Range and direction:** `[0, 1]`; higher is better. It is zero when precision
and recall are both zero.

### Formula Recognition Similarity (CDM)

**Question:** How visually and structurally close is each recognized formula to
its reference?

The official evaluator renders each reference and predicted formula, then
matches their character regions and positions. It compares matched characters
against extra predicted and missed reference characters. The reported score is
first averaged over reference formulas within each document, including zero for
misses, and then macro-averaged across eligible documents.

In plain language:

```text
                  2 × matched characters
CDM = ──────────────────────────────────────────────
      2 × matched + extra + missed character regions
```

```text
CDM = 1.0  → perfect rendered-character match
CDM = 0.8  → mostly correct formula
CDM = 0.0  → no useful match
```

**Example:** CDM scores `1.0`, `0.9`, `1.0`, and `0.7` in one document produce
that document's Mean CDM `0.90`. A missed formula receives zero. Macro-average
the eligible document values rather than pooling formulas across the corpus.

CDM compares rendered character positions, so equivalent renderings are not
penalized merely for using different LaTeX source.

**Range and direction:** `[0, 1]`; higher is better.

### Formula Exact Match (ExpRate@CDM)

**Question:** How often was a formula reconstructed perfectly?

In plain language:

```text
reference formulas in the document whose CDM score equals 1
───────────────────────────────────────────────────────────
               all reference formulas in the document
```

**Example:**

```text
CDM scores: 1.0, 0.9, 1.0, 0.7

Mean CDM    = (1.0 + 0.9 + 1.0 + 0.7) / 4 = 0.90
ExpRate@CDM = 2 / 4 = 0.50
```

The formulas are generally close, but only half are completely correct. Mean
CDM and ExpRate@CDM therefore answer different questions. Macro-average the
document exact-match fractions equally; a missed formula contributes zero.
With no reference formulas, both recognition tasks are inapplicable. A missing
or known invalid predicted formula receives zero; an official evaluator failure
is unavailable. Applicable failed first attempts are null/unavailable. An
empty or invalid claimed reference formula must be repaired before evaluation.

**Range and direction:** `[0, 1]`; higher is better.

The evaluator and its exact revision are pinned from the [official OmniDocBench evaluation
code](https://github.com/opendatalab/OmniDocBench); EduMind does not substitute a
home-grown LaTeX edit score. The CDM and ExpRate@CDM definitions come from the
[CVPR 2025 CDM paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Image_Over_Text_Transforming_Formula_Recognition_Evaluation_with_Character_Detection_CVPR_2025_paper.html).

## Reliability and failure behavior

**Primary metrics:** None.

Reliability metrics remain required where measured, but none summarizes
extraction quality. Unexpected empty outputs, duplication, successful-output
repeatability, and attempt failures answer different questions. Quality and
duplication observations use the first measured output; repeatability and
Attempt Failure Rate use all three scheduled measured attempts. They do not
automatically select or disqualify a candidate.

### Unexpected Empty Output Rate

**Question:** How often does extraction complete empty when the source requires
usable content?

In plain language:

```text
documents whose valid first output is unexpectedly empty
────────────────────────────────────────────────────────
                  all scheduled documents
```

Determine emptiness from usable output units, not raw string length: projected
prose tokens, symbol-preserving code/formulas, and valid annotated structures.
Correct empty output on a verified blank source is not unexpected. Prose
normalization does not erase a table, formula, or other structural object.
Fatal attempt errors are not completed empty results; Attempt Failure Rate
reports them separately.
If the benchmark cannot account for all scheduled first attempts, the rate is
incomplete rather than silently reducing its denominator.

**Example:** Among 100 documents, two valid first outputs are empty despite
required content, and three others are correctly empty on blank sources:

```text
Unexpected Empty Output Rate = 2 / 100 = 0.02
```

**Range and direction:** `[0, 1]`; lower is better. Failures are not unexpected
empty successes; read this rate together with Attempt Failure Rate and quality.

### Duplicate Content Rate

**Question:** How much repeated token content did the parser add without source
support?

In plain language:

```text
unsupported repeated content units
──────────────────────────────────
       predicted content units
```

The units are the projected whitespace-separated occurrences used by prose
Content F1. For each token occurring at least twice in the prediction, count
its positive excess over the reference occurrence count. For example, four
predicted copies versus three reference copies contribute one; two predicted
copies of a token absent from the reference contribute two. A token occurring
only once contributes none to this diagnostic. This is a repeated-excess-token
diagnostic, not proof of which paragraph or structure was duplicated.
Legitimate repetitions supported by the reference do not count, nor does one
isolated unsupported token.

**Example:** If 20 of 1,000 predicted units are unsupported repetitions, the
rate is `20 / 1,000 = 0.02`.

**Range and direction:** `[0, 1]`; lower is better. Valid empty projected
prediction receives zero: there is no repeated content. Quality and Unexpected
Empty Output Rate distinguish correct emptiness from missing content. A failed
first attempt has unavailable duplication diagnostics, with its failure and
completed-output coverage reported; it is not a measured zero-duplication output.

### Repeatability Success Rate

**Question:** How often do two scheduled attempts both deliver the same valid
structured output?

Development, validation, and locked evaluate all documents in their own split
three times with the same settings and inference seed. Compare pairs `(1,2)`,
`(1,3)`, and `(2,3)`. A pair succeeds only when both attempts completed with valid
canonical outputs and those outputs agree. An error is never a valid agreeing
output, even when two error messages are identical.

The canonical comparison includes text, page attribution, element types and
order, hierarchy, tables, formulas, and normalized boxes. Exclude execution
timestamps, timing, and incidental run/element IDs; canonicalized element IDs
must retain parent relationships and ordering. Coordinate units and numeric
comparison precision are frozen and recorded, not selected after observing
differences. Each attempt must execute extraction rather than return a cached
previous result. Model and runtime-buffer reuse is expected.

In plain language, for one document:

```text
successful identical-output attempt pairs
────────────────────────────────────────
       three scheduled attempt pairs
```

**Examples:** `A`, `B`, and `C` are distinct valid canonical outputs. Reordering
the attempts does not change repeatability, although quality still uses attempt 1.

| Attempts | Successful identical pairs | Document repeatability |
|---|---:|---:|
| `A, A, A` | 3 | 1 |
| `A, A, B` | 1 | 1/3 |
| `A, B, C` | 0 | 0 |
| `A, A, failure` | 1 | 1/3 |
| `A, B, failure` | 0 | 0 |
| `A, failure, failure` | 0 | 0 |
| `failure, failure, failure` | 0 | 0 |

Macro-average the document values. Three identical incorrect or empty outputs
can receive one; quality judges their correctness. This combines successful
delivery and agreement, not pure mathematical determinism. Attempt Failure Rate
separates execution errors from differing valid outputs.

**Range and direction:** `[0, 1]`; higher is better. Every complete three-attempt
record has a numeric score, including zero for all failures. One scheduled
attempt, as in smoke, does not measure repeatability. Interrupted execution with
missing required attempts is incomplete, not imputed as all failures. Retain the
null and reason in artifacts; neither case silently contributes a numeric score.

### Attempt Failure Rate

**Question:** How often does a measured extraction attempt fail to deliver a
valid output?

For each document:

```text
failed measured extraction attempts
───────────────────────────────────
   scheduled measured attempts
```

A crash, request timeout, or unusable complete output schema is an attempt
failure. A completed valid empty output is not an execution failure. Recoverable
malformed blocks remain warnings in an otherwise valid output and affect quality
through missing or incorrect content; they do not turn a whole attempt into a
failure. An evaluator failure is separate from extraction-attempt failure.

**Example:** For `A, A, failure`, Attempt Failure Rate is `1 / 3`, Repeatability
Success Rate is `1 / 3`, and quality comes only from the first `A`. For
`failure, A, A`, the same reliability rates apply but first-attempt quality
is null/unavailable. Macro-average document failure
fractions; with three attempts for every document this also equals total failed
attempts divided by total scheduled attempts.

**Range and direction:** `[0, 1]`; lower is better. Attempt all scheduled
repetitions after recoverable attempt failures and preserve a row for every
attempt in `timings.parquet`. If a process or infrastructure failure prevents
the remaining attempts, mark them not executed and the measurement incomplete;
do not call unexecuted requests observed failures. Loading/warmup failures are
setup failures, not invented failures for every unexecuted document. Every
first-attempt failure remains visible independently of later outcomes.

## Operational performance

**Primary metrics:** None.

Operational metrics describe different resource and latency tradeoffs rather
than one universal notion of quality. Cold loading, first-item latency, warm
latency, whole-document latency, throughput, RAM, VRAM, and temporary disk are
reported individually. The important constraint depends on the intended
deployment; for example, an interactive application may emphasize warm p95
latency while batch ingestion may emphasize pages per minute.

### Cold Model-Load Time

**Question:** How long does the fresh worker take to initialize the candidate
before warmup?

Time model/runtime construction, explicit pipeline initialization, and device
placement in a fresh worker; synchronize before ending the timer. Required lazy
model preparation must not be silently shifted into the warmup. The model is
absent from the process and device at the start, but the operating system's disk
cache is not forcibly cleared. Run one complete warmup only after loading ends.

**Example:** If initialization starts at `0.4 s` and the candidate is ready at
`5.4 s`, Cold Model-Load Time is `5.0 s`.

**Range and direction:** Non-negative seconds; lower is better. One observed
load has no confidence interval. Missing timing or incomplete loading is
unavailable, not a completed zero-second load; retain elapsed time to failure.

### First-item Latency

**Question:** What initialization cost does the first request experience?

In plain language:

```text
first extraction completion time − fresh process start time
```

It includes worker startup, model initialization, and the first complete
representative request. That first request is the warmup in the measured-worker
lifecycle; retaining its first-item completion time does not put its output or
latency into quality or steady-state aggregates. This is distinct from Cold
Model-Load Time, which ends before that request begins.

**Example:** If the process starts at `0.0 s` and the first result completes at
`8.4 s`, First-item Latency is `8.4 s`.

**Range and direction:** Non-negative seconds; lower is better. If the first
request fails, successful first-item latency is unavailable; retain the elapsed
time and failure separately. A single observation has no confidence interval.

### p50 and p95 Warm Latency per Page

**Question:** What are the typical and slow-tail steady-state page speeds?

Use successful measured request durations after the one warmup. For each
document, take the median successful attempt duration and normalize it by the
verified number of physical input pages, not the number of pages emitted by the
parser. In plain language:

```text
median successful warm document latency
──────────────────────────────
  verified physical input pages
```

Each observation is a document-normalized page speed, not the latency of an
individually timed page. The benchmark then reports:

- p50: the median, representing typical performance;
- p95: the estimated 95th percentile of those document-normalized observations,
  using the frozen quantile method.

**Example:** A p95 of `2.4 seconds/page` describes the slow tail of the normalized
document speeds. It does not mean every individual page was timed separately or
that exactly 95% of a small observed sample falls below an interpolated bound.

**Range and direction:** Non-negative seconds/page; lower is better. Native
DOCX without a fixed renderer has no physical-page latency. Missing or invalid
input page counts make the normalization unavailable or invalid-reference,
not division by zero. A document with no successful measured attempts has
unavailable successful-completion latency. Record successful and failed attempt
counts; do not substitute fast failure times as completed-page timings.

### Complete Document Latency

**Question:** How long does a user wait for a complete source?

This is the warm end-to-end request time for one whole source, including
preprocessing, extraction, postprocessing, validation, and synchronization but
excluding model loading, warmup, and official scoring. Take each document's
median successful measured attempt as its observation, then calculate p50/p95
across documents. It is retained alongside per-page latency because a large PDF
can have reasonable page speed but still require a long total wait.

**Example:** For document times `2`, `3`, `4`, `5`, and `11` seconds, the median
is `4 seconds`; the slow `11-second` document influences the upper tail. The
benchmark reports the exact p95 using its fixed quantile implementation.

**Range and direction:** Non-negative seconds/document; lower is better. Results
are also reported by document group. With no successful completions the latency
is unavailable, not zero. Artifacts retain all attempt durations, including
time-to-failure, and successful-completion coverage; a partial successful-latency
aggregate must be read with Attempt Failure Rate.

### Batch Pages per Minute

**Question:** What sustained batch capacity does the parser provide?

In plain language:

```text
60 × successfully processed pages
─────────────────────────────────
       batch duration in seconds
```

It measures sustained extraction capacity, not the latency of one request.
Count physical input pages from successfully completed measured attempts, not
predicted page records or correctly recognized tokens. The measured batch
interval includes failed-attempt time and dispatch overhead but excludes model
loading, warmup, and offline scoring. A valid empty extraction can be an
execution success even when its quality is zero; throughput must be interpreted
with quality and failure results. Page throughput is inapplicable to native
DOCX without fixed physical pagination.

**Example:** Processing 120 pages in 180 seconds gives
`60 × 120 / 180 = 40 pages/minute`.

**Range and direction:** Non-negative pages/minute; higher is better. Zero
completed pages with positive measured batch duration gives zero. No measured
batch, missing timing, or non-positive duration is unavailable, never division
by zero or an invented throughput.

### Peak Process-Tree RAM

**Question:** How much system memory does the complete extractor require?

At every sampling point, add the resident RAM of the benchmark candidate and
all extractor child processes. The largest observed total is reported.

**Example:** If sampled process-tree totals are `1,200`, `2,450`, and `2,100`
MiB, Peak Process-Tree RAM is `2,450 MiB`.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

### Peak Device VRAM

**Question:** How much total memory is occupied on the assigned GPU during extraction?

Report the largest raw NVML device-memory sample under the shared CUDA resource
contract, including loading, warmup, and measured extraction.

**Example:** Device-memory samples of `700`, `1,800`, and `1,500` MiB produce Peak
Device VRAM `1,800 MiB`; an idle baseline is not subtracted.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
CPU execution records null/inapplicable; missing required CUDA telemetry records
null/unavailable. Retain measured samples after failure with their lifecycle scope.

### Peak Temporary Disk

**Question:** How much additional working-disk space does extraction require?

In plain language:

```text
largest temporary-file footprint during extraction
− temporary-file footprint before extraction
```

It shows the maximum extra working-disk space required by the profile.

**Example:** If temporary storage starts at `20 MiB` and reaches `370 MiB`, Peak
Temporary Disk is `350 MiB`.

**Range and direction:** Non-negative MiB; lower is better.

p50/p95 latency intervals are reported only when enough independent source
documents support them; pages and attempts do not increase that count. A single
first-item measurement, throughput batch, or peak RAM/VRAM/temporary-disk
observation is reported without a confidence interval. If an operational
measurement is repeated independently,
an interval may be reported only with the repetition count and aggregation unit.
Missing RAM, VRAM, or disk instrumentation is recorded as unavailable, never as
zero.

## Worked candidate interpretation

Suppose one extraction profile produces:

```text
Content Precision           = 0.98
Content Recall              = 0.75
Reading Order NED           = 0.05
Page Coverage               = 1.00
Table Detection Recall      = 1.00
Table Content F1            = 0.90
TEDS                        = 0.55
TEDS-S                      = 0.60
Repeatability Success Rate  = 0.95
Attempt Failure Rate        = 0.00
p95 warm latency/page       = 3.2 seconds
```

These quality values are document-macro averages. This means:

- document-level text precision is high on average;
- document-level text recovery is incomplete on average (`Content Recall = 0.75`),
  not a claim that exactly 25% of all corpus tokens were missed;
- the complete element sequence requires relatively few edits to match the reference;
- every content-bearing reference page produced some matching content;
- every reference table was detected;
- table text was mostly recovered, but the low TEDS and TEDS-S show that the
  complete table and its structure were reconstructed poorly despite successful
  detection;
- 95% of scheduled attempt pairs delivered identical valid structured outputs;
- no measured extraction attempt failed; and
- the estimated 95th percentile of document-normalized warm page speeds is
  3.2 seconds/page.

No single number communicates all of these facts. That is why the benchmark
reports the metrics separately instead of calculating a weighted overall score.

## Document-extraction confidence intervals

A confidence interval expresses uncertainty in an aggregate calculated from
multiple independent observations. It does not describe the possible range of
an individual prediction, and it must not be attached to a value merely because
the value is reported.

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Development, validation, and locked text, page, layout, table, and formula aggregates | Yes | Calculated from the contributing independent samples. |
| Development, validation, and locked reliability rates | Yes | Calculated across scheduled independent samples. |
| Development, validation, and locked p50/p95 latency | Conditional | Reported when enough independent source documents support the percentile estimate; pages and attempts remain dependent observations. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| Statuses, revisions, checksums, configuration values | No | These are states or fixed facts rather than sampled estimates. |
| One first-item or cold-load measurement | No | One observation cannot estimate uncertainty. |
| One throughput batch | No | Report the observed batch throughput. |
| One peak RAM, VRAM, or temporary-disk measurement | No | Report the observed peak. |
| Repeated independent operational measurements | Conditional | An interval is allowed only when the repetition count and aggregation unit are recorded. |

When an interval is required but the available independent observations are too
few to support it, the report keeps the point estimate, omits the interval, and
marks the metric as descriptive rather than authoritative selection evidence.
EduMind does not invent a zero-width interval.

### Calculation

Eligible development, validation, and locked sample-based metrics use 10,000 bootstrap resamples with
seed 42:

1. Treat each source document as the independent resampling unit.
2. Resample documents with replacement.
3. Recalculate the aggregate for every resample.
4. Use the 2.5th and 97.5th percentiles as the 95% interval bounds.

Document-group metrics resample only the samples in that group. Conditional
tasks such as table reconstruction or formula recognition draw from reference-fixed
eligible source records, including failed or unavailable first outputs. Do not
prefilter to successful outputs before drawing. Within each draw, aggregate only
defined numeric values and retain contributing counts and undefined-draw reasons. Verified-negative detection documents already have defined
per-document scores under the empty convention; they are not dropped from
bootstrap draws. Preserve all dependent pages, objects, capture variants, and
attempts within their declared independent source unit.

Confidence intervals are calculated after execution from saved document-level
values, not during an inference attempt. Quality intervals use first-attempt
quality values; repeatability intervals use the document's three-attempt pair
score; Attempt Failure Rate intervals use its failed-attempt fraction. With
100 independent documents and three attempts each there are still 100 independent
units, not 300 attempts or pairs. Resampling 10,000 times performs arithmetic on
saved results and does not launch 10,000 model inferences.

No eligible independent documents means no aggregate or interval. Below the
frozen support minimum, keep the point estimate and null bounds with the reason.
Where dependent capture variants share one source, the independent source count
can be smaller than the eligible input-document count; report both. If every
observed document value is identical, percentile bootstrap can return equal
bounds, such as `[1, 1]`; retain the calculated bounds with a degenerate-interval
flag. That result is not proof of certainty on future documents. Missing required
evaluations do not become a normal authoritative aggregate by dropping rows.
Retain defined/undefined draw counts whenever an aggregate genuinely cannot be
calculated in a draw.

### Interpretation

For example:

```text
Content F1 = 0.91
95% CI     = [0.89, 0.93]
```

The benchmark estimates an aggregate Content F1 of `0.91`; variation across the
sampled documents produces the reported uncertainty interval. A narrower
interval means the aggregate is estimated more precisely.
