# Benchmark metric reference

[Benchmark overview](overview.md) · [Experiment sequence and rationale](methodology.md) ·
[Benchmark runbook](running.md)

This page defines EduMind's approved benchmark metric contracts. The
[methodology](methodology.md) says **where** each metric is used and why; this
page says **what the value means and how it is calculated**. A development or validation
result is authoritative only when its runner implements the applicable contract
exactly and records every required value. Higher is better unless a metric is
marked lower-is-better.

## Shared conventions

- Prose comparison uses one symmetric, evaluation-only projection on both the
  reference and prediction: Unicode NFC, case-folding, replacement of Unicode
  punctuation with spaces, and whitespace collapse. The resulting whitespace-
  separated units are used by prose Content, Exact Match, Token F1, and
  ROUGE-L; prose CER and WER operate on the same projected strings. Raw outputs
  remain unchanged in artifacts. The projection does not dehyphenate words,
  correct spelling, rewrite numbers, remove headers, or alter formulas, code,
  layout trees, or table trees.
- Source and evidence spans are half-open intervals: `[start, end)`.
- Empty denominators use the explicit behavior stated below; they never produce
  fabricated zero-quality observations.
- Development, validation, and locked runs retain one row per sample before aggregation.
- p50, p95, and p99 are latency percentiles. Throughput is completed operations
  divided by measured wall-clock time.
- Eligible development, validation, and locked sample-based aggregates use 10,000
  bootstrap resamples with seed 42 and 95% confidence intervals. Counts,
  statuses, fixed identifiers, and single operational observations do not
  receive intervals.
- Normalized precision, recall, F1, accuracy, coverage, nDCG, and correctness
  values lie in `[0, 1]`. CER and WER are non-negative and can exceed 1 when
  insertions outnumber reference units. Human rubric scores use their stated
  `0–2` or `0–1` scales. Time, memory, storage, and throughput are non-negative
  and have no fixed upper bound.

## Document extraction

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

Each metric now has one self-contained subsection. Its question, equation,
plain-language calculation, example, interpretation, valid range, and preferred
direction are kept together. Category introductions contain only rules shared by
multiple metrics, such as the element-matching protocol.

### Metric summary

The following tables provide the complete metric list for readers who only need
to know what the document-extraction benchmark measures. The detailed contracts,
examples, and confidence-interval rules follow the summary.

#### Text content and recognition

| Metric | Question answered |
|---|---|
| Content Precision | How much extracted content is supported by the reference? |
| Content Recall | How much required content was recovered? |
| Content F1 | Does the extractor balance correct output and complete output? |
| Character Error Rate (CER) | How severe are character-recognition errors? |
| Word Error Rate (WER) | How severe are complete-word errors? |
| Reading Order Accuracy | Is recovered content presented in the correct sequence? |

#### Pages

| Metric | Question answered |
|---|---|
| Page Coverage | Did every expected page produce relevant content? |
| Page Content F1 | Was the correct content recovered within the correct page? |
| Page Attribution Accuracy | Was extracted content assigned to the correct page number? |
| Duplicate Page Rate | How often did the parser repeat a page? |

#### Layout and document structure

| Metric | Question answered |
|---|---|
| Layout Element Precision | How many predicted document elements are real? |
| Layout Element Recall | How many required document elements were detected? |
| Layout Element F1 | Does layout detection balance false and missed elements? |
| Element Type Accuracy | Were matched headings, paragraphs, lists, captions, and other blocks classified correctly? |
| Hierarchy Accuracy | Were heading levels, list nesting, and parent-child relationships preserved? |
| Mean Bounding-Box IoU | Were matched elements localized correctly on the page? |

#### Tables

| Metric | Question answered |
|---|---|
| Table Detection Precision | How many predicted tables are real tables? |
| Table Detection Recall | How many reference tables were found? |
| Table Detection F1 | Does table detection balance extra and missed tables? |
| Table Content Precision | How much extracted table text is supported by the reference? |
| Table Content Recall | How much reference table text was recovered? |
| Table Content F1 | Was the textual content inside tables recovered? |
| TEDS | How similar is the complete predicted table tree, including structure and cell text, to the reference? |
| TEDS-S | Were rows, columns, headers, merged cells, and spans reconstructed correctly when cell text is ignored? |

#### Mathematical formulas

| Metric | Question answered |
|---|---|
| Formula Detection Precision | How many predicted formula regions are real formulas? |
| Formula Detection Recall | How many reference formulas were found? |
| Formula Detection F1 | Does formula detection balance extra and missed formulas? |
| Formula Recognition Similarity (CDM) | How visually and structurally close is each recognized formula to its reference? |
| Formula Exact Match (ExpRate@CDM) | How often was a formula reconstructed perfectly? |

#### Reliability and failure behavior

| Metric | Question answered |
|---|---|
| Empty Output Rate | How often does extraction complete without producing usable content? |
| Duplicate Content Rate | How much substantial content did the parser repeat without source support? |
| Structured-output Determinism | Does the same input produce the same complete structured result? |
| Candidate Failure Rate | How often does the candidate end with a fatal error instead of a scoreable result? |

#### Operational performance

| Metric | Question answered |
|---|---|
| First-item Latency | What initialization cost does the first request experience? |
| p50/p95 Warm Latency per Page | What are the typical and slow-tail steady-state page speeds? |
| Complete Document Latency | How long does a user wait for a complete source? |
| Batch Pages per Minute | What sustained batch capacity does the parser provide? |
| Peak Process-Tree RAM | How much system memory does the complete extractor require? |
| Peak VRAM | How much GPU memory does extraction require? |
| Peak Temporary Disk | How much additional working-disk space does extraction require? |

This contract replaces the legacy exact-line, adjacency-table, and raw-LaTeX
scorers. The executable document runner uses these names, applicability rules,
group aggregates, sample counts, and confidence intervals directly.

All candidates receive the same canonical reference and output conversion.
The prose projection defined above handles harmless representation differences;
it does not repair words, remove page content, deduplicate text, or otherwise
clean a candidate's extraction.
Each metric is reported as a total across the documents on which it is defined
and separately for applicable document groups such as `image`, `pdf_scanned`,
and `docx`. A conditional metric is absent, not zero, when its required
annotation does not apply. Qualifying aggregates receive a confidence interval
under the separate policy below.

Text error and content metrics are calculated per eligible document and then
macro-averaged, so a long PDF cannot dominate every short document. Detection
precision, recall, and F1 pool `TP`, `FP`, and `FN` across the annotated samples
in the reported group. Table-content, table-structure, and formula-recognition
scores macro-average their eligible reference objects, assigning zero to a
missed reference object. Bootstrap resampling always uses the document as the
independent unit and recalculates the complete aggregate from that resample.

Eligibility comes from each authoritative reference's explicit
`reference_capabilities` list: `text`, `pages`, `reading_order`,
`layout_boxes`, `element_types`, `hierarchy`, `tables`, and `formulas`. Smoke
fixtures may infer this list from inline annotations. A missing capability omits
the metric; it never fabricates a zero. Table/formula capabilities use explicit
`has_table:false` and `has_formula:false` negatives in detection counts.

### Prose scoring projection

Before prose text is compared, both sides undergo the same four representation
steps: Unicode NFC, case-folding, punctuation replacement with spaces, and
whitespace collapse. This prevents harmless typography from deciding a score.

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

### Text content and recognition

Repeated token occurrences are counted; token sets are not used.

**Primary metrics:** Content F1 and Reading Order Accuracy.

- **Content F1** is primary because it summarizes whether the parser recovered
  the required text without adding unsupported text. It balances content
  precision and recall in one category-level measure.
- **Reading Order Accuracy** is also primary because correct words can still be
  unusable when headings, columns, paragraphs, or list items are returned in the
  wrong sequence. It measures a different outcome from Content F1.
- **Content Precision and Content Recall** are secondary metrics. They separate
  hallucinated or extra content from missing content and therefore explain why
  Content F1 changed.
- **CER and WER** are supporting error diagnostics. They reveal character-level
  and word-level recognition mistakes, but neither alone measures both content
  completeness and unsupported output as directly as Content F1.

#### Content Precision

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

**Range and direction:** `[0, 1]`; higher is better. An empty prediction receives
zero rather than producing an undefined value.

#### Content Recall

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

**Range and direction:** `[0, 1]`; higher is better.

#### Content F1

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

**Range and direction:** `[0, 1]`; higher is better. F1 is zero when precision
and recall are both zero.

#### Character Error Rate (CER)

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
when insertions outnumber reference characters.

#### Word Error Rate (WER)

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
when insertions outnumber reference words.

#### Reading Order Accuracy

**Question:** Is recovered content presented in the correct sequence?

The evaluator considers every pair of one-to-one matched document elements and
checks whether that pair has the same relative order in the reference and
prediction.

In plain language:

```text
correctly ordered matched-element pairs
───────────────────────────────────────
      all comparable matched pairs
```

**Example:**

```text
Reference:  Heading → Paragraph → Caption
Prediction: Paragraph → Heading → Caption

Heading–Paragraph: wrong
Heading–Caption:   correct
Paragraph–Caption: correct

Reading Order Accuracy = 2 / 3 = 0.67
```

This metric is especially important for multi-column pages, where the text can
be recognized correctly but returned in an unusable order.

**Range and direction:** `[0, 1]`; higher is better. Samples with fewer than two
matched elements are ineligible rather than being assigned an artificial zero
or perfect score.

### Pages

**Primary metric:** Page Content F1.

- **Page Content F1** is primary because it requires the parser to recover the
  correct content on the correct page. It captures more than merely producing
  some output for each page.
- **Page Coverage** is supporting because a page can count as covered after only
  a small amount of relevant content is recovered.
- **Page Attribution Accuracy** is supporting because it evaluates page labels
  only for content that was already matched; it does not measure missing or extra
  page content.
- **Duplicate Page Rate** is supporting because it isolates one narrow failure:
  repeating pages that should appear once.

#### Page Coverage

**Question:** Did every expected page produce relevant content?

Each reference content-bearing page counts as covered when it contains at least
one matched content unit.

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

**Range and direction:** `[0, 1]`; higher is better.

#### Page Content F1

**Question:** Was the correct content recovered within the correct page?

Content F1 is calculated separately for every reference or unexpected predicted
page and then averaged. Missing reference pages and unexpected predicted pages
receive zero.

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

#### Page Attribution Accuracy

**Question:** Was extracted content assigned to the correct page number?

For this metric, one-to-one content matching maximizes Content F1 and requires
`Content F1 >= 0.5`; page numbers are deliberately ignored until after the
content pairs are formed. The metric then counts how many matched elements have
the correct page number.

In plain language:

```text
matched elements assigned to the correct page
─────────────────────────────────────────────
    matched elements with page annotations
```

**Example:** An extractor may recover a paragraph correctly but assign it to
page 2 instead of page 3. The paragraph still contributes to text recovery, but
it fails Page Attribution Accuracy.

**Range and direction:** `[0, 1]`; higher is better. It is omitted when there
are no matched elements with page annotations.

#### Duplicate Page Rate

**Question:** How often did the parser repeat a page?

Two page records are treated as near-duplicates when their Content F1 is at
least `0.95`; this fixed threshold is applied to every extraction profile.

In plain language:

```text
unsupported duplicated page records
────────────────────────────────────
       predicted page records
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
duplicate.

**Range and direction:** `[0, 1]`; lower is better. The metric is omitted when
the candidate predicts no pages; Empty Output Rate records that failure.

### Layout and document structure

The layout set contains headings, paragraphs, list items, captions, figures,
code blocks, and other annotated non-table/non-formula elements. Tables and
formulas are evaluated in their own sections.

**Primary metrics:** Layout Element F1, Element Type Accuracy, and Hierarchy
Accuracy.

- **Layout Element F1** is primary because it summarizes whether the expected
  document elements were found without inventing extra elements.
- **Element Type Accuracy** is primary because detecting a region is not enough:
  the parser must distinguish headings, paragraphs, list items, captions, and
  other element types.
- **Hierarchy Accuracy** is primary because parent-child relationships, heading
  levels, and list nesting determine whether the recovered document structure is
  usable. Neither detection nor type classification measures these relations.
- **Layout Element Precision and Recall** are secondary metrics. They distinguish
  extra detected elements from missed elements and explain Layout Element F1.
- **Mean Bounding-Box IoU** is supporting because it diagnoses localization
  quality after elements have been matched. Precise boxes are useful, but they do
  not by themselves prove that the correct elements, types, or hierarchy were
  recovered.

Reference and predicted elements are matched one-to-one. When boxes are
available, matching maximizes bounding-box Intersection over Union (IoU), with
`IoU >= 0.5` required for a match. For native documents without boxes, matching
maximizes element Content F1 and requires `Content F1 >= 0.5`. The input format
therefore determines one explicit matching rule; the runner does not silently
mix the two rules within a document comparison.
Visual layout, table, and formula pairs must also have equal page numbers;
identical boxes on different pages never match. Page Attribution Accuracy is
the deliberate exception: it first matches content while ignoring page, then
scores whether the predicted page label is correct.

#### Layout Element Precision

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

#### Layout Element Recall

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

#### Layout Element F1

**Question:** Does layout detection balance false and missed elements?

Layout Element F1 combines Layout Element Precision and Recall calculated from
the pooled matched, extra, and missed element counts.

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

#### Element Type Accuracy

**Question:** Were headings, paragraphs, lists, captions, and other matched
blocks classified correctly?

Only one-to-one matched elements are considered. Each receives credit when its
predicted type equals its reference type.

In plain language:

```text
matched elements assigned the correct type
──────────────────────────────────────────
             matched elements
```

**Example:** A heading detected at the correct location but labelled as a
paragraph counts as a successful detection and an incorrect type. Detection
metrics and type accuracy therefore answer different questions.

**Range and direction:** `[0, 1]`; higher is better. It is omitted when there
are no matched elements.

#### Hierarchy Accuracy

**Question:** Were heading levels, list nesting, and parent-child relationships
preserved?

Only matched elements with hierarchy annotations are considered. An element
receives credit when both its parent relationship and hierarchy level are
correct.

In plain language:

```text
matched elements with correct parent and level
──────────────────────────────────────────────
    matched elements with hierarchy labels
```

It checks relationships such as subsection-to-section attachment, list nesting,
and heading level.

**Example:** A detected level-two heading incorrectly promoted to level one
fails Hierarchy Accuracy even though the heading itself was found.

**Range and direction:** `[0, 1]`; higher is better. It is omitted when no
matched elements have hierarchy annotations.

#### Mean Bounding-Box Intersection over Union (IoU)

**Question:** Were matched elements localized correctly on the page?

In plain language, IoU for one matched element is:

```text
area shared by reference and predicted boxes
───────────────────────────────────────────
 area covered by either of the two boxes
```

The benchmark averages this value over matched elements that have both
reference and predicted boxes.

**Example:**

```text
IoU = 1.0  → identical boxes
IoU = 0.8  → boxes mostly agree
IoU = 0.0  → boxes do not overlap
```

Layout Recall tells us whether elements were found; Mean IoU tells us how
accurately the found elements were localized.

**Range and direction:** `[0, 1]`; higher is better. It is omitted when there
are no matched elements with both boxes.

The previous exact-line Block F1 is not part of this contract. Exact line equality
is too brittle for layout detection and does not measure element types, geometry,
or hierarchy.

### Tables

Table metrics are calculated only for samples with table annotations. Detection
uses one-to-one table-region matching at `IoU >= 0.5` when boxes are available.
For native documents without boxes, one-to-one matching uses table Content F1
with the same fixed `0.5` threshold used by the executable evaluator. Results
are reported as a total and by document group.

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

#### Table Detection Precision

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

#### Table Detection Recall

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

#### Table Detection F1

**Question:** Does table detection balance extra and missed tables?

Table Detection F1 combines Table Detection Precision and Recall calculated
from the pooled matched, extra, and missed table-region counts.

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

#### Table Content Precision, Recall, and F1

**Question:** How much extracted table text is supported, how much reference
table text was recovered, and does the result balance both?

Each reference table is paired with its matched prediction. A missed table is
paired with an empty prediction. Repeated token occurrences are counted.

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

Aggregate result =
sum of reference-table Content F1 scores
────────────────────────────────────────
         number of reference tables
```

Precision exposes unsupported table text; recall exposes missing table text;
F1 summarizes their balance. A missed reference table receives zero recall and
F1.

**Example:** If a table prediction contains ten text units, eight match, and the
reference contains twelve units, precision is `8 / 10 = 0.80`, recall is
`8 / 12 = 0.67`, and F1 is approximately `0.73`. For per-table F1 values
`1.00`, `0.80`, and `0.00` for a missed table, the aggregate is `0.60`.

**Range and direction:** All three values lie in `[0, 1]`; higher is better.

#### TEDS

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
Table Content Precision/Recall/F1 and TEDS-S remain necessary diagnostics:
they show whether a TEDS loss came from missing or extra text, structure, or
both.

**Range and direction:** `[0, 1]`; higher is better. A missed table receives
zero.

#### TEDS-S

**Question:** Were rows, columns, headers, merged cells, and spans reconstructed
correctly?

In plain language:

```text
TEDS-S = 1 − normalized table-tree edit distance
```

The evaluator represents each reference and prediction as a tree of rows,
columns, headers, cells, and spans. It measures how many tree edits separate
them, normalizes that distance for table size, converts it to similarity, and
then averages the scores across reference tables. Cell text is removed before
comparison, so this score focuses on structure:

```text
TEDS-S = 1.0       → identical structure
TEDS-S near 1.0    → small structural differences
TEDS-S near 0.0    → severely incorrect structure
```

**Example:** A parser may recover every table word but put the words into the
wrong columns. Table Content F1 can remain high while TEDS-S exposes the
structural failure. Conversely, high TEDS-S with low Table Content Recall means
the shape is right but text is missing.

**Range and direction:** `[0, 1]`; higher is better. A missed table receives
zero.

The benchmark uses the pinned official scorer rather than a custom
approximation. OmniDocBench documents TEDS and TEDS-S in its
[official evaluation repository](https://github.com/opendatalab/OmniDocBench).

### Mathematical formulas

Formula metrics are calculated only for samples with formula annotations and are
reported as a total and by document group. Detection uses one-to-one region
matching at `IoU >= 0.5` when boxes are available. For native documents without
boxes, one-to-one matching uses formula Content F1 with a `0.5` threshold.

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

#### Formula Detection Precision

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

#### Formula Detection Recall

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

#### Formula Detection F1

**Question:** Does formula detection balance extra and missed formulas?

Formula Detection F1 combines Formula Detection Precision and Recall calculated
from the pooled matched, extra, and missed formula-region counts.

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

#### Formula Recognition Similarity (CDM)

**Question:** How visually and structurally close is each recognized formula to
its reference?

The official evaluator renders each reference and predicted formula, then
matches their character regions and positions. It compares matched characters
against extra predicted and missed reference characters. The reported score is
the average across reference formulas.

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

**Example:** CDM scores `1.0`, `0.9`, `1.0`, and `0.7` produce Mean CDM `0.90`.
A missed formula receives zero.

CDM compares rendered character positions, so equivalent renderings are not
penalized merely for using different LaTeX source.

**Range and direction:** `[0, 1]`; higher is better.

#### Formula Exact Match (ExpRate@CDM)

**Question:** How often was a formula reconstructed perfectly?

In plain language:

```text
formulas whose CDM score equals 1
────────────────────────────────
      all reference formulas
```

**Example:**

```text
CDM scores: 1.0, 0.9, 1.0, 0.7

Mean CDM    = (1.0 + 0.9 + 1.0 + 0.7) / 4 = 0.90
ExpRate@CDM = 2 / 4 = 0.50
```

The formulas are generally close, but only half are completely correct. Mean
CDM and ExpRate@CDM therefore answer different questions.

**Range and direction:** `[0, 1]`; higher is better.

The evaluator and its exact revision are pinned from the [official OmniDocBench evaluation
code](https://github.com/opendatalab/OmniDocBench); EduMind does not substitute a
home-grown LaTeX edit score. The CDM and ExpRate@CDM definitions come from the
[CVPR 2025 CDM paper](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_Image_Over_Text_Transforming_Formula_Recognition_Evaluation_with_Character_Detection_CVPR_2025_paper.html).

### Reliability and failure behavior

**Primary metrics:** None.

All reliability metrics remain required, but none summarizes extraction quality.
Empty outputs, duplicated content, nondeterministic results, and fatal failures
are different operational failure modes and should not be combined into one
preferred number. They are interpreted individually when judging whether a
candidate is dependable enough to use.

#### Empty Output Rate

**Question:** How often does extraction complete without producing usable
content?

In plain language:

```text
samples producing no usable content
───────────────────────────────────
         all scheduled samples
```

**Example:** If two of 100 documents return empty structured documents:

```text
Empty Output Rate = 2 / 100 = 0.02
```

**Range and direction:** `[0, 1]`; lower is better.

#### Duplicate Content Rate

**Question:** How much substantial content did the parser repeat without source
support?

In plain language:

```text
unsupported repeated content units
──────────────────────────────────
       predicted content units
```

The content units are the same projected whitespace-separated occurrences used
by Content F1. For each unit, predicted occurrences beyond the number supported
by the reference count only when that token was repeated in the prediction. The
metric therefore exposes repeated paragraphs, page content, tables, or formulas
without treating one isolated wrong token as a duplication error.

**Example:** If 20 of 1,000 predicted content units are unsupported repetitions,
Duplicate Content Rate is `20 / 1,000 = 0.02`.

**Range and direction:** `[0, 1]`; lower is better. It is omitted when the
candidate produces no content; Empty Output Rate records that case.

#### Structured-output Determinism

**Question:** Does the same input produce the same complete structured result?

The fingerprint covers text, page attribution, element types and order,
hierarchy, tables, formulas, and normalized boxes; timing and random run IDs are
excluded.

In plain language:

```text
samples with identical output in every repetition
─────────────────────────────────────────────────
            repeatedly tested samples
```

**Example:** If 19 of 20 repeatedly executed documents produce an identical
canonical fingerprint:

```text
Structured-output Determinism = 19 / 20 = 0.95
```

**Range and direction:** `[0, 1]`; higher is better.

#### Candidate Failure Rate

**Question:** How often does the candidate end with a fatal error instead of a
scoreable result?

In plain language:

```text
samples ending in fatal errors
──────────────────────────────
     all scheduled samples
```

**Example:** If one of 100 scheduled documents crashes, Candidate Failure Rate
is `1 / 100 = 0.01`.

An empty output means the extractor completed but returned no usable content. A
candidate failure means it crashed or ended with a fatal error. Keeping both
metrics separates silent extraction failures from execution failures.

**Range and direction:** `[0, 1]`; lower is better.

Every scheduled sample remains visible. A recoverable empty or malformed output
is scored as an empty prediction: it contributes zero to every applicable
recall, F1, coverage, accuracy, or similarity aggregate while precision follows
that metric's documented empty-denominator rule. A fatal candidate error with a
valid per-sample error record follows the same quality treatment and also
increments Candidate Failure Rate. A failure that prevents the required
per-sample record makes the benchmark invocation incomplete and therefore
non-authoritative. Failed difficult documents can therefore never disappear
from the denominator and make a candidate look artificially strong.
Every configured repetition is attempted and written to `timings.parquet`, even
after an earlier attempt fails. If any measured repetition fails, the sample is
scored as empty, Candidate Failure Rate is one, and Structured-output
Determinism is zero. Successful-page throughput still divides by all measured
attempt time, including failed attempts.

### Operational performance

**Primary metrics:** None.

Operational metrics describe different resource and latency tradeoffs rather
than one universal notion of quality. First-item latency, warm latency,
whole-document latency, throughput, RAM, VRAM, and temporary disk are therefore
reported individually. The important constraint depends on the intended
deployment; for example, an interactive application may emphasize warm p95
latency while batch ingestion may emphasize pages per minute.

#### First-item Latency

**Question:** What initialization cost does the first request experience?

In plain language:

```text
first extraction completion time − fresh process start time
```

It includes model initialization and the first extraction, so it represents the
delay experienced by the first request in a fresh process.

**Example:** If the process starts at `0.0 s` and the first result completes at
`8.4 s`, First-item Latency is `8.4 s`.

**Range and direction:** Non-negative seconds; lower is better.

#### p50 and p95 Warm Latency per Page

**Question:** What are the typical and slow-tail steady-state page speeds?

In plain language, first calculate this for every document:

```text
complete warm document latency
──────────────────────────────
       processed pages
```

The benchmark then reports:

- p50: the median, representing typical performance;
- p95: the tail value at or below which 95% of measured per-page observations
  fall.

**Example:** A p95 of `2.4 seconds/page` means 95% of measured warm per-page
observations were no slower than 2.4 seconds.

**Range and direction:** Non-negative seconds/page; lower is better.

#### Complete Document Latency

**Question:** How long does a user wait for a complete source?

This is the end-to-end time for one whole source. Its p50 and p95 answer how long
a user typically waits and how long difficult documents take. It is retained
alongside per-page latency because a large PDF can have reasonable page speed
but still require a long total wait.

**Example:** For document times `2`, `3`, `4`, `5`, and `11` seconds, the median
is `4 seconds`; the slow `11-second` document influences the upper tail. The
benchmark reports the exact p95 using its fixed quantile implementation.

**Range and direction:** Non-negative seconds/document; lower is better. Results
are also reported by document group.

#### Batch Pages per Minute

**Question:** What sustained batch capacity does the parser provide?

In plain language:

```text
60 × successfully processed pages
─────────────────────────────────
       batch duration in seconds
```

It measures sustained extraction capacity, not the latency of one request.

**Example:** Processing 120 pages in 180 seconds gives
`60 × 120 / 180 = 40 pages/minute`.

**Range and direction:** Non-negative pages/minute; higher is better. It is
omitted when no real sustained batch is measured.

#### Peak Process-Tree RAM

**Question:** How much system memory does the complete extractor require?

At every sampling point, add the resident RAM of the benchmark candidate and
all extractor child processes. The largest observed total is reported.

**Example:** If sampled process-tree totals are `1,200`, `2,450`, and `2,100`
MiB, Peak Process-Tree RAM is `2,450 MiB`.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

#### Peak VRAM

**Question:** How much GPU memory does extraction require?

Report the largest GPU-memory allocation attributable to the candidate during
the measured extraction.

**Example:** GPU-memory samples of `700`, `1,800`, and `1,500` MiB produce Peak
VRAM `1,800 MiB`.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

#### Peak Temporary Disk

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

p50/p95 latency intervals are reported only when enough independent document or
page observations support them. A single first-item measurement, throughput
batch, or peak RAM/VRAM/temporary-disk observation is reported without a
confidence interval. If an operational measurement is repeated independently,
an interval may be reported only with the repetition count and aggregation unit.
Missing RAM, VRAM, or disk instrumentation is recorded as unavailable, never as
zero.

### Worked candidate interpretation

Suppose one extraction profile produces:

```text
Content Precision       = 0.98
Content Recall          = 0.75
Reading Order Accuracy  = 0.95
Page Coverage           = 1.00
Table Detection Recall  = 1.00
Table Content F1        = 0.90
TEDS                    = 0.55
TEDS-S                  = 0.60
Candidate Failure Rate  = 0.00
p95 warm latency/page   = 3.2 seconds
```

This means:

- nearly all extracted text is supported by the reference;
- approximately 25% of reference text was not recovered;
- recovered elements are mostly in the correct reading order;
- every reference page produced some matching content;
- every reference table was detected;
- table text was mostly recovered, but the low TEDS and TEDS-S show that the
  complete table and its structure were reconstructed poorly despite successful
  detection;
- no scheduled sample ended in a fatal error; and
- 95% of measured warm per-page observations took no more than 3.2 seconds.

No single number communicates all of these facts. That is why the benchmark
reports the metrics separately instead of calculating a weighted overall score.

### Document-extraction confidence intervals

A confidence interval expresses uncertainty in an aggregate calculated from
multiple independent observations. It does not describe the possible range of
an individual prediction, and it must not be attached to a value merely because
the value is reported.

#### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Development, validation, and locked text, page, layout, table, and formula aggregates | Yes | Calculated from the contributing independent samples. |
| Development, validation, and locked reliability rates | Yes | Calculated across scheduled independent samples. |
| Development, validation, and locked p50/p95 latency | Conditional | Reported when enough independent document or page observations support the percentile estimate. |
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

#### Calculation

Eligible development, validation, and locked sample-based metrics use 10,000 bootstrap resamples with
seed 42:

1. Treat each source document as the independent resampling unit.
2. Resample documents with replacement.
3. Recalculate the aggregate for every resample.
4. Use the 2.5th and 97.5th percentiles as the 95% interval bounds.

Document-group metrics resample only the samples in that group. Conditional
metrics such as table structure or formula recognition resample only the samples
eligible for that metric. If a pooled detection resample contains neither a
reference nor a prediction,
its precision/recall/F1 denominator is undefined and that draw is excluded from
that metric's percentile calculation rather than converted to zero. The result
artifact records the number of defined resamples.

#### Interpretation

For example:

```text
Content F1 = 0.91
95% CI     = [0.89, 0.93]
```

The benchmark estimates an aggregate Content F1 of `0.91`; variation across the
sampled documents produces the reported uncertainty interval. A narrower
interval means the aggregate is estimated more precisely.

## Audio extraction

The ASR benchmark evaluates the complete ordered transcript, its timestamps,
catastrophic output behavior, and the cost of the recorded runtime profile. It
does not use Content F1 or Reading Order Accuracy: unlike a two-dimensional
page, audio already defines one chronological sequence.

### Metric summary

#### Recognition quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Corpus Word Error Rate (WER) | Primary | How wrong is the complete ordered word transcript? | Lower |
| Corpus Character Error Rate (CER) | Secondary | How severe are character-level spelling, name, and number errors? | Lower |

#### WER diagnostics

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Word Substitution Rate | Diagnostic | How often is a spoken word recognized as a different word? | Lower |
| Word Deletion Rate | Diagnostic | How much spoken content is omitted? | Lower |
| Word Insertion Rate | Diagnostic | How much unsupported word content is added? | Lower |

#### Timestamp quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Timestamp Boundary MAE | Primary | How far are aligned segment starts and ends from the reference boundaries? | Lower |
| Timestamp Alignment Coverage | Primary | What proportion of timed reference segments received a valid alignment? | Higher |

#### Reliability and failure behavior

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Empty Transcript Rate | Diagnostic | How often does speech-containing audio produce no lexical text? | Lower |
| Nonspeech False-Transcription Rate | Diagnostic | How often does verified nonspeech audio produce lexical text? | Lower |
| Repeat Transcript Agreement Rate | Diagnostic | How often do repeated measured runs return the same scored transcript? | Higher |

#### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Complete-Pipeline Real-Time Factor | Operational | How much processing time is required relative to audio duration? | Lower |
| p50 Warm Clip Latency | Operational | What is normal warm processing latency? | Lower |
| p95 Warm Clip Latency | Operational | What is slow-case warm processing latency? | Lower |
| Cold Model-Load Time | Operational | How long does initial model loading take? | Lower |
| Peak Process-Tree RAM | Operational | How much total system memory does the candidate require? | Lower |
| Peak VRAM | Operational | How much GPU memory does the candidate require? | Lower |

These 16 metrics are the frozen ASR evaluation contract. Technical-Term
Accuracy is excluded because EduMind has no fixed subject vocabulary;
diarization metrics remain out of scope until speaker identification becomes a
product requirement.

### Recognition quality

#### Corpus Word Error Rate

**Question:** How wrong is the complete ordered word transcript?

Corpus WER uses the fixed ASR evaluator normalization described in the shared
conventions. Edit counts are pooled across clips before division; the benchmark
does not average independently calculated clip error rates.

```text
Corpus WER =
all word substitutions + deletions + insertions
------------------------------------------------
        all words in the reference clips
```

**Example:** Across the complete corpus, 1,000 reference words with 40
substitutions, 20 deletions, and 10 insertions produce WER `70 / 1,000 = 0.07`.

WER is primary because ASR must reproduce the ordered spoken sequence in one
value. The component rates below explain its failure type rather than replacing
it.

**Range and direction:** WER is non-negative and lower is better. It can exceed
`1` when insertions outnumber reference words. A speech set with no reference
words is invalid.

#### Corpus Character Error Rate

**Question:** How severe are character-level transcription errors that may be
hidden by whole-word scoring?

Corpus CER uses the same pooled calculation over characters:

```text
Corpus CER =
all character substitutions + deletions + insertions
-----------------------------------------------------
          all characters in the reference clips
```

**Example:** Fifty character edits across 5,000 reference characters produce
CER `50 / 5,000 = 0.01`.

CER supports WER by exposing small spelling, number, abbreviation, and name
errors that a whole-word error treats as one event.

**Range and direction:** CER is non-negative and lower is better. It can exceed
`1` when insertions outnumber reference characters. A speech set with no
reference characters is invalid.

### WER diagnostics

The three diagnostic rates reuse the exact word alignment and pooled reference
word count used by Corpus WER. Together they add up to WER, but each answers a
different failure question.

#### Word Substitution Rate

**Question:** How often is a spoken reference word replaced by a different
predicted word?

```text
Word Substitution Rate =
substituted reference words
---------------------------
all reference words
```

**Example:** Forty substitutions among 1,000 reference words produce `0.04`.

This distinguishes word confusion from omitted or invented speech.

**Range and direction:** Non-negative and lower is better. It cannot exceed `1`
because each reference word can be substituted at most once.

#### Word Deletion Rate

**Question:** How much spoken reference content is missing from the transcript?

```text
Word Deletion Rate =
deleted reference words
-----------------------
all reference words
```

**Example:** Twenty deletions among 1,000 reference words produce `0.02`.

This exposes omissions that are especially harmful when lectures contain
definitions, negations, or instructions.

**Range and direction:** Lies in `[0, 1]`; lower is better.

#### Word Insertion Rate

**Question:** How much unsupported word content did the model add?

```text
Word Insertion Rate =
inserted predicted words
------------------------
all reference words
```

**Example:** Ten insertions against 1,000 reference words produce `0.01`.

This exposes invented speech that is not present in the recording.

**Range and direction:** Non-negative and lower is better. It can exceed `1`
when the model inserts more words than the entire reference contains.

### Timestamp quality

#### Timestamp Boundary MAE and Timestamp Alignment Coverage

**Question:** How accurate are the predicted start/end times, and how much of
the timed reference could actually be aligned?

For each reference segment, the evaluator enumerates contiguous spans of
predicted timestamp units and keeps spans whose normalized token Content F1 is
at least `0.5`. Dynamic programming selects ordered, non-overlapping one-to-one
matches by maximum total similarity, then match count, then earliest spans.
No predicted unit can be reused. This lets a segment-level reference match
several word-level predictions while preventing one broad prediction from
covering multiple references. The span envelope supplies its minimum start and
maximum end. Timestamp Boundary MAE is then calculated as follows:

1. Find the absolute start-time error for every aligned segment.
2. Find the absolute end-time error for every aligned segment.
3. Average all start and end errors.

```text
Timestamp Alignment Coverage =
reference segments successfully aligned
---------------------------------------
reference segments with timestamps
```

**Example:** If 18 of 20 reference segments align, coverage is `18 / 20 =
0.90`. If their 36 start/end boundaries have 7.2 seconds of total absolute
error, Timestamp Boundary MAE is `7.2 / 36 = 0.20 seconds`.

Boundary MAE alone can look excellent when only easy segments align. Coverage
shows how much of the timed reference actually contributed. The two metrics are
therefore interpreted together: low MAE and high coverage. When no segments
align, MAE is undefined rather than fabricated as zero, while coverage is zero.

**Range and direction:** Boundary MAE is a non-negative number of seconds and
lower is better. Alignment Coverage lies in `[0, 1]` and higher is better.

### Reliability and failure behavior

#### Empty Transcript Rate

**Question:** How often does a valid speech clip produce no lexical transcript?

For speech-containing clips:

```text
Empty Transcript Rate =
speech clips producing no lexical text
--------------------------------------
       speech clips evaluated
```

**Example:** Four empty transcripts from 100 speech clips produce a rate of
`4 / 100 = 0.04`.

This exposes complete transcription failures that can be diluted inside corpus
WER. A process crash is a failed candidate run, not an empty transcript.

An empty transcript with an empty timestamp sequence remains a scoreable speech
sample: all reference words and characters are deletions, timestamp coverage is
zero, Boundary MAE is null, and this rate increments. Non-empty transcript text
without timestamps and empty text with lexical timestamp segments are
contradictory fatal outputs.

**Range and direction:** The rate lies in `[0, 1]`; lower is better. A dataset
without speech clips is invalid for this metric.

#### Nonspeech False-Transcription Rate

**Question:** How often does the model invent lexical speech on verified
nonspeech audio?

For the fixed silence, music-without-lyrics, background-noise, and environmental
sound controls:

```text
Nonspeech False-Transcription Rate =
nonspeech clips producing lexical text
---------------------------------------
       nonspeech clips evaluated
```

**Example:** Three controls producing text among 20 nonspeech clips produce a
rate of `3 / 20 = 0.15`.

WER cannot score these controls because their references contain zero words.
This rate directly measures invented speech where no spoken reference exists.

**Range and direction:** The rate lies in `[0, 1]`; lower is better. It is
undefined when no reliability controls were evaluated.

#### Repeat Transcript Agreement Rate

**Question:** How often does the same model profile return the same transcript
when it processes the same clip repeatedly?

For each speech clip, the three measured transcripts receive the same prose
projection used for WER. The clip receives `1` only when all three projected
transcripts are identical and `0` otherwise. The benchmark then averages those
clip values:

```text
Repeat Transcript Agreement Rate =
speech clips with identical repeated transcripts
-------------------------------------------------
             speech clips evaluated
```

**Example:** If 47 of 50 clips have identical transcripts in all three measured
runs, agreement is `47 / 50 = 0.94`.

This is a determinism diagnostic, not a correctness score. A model can repeat
the same wrong transcript perfectly, so the value must be read with WER. The
metric uses the repetitions already collected for latency and adds no model
inference.

**Range and direction:** `[0, 1]`; higher is better. Smoke uses one measured
run and therefore cannot provide meaningful determinism evidence.

### Operational performance

#### Complete-Pipeline Real-Time Factor

**Question:** How much complete processing time is required relative to audio
duration?

```text
Complete-Pipeline Real-Time Factor =
total transcription-and-timestamp processing time
--------------------------------------
          total audio duration
```

**Example:** Processing 60 minutes of audio in 15 minutes produces RTF `15 / 60
= 0.25`.

An RTF below `1` means processing is faster than audio playback.

**Range and direction:** RTF is non-negative and lower is better.

#### p50 and p95 Warm Clip Latency

**Question:** What are the typical and slow-tail steady-state transcription
times per clip?

For each clip, the benchmark takes the median latency of its measured warm
repetitions. It then reports p50 across clips as typical latency and p95 as the
slow tail.

**Example:** A p95 of `5.2 seconds` means 95% of the per-clip warm latency
observations are at or below 5.2 seconds.

**Range and direction:** Non-negative seconds per clip; lower is better.

#### Cold Model-Load Time

**Question:** How long does a fresh worker need to make the ASR model ready?

Measure from the start of model construction until the complete ASR profile is
ready, before warmups or transcription. Any component needed before the first
request belongs to cold load; work performed later belongs to the measured
complete-pipeline latency and cannot be hidden from both measurements.

**Range and direction:** Non-negative seconds; lower is better.

#### Peak Process-Tree RAM and Peak VRAM

**Question:** What maximum system and GPU memory does the complete candidate
execution require?

Peak Process-Tree RAM is the largest sampled resident-memory total for the
worker and its child processes. Peak VRAM is the largest GPU-memory allocation
attributable to that worker. Every result records the explicit CPU or GPU
profile; silent device fallback is invalid.

**Example:** Process-tree samples peaking at `3,200 MiB` and GPU samples peaking
at `2,100 MiB` produce those two reported peaks.

**Range and direction:** Non-negative MiB; lower is better at equal quality. A
confirmed CPU-only profile reports zero VRAM; unavailable measurement is not
converted to zero.

### Worked candidate interpretation

Suppose one ASR profile produces:

```text
Corpus WER                         = 0.08
Word Deletion Rate                 = 0.04
Timestamp Boundary MAE             = 0.32 seconds
Timestamp Alignment Coverage       = 0.94
Nonspeech False-Transcription Rate = 0.10
Complete-Pipeline RTF              = 0.40
Peak VRAM                          = 2,100 MiB
```

The profile transcribes faster than playback and aligns most timed segments,
but half of its word errors come from omitted speech and it invents text on 10%
of nonspeech controls. Those failure modes remain visible even though the total
WER is relatively low. No weighted overall score combines these values.

### Audio confidence intervals

#### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Development, validation, and locked WER, CER, and substitution/deletion/insertion rates | Yes | Resample complete speech clips and recalculate the pooled counts. |
| Development, validation, and locked Timestamp Alignment Coverage | Yes | Resample complete timed speech clips. |
| Development, validation, and locked Timestamp Boundary MAE | Yes, when defined | Use resampled clips containing valid aligned boundaries. |
| Development, validation, and locked Empty Transcript Rate | Yes | Resample speech clips. |
| Development, validation, and locked Nonspeech False-Transcription Rate | Yes | Resample the separate nonspeech controls. |
| Development, validation, and locked Repeat Transcript Agreement Rate | Yes | Resample complete speech clips with their already-computed agreement flags. |
| Development, validation, and locked Real-Time Factor | Yes | Resample complete speech clips with their measured processing time and duration. |
| Development, validation, and locked p50/p95 warm latency | Conditional | Report only when enough independent clips support the percentile estimate. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| One cold-load measurement | No | One observation cannot estimate uncertainty. |
| One observed peak RAM or VRAM value | No | Report the observed peak without invented bounds. |

#### Calculation

Development, validation, and locked runs use 10,000 bootstrap resamples with seed 42:

1. Treat each complete clip as the independent unit.
2. Resample speech clips with replacement; resample nonspeech controls
   separately for their reliability rate.
3. Recalculate each eligible aggregate from the sampled counts, timestamps,
   durations, and latencies.
4. Use the 2.5th and 97.5th percentiles as the 95% bounds.

A resample with no timestamp match still contributes to recognition,
reliability, latency, and zero Alignment Coverage. Its undefined Boundary MAE
does not enter the MAE percentile calculation.

If no reference segment aligns anywhere in the complete candidate run,
Timestamp Boundary MAE is stored as null, Alignment Coverage is `0`, and the run
remains successful. This reports the absence of a measurable boundary without
inventing either a perfect or infinitely bad time error. The null is preserved
in `candidate.json` and `summary.json`; its MLflow scalar key is absent because
MLflow scalar metrics cannot represent null.

#### Interpretation

Corpus WER `0.08` with a 95% CI of `[0.07, 0.10]` means the observed aggregate
error rate is 8%, while clip resampling estimates a plausible range of 7% to
10%. A narrower interval indicates a more precise corpus estimate; it does not
mean that every individual clip has an error rate inside that range.

## Video extraction

The video experiment compares keyframe configurations while holding the
document parser and ASR fixed. Its quality metrics therefore answer two focused
questions: did the selected frames recover the useful visible content, and did
they place that content at a useful time? Spoken and visible tokens are not
combined into one score because the much longer transcript would dominate it.

### Metric summary

#### Visual-content quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Visual Content F1 | Primary | Does the configuration balance correct visible text with complete visible-text recovery? | Higher |
| Visual Content Precision | Secondary | How much extracted visible text is supported by the reference? | Higher |
| Visual Content Recall | Secondary | How much verified visible text was recovered from selected frames? | Higher |

#### Visual timestamp quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Mean Visual First-Detection Delay | Primary | After visible text first appears, how long does the strategy take to capture it? | Lower |
| Timed Visual Occurrence Coverage | Primary | What proportion of verified timed visible-text occurrences were captured at least once while visible? | Higher |

#### Reliability and failure behavior

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Duplicate Visual Text Rate | Diagnostic | How much extracted visible content was repeated because similar frames were selected repeatedly? | Lower |

#### Frozen-input diagnostic

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Frozen-ASR Transcript WER | Diagnostic | Is the frozen audio transcript sufficiently understood when interpreting the later combined pipeline? | Lower |

#### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Visual Real-Time Factor | Operational | How much keyframe-selection and visual-parsing time is required relative to video duration? | Lower |
| p50 Warm Visual Latency | Operational | What is normal warm visual-processing time for one video? | Lower |
| p95 Warm Visual Latency | Operational | What is slow-case warm visual-processing time? | Lower |
| Cold Visual-Pipeline Load Time | Operational | How long does initial loading of the keyframe and document-parser path take? | Lower |
| Peak Visual Process-Tree RAM | Operational | How much system memory does the visual path require? | Lower |
| Peak Visual VRAM | Operational | How much GPU memory does the visual path require? | Lower |

#### Workload descriptor

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Mean Selected Frames per Video | Workload descriptor | How many frames does the configuration send to the document parser on average? | Descriptive |

### Visual-content quality

#### Visual Content Precision, Recall, and F1

**Question:** How much extracted visible content is supported, how much required
visible content was recovered, and does the configuration balance both?

The evaluator compares the distinct normalized visible content extracted from
the selected frames with the human-verified visible content. Repeated copies of
the same text do not earn additional matches; repetition is measured separately
by Duplicate Visual Text Rate.

```text
Visual Content Precision =
matched extracted visible-content units
---------------------------------------
all extracted visible-content units

Visual Content Recall =
matched reference visible-content units
---------------------------------------
all reference visible-content units

Visual Content F1 =
the balance between Visual Content Precision and Visual Content Recall
```

**Example:** The reference contains 100 visible-content units. A configuration
extracts 90 units, 80 of which match the reference. Precision is `80 / 90 =
0.89`, recall is `80 / 100 = 0.80`, and F1 is approximately `0.84`.

F1 is primary because it prevents a configuration from looking good by
extracting almost nothing or by extracting large amounts of unsupported text.
Precision and recall are supporting diagnostics that explain whether a lower F1
comes from extra text or missing text.

Corpus aggregates are recomputed from the summed per-video distinct-reference,
distinct-prediction, and matched-unit counts. They are not an unweighted mean of
the per-video ratios. Timed occurrence coverage and Duplicate Visual Text Rate
are pooled from their corresponding occurrence counts for the same reason.

**Range and direction:** All three values lie in `[0, 1]`; higher is better. If
both reference and prediction contain no visible content, the sample is not
eligible rather than being assigned perfect quality.

### Visual timestamp quality

#### Mean Visual First-Detection Delay and Timed Visual Occurrence Coverage

**Question:** Did the strategy capture each timed visible-text occurrence while
it was on screen, and how long after appearance did the first capture happen?

Each verified visible-text occurrence has text plus an interval during which it
is visible. A reference occurrence is covered when a selected frame inside that
interval yields matching text. Its delay is the first matching frame time minus
the reference start time.

The versioned video protocol supplies the normalized Content F1 eligibility threshold and
freezes frame-time tolerance at zero, so a frame outside the verified interval
is never eligible. Matching is one-to-one: each reference occurrence and
each normalized text unit from a selected frame can be used at most once.
Cardinality is maximized first, then the earliest eligible detections are chosen,
so repeated frames cannot inflate coverage and delay really is first detection.

```text
Timed Visual Occurrence Coverage =
timed reference occurrences captured while visible
--------------------------------------------------
          all timed reference occurrences

Mean Visual First-Detection Delay =
sum of first matching frame time minus reference start time
-----------------------------------------------------------
                covered reference occurrences
```

**Example:** Suppose 8 of 10 timed occurrences are captured while visible, so
coverage is `0.80`. If their first matching frames arrive a total of 12 seconds
after their verified starts, mean first-detection delay is `12 / 8 = 1.5
seconds`.

The two values must be read together. Low delay with low coverage means the
strategy quickly captured only an easy subset. A useful result has high
coverage and low delay. If nothing is covered, coverage is zero and delay is
undefined rather than reported as a false zero.

**Range and direction:** Coverage lies in `[0, 1]` and higher is better. Delay
is a non-negative number of seconds and lower is better; it is null when no
timed occurrence is covered.

### Reliability and failure behavior

#### Duplicate Visual Text Rate

**Question:** How much output repeats content already recovered from an
unchanged or repeatedly selected frame?

```text
Duplicate Visual Text Rate =
repeated normalized visible-content occurrences after their first occurrence
-------------------------------------------------------------------------
all extracted visible-content occurrences
```

**Example:** If a configuration extracts 50 visible-text occurrences and 15 are
repeated copies of content already recovered from unchanged frames, the rate is
`15 / 50 = 0.30`.

This metric exposes wasted document-parser work and repeated downstream context.
It is separate from content precision so duplicates cannot obscure whether the
text itself is supported by the video.

**Range and direction:** The rate lies in `[0, 1]`; lower is better. An output
with no visible-content occurrences is undefined for duplication and is handled
by the visual-content metrics instead.

### Frozen-input diagnostic

#### Frozen-ASR Transcript WER

**Question:** How accurately does the already selected ASR transcribe the audio
of this video corpus?

The ASR profile and its audio output are frozen before frame selection begins.
Transcript WER is therefore calculated once for the shared ASR output and
stored on the phase's frozen-ASR child run. It is a diagnostic for understanding the final
combined extraction, not a metric for ranking keyframe configurations.

Its calculation, range, and direction use the Corpus WER contract in the audio
section. It is calculated once per video phase rather than repeated for every
keyframe configuration.

### Operational performance

#### Visual Real-Time Factor

**Question:** How much keyframe-selection and visual-parsing time is required
relative to video duration?

```text
Visual Real-Time Factor =
keyframe-selection and visual-parsing time
------------------------------------------
               video duration
```

**Example:** Extracting a four-minute video in one minute produces an RTF of
`1 / 4 = 0.25`. A value below 1 means extraction is faster than playback.

**Range and direction:** RTF is non-negative and lower is better.

#### p50 and p95 Warm Visual Latency

**Question:** What are the typical and slow-tail visual-processing times after
the document parser is loaded?

For each video, take the median latency of its measured warm repetitions. The
benchmark reports p50 across videos as typical latency and p95 as the slow tail.

**Example:** A p95 of `48 seconds` means 95% of measured warm video latencies
are at or below 48 seconds.

**Range and direction:** Non-negative seconds per video; lower is better.

#### Cold Visual-Pipeline Load Time

**Question:** How long does a fresh worker need to load the keyframe and frozen
document-parser path required by the video configuration?

Measure model construction and loading before warmups or video extraction. It
is kept separate from warm latency so startup does not distort steady-state
performance.

**Range and direction:** Non-negative seconds; lower is better.

#### Peak Visual Process-Tree RAM and Peak Visual VRAM

**Question:** What maximum system and GPU memory does the visual path require?

Peak RAM is the largest sampled resident-memory total across the worker and its
child processes. Peak VRAM is the largest GPU-memory allocation attributable to
that worker.

**Example:** If process-tree RAM peaks at `5,600 MiB` and GPU memory peaks at
`2,900 MiB`, those are the reported resource values.

**Range and direction:** Non-negative MiB; lower is better at equal quality. A
confirmed CPU-only profile reports zero VRAM; unavailable measurement is not
converted to zero.

#### Mean Selected Frames per Video

**Question:** How many frames does the strategy send to the frozen document
parser, on average?

```text
Mean Selected Frames per Video =
total selected frames
---------------------
videos evaluated
```

**Example:** Selecting 360 frames across 30 videos produces a mean of `12`
frames per video.

**Range and direction:** Non-negative frames per video. Lower is preferable
only when visual-content and timestamp quality are preserved; it is a cost
diagnostic rather than a standalone selection objective.

### Worked candidate interpretation

Suppose one keyframe configuration produces:

```text
Visual Content Precision           = 0.92
Visual Content Recall              = 0.76
Visual Content F1                  = 0.83
Mean Visual First-Detection Delay  = 1.5 seconds
Timed Visual Occurrence Coverage   = 0.80
Duplicate Visual Text Rate         = 0.18
Mean Selected Frames per Video     = 14
Visual Real-Time Factor            = 0.35
```

Most extracted visible content is supported by the reference, but approximately
one quarter of the required content is still missed. The matched content is
captured quickly when found, although 20% of timed occurrences were never
captured while visible. Eighteen
percent duplicate output suggests that the strategy still selects some
unchanged frames. The frame count and RTF show the processing cost that produced
this quality. No weighted overall score combines these values.

### Video confidence intervals

#### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Development, validation, and locked Visual Content Precision/Recall/F1 | Yes | Resample complete videos and recalculate each aggregate. |
| Development, validation, and locked Timed Visual Occurrence Coverage | Yes | Resample complete videos with their timed visible references. |
| Development, validation, and locked Mean Visual First-Detection Delay | Yes, when defined | Use resampled videos containing covered timed occurrences. |
| Development, validation, and locked Duplicate Visual Text Rate | Yes | Resample complete videos with their duplicate counts. |
| Development, validation, and locked Frozen-ASR Transcript WER | Yes | Resample the shared per-video ASR outputs; report the result on the frozen-ASR child run. |
| Development, validation, and locked Visual Real-Time Factor and Mean Selected Frames per Video | Yes | Resample complete videos with their duration, visual-processing time, and frame counts. |
| Development, validation, and locked p50/p95 warm latency | Conditional | Report only when enough independent videos support the percentile estimate. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| One cold-load measurement | No | One observation cannot estimate uncertainty. |
| One observed peak RAM or VRAM value | No | Report the observed peak without invented bounds. |

#### Calculation

Development, validation, and locked runs use 10,000 bootstrap resamples with seed 42:

1. Treat each complete video as the independent unit.
2. Resample videos with replacement.
3. Recalculate each eligible aggregate from the sampled visual matches, timed
   occurrences, duplicates, durations, latencies, and frame counts.
4. Use the 2.5th and 97.5th percentiles as the 95% bounds.

A resample with no covered timed occurrence still contributes zero Timed Visual
Occurrence Coverage and contributes normally to the other defined metrics. Its
undefined First-Detection Delay does not enter the delay percentile calculation.

#### Interpretation

Visual Content F1 `0.84` with a 95% CI of `[0.79, 0.88]` means the point
estimate summarizes the observed videos, while video resampling estimates the
uncertainty around it. A narrow interval indicates a more precise corpus-level
estimate; it does not describe the range of individual-video F1 values.

## Chunking and embedding

The chunking/embedding experiment evaluates one complete
`chunker|embedding` pair at a time with exact cosine search. Three primary metric
families answer different selection questions; alpha-nDCG is retained as a
novelty diagnostic. None is combined into a weighted score.

### Metric summary

#### Retrieval quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| nDCG@3/@5 | Primary | Are chunks containing verified evidence ranked near the top? | Higher |
| Evidence-unit Recall@3/@5 | Primary | How much of the required evidence is present in the retrieved set? | Higher |
| Evidence-token Precision@3/@5 | Primary | How concentrated is the retrieved text around verified evidence? | Higher |
| alpha-nDCG@3/@5 | Diagnostic | On multi-evidence questions, is new evidence placed early instead of repeatedly covering evidence already retrieved? | Higher |

`@3` and `@5` mean that the same calculation is performed over the first three
and first five ranked chunks. Both are reported because Final RAG evaluates
both context counts.

#### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Corpus-Build Time | Operational | How long does chunking, document embedding, and searchable-matrix preparation take? | Lower |
| Corpus-Build Throughput | Operational | How many original source tokens are processed per second? | Higher |
| p50 Warm Query Latency | Operational | What is normal query-embedding and exact-search latency? | Lower |
| p95 Warm Query Latency | Operational | What is slow-case warm query latency? | Lower |
| Peak Process-Tree RAM | Operational | How much total system memory does the pair require? | Lower at equal quality |
| Peak VRAM | Operational | How much GPU memory does the pair require? | Lower at equal quality |

#### Workload and storage descriptors

| Value | Role | Question answered | Direction |
|---|---|---|---|
| Corpus Counts | Workload descriptor | How many documents and answerable/unanswerable questions were represented in the run? | Descriptive |
| Source Tokens | Workload descriptor | How large was the original corpus before chunk overlap? | Descriptive |
| Indexed-Token Occurrences | Workload descriptor | How much text was embedded after overlap and repeated context were counted? | Descriptive |
| Chunk Count | Workload descriptor | How many searchable vectors and metadata records did the strategy create? | Descriptive |
| Mean/p95 Chunk Tokens | Workload descriptor | What typical and long-tail chunk sizes did the strategy actually produce? | Descriptive |
| Embedding Dimension and Dtype | Storage descriptor | What shape and numeric representation did each vector use? | Descriptive |
| Embedding-Matrix Bytes | Storage descriptor | How much storage did the complete chunk-vector matrix occupy? | Descriptive |

### Retrieval quality

#### nDCG@3/@5

**Question:** How early do relevant chunks appear when repeated evidence is not
penalized?

nDCG gives more credit when chunks containing verified evidence appear earlier.
A chunk is relevant when it completely covers at least one required evidence
unit. Covering an evidence unit already found in an earlier chunk does not
reduce that chunk's relevance credit.

**Example:** Moving a relevant chunk from rank 4 to rank 2 improves nDCG. Two
chunks containing the same relevant passage can both receive relevance credit.

nDCG is primary because conventional relevance ranking is the direct job of the
chunker/embedding pair. Evidence-unit Recall separately shows whether repetition
displaced other required evidence.

**Range and direction:** `[0, 1]`; higher is better. A question for which no
candidate chunk contains verified evidence receives zero.

#### Evidence-unit Recall@3/@5

**Question:** How many required evidence units appear anywhere in the first
three or first five chunks, regardless of their order?

Each required evidence unit counts once when at least one chunk inside the
cutoff contains it completely. Repeated copies do not add credit, and partial
fragments do not count as recovered units.

**Example:** If the first three chunks recover one of three required units and
the next two recover another, Recall@3 is approximately `0.33` and Recall@5 is
approximately `0.67`.

This primary metric catches rankings that look good near the top but still miss
part of the evidence needed for a complete answer. It also makes a separate Hit
Rate unnecessary.

**Range and direction:** `[0, 1]`; higher is better.

#### Evidence-token Precision@3/@5

**Question:** What share of the tokens returned in the first three or first five
chunks is relevant evidence?

The metric compares evidence-bearing source tokens with all source tokens
inside the cutoff. Every candidate uses the same evaluation tokenizer, and text
repeated through chunk overlap is counted each time it is returned.

**Example:** If 160 of 600 retrieved tokens are relevant evidence, precision is
approximately `0.27`. The remaining tokens are additional context not counted
as relevant evidence by the reference annotations.

This primary metric does not impose a context budget or penalize a chunk merely
for being large. It reports how concentrated the returned context is around the
verified evidence.

**Range and direction:** `[0, 1]`; higher is better. Empty retrieved text
receives zero.

#### Alpha-nDCG@3/@5

**Question:** Are different required evidence units placed early instead of
being displaced by repeated evidence?

Alpha-nDCG rewards useful evidence more when it appears near the top and reduces
the credit for later chunks that repeat the same evidence. With the frozen
`alpha=0.5` setting, each repetition receives half the remaining novelty credit.

**Example:** If the first two chunks contain the same evidence and the third
contains different evidence, the second chunk is discounted. Moving the
different evidence to rank 2 improves the score.

Here, repetition is not decided by text similarity. It means that a chunk
covers an evidence-unit ID already covered by a higher-ranked chunk. Exact
duplicate chunk IDs are forbidden separately by the retrieval contract.

This is diagnostic rather than primary because repeated relevant evidence is
not automatically a retrieval failure. It can still explain why a candidate
with strong conventional ranking quality covers fewer distinct evidence units.
The metric is calculated only for questions with at least two distinct gold
evidence units; it is omitted for other questions, and its eligible question
and document counts are reported.

**Range and direction:** `[0, 1]`; higher is better. An eligible question for
which no candidate chunk contains verified evidence receives zero.

### Why the metrics are not interchangeable

| Metric | What changes it | What it does not answer directly |
|---|---|---|
| nDCG@3/@5 | The ranks of chunks that contain complete evidence | Whether all distinct evidence units were found or how much extra text was returned |
| Evidence-unit Recall@3/@5 | Whether each required evidence unit appears inside the cutoff | Whether the recovered units were ordered well or surrounded by extra context |
| Evidence-token Precision@3/@5 | How much returned text is annotated evidence | Whether all required units were found or ranked early |
| alpha-nDCG@3/@5 | On eligible questions, the order in which different evidence units appear | Whether repetition actually harms the downstream answer |

The three primary families are complementary. Reordering the same chunks can
change nDCG without changing recall or token precision. Adding non-evidence text
can lower token precision without changing relevance order or recovered units.
Missing one required unit lowers recall even when the remaining relevant chunks
are ranked early. Alpha-nDCG deliberately overlaps with nDCG, but is kept only
to diagnose novelty on questions where novelty is measurable.

### Worked candidate interpretation

Suppose a valid pair reports:

```text
Evidence-unit Recall@5       = 0.82
Evidence-token Precision@5  = 0.44
nDCG@5                       = 0.81
alpha-nDCG@5                 = 0.74
```

The nDCG result says that relevant chunks generally appear early. Recall says
that 82% of the required evidence units are present somewhere in the first five
chunks, leaving 18% missing. Token precision says that 44% of the returned
tokens are relevant evidence and 56% are
additional context according to the reference annotations. On the eligible
multi-evidence subset, the lower alpha-nDCG result suggests that repeated
evidence sometimes appears before new evidence. It does not by itself declare
those repetitions harmful. The same interpretation is performed separately at
`@3`. No formula combines these values.

### Operational performance

The phase calls preparation of the searchable embedding matrix **corpus build**.
It is not the later vector-server indexing experiment.

#### Corpus-build elapsed time and throughput

**Question:** How much steady-state work is required to chunk and embed the
fixed corpus?

Timing begins after the model and tokenizer are loaded. It includes chunk
creation, document embedding, and embedding-matrix/metadata assembly, and
excludes downloads and environment installation.

Corpus-build throughput divides the number of original source tokens in the
frozen corpus by corpus-build wall-clock seconds. Original source tokens use the
fixed evaluation tokenizer. Indexed-token occurrences are not used because
overlap would otherwise reward a candidate for duplicating text.

**Range and direction:** elapsed seconds are non-negative and lower is better at
equal quality; source tokens/second are non-negative and higher is better at
equal quality.

#### Warm query latency p50 and p95

**Question:** What are normal and slow-tail times for query embedding plus exact
cosine top-20 search?

After warmup, execute every query for the configured measured repetitions. Use
that query's median repetition as its latency observation, then calculate p50
and p95 across eligible questions. Search timing includes query tokenization,
query embedding, cosine scoring, deterministic ordering, and top-20 selection.
It excludes corpus build.

p99 is not authoritative in this phase because the corpus does not provide
enough thousands of independent query requests to estimate a stable one-percent
tail. The vector-server load benchmark measures p99 under controlled
concurrency.

**Range and direction:** non-negative milliseconds per query; lower is better at
equal quality.

#### Peak process-tree RAM and peak VRAM

**Question:** What maximum system and GPU memory does the complete pair require
during corpus build and query evaluation?

Peak RAM is the largest sampled resident-memory total across the benchmark
worker and its child processes. Peak VRAM uses the shared process-attributed
resource-monitor contract. The artifact identifies the measurement method.

**Range and direction:** non-negative MiB; lower is better at equal quality. A
confirmed CPU-only run may report zero VRAM; unavailable GPU instrumentation is
not converted to zero.

### Workload and storage descriptors

These values explain operational outcomes. They are numeric and chartable but
are not quality metrics or independent winner-selection objectives.

Source tokens count the frozen corpus once with `tiktoken:cl100k_base`, while
indexed-token occurrences count all generated chunks, including overlap. This
separates fixed corpus size from the embedding workload created by a strategy.
Chunk counts and chunk-length summaries explain search work and the actual size
distribution produced by each strategy.

The recorded embedding-matrix byte count is the authoritative storage value and
is cross-checked against chunk count, embedding dimension, and stored dtype.
These descriptors explain quality and speed differences but cannot compensate
for worse retrieval quality.

### Chunking and embedding confidence intervals

#### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Development and validation nDCG, Evidence-unit Recall, Evidence-token Precision, and eligible alpha-nDCG at @3/@5 | Yes | Resample source documents and recalculate each aggregate. |
| Text, table, formula, and mixed evidence slices | Yes, when enough documents contribute | Resample only the contributing source documents. |
| p50/p95 warm query latency | Conditional | Report only when enough independent query observations support the percentile estimate. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| Corpus-build time and throughput | No | One corpus-build observation cannot estimate uncertainty. |
| Peak RAM and VRAM | No | Report the observed peak without invented bounds. |
| Workload and storage descriptors | No | These are observed properties of the candidate and fixed corpus, not sampled quality estimates. |

#### Calculation

Quality is first calculated for each answerable question. Questions are averaged
within their source document so a paper with many questions cannot dominate the
result. Development and validation runs then use 10,000 bootstrap resamples of complete
documents with seed 42 and take the 2.5th and 97.5th percentiles as the 95%
confidence bounds.

If too few documents contribute to a slice or conditional latency interval, the
point estimate remains available but the interval is omitted rather than
reported as artificially precise.

#### Interpretation

An nDCG@5 of `0.81` with a 95% confidence interval of `[0.77, 0.85]` means
`0.81` is the observed aggregate, while resampling complete source documents
estimates its uncertainty. The interval does not describe the range of
individual-question scores.

## Retrieval and reranking

This phase compares 15 complete `retriever|reranker` candidates: Dense, BM25, and
RRF, each with no reranker or one of four learned rerankers. Quality uses the
same frozen evidence representation and evaluation tokenizer as the
chunking/embedding benchmark. No context-token budget is applied and no metric
families are combined into a weighted score.

### Metric summary

#### Retrieval quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| nDCG@3/@5 | Primary | Does the complete stack place chunks containing verified evidence near the top? | Higher |
| Evidence-unit Recall@3/@5 | Primary | How much of the required evidence is present in the first three or five chunks? | Higher |
| Evidence-token Precision@3/@5 | Primary | How concentrated are the first three or five chunks around verified evidence? | Higher |
| alpha-nDCG@3/@5 | Diagnostic | On eligible multi-evidence questions, does the ranking surface new evidence early instead of repeatedly covering evidence already found? | Higher |
| Candidate-pool Evidence-unit Recall@20 | Diagnostic | Did the first-stage top-20 pool contain the required evidence before any reranker reordered it? | Higher |
| Ranking Agreement | Validity gate | Does repeated inference return the same complete ordering? | Must equal 1.0 |

`@3` and `@5` mean that the calculation uses the first three and first five
ranked chunks. Both cutoffs are primary because the later complete-system
experiment evaluates both context counts. The top-20 metric diagnoses the
first-stage ceiling; it is not another selection cutoff.

#### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Full-Stack Warm Latency p50/p95 | Operational | How long does the deployable first-stage-plus-reranker path usually take, and how slow is its warm tail? | Lower |
| First-Stage Warm Latency p50/p95 | Operational | How much of total query time belongs to Dense, BM25, or RRF retrieval? | Lower |
| Reranker Warm Latency p50/p95 | Operational | For reranked candidates, what additional query time does learned reranking require? | Lower |
| Cold Initialization Time | Operational | How long does the complete candidate take to become ready from a fresh worker? | Lower |
| Peak Process-Tree RAM | Operational | What maximum system memory does the candidate require? | Lower |
| Peak VRAM | Operational | What maximum GPU memory does the candidate require? | Lower |
| Index-Build Time | Operational | How long does preparation of the first-stage searchable state take? | Lower |
| Index Bytes | Storage descriptor | How much stored searchable state does the first-stage retriever require? | Descriptive |

#### Workload descriptors

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Corpus, Query, Chunk, and Pool Counts | Workload descriptor | What fixed data and ranked-list sizes were actually processed? | Descriptive |
| Mean/p95 Retrieved Tokens@3/@5 | Workload descriptor | How much canonical source text would each cutoff pass downstream? | Descriptive |
| Mean/p95 Reranker Input Tokens | Workload descriptor | How much native-tokenizer input did a learned reranker process per query? | Descriptive |
| Reranker Snapshot Bytes | Storage descriptor | How much local model storage does a reranker require? | Descriptive |

Workload and storage values explain latency and resource differences. They do
not compensate for worse retrieval quality and are not combined with quality
into a winner score.

### Retrieval quality

#### nDCG@3/@5

**Question:** How early does the complete stack place relevant chunks when
repeated evidence is not treated as an error?

nDCG gives more credit when chunks containing complete verified evidence appear
at earlier ranks. A chunk remains relevant when it covers evidence already seen
in another chunk, so the metric evaluates conventional relevance ordering rather
than diversity.

For each question, the ideal ranking is derived from the relevance labels of
every chunk in the complete frozen corpus. It is not derived from Dense, BM25,
RRF, or any candidate's top-20 pool. All 15 candidates therefore use the same
ideal denominator for that question; a retriever cannot make its normalization
easier by failing to retrieve relevant chunks.

**Example:** Moving an evidence-bearing chunk from rank 5 to rank 2 improves
nDCG. Two high-ranked chunks that contain the same verified passage may both
receive relevance credit.

This is primary because ordering relevant material is the direct job shared by
the retriever and reranker. Evidence-unit Recall separately reveals whether
repetition displaced other required evidence.

**Range and direction:** `[0, 1]`; higher is better. A question with no verified
evidence in the scored results receives zero.

#### Evidence-unit Recall@3/@5

**Question:** How many distinct required evidence units appear anywhere inside
the cutoff?

Each gold evidence-unit ID counts once when at least one of the first three or
five chunks contains that unit completely. Rank does not change the credit,
repeated coverage does not add credit, and an incomplete fragment does not count
as a recovered unit.

**Example:** If a question needs three evidence units and the first three chunks
recover one, Recall@3 is about `0.33`. If ranks 4 and 5 add a second unit,
Recall@5 is about `0.67`.

This is primary because an early-looking ranking may still omit evidence needed
for a complete answer. It also makes a separate Hit Rate unnecessary.

**Range and direction:** `[0, 1]`; higher is better.

#### Evidence-token Precision@3/@5

**Question:** What share of the returned token occurrences is annotated as
verified evidence?

Canonical chunk text is measured with `tiktoken:cl100k_base`. Evidence-bearing
source intervals contribute relevant tokens; all returned source tokens form
the denominator. Overlapping text returned in multiple chunks is counted each
time because it consumes downstream context each time.

**Example:** If the first five chunks contain 600 token occurrences and 240 are
inside verified evidence intervals, Evidence-token Precision@5 is `0.40`. The
remaining 60% is additional text, not necessarily incorrect text.

This is primary because nDCG and recall can be high while the selected chunks
still contain substantial unrelated material.

**Range and direction:** `[0, 1]`; higher is better.

#### alpha-nDCG@3/@5

**Question:** On questions that require multiple distinct evidence units, does
the ranking introduce new evidence early?

Alpha-nDCG discounts a lower-ranked chunk only when it covers an evidence-unit
ID already covered higher in the ranking. It does not use text similarity to
decide that two chunks are duplicates, and it does not declare repeated relevant
evidence inherently bad.

**Example:** If ranks 1 and 2 contain the same evidence unit and rank 3 contains
a different required unit, moving the different unit to rank 2 improves
alpha-nDCG while ordinary nDCG may remain high.

The metric is diagnostic because novelty is useful context information but is
not the retriever's only responsibility. It uses fixed `alpha=0.5` and is
calculated only for questions with at least two distinct gold evidence units.
For other questions it is omitted rather than reported as zero.

Its ideal ordering is built deterministically from the complete frozen corpus,
choosing chunks that add the most not-yet-covered evidence at each rank. It is
never constructed from the candidate's retrieved pool.

**Range and direction:** `[0, 1]`; higher is better on its eligible subset.

#### Candidate-pool Evidence-unit Recall@20

**Question:** Did the first-stage retriever give its rerankers an adequate set of
candidates?

The metric measures distinct required evidence units found anywhere in the
frozen top-20 pool before learned reranking. A reranker cannot recover evidence
that is absent from this pool. The value therefore separates a first-stage miss
from a poor reordering decision.

It is recorded once on each of `dense|none`, `bm25|none`, and `rrf|none`. The
four reranker children reference the matching pool artifact and checksum instead
of copying the same value as if they had created it.

**Range and direction:** `[0, 1]`; higher is better as a diagnostic ceiling.

#### Ranking Agreement

**Question:** Does the same candidate produce the same ordered chunk IDs when
the query is repeated under identical conditions?

The first measured repetition is the designated ordering. Every later measured
repetition is an agreement only when its complete ordered top-20 chunk-ID list
is identical at every position. Ranking Agreement is the number of matching
query/repetition comparisons divided by the number expected. A failed
repetition counts as a disagreement. It catches unstable tie handling or
nondeterministic model behavior that could make quality results irreproducible;
it does not require floating-point scores themselves to be bit-identical.

The required authoritative value is `1.0`. A lower value is a validity failure,
not a quality trade-off.

### Why the quality metrics are not interchangeable

| Metric | What changes it | What it does not answer directly |
|---|---|---|
| nDCG@3/@5 | The ranks of evidence-bearing chunks | Whether every distinct unit was recovered or how much extra text was returned |
| Evidence-unit Recall@3/@5 | Which distinct required units appear within the cutoff | Whether those units appeared early or were surrounded by unrelated text |
| Evidence-token Precision@3/@5 | The proportion of returned tokens inside verified evidence | Whether all required units were found or ordered early |
| alpha-nDCG@3/@5 | The order in which distinct evidence-unit IDs first appear | Whether repeated relevant evidence actually harms the downstream answer |
| Pool Recall@20 | Evidence available to a reranker before reordering | Whether the final top-three or top-five order is good |

Together, the primary metrics answer ordering, completeness, and concentration.
The diagnostics then explain whether a weakness came from a limited first-stage
pool, repeated evidence, or unstable execution.

### Operational measurements

#### Warm latency

After warmup, each query is executed for every configured measured repetition.
Full-stack latency contains all work needed by the candidate. For a reranked
candidate it includes live first-stage retrieval, reranker tokenization and
inference, deterministic score ordering, and top-result selection.

First-stage latency isolates Dense, BM25, or RRF work. Reranker latency isolates
only learned reranking and is omitted for the no-reranker option. Components are
measured inside the same live execution; cached quality pools cannot make the
operational path look faster.

For each latency family, the median repetition for a query is the query-level
observation. p50 describes a typical warm query and p95 describes the slow tail.
p99 is reserved for the vector-server load benchmark, where enough controlled
requests exist to estimate it reliably.

#### Cold initialization and peak resources

Cold initialization uses a fresh worker and ends when the complete candidate is
ready to accept a query. It includes loading the required retriever state,
tokenizers, and reranker when applicable; downloads and environment installation
remain outside the benchmark.

Peak RAM is the largest sampled resident-memory total across the worker and its
child processes. Peak VRAM uses process-attributed GPU sampling. A confirmed
CPU-only run may report zero VRAM; missing GPU instrumentation is reported as
unavailable, not converted to zero.

#### Index build and storage

Index-build time begins after required models are loaded and ends when the
first-stage searchable state is ready. Index bytes are measured from that state.
The Dense and BM25 no-reranker owners record their own build time and index
bytes; their four reranker children reference those artifacts instead of copying
the values. RRF references both checksummed indexes and builds no third search
index. Its required bytes are the unique sum of the Dense and BM25 indexes, and
its incremental fusion-index bytes are zero.

### Workload descriptors

Corpus, question, chunk, and pool counts verify what the run processed. Mean and
p95 retrieved tokens at @3 and @5 use canonical source text and the fixed
evaluation tokenizer. They describe how much text each final ranking would pass
downstream without imposing a token budget.

Reranker input-token summaries use that reranker's native tokenizer over all 20
query-passage inputs. For each query, the token counts of its individual pairs
are summed first; mean and p95 are then calculated across those per-query totals.
The individual pairs are not treated as independent workload observations.

For example, queries requiring `2,000`, `2,400`, and `4,000` reranker input
tokens have a mean workload of `2,800` tokens per query. Averaging the lengths
of all individual pairs would instead describe the average pair and could hide
the expensive `4,000`-token request. Reranker input-token summaries are omitted
for no-reranker candidates. Snapshot bytes describe local model storage and are
recorded from the immutable model cache.

### Validity gates and eligibility

An authoritative child is valid only when every expected query is processed,
all ranking scores are finite, input truncation is zero, repeated rankings agree,
and each reranker output is an exact permutation of the referenced checksummed
pool. A failed query, pool mismatch, duplicate/missing chunk ID, nonfinite score,
or truncated input makes the child failed and the parent comparison incomplete.
There is no average failure-rate metric that can hide these errors.

Quality metrics include answerable questions only. Every overall result and
text, table, formula, or mixed slice records eligible question and document
counts. Alpha-nDCG separately records its smaller multi-evidence eligibility
counts. Inapplicable fields are absent rather than filled with zero.

### Aggregation and confidence intervals

Quality is calculated per eligible question, averaged within each source
document, and macro-averaged across documents. This prevents a paper with many
questions from dominating the result. Development and validation runs use 10,000
bootstrap resamples of complete documents with seed 42; the 2.5th and 97.5th
percentiles form the 95% confidence interval. Evidence slices repeat the same
calculation over contributing documents.

Smoke values receive no authoritative interval. One-off cold initialization,
index build, index bytes, peak resources, and workload descriptors receive point
observations but no invented interval. Warm latency intervals are reported only
when enough independent query observations support them.

## Vector-server correctness and performance

The NumPy exact cosine search result is the oracle for ANN quality; it is not a
production candidate. Results are reported separately by `K`, filter
selectivity, concurrency, vector dimension, and workload profile rather than
pooling unlike conditions into one score.

### Metric summary

#### Search and filter quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| ANN Recall@3/@5/@10 | Primary | How many exact nearest neighbours does the approximate server preserve at application-relevant depths? | Higher |
| Filtered ANN Recall@3/@5/@10 | Primary | Does the server preserve exact neighbours after applying the required metadata filter? | Higher |
| Filter Correctness | Validity gate | Do all returned records satisfy every requested predicate? | Must equal 1.0 |
| Empty-Filter Correctness | Validity gate | Does a filter with no valid match return an empty result? | Must equal 1.0 |
| ANN and Filtered ANN Recall@1 | Secondary | Does the server preserve the single nearest result? | Higher |
| Replacement, Deletion, Persistence, and ANN-Index Correctness | Validity gate | Does the server preserve required state semantics and actually use the configured ANN index? | Must pass |

#### Performance and resource measurements

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Unfiltered/Filtered Latency p50 | Diagnostic | What does a typical successful request take? | Lower |
| Unfiltered/Filtered Latency p95/p99 | Operational | How slow is the warm request tail under each concurrency? | Lower |
| Query Throughput | Operational | How many requests complete successfully per wall-clock second? | Higher |
| Request Error Rate | Operational | What share of submitted requests fail? | Lower |
| Build Time and Build Throughput | Operational | How long does initial indexing take and how many vectors are indexed per second? | Lower / Higher |
| Incremental Upsert/Delete Throughput | Operational | How quickly can the ready server apply each mutation workload? | Higher |
| Restart Readiness | Operational | How long until persisted state is queryable after restart? | Lower |
| First-Query Latency after Restart | Diagnostic | How slow is the first successful query after readiness? | Lower |
| Peak Server RAM and Persistent Storage | Operational | What memory and disk footprint does the prepared server require? | Lower at equal correctness and quality |

### Search and filter quality

#### ANN Recall@K

**Question:** How faithfully does approximate search preserve exact nearest
neighbours?

For each query, NumPy ranks the complete frozen vector corpus by exact cosine
similarity. ANN Recall compares the server's returned IDs with the first `K`
oracle IDs. Order inside the returned set does not change this metric; the
question is whether the exact neighbours remain available. If the corpus has
fewer than `K` eligible records, the denominator is the number the oracle can
actually return. Duplicate returned IDs invalidate the request rather than
earning repeated credit.

A failed server request is recorded with zero recall and also increments Request
Error Rate, so failures cannot disappear through eligibility filtering. Recall
is reported independently at `K=1`, `3`, `5`, and `10`; the `@1` value is
secondary and the other three are primary.

**Range and direction:** `[0, 1]`; higher is better.

#### Filtered ANN Recall@K

**Question:** Does approximate search remain faithful after metadata filtering?

The evaluator applies exactly the same frozen predicate to the oracle corpus and
the server request, then compares their IDs as above. Results remain separated
by filter-selectivity band. A query whose exact filtered result is empty is not
eligible for Filtered ANN Recall; it is evaluated by Empty-Filter Correctness.
A failed non-empty filtered request receives zero recall and increments Request
Error Rate.

**Range and direction:** `[0, 1]`; higher is better.

#### Filter Correctness

**Question:** Does every returned record obey the full requested predicate?

A filtered request receives `1` only when every returned record satisfies every
part of the predicate, including conjunctions. It receives `0` when any returned
record violates a predicate or the request fails. Returning too few otherwise
valid records does not reduce this metric because Filtered ANN Recall already
measures missing neighbours.

The aggregate is the mean of the request-level pass values, reported separately
for each filter-selectivity band.

**Range and direction:** `[0, 1]`; higher is better and `1.0` is required for a
conformant server.

#### Empty-Filter Correctness

**Question:** Does the server correctly return nothing when no record matches?

Each verified-empty predicate receives `1` only when the request succeeds and
returns no records. A non-empty response or request failure receives `0`. The
aggregate is the mean over verified-empty requests.

**Range and direction:** `[0, 1]`; `1.0` is required.

### Conformance validity gates

Replacement checks require an upserted ID to expose only its new vector and
metadata. Deletion checks require removed IDs and complete removed documents to
be absent from later search and filtering. Persistence checks require committed
records and the configured ANN index to remain available after restart. Health,
cosine behavior, wrong-dimension rejection, compound filters, and real ANN-index
use are binary gates under the same rule: every required check must pass.

A failed gate makes the server profile non-conformant. Performance measurements
may remain available for diagnosis, but the profile cannot be selected by
trading a correctness failure against speed.

### Performance and resources

Warm latency starts before client serialization and ends after the complete
loopback response is decoded. Only successful requests have a latency value;
failures remain visible through Request Error Rate. p50 describes the typical
request, while p95 and p99 describe the tail. p99 is reported only for workload
cells with enough submitted requests to estimate it; otherwise it is null with
an `insufficient_requests` status.

Query Throughput counts successful responses over the complete measured wall
time at each concurrency. Request Error Rate uses every submitted request as its
denominator. These two values are always read together: failed traffic cannot
make throughput look successful.

Build Time starts when a ready empty server receives the first vector and ends
when all submitted vectors are queryable. Build Throughput uses successfully
indexed vectors over that same elapsed time. Incremental upsert and delete
throughput are measured separately on a ready populated index and include the
time until each mutation is visible to queries.

Restart Readiness starts when restart is requested and ends when health checks
pass and the persisted ANN index answers its verification query. First-Query
Latency times the first successful query after readiness and is not mixed into
warm latency percentiles.

Peak server RAM is the largest sampled resident-memory total for server
processes or containers during the measured phase. Persistent Storage is the
on-disk server state after synchronization and before teardown. Client resource
measurements are labeled separately and are never added to server peaks.

### Eligibility, aggregation, and confidence intervals

Search-quality metrics are calculated once per frozen query and then averaged
within each workload cell. Filtered results are also averaged independently per
selectivity band. Query identities and conditions remain aligned across servers.
Development and validation quality intervals use 10,000 bootstrap resamples of complete
query IDs with seed 42. A resampled query carries all of its compared server
results and filter conditions.

Latency percentiles and their intervals use successful request observations
within one fixed workload cell. Build, mutation, restart, resource, and storage
values are observed phase-level measurements and receive no fabricated
confidence interval. Every table reports submitted, successful, failed, and
eligible request counts. Null means a defined eligibility condition was not met;
it never means zero.

## Generation and final-answer quality

This section covers generation on frozen evidence and the same automated metrics
when generation is embedded in Final RAG. Blinded human judgments remain the
authority for claim-level faithfulness and answer quality; automated metrics
answer narrower, reproducible questions.

### Metric summary

#### Automated quality and validity

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Citation Precision/Recall/F1 | Primary | On answerable questions, are citations correct and do they cover the required evidence? | Higher |
| Answerability Balanced Accuracy | Primary | Does the model distinguish answerable from unanswerable questions without the majority class dominating? | Higher |
| Unsupported Answer Rate | Primary | How often does the model give a substantive answer to an unanswerable question? | Lower |
| Malformed Output Rate | Primary | How often does a completed response violate the required answer/citation schema? | Lower |
| Token F1 | Secondary | How much accepted answer content is recovered even when wording differs? | Higher |
| Exact Match and ROUGE-L | Diagnostic | How often is wording exact, and how similar is its sequence to an accepted answer? | Higher |
| Refusal Precision/Recall/F1 | Diagnostic | Are refusals reserved for unanswerable questions, and are those questions actually refused? | Higher |
| HHEM Faithfulness | Diagnostic | Does a pinned local model judge a substantive answer supported by the supplied evidence? | Higher |
| Repeat Output Agreement | Diagnostic | Does deterministic generation return the same visible answer and citations? | Higher |

#### Operational measurements and descriptors

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Time to First Token | Operational | How long does a warm request wait before generation begins? | Lower |
| Prompt-Evaluation Time | Operational | How long is spent processing the prompt before decoding? | Lower |
| Generation Time and Generated Tokens/Second | Operational | How long does decoding take and at what observed rate? | Lower / Higher |
| Total Response Latency p50/p95 | Operational | How long does the complete warm generation request usually take, and how slow is its tail? | Lower |
| Cold Model-Load Time | Operational | How long does a fresh worker need to make the generator ready? | Lower |
| Peak Process-Tree RAM and Peak VRAM | Operational | What host and device memory does the profile require? | Lower at equal quality |
| Prompt, Visible-Answer, Reasoning, and Generated Token Counts | Workload descriptor | How much native-tokenizer input and output produced the quality and latency results? | Descriptive |

### Citation quality

Citation Precision, Recall, and F1 are calculated only for answerable questions,
which always have one or more verified gold evidence units. Unanswerable
questions are handled by answerability and refusal metrics rather than receiving
artificially perfect citation scores.

A citation is automatically supported only when it is a valid identifier for a
supplied numbered evidence block and that block completely covers at least one
required gold evidence unit. Text similarity alone does not make a citation
correct. Repeated references to the same block count once. An unknown citation
ID counts as produced but unsupported and also makes the response malformed.

- **Citation Precision** asks what share of the distinct citations produced are
  supported. An answerable response with no citations receives zero.
- **Citation Recall** asks what share of the distinct required gold evidence
  units are covered by at least one supported citation. A cited block can cover
  more than one unit when its frozen intervals genuinely contain them.
- **Citation F1** balances those two results. It is zero when either no required
  evidence is cited or no produced citation is supported.

All three lie in `[0, 1]`; higher is better. A malformed answerable response
receives zero for all three so protocol failures are not removed from the
quality denominator.

### Answerability, refusal, and output validity

A response is a **refusal** only when it uses the frozen refusal representation
and contains no substantive answer. Every other completed response is a
substantive answer. Answerability Balanced Accuracy averages the recall of the
answerable class and the unanswerable class; both classes must occur in an
authoritative split. It therefore cannot be inflated by always choosing the
larger class.

Unsupported Answer Rate is the share of unanswerable questions that receive a
substantive answer. It does not claim to measure whether every statement in an
answerable response is faithful; that broader question belongs to blinded human
review, with HHEM retained only as a diagnostic.

Refusal Precision asks what share of refusals were issued for unanswerable
questions. Refusal Recall asks what share of unanswerable questions were
refused. Refusal F1 balances them. A profile that never refuses receives zero
precision, recall, and F1 when the split contains unanswerable questions.

Malformed Output Rate includes responses that cannot be parsed into the required
answer/citation schema, use unknown citation IDs, mix the refusal marker with a
substantive answer, or omit required fields. A completed but malformed response
is still a scored sample. An inference crash is instead a validity failure: the
child fails and the parent comparison is incomplete rather than averaging only
the surviving questions.

### Accepted-answer similarity

Exact Match, Token F1, and ROUGE-L apply to answerable questions. A refusal or
malformed response to an answerable question receives zero. When several
accepted answers exist, the highest score across those references is used.

- **Exact Match** requires the normalized predicted answer to equal an accepted
  answer exactly.
- **Token F1** uses repeated normalized token occurrences, so it rewards partial
  recovery without treating repeated words as a set.
- **ROUGE-L** rewards an in-order common token sequence and can distinguish two
  answers with similar words but different ordering.

Unanswerable questions are ineligible for these three metrics because they have
no accepted substantive answer. Exact Match and ROUGE-L are diagnostic; Token
F1 is the secondary automated answer-correctness measure.

### Automated faithfulness and repeatability

HHEM scores every parsable substantive answer against exactly the supplied
evidence blocks. It is omitted when a candidate produces no eligible
substantive answers. The pinned HHEM checkpoint, revision, prompt construction,
and score direction are recorded. Its result never replaces human Faithfulness.

For Repeat Output Agreement, the first measured response is designated and each
later repetition agrees only when its normalized visible answer and ordered
distinct citation IDs are identical. Hidden reasoning text does not affect the
agreement value but its token count remains a workload descriptor. A failed
repetition is a disagreement and also fails the candidate's completeness gate.
Smoke with one measured response has no meaningful agreement value.

### Operational measurement

Time to First Token starts immediately before the warm generation call and ends
when the first generated token is available. Prompt-Evaluation Time uses the
runtime's measured prefill interval when exposed; otherwise it is null with an
`unsupported_by_runtime` status rather than inferred by subtraction.
Generation Time runs from the first generated token through completion, and
Generated Tokens/Second uses all generated native-tokenizer tokens over that
interval. Both are null if no token is generated.

Total Response Latency covers prompt preparation, tokenization, model prefill,
reasoning and visible-answer decoding, output parsing, and citation validation.
For Final RAG, separate server-call, retrieval/reranking, context-packing, and
generation timings are also reported; end-to-end latency contains all of them.
The median repetition is the per-question warm observation used for p50 and p95.

Cold Model-Load Time is measured once in a fresh worker before warmup. Peak RAM
includes the worker process tree; Peak VRAM uses process-attributed device
measurement and must be non-zero for an authoritative CUDA child. Prompt,
reasoning, visible-answer, and total generated token counts use each generator's
native tokenizer and are reported per question before mean and p95 summaries.
Unavailable separate reasoning counts remain null rather than being estimated.

### Human review metrics

For Final RAG, one blinded reviewer scores Faithfulness, Answer Correctness,
Completeness, and Citation Accuracy on the frozen `0` to `2` rubric, plus
Answerability Correctness on `0` or `1`. A score of `0` means the requirement is
not met, `1` means partly met, and `2` means fully met. The reviewer sees the
same question, accepted answer, and supplied evidence for each anonymous system.
These values are reported separately; they are never averaged into a universal
quality score. A single reviewer does not support an inter-reviewer agreement
claim.

### Eligibility, aggregation, and confidence intervals

Automated values are first calculated per question, averaged within each source
document, and macro-averaged across documents. This prevents a paper with many
questions from dominating. Development and validation runs use 10,000 bootstrap
resamples of complete documents with seed 42. Answerable-only, unanswerable-only,
substantive-answer-only, and evidence-type results always report their eligible
question and document counts.

Warm latency and token-workload summaries use the same fixed question set and
report their observation counts. Cold load and observed RAM/VRAM peaks are
single-run measurements and receive no fabricated interval. Null is reserved
for explicitly ineligible or runtime-unsupported values. A missing required
sample, artifact, or metric is a failed child, not a null score.

## Aggregation and interpretation

Aggregate metrics never replace sample rows. Eligible development, validation, and locked sample-based
metrics report the mean (or named percentile), a 95% interval, the number of
contributing samples, and failures. Conditional metrics such as table structure
or timestamps also report their sample count. Smoke values, counts, statuses,
fixed identifiers, and single operational observations do not have authoritative
intervals. An engineer reviews the complete evidence; EduMind does not combine
unrelated metrics into a weighted overall score or promote a candidate
automatically.
