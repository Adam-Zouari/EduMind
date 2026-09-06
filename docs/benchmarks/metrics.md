# Benchmark metric reference

[Benchmark overview](overview.md) · [Experiment sequence and rationale](methodology.md) ·
[Benchmark runbook](running.md)

This page defines EduMind's approved benchmark metric contracts. The
[methodology](methodology.md) says **where** each metric is used and why; this
page says **what the value means and how it is calculated**. A standard or full
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
- Standard, full, and locked runs retain one row per sample before aggregation.
- p50, p95, and p99 are latency percentiles. Throughput is completed operations
  divided by measured wall-clock time.
- Eligible standard, full, and locked sample-based aggregates use 10,000 bootstrap resamples
  with seed 42 and 95% confidence intervals. Paired comparisons resample aligned
  samples, not unrelated aggregate values. Counts, statuses, fixed identifiers,
  and single operational observations do not receive intervals.
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

The benchmark uses the pinned official evaluator rather than a custom
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
| Standard, full, and locked text, page, layout, table, and formula aggregates | Yes | Calculated from the contributing independent samples. |
| Standard, full, and locked reliability rates | Yes | Calculated across scheduled independent samples. |
| Standard, full, and locked p50/p95 latency | Conditional | Reported when enough independent document or page observations support the percentile estimate. |
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

Eligible standard, full, and locked sample-based metrics use 10,000 bootstrap resamples with
seed 42:

1. Treat each source document as the independent resampling unit.
2. Resample documents with replacement.
3. Recalculate the aggregate for every resample.
4. Use the 2.5th and 97.5th percentiles as the 95% interval bounds.

Document-group metrics resample only the samples in that group. Conditional
metrics such as table structure or formula recognition resample only the samples
eligible for that metric. Paired candidate comparisons resample aligned sample
IDs together so both candidates are evaluated on the same resampled cases.
If a pooled detection resample contains neither a reference nor a prediction,
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
interval means the aggregate is estimated more precisely. Overlapping intervals
do not by themselves prove that candidates are equal, and non-overlapping
intervals do not replace an aligned paired comparison when a formal difference
claim is made.

## Audio extraction

The ASR benchmark evaluates the complete ordered transcript, its timestamps,
catastrophic output behavior, and the cost of the recorded runtime profile. It
does not use Content F1 or Transcript Order Accuracy: unlike a two-dimensional
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

After the stage applies one fixed one-to-one segment-alignment rule, Timestamp
Boundary MAE is calculated as follows:

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
total transcription-and-alignment time
--------------------------------------
          total audio duration
```

**Example:** Processing 60 minutes of audio in 15 minutes produces RTF `15 / 60
= 0.25`.

An RTF below `1` means processing is faster than audio playback. Qwen's complete
time includes transcription, unloading, loading the forced aligner, and
alignment.

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

Measure from the start of model construction until loading completes, before
warmups or transcription. Qwen reports the complete required loading policy;
its forced-aligner reload remains part of complete-pipeline latency rather than
being hidden in this first load value.

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
| Standard, full, and locked WER, CER, and substitution/deletion/insertion rates | Yes | Resample complete speech clips and recalculate the pooled counts. |
| Standard, full, and locked Timestamp Alignment Coverage | Yes | Resample complete timed speech clips. |
| Standard, full, and locked Timestamp Boundary MAE | Yes, when defined | Use resampled clips containing valid aligned boundaries. |
| Standard, full, and locked Empty Transcript Rate | Yes | Resample speech clips. |
| Standard, full, and locked Nonspeech False-Transcription Rate | Yes | Resample the separate nonspeech controls. |
| Standard, full, and locked Repeat Transcript Agreement Rate | Yes | Resample complete speech clips with their already-computed agreement flags. |
| Standard, full, and locked Real-Time Factor | Yes | Resample complete speech clips with their measured processing time and duration. |
| Standard, full, and locked p50/p95 warm latency | Conditional | Report only when enough independent clips support the percentile estimate. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| One cold-load measurement | No | One observation cannot estimate uncertainty. |
| One observed peak RAM or VRAM value | No | Report the observed peak without invented bounds. |

#### Calculation

Standard, full, and locked runs use 10,000 bootstrap resamples with seed 42:

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
| Mean Selected Frames per Video | Cost diagnostic | How many frames does the configuration send to the document parser on average? | Lower only when quality is preserved |

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
| Standard, full, and locked Visual Content Precision/Recall/F1 | Yes | Resample complete videos and recalculate each aggregate. |
| Standard, full, and locked Timed Visual Occurrence Coverage | Yes | Resample complete videos with their timed visible references. |
| Standard, full, and locked Mean Visual First-Detection Delay | Yes, when defined | Use resampled videos containing covered timed occurrences. |
| Standard, full, and locked Duplicate Visual Text Rate | Yes | Resample complete videos with their duplicate counts. |
| Standard, full, and locked Frozen-ASR Transcript WER | Yes | Resample the shared per-video ASR outputs; report the result on the frozen-ASR child run. |
| Standard, full, and locked Visual Real-Time Factor and Mean Selected Frames per Video | Yes | Resample complete videos with their duration, visual-processing time, and frame counts. |
| Standard, full, and locked p50/p95 warm latency | Conditional | Report only when enough independent videos support the percentile estimate. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| One cold-load measurement | No | One observation cannot estimate uncertainty. |
| One observed peak RAM or VRAM value | No | Report the observed peak without invented bounds. |

#### Calculation

Standard, full, and locked runs use 10,000 bootstrap resamples with seed 42:

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

## Retrieval quality

For chunk interval `[c_start, c_end)` and evidence interval
`[e_start, e_end)`, overlap is:

```python
max(0, min(c_end, e_end) - max(c_start, e_start))
```

Retrieved intervals are merged before coverage is counted, so overlapping
chunks cannot receive duplicate credit.

| Metric | Definition | Direction |
|---|---|---|
| Context Recall@K | Unique gold-evidence characters covered by the first K chunks divided by unique gold-evidence characters. | Higher |
| Context Precision@K | Rank-aware average precision of evidence-bearing chunks in the first K results. | Higher |
| Context Recall under 2,048 tokens | Evidence coverage after ranked chunks are packed without exceeding the token budget. | Higher |
| Precision@K | Evidence-bearing results in the first K divided by K. | Higher |
| Recall@K | Relevant results returned in the first K divided by all relevant results. | Higher |
| Hit Rate@K | 1 when at least one of the first K results is relevant; otherwise 0. | Higher |
| MAP@K | Mean of per-query average precision through rank K. | Higher |
| MRR | Mean reciprocal rank of the first relevant result. | Higher |

Each chunk's graded relevance is its covered gold-evidence length divided by its
own length, capped at 1. Discounted cumulative gain is:

```text
DCG@K = sum((2^grade_i - 1) / log2(i + 1), i=1..K)
nDCG@K = observed DCG@K / ideal DCG@K
```

`nDCG@3` and `nDCG@5` reward putting highly relevant evidence early. Context
recall measures evidence coverage, while context precision penalizes wasting
limited context on irrelevant chunks.

## Vector-server correctness and performance

The NumPy exact-neighbor result is the oracle for ANN metrics; it is not a
production candidate.

| Metric | Definition | Direction |
|---|---|---|
| ANN Recall@K | Size of the intersection between approximate and exact top-K IDs divided by the number of exact IDs available through K. | Higher |
| Filtered ANN Recall@K | The same calculation after applying the identical metadata predicate to the exact oracle and server query. | Higher |
| Filter Correctness | Per query, 1 only when every returned row satisfies every tested predicate; aggregate output is the fraction of queries passing completely. | Higher |
| Empty-Filter Correctness | 1 when a predicate with no matching records returns no records; otherwise 0. | Higher |
| Replacement/Deletion/Persistence Correctness | Binary checks that replacement removes stale chunks, deletion removes all target records, and records survive a server restart. | Higher |
| Error Rate | Failed requests divided by submitted requests. | Lower |
| Throughput | Successful requests or ingested vectors divided by wall-clock seconds. | Higher |

Latency includes client serialization and loopback transport because both are
part of the user-visible server path. Resource results identify client and
server measurements separately.

## Generation and final-answer quality

Human reviewers score Faithfulness, Answer Correctness, Completeness, and
Citation Accuracy from 0 to 2 using the blinded review rubric. These judgments
are authoritative; automated text scores are diagnostics.

| Metric | Definition | Direction |
|---|---|---|
| Citation Precision | Distinct supported citations divided by distinct citations produced. | Higher |
| Citation Recall | Distinct supported evidence items cited divided by supported evidence items available. When no support exists, recall is 1 only if no citation is produced. | Higher |
| Citation F1 | Harmonic mean of citation precision and recall. | Higher |
| Answerability Balanced Accuracy | Mean recall across the answerable and unanswerable classes that occur in the evaluated set. | Higher |
| Exact Match | 1 when normalized answer tokens exactly equal an accepted answer; otherwise 0. | Higher |
| Token F1 | Multiset token overlap F1 between prediction and accepted answer. The best score over accepted references is used. | Higher |
| ROUGE-L | F1 derived from the longest common token subsequence. The best score over accepted references is used. | Higher |
| Refusal Precision/Recall/F1 | Classification metrics for refusing unanswerable questions. | Higher |
| Unsupported Answer Rate | Fraction of answers asserting content that is unsupported by the supplied evidence. | Lower |
| Malformed Output Rate | Fraction that violates the required answer/citation format. | Lower |
| NLI/HHEM Faithfulness | Pinned local factual-consistency model score used only as an automated diagnostic. | Higher |

Operational generation metrics are time to first generated token, total
response latency, prompt-evaluation time, generated tokens per second, token
counts, cold load time, and peak RAM/VRAM. End-to-end Final RAG latency includes
retrieval, reranking, context packing, prompting, and generation.

## Aggregation and interpretation

Aggregate metrics never replace sample rows. Eligible standard, full, and locked sample-based
metrics report the mean (or named percentile), a 95% interval, the number of
contributing samples, and failures. Conditional metrics such as table structure
or timestamps also report their sample count. Smoke values, counts, statuses,
fixed identifiers, and single operational observations do not have authoritative
intervals. An engineer reviews the complete evidence; EduMind does not combine
unrelated metrics into a weighted overall score or promote a candidate
automatically.
