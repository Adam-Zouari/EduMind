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

- Content, Exact Match, Token F1, and ROUGE-L use case-folded `\w+` tokens.
  CER operates on the strings supplied by the stage, and WER splits those
  strings on whitespace. An evaluator therefore applies its fixed canonical
  representation rules before scoring rather than hiding them inside these two
  metrics.
- Source and evidence spans are half-open intervals: `[start, end)`.
- Empty denominators use the explicit behavior stated below; they never produce
  fabricated zero-quality observations.
- Standard and full runs retain one row per sample before aggregation.
- p50, p95, and p99 are latency percentiles. Throughput is completed operations
  divided by measured wall-clock time.
- Eligible standard/full sample-based aggregates use 10,000 bootstrap resamples
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
outcomes. The remaining metrics are **supporting metrics**: they explain the
primary results or expose a narrower failure mode. Primary status does not assign
weights, combine categories into an overall score, or select a candidate
automatically.

Each metric now has one self-contained subsection. Its question, equation,
plain-language calculation, example, interpretation, valid range, and preferred
direction are kept together. Category introductions contain only rules shared by
multiple metrics, such as the element-matching protocol.

### Metric summary

The following tables provide the complete metric list for readers who only need
to know what the document-extraction benchmark measures. The detailed contracts
and examples follow the summary.

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
| Table Content F1 | Was the textual content inside tables recovered? |
| Table Structure Score (TEDS-S) | Were rows, columns, headers, merged cells, and spans reconstructed correctly? |

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

This section replaces the legacy exact-line, adjacency-table, and raw-LaTeX
scorers. Until the runner is aligned with this contract, document smoke runs may
validate wiring, but their metrics cannot support an authoritative parser
comparison.

All candidates receive the same canonical reference and output conversion.
This evaluation-only conversion handles representational equivalence; it does
not repair words, remove page content, deduplicate text, or otherwise clean a
candidate's extraction.
Each metric is reported as a total across the documents on which it is defined
and separately for applicable document groups such as `image`, `pdf_scanned`,
and `docx`. A conditional metric is absent, not zero, when its required
annotation does not apply. Every sample-based aggregate records `sample_count`
as reporting metadata. Qualifying aggregates also record a confidence interval
under the separate policy below. The benchmark plan records the total scheduled
samples.

Text error and content metrics are calculated per eligible document and then
macro-averaged, so a long PDF cannot dominate every short document. Detection
precision, recall, and F1 pool `TP`, `FP`, and `FN` across the annotated samples
in the reported group. Table-content, table-structure, and formula-recognition
scores macro-average their eligible reference objects, assigning zero to a
missed reference object. Bootstrap resampling always uses the document as the
independent unit and recalculates the complete aggregate from that resample.

### Text content and recognition

Repeated token occurrences are counted; token sets are not used.

**Primary metrics:** Content F1 and Reading Order Accuracy.

- **Content F1** is primary because it summarizes whether the parser recovered
  the required text without adding unsupported text. It balances content
  precision and recall in one category-level measure.
- **Reading Order Accuracy** is also primary because correct words can still be
  unusable when headings, columns, paragraphs, or list items are returned in the
  wrong sequence. It measures a different outcome from Content F1.
- **Content Precision and Content Recall** are supporting metrics. They separate
  hallucinated or extra content from missing content and therefore explain why
  Content F1 changed.
- **CER and WER** are supporting error diagnostics. They reveal character-level
  and word-level recognition mistakes, but neither alone measures both content
  completeness and unsupported output as directly as Content F1.

#### Content Precision

**Question:** How much extracted content is supported by the reference?

For reference token counts $c_r(t)$ and predicted token counts $c_p(t)$, the
number of matched token occurrences is:

$$
TP_{\text{token}}
= \sum_{t \in V}\min\!\left(c_r(t),c_p(t)\right)
$$

$V$ is the set of distinct tokens appearing in either text and $m$ is the
number of predicted tokens. Content Precision is:

$$
P_{\text{content}}=\frac{TP_{\text{token}}}{m}
$$

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

Let $V$ be the tokens appearing in either text, $c_r(t)$ and $c_p(t)$ their
reference and prediction occurrence counts, and $n$ the number of reference
tokens:

$$
TP_{\text{token}}
=\sum_{t\in V}\min\!\left(c_r(t),c_p(t)\right)
$$

$$
R_{\text{content}}=\frac{TP_{\text{token}}}{n}
$$

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

Let $TP_{\text{token}}$ be multiset token overlap, so repeated occurrences are
counted up to the smaller reference/prediction count. Let $m$ and $n$ be the
prediction and reference token counts:

$$
TP_{\text{token}}
=\sum_{t\in V}\min\!\left(c_r(t),c_p(t)\right)
$$

$$
P_{\text{content}}=\frac{TP_{\text{token}}}{m},
\qquad
R_{\text{content}}=\frac{TP_{\text{token}}}{n}
$$

$$
F1_{\text{content}}
=\frac{2P_{\text{content}}R_{\text{content}}}
       {P_{\text{content}}+R_{\text{content}}}
$$

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

Let $S_{char}$, $D_{char}$, and $I_{char}$ be the minimum Levenshtein
substitutions, deletions, and insertions needed to transform the reference into
the prediction:

$$
CER=\frac{S_{char}+D_{char}+I_{char}}
         {N_{\text{reference characters}}}
$$

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

CER exposes errors such as `0` instead of `O`, `rn` instead of `m`, missing
punctuation, and misspelled technical terms. A perfect result has `CER = 0`.

**Range and direction:** `[0, infinity)`; lower is better. CER can exceed `1`
when insertions outnumber reference characters.

#### Word Error Rate (WER)

**Question:** How severe are complete-word errors?

Let $S_{word}$, $D_{word}$, and $I_{word}$ be the minimum word-level
substitutions, deletions, and insertions:

$$
WER=\frac{S_{word}+D_{word}+I_{word}}
         {N_{\text{reference words}}}
$$

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

Let $L$ be the number of one-to-one matched document elements and $C$ the
number of their unordered pairs whose relative order agrees in the reference
and prediction:

$$
ROA=\frac{C}{\binom{L}{2}}
   =\frac{2C}{L(L-1)},
\qquad L\ge2
$$

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

Let $G$ be the set of reference content-bearing pages and $TP_j$ the number of
matched content units on reference page $j$:

$$
\operatorname{PageCoverage}
= \frac{\displaystyle\sum_{j \in G}
          \mathbf{1}\!\left[TP_j > 0\right]}
       {|G|}
$$

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

Let $G$ and $P$ be the reference and predicted page-ID sets. $F1_j$ is Content
F1 calculated using only the reference and predicted content assigned to page
$j$:

$$
\operatorname{PageContentF1}
= \frac{1}{|G \cup P|}
  \sum_{j \in G \cup P}F1_j
$$

Using $G\cup P$ includes missing reference pages and unexpected predicted pages,
both of which receive zero.

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

Let $A$ be the set of matched content elements with reference page annotations:

$$
\operatorname{PageAttributionAccuracy}
= \frac{\displaystyle\sum_{a \in A}
          \mathbf{1}\!\left[
          \operatorname{page}_{pred}(a)=\operatorname{page}_{ref}(a)
          \right]}
       {|A|}
$$

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

Let $D_{page}$ be unsupported duplicate predicted pages under the fixed
near-duplicate rule:

$$
\operatorname{DuplicatePageRate}
= \frac{D_{page}}{N_{\text{predicted page records}}}
$$

In plain language:

```text
unsupported duplicated page records
────────────────────────────────────
       predicted page records
```

**Example:**

```text
Predicted pages:      1, 2, 2, 3
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
- **Layout Element Precision and Recall** are supporting metrics. They distinguish
  extra detected elements from missed elements and explain Layout Element F1.
- **Mean Bounding-Box IoU** is supporting because it diagnoses localization
  quality after elements have been matched. Precise boxes are useful, but they do
  not by themselves prove that the correct elements, types, or hierarchy were
  recovered.

Reference and predicted elements are matched one-to-one. When boxes are
available, matching maximizes bounding-box Intersection over Union (IoU), with
`IoU >= 0.5` required for a match. When a corpus lacks boxes, its pinned official
element matcher is used instead; matching protocols are never mixed silently
inside one comparison.

#### Layout Element Precision

**Question:** How many predicted document elements are real?

$TP_e$ is the number of one-to-one matched elements and $FP_e$ the number of
extra predicted elements:

$$
P_e=\frac{TP_e}{TP_e+FP_e}
$$

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

$FN_e$ is the number of missed reference elements:

$$
R_e=\frac{TP_e}{TP_e+FN_e}
$$

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

For the pooled matched, extra, and missed element counts:

$$
P_e=\frac{TP_e}{TP_e+FP_e},
\qquad
R_e=\frac{TP_e}{TP_e+FN_e}
$$

$$
F1_e=\frac{2P_eR_e}{P_e+R_e}
$$

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

Let $M_e$ be the set of matched elements:

$$
\operatorname{TypeAccuracy}
=\frac{\displaystyle\sum_{e\in M_e}
        \mathbf{1}\!\left[
        \operatorname{type}_{pred}(e)=\operatorname{type}_{ref}(e)
        \right]}
       {|M_e|}
$$

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

Let $H_e$ be matched elements with hierarchy annotations:

$$
\operatorname{HierarchyAccuracy}
=\frac{\displaystyle\sum_{e\in H_e}
        \mathbf{1}\!\left[
        \operatorname{parent}_{pred}(e)=\operatorname{parent}_{ref}(e)
        \land
        \operatorname{level}_{pred}(e)=\operatorname{level}_{ref}(e)
        \right]}
       {|H_e|}
$$

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

For boxes $a$ and $b$:

$$
\operatorname{IoU}(a,b)
=\frac{\operatorname{area}(a\cap b)}
       {\operatorname{area}(a\cup b)}
$$

For the matched elements $B_e$ that have reference and predicted boxes:

$$
\operatorname{MeanIoU}
=\frac{1}{|B_e|}
  \sum_{e\in B_e}
  \operatorname{IoU}\!\left(b_e^{ref},b_e^{pred}\right)
$$

In plain language, IoU for one matched element is:

```text
area shared by reference and predicted boxes
───────────────────────────────────────────
 area covered by either of the two boxes
```

The benchmark averages this value over matched elements with boxes.

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
uses one-to-one table-region matching at `IoU >= 0.5`; crop-level datasets use
their explicit table identities. Results are reported overall and by table
attributes such as bordered/borderless and merged-cell presence when those
labels exist.

Table evaluation answers three separate questions: was the table found, was
its text recovered, and was its structure reconstructed?

**Primary metrics:** Table Detection F1, Table Content F1, and Table Structure
Score (TEDS-S).

- **Table Detection F1** is primary because it summarizes whether tables were
  found without producing false table regions.
- **Table Content F1** is primary because finding a table does not show whether
  the text inside its cells was recovered correctly.
- **TEDS-S** is primary because correct table text can still be assigned to the
  wrong rows, columns, headers, or spans. It measures structural reconstruction,
  which the other two primary metrics do not.
- **Table Detection Precision and Recall** are supporting metrics. They reveal
  whether a Detection F1 result is limited by false table regions or missed
  tables.

#### Table Detection Precision

**Question:** How many predicted tables are real tables?

Let $TP_t$ be one-to-one matched table regions and $FP_t$ extra predicted
tables:

$$
P_t=\frac{TP_t}{TP_t+FP_t}
$$

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

Let $FN_t$ be missed reference tables:

$$
R_t=\frac{TP_t}{TP_t+FN_t}
$$

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

For the pooled matched, extra, and missed table-region counts:

$$
P_t=\frac{TP_t}{TP_t+FP_t},
\qquad
R_t=\frac{TP_t}{TP_t+FN_t}
$$

$$
F1_t=\frac{2P_tR_t}{P_t+R_t}
$$

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

#### Table Content F1

**Question:** Was the textual content inside tables recovered?

Each of the $N_t$ reference tables is paired with its matched prediction. A
missed table is paired with an empty prediction:

$$
\operatorname{TableContentF1}
=\frac{1}{N_t}
  \sum_{j=1}^{N_t}
  \operatorname{ContentF1}(r_j,p_j)
$$

In plain language:

```text
sum of reference-table Content F1 scores
────────────────────────────────────────
         number of reference tables
```

It measures the words recovered from inside the tables. A missed reference
table receives zero.

**Example:** For table Content F1 values `1.00`, `0.80`, and `0.00` for a missed
table, the result is `(1.00 + 0.80 + 0.00) / 3 = 0.60`.

**Range and direction:** `[0, 1]`; higher is better.

#### Table Structure Score (TEDS-S)

**Question:** Were rows, columns, headers, merged cells, and spans reconstructed
correctly?

$T(r_j)$ and $T(p_j)$ are the reference and predicted HTML trees after cell
text is removed. $\operatorname{TED}$ is tree-edit distance and $|T|$ is the
number of nodes:

$$
\operatorname{TEDS\text{-}S}_j
=1-
 \frac{\operatorname{TED}\!\left(T(r_j),T(p_j)\right)}
      {\max\!\left(|T(r_j)|,|T(p_j)|\right)}
$$

The reported score is the mean over all reference tables:

$$
\operatorname{TableStructureScore}
=\frac{1}{N_t}
  \sum_{j=1}^{N_t}\operatorname{TEDS\text{-}S}_j
$$

In plain language:

```text
TEDS-S = 1 − normalized table-tree edit distance
```

The evaluator represents each table as a tree of rows, columns, headers, cells,
and spans. Cell text is removed before comparison, so this score focuses on
structure:

```text
TEDS-S = 1.0       → identical structure
TEDS-S near 1.0    → small structural differences
TEDS-S near 0.0    → severely incorrect structure
```

**Example:** A parser may find a table and recover every word but put the words
into the wrong columns. Detection Recall and Table Content F1 can remain high,
while TEDS-S exposes the structural failure.

**Range and direction:** `[0, 1]`; higher is better. A missed table receives
zero.

The benchmark uses the pinned official evaluator rather than EduMind's former
row/column-adjacency approximation. OmniDocBench documents TEDS and TEDS-S in its
[official evaluation repository](https://github.com/opendatalab/OmniDocBench).

### Formulas

Formula metrics are calculated only for samples with formula annotations and are
reported as a total and separately for inline and display formulas. Detection
uses one-to-one region matching at `IoU >= 0.5` when boxes are available.

**Primary metrics:** Formula Detection F1 and Formula Exact Match
(ExpRate@CDM).

- **Formula Detection F1** is primary because it summarizes whether formula
  regions were found without hallucinating extra formula regions.
- **ExpRate@CDM** is primary because mathematical meaning can change after one
  wrong symbol. It reports the proportion of reference formulas reconstructed
  perfectly under the CDM evaluator.
- **Formula Detection Precision and Recall** are supporting metrics. They expose
  whether Detection F1 is limited by false formula regions or missed formulas.
- **Formula Recognition Similarity (CDM)** is supporting because it shows how
  close imperfect recognitions are to the reference. It is valuable for error
  analysis, but a high average similarity can hide formulas with small,
  meaning-changing mistakes; ExpRate@CDM makes perfect reconstruction explicit.

#### Formula Detection Precision

**Question:** How many predicted formula regions are real formulas?

$TP_f$ is the number of matched formula regions and $FP_f$ extra predicted
formula regions:

$$
P_f=\frac{TP_f}{TP_f+FP_f}
$$

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

$FN_f$ is the number of missed reference formula regions:

$$
R_f=\frac{TP_f}{TP_f+FN_f}
$$

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

For the pooled matched, extra, and missed formula-region counts:

$$
P_f=\frac{TP_f}{TP_f+FP_f},
\qquad
R_f=\frac{TP_f}{TP_f+FN_f}
$$

$$
F1_f=\frac{2P_fR_f}{P_f+R_f}
$$

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

The official evaluator renders the reference and predicted formulas, then
matches their characters and positions. For formula $j$:

$$
\operatorname{CDM}(r_j,p_j)
=\frac{2TP_{cdm,j}}
       {2TP_{cdm,j}+FP_{cdm,j}+FN_{cdm,j}}
$$

$TP_{cdm,j}$ is valid matched character regions; $FP_{cdm,j}$ and $FN_{cdm,j}$
are extra predicted and missed reference character regions. The reported score
averages the $N_f$ reference formulas:

$$
\operatorname{MeanCDM}
=\frac{1}{N_f}
  \sum_{j=1}^{N_f}\operatorname{CDM}(r_j,p_j)
$$

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

$$
\operatorname{ExpRate@CDM}
=\frac{1}{N_f}
  \sum_{j=1}^{N_f}
  \mathbf{1}\!\left[\operatorname{CDM}(r_j,p_j)=1\right]
$$

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
home-grown LaTeX edit score. The CDM and ExpRate@CDM formulas come from the
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

$$
\operatorname{EmptyOutputRate}
= \frac{1}{N_{scheduled}}
  \sum_{i=1}^{N_{scheduled}}
  \mathbf{1}\!\left[\operatorname{output}_i=\varnothing\right]
$$

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

$$
\operatorname{DuplicateContentRate}
=\frac{N_{\text{unsupported duplicate units}}}
       {N_{\text{predicted content units}}}
$$

In plain language:

```text
unsupported repeated content units
──────────────────────────────────
       predicted content units
```

It exposes repeated paragraphs, page content, tables, or formulas that do not
represent legitimate repetition in the source.

**Example:** If 20 of 1,000 predicted content units are unsupported repetitions,
Duplicate Content Rate is `20 / 1,000 = 0.02`.

**Range and direction:** `[0, 1]`; lower is better. It is omitted when the
candidate produces no content; Empty Output Rate records that case.

#### Structured-output Determinism

**Question:** Does the same input produce the same complete structured result?

Let $N_{repeated}$ be samples deliberately executed more than once. The
fingerprint covers text, page attribution, element types and order, hierarchy,
tables, formulas, and normalized boxes; timing and random run IDs are excluded:

$$
\operatorname{Determinism}
=\frac{1}{N_{repeated}}
  \sum_{i=1}^{N_{repeated}}
  \mathbf{1}\!\left[
  |\operatorname{Fingerprints}(i)|=1
  \right]
$$

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

$$
\operatorname{CandidateFailureRate}
=\frac{1}{N_{scheduled}}
  \sum_{i=1}^{N_{scheduled}}
  \mathbf{1}\!\left[\text{sample }i\text{ ends in a fatal error}\right]
$$

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

Every scheduled sample remains visible. A recoverable failure is represented by
an explicit per-sample result and contributes to Empty Output or Candidate
Failure Rate; it is never dropped from quality aggregation. A failure that
prevents the required per-sample record makes the benchmark invocation
incomplete and therefore non-authoritative.

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

$$
T_{first}
= t_{\text{first extraction complete}}-t_{\text{fresh process start}}
$$

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

For document $i$, let $T_i$ be complete warm latency and $p_i$ its processed
page count:

$$
L_i=\frac{T_i}{p_i}
$$

$$
\operatorname{p50}_{page}=Q_{0.50}(L_1,\ldots,L_N),
\qquad
\operatorname{p95}_{page}=Q_{0.95}(L_1,\ldots,L_N)
$$

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

For complete document times $T_1,\ldots,T_N$:

$$
\operatorname{p50}_{document}=Q_{0.50}(T_1,\ldots,T_N),
\qquad
\operatorname{p95}_{document}=Q_{0.95}(T_1,\ldots,T_N)
$$

This is the end-to-end time for one whole source. Its p50 and p95 answer how long
a user typically waits and how long difficult documents take. It is retained
alongside per-page latency because a large PDF can have reasonable page speed
but still require a long total wait.

**Example:** For document times `2`, `3`, `4`, `5`, and `11` seconds, the median
is `4 seconds`; the slow `11-second` document influences the upper tail. The
benchmark reports the exact p95 using its fixed quantile implementation.

**Range and direction:** Non-negative seconds/document; lower is better. Results
are also reported by modality and page-count bucket.

#### Batch Pages per Minute

**Question:** What sustained batch capacity does the parser provide?

$$
\operatorname{PagesPerMinute}
=\frac{60N_{\text{successful pages}}}{T_{\text{batch seconds}}}
$$

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

$$
\operatorname{PeakRAM}
=\max_t\left(
  \sum_{p\in\mathcal{P}_{candidate}(t)}
  \operatorname{ResidentMemory}(p,t)
  \right)
$$

At every sampling point, add the resident RAM of the benchmark candidate and
all extractor child processes. The largest observed total is reported.

**Example:** If sampled process-tree totals are `1,200`, `2,450`, and `2,100`
MiB, Peak Process-Tree RAM is `2,450 MiB`.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

#### Peak VRAM

**Question:** How much GPU memory does extraction require?

$$
\operatorname{PeakVRAM}
=\max_t\operatorname{CandidateGPUMemory}(t)
$$

Report the largest GPU-memory allocation attributable to the candidate during
the measured extraction.

**Example:** GPU-memory samples of `700`, `1,800`, and `1,500` MiB produce Peak
VRAM `1,800 MiB`.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

#### Peak Temporary Disk

**Question:** How much additional working-disk space does extraction require?

$$
\operatorname{PeakTempDisk}
=\max_t\left(
  \operatorname{TempBytes}(t)-\operatorname{TempBytes}_{before}
  \right)
$$

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
Table Structure Score   = 0.55
Candidate Failure Rate  = 0.00
p95 warm latency/page   = 3.2 seconds
```

This means:

- nearly all extracted text is supported by the reference;
- approximately 25% of reference text was not recovered;
- recovered elements are mostly in the correct reading order;
- every reference page produced some matching content;
- every reference table was detected;
- table rows, columns, headers, or spans were reconstructed poorly despite the
  successful table detection;
- no scheduled sample ended in a fatal error; and
- 95% of measured warm per-page observations took no more than 3.2 seconds.

No single number communicates all of these facts. That is why the benchmark
reports the metrics separately instead of calculating a weighted overall score.

### Required document-extraction result groups

```text
text:
  Content Precision, Content Recall, Content F1, CER, WER,
  Reading Order Accuracy

pages:
  Page Coverage, Page Content F1, Page Attribution Accuracy,
  Duplicate Page Rate

layout:
  Layout Element Precision, Recall, F1, Element Type Accuracy,
  Hierarchy Accuracy, Mean Bounding-Box IoU

tables (conditional):
  Detection Precision, Recall, F1, Content F1, TEDS-S

formulas (conditional):
  Detection Precision, Recall, F1, CDM, ExpRate@CDM

reliability:
  Empty Output Rate, Duplicate Content Rate,
  Structured-output Determinism, Candidate Failure Rate

operational:
  First-item Latency, p50/p95 Warm Latency per Page,
  Complete Document Latency, Batch Pages per Minute,
  Peak Process-Tree RAM, Peak VRAM, Peak Temporary Disk
```

This grouping supports a complete conclusion without pretending that text,
layout, structure, reliability, and cost are interchangeable. No weighted
overall score is calculated.

### Reporting metadata

Every sample-based aggregate records `sample_count`, the number of samples that
contributed to its point estimate. It is reporting metadata, not a quality or
operational metric, and no confidence interval is calculated for it.

```text
text.content_f1 = 0.91
text.content_f1.sample_count = 120

tables.structure_score = 0.84
tables.structure_score.sample_count = 18
```

This makes the evidence volume visible without treating the count as part of
the confidence-interval calculation or as another performance result.

## Confidence intervals

A confidence interval expresses uncertainty in an aggregate calculated from
multiple independent observations. It does not describe the possible range of
an individual prediction, and it must not be attached to a value merely because
the value appears in MLflow.

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Standard/full text, page, layout, table, and formula aggregates | Yes | Calculated from the contributing independent samples. |
| Standard/full reliability rates | Yes | Calculated across scheduled independent samples. |
| Standard/full p50/p95 latency | Conditional | Reported when enough independent document or page observations support the percentile estimate. |
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

Eligible standard/full sample-based metrics use 10,000 bootstrap resamples with
seed 42:

1. Treat the document, page, query, clip, or other declared sample as the
   independent resampling unit.
2. Resample those units with replacement.
3. Recalculate the aggregate for every resample.
4. Use the 2.5th and 97.5th percentiles as the 95% interval bounds.

If `θ(X)` is the aggregate metric calculated from samples `X`, and `X_b*` is
bootstrap resample `b`, the reported percentile interval is:

```text
θ_b* = θ(X_b*)                                      for b = 1, …, 10,000
95% CI = [Q0.025({θ_b*}), Q0.975({θ_b*})]
```

Document-group metrics resample only the samples in that group. Conditional
metrics such as table structure or formula recognition resample only the samples
eligible for that metric. Paired candidate comparisons resample aligned sample
IDs together so both candidates are evaluated on the same resampled cases.

### MLflow names

The point estimate keeps the simple metric name. Interval bounds are attached to
it:

```text
text.content_f1
text.content_f1.ci_lower
text.content_f1.ci_upper

text.pdf_scanned.content_f1
text.pdf_scanned.content_f1.ci_lower
text.pdf_scanned.content_f1.ci_upper

tables.structure_score
tables.structure_score.ci_lower
tables.structure_score.ci_upper
```

A value without `ci_lower` and `ci_upper` has no reported interval. Missing
bounds are not interpreted as zero.

### Interpretation

For example:

```text
text.content_f1 = 0.91
95% CI          = [0.89, 0.93]
```

The benchmark estimates an aggregate Content F1 of `0.91`; variation across the
sampled documents produces the reported uncertainty interval. A narrower
interval means the aggregate is estimated more precisely. Overlapping intervals
do not by themselves prove that candidates are equal, and non-overlapping
intervals do not replace an aligned paired comparison when a formal difference
claim is made.

## Audio extraction

Audio uses CER and WER as defined above, with speech-specific alignment and cost
metrics:

| Metric | Definition | Direction |
|---|---|---|
| Timestamp Mean Absolute Error (MAE) | Mean absolute difference between aligned reference and predicted timestamps. | Lower |
| Segment Boundary MAE | MAE over aligned segment start and end times. | Lower |
| Timestamp Alignment Coverage | Reference timed units with a valid predicted alignment divided by timed reference units. | Higher |
| Real-Time Factor | Complete transcription-and-alignment seconds divided by audio duration seconds. | Lower |

The timestamp helper accepts already aligned arrays of equal non-zero length. A
stage must perform and record the alignment before calling it; it must not
truncate unequal arrays silently. The audio section will be expanded to the same
full contract format after its metric audit.

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

Aggregate metrics never replace sample rows. Eligible standard/full sample-based
metrics report the mean (or named percentile), a 95% interval, the number of
contributing samples, and failures. Conditional metrics such as table structure
or timestamps also report their sample count. Smoke values, counts, statuses,
fixed identifiers, and single operational observations do not have authoritative
intervals. An engineer reviews the complete evidence; EduMind does not combine
unrelated metrics into a weighted overall score or promote a candidate
automatically.
