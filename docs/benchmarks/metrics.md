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

### Text content and recognition

For normalized reference-token counts `R(t)` and predicted-token counts `P(t)`,
let:

```text
M = sum over tokens t of min(R(t), P(t))
```

Repeated token occurrences are counted; token sets are not used.

| Metric | Calculation | Question answered | Range and direction |
|---|---|---|---|
| Content Precision | `M / sum(P(t))` | How much extracted content is supported by the reference? | `[0, 1]`, higher is better |
| Content Recall | `M / sum(R(t))` | How much required content was recovered? | `[0, 1]`, higher is better |
| Content F1 | `2PR / (P + R)` | How well does the parser balance additional and missing content? | `[0, 1]`, higher is better |
| Character Error Rate (CER) | Character-level Levenshtein substitutions + deletions + insertions, divided by reference characters | How severe are character-recognition errors? | `[0, infinity)`, lower is better |
| Word Error Rate (WER) | Word-level Levenshtein substitutions + deletions + insertions, divided by reference words | How severe are complete word errors? | `[0, infinity)`, lower is better |
| Reading Order Accuracy | Concordant pairs of matched layout elements divided by all comparable matched-element pairs | Is recovered content presented in the correct sequence? | `[0, 1]`, higher is better |

Valid document samples have non-empty references. An empty prediction receives
zero Content Precision, Recall, and F1 and is also counted by Empty Output Rate.
CER and WER may exceed 1 when insertions outnumber reference units.

Content Precision, Recall, and F1 are intentionally retained together. Precision
identifies additional output, Recall identifies omissions, and F1 summarizes the
balance. CER and WER are also distinct: a one-character substitution may have a
small character cost while making an entire word incorrect.

Reading order is evaluated over one-to-one matched document elements, not only
words that happen to be unique. For every pair of matched elements, the metric
checks whether their relative order agrees in reference and prediction. Samples
with fewer than two matched elements are ineligible for this metric rather than
being assigned an artificial perfect or zero value.

### Pages

| Metric | Calculation | Question answered | Range and direction |
|---|---|---|---|
| Page Coverage | Reference content-bearing pages with at least one matched predicted content element, divided by reference content-bearing pages | Did every expected page produce relevant content? | `[0, 1]`, higher is better |
| Page Content F1 | Content F1 calculated within the same page identifier and macro-averaged across reference and unexpected predicted pages | Was the correct content recovered within the correct page? | `[0, 1]`, higher is better |
| Page Attribution Accuracy | Matched elements assigned to their reference page, divided by matched elements carrying page annotations | Was extracted content assigned to the correct page number? | `[0, 1]`, higher is better |
| Duplicate Page Rate | Unsupported duplicate predicted pages divided by predicted pages | How often did the parser repeat a page? | `[0, 1]`, lower is better |

Page Coverage is not simply a count of non-empty outputs: a page counts as
covered only when some reference content is matched. Page Content F1 penalizes
missing, incorrect, and unexpected page content. Page Attribution Accuracy looks
only at the placement of matched elements, so low attribution is distinguishable
from low content recovery.

Duplicate-page detection uses a fixed, versioned near-duplicate rule over
canonical page content. Repetition already present in the reference is excluded;
otherwise legitimate repeated covers, forms, or headers could be mislabeled as
parser duplication.

### Layout and document structure

The layout set contains headings, paragraphs, list items, captions, figures,
code blocks, and other annotated non-table/non-formula elements. Tables and
formulas are evaluated in their own sections.

Reference and predicted elements are matched one-to-one. When boxes are
available, matching maximizes bounding-box Intersection over Union (IoU), with
`IoU >= 0.5` required for a match. When a corpus lacks boxes, its pinned official
element matcher is used instead; matching protocols are never mixed silently
inside one comparison.

| Metric | Calculation | Question answered | Range and direction |
|---|---|---|---|
| Layout Element Precision | Matched predicted elements divided by predicted elements | How many predicted document elements are real? | `[0, 1]`, higher is better |
| Layout Element Recall | Matched reference elements divided by reference elements | How many required document elements were detected? | `[0, 1]`, higher is better |
| Layout Element F1 | Harmonic mean of layout precision and recall | How well does element detection balance false and missed elements? | `[0, 1]`, higher is better |
| Element Type Accuracy | Matched elements with the correct semantic type, divided by matched elements | Were headings, paragraphs, lists, captions, and other blocks classified correctly? | `[0, 1]`, higher is better |
| Hierarchy Accuracy | Eligible matched elements with the correct parent and hierarchy level, divided by eligible matched elements | Were heading levels, list nesting, and parent-child relationships preserved? | `[0, 1]`, higher is better |
| Mean Bounding-Box IoU | Mean geometric IoU over matched elements with reference boxes | Were elements localized correctly on the page? | `[0, 1]`, higher is better |

The previous exact-line Block F1 is not part of this contract. Exact line equality
is too brittle for layout detection and does not measure element types, geometry,
or hierarchy.

### Tables

Table metrics are calculated only for samples with table annotations. Detection
uses one-to-one table-region matching at `IoU >= 0.5`; crop-level datasets use
their explicit table identities. Results are reported overall and by table
attributes such as bordered/borderless and merged-cell presence when those
labels exist.

| Metric | Calculation | Question answered | Range and direction |
|---|---|---|---|
| Table Detection Precision | Matched predicted tables divided by predicted tables | How many predicted tables are real tables? | `[0, 1]`, higher is better |
| Table Detection Recall | Matched reference tables divided by reference tables | How many reference tables were found? | `[0, 1]`, higher is better |
| Table Detection F1 | Harmonic mean of table-detection precision and recall | How well does detection balance extra and missing tables? | `[0, 1]`, higher is better |
| Table Content F1 | Token-content F1 for matched tables, macro-averaged over reference tables; a missing reference table receives zero | Was the textual content inside tables recovered? | `[0, 1]`, higher is better |
| Table Structure Score (TEDS-S) | Tree-Edit-Distance-based Similarity after cell text is removed | Were rows, columns, headers, merged cells, and spans reconstructed correctly? | `[0, 1]`, higher is better |

TEDS-S is used for structure because it separates structural reconstruction from
cell recognition. Table Content F1 then answers the separate text question. The
benchmark uses the pinned official evaluator rather than EduMind's former
row/column-adjacency approximation. OmniDocBench documents TEDS and TEDS-S in its
[official evaluation repository](https://github.com/opendatalab/OmniDocBench).

### Formulas

Formula metrics are calculated only for samples with formula annotations and are
reported as a total and separately for inline and display formulas. Detection
uses one-to-one region matching at `IoU >= 0.5` when boxes are available.

| Metric | Calculation | Question answered | Range and direction |
|---|---|---|---|
| Formula Detection Precision | Matched predicted formulas divided by predicted formulas | How many predicted formulas are real formulas? | `[0, 1]`, higher is better |
| Formula Detection Recall | Matched reference formulas divided by reference formulas | How many reference formulas were found? | `[0, 1]`, higher is better |
| Formula Detection F1 | Harmonic mean of formula-detection precision and recall | How well does detection balance extra and missing formulas? | `[0, 1]`, higher is better |
| Formula Recognition Similarity (CDM) | Official Character Detection Matching score, normalized to `[0, 1]`, macro-averaged over reference formulas; a missing formula receives zero | How visually and structurally close is the reconstructed mathematical expression? | `[0, 1]`, higher is better |
| Formula Exact Match (ExpRate@CDM) | Reference formulas recognized exactly according to the CDM evaluator, divided by reference formulas | How often is a formula reconstructed perfectly? | `[0, 1]`, higher is better |

CDM compares rendered formula characters and their spatial relationships, so
equivalent renderings are not penalized merely for using different LaTeX source.
Exact Match remains useful because a high average similarity can coexist with a
low percentage of completely correct formulas. The evaluator and its exact
revision are pinned from the [official OmniDocBench evaluation
code](https://github.com/opendatalab/OmniDocBench); EduMind does not substitute a
home-grown LaTeX edit score.

### Reliability and failure behavior

| Metric | Calculation | Question answered | Range and direction |
|---|---|---|---|
| Empty Output Rate | Scheduled samples producing no canonical content, divided by scheduled samples | How often does extraction fail completely? | `[0, 1]`, lower is better |
| Duplicate Content Rate | Unsupported duplicated canonical content units divided by predicted content units | How often does the parser repeat substantial content? | `[0, 1]`, lower is better |
| Structured-output Determinism | Samples whose repeated canonical-output fingerprints are identical, divided by repeatedly measured samples | Does the same input produce the same complete structured result? | `[0, 1]`, higher is better |
| Candidate Failure Rate | Scheduled samples ending in a fatal extraction error, divided by scheduled samples | How often can the candidate not produce a scoreable result? | `[0, 1]`, lower is better |

The determinism fingerprint includes text, pages, element types and order,
hierarchy, tables, formulas, and normalized boxes. Timings, temporary paths, and
random run identifiers are excluded.

Every scheduled sample remains visible. A recoverable failure is represented by
an explicit per-sample result and contributes to Empty Output or Candidate
Failure Rate; it is never dropped from quality aggregation. A failure that
prevents the required per-sample record makes the benchmark invocation
incomplete and therefore non-authoritative.

### Operational performance

| Metric | Calculation | Question answered | Range and direction |
|---|---|---|---|
| First-item Latency | Wall-clock time for model initialization plus the first extraction in a fresh candidate process | What initialization cost does the first user request experience? | Seconds, lower is better |
| p50 Warm Latency per Page | Median warm document latency divided by processed pages | What is typical steady-state page speed? | Seconds/page, lower is better |
| p95 Warm Latency per Page | 95th percentile of warm per-page latency | How slow are difficult steady-state pages? | Seconds/page, lower is better |
| Complete Document Latency | End-to-end wall-clock time for each whole source; p50/p95 are reported by modality and page-count bucket | How long does a user wait for a complete document? | Seconds/document, lower is better |
| Batch Pages per Minute | Successfully processed pages divided by measured batch wall time | What sustained batch capacity does the parser provide? | Pages/minute, higher is better |
| Peak Process-Tree RAM | Peak resident memory of the benchmark process plus extractor child processes | How much system memory does the complete extractor require? | MiB, lower is better |
| Peak VRAM | Peak GPU memory attributable to the candidate process tree | How much GPU memory does extraction require? | MiB, lower is better |
| Peak Temporary Disk | Maximum additional temporary-file footprint during extraction | How much working disk space is required? | MiB, lower is better |

Batch Pages per Minute is retained only when measured with a real sustained
batch. It is omitted when it would merely be the arithmetic inverse of sequential
latency. Missing RAM, VRAM, or disk instrumentation is recorded as unavailable;
it is never silently converted to zero.

p50/p95 latency intervals are reported only when enough independent document or
page observations support them. A single first-item measurement, throughput
batch, or peak RAM/VRAM/temporary-disk observation is reported without a
confidence interval. If an operational measurement is repeated independently,
an interval may be reported only with the repetition count and aggregation unit.

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
