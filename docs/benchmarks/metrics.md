# Benchmark metric reference

[Benchmark overview](overview.md) · [Experiment sequence and rationale](methodology.md) ·
[Benchmark runbook](running.md)

This page defines how EduMind calculates benchmark metrics. The
[methodology](methodology.md) says **where** each metric is used and why; this
page says **what the value means**. Higher is better unless a metric is marked
lower-is-better.

## Shared conventions

- Content, Exact Match, Token F1, and ROUGE-L use case-folded `\w+` tokens.
  CER operates on the strings supplied by the stage, and WER splits those
  strings on whitespace. A stage therefore performs its fixed canonical
  normalization before scoring rather than hiding it inside these two metrics.
- Source and evidence spans are half-open intervals: `[start, end)`.
- Empty denominators use the explicit behavior stated below; they never produce
  fabricated zero-quality observations.
- Standard and full runs retain one row per sample before aggregation.
- p50, p95, and p99 are latency percentiles. Throughput is completed operations
  divided by measured wall-clock time.
- Confidence intervals use 10,000 bootstrap resamples with seed 42. Paired
  comparisons resample aligned samples, not unrelated aggregate values.
- Normalized precision, recall, F1, accuracy, coverage, nDCG, and correctness
  values lie in `[0, 1]`. CER and WER are non-negative and can exceed 1 when
  insertions outnumber reference units. Human rubric scores use their stated
  `0–2` or `0–1` scales. Time, memory, storage, and throughput are non-negative
  and have no fixed upper bound.

## Text extraction and transcription

Let `d(reference, prediction)` be Levenshtein edit distance.

| Metric | Definition | Direction |
|---|---|---|
| Character Error Rate (CER) | Character edit distance on the supplied strings divided by the number of reference characters. | Lower |
| Word Error Rate (WER) | Edit distance over whitespace-separated words divided by the number of reference words. | Lower |
| Content Precision | Matched predicted word-token occurrences divided by predicted occurrences. Repeated words are counted as a multiset. | Higher |
| Content Recall | Matched reference word-token occurrences divided by reference occurrences. | Higher |
| Content F1 | Harmonic mean of content precision and recall. | Higher |
| Missing Text/Speech Rate | Unmatched reference occurrences divided by reference occurrences. | Lower |
| Hallucinated Text/Speech Rate | Unmatched predicted occurrences divided by predicted occurrences. | Lower |
| Reading Order Accuracy | Fraction of comparable token pairs that occur in the same relative order in reference and prediction. Only tokens occurring once in both texts are comparable. | Higher |
| Block Precision/Recall/F1 | Exact normalized-line matches treated as a multiset. | Higher |
| Empty Output Rate | Fraction of samples producing no normalized content. | Lower |
| Duplicate Text Rate | Fraction of repeated normalized output units beyond their first occurrence. | Lower |

CER is useful for spelling and OCR errors; WER reflects errors at readable-word
level. Content F1 tolerates ordering differences, so it is paired with Reading
Order Accuracy rather than used alone.

### Pages and structured document content

| Metric | Definition | Direction |
|---|---|---|
| Page Coverage | Reference pages with non-empty matched output divided by reference pages. | Higher |
| Page Content F1 | Content F1 calculated within each page, then averaged over annotated pages. | Higher |
| Page Attribution Accuracy | Predicted page texts whose most similar reference page has the same page number, divided by the total number of reference pages. | Higher |
| Duplicate Page Rate | Repeated normalized page outputs beyond their first occurrence divided by produced pages. | Lower |
| Table Detection F1 | F1 over matched reference and predicted tables. A greedy match requires content similarity of at least 0.5. | Higher |
| Table Content F1 | Content F1 over matched table cells/text. | Higher |
| Table Structure F1 | F1 over row/column adjacency relations in matched tables. | Higher |
| Formula Detection F1 | F1 over reference and predicted formulas; greedy matching requires normalized similarity of at least 0.5. | Higher |
| Formula LaTeX Similarity | `1 - edit_distance / max(reference_length, prediction_length, 1)` after LaTeX normalization. | Higher |
| Formula Exact Match | Fraction of matched formulas with identical normalized LaTeX. | Higher |

Table and formula metrics are calculated only for samples carrying the required
annotations. Reports must include the contributing sample count.

### Timestamps and audio cost

| Metric | Definition | Direction |
|---|---|---|
| Timestamp Mean Absolute Error (MAE) | Mean absolute difference between aligned reference and predicted timestamps. | Lower |
| Segment Boundary MAE | MAE over aligned segment start and end times. | Lower |
| Timestamp Alignment Coverage | Reference timed units with a valid predicted alignment divided by timed reference units. | Higher |
| Real-Time Factor | Complete transcription-and-alignment seconds divided by audio duration seconds. | Lower |

The current timestamp helper accepts already aligned arrays of equal non-zero
length. A stage must perform and record the alignment before calling it; it must
not truncate unequal arrays silently.

## Normalization

For reference text `r`, corrupted observed text `o`, and normalized output `n`:

```text
B = edit_distance(r, o)        original corruption
A = edit_distance(r, n)        corruption remaining
C = edit_distance(o, n)        edits made by normalization
R = max(0, B - A)              useful repair
```

| Metric | Definition | Direction |
|---|---|---|
| Content Preservation Recall | Reference content retained after normalization divided by reference content. | Higher |
| Corruption Removal Recall | `R / B`; equals 1 when the input has no corruption and remains correct. | Higher |
| Corruption Removal Precision | `R / C`; unnecessary or harmful edits increase `C` without increasing `R`. | Higher |
| Corruption Removal F1 | Harmonic mean of removal precision and recall. | Higher |
| Accidental Deletion/Merge Rate | Reference units deleted or incorrectly joined by normalization divided by reference units. | Lower |

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

Aggregate metrics never replace sample rows. Reports show the mean (or named
percentile), a 95% interval, the number of contributing samples, and failures.
Conditional metrics such as table structure or timestamps also report their
eligible subset size. An engineer reviews the complete evidence; EduMind does
not combine unrelated metrics into a weighted overall score or promote a
candidate automatically.
