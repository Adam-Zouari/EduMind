# Retrieval and reranking metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Retrieval quality](#retrieval-quality-1)
- [Why the quality metrics are not interchangeable](#why-the-quality-metrics-are-not-interchangeable)
- [Operational measurements](#operational-measurements)
- [Workload descriptors](#workload-descriptors-1)
- [Validity gates and eligibility](#validity-gates-and-eligibility)
- [Aggregation and confidence intervals](#aggregation-and-confidence-intervals)

The declared roster contains 15 complete `retriever|reranker` candidates:
Dense, BM25, and RRF, each with no reranker or one of four learned rerankers.
Development compares the preflight-qualified subset. Quality uses the
same frozen evidence representation and evaluation tokenizer as the
chunking/embedding benchmark. No context-token budget is applied and no metric
families are combined into a weighted score.

## Metric summary

### Retrieval quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| nDCG@3/@5 | Primary | Does the complete stack place chunks containing more required evidence near the top? | Higher |
| Evidence-unit Recall@3/@5 | Primary | How much of the required evidence is present in the first three or five chunks? | Higher |
| Evidence-token Precision@3/@5 | Primary | How concentrated are the first three or five chunks around verified evidence? | Higher |
| alpha-nDCG@3/@5 | Diagnostic | How early does the ranking surface required evidence when repeated coverage receives less credit? | Higher |
| Candidate-pool Evidence-unit Recall@20 | Diagnostic | Did the first-stage top-20 pool contain the required evidence before any reranker reordered it? | Higher |
| Ranking Agreement | Validity gate | Does repeated inference return the same complete ordering? | Must equal 1.0 |

`@3` and `@5` mean that the calculation uses the first three and first five
ranked chunks. Both cutoffs are primary so the engineer can compare the two
context counts before freezing the final system's top-K. The top-20 metric
diagnoses the first-stage ceiling; it is not another selection cutoff.

### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Full-Stack Warm Latency p50/p95 | Operational | How long does the deployable first-stage-plus-reranker path usually take, and how slow is its warm tail? | Lower |
| First-Stage Warm Latency p50/p95 | Operational | How much of total query time belongs to Dense, BM25, or RRF retrieval? | Lower |
| Reranker Warm Latency p50/p95 | Operational | For reranked candidates, what additional query time does learned reranking require? | Lower |
| Cold Initialization Time | Operational | How long does the complete candidate take to become ready from a fresh worker? | Lower |
| Peak Process-Tree RAM | Operational | What maximum system memory does the candidate require? | Lower |
| Peak Device VRAM | Operational | What maximum total GPU memory is occupied during candidate execution? | Lower |
| Index-Build Time | Operational | How long does preparation of the first-stage searchable state take? | Lower |
| Index Bytes | Storage descriptor | How much stored searchable state does the first-stage retriever require? | Descriptive |

### Workload descriptors

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Corpus, Query, Chunk, and Pool Counts | Workload descriptor | What fixed data and ranked-list sizes were actually processed? | Descriptive |
| Mean/p95 Retrieved Tokens@3/@5 | Workload descriptor | How much canonical source text would each cutoff pass downstream? | Descriptive |
| Mean/p95 Reranker Input Tokens | Workload descriptor | How much native-tokenizer input did a learned reranker process per query? | Descriptive |
| Reranker Snapshot Bytes | Storage descriptor | How much local model storage does a reranker require? | Descriptive |

Workload and storage values explain latency and resource differences. They do
not compensate for worse retrieval quality and are not combined with quality
into a winner score.

## Retrieval quality

### nDCG@3/@5

**Question:** How early does the complete stack place evidence-rich chunks when
repeated evidence keeps its relevance credit?

The linear evidence-count definition is the same as in chunking and embedding:
each chunk's gain is the number of distinct required units it completely covers,
used directly with the logarithmic rank discount. A unit counts once within
each chunk; coverage in an earlier chunk does not reduce a later chunk's gain.
The metric evaluates relevance ordering, not diversity-aware selection.

For each question, the exact ideal ranking sorts the gains of every chunk in
the complete frozen corpus. It is not derived from Dense, BM25, RRF, or any
candidate's top-20 pool. All evaluated stacks therefore use the same ideal
denominator for that question; a retriever cannot make its normalization easier
by failing to retrieve relevant chunks.

**Example:** A chunk covering three required units receives gain `3`; one
covering one receives `1`. Moving the three-unit chunk ahead of the one-unit
chunk improves nDCG. Two distinct chunks covering the same three units each
retain gain `3`, even when placed consecutively.

This is primary because the retriever and reranker are compared on relevance
ranking. Repeated relevant chunks are not automatically ranking errors.
Evidence-unit Recall measures distinct coverage, while diagnostic alpha-nDCG
describes how early new evidence is introduced.

**Range and direction:** `[0, 1]`; higher is better. A question with no verified
evidence in the scored results receives zero.

### Evidence-unit Recall@3/@5

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
for a complete answer.

**Range and direction:** `[0, 1]`; higher is better.

### Evidence-token Precision@3/@5

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

### alpha-nDCG@3/@5

**Question:** How early does the ranking surface required evidence when repeated
coverage receives less credit?

Alpha-nDCG discounts a lower-ranked chunk only when it covers an evidence-unit
ID already covered higher in the ranking. It does not use text similarity to
decide that two chunks are duplicates, and it does not declare repeated relevant
evidence inherently bad.

**Example:** If ranks 1 and 2 contain the same evidence unit and rank 3 contains
a different required unit, moving the different unit to rank 2 improves
alpha-nDCG while ordinary nDCG may remain high.

The metric is diagnostic because it adds a novelty preference beyond the
relevance-ranking objective. A lower score does not make repeated relevant
chunks irrelevant or establish that their repetition harms answers. It uses
fixed `alpha=0.5` and is calculated for every answerable question with at least
one gold evidence unit, including single-unit questions. Its eligible questions
and documents are the same as those of the primary retrieval metrics.
Unanswerable questions are excluded from all four quality metrics.

Normalization uses the same
[deterministic greedy approximation](../chunking_embedding/metrics.md#alpha-ndcg35)
as the chunking/embedding benchmark. At each rank, the unused chunk with the largest
remaining alpha gain is chosen, including discounted credit for repeated units;
ties follow the frozen corpus order. All candidates share this baseline from
the complete frozen corpus, never from a retrieved pool. Scores are capped at
`1`; the baseline is not guaranteed to be the exact maximum. The development-only
greedy-versus-exact review is tracked in
[pending-data-review.md](../../pending-data-review.md).

**Range and direction:** `[0, 1]`; higher is better. An answerable question with
no complete evidence unit in its scored results receives zero.

### Candidate-pool Evidence-unit Recall@20

**Question:** Did the first-stage retriever give its rerankers an adequate set of
candidates?

The metric measures distinct required evidence units found anywhere in the
frozen top-20 pool before learned reranking. A reranker cannot recover evidence
that is absent from this pool. The value therefore separates a first-stage miss
from a poor reordering decision.

It is recorded once on each of `dense|none`, `bm25|none`, and `rrf|none`. The
four reranker children reference the matching pool-collection artifact and
checksum instead of copying the same value as if they had created it.

**Range and direction:** `[0, 1]`; higher is better as a diagnostic ceiling.

### Ranking Agreement

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

## Why the quality metrics are not interchangeable

| Metric | What changes it | What it does not answer directly |
|---|---|---|
| nDCG@3/@5 | The ranks and complete evidence-unit counts of chunks | Whether every distinct unit was recovered or how much extra text was returned |
| Evidence-unit Recall@3/@5 | Which distinct required units appear within the cutoff | Whether those units appeared early or were surrounded by unrelated text |
| Evidence-token Precision@3/@5 | The proportion of returned tokens inside verified evidence | Whether all required units were found or ordered early |
| alpha-nDCG@3/@5 | When evidence appears and how often its unit IDs have already been covered | Whether repeated relevant evidence actually harms the downstream answer |
| Pool Recall@20 | Evidence available to a reranker before reordering | Whether the final top-three or top-five order is good |

Together, the primary metrics answer ordering, completeness, and concentration.
The diagnostics describe first-stage pool limits and repeated coverage. Ranking
Agreement separately checks execution stability; repeated relevant evidence is
not itself a relevance-ranking failure.

## Operational measurements

### Warm latency

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

### Cold initialization and peak resources

Cold initialization uses a fresh worker and ends when the complete candidate is
ready to accept a query. It includes loading the required retriever state,
tokenizers, and reranker when applicable; downloads and environment installation
remain outside the benchmark.

Peak RAM is the largest sampled resident-memory total across the worker and its
child processes. Peak Device VRAM uses the shared raw NVML device-total contract.
A confirmed CPU-only run may report zero VRAM; missing GPU instrumentation is reported as
unavailable, not converted to zero.

### Index build and storage

Index-build time begins after required models are loaded and ends when the
first-stage searchable state is ready. Index bytes are measured from that state.
The Dense and BM25 no-reranker owners record their own build time and index
bytes; their four reranker children reference those artifacts instead of copying
the values. RRF references both checksummed indexes and builds no third search
index. Its required bytes are the unique sum of the Dense and BM25 indexes, and
its incremental fusion-index bytes are zero.

## Workload descriptors

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

## Validity gates and eligibility

An authoritative child is valid only when every expected query is processed,
all ranking scores are finite, input truncation is zero, repeated rankings agree,
and each reranker output is an exact permutation of the referenced checksummed
pool. A failed query, pool mismatch, duplicate/missing chunk ID, nonfinite score,
or truncated input makes the child failed and the parent comparison incomplete.
There is no average failure-rate metric that can hide these errors.

Quality metrics include answerable questions only. Every overall result and
text, table, formula, or mixed slice records eligible question and document
counts shared by all four retrieval-quality metrics. Single-unit questions are
included in alpha-nDCG. Inapplicable operational fields are absent rather than
filled with zero.

## Aggregation and confidence intervals

Quality is calculated per eligible question, averaged within each source
document, and macro-averaged across documents. This prevents a paper with many
questions from dominating the result. Development, validation, and locked runs use 10,000
bootstrap resamples of complete documents with seed 42; the 2.5th and 97.5th
percentiles form the 95% confidence interval. Evidence slices repeat the same
calculation over contributing documents.

Smoke values receive no authoritative interval. One-off cold initialization,
index build, index bytes, peak resources, and workload descriptors receive point
observations but no invented interval. Warm latency intervals are reported only
when enough independent source documents support them. Resampling keeps all
query observations from each sampled document together; it does not treat
related questions as independent evidence.
