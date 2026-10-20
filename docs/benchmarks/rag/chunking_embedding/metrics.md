# Chunking and embedding metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Eligibility and result status](#eligibility-and-result-status)
- [Retrieval quality](#retrieval-quality-1)
- [Why the metrics are not interchangeable](#why-the-metrics-are-not-interchangeable)
- [Worked candidate interpretation](#worked-candidate-interpretation)
- [Ranking consistency](#ranking-consistency-1)
- [Operational performance](#operational-performance-1)
- [Workload and storage descriptors](#workload-and-storage-descriptors-1)
- [Aggregation and comparison completeness](#aggregation-and-comparison-completeness)
- [Chunking and embedding confidence intervals](#chunking-and-embedding-confidence-intervals)

The chunking/embedding experiment evaluates one complete
`chunker|embedding` pair at a time with exact cosine search. Three primary metric
families answer different selection questions; alpha-nDCG is retained as a
novelty diagnostic. None is combined into a weighted score.

The four quality families adopt the
[shared retrieval-evidence definitions](../../metrics.md#retrieval-evidence-quality).
The applications below use each candidate pair's complete generated corpus,
not another pair's corpus or the retrieved top-20 pool.

The [implementation boundary](methodology.md#implementation-boundary) identifies
pending operational-breakdown, repeatability, and reporting changes. Definitions
below are the approved contract, not aliases for unmodified older output keys.

## Metric summary

### Retrieval quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| nDCG@3/@5 | Primary | Are chunks containing more required evidence ranked near the top? | Higher |
| Evidence-unit Recall@3/@5 | Primary | How much of the required evidence is present in the retrieved set? | Higher |
| Evidence-token Precision@3/@5 | Primary | How concentrated is the retrieved text around verified evidence? | Higher |
| alpha-nDCG@3/@5 | Diagnostic | How early does required evidence appear when repeated coverage receives less credit? | Higher |

`@3` and `@5` mean that the same calculation is performed over the first three
and first five ranked chunks. Both are reported to compare retrieval quality
at the two candidate context counts before the final system's top-K is frozen.

### Ranking consistency

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Ranking Repeatability | Diagnostic | How often do repeated requests deliver the same valid ordered top-five result? | Higher |

### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Cold Model-Load Time | Operational | What startup delay makes the embedding model and tokenizer ready? | Lower |
| Chunking Time | Operational | How expensive is creating the complete chunk inventory? | Lower at equal quality |
| Chunk-Embedding Time | Operational | How expensive is producing the final vectors for all chunks? | Lower at equal quality |
| Corpus-Build Time | Operational | How long does chunking, document embedding, and searchable-matrix preparation take? | Lower |
| Corpus-Build Throughput | Operational | How many original source tokens are processed per second? | Higher |
| Query-Embedding Latency p50/p95 | Operational | What typical and slow-tail query cost belongs to producing its vector? | Lower at equal quality |
| Vector Search Latency p50/p95 | Operational | What typical and slow-tail query cost belongs to searching and selecting results? | Lower at equal quality |
| p50 Warm Query Latency | Operational | What is normal query-embedding and exact-search latency? | Lower |
| p95 Warm Query Latency | Operational | What is slow-case warm query latency? | Lower |
| Peak Process-Tree RAM | Operational | What peak host memory is observed for the worker and its children? | Lower at equal quality |
| Peak Device VRAM | Operational | How much total GPU memory is occupied while the pair runs? | Lower at equal quality |

### Workload and storage descriptors

| Value | Role | Question answered | Direction |
|---|---|---|---|
| Corpus Counts | Workload descriptor | How many documents and answerable/unanswerable questions were represented in the run? | Descriptive |
| Source Tokens | Workload descriptor | How large was the original corpus before chunk overlap? | Descriptive |
| Indexed-Token Occurrences | Workload descriptor | How much source text do the generated chunks represent, including overlap and repeated context? | Descriptive |
| Chunk Count | Workload descriptor | How many searchable vectors and metadata records did the strategy create? | Descriptive |
| Mean/p95 Chunk Tokens | Workload descriptor | What typical and long-tail chunk sizes did the strategy actually produce? | Descriptive |
| Mean/p95 Retrieved Context Tokens@3/@5 | Workload descriptor | How much text do the actual selected prefixes return on average and in the upper tail? | Descriptive |
| Embedding Dimension and Dtype | Storage descriptor | What shape and numeric representation did each vector use? | Descriptive |
| Embedding-Matrix Bytes | Storage descriptor | How large is the complete matrix's raw vector payload? | Descriptive |

## Eligibility and result status

Only answerable questions with reviewed gold evidence are executed and scored.
Unanswerable questions retained in a shared manifest are input counts for later
stages, not retrieval requests or empty-reference quality comparisons here.
Invalid or missing required evidence is a
[data-validation failure](../../data-validation.md#chunking-and-embedding-inputs),
not a zero or null observation from a candidate that never ran.

Quality uses the designated first measured ranking under the
[shared availability rules](../../metrics.md#shared-availability-rules).

Score all four quality families once at both cutoffs from a valid first ranking.
Missing evidence in valid retrieved chunks receives the normal poor numeric
score. Failures and missing records follow the
[shared availability rules](../../metrics.md#shared-availability-rules);
slice eligibility and partial results follow
[aggregation and comparison completeness](#aggregation-and-comparison-completeness).

Generated chunks, vectors, and rankings have separate integrity checks. Exact
search on a valid nonempty index returns `min(20, chunk_count)` unique corpus
chunk IDs with finite scores. An unexpected empty or short ranking, token-free
chunk, invalid ID, or corrupted vector is an execution/integrity failure, not
a scored empty prediction. In particular, no valid returned source tokens makes
applicable quality null/unavailable, not a scored zero-token comparison. A
legitimate corpus containing fewer than three or five chunks uses the available
chunks at the unchanged cutoff.

Scoring checks require finite, in-range reported quality values. A substantive
nDCG above one indicates an integrity/calculation error, not a normal quality
case; numerical tolerances are code invariants. Alpha-nDCG's retained raw ratio
is explicitly different because its greedy baseline can be exceeded.

Shared status/support meanings apply to each result. The
[methodology](methodology.md#result-completeness-and-advancement) defines
which completed results can enter selection comparisons.

## Retrieval quality

### nDCG@3/@5

**Question:** How early do evidence-rich chunks appear when repeated evidence
keeps its relevance credit?

Apply the [shared linear-gain nDCG rule](../../metrics.md#retrieval-evidence-quality)
to each question and candidate pair. The exact ideal comes from that pair's
complete generated corpus; distinct relevant chunks keep full evidence credit
even when their text or evidence repeats.

In plain words:

```text
discounted gain at rank r = complete evidence units in that chunk / log2(r + 1)
DCG@K = sum of those discounted gains through rank K
nDCG@K = observed DCG@K / ideal DCG@K over the complete corpus
```

**Example:** Suppose the corpus has only two evidence-bearing chunks: one covers
one required unit and the other covers all three. At `@3`, ordering their gains
as `[3, 1, 0]` gives `1.00`, while `[1, 3, 0]` gives approximately `0.80`. Both
orders recover the same units and return the same text, but the first places
more evidence earlier.

The rank discounts at ranks 1, 2, and 3 are `1`, approximately `0.631`,
and `0.5`. Thus the ideal DCG is `3 + 1 × 0.631 = 3.631`; the reversed
order gives `1 + 3 × 0.631 = 2.893`, and `2.893 / 3.631 ≈ 0.797`.
Use unrounded values for actual scoring.

nDCG is primary because this benchmark evaluates relevance ranking rather than
diversity-aware selection. Evidence-unit Recall separately measures distinct
coverage; alpha-nDCG diagnoses novelty without treating repeated relevant
evidence as a relevance-ranking error.

**Range and direction:** `[0, 1]`; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Zero ideal gain over the complete candidate corpus | Every chunk splits the question's required units, so no chunk covers a complete unit. | `0` (`scored`) | The explicit zero-denominator convention keeps boundary fragmentation in the quality result rather than excluding the pair. |

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
Evidence-slice and aggregate status follow
[this suite's completeness rules](#aggregation-and-comparison-completeness).

### Evidence-unit Recall@3/@5

**Question:** How many required evidence units appear anywhere in the first
three or first five chunks, regardless of their order?

Each required evidence unit counts once when at least one chunk inside the
cutoff contains it completely. Repeated copies do not add credit, and partial
fragments do not count as recovered units. Every eligible question has at least
one reviewed gold unit, so the denominator is positive; missing required units
are an [input-validation issue](../../data-validation.md#chunking-and-embedding-inputs),
not an empty-reference scoring convention.

```text
Evidence-unit Recall@K = distinct required units completely covered by top K
                        / all required units for the question
```

**Example:** If the first three chunks recover one of three required units and
the next two recover another, Recall@3 is approximately `0.33` and Recall@5 is
approximately `0.67`.

This primary metric catches rankings that look good near the top but still miss
part of the evidence needed for a complete answer. Recovering no complete units
gives zero under the ordinary calculation.

**Range and direction:** `[0, 1]`; higher is better.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
Evidence-slice and aggregate status follow
[this suite's completeness rules](#aggregation-and-comparison-completeness).

### Evidence-token Precision@3/@5

**Question:** What share of the tokens returned in the first three or first five
chunks is relevant evidence?

Measure canonical chunk text with `tiktoken:cl100k_base`. A token occurrence is
relevant when its source span has positive overlap with a verified evidence
interval in the same document. Count it once even if several gold intervals
overlap it. All returned source-token occurrences form the denominator; text
repeated in different chunks counts each time in both numerator and denominator.
Native model prefixes and special inference tokens are not source content.
The valid nonempty ranking and token-bearing chunks make the denominator
positive; see the [output-integrity prerequisites](#eligibility-and-result-status).

```text
Evidence-token Precision@K = evidence-bearing returned token occurrences
                            / all returned source-token occurrences
```

**Example:** If 160 of 600 retrieved tokens are relevant evidence, precision is
approximately `0.27`. The remaining tokens are additional context not counted
as relevant evidence by the reference annotations.

This primary metric does not impose a context budget or penalize a chunk merely
for being large. It reports how concentrated the returned context is around the
verified evidence. Valid returned text with no evidence-bearing tokens receives
zero under the ordinary calculation.

**Range and direction:** `[0, 1]`; higher is better.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
Evidence-slice and aggregate status follow
[this suite's completeness rules](#aggregation-and-comparison-completeness).

### Alpha-nDCG@3/@5

**Question:** How early does required evidence appear when repeated coverage
receives less credit?

Alpha-nDCG rewards useful evidence more when it appears near the top and reduces
the credit for later chunks that repeat the same evidence. With the frozen
`alpha=0.5` setting, each repetition receives half the remaining novelty credit.

In plain words:

```text
novelty gain of a chunk = sum, over its complete gold units, of
                         0.5 raised to the number of earlier chunks covering that unit
alpha-DCG@K = sum of novelty gains / log2(rank + 1) through rank K
alpha-nDCG@K = min(1, observed alpha-DCG@K / greedy-baseline alpha-DCG@K)
```

Here, repetition is not decided by text similarity. It means that a chunk
covers an evidence-unit ID already covered by a higher-ranked chunk.

**Example:** If the first two chunks contain the same evidence and the third
contains different evidence, the second chunk is discounted. Moving the
different evidence to rank 2 improves the score.

For two required units, consider three corpus chunks `A={G1}`, `B={G1}`,
and `C={G2}`. Order `A,B,C` has novelty gains `1,0.5,1`; greedy order
`A,C,B` has `1,1,0.5`. At `@3`, their discounted sums are approximately
`1 + 0.5 × 0.631 + 1 × 0.5 = 1.815` and
`1 + 1 × 0.631 + 0.5 × 0.5 = 1.881`. The reported ratio is about
`0.965`, rather than `1`, even though both orders cover both units.

This is diagnostic rather than primary because novelty is a separate objective
from relevance ranking. Repeated relevant evidence is not automatically a
retrieval failure, and a low novelty score does not establish that the chunks
are irrelevant or that repetition harms the downstream answer.
The metric is calculated for every answerable question with at least one gold
evidence unit, using the same eligible questions and documents as the three
primary retrieval metrics. With one unit, its first occurrence receives full
credit and later occurrences receive discounted credit; the same rule is used
in the greedy normalization.

Apply the [shared greedy normalization and score-cap rules](../../metrics.md#retrieval-evidence-quality)
to this pair's complete corpus. A score of one means the ranking meets or
exceeds that greedy baseline, not necessarily an exact ideal.
The [data-review checklist](../../pending-data-review.md) records the development-only
comparison with exact normalization before the method is frozen for held-out
evaluation.

**Range and direction:** `[0, 1]`; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Zero greedy-baseline gain over the complete candidate corpus | No chunk covers any complete gold unit for an eligible question. | `0` (`scored`) | Keep fragmentation in the quality result using the explicit zero-denominator convention. |
| Observed gain exceeds the greedy normalization baseline | The uncapped ratio is `1.02`. | Report `1` (`scored`); retain raw `1.02` and the greedy-baseline-exceeded flag. | The deterministic greedy baseline is approximate. Exceeding it is not itself a failed evaluation; the raw ratio and flag are per-question diagnostics, not another primary metric. |

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
Evidence-slice and aggregate status follow
[this suite's completeness rules](#aggregation-and-comparison-completeness).

## Why the metrics are not interchangeable

| Metric | What changes it | What it does not answer directly |
|---|---|---|
| nDCG@3/@5 | The ranks and complete evidence-unit counts of chunks | Whether all distinct evidence units were found or how much extra text was returned |
| Evidence-unit Recall@3/@5 | Whether each required evidence unit appears inside the cutoff | Whether the recovered units were ordered well or surrounded by extra context |
| Evidence-token Precision@3/@5 | How much returned text is annotated evidence | Whether all required units were found or ranked early |
| alpha-nDCG@3/@5 | When evidence appears and how often it has already been covered | Whether repetition actually harms the downstream answer |

The three primary families are complementary. Reordering the same chunks can
change nDCG without changing recall or token precision. Adding non-evidence text
can lower token precision without changing relevance order or recovered units.
Missing one required unit lowers recall even when the remaining relevant chunks
are ranked early. Alpha-nDCG deliberately overlaps with nDCG, but adds a
repeated-evidence discount as a diagnostic, not a primary relevance criterion.

## Worked candidate interpretation

Suppose a valid pair reports:

```text
Evidence-unit Recall@5       = 0.82
Evidence-token Precision@5  = 0.44
nDCG@5                       = 0.81
alpha-nDCG@5                 = 0.74
```

The nDCG result summarizes how early evidence-rich chunks appear. After question
scores are averaged within documents and then across documents, average
evidence recovery is `0.82` and average evidence-token precision is `0.44`.
These describe completeness and concentration, not pooled percentages of all
corpus evidence units or returned tokens. Alpha-nDCG uses the same answerable
questions and describes novelty under its repeated-evidence discount. Its
value is not directly comparable with nDCG because the gains and normalization
differ. Inspect the evidence matches to
identify repeated coverage; neither the score nor the gap establishes that
repetition harms answers. The same interpretation is performed separately at
`@3`. No formula combines these values.

## Ranking consistency

### Ranking Repeatability

**Question:** How often do repeated requests deliver the same valid ordered
top-five chunks?

Apply [shared pairwise repeatability](../../metrics.md#pairwise-repeatability)
to all three unordered pairs of measured attempts for each answerable question.
Compare ordered top-five IDs, or all available IDs in a legitimately smaller
corpus, not bit-identical scores. Average question pair scores within documents,
then macro-average documents. Every attempt executes retrieval; it does not
return a cached earlier ranking.

```text
Question Ranking Repeatability = identical valid top-five pairs / scheduled pairs
```

**Example:** `A,A,B` and `failure,A,A` each have one agreeing valid pair among
three, giving `1/3`. Three executed failures give zero. Three identical valid
rankings give one even when they retrieve irrelevant text.

This diagnoses repeatable delivery, not relevance. A different valid order does
not invalidate a candidate; unknown/duplicate IDs, incorrect provenance, and
other output-integrity violations retain their separate requirements. This
suite examines the selected top five; retrieval–reranking examines its complete
normally top-20 ranking. Those declared depths are not interchangeable.

**Range and direction:** `[0,1]`; higher is more repeatable, not necessarily
higher quality.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No attempt pairs are scheduled | One-attempt smoke | `null` (`inapplicable`) | No repeatability task was measured. |

Missing required execution or records follows the shared pair rule, not a
surviving-pair average. Supported source-document intervals follow
[this suite's CI table](#which-values-receive-an-interval).

## Operational performance

The phase calls preparation of the searchable embedding matrix **corpus build**.
It is not the later vector-server indexing experiment.

Operational availability follows the
[shared rules](../../metrics.md#shared-availability-rules). Each build component
requires its own completed phase; total corpus-build values require the complete
searchable matrix. Query-component latency uses each query's median successful
completed component duration, and complete-query latency uses successful complete
requests. A completed embedding can survive a later search failure. Keep each
cohort's support; component success does not establish whole-query success.
Apply the shared
[percentile](../../metrics.md#latency-percentiles) and
[resource](../../metrics.md#resource-measurements) definitions with the lifecycle
windows below. Partial-resource scope is recorded as specified by the
[run structure](methodology.md#mlflow-result-structure); CI applicability remains
in this page's [interval table](#which-values-receive-an-interval).

Use the [shared timing boundaries and naming convention](../../metrics.md#latency-and-warm-latency).
Build components explain total preparation, and query components explain total
query time; they are not independent quality benefits to count again. Preserve
assembly, validation, and dispatch overhead in the observed enclosing interval.

### Cold Model-Load Time

**Question:** What startup delay makes the embedding model and tokenizer ready?

Start before their construction/loading and stop at full readiness on the
requested device, after required placement checks and synchronization. Exclude
chunking, corpus embedding, warmup, and measured queries. Cold means a fresh
worker/model, not forcibly cleared operating-system disk caches.

**Example:** Construction starts at second `0` and readiness occurs at second
`4`: Cold Model-Load Time is `4 seconds`. The corpus-build timer starts afterward.

**Range and direction:** Non-negative seconds; lower is better at equal scope.

**Edge cases and availability:**

[Shared completed-phase rules apply.](../../metrics.md#dependent-metrics-and-partial-observations)
A completed load survives later failure; interrupted loading retains elapsed-
to-failure separately. There is no additional metric-specific empty convention.

### Chunking Time

**Question:** How expensive is creating the complete corpus's passages?

Time canonical-document processing through the complete chunk inventory: boundary
selection, overlaps, text, IDs, source offsets, and chunk metadata. Include native
checks and embedding calls used to choose semantic boundaries. Exclude model
loading and final chunk-vector production. The completed inventory must satisfy
its source/offset contract; a partial inventory is not a completed phase.

**Example:** Ten documents become 800 complete chunk records in `6 seconds`.
That is chunk creation cost, not the time to create the final 800 search vectors.
The semantic strategy's boundary embeddings belong to this phase.

**Range and direction:** Non-negative seconds; lower is better at equal quality.

**Edge cases and availability:**

[Shared completed-phase rules apply.](../../metrics.md#dependent-metrics-and-partial-observations)
No additional metric-specific exception is needed. Retain a completed chunking
duration if later vector production fails; invalid empty inventories remain
runtime-integrity failures, not new duration conventions.

### Chunk-Embedding Time

**Question:** How expensive is producing the final searchable chunk vectors?

After the chunk inventory is ready, time final-chunk native preparation and
length checks, model inference, normalization, stored-dtype conversion, and
required transfer/synchronization through the complete final vector set.
Semantic-boundary embeddings are not counted again. Searchable matrix/metadata
assembly and final corpus validation remain inside the enclosing build timer.

**Example:** The 800 completed chunks require `24 seconds` to produce their
final vectors. Embedding 700 before interruption does not establish an 800-chunk
completion time; its known progress remains diagnostic evidence.

**Range and direction:** Non-negative seconds; lower is better at equal quality.

**Edge cases and availability:**

[Shared completed-phase rules apply.](../../metrics.md#dependent-metrics-and-partial-observations)
No additional metric-specific exception is needed. Missing vectors and invalid
native inputs retain their separate integrity/input outcomes.

### Corpus-Build Time

**Question:** How long does preparation of the complete searchable corpus take?

Timing begins after the model and tokenizer are loaded and ends when the complete
searchable corpus is ready. It includes chunk creation, document embedding, and
embedding-matrix/metadata assembly and validation, and required device
synchronization. It excludes downloads, full data validation and validated-input
verification, model/tokenizer loading, environment installation, query warmup,
and offline metric scoring. Semantic
boundary embedding, native-input checks on generated inference inputs, and
first-use work performed during corpus build belong to this timer. Build the
corpus once; query repetitions do not rebuild it.

**Example:** Chunking takes `6 seconds`, final chunk embedding `24 seconds`, and
assembly/validation `2 seconds`. Corpus-Build Time is the observed `32 seconds`,
not only the embedding component. A completed build taking 80 seconds similarly
reports `80` as the complete preparation cost.

**Range and direction:** Non-negative seconds; lower is better at equal quality.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
See this page's [operational completion rules](#operational-performance-1).

### Corpus-Build Throughput

**Question:** How quickly does the pair prepare the fixed source workload?

Corpus-build throughput divides the number of original source tokens in the
frozen corpus by corpus-build wall-clock seconds. Original source tokens use the
fixed evaluation tokenizer. Indexed-token occurrences are not used because
overlap would otherwise reward a candidate for duplicating text.
Use the unrounded completed duration.
The required nonempty corpus is checked during
[data validation](../../data-validation.md#chunking-and-embedding-inputs).

**Example:** Preparing 100,000 source tokens in 80 seconds gives 1,250 source
tokens/second, regardless of how many overlapping chunk-token occurrences were
embedded.

On the same fixed source workload this rate is derived from Corpus-Build Time;
it provides a useful capacity unit, not an independent advantage to count twice.

**Range and direction:** Non-negative source tokens/second; higher is better at
equal quality.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Completed build has zero measured duration | The retained, unrounded completed duration is `0` seconds. | `null` (`unavailable`) | The throughput denominator is zero. Do not substitute a tiny constant or report infinity. |

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
Incomplete builds and invalid timers follow the
[operational completion rules](#operational-performance-1).

### Query-Embedding Latency p50/p95

The component starts with preparation of the current question and ends when its
validated query vector is ready for search. Include native tokenization/length
checks, embedding inference, normalization, and required synchronization. Exclude
corpus embedding and vector search. Use each query's median completed component
duration before applying the shared percentiles.

#### Query-Embedding Latency p50

**Question:** What typical query time belongs to preparing its search vector?

Take p50 of the per-query query-embedding observations.

**Example:** Query observations `8,10,12,14,16 ms` give p50 `12 ms`. A request
whose embedding completes in 12 ms keeps that component observation if search
subsequently fails.

**Range and direction:** Non-negative milliseconds; lower is better at equal quality.

**Edge cases and availability:**

Apply [shared availability](../../metrics.md#shared-availability-rules) and
[percentile rules](../../metrics.md#latency-percentiles). One observation is its
own p50. If no completed component observation contributes, the aggregate is
`null/unavailable`, not zero.

#### Query-Embedding Latency p95

**Question:** What is the slow tail of question-vector preparation?

Take p95 over the same per-query component observations, not all attempts pooled.

**Example:** `8,10,12,14,16 ms` gives position `3.8` and p95
`14 + 0.8 × (16 - 14) = 15.6 ms`.

**Range and direction:** Non-negative milliseconds; lower is better at equal quality.

**Edge cases and availability:**

Apply the shared [percentile](../../metrics.md#latency-percentiles) and
[availability](../../metrics.md#shared-availability-rules) rules. One observation
is its own p95; sufficient source support is separately required for bounds.

### Vector Search Latency p50/p95

Start from the ready query vector and searchable matrix; stop after cosine
scoring, deterministic ordering/tie-breaking, top-20 selection, output validation,
and required synchronization. This suite executes exact NumPy cosine search;
the display name does not imply a vector server or approximate index. Use each
query's median successful completed search duration.

#### Vector Search Latency p50

**Question:** What typical query time belongs to searching the matrix?

Take p50 over the per-query vector-search observations.

**Example:** `4,6,8,10,12 ms` gives p50 `8 ms`. This can grow when a chunker
creates more vectors, even with an unchanged query-embedding model.

**Range and direction:** Non-negative milliseconds; lower is better at equal quality.

**Edge cases and availability:**

Apply [shared availability](../../metrics.md#shared-availability-rules) and
[percentile rules](../../metrics.md#latency-percentiles). Search never entered
after embedding failure is unexecuted, not a fabricated search failure.

#### Vector Search Latency p95

**Question:** What is the slow tail of matrix search and result selection?

Take p95 over the same per-query search observations.

**Example:** `4,6,8,10,12 ms` gives p95 `10 + 0.8 × (12 - 10) = 11.6 ms`.
This is not complete-query p95 minus query-embedding p95.

**Range and direction:** Non-negative milliseconds; lower is better at equal quality.

**Edge cases and availability:**

Apply the shared [percentile](../../metrics.md#latency-percentiles) and
[availability](../../metrics.md#shared-availability-rules) rules. A completed
search survives later independent offline-scoring failure.

### p50 Warm Query Latency

**Question:** What is typical warm query-embedding and exact-search latency?

After one representative query warmup, take each answerable query's median
successful measured latency as its observation, then calculate p50 across those
observations. Timing includes tokenization and native-input checks, query
embedding, cosine scoring, deterministic ordering, top-20 selection, output
validation, and required device synchronization. Loading, corpus build, warmup,
full data validation and validated-input verification, and offline quality
scoring are outside it.

**Example:** Successful attempts of 20 and 24 milliseconds give a query
observation of 22 milliseconds. Query observations of 10, 20, 30, 40, and 50
milliseconds give p50 of 30 milliseconds: the typical successful query time,
not an arithmetic mean of all measured attempts.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
See the [successful-observation and percentile rules](#operational-performance-1)
and [CI support requirements](#chunking-and-embedding-confidence-intervals).

### p95 Warm Query Latency

**Question:** What is the slow-tail warm query-embedding and exact-search time?

Use the same per-query successful-attempt medians as p50 and calculate their
linearly interpolated 95th percentile. Do not pool repetitions as independent
queries or include elapsed times from failed requests.

Component timings explain this complete path; their p95 values must not be
added to derive its p95.

**Example:** Query observations of 10, 20, 30, 40, and 50 milliseconds have
zero-based percentile position `(5 - 1) × 0.95 = 3.8`, giving
`40 + 0.8 × (50 - 40) = 48` milliseconds. This describes the slow tail rather
than the typical query; p50 is not a mean and does not replace p95.

p99 is not reported in this phase. The vector-server load benchmark measures
it under controlled concurrency; it is not part of this pair's metric contract.

**Range and direction:** Non-negative milliseconds per query; lower is better at
equal quality.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
See the [successful-observation and percentile rules](#operational-performance-1)
and [CI support requirements](#chunking-and-embedding-confidence-intervals).

### Peak Process-Tree RAM

**Question:** How much peak host memory does candidate execution require?

Use [Peak Process-Tree RAM](../../metrics.md#peak-process-tree-ram). This suite's
window starts before model loading, includes corpus build, warmup, and measured
queries, and ends before offline scoring/bootstrap analysis.

**Example:** A highest simultaneous process-tree sample of 2,800 MiB reports
2,800 MiB, not the sum of separately observed process peaks.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
See the [resource-sampling and scope rules](#operational-performance-1).

### Peak Device VRAM

**Question:** How much of the assigned GPU's memory is occupied at the peak?

Use [Peak Device VRAM](../../metrics.md#peak-device-vram) over the same lifecycle
window as this pair's RAM measurement.

**Example:** An observed 2,900 MiB peak reports 2,900 MiB without subtracting
the idle baseline. It is the highest sampled value, not a promise that no higher
allocation occurred between samples.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
See the [resource-sampling and scope rules](#operational-performance-1).

## Workload and storage descriptors

These values explain operational outcomes, not retrieval quality. Numeric
descriptors are chartable; dtype is a recorded parameter. None independently
selects a winner or compensates for worse quality.

Input counts describe the full verified workload. Generated-chunk descriptors
may remain known even if embedding later fails. If chunk generation stops early,
label any observed inventory counts/summaries as partial in artifacts rather
than presenting them as full-corpus totals. A missing inventory is unavailable,
not an observed zero; a completed generation that produces no chunks records
Chunk Count = 0 and an invalid empty-index outcome.
Verified input counts remain known even if candidate execution later fails.
An incomplete matrix's final payload size is null/unavailable: retain known
progress in existing artifacts without introducing a partial-matrix-bytes metric.
Missing measurements and records otherwise follow the
[shared availability rules](../../metrics.md#shared-availability-rules).

### Corpus Counts

Record document, answerable-question, and unanswerable-question counts
separately. Only answerable questions enter this benchmark's retrieval-quality
and query-latency evaluation; unanswerable questions retained in a shared input
manifest are a workload count, not measured retrieval requests.

#### Document Count

**Question:** How many documents does the frozen searchable corpus contain?

Count the document records in the verified corpus once each. Documents remain
searchable even when their questions are unanswerable; this is an input count,
not the count of documents successfully processed by the candidate.

**Example:** A corpus of 40 documents reports `40`, regardless of how many
questions each document supplies. The count describes the searchable workload,
not independent query or attempt support.

**Range and direction:** Non-negative integer count; descriptive.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
See the [verified-workload and inventory rules](#workload-and-storage-descriptors-1).

#### Answerable-Question Count

**Question:** How many questions are scheduled for retrieval evaluation?

Count the verified manifest's answerable questions once each, not once per
measured repetition. These questions supply retrieval-quality and query-latency
tasks; the count does not depend on how many requests later succeed.

**Example:** A manifest with 120 answerable questions supplies `120` evaluated
queries. Three measured attempts per query do not change this input count to
360; attempts are separate execution records.

**Range and direction:** Non-negative integer count; descriptive.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
See the [verified-workload and inventory rules](#workload-and-storage-descriptors-1).

#### Unanswerable-Question Count

**Question:** How many questions are retained for later stages but not executed
in this phase?

Count the verified manifest's unanswerable questions once each. This descriptor
does not add retrieval requests or empty-reference quality comparisons.

**Example:** A manifest with 120 answerable questions and 30 unanswerable
questions reports an unanswerable count of `30` but supplies 120 evaluated
queries, not 150. The descriptor explains the difference between the shared
question inventory and this phase's workload.

**Range and direction:** Non-negative integer count; descriptive.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
See the [verified-workload and inventory rules](#workload-and-storage-descriptors-1).

### Source Tokens

**Question:** How much original source text does the pair process?

Count each canonical document once using `tiktoken:cl100k_base`, before chunking
or overlap. The count is the same for every pair using the same frozen corpus
and is the numerator of Corpus-Build Throughput.

**Example:** A corpus containing 100,000 source tokens still has 100,000 source
tokens when a chunker repeats some text through overlap.

**Range and direction:** Non-negative integer count; descriptive.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
See the [verified-workload and inventory rules](#workload-and-storage-descriptors-1).

### Indexed-Token Occurrences

**Question:** How much source text is represented across all generated chunks?

Sum the fixed-tokenizer counts of the chunk texts. Text repeated in overlapping
chunks counts each time. Model-specific prefixes and special inference tokens
are not source-token occurrences; native input lengths belong to the separate
compatibility checks.

**Example:** A 100,000-token corpus may produce 130,000 indexed-token
occurrences because overlap repeats source text. This helps explain embedding
work without inflating source-token throughput.

**Range and direction:** Non-negative integer count; descriptive.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
See the [generated-inventory rules](#workload-and-storage-descriptors-1).

### Chunk Count

**Question:** How many searchable passages does the strategy create?

Count generated chunks. A complete index contains one embedding row per chunk;
the count therefore explains matrix size and exact-search work.

**Example:** A strategy producing 800 chunks requires 800 chunk-vector rows in
its complete matrix.

**Range and direction:** Non-negative integer count; descriptive.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
The [generated-inventory rules](#workload-and-storage-descriptors-1) distinguish
a missing inventory from an observed zero count and preserve partial scope;
an empty index is not an integrity-valid build.

### Mean Chunk Tokens

**Question:** How many tokens does a generated chunk contain on average?

Count each generated chunk with `tiktoken:cl100k_base`, then divide the sum
of those lengths by Chunk Count. Overlap counts again in each chunk. This is
an arithmetic mean over chunks, not a document-macro score or configured target.

**Example:** Chunk lengths `100, 200, 300, 400, 500` give
`(100 + 200 + 300 + 400 + 500) / 5 = 300 tokens/chunk`.
The mean summarizes average size but can hide long chunks.

**Range and direction:** Non-negative tokens/chunk; descriptive.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No chunks to average | The completed generated inventory has Chunk Count = `0`. | `null` (`unavailable`) | The mean's denominator is zero; this is not an observed mean of zero tokens. |

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
See the [generated-inventory and partial-scope rules](#workload-and-storage-descriptors-1).

### p95 Chunk Tokens

**Question:** How long are chunks near the upper end of the size distribution?

Sort the same fixed-tokenizer chunk lengths and take their linearly
interpolated 95th percentile. This is a size percentile, not a confidence bound
or a native model-input limit.

**Example:** For `100, 200, 300, 400, 500`, the percentile lies 80% of the
way from `400` to `500`: `400 + 0.8 × 100 = 480 tokens`. The mean is `300`,
so a mean alone would conceal the longer passages. An interpolated size need
not be an integer even though each observed chunk length is an integer.

**Range and direction:** Non-negative tokens; descriptive.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No chunk-length observations | The completed generated inventory has Chunk Count = `0`. | `null` (`unavailable`) | There is no size distribution from which to calculate p95, not a measured zero-token percentile. |

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
The [shared percentile convention](../../metrics.md#latency-percentiles)
makes a single chunk's p95 equal its length. See the
[generated-inventory and partial-scope rules](#workload-and-storage-descriptors-1)
and [descriptor CI rules](#which-values-receive-an-interval).

### Mean Retrieved Context Tokens@3/@5

**Question:** How much text do the selected chunks return per query on average?

Apply the [shared retrieved-context workload contract](../../metrics.md#retrieved-context-workload)
to this pair's first ranking. Sum fixed-tokenizer lengths inside each prefix,
then average those per-query totals separately at @3 and @5. This is not average
chunk size or document-macro quality.

```text
Mean Retrieved Context Tokens@K = sum of complete query-prefix token totals
                                 / contributing queries
```

**Example:** Chunk lengths `180,220,200,300,100` give that query 600 tokens at
@3 and 1,000 at @5. Across ten queries, nine 500-token contexts and one 5,000-token
context give mean `950`, while p50 would be `500`. The mean describes average
volume, including the large context; it is not a latency prediction or limit.

**Range and direction:** Non-negative tokens/query; descriptive.

**Edge cases and availability:**

The valid workload has a positive contributing denominator. No additional
valid-empty convention is needed. Use shared
[context-count availability](../../metrics.md#retrieved-context-workload) for
missing/partial totals; later rankings never repair a failed first observation.

### p95 Retrieved Context Tokens@3/@5

**Question:** How large is the upper tail of selected context volumes?

Take the [shared interpolated p95](../../metrics.md#latency-percentiles) of the
same complete per-query totals, independently for each cutoff. This is a size
percentile, not a confidence interval or model-native input capacity.

**Example:** @3 totals `400,600,800` give p95 `780 tokens`; @5 totals
`600,900,1,200` give `1,170 tokens`. A mean can conceal the larger contexts this
descriptor makes visible. It imposes no context budget.

**Range and direction:** Non-negative tokens; descriptive. Interpolation may
produce a fractional count, for example `600.95` from totals `600,601`.

**Edge cases and availability:**

One contributing query's p95 equals its count under the shared percentile rule.
If eligible queries exist but no complete total contributes, the aggregate is
`null/unavailable`, not a zero-token context. Missing/partial totals follow
[shared context-count availability](../../metrics.md#retrieved-context-workload).

### Embedding Dimension and Dtype

**Question:** How many coordinates does each vector contain, and how are they
stored?

Record the model's embedding dimension and the matrix's stored dtype. Inference
dtype and stored dtype are separate settings: CUDA inference can use FP16 while
the resulting vectors are stored as FP32.

**Example:** A dimension of 768 with stored `float32` means each vector contains
768 four-byte coordinates.

**Range and direction:** Dimension is a positive integer; dtype is a parameter.
Both are descriptive, not higher-is-better quality measures.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)

### Embedding-Matrix Bytes

**Question:** How large is the complete raw chunk-vector payload?

Record the complete dense matrix's raw payload bytes, excluding metadata,
serialization headers, and model weights. For the stored contiguous FP32
matrix, cross-check its actual payload against chunk count, embedding dimension,
and four bytes per coordinate.

```text
Embedding-Matrix Bytes = chunk rows × coordinates per row × bytes per coordinate
```

**Example:** 1,000 vectors with 768 FP32 coordinates occupy 3,072,000 raw
payload bytes. A serialized `.npy` file can legitimately be larger because it
also contains a header; its file size is not this descriptor.

Matrix storage inconsistent with its declared shape, dtype, and storage format
is an integrity failure. Raw payload checks must not be applied as exact
file-size checks to a serialized or compressed artifact.

**Range and direction:** Non-negative integer bytes; descriptive.

**Edge cases and availability:**

[Shared availability rules apply.](../../metrics.md#shared-availability-rules)
Only a complete matrix has a final payload-size observation; see the
[storage-availability rules](#workload-and-storage-descriptors-1).

## Aggregation and comparison completeness

For each quality metric, first average defined first-ranking question scores
within a source document, then macro-average those document values. Calculate
every metric and evidence slice independently. A complete quality result covers all
planned answerable questions; all four families share reference eligibility,
but retain metric-specific contributing and unavailable counts if scoring fails.

| Case | Aggregate and comparison treatment |
|---|---|
| No eligible questions in an optional slice | Null/inapplicable; report zero eligible support. |
| Eligible questions exist, none has a usable score | Null/unavailable; preserve all failed/unavailable records. |
| Some required question scores are unavailable | Retain a partial diagnostic aggregate and exact support; do not place it in the complete-result leaderboard or paired selection comparisons. |
| Missing required request or result records | Execution is incomplete; observed results do not repair absent records. |

For example, quality of `0.90` on 8 of 10 queries is conditional on those eight
usable outputs and cannot establish superiority over `0.85` on all 10. Keep the
partial estimate in artifacts with its status, rather than logging it as a
normal `quality.*` aggregate. Unavailable scores are neither imputed as zero
nor silently discarded from the account of what was scheduled.

Apply the [shared completeness distinction](../../metrics.md#execution-accounting-and-result-completeness)
to execution coverage, quality coverage, and integrity. A later request failure
does not automatically invalidate complete first-ranking quality or a fully
accounted invocation. Missing required first scores remain partial; declared
hardware/integrity violations remain enforced. Normal quality comparisons need
complete valid quality, and the engineer reviews recorded failures separately.

## Chunking and embedding confidence intervals

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Development, validation, and locked nDCG, Evidence-unit Recall, Evidence-token Precision, and alpha-nDCG at @3/@5 | When defined and sufficiently supported | Resample source documents and recalculate each aggregate. Partial estimates and any bounds remain diagnostic, not complete selection evidence. |
| Text, table, formula, and mixed evidence slices | When defined and sufficiently supported | Resample eligible source documents with their first-ranking and failure records; do not prefilter to successful contributors. |
| Ranking Repeatability | When complete and sufficiently supported | Resample source documents with every query's scheduled attempt pairs and recalculate the macro pair score. |
| p50/p95 complete-query, query-embedding, and vector-search latency | Conditional | Resample source documents with each component's own query medians; otherwise retain defined descriptive points without bounds. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| Cold load, chunking, chunk embedding, corpus-build time and throughput | No | Each is one completed phase/build observation, not independent repeated builds. |
| Peak Process-Tree RAM and Peak Device VRAM | No | Report the observed peak without invented bounds. |
| Workload and storage descriptors | No | These are observed properties of the candidate and fixed corpus, not sampled quality estimates. |

### Calculation

Apply the [shared CI procedure](../../metrics.md#confidence-intervals) to complete
source documents eligible for the metric/slice, carrying their question,
first-ranking, timing, and failure records. Each quality draw recalculates the
question-then-document macro statistic. Repeatability draws recalculate the
question-then-document pair statistic from all scheduled outcomes. Latency draws
collect the sampled documents' per-query successful component/request medians
and recalculate each p50/p95 separately, not a macro-average of document latencies.

The shared support, undefined-draw, duplicate-draw, and degenerate-bound rules,
including the [support-freeze deadline](../../metrics.md#confidence-intervals),
apply. The
[data-review checklist](../../pending-data-review.md#chunking-embedding-retrieval-and-generation)
tracks those decisions.

### Interpretation

An nDCG@5 of `0.81` with a 95% confidence interval of `[0.77, 0.85]` means
`0.81` is the observed aggregate, while resampling complete source documents
estimates its uncertainty. The interval does not describe the range of
individual-question scores.
