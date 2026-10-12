# Chunking and embedding metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Eligibility and result status](#eligibility-and-result-status)
- [Retrieval quality](#retrieval-quality-1)
- [Why the metrics are not interchangeable](#why-the-metrics-are-not-interchangeable)
- [Worked candidate interpretation](#worked-candidate-interpretation)
- [Operational performance](#operational-performance-1)
- [Workload and storage descriptors](#workload-and-storage-descriptors-1)
- [Aggregation and comparison completeness](#aggregation-and-comparison-completeness)
- [Chunking and embedding confidence intervals](#chunking-and-embedding-confidence-intervals)

The chunking/embedding experiment evaluates one complete
`chunker|embedding` pair at a time with exact cosine search. Three primary metric
families answer different selection questions; alpha-nDCG is retained as a
novelty diagnostic. None is combined into a weighted score.

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

### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Corpus-Build Time | Operational | How long does chunking, document embedding, and searchable-matrix preparation take? | Lower |
| Corpus-Build Throughput | Operational | How many original source tokens are processed per second? | Higher |
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
| Embedding Dimension and Dtype | Storage descriptor | What shape and numeric representation did each vector use? | Descriptive |
| Embedding-Matrix Bytes | Storage descriptor | How large is the complete matrix's raw vector payload? | Descriptive |

## Eligibility and result status

Only answerable questions with reviewed gold evidence are executed and scored.
Unanswerable questions retained in a shared manifest are input counts for later
stages, not retrieval requests or empty-reference quality comparisons here.
Invalid or missing required evidence is a
[data-validation failure](../../data-validation.md#chunking-and-embedding-inputs),
not a zero or null observation from a candidate that never ran.

Quality uses the designated first measured ranking. A failed first request is
not an empty retrieval result; later success cannot replace it. Conversely, a
later failed repetition does not erase a valid first ranking's quality. Each
attempt keeps its own outcome and timing record.

| Result | Value and treatment |
|---|---|
| Valid first ranking | Score all four quality families once at both cutoffs. Missing evidence in valid retrieved chunks receives the normal poor numeric score. |
| Failed first request or invalid first ranking | Applicable quality is null/unavailable, with the error and planned support retained. |
| Optional evidence slice with no eligible questions | Aggregate is null/inapplicable; no retrieval task exists in that slice. |
| Eligible questions exist, but none has a usable first ranking | Aggregate is null/unavailable, not zero. |
| Scheduled requests or required records are missing | Execution is incomplete; retain known observations without inventing failed or successful requests. |

Generated chunks, vectors, and rankings have separate integrity checks. Exact
search on a valid nonempty index returns `min(20, chunk_count)` unique corpus
chunk IDs with finite scores. An unexpected empty or short ranking, token-free
chunk, invalid ID, or corrupted vector is an execution/integrity failure, not
a scored empty prediction. A legitimate corpus containing fewer than three or
five chunks uses the available chunks at the unchanged cutoff.

Scoring checks require finite, in-range reported quality values. A substantive
nDCG above one indicates an integrity/calculation error, not a normal quality
case; numerical tolerances are code invariants. Alpha-nDCG's retained raw ratio
is explicitly different because its greedy baseline can be exceeded.

Retain null values with `inapplicable`, `unavailable`, or `incomplete` status and
a reason in artifacts. MLflow omits their scalar keys rather than recording
zero. Report scheduled, eligible, scored, failed, and unavailable support.
The [methodology](methodology.md#result-completeness-and-advancement) defines
which completed results can enter selection comparisons.

## Retrieval quality

### nDCG@3/@5

**Question:** How early do evidence-rich chunks appear when repeated evidence
keeps its relevance credit?

nDCG uses linear evidence-count gains: a chunk receives one point for each
distinct required evidence unit it completely covers. A chunk containing no
complete unit receives `0`; one containing three receives `3`. Each unit counts
once within a chunk, but its presence in an earlier chunk does not reduce the
credit of a later chunk. Distinct chunks containing the same evidence or text
can therefore both receive full relevance credit.

The count is used directly, not converted into an exponential gain. Earlier
ranks receive more weight through the usual logarithmic rank discount. For each
question and pair, the ideal ranking sorts the gains of every chunk in that
pair's complete corpus from highest to lowest and applies the same discount and
cutoff. The reported score compares the retrieved ranking with this exact
ideal; the ideal is never restricted to the retrieved top-20 list.

**Example:** Suppose the corpus has only two evidence-bearing chunks: one covers
one required unit and the other covers all three. At `@3`, ordering their gains
as `[3, 1, 0]` gives `1.00`, while `[1, 3, 0]` gives approximately `0.80`. Both
orders recover the same units and return the same text, but the first places
more evidence earlier.

nDCG is primary because this benchmark evaluates relevance ranking rather than
diversity-aware selection. Evidence-unit Recall separately measures distinct
coverage; alpha-nDCG diagnoses novelty without treating repeated relevant
evidence as a relevance-ranking error.

**Range and direction:** `[0, 1]`; higher is better. A question for which no
chunk in the complete candidate corpus contains any complete required evidence
unit receives zero, even though its ideal gain is zero. This convention keeps
boundary fragmentation in the quality result instead of excluding a poor pair.

### Evidence-unit Recall@3/@5

**Question:** How many required evidence units appear anywhere in the first
three or first five chunks, regardless of their order?

Each required evidence unit counts once when at least one chunk inside the
cutoff contains it completely. Repeated copies do not add credit, and partial
fragments do not count as recovered units.

**Example:** If the first three chunks recover one of three required units and
the next two recover another, Recall@3 is approximately `0.33` and Recall@5 is
approximately `0.67`.

This primary metric catches rankings that look good near the top but still miss
part of the evidence needed for a complete answer.

**Range and direction:** `[0, 1]`; higher is better.
Every eligible question has at least one gold unit, so its denominator is
positive. Recovering no complete units gives zero normally; an answerable
question lacking gold units must be repaired before execution.

### Evidence-token Precision@3/@5

**Question:** What share of the tokens returned in the first three or first five
chunks is relevant evidence?

Measure canonical chunk text with `tiktoken:cl100k_base`. A token occurrence is
relevant when its source span has positive overlap with a verified evidence
interval in the same document. Count it once even if several gold intervals
overlap it. All returned source-token occurrences form the denominator; text
repeated in different chunks counts each time in both numerator and denominator.
Native model prefixes and special inference tokens are not source content.

**Example:** If 160 of 600 retrieved tokens are relevant evidence, precision is
approximately `0.27`. The remaining tokens are additional context not counted
as relevant evidence by the reference annotations.

This primary metric does not impose a context budget or penalize a chunk merely
for being large. It reports how concentrated the returned context is around the
verified evidence.

**Range and direction:** `[0, 1]`; higher is better. Valid returned text with no
evidence-bearing tokens receives zero. No valid returned tokens violates this
benchmark's output contract and gives null/unavailable, not a scored zero.

### Alpha-nDCG@3/@5

**Question:** How early does required evidence appear when repeated coverage
receives less credit?

Alpha-nDCG rewards useful evidence more when it appears near the top and reduces
the credit for later chunks that repeat the same evidence. With the frozen
`alpha=0.5` setting, each repetition receives half the remaining novelty credit.

**Example:** If the first two chunks contain the same evidence and the third
contains different evidence, the second chunk is discounted. Moving the
different evidence to rank 2 improves the score.

Here, repetition is not decided by text similarity. It means that a chunk
covers an evidence-unit ID already covered by a higher-ranked chunk. Exact
duplicate chunk IDs are forbidden separately by the retrieval contract.

This is diagnostic rather than primary because novelty is a separate objective
from relevance ranking. Repeated relevant evidence is not automatically a
retrieval failure, and a low novelty score does not establish that the chunks
are irrelevant or that repetition harms the downstream answer.
The metric is calculated for every answerable question with at least one gold
evidence unit, using the same eligible questions and documents as the three
primary retrieval metrics. With one unit, its first occurrence receives full
credit and later occurrences receive discounted credit; the same rule is used
in the greedy normalization.

Normalization uses a deterministic greedy approximation over the pair's
complete chunk corpus. At each rank, it chooses the unused chunk with the
largest remaining alpha gain, including discounted credit for previously seen
units; ties follow the frozen corpus order. This is an approximation, not a
guaranteed maximum. Scores are capped at `1`, so a perfect score means the
ranking meets or exceeds this greedy baseline, not necessarily the exact ideal.
Retain the uncapped ratio and a greedy-baseline-exceeded flag in the per-question
artifact whenever capping occurs; they are diagnostic details, not another
primary metric. A raw ratio above one is not itself a failed evaluation.
The [data-review checklist](../../pending-data-review.md) records the development-only
comparison with exact normalization before the method is frozen for held-out
evaluation. The [original alpha-nDCG paper](https://plg.uwaterloo.ca/~gvcormac/novelty.pdf)
describes this practical greedy approximation.

**Range and direction:** `[0, 1]`; higher is better. An eligible question for
which no corpus chunk covers any complete gold unit receives zero when the
greedy denominator is zero. Failed execution or scoring remains unavailable.

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

## Operational performance

The phase calls preparation of the searchable embedding matrix **corpus build**.
It is not the later vector-server indexing experiment.

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

**Example:** A completed build taking 80 seconds reports `80`, even if a later
query fails. If the build itself fails after 30 seconds, completed build time is
null/unavailable; retain the 30-second elapsed-to-failure observation separately.

**Range and direction:** Non-negative seconds; lower is better at equal quality.
A failed build or missing/non-finite timer has no completed build-time value.

### Corpus-Build Throughput

**Question:** How quickly does the pair prepare the fixed source workload?

Corpus-build throughput divides the number of original source tokens in the
frozen corpus by corpus-build wall-clock seconds. Original source tokens use the
fixed evaluation tokenizer. Indexed-token occurrences are not used because
overlap would otherwise reward a candidate for duplicating text.

**Example:** Preparing 100,000 source tokens in 80 seconds gives 1,250 source
tokens/second, regardless of how many overlapping chunk-token occurrences were
embedded.

**Range and direction:** Non-negative source tokens/second; higher is better at
equal quality. No completed build or a zero/invalid completed duration gives
null/unavailable. Use the unrounded duration; do not substitute a tiny constant
or report infinity. An empty required corpus fails data validation.

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
observation of 22 milliseconds even if its third attempt fails. That failure's
elapsed time remains a failure record, not a completed retrieval latency.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality. A query with no successful timed attempt has no completed latency.
With no contributing query observations, p50 is null/unavailable. With one,
p50 equals that observation. Missing required timing records are incomplete.

### p95 Warm Query Latency

**Question:** What is the slow-tail warm query-embedding and exact-search time?

Use the same per-query successful-attempt medians as p50 and calculate their
linearly interpolated 95th percentile. Do not pool repetitions as independent
queries or include elapsed times from failed requests.

**Example:** Query observations of 10, 20, 30, 40, and 50 milliseconds give p50
of 30 and descriptive p95 of 48 milliseconds. With so few observations, the tail
estimate is fragile; interval eligibility still follows the frozen independent-
document support requirement, not the number of bootstrap draws.

p99 is not reported in this phase. The vector-server load benchmark measures
it under controlled concurrency; it is not part of this pair's metric contract.

**Range and direction:** Non-negative milliseconds per query; lower is better at
equal quality. With no observations, p95 is null/unavailable; with one, it equals
that observation. Small samples retain descriptive percentiles, with null
confidence bounds when independent support is insufficient. p50 is not a mean
and does not replace p95.

### Peak Process-Tree RAM

**Question:** How much peak host memory does candidate execution require?

Take the maximum simultaneous sampled resident-memory sum across the worker and
its child processes. Do not add each process's independent peak. Monitoring
starts before model loading, includes corpus build, warmup, and measured query
execution, and stops before offline scoring and bootstrap analysis.

**Example:** A highest simultaneous process-tree sample of 2,800 MiB reports
2,800 MiB. If execution stops early, retain the observed value with partial
measurement scope rather than claiming completion of the whole workload.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
No usable samples gives null/unavailable. Partial-window peaks retain their
window, method, and failure reason in artifacts and the scope tag described in
the [run structure](methodology.md#mlflow-result-structure).

### Peak Device VRAM

**Question:** How much of the assigned GPU's memory is occupied at the peak?

Use the shared raw NVML `memory.used` device-total contract over the same
execution window. Record method, GPU identity, idle baseline, and samples;
neither subtract the baseline nor infer model placement from this value.

**Example:** An observed 2,900 MiB peak before an OOM remains 2,900 MiB with
partial scope. It is the highest sampled value, not a promise that no higher
allocation occurred between samples or that the remaining workload would fit.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
CPU-only execution records null/inapplicable; missing required CUDA telemetry
records null/unavailable, never zero. Preserve valid partial-window observations
on failed children. A hardware-limit breach does not clamp or erase measured
peaks; resource eligibility remains a separate gate.

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

### Corpus Counts

**Question:** How many documents and questions does the frozen workload contain?

Record document, answerable-question, and unanswerable-question counts
separately. Only answerable questions enter this benchmark's retrieval-quality
and query-latency evaluation; unanswerable questions retained in a shared input
manifest are a workload count, not measured retrieval requests.

**Example:** A manifest with 40 documents, 120 answerable questions, and 30
unanswerable questions supplies 120 evaluated queries, not 150.

**Range and direction:** Non-negative integer counts; descriptive. Verified
input counts remain known even if candidate execution later fails.

### Source Tokens

**Question:** How much original source text does the pair process?

Count each canonical document once using `tiktoken:cl100k_base`, before chunking
or overlap. The count is the same for every pair using the same frozen corpus
and is the numerator of Corpus-Build Throughput.

**Example:** A corpus containing 100,000 source tokens still has 100,000 source
tokens when a chunker repeats some text through overlap.

**Range and direction:** Non-negative integer count; descriptive.

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

### Chunk Count

**Question:** How many searchable passages does the strategy create?

Count generated chunks. A complete index contains one embedding row per chunk;
the count therefore explains matrix size and exact-search work.

**Example:** A strategy producing 800 chunks requires 800 chunk-vector rows in
its complete matrix.

**Range and direction:** Non-negative integer count; descriptive. Zero generated
chunks is an observed count, not a successful empty index for the required
nonempty workload.

### Mean and p95 Chunk Tokens

**Question:** How large are typical chunks and the long tail of the chunk sizes?

Calculate the arithmetic mean and the linearly interpolated 95th percentile of
generated chunk lengths using `tiktoken:cl100k_base`. These describe actual
boundaries, not just the configured target size.

**Example:** Mean chunk length of 310 tokens and p95 of 500 tokens show that
most chunks are shorter than the largest passages the strategy produces.

**Range and direction:** Non-negative token counts; descriptive. With no
generated chunks, both summaries are null/unavailable rather than zero; retain
Chunk Count = 0 and the execution failure separately.

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

### Embedding-Matrix Bytes

**Question:** How large is the complete raw chunk-vector payload?

Record the complete dense matrix's raw payload bytes, excluding metadata,
serialization headers, and model weights. For the stored contiguous FP32
matrix, cross-check its actual payload against chunk count, embedding dimension,
and four bytes per coordinate.

**Example:** 1,000 vectors with 768 FP32 coordinates occupy 3,072,000 raw
payload bytes. A serialized `.npy` file can legitimately be larger because it
also contains a header; its file size is not this descriptor.

Matrix storage inconsistent with its declared shape, dtype, and storage format
is an integrity failure. Raw payload checks must not be applied as exact
file-size checks to a serialized or compressed artifact.

**Range and direction:** Non-negative integer bytes; descriptive. If no complete
matrix exists, its final payload size is null/unavailable. Record the build
failure and known progress in existing artifacts; no partial-matrix-bytes metric
is introduced.

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

Quality, successful-latency support, and execution validity are separate. A
later request failure may leave first-ranking quality complete while still
failing the candidate's run-level execution contract. Only complete, valid
children enter normal comparisons and engineer advancement decisions.

## Chunking and embedding confidence intervals

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Development, validation, and locked nDCG, Evidence-unit Recall, Evidence-token Precision, and alpha-nDCG at @3/@5 | When defined and sufficiently supported | Resample source documents and recalculate each aggregate. Partial estimates and any bounds remain diagnostic, not complete selection evidence. |
| Text, table, formula, and mixed evidence slices | When defined and sufficiently supported | Resample eligible source documents with their first-ranking and failure records; do not prefilter to successful contributors. |
| p50/p95 warm query latency | Conditional | Report bounds only when enough independent source documents support the query-level percentile; otherwise retain any defined descriptive point estimate without bounds. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| Corpus-build time and throughput | No | One corpus-build observation cannot estimate uncertainty. |
| Peak Process-Tree RAM and Peak Device VRAM | No | Report the observed peak without invented bounds. |
| Workload and storage descriptors | No | These are observed properties of the candidate and fixed corpus, not sampled quality estimates. |

### Calculation

Development, validation, and locked use 10,000 bootstrap resamples with seed 42
and the 2.5th and 97.5th percentiles as 95% bounds. These are calculations on
saved records, not new inference. Resample complete source documents eligible
for the metric/slice, retaining their question, first-ranking, timing, and
failure records. Do not prefilter failed questions or treat repetitions as
independent papers.

Each draw recalculates the same question-then-document macro quality statistic.
If a document is drawn twice, its result contributes twice; do not collapse
drawn copies by source ID. Independent support counts refer to the original
contributing documents, not the repeated copies or number of bootstrap draws.
For latency, collect the sampled documents' per-query successful-attempt medians
and recalculate p50/p95; do not substitute a macro-average of document latencies.
Record defined and undefined draw counts per metric. Undefined draws are omitted
only from that metric's bounds, not from saved records or other defined results.

If the frozen minimum independent support is not met, keep any defined point
estimate with null bounds and an insufficient-support reason. A single document
cannot establish an interval by resampling it repeatedly. Slice/support
thresholds are reviewed and frozen before held-out evaluation; the
[data-review checklist](../../pending-data-review.md#chunking-embedding-retrieval-and-generation)
tracks those decisions.

With sufficient support, identical bootstrap estimates retain their calculated
bounds, such as `[0, 0]`, with a degeneracy flag. This means the observed
resamples show no variation, not that future-document performance is certain.
Null estimates or bounds remain null with their reason in artifacts; MLflow
omits the corresponding scalar keys.

### Interpretation

An nDCG@5 of `0.81` with a 95% confidence interval of `[0.77, 0.85]` means
`0.81` is the observed aggregate, while resampling complete source documents
estimates its uncertainty. The interval does not describe the range of
individual-question scores.
