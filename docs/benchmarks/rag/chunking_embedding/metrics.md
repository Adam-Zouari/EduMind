# Chunking and embedding metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Retrieval quality](#retrieval-quality-1)
- [Why the metrics are not interchangeable](#why-the-metrics-are-not-interchangeable)
- [Worked candidate interpretation](#worked-candidate-interpretation)
- [Operational performance](#operational-performance-1)
- [Workload and storage descriptors](#workload-and-storage-descriptors-1)
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
| Peak Process-Tree RAM | Operational | How much total system memory does the pair require? | Lower at equal quality |
| Peak Device VRAM | Operational | How much total GPU memory is occupied while the pair runs? | Lower at equal quality |

### Workload and storage descriptors

| Value | Role | Question answered | Direction |
|---|---|---|---|
| Corpus Counts | Workload descriptor | How many documents and answerable/unanswerable questions were represented in the run? | Descriptive |
| Source Tokens | Workload descriptor | How large was the original corpus before chunk overlap? | Descriptive |
| Indexed-Token Occurrences | Workload descriptor | How much text was embedded after overlap and repeated context were counted? | Descriptive |
| Chunk Count | Workload descriptor | How many searchable vectors and metadata records did the strategy create? | Descriptive |
| Mean/p95 Chunk Tokens | Workload descriptor | What typical and long-tail chunk sizes did the strategy actually produce? | Descriptive |
| Embedding Dimension and Dtype | Storage descriptor | What shape and numeric representation did each vector use? | Descriptive |
| Embedding-Matrix Bytes | Storage descriptor | How much storage did the complete chunk-vector matrix occupy? | Descriptive |

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
candidate chunk contains verified evidence receives zero.

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

### Evidence-token Precision@3/@5

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
in the greedy normalization. Unanswerable questions have no gold evidence and
are excluded from all four retrieval-quality metrics.

Normalization uses a deterministic greedy approximation over the pair's
complete chunk corpus. At each rank, it chooses the unused chunk with the
largest remaining alpha gain, including discounted credit for previously seen
units; ties follow the frozen corpus order. This is an approximation, not a
guaranteed maximum. Scores are capped at `1`, so a perfect score means the
ranking meets or exceeds this greedy baseline, not necessarily the exact ideal.
The [data-review checklist](../../pending-data-review.md) records the development-only
comparison with exact normalization before the method is frozen for held-out
evaluation. The [original alpha-nDCG paper](https://plg.uwaterloo.ca/~gvcormac/novelty.pdf)
describes this practical greedy approximation.

**Range and direction:** `[0, 1]`; higher is better. An eligible question for
which no candidate chunk contains verified evidence receives zero.

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

### Corpus-build elapsed time and throughput

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

### Warm query latency p50 and p95

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

### Peak process-tree RAM and peak device VRAM

**Question:** What peak host-process and total device memory accompany corpus
build and query evaluation?

Peak RAM is the largest sampled resident-memory total across the benchmark
worker and its child processes. Peak Device VRAM uses the shared raw NVML
device-total contract. The artifact identifies the measurement method.

**Range and direction:** non-negative MiB; lower is better at equal quality. CPU-only execution records VRAM as null/inapplicable; missing required CUDA
instrumentation records null/unavailable, never zero.

## Workload and storage descriptors

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

## Chunking and embedding confidence intervals

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Development, validation, and locked nDCG, Evidence-unit Recall, Evidence-token Precision, and alpha-nDCG at @3/@5 | Yes | Resample source documents and recalculate each aggregate. |
| Text, table, formula, and mixed evidence slices | Yes, when enough documents contribute | Resample only the contributing source documents. |
| p50/p95 warm query latency | Conditional | Report only when enough independent source documents support the query-level percentile estimate. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| Corpus-build time and throughput | No | One corpus-build observation cannot estimate uncertainty. |
| Peak Process-Tree RAM and Peak Device VRAM | No | Report the observed peak without invented bounds. |
| Workload and storage descriptors | No | These are observed properties of the candidate and fixed corpus, not sampled quality estimates. |

### Calculation

Quality is first calculated for each answerable question. Questions are averaged
within their source document so a paper with many questions cannot dominate the
result. Development, validation, and locked runs then use 10,000 bootstrap resamples of complete
documents with seed 42 and take the 2.5th and 97.5th percentiles as the 95%
confidence bounds.

If too few documents contribute to a slice or conditional latency interval, the
point estimate remains available but the interval is omitted rather than
reported as artificially precise.

### Interpretation

An nDCG@5 of `0.81` with a 95% confidence interval of `[0.77, 0.85]` means
`0.81` is the observed aggregate, while resampling complete source documents
estimates its uncertainty. The interval does not describe the range of
individual-question scores.
