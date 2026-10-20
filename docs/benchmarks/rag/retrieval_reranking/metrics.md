# Retrieval and reranking metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Scoring inputs and result status](#scoring-inputs-and-result-status)
- [Retrieval quality](#retrieval-quality-1)
- [Ranking consistency](#ranking-consistency-1)
- [Why the quality metrics are not interchangeable](#why-the-quality-metrics-are-not-interchangeable)
- [Operational performance](#operational-performance-1)
- [Workload descriptors](#workload-descriptors-1)
- [Storage descriptors](#storage-descriptors)
- [Worked candidate interpretation](#worked-candidate-interpretation)
- [Aggregation and confidence intervals](#aggregation-and-confidence-intervals)

This benchmark compares complete retriever/reranker stacks on one frozen
chunker/embedding pair. Dense, BM25, and RRF each run with no reranker or one
of four learned rerankers: 15 declared stacks, with development restricted to
the hardware-qualified subset. The selected chunker, embedding, source corpus,
questions, and evidence annotations remain the same for every stack.

The three primary quality families measure ordering, completeness, and
concentration of evidence. Diagnostics explain pool limitations and repeated
coverage. Operational and storage measurements explain their cost. No weighted
score combines the families, and no context-token budget changes the evaluated
ranked chunks. Every metric below has a question, calculation, worked example,
interpretation, range/direction, and availability rules. The
[methodology](methodology.md) defines pool ownership, advancement, MLflow
artifacts, and paired comparisons.

These definitions are the approved reporting contract, not a claim that every
current output already satisfies it. The
[implementation boundary](methodology.md#implementation-boundary) identifies
pending runner/protocol alignment. Worked examples are illustrative, not
measured results.

## Metric summary

### Retrieval quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| nDCG@3/@5 | Primary | Are chunks containing more required evidence placed near the top? | Higher |
| Evidence-unit Recall@3/@5 | Primary | How much distinct required evidence is present in the selected context? | Higher |
| Evidence-token Precision@3/@5 | Primary | How concentrated is that context around verified evidence? | Higher |
| alpha-nDCG@3/@5 | Diagnostic | How early does new evidence appear when repeated coverage earns less credit? | Higher |
| Candidate-pool Evidence-unit Recall@20 | Diagnostic | How much evidence was available before reranking? | Higher |

Both @3 and @5 are reporting cutoffs for the three primary families.
They use prefixes of the same first measured ranking. The top-20 pool is the
reranker's input and a diagnostic depth, not another final context cutoff.

### Ranking consistency

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Ranking Repeatability | Diagnostic | How often do repeated requests return the same complete valid ordering? | Higher |

### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Full-Stack Warm Latency p50/p95 | Operational | What are typical and slow-tail complete query times? | Lower |
| Retrieval Warm Latency p50/p95 | Operational | How much time belongs to the first-stage retriever? | Lower |
| Reranker Warm Latency p50/p95 | Operational | How much time belongs to learned reranking? | Lower |
| Cold Initialization Time | Operational | How much fresh query-runtime initialization time is required, excluding offline index build? | Lower |
| Peak Process-Tree RAM | Operational | What peak host memory does the worker and its children occupy? | Lower at equal quality |
| Peak Device VRAM | Operational | What peak total memory is occupied on the assigned GPU? | Lower at equal quality |
| Index Build Time | Operational, separate Dense/BM25 owners | How long does preparing each searchable index take? | Lower |

Each latency family has separate p50 and p95 scalars on the complete-stack
child. The retriever's identity comes from that child; Dense, BM25, and RRF do
not need three differently named retrieval-latency metrics. Units are explicit
in the [MLflow keys](methodology.md#mlflow-result-structure).

### Workload descriptors

| Value | Role | Question answered | Direction |
|---|---|---|---|
| Mean Retrieved Context Tokens@3/@5 | Workload descriptor | How much source text does the final ranked context contain on average? | Descriptive |
| p95 Retrieved Context Tokens@3/@5 | Workload descriptor | How large is the long tail of final context sizes? | Descriptive |
| Mean/p95 Reranker Input Tokens | Workload descriptor | How much native-tokenizer input is prepared for one reranked query? | Descriptive |

Corpus/query/chunk counts belong to the parent input inventory, and requested
pool depth is a protocol parameter. Actual pool sizes and optional
Candidate-Pool Tokens@20 belong to pool artifacts. They explain and verify the
workload without becoming mandatory leaderboard metrics.

### Storage descriptors

| Value | Role | Question answered | Direction |
|---|---|---|---|
| Index Size | Storage descriptor, separate Dense/BM25 owners | How much searchable state does each independent index require? | Descriptive |
| Index Size, with Dense/BM25 breakdown | Storage descriptor, RRF owner | What total storage does RRF require, and how much belongs to each dependency? | Descriptive |

Descriptors explain operational differences. A smaller value is not
automatically better and cannot compensate for worse quality.

## Scoring inputs and result status

### Evidence and cutoffs

Only answerable questions with reviewed gold evidence are executed and scored.
Unanswerable questions in a shared manifest are inventory for later stages,
not empty-reference retrieval comparisons. Missing required evidence or invalid
source intervals fail [data validation](../../data-validation.md#chunking-and-embedding-inputs)
before inference. The required canonical corpus and answerable-question
allocation must be nonempty; missing required inputs fail preparation rather
than supplying an empty retrieval comparison.

A chunk covers a gold unit only when that single chunk contains all required
half-open source intervals in the same canonical document. Token precision
instead counts source-token overlap: a fragment can contribute relevant tokens
without recovering a complete unit.

The four @3/@5 families use the
[shared retrieval-evidence definitions](../../metrics.md#retrieval-evidence-quality).
Chunking/embedding candidates can create different corpora; here every stack
shares the same complete frozen corpus and per-question normalization baselines.

### Attempts, failures, and integrity

Quality uses the designated first measured ranking after warmup, with later
attempts supplying timing and Ranking Repeatability. Apply the
[shared availability/status rules](../../metrics.md#shared-availability-rules)
to failed outputs, missing records, and task-free slices.

Each reranker must receive the correct checksummed query pool and return exactly
its chunk IDs once each, possibly in a different order. It cannot add, remove,
duplicate, or replace chunks. Final top-K must be the ordered prefix of that
validated complete ranking, not an arbitrary subset. Nonfinite scores, silent
truncation, invalid IDs, and unexpectedly empty/short rankings are integrity failures, not valid
zero-quality outputs. Distinct IDs containing identical text remain legitimate;
matching text does not make their source identities interchangeable.
A ranking with no valid returned source tokens is likewise an output-integrity
failure: token precision is null/unavailable, not a scored empty prediction.
An upstream index required to be nonempty but producing zero chunks is invalid;
retain the observed zero without calling it successful indexing.

The expected pool length is min(20, frozen chunk count). A legitimately smaller
corpus uses every available chunk without padding or renaming @3/@5. The smoke
fixture has 30 chunks and therefore requires 20 results. Returning 12 when 100
chunks exist is an integrity failure; returning all 12 when only 12 exist is not.

Apply the [shared completeness distinction](../../metrics.md#execution-accounting-and-result-completeness):
request-failure counts describe observed outcomes, not fabricated irrelevant
chunks or an implicit zero-error acceptance rule. Missing required execution,
partial quality, and failed integrity each retain their own consequences.

Integrity checks verify pool provenance, exact live retrieval order against the
frozen reference, output permutation/prefix, finite scores, and absence of
truncation. They remain enforced checks with detailed outcomes in
`validation_report.json`, not a separate `pool_collection_match` leaderboard
indicator. Missing verification is not a pass. A violation stops the affected
attempt's downstream work; shared invalid inputs also stop dependent work.
Preserve independent completed measurements and safe independent execution,
but do not accept the affected evaluation as complete authoritative evidence.

See [what belongs in a metric's edge-case block](../../metrics.md#what-belongs-in-a-metrics-edge-case-block)
for the separation between valid metric exceptions, rejected inputs, and
runtime integrity failures.

Build time and final index size require a verified complete index; incomplete
preparation retains progress separately. Partial resource scope is stored in
`resources.parquet`/`candidate.json` under the
[shared resource rules](../../metrics.md#resource-measurements).

## Retrieval quality

### nDCG@3/@5

**Question:** How early does the stack place chunks containing more required
evidence?

Apply the [shared linear-gain nDCG rule](../../metrics.md#retrieval-evidence-quality)
to the frozen chunk corpus. Its exact ideal is identical for every compared
stack on that question, regardless of which top-20 pool was retrieved.

In plain language:

    discounted evidence gain of the returned ranking
    ------------------------------------------------
     discounted evidence gain of the ideal ranking

**Example:** The only evidence-bearing chunks have gains 3 and 1. At @3,
ideal gains [3, 1, 0] give discounted gain about 3.631. Order [1, 3, 0]
gives about 2.893, so nDCG is approximately 0.80. Both orders can recover the
same units; one places evidence-rich content earlier.
If ranks 4 and 5 add zero gain and the ideal also has no further positive gain,
the same example gives approximately 0.80 at @5.

This is primary because the benchmark compares relevance ordering. Repeated
relevant chunks are not automatically ranking errors. A valid ranking with no
complete evidence scores zero when the ideal gain is positive. A substantive
nDCG above one is a scoring-integrity error, not a value to clamp.

**Range and direction:** [0, 1]; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Zero ideal gain in a valid frozen corpus | Gold evidence exists, but every chunk fragments it and contains no complete unit. | `0` (`scored`), with the shared upstream-fragmentation limitation recorded. | Apply the [shared zero-ideal convention](../../metrics.md#retrieval-evidence-quality). |

[Shared availability rules](../../metrics.md#shared-availability-rules) apply.

### Evidence-unit Recall@3/@5

**Question:** How much distinct required evidence appears anywhere inside the
cutoff?

Count a unit once when at least one selected chunk completely contains it.
Position within the cutoff and repeated copies add no credit. Eligible
questions always have reviewed gold units, so the denominator is positive.

In plain language:

    distinct required units recovered in the selected chunks
    --------------------------------------------------------
               required units for the question

**Example:** One recovered unit out of three gives Recall@3 about 0.33.
If ranks 4 and 5 add a second, Recall@5 is about 0.67.

This is primary because a convincing early ranking can still omit evidence
needed for a complete answer. No recovered units gives zero normally.

**Range and direction:** [0, 1]; higher is better.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[retrieval eligibility and integrity](#scoring-inputs-and-result-status).

### Evidence-token Precision@3/@5

**Question:** What share of returned source-token occurrences is verified
evidence?

Apply the [shared evidence-token counting rule](../../metrics.md#retrieval-evidence-quality)
to canonical text in the final @3/@5 prefixes. This counts returned source-token
occurrences, not native reranker inputs or distinct evidence units.

In plain language:

    returned token occurrences overlapping verified evidence
    ---------------------------------------------------------
                 all returned source-token occurrences

**Example:** Top-three chunks contain 120 relevant occurrences among 300:
Precision@3 is 120 / 300 = 0.40. Ranks 4 and 5 add another 120 relevant
occurrences among 300, so Precision@5 is 240 / 600 = 0.40.
The remaining 60% is additional context not labelled as
evidence, not necessarily false text.

This is primary because high nDCG and recall do not establish concentrated
context. Valid returned text with no relevant tokens gives zero; an invalid
returned-token denominator follows the page-wide
[output-integrity rules](#attempts-failures-and-integrity).

**Range and direction:** [0, 1]; higher is better.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[retrieval eligibility and integrity](#scoring-inputs-and-result-status).

### alpha-nDCG@3/@5

**Question:** How early does new evidence appear when repeated coverage receives
less credit?

Start from complete-unit coverage. With alpha = 0.5, halve a unit's contribution
for each earlier selected chunk already covering it, then apply the logarithmic
rank discount.

In plain language:

    discounted novelty gain of the returned ranking
    ------------------------------------------------
    discounted novelty gain of the greedy baseline

The [shared greedy baseline](../../metrics.md#retrieval-evidence-quality) uses
the complete frozen corpus and is the same for every stack on the question.

**Example:** Chunks cover E1, E1, and E2. At @3, order E1, E1, E2 gives gain
about 1.815; greedy order E1, E2, E1 gives about 1.881. Their ratio is
approximately 0.965. Ordinary nDCG can be 1 for both because all three chunks
have relevance gain one.
If no other chunk covers a required unit, adding zero-gain ranks 4 and 5 leaves
the same alpha-nDCG at @5. Extra ranks can otherwise change both numerator
and baseline; @3 and @5 are calculated separately.

This diagnostic adds a novelty preference. It does not establish that repeated
evidence is irrelevant or harms answers. Single-unit questions remain eligible.
The development-only exact-normalization review is in
[pending-data-review](../../pending-data-review.md).

**Range and direction:** [0, 1]; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Zero greedy-baseline gain in a valid corpus | No corpus chunk contains any complete required unit. | `0` (`scored`), under the zero-baseline convention. | Follow the shared upstream-fragmentation convention rather than dividing by zero. |
| A valid ranking exceeds the greedy baseline | The uncapped novelty-gain ratio is `1.02`. | Report `1` (`scored`); retain `1.02` and the greedy-baseline-exceeded flag. | The deterministic greedy denominator is not a guaranteed maximum; exceeding it is not itself evaluation failure. |

These conventions match [chunking/embedding](../chunking_embedding/metrics.md#retrieval-quality-1).
[Shared availability rules](../../metrics.md#shared-availability-rules) apply
to failed rankings or evaluations.

### Candidate-pool Evidence-unit Recall@20

**Question:** Did first-stage retrieval make the required evidence available to
its rerankers?

Apply Evidence-unit Recall to the verified pool before learned reranking.

In plain language:

    distinct required units found anywhere in the first-stage pool
    ---------------------------------------------------------------
                  required units for the question

**Example:** The question requires E1, E2, and E3; the pool contains E1
and E2. Pool recall is 2 / 3, about 0.67. No permutation can recover E3.
High pool recall but low final Recall@5 indicates evidence was available before
the final selection.

Record this once on dense|none, bm25|none, or rrf|none. Learned-reranker
children reference that pool collection instead of copying its diagnostic.
When no owner child is evaluated, retain the prepared pool observation without
inventing a baseline run.

Eligible questions have a positive gold-unit denominator. A valid pool with
no recovered evidence scores zero. A legitimate corpus smaller than 20 uses
its complete available pool; record requested depth and actual size separately.

**Range and direction:** [0, 1]; higher is better as a diagnostic ceiling.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[pool verification and result status](#attempts-failures-and-integrity).

## Ranking consistency

### Ranking Repeatability

**Question:** How often do identical requests reproduce the complete valid
chunk ordering?

Apply [pairwise repeatability](../../metrics.md#pairwise-repeatability) to the
complete ordered result, normally 20 IDs or all available IDs for a legitimately
smaller corpus. All IDs must have the same positions; floating-point scores
need not be bit-identical.

In plain language:

    scheduled attempt pairs with identical valid complete rankings
    --------------------------------------------------------------
                     scheduled attempt pairs

**Example:** A, A, A gives 3 / 3 = 1. A, A, B gives 1 / 3; so does
failure, A, A, because (2,3) still agrees. A with two failures gives zero,
and three executed failures give zero. Identical error messages are not
successful rankings and never agree.

Calculate the pair rate for each question, then average within source documents
and macro-average documents. Retain scheduled/matching pair counts and complete
attempt outcomes so a high score cannot hide failed execution.

This diagnostic measures output-order repeatability, not repeatability of every
quality score or latency. It does not require a perfect score for selection.
Declared hardware and integrity requirements remain enforced. Recorded request
failures retain their own outcomes rather than automatically creating a new
zero-error eligibility rule.
For learned rerankers, the enforced live-pool check already establishes retrieval
order consistency among passing executions; no extra retrieval-repeatability
scalar is required. For `none`, the complete output is the retrieval ordering.
Unlike chunking/embedding's top-five diagnostic, this examines the reranker's full
input/output ordering and can expose changes hidden below rank five.

**Range and direction:** [0, 1]; higher is more repeatable, not necessarily more
relevant.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| One-attempt smoke | One measured ranking, hence no scheduled pair. | `null` (`inapplicable`). | A single attempt cannot measure pairwise ranking repeatability. |

Availability follows the shared
[pairwise-repeatability contract](../../metrics.md#pairwise-repeatability).

## Why the quality metrics are not interchangeable

| Metric | What changes it | What it does not answer directly |
|---|---|---|
| nDCG@3/@5 | Ranks and complete evidence counts | Whether every distinct unit was found or how much extra text was returned |
| Evidence-unit Recall@3/@5 | Distinct units recovered inside the cutoff | Whether units appeared early or were surrounded by extra text |
| Evidence-token Precision@3/@5 | Share of returned tokens overlapping evidence | Whether complete units were recovered or ordered early |
| alpha-nDCG@3/@5 | Evidence position and earlier repetition count | Whether repetition harms the final answer |
| Pool Recall@20 | Evidence available before reranking | Whether the final top-three/top-five order is good |

Reordering the same selected chunks can change nDCG without changing recall
or token precision. Missing a required unit lowers recall; extra non-evidence
context lowers token precision. Pool recall diagnoses the first-stage ceiling;
alpha-nDCG diagnoses novelty under a different objective.

## Operational performance

Warm means initialization and warmup have finished. It does not mean replaying
a saved ranking. Each measured request executes its operations, including
required preparation, input/output checks, ordering, and device synchronization.
Downloads, full data validation/input verification, cold initialization,
warmup, and offline quality/CI calculations are outside warm query timing.

For every latency family, take each query's median successful completed
component/request duration, then calculate percentiles across query observations
using the [shared percentile calculation](../../metrics.md#latency-percentiles).
Keep support per component: retrieval can complete before its reranker fails.
Interval applicability is listed under
[aggregation and confidence intervals](#aggregation-and-confidence-intervals);
p99 belongs to the separate vector-server load experiment.

### Full-Stack Warm Latency p50

**Question:** What is typical complete query time after warmup?

The complete path includes live retrieval, learned-reranker tokenization,
inference, ordering when applicable, and validated result selection.

In plain language:

    p50 of per-query median successful full-stack durations

**Example:** Query observations 10, 20, 30, 40, and 50 ms give p50 30 ms.
Successful attempts 20 and 24 ms give one query observation 22 ms even if its
third attempt fails.

This is deployable warm-request cost, not the cost of replaying a quality pool.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[warm-latency observation rules](#operational-performance-1).

### Full-Stack Warm Latency p95

**Question:** What is slow-tail complete warm query time?

Use the same query observations as p50 and calculate their 95th percentile.

**Example:** Observations 10, 20, 30, 40, and 50 ms give descriptive p95
at position 3.8: 40 + 0.8 × (50 - 40) = 48 ms, not the maximum or an average.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[warm-latency observation rules](#operational-performance-1).

### Retrieval Warm Latency p50

**Question:** What is typical retrieval time before learned reranking?

Measure the selected first stage inside the live request. Dense includes query
preparation/embedding and exact search. BM25 includes lexical query preparation
and scoring. RRF includes both source retrievals and fusion. Do not read cached
quality pools as a substitute for these operations.

In plain language:

    p50 of per-query median successful retrieval durations

**Example:** Retrieval observations 4, 6, and 8 ms give p50 6 ms.
Retrieval completed in 8 ms remains known if its downstream reranker fails.
Component support does not establish full-stack success.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[warm-latency observation rules](#operational-performance-1).

### Retrieval Warm Latency p95

**Question:** How slow is the retrieval tail?

Calculate p95 from the same per-query retrieval observations used for its p50.

**Example:** Observations 4, 6, and 8 ms give position 1.9:
6 + 0.9 × (8 - 6) = 7.8 ms. This summarizes the slow tail of retrieval,
not the difference between full-stack and reranker percentiles.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[warm-latency observation rules](#operational-performance-1).

### Reranker Warm Latency p50

**Question:** What typical additional time does learned reranking require?

Time complete native preparation, scoring, ordering, validation, and required
synchronization for the learned reranker's pool.

In plain language:

    p50 of per-query median successful reranking durations

**Example:** Observations 20, 30, and 40 ms give p50 30 ms.
This is the learned component's complete warm cost, not just its model-forward
pass or full-stack latency minus retrieval's median.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No learned reranker | `dense\|none`. | `null` (`inapplicable`); omit the scalar. | The stack has no learned-reranking task; its cost is not a measured zero. |

[Shared availability rules](../../metrics.md#shared-availability-rules) and
[warm-latency observation rules](#operational-performance-1) apply when a
reranker exists but completed timings are unavailable.

### Reranker Warm Latency p95

**Question:** How slow is the learned reranking tail?

Calculate p95 over the same query observations as its p50.

**Example:** Observations 20, 30, and 40 ms give position 1.9:
30 + 0.9 × (40 - 30) = 39 ms. This describes the reranker's slow tail,
not a bound on the complete stack's tail.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No learned reranker | `bm25\|none`. | `null` (`inapplicable`); omit the scalar. | There is no learned-reranking latency distribution to summarize. |

[Shared availability rules](../../metrics.md#shared-availability-rules) and
[warm-latency observation rules](#operational-performance-1) apply when a
reranker exists but completed timings are unavailable.

### Cold Initialization Time

**Question:** How much fresh query-runtime initialization time is required,
apart from the one-time offline index build?

Include required model/tokenizer/reranker loading, device placement, and
query-runtime initialization before warmup. Exclude the separately timed
offline index-build interval and separately identified input-preflight interval.
The current counter derives this value from those disjoint recorded intervals;
it is narrower than wall time from worker start through all corpus preparation.
Operating-system disk caches are not forcibly cleared. Downloads, environment
installation, and offline quality scoring remain outside initialization.

In plain language:

    initialization elapsed time
    - separately timed offline index build
    - separately timed input preflight

**Example:** Model loading takes 3 seconds, offline index build 12 seconds,
input preflight 0.2 seconds, and remaining reranker/runtime loading 2 seconds.
Initialization elapsed is 17.2 seconds; Cold Initialization Time is
17.2 - 12 - 0.2 = 5 seconds. Index Build Time separately reports 12 seconds.
Do not describe the five-second scalar as the full 17.2-second first readiness
wait, or count offline build again as part of cold loading.

BM25 without a learned model still has runtime initialization; model absence
does not imply zero initialization time.
Negative or nonfinite derived time is an unavailable instrumentation error,
not a value to clamp to zero.

**Range and direction:** Non-negative seconds; lower is better.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[completed-measurement status rules](#attempts-failures-and-integrity).

### Peak Process-Tree RAM

**Question:** What peak host memory does the complete stack occupy?

Use [Peak Process-Tree RAM](../../metrics.md#peak-process-tree-ram). This stack's
window includes loading, index preparation, warmup, and measured queries,
excluding offline scoring/bootstrap calculations.

**Example:** Totals 1,200, 2,800, and 2,100 MiB give peak 2,800 MiB.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[resource-window status rules](#attempts-failures-and-integrity).

### Peak Device VRAM

**Question:** What peak total GPU memory is occupied during candidate execution?

Use [Peak Device VRAM](../../metrics.md#peak-device-vram) over the same candidate
lifecycle as this stack's RAM measurement.

**Example:** Samples 700, 2,100, and 1,900 MiB give peak 2,100 MiB.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| CPU-only execution | CPU smoke without an assigned GPU. | `null` (`inapplicable`). | No GPU-memory task exists; this differs from a missing CUDA measurement. |

[Shared availability rules](../../metrics.md#shared-availability-rules) govern
missing required CUDA telemetry (`null/unavailable`) and
[partial resource windows](#attempts-failures-and-integrity).

### Index Build Time — Dense

**Question:** How long does preparation of the Dense searchable state take?

After required model readiness, time complete Dense vectors/metadata/searchable
state preparation through validation and required synchronization. Distinguish
building state from loading already-built state into a query worker.

In plain language:

    validated Dense state readiness - Dense build start

**Example:** A build from second 2 to second 14 reports 12 seconds once on
the Dense owner, normally dense|none. Reranked children reference the same
observation. If no baseline child is evaluated, the preparation input artifact
owns the observation; no baseline run is invented.

**Range and direction:** Non-negative seconds; lower is better.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[completed-build status rules](#attempts-failures-and-integrity).

### Index Build Time — BM25

**Question:** How long does preparation of the lexical searchable state take?

Time lexical corpus preparation, BM25 construction, and validation separately
from Dense preparation.

In plain language:

    validated BM25 state readiness - BM25 build start

**Example:** BM25 preparation taking 0.8 seconds reports 0.8 on bm25|none,
not the Dense owner's 12 seconds. Reranked BM25 children reference this owner.

RRF depends on both owner indexes and builds no third searchable index. Fusion
belongs to Retrieval Warm Latency, not an invented third index build.
Both independent owners use the same `operational.index_build_seconds` key.
RRF's `index_build.json` references their separate observations; it does not
copy component build scalars or invent a combined third build time.

**Range and direction:** Non-negative seconds; lower is better.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[completed-build status rules](#attempts-failures-and-integrity).

## Workload descriptors

Selected-context means and percentiles adopt the
[shared retrieved-context workload contract](../../metrics.md#retrieved-context-workload).
The examples below retain this suite's selected-stack interpretation.

Verified input counts remain known after failure. Actual generated or measured
quantities keep their own availability and complete/partial scope.

Incomplete pool collections retain observed diagnostics and planned/observed
query support, not a complete collection result. Keep missing/unexecuted
queries separate from actually emitted empty pools and from observed
short-pool counts. An emitted empty pool has observed size zero but is invalid
output; preserve its row and duplicate-ID errors rather than confusing it with
a missing pool.

Selected-context counts require valid first rankings and successful token
counting. A failed count is unavailable; a partial set is a labelled workload
diagnostic. Complete counts survive later offline quality-evaluator failure.
For native reranker-input counts, only a fully tokenized pool supplies a
complete query total. Prepared totals survive later inference failure without
implying that all pairs completed inference. Retain known native lengths when
rejecting oversized inputs; never truncate to make a complete pool fit. Keep
prepared-input counts and completed-pair outcomes distinguishable in artifacts.

### Inventory and pool diagnostics

The following counts are stored in parent input inventory/parameters and pool
artifacts, not required scalar metrics. Pool-size summaries are optional
integrity diagnostics. Required eligibility/failure support remains in result
artifacts and validity counters; inventory does not replace it.

#### Corpus Document Count

**Question:** How many canonical documents are searchable?

Count distinct documents in the verified frozen corpus, including distractors,
not returned chunks.

**Example:** A corpus of 100 papers reports 100 even if execution later fails
before its first query.

**Range and direction:** Non-negative integer; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[verified-inventory status rules](#workload-descriptors-1).

#### Answerable Query Count

**Question:** How many reviewed retrieval questions are scheduled?

Count eligible answerable questions once each, not once per attempt.
Unanswerable shared-manifest questions are not timed or scored here.

**Example:** 120 questions with three attempts mean 120 queries and 360
scheduled attempts, not necessarily 360 successful requests.
Preserve processed/successful/failed/unexecuted counts separately; an absent
required allocation is rejected under [input prerequisites](#evidence-and-cutoffs).

**Range and direction:** Non-negative integer; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[verified-inventory status rules](#workload-descriptors-1).

#### Chunk Count

**Question:** How many canonical passages are searchable?

Count unique chunk records in the frozen corpus, not rows returned across
queries. Distinct IDs can legitimately contain identical text.

**Example:** 800 chunks remain 800 when 100 queries each produce 20 results.

**Range and direction:** Non-negative integer; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[index-integrity prerequisites](#attempts-failures-and-integrity).

#### Requested Pool Size

**Question:** How many first-stage candidates are requested per question?

Record the protocol depth, currently 20, separately from actual production.

**Example:** Requested 20 remains 20 if a broken retriever returns 12;
retain the observed short pool and failure separately.

**Range and direction:** Positive integer; descriptive execution control.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[verified-inventory status rules](#workload-descriptors-1).

#### Mean Actual Pool Size

**Question:** How many chunks does a produced pool contain on average?

Count unique IDs per observed pool and average the query counts. Do not count
the same frozen pool repeatedly for rerankers or repetitions.

In plain language:

    total observed pool members / observed query pools

**Example:** Sizes 20, 20, and 12 give 52 / 3, approximately 17.33.
With at least 20 searchable chunks, the last pool violates the output contract.
If the entire corpus has 12 chunks, every valid pool instead has size 12.

**Range and direction:** Non-negative chunks/query; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[pool-diagnostic status rules](#workload-descriptors-1).

#### Minimum Actual Pool Size

**Question:** What is the smallest produced query pool?

Take the minimum of the same per-query counts.

**Example:** Sizes 20, 20, and 12 give minimum 12, exposing a short pool
that an average alone can obscure.

**Range and direction:** Non-negative integer; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[pool-diagnostic status rules](#workload-descriptors-1).

#### Short-Pool Query Count

**Question:** How many produced pools are shorter than requested?

Count observed pools below the requested depth. Missing pools are not invented
zero-sized observations; retain missing/unexecuted queries separately.

**Example:** Sizes 20, 20, and 12 with depth 20 give short-pool count one.
With 100 corpus chunks that is an integrity failure, not weak relevance.
With only 12 corpus chunks, pools of 12 are short relative to the requested
depth but complete relative to the available corpus; retain that distinction.
A complete collection without short pools gives zero.

**Range and direction:** Non-negative integer; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[pool-diagnostic status rules](#workload-descriptors-1).

### Mean Retrieved Context Tokens@3/@5

**Question:** How much canonical source text would the selected context pass
downstream on average?

Apply the shared context-token calculation to the valid designated first ranking
after any enabled reranking, then average per-query totals. This descriptor is
query-weighted, not document-macro quality or average chunk size.

In plain language:

    sum of selected-context token totals / contributing queries

**Example:** Three queries have @3 totals 400, 600, and 800: mean
(400 + 600 + 800) / 3 = 600 tokens/query. Their @5 totals are 600, 900,
and 1,200: mean (600 + 900 + 1,200) / 3 = 900 tokens/query.
Each cutoff describes its own selected context. This imposes no token budget
and is not inherently better when smaller.
The mean describes average returned volume, while p50 would describe the middle
query; p95 separately describes larger contexts.

**Range and direction:** Non-negative tokens/query; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[token-count status rules](#workload-descriptors-1).

### p95 Retrieved Context Tokens@3/@5

**Question:** How large is the long tail of selected context sizes?

Calculate the interpolated p95 of the same per-query totals.
Use the [shared percentile calculation](../../metrics.md#latency-percentiles).

**Example:** The @3 totals 400, 600, and 800 give position 1.9 and
600 + 0.9 × (800 - 600) = 780 tokens. The @5 totals 600, 900, and 1,200
give 900 + 0.9 × (1,200 - 900) = 1,170 tokens. These are context-size
statistics rather than latency or correctness.

**Range and direction:** Non-negative tokens/query; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[token-count status rules](#workload-descriptors-1).

### Candidate-Pool Tokens@20

**Question:** How much canonical source text enters the complete reranking pool?

Store an optional per-query field in the pool artifact. Sum
`tiktoken:cl100k_base` lengths of the verified pool's canonical chunks, including
repeated text occurrences across distinct chunks. For RRF, count the fused pool,
not both source lists. This is not an additional mandatory MLflow metric.

**Example:** A verified 20-chunk pool totals 6,000 source tokens. Reranking its
exact permutation still totals 6,000; selecting the final five might total
1,500. The first count describes candidate preparation, the second downstream
context. Native reranker templates and the repeated query belong to Reranker
Input Tokens, not this source-text count.

A valid smaller corpus counts its actual pool. Aggregate from saved fields when
needed rather than duplicating it as a separate Mean Retrieved Tokens@20 metric.
Only a complete verified pool supplies a complete count; partial progress
remains an artifact diagnostic.

**Range and direction:** Non-negative tokens/query; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[pool-verification rules](#attempts-failures-and-integrity).

### Mean Reranker Input Tokens

**Question:** How much native-tokenizer input is prepared for one learned
reranking query?

Tokenize each query-passage pair with the reranker's own tokenizer and required
template/special tokens. Sum pair lengths within a question before averaging
question totals.

In plain language:

    sum of complete per-query native-token totals / contributing queries

**Example:** Query totals 2,000, 2,400, and 4,000 give mean
8,400 / 3 = 2,800 tokens/query. Averaging pair lengths answers a different
question and hides total request workload.

This is not the fixed source tokenizer used by Evidence-token Precision.
The same passage can have different native lengths for different rerankers.

**Range and direction:** Non-negative tokens/query; descriptive.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No learned reranker | `rrf\|none`. | `null` (`inapplicable`); omit the scalar. | No native reranker template or tokenizer workload is prepared. |

[Shared availability rules](../../metrics.md#shared-availability-rules) and
[prepared-input status rules](#workload-descriptors-1) apply when a reranker
exists but complete token totals are unavailable.

### p95 Reranker Input Tokens

**Question:** How large is the native-input workload for expensive queries?

Calculate p95 over complete per-query sums, not all pairs pooled globally.
Use the [shared percentile calculation](../../metrics.md#latency-percentiles).

**Example:** Totals 2,000, 2,400, and 4,000 give position 1.9 and
2,400 + 0.9 × (4,000 - 2,400) = 3,840 tokens/query. This describes the
tail of prepared per-query workload, not the largest individual pair length.

**Range and direction:** Non-negative tokens/query; descriptive.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No learned reranker | `dense\|none`. | `null` (`inapplicable`); omit the scalar. | No native reranker-input distribution exists to summarize. |

[Shared availability rules](../../metrics.md#shared-availability-rules) and
[prepared-input status rules](#workload-descriptors-1) apply when a reranker
exists but complete token totals are unavailable.

## Storage descriptors

Dense and BM25 each record the same Index Size field on their own owner.
Measure the complete declared searchable-state representation, excluding model
weights and unrelated artifacts. State whether bytes describe raw arrays or
serialized storage; headers and metadata can legitimately make files larger
than the vector payload alone.

### Index Size — Dense

**Question:** How much stored searchable state does semantic retrieval require?

Add the bytes of unique components included in the declared Dense state.

**Example:** A representation with 3,072,000 vector-payload bytes and 128,000
metadata bytes totals 3,200,000 bytes. A serialized representation that also
stores a header can legitimately be larger; record its actual measurement
boundary rather than enforcing the raw-payload formula on the whole file.

This is index storage, not model VRAM or the reranker's checkpoint size.
For raw matrices, bytes must agree with shape and dtype. For serialized state,
validate the declared format and included components as well: a `.npy` header
is not a payload-integrity error. A contradictory shape, dtype, or storage
manifest is an integrity failure, not a valid unusually small index.
Rerankers reference the owner rather than claiming separately built copies.

**Range and direction:** Non-negative bytes; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[verified-index status rules](#attempts-failures-and-integrity).

### Index Size — BM25

**Question:** How much stored searchable state does lexical retrieval require?

Add unique lexical-index components in the declared representation, not Dense
vectors or learned model weights.

**Example:** Lexical state measured as 400,000 bytes of one required component
and 100,000 bytes of another totals 500,000 bytes. Report 500,000 on the BM25
owner; do not copy the Dense owner's 3,200,000 bytes or add it to pure BM25.

This separates the disk/search-state cost of lexical and semantic retrieval.
RRF references this state alongside Dense rather than creating a third copy.

**Range and direction:** Non-negative bytes; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[verified-index status rules](#attempts-failures-and-integrity).

### Index Size — RRF and component sizes

RRF's storage values describe dependencies, not a third index. RRF creates no
additional persistent search index, so no constant-zero incremental-index
metric is reported. Query-time buffers belong to execution memory; saved pools
are artifacts rather than search indexes.

#### Index Size — RRF

**Question:** How much index storage must exist for RRF to operate?

Add unique required Dense and BM25 state sizes, not repeated references.

In plain language:

    required Dense index bytes + required BM25 index bytes

**Example:** Dense requires 12,000,000 bytes and BM25 requires 3,000,000:
RRF reports Index Size 15,000,000, Dense Index Size 12,000,000, and BM25
Index Size 3,000,000 bytes. The breakdown references the same verified source
observations rather than independently remeasuring or counting them twice.

Both verified component-size measurements are required for the complete total.
If either is missing, the total is unavailable, not the remaining index's size;
preserve the known component size separately.

**Range and direction:** Non-negative bytes; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[verified-index status rules](#attempts-failures-and-integrity).

#### Dense Index Size — RRF

**Question:** How much of RRF's required index storage belongs to Dense?

Reference the verified Dense owner's complete searchable-state byte count,
using the same declared representation and measurement boundary. Do not count
it again as an independently built RRF index.

**Example:** Dense state of 12,000,000 bytes gives Dense Index Size
12,000,000 bytes on RRF, even when its BM25 dependency adds 3,000,000 bytes
to the total. This identifies the semantic dependency's share, not RRF's
incremental storage.

**Range and direction:** Non-negative bytes; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[verified-index status rules](#attempts-failures-and-integrity).

#### BM25 Index Size — RRF

**Question:** How much of RRF's required index storage belongs to BM25?

Reference the verified BM25 owner's complete searchable-state byte count,
using its declared lexical representation. Do not substitute Dense bytes or
remeasure a supposed third index.

**Example:** BM25 state of 3,000,000 bytes gives BM25 Index Size
3,000,000 bytes on RRF, separately from the 12,000,000-byte Dense dependency.
This identifies lexical storage without counting the same state twice.

**Range and direction:** Non-negative bytes; descriptive.

**Edge cases and availability:**

Shared availability rules apply. See the
[shared rules](../../metrics.md#shared-availability-rules) and
[verified-index status rules](#attempts-failures-and-integrity).

Model snapshot size is preparation metadata (`snapshot_size_bytes`) in the model
lock, not a per-candidate benchmark metric. See the
[installation guide](../../../setup/installation.md#snapshot-size-metadata).

## Worked candidate interpretation

Suppose a valid stack reports:

    nDCG@5                         = 0.84
    Evidence-unit Recall@5         = 0.72
    Evidence-token Precision@5     = 0.40
    alpha-nDCG@5                   = 0.78
    referenced pool Recall@20      = 0.90
    Full-Stack Warm Latency p50    = 45 ms
    Retrieval Warm Latency p50     = 8 ms
    Reranker Warm Latency p50      = 35 ms

Evidence-rich chunks are ordered well on average, but the selected context
does not recover every required unit. Forty percent is document-macro average
token precision, not a pooled percentage of all corpus tokens. Higher pool
recall shows that some omitted evidence was available before top-five selection.
Alpha-nDCG describes novelty, not a subtraction from ordinary nDCG. Component
medians need not sum to the full-stack median: they summarize query observations
separately. Review the same families at @3 before freezing downstream top-K.

## Aggregation and confidence intervals

### Aggregation and completeness

Average defined first-ranking question scores within each source document,
then macro-average document values. Apply the same procedure to pool recall
and each evidence slice. All four @3/@5 families share reference eligibility
but retain metric-specific scored/unavailable support.

**Example:** A document with ten questions averaging 0.8 and another with two
averaging 0.4 give document-macro (0.8 + 0.4) / 2 = 0.6, not the
question-weighted (10 × 0.8 + 2 × 0.4) / 12, approximately 0.73.

A slice with no eligible questions is inapplicable; eligible questions without
usable scores make its aggregate unavailable. Partial required quality remains an artifact
diagnostic, not a normal complete-result leaderboard value or paired selection
comparison. For example, 0.90 on eight of ten questions cannot establish
superiority over 0.85 on all ten. Complete first-ranking quality can survive a
later failed attempt. Execution coverage, quality coverage, and integrity follow
the shared completeness distinction, not a blanket zero-request-failure gate.

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---|---|
| Development/validation/locked @3/@5 quality and pool recall | When defined and sufficiently supported | Resample independent source documents and recalculate the macro statistic. |
| Text, table, formula, and mixed quality slices | When defined and sufficiently supported | Preserve eligible source question/failure records. |
| Full-stack, retrieval, and reranker warm p50/p95 | Conditional | Resample source documents with query-level successful-attempt medians; recalculate percentiles. |
| Ranking Repeatability | When complete and sufficiently supported | Resample source documents with all scheduled query pairs and recalculate question-then-document means. |
| Validity counters | No selection-uncertainty interval | Retain actual checks, support, and failure details. |
| Smoke | No authoritative interval | Wiring evidence only. |
| One cold initialization, index build, or observed memory peak | No | One lifecycle observation cannot estimate uncertainty. |
| Workload/storage descriptors | No | Observed properties, not invented sampling uncertainty. |

### Calculation

Apply the [shared CI procedure](../../metrics.md#confidence-intervals) to the
reference-eligible source-document inventory and its complete query/failure
records. Recalculate document-macro quality and pool recall; latency draws use
the sampled documents' query-level successful-attempt medians before calculating
each component's percentile. Repeatability draws recalculate each question's
scheduled pair score and document-macro mean; they do not create new rankings or
treat pairs as independent documents. The shared support, undefined-draw,
paired-draw, and degenerate-bound rules apply. Diagnostic status does not waive
source-support requirements; null bounds do not erase a defined descriptive point.

**Example:** nDCG@5 0.84 with CI [0.79, 0.88] describes the observed macro
score and source-sampling uncertainty, not individual-question score ranges.

Paired comparisons use the same source draws for both complete valid candidates.
Their schemas and CSV/Parquet artifacts belong in the
[methodology](methodology.md#paired-comparisons-and-advancement).
