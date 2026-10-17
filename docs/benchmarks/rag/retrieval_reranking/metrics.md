# Retrieval and reranking metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Scoring inputs and result status](#scoring-inputs-and-result-status)
- [Retrieval quality](#retrieval-quality-1)
- [Ranking consistency](#ranking-consistency)
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
| Ranking Agreement | Validity gate | Do later measured attempts reproduce the first complete ordering? | Must equal 1 for authoritative validity |

### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Full-Stack Warm Latency p50/p95 | Operational | What are typical and slow-tail complete query times? | Lower |
| Retrieval Warm Latency p50/p95 | Operational | How much time belongs to the first-stage retriever? | Lower |
| Reranker Warm Latency p50/p95 | Operational | How much time belongs to learned reranking? | Lower |
| Cold Initialization Time | Operational | How much fresh query-runtime initialization time is required, excluding offline index build? | Lower |
| Peak Process-Tree RAM | Operational | What peak host memory does the worker and its children occupy? | Lower at equal quality |
| Peak Device VRAM | Operational | What peak total memory is occupied on the assigned GPU? | Lower at equal quality |
| Dense Index-Build Time | Operational, Dense owner | How long does preparing the searchable Dense state take? | Lower |
| BM25 Index-Build Time | Operational, BM25 owner | How long does preparing the searchable lexical state take? | Lower |

Retrieval latency is the same measurement called first-stage latency in
artifact keys. This reader-facing name does not rename the existing
operational.first_stage_latency_ms_p50/p95 keys.

### Workload descriptors

| Value | Role | Question answered | Direction |
|---|---|---|---|
| Corpus Document Count | Workload descriptor | How many canonical documents are searchable? | Descriptive |
| Answerable Query Count | Workload descriptor | How many reviewed questions are scheduled? | Descriptive |
| Chunk Count | Workload descriptor | How many canonical passages are searchable? | Descriptive |
| Requested Pool Size | Workload descriptor | How many candidates does each retriever request? | Descriptive |
| Mean/Minimum Actual Pool Size | Workload descriptor | How large are the produced query pools? | Descriptive |
| Short-Pool Query Count | Workload descriptor | How many pools contain fewer than requested? | Descriptive |
| Mean/p95 Retrieved Tokens@3/@5 | Workload descriptor | How much source text would the ranked context pass downstream? | Descriptive |
| Mean/p95 Reranker Input Tokens | Workload descriptor | How much native-tokenizer input is prepared for one reranked query? | Descriptive |

### Storage descriptors

| Value | Role | Question answered | Direction |
|---|---|---|---|
| Dense/BM25 Index Bytes | Storage descriptor, index owners | How much searchable state does each independent index require? | Descriptive |
| RRF Required Index Bytes | Storage descriptor, RRF owner | What total index storage does RRF depend on? | Descriptive |
| Reranker Snapshot Bytes | Storage descriptor | How much local model storage does the learned reranker require? | Descriptive |

Descriptors explain operational differences. A smaller value is not
automatically better and cannot compensate for worse quality.

## Scoring inputs and result status

### Evidence and cutoffs

Only answerable questions with reviewed gold evidence are executed and scored.
Unanswerable questions in a shared manifest are inventory for later stages,
not empty-reference retrieval comparisons. Missing required evidence or invalid
source intervals fail data validation before inference.

A chunk covers a gold unit only when that single chunk contains all required
half-open source intervals in the same canonical document. Token precision
instead counts source-token overlap: a fragment can contribute relevant tokens
without recovering a complete unit.

The four @3/@5 families use the
[chunking/embedding definitions](../chunking_embedding/metrics.md#retrieval-quality-1).
Chunking/embedding candidates can create different corpora; here every stack
shares the same complete frozen corpus and per-question normalization baselines.

### Attempts, failures, and integrity

Quality uses the designated first measured ranking after warmup. Later attempts
supply timing and ranking-consistency evidence, not extra quality scores.
Later success never replaces a failed first ranking; later failure does not
erase valid first-ranking quality.

| Result | Value and treatment |
|---|---|
| Valid first ranking with no recovered evidence | Normal poor numeric quality scores. |
| Failed first request or invalid first ranking | Applicable quality is null/unavailable; retain the error and planned support. |
| Valid ranking but one metric cannot be evaluated | That metric is unavailable; preserve independently valid scores and the ranking for rescoring. |
| Optional evidence slice has no eligible questions | Null/inapplicable aggregate. |
| Eligible questions exist but none is scorable | Null/unavailable aggregate. |
| Required execution or records are missing | Incomplete execution, not invented observations. |

Each reranker must receive the correct checksummed query pool and return exactly
its chunk IDs once each, possibly in a different order. It cannot add, remove,
duplicate, or replace chunks. Nonfinite scores, silent truncation, invalid IDs,
and unexpectedly empty/short rankings are integrity failures, not valid
zero-quality outputs. Distinct IDs containing identical text remain legitimate;
matching text does not make their source identities interchangeable.

The expected pool length is min(20, frozen chunk count). A legitimately smaller
corpus uses every available chunk without padding or renaming @3/@5. The smoke
fixture has 30 chunks and therefore requires 20 results. Returning 12 when 100
chunks exist is an integrity failure; returning all 12 when only 12 exist is not.

| Status | Meaning |
|---|---|
| scored | Defined numeric result under the metric's contract. |
| inapplicable | No task exists, such as learned-reranker latency without a reranker. |
| unavailable | The task exists but an output, evaluation, or measurement is unavailable. |
| incomplete | Required execution or records are absent. |
| invalid_reference | Inputs violate the contract; repair before candidate execution. |

Report scheduled, eligible, scored, failed, unavailable, and independent-source
support. Nulls/reasons remain in JSON/Parquet; MLflow omits null scalar keys.
A failed child may retain valid measurements but is not a complete selection
result. Failure counts are validity evidence, not fabricated irrelevant chunks.

## Retrieval quality

### nDCG@3/@5

**Question:** How early does the stack place chunks containing more required
evidence?

A chunk receives one point per distinct gold unit it completely contains. Use
that count directly, not an exponential gain. Divide each gain by the base-two
logarithm of its rank plus one, then sum inside the cutoff.

In plain language:

    discounted evidence gain of the returned ranking
    ------------------------------------------------
     discounted evidence gain of the ideal ranking

The exact ideal sorts gains over every chunk in the complete frozen corpus,
using the same cutoff/discount. It is never derived from a retrieved pool.
Repeated evidence retains its relevance credit in each distinct chunk.

**Example:** The only evidence-bearing chunks have gains 3 and 1. At @3,
ideal gains [3, 1, 0] give discounted gain about 3.631. Order [1, 3, 0]
gives about 2.893, so nDCG is approximately 0.80. Both orders can recover the
same units; one places evidence-rich content earlier.
If ranks 4 and 5 add zero gain and the ideal also has no further positive gain,
the same example gives approximately 0.80 at @5.

This is primary because the benchmark compares relevance ordering. Repeated
relevant chunks are not automatically ranking errors.

**Range and direction:** [0, 1]; higher is better. Valid rankings containing
no complete evidence get zero. If no valid corpus chunk contains any complete
required unit, follow the chunking/embedding zero-ideal convention: zero with
the shared upstream-fragmentation limitation recorded. Corrupt annotations and
evaluator errors are not zero-ideal cases. Failed first rankings are unavailable.
A substantive nDCG above one is a scoring-integrity error, not a value to clamp.

### Evidence-unit Recall@3/@5

**Question:** How much distinct required evidence appears anywhere inside the
cutoff?

Count a unit once when at least one selected chunk completely contains it.
Position within the cutoff and repeated copies add no credit.

In plain language:

    distinct required units recovered in the selected chunks
    --------------------------------------------------------
               required units for the question

**Example:** One recovered unit out of three gives Recall@3 about 0.33.
If ranks 4 and 5 add a second, Recall@5 is about 0.67.

This is primary because a convincing early ranking can still omit evidence
needed for a complete answer.

**Range and direction:** [0, 1]; higher is better. No recovered units gives
zero normally. Eligible questions always have gold units; missing annotations
fail data validation. Failed first rankings are unavailable.

### Evidence-token Precision@3/@5

**Question:** What share of returned source-token occurrences is verified
evidence?

Measure canonical chunk text using tiktoken:cl100k_base. A token occurrence is
relevant when its source span has positive overlap with a gold evidence interval
in the same document. Count it once despite overlapping gold intervals. Text
returned through different chunks counts again in numerator and denominator.
Native model prefixes and special inference tokens are not source content.

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
context.

**Range and direction:** [0, 1]; higher is better. Valid returned text with
no relevant tokens gives zero. No valid returned tokens is an output-integrity
failure and gives null/unavailable, not a scored empty prediction. Failed first
rankings are unavailable.

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

The baseline uses the complete frozen corpus. At each rank choose the unused
chunk with the largest remaining novelty gain; ties follow frozen corpus order.
This deterministic greedy baseline is not guaranteed to be the exact maximum.

**Example:** Chunks cover E1, E1, and E2. At @3, order E1, E1, E2 gives gain
about 1.815; greedy order E1, E2, E1 gives about 1.881. Their ratio is
approximately 0.965. Ordinary nDCG can be 1 for both because all three chunks
have relevance gain one.
If no other chunk covers a required unit, adding zero-gain ranks 4 and 5 leaves
the same alpha-nDCG at @5. Extra ranks can otherwise change both numerator
and baseline; @3 and @5 are calculated separately.

This diagnostic adds a novelty preference. It does not establish that repeated
evidence is irrelevant or harms answers. Single-unit questions remain eligible.

**Range and direction:** [0, 1]; higher is better. No complete unit anywhere
in the valid corpus gives zero under the zero-baseline convention. Otherwise cap
the reported ratio at one and retain the uncapped ratio and greedy-baseline-
exceeded flag when capping occurs, as in chunking/embedding. Exceeding a greedy
baseline is not itself evaluation failure. Failed first rankings or evaluator
errors are unavailable. The development-only exact-normalization review is in
[pending-data-review](../../pending-data-review.md).

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

**Range and direction:** [0, 1]; higher is better as a diagnostic ceiling.
Valid pools with no recovered evidence get zero. Missing, invalid, or unverified
pools are unavailable. Downstream reranker failure does not erase a valid
pool-owner measurement. A legitimate corpus smaller than 20 uses its complete
available pool, with the requested depth and actual size recorded separately.

## Ranking consistency

### Ranking Agreement

**Question:** Do later identical requests reproduce the first complete measured
chunk ordering?

The first ranking is the anchor. Compare each later attempt's ordered top-20
chunk IDs with it. A comparison agrees only when both attempts are valid and
all IDs have the same positions. Floating-point scores need not be bit-identical.

In plain language:

    later measured rankings matching the valid first ranking
    --------------------------------------------------------
                 expected later-ranking comparisons

**Example:** With three attempts A, A, B, comparisons (1,2) and (1,3) give
1 / 2. A, A, failure also gives 1 / 2. Failure, A, A gives zero because
no valid anchor exists. Identical error messages never agree.

This is the current validity gate, not an extra quality score. It differs
from extraction's all-pairs repeatability and chunking/embedding's top-five
consistency check; these definitions are not interchangeable.

**Range and direction:** [0, 1]; authoritative validity requires one.
All recorded comparisons failing gives zero. One-attempt smoke is
null/inapplicable. Missing/unexecuted required outcomes are incomplete, not
invented disagreements or a favorable surviving-only rate.

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
with linear interpolation. Failed elapsed times remain artifacts, not completed
latency. Keep support per component: retrieval can complete before its reranker
fails.

For n sorted query observations, the zero-based percentile position is
(n - 1) × p, where p is 0.50 or 0.95. Interpolate adjacent observations when
the position is fractional; do not round the position to a measured request.

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
quality. No completed timed queries gives null/unavailable. One observation
gives that observation as p50. Missing/nonfinite timers are unavailable.

### Full-Stack Warm Latency p95

**Question:** What is slow-tail complete warm query time?

Use the same query observations as p50 and calculate their 95th percentile.

**Example:** Observations 10, 20, 30, 40, and 50 ms give descriptive p95
at position 3.8: 40 + 0.8 × (50 - 40) = 48 ms, not the maximum or an average.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality. No observations is unavailable; one gives that observation. Small
samples retain descriptive points without unsupported CI bounds. p95 is not a
95% confidence bound. p99 belongs to the vector-server load experiment.

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

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality. No completed retrieval timing is unavailable. Component support does
not establish full-stack success.

### Retrieval Warm Latency p95

**Question:** How slow is the retrieval tail?

Calculate p95 from the same per-query retrieval observations used for its p50.

**Example:** Observations 4, 6, and 8 ms give position 1.9:
6 + 0.9 × (8 - 6) = 7.8 ms.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality. The same unavailable/support rules apply as full-stack latency.
Do not derive it by subtracting reranker p95 from full-stack p95.

### Reranker Warm Latency p50

**Question:** What typical additional time does learned reranking require?

Time complete native preparation, scoring, ordering, validation, and required
synchronization for the learned reranker's pool.

In plain language:

    p50 of per-query median successful reranking durations

**Example:** Observations 20, 30, and 40 ms give p50 30 ms.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality. No-reranker stacks are null/inapplicable and omit the scalar.
Failed/never-started reranking has unavailable completed latency, not zero.
No completed observations makes the aggregate unavailable.

### Reranker Warm Latency p95

**Question:** How slow is the learned reranking tail?

Calculate p95 over the same query observations as its p50.

**Example:** Observations 20, 30, and 40 ms give position 1.9:
30 + 0.9 × (40 - 30) = 39 ms.

**Range and direction:** Non-negative milliseconds; lower is better at equal
quality. No reranker is inapplicable; no completed observations is unavailable.
Keep component support and failed elapsed times. Component percentiles need not
sum to the full-stack percentile.

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
17.2 - 12 - 0.2 = 5 seconds. Index-Build Time separately reports 12 seconds.
Do not describe the five-second scalar as the full 17.2-second first readiness
wait, or count offline build again as part of cold loading.

**Range and direction:** Non-negative seconds; lower is better. Failure
before readiness is unavailable, with elapsed-to-setup-failure retained.
Successful readiness survives later inference failure. One observation has no CI.

### Peak Process-Tree RAM

**Question:** What peak host memory does the complete stack occupy?

At each sample sum resident memory of the worker and living child processes.
Take the largest simultaneous total, not the sum of independent process peaks.
Monitor loading, index preparation, warmup, and measured queries, excluding
offline scoring/bootstrap calculations.

**Example:** Totals 1,200, 2,800, and 2,100 MiB give peak 2,800 MiB.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
No samples is unavailable. Preserve observed peaks after failure with partial
lifecycle scope in resources.parquet/candidate.json; they do not establish
that the remaining workload would fit.

### Peak Device VRAM

**Question:** What peak total GPU memory is occupied during candidate execution?

Take the largest raw NVML memory.used sample on the assigned GPU. Record
method, GPU identity, idle baseline, and samples; do not subtract the baseline
or infer model placement from this value.

**Example:** Samples 700, 2,100, and 1,900 MiB give peak 2,100 MiB.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
CPU execution is null/inapplicable; missing required CUDA telemetry is
null/unavailable, never zero. Preserve peaks after OOM/hardware-gate failure
with partial scope, without clamping to the protocol cap. One peak has no CI.

### Dense Index-Build Time

**Question:** How long does preparation of the Dense searchable state take?

After required model readiness, time complete Dense vectors/metadata/searchable
state preparation through validation and required synchronization. Distinguish
building state from loading already-built state into a query worker.

In plain language:

    validated Dense state readiness - Dense build start

**Example:** A build from second 2 to second 14 reports 12 seconds once on
dense|none. Reranked Dense children reference that owner.

**Range and direction:** Non-negative seconds; lower is better. Failed builds
have unavailable completed time, with elapsed-to-failure retained. Completed
builds survive later failures. One build has no CI.

### BM25 Index-Build Time

**Question:** How long does preparation of the lexical searchable state take?

Time lexical corpus preparation, BM25 construction, and validation separately
from Dense preparation.

In plain language:

    validated BM25 state readiness - BM25 build start

**Example:** BM25 preparation taking 0.8 seconds reports 0.8 on bm25|none,
not the Dense owner's 12 seconds. Reranked BM25 children reference this owner.

RRF depends on both owner indexes and builds no third searchable index. Fusion
belongs to Retrieval Warm Latency, not an invented third index build.

**Range and direction:** Non-negative seconds; lower is better. Failed builds
are unavailable; completed builds survive later failure. One build has no CI.

## Workload descriptors

Verified input counts remain known after failure. Actual generated or measured
quantities keep their own availability and complete/partial scope.

### Corpus Document Count

**Question:** How many canonical documents are searchable?

Count distinct documents in the verified frozen corpus, including distractors,
not returned chunks.

**Example:** A corpus of 100 papers reports 100 even if execution later fails
before its first query.

**Range and direction:** Non-negative integer; descriptive. Missing required
inputs fail preparation rather than supplying a valid empty corpus.

### Answerable Query Count

**Question:** How many reviewed retrieval questions are scheduled?

Count eligible answerable questions once each, not once per attempt.
Unanswerable shared-manifest questions are not timed or scored here.

**Example:** 120 questions with three attempts mean 120 queries and 360
scheduled attempts, not necessarily 360 successful requests.

**Range and direction:** Non-negative integer; descriptive. Preserve
processed/successful/failed/unexecuted counts separately. Missing required
answerable allocation is invalid input, not perfect empty-reference quality.

### Chunk Count

**Question:** How many canonical passages are searchable?

Count unique chunk records in the frozen corpus, not rows returned across
queries. Distinct IDs can legitimately contain identical text.

**Example:** 800 chunks remain 800 when 100 queries each produce 20 results.

**Range and direction:** Non-negative integer; descriptive. A required
nonempty index producing zero chunks is invalid; retain observed zero without
calling it successful indexing.

### Requested Pool Size

**Question:** How many first-stage candidates are requested per question?

Record the protocol depth, currently 20, separately from actual production.

**Example:** Requested 20 remains 20 if a broken retriever returns 12;
retain the observed short pool and failure separately.

**Range and direction:** Positive integer; descriptive execution control.

### Mean Actual Pool Size

**Question:** How many chunks does a produced pool contain on average?

Count unique IDs per observed pool and average the query counts. Do not count
the same frozen pool repeatedly for rerankers or repetitions.

In plain language:

    total observed pool members / observed query pools

**Example:** Sizes 20, 20, and 12 give 52 / 3, approximately 17.33.
With at least 20 searchable chunks, the last pool violates the output contract.
If the entire corpus has 12 chunks, every valid pool instead has size 12.

**Range and direction:** Non-negative chunks/query; descriptive. No produced
pool is unavailable, not zero. Incomplete collections retain a partial
diagnostic and planned/observed query support, not a complete collection result.

### Minimum Actual Pool Size

**Question:** What is the smallest produced query pool?

Take the minimum of the same per-query counts.

**Example:** Sizes 20, 20, and 12 give minimum 12, exposing a short pool
that an average alone can obscure.

**Range and direction:** Non-negative integer; descriptive. No produced pools
is unavailable. Incomplete collections retain partial scope.

### Short-Pool Query Count

**Question:** How many produced pools are shorter than requested?

Count observed pools below the requested depth. Missing pools are not invented
zero-sized observations; retain missing/unexecuted queries separately.

**Example:** Sizes 20, 20, and 12 with depth 20 give short-pool count one.
With 100 corpus chunks that is an integrity failure, not weak relevance.
With only 12 corpus chunks, pools of 12 are short relative to the requested
depth but complete relative to the available corpus; retain that distinction.

**Range and direction:** Non-negative integer; descriptive. A complete
collection without short pools gives zero. Partial collections retain their
observed count and missing-query accounting rather than a complete total.

### Mean Retrieved Tokens@3/@5

**Question:** How much canonical source text would the selected context pass
downstream on average?

For each valid first ranking, sum tiktoken:cl100k_base occurrences in its first
three/five chunks. Average per-query totals. Text repeated across chunks counts
again; model prefixes do not count.

In plain language:

    sum of selected-context token totals / contributing queries

**Example:** Three queries have @3 totals 400, 600, and 800: mean
(400 + 600 + 800) / 3 = 600 tokens/query. Their @5 totals are 600, 900,
and 1,200: mean (600 + 900 + 1,200) / 3 = 900 tokens/query.
Each cutoff describes its own selected context. This imposes no token budget
and is not inherently better when smaller.

**Range and direction:** Non-negative tokens/query; descriptive. Failed first
rankings have unavailable counts. No contributing ranking is unavailable;
a partial set is a labelled workload diagnostic.

### p95 Retrieved Tokens@3/@5

**Question:** How large is the long tail of selected context sizes?

Calculate the interpolated p95 of the same per-query totals.

**Example:** The @3 totals 400, 600, and 800 give position 1.9 and
600 + 0.9 × (800 - 600) = 780 tokens. The @5 totals 600, 900, and 1,200
give 900 + 0.9 × (1,200 - 900) = 1,170 tokens. These are context-size
statistics rather than latency or correctness.

**Range and direction:** Non-negative tokens/query; descriptive. No complete
observations is unavailable; one gives that observation. Fixed-workload
descriptors receive no invented CI.

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

**Range and direction:** Non-negative tokens/query; descriptive. No learned
reranker is inapplicable. Partially tokenized pools have no complete total.
Fully prepared totals can survive later inference failure, but do not imply
that all pairs completed inference.

### p95 Reranker Input Tokens

**Question:** How large is the native-input workload for expensive queries?

Calculate p95 over complete per-query sums, not all pairs pooled globally.

**Example:** Totals 2,000, 2,400, and 4,000 give position 1.9 and
2,400 + 0.9 × (4,000 - 2,400) = 3,840 tokens/query.

**Range and direction:** Non-negative tokens/query; descriptive. No reranker
is inapplicable; no complete totals is unavailable. Prepared-input counts and
completed-pair outcomes remain distinguishable in artifacts.

## Storage descriptors

Dense and BM25 each record the same Index Bytes field on their own owner.
Measure the complete declared searchable-state representation, excluding model
weights and unrelated artifacts. State whether bytes describe raw arrays or
serialized storage; headers and metadata can legitimately make files larger
than the vector payload alone.

### Dense Index Bytes

**Question:** How much stored searchable state does semantic retrieval require?

Add the bytes of unique components included in the declared Dense state.

**Example:** A representation with 3,072,000 vector-payload bytes and 128,000
metadata bytes totals 3,200,000 bytes. A serialized representation that also
stores a header can legitimately be larger; record its actual measurement
boundary rather than enforcing the raw-payload formula on the whole file.

This is index storage, not model VRAM or the reranker's checkpoint size.

**Range and direction:** Non-negative bytes; descriptive. Incomplete/unverified
state has unavailable final size, with progress retained in artifacts. A
verified index's size survives later query failure. Rerankers reference the
owner rather than claiming separately built copies.

### BM25 Index Bytes

**Question:** How much stored searchable state does lexical retrieval require?

Add unique lexical-index components in the declared representation, not Dense
vectors or learned model weights.

**Example:** Lexical state measured as 400,000 bytes of one required component
and 100,000 bytes of another totals 500,000 bytes. Report 500,000 on the BM25
owner; do not copy the Dense owner's 3,200,000 bytes or add it to pure BM25.

This separates the disk/search-state cost of lexical and semantic retrieval.
**Range and direction:** Non-negative bytes; descriptive. Unfinished/unverified
state is unavailable as a completed index; verified size survives later failure.
RRF references this state alongside Dense rather than creating a third copy.

### RRF Required Index Bytes

**Question:** How much index storage must exist for RRF to operate?

Add unique required Dense and BM25 state sizes, not repeated references.

In plain language:

    required Dense index bytes + required BM25 index bytes

**Example:** Dense requires 12 MiB and BM25 requires 3 MiB: RRF depends on
15 MiB, not zero.

**Range and direction:** Non-negative bytes; descriptive. A missing required
dependency makes the complete total unavailable, not the remaining index's size.
This is dependency storage, not a third index.
RRF creates no additional persistent search index, so no constant-zero
incremental-index metric is reported. Query-time buffers belong to execution
memory; saved pools are artifacts rather than search indexes.

### Reranker Snapshot Bytes

**Question:** How much local model storage does the learned reranker require?

Measure the prepared verified snapshot from the model lock, excluding indexes
and run artifacts. Do not multiply by query count.

**Example:** A 300 MiB snapshot remains 300 MiB across 100 queries or separate
Dense/BM25 candidates referencing the same snapshot.

**Range and direction:** Non-negative bytes; descriptive. No learned reranker
is inapplicable. Incomplete/unverified snapshots are unavailable as complete
model storage. Verified size survives inference failure.

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
later failed attempt while the candidate still fails run-level validity.

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---|---|
| Development/validation/locked @3/@5 quality and pool recall | When defined and sufficiently supported | Resample independent source documents and recalculate the macro statistic. |
| Text, table, formula, and mixed quality slices | When defined and sufficiently supported | Preserve eligible source question/failure records. |
| Full-stack, retrieval, and reranker warm p50/p95 | Conditional | Resample source documents with query-level successful-attempt medians; recalculate percentiles. |
| Ranking Agreement and validity counters | No selection-uncertainty interval | Retain expected comparisons and actual gate outcomes. |
| Smoke | No authoritative interval | Wiring evidence only. |
| One cold initialization, index build, or observed memory peak | No | One lifecycle observation cannot estimate uncertainty. |
| Workload/storage descriptors | No | Observed properties, not invented sampling uncertainty. |

### Calculation

Development, validation, and locked use 10,000 source-document bootstrap draws
with replacement, seed 42. Recalculate each metric's own statistic; the 2.5th
and 97.5th percentiles give 95% bounds. Draw from the reference-eligible source
inventory with failure/unavailable records, not only successful questions.
If a document is drawn twice, its observations contribute twice. Repetitions
are not independent documents.

Record defined/undefined draws per metric. Exclude an undefined draw only from
that metric's bounds, not saved records or other defined metrics. Below frozen
independent support, retain defined points with null bounds and reasons.
Supported equal bounds such as [0, 0] remain with degeneracy flags, not claims
of certainty about future documents.

**Example:** nDCG@5 0.84 with CI [0.79, 0.88] describes the observed macro
score and source-sampling uncertainty, not individual-question score ranges.
Ten thousand draws perform saved-record arithmetic, not new inference.

Paired comparisons use the same source draws for both complete valid candidates.
Their schemas and CSV/Parquet artifacts belong in the
[methodology](methodology.md#paired-comparisons-and-advancement).
