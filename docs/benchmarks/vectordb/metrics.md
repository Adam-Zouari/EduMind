# Vector database metrics

[Shared metric conventions](../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Scoring inputs, eligibility, and result status](#scoring-inputs-eligibility-and-result-status)
- [Search quality](#search-quality-1)
- [Application retrieval quality](#application-retrieval-quality-1)
- [Diagnostics](#diagnostics-1)
- [Reliability and filter correctness](#reliability-and-filter-correctness-1)
- [Why the quality metrics are not interchangeable](#why-the-quality-metrics-are-not-interchangeable)
- [Operational performance](#operational-performance-1)
- [Storage descriptors](#storage-descriptors-1)
- [Worked result interpretation](#worked-result-interpretation)
- [Aggregation and confidence intervals](#aggregation-and-confidence-intervals)

This benchmark evaluates database–configuration pairs. ANN quality measures
preservation of exact nearest neighbours; evidence quality checks the frozen
retrieval stack's educational usefulness. Latency, reliability, maintenance and
resource measurements answer separate questions rather than forming one weighted
score. Primary status does not select a winner automatically.

Each metric follows the document-extraction reference: question, calculation,
example, interpretation, range/direction, and **Edge cases and availability**.
The [shared availability rules](../metrics.md#shared-availability-rules) own
failures, missing records and partial observations. Local blocks add only genuine
metric-specific exceptions, not bad-data cases or repeated general rules.
All numbers below illustrate arithmetic, not measured database results.

The [implementation boundary](methodology.md#implementation-status) applies to
this entire reporting contract. Old keys/measurements must not be renamed to imply
the revised calculations already execute.

## Metric summary

### Search quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| ANN Recall@20 | Primary | How many exact nearest neighbours does unfiltered search preserve? | Higher |
| Filtered ANN Recall@20 | Primary | How many exact nearest neighbours inside the allowed subset are preserved? | Higher |

### Application retrieval quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| nDCG@3/@5 | Secondary | Does the database-backed stack rank evidence-rich chunks early? | Higher |
| Evidence-unit Recall@3/@5 | Secondary | How much required evidence reaches the selected context? | Higher |
| Evidence-token Precision@3/@5 | Secondary | How much selected context consists of verified evidence? | Higher |

### Diagnostics

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Candidate-pool Evidence-unit Recall@20 | Diagnostic | How much required evidence is available before reranking? | Higher |
| Ranking Repeatability | Diagnostic | Does the database reproduce the same ordered retrieved IDs? | Higher stability, not necessarily better relevance |

### Reliability and filter correctness

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Request Error Rate — Search | Diagnostic | How often do submitted searches fail to execute successfully? | Lower |
| Request Error Rate — Insert | Diagnostic | How often do submitted document insertions fail to complete? | Lower |
| Request Error Rate — Update | Diagnostic | How often do submitted document replacements fail to complete? | Lower |
| Request Error Rate — Delete | Diagnostic | How often do submitted document deletions fail to complete? | Lower |
| Filter Violation Rate | Diagnostic | How often does a completed checkable filtered response contain forbidden records? | Lower |

### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Database Search Latency p50 | Operational | What is a typical completed database request's duration? | Lower |
| Database Search Latency p95 | Operational | How slow is the completed database-request tail? | Lower |
| Full-Stack Retrieval Latency p50 | Operational | What is a typical complete retrieval/reranking wait? | Lower |
| Full-Stack Retrieval Latency p95 | Operational | How slow is the complete retrieval/reranking tail across questions? | Lower |
| QPS | Operational | How many successful search requests finish per wall-clock second? | Higher |
| Initial Build Time | Operational | How long until prepared data and configured indexes are query-ready? | Lower |
| Insert Latency p50/p95 | Operational | How long until prepared documents become available? | Lower |
| Update Latency p50/p95 | Operational | How long until document replacements are fully reflected? | Lower |
| Delete Latency p50/p95 | Operational | How long until documents are confirmed absent? | Lower |
| Startup Time | Operational | How long until an existing stopped database can serve its preserved data? | Lower |
| Peak Server RAM | Operational | What is the highest observed server memory use in the declared lifecycle? | Lower at comparable quality/workload |

### Storage descriptors

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Persistent Storage Size | Descriptor | How much persistent state does the completed database require? | No quality direction; smaller is a footprint advantage at comparable functionality |

Workload counts, dimensions, concurrency, filters, requested/actual result counts,
and document operation sizes remain parameters or artifact fields. Actual
execution-support counters use the shared `validity.*` reporting convention; they
are not extra selection metrics. See [MLflow structure](methodology.md#results-and-mlflow-structure).

## Scoring inputs, eligibility, and result status

Quality uses the designated first measured response for each query/condition.
Later attempts supply timing and diagnostic database-ranking comparisons, not
replacement quality. Reference eligibility is fixed before execution. Apply
[shared completeness accounting](../metrics.md#execution-accounting-and-result-completeness)
to failed, unavailable and not-executed outcomes. Partial required quality is
retained as labelled diagnostic evidence, not a normal complete-quality aggregate.

Invalid vectors, missing evidence, incorrect filter references, and split leakage
belong to [data validation](../data-validation.md). Duplicate/unknown output IDs,
nonfinite required scores and unverified provenance belong to runtime integrity.
Do not bury these cases in metric-specific empty-denominator conventions.

Each filter is evaluated against trusted source metadata. Search execution
success means a completed response with a usable envelope, known unique IDs and
finite required scores. Filter correctness is checked separately. Use the same
execution-success cohort for Search Request Error Rate, database latency and QPS:

| Search outcome | Execution error? | Include in successful QPS/database latency? | Filter observation |
|---|---|---|---|
| Usable response, all records allowed | No | Yes | `0` when filtering applies. |
| Usable response, known forbidden record returned | No | Yes, with correctness failure retained | `1`; dependent ANN/integration quality is unavailable. |
| Timeout, transport failure, unusable envelope, duplicate/unknown IDs, or nonfinite required scores | Yes | No; retain elapsed-to-failure/invalid response separately | Unavailable if filtering was required; do not infer a violation or pass. |

Thus successful execution is not a claim of correct filtering or accepted
benchmark evidence. A filter-violating child's retained performance cannot be
presented as a conformant comparison. An execution failure supplies no checkable
filter response. Low recall and legitimate alternative tied neighbours are not
request errors. Missing outcome/verification records remain incomplete rather
than presumed success or failure.

Well-formed short or empty ANN responses against nonempty references retain
their low recall. Integration has a stricter declared pool contract:
`min(20, eligible corpus size)` for each required source retrieval. Unexpectedly
short/empty pools leave dependent evidence metrics unavailable, not zero. Do not
silently rerun, pad, or refill them. A genuinely smaller eligible corpus uses
all available chunks. Its valid final ranking must still contain source tokens.

Application metrics use answerable questions and the
[shared retrieval-evidence definitions](../metrics.md#retrieval-evidence-quality).
Unanswerable questions are inapplicable, not empty gold comparisons. Complete-unit
coverage uses source intervals; token precision uses `tiktoken:cl100k_base` source
token occurrences. The nDCG ideal uses the entire eligible frozen corpus, never
only the retrieved pool. A reranker must permute its actual verified pool; the
exact-retrieval suite's repeated-live-pool equality gate does not apply here.

### Exact neighbours and boundary ties

For each query, calculate exhaustive distances from the frozen vectors with the
common evaluator. Filter before selecting neighbours. Let the effective result
count be the smaller of 20 and the eligible record count. If it is positive,
find that position's exact distance and split the eligible records into:

1. records strictly closer than the cutoff;
2. records tied at the cutoff; and
3. records farther away.

The number of **boundary slots** is the effective result count minus the number
of strictly closer records. Credit each recovered strictly closer ID once.
Credit recovered boundary IDs only up to the number of boundary slots. Extra
boundary alternatives cannot compensate for a missing strictly closer record.
This calculates the best ID overlap with a valid exact top-result set without
enumerating all possible tied sets. Do not increase K or ask the database to
return every tied record.

Use canonical evaluator distances, not backend-reported rounded similarities.
Default to exact equality. Any justified nonzero numerical tolerance must be
fixed before comparison and recorded with evaluator precision/version. With
absolute tolerance `t`, boundary records satisfy `abs(distance - cutoff) <= t`
and strictly closer records satisfy `distance < cutoff - t`. If a relative
component is adopted, its rule for deriving `t` must also be frozen. Tolerance
is numerical policy, not a permitted relevance loss tuned to improve recall.

**Worked tie example:** At @3 for illustration, distances are A=0.10,
B=C=D=0.20, E=0.30. A is strictly closer and two boundary slots remain:

| Returned IDs | Credit | Recall |
|---|---|---:|
| A,B,C | A and two boundary records | 1 |
| A,C,D | An equally valid exact result | 1 |
| A,B,E | A and one boundary record | 2/3 |
| B,C,D | Two boundary slots; A is missing | 2/3 |

The benchmark uses @20; the small example only explains the arithmetic. Record
cutoff distance, effective count, tolerance, strict/boundary counts, slots and
credited counts with each scored query. Returned IDs and reference checksums
allow independent verification. Repeatability still compares actual ordered IDs;
two different valid tied answers need not agree.

## Search quality

### ANN Recall@20

**Question:** How many of the exact nearest neighbours survive unfiltered search?

Apply the boundary-tie rule to the first 20 returned IDs, with no ordering credit
inside that prefix. In plain language:

```text
recovered strictly closer records + credited boundary records
------------------------------------------------------------
                 effective requested count
```

**Example:** With 20 eligible nearest-neighbour slots and no boundary ambiguity,
18 recovered neighbours give `18 / 20 = 0.90`. This measures vector-search
accuracy, not whether those neighbours answer the educational question.

Average query scores within the declared condition; do not pool recovered counts
when effective denominators differ. Compare speed at comparable recall.

**Range and direction:** `[0,1]`; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Fewer eligible records than 20 | Only 8 records exist; 6 are recovered. | `6/8 = 0.75` (`scored`). | Score the available task, not 20 imaginary records. |
| Empty valid response against a nonempty reference | No IDs returned for 20 reference slots. | `0` (`scored`). | No neighbours were recovered; this is not a runtime crash. |
| More records tie at the cutoff than there are slots | 18 strict neighbours and 5 tied alternatives. | Credit at most 2 boundary records. | Equally close alternatives are interchangeable only in those slots. |

An empty required unfiltered corpus fails input validation. Shared availability
rules govern failed requests; they do not receive this valid-empty zero.

### Filtered ANN Recall@20

**Question:** How many exact neighbours inside the allowed subset survive search?

Use the same calculation, after restricting the exact reference and the request
to the same frozen predicate. The denominator is `min(20, eligible records)`,
not the unfiltered corpus size. Report conditions separately.

**Example:** A document filter permits 100 records. Recovering 17 of its exact
nearest 20 gives `17/20 = 0.85`. A different filter permitting only 8 records,
of which 6 are returned, gives `6/8 = 0.75`.

A missed eligible neighbour is not necessarily a filter violation: another
returned record may satisfy the filter but be farther away.

**Range and direction:** `[0,1]`; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Filter legitimately permits no records | An empty-match control returns no IDs. | `null` (`inapplicable`). | There are no nearest neighbours to recover; filter behaviour is checked separately. |
| Filter permits fewer than 20 records | All 3 permitted records are returned. | `1` (`scored`). | All available neighbours were recovered. |
| Valid empty result with eligible neighbours | Filter permits 30 records but returns none. | `0` (`scored`). | The nonempty reference supplies a positive denominator. |
| Boundary ties in the eligible subset | Several permitted vectors tie for the final slot. | Apply the same boundary-slot credit. | Filtering does not change the tie policy. |

Shared availability and output-integrity rules apply; forbidden records are not
silently discarded to turn an invalid response into an accepted recall score.

## Application retrieval quality

### nDCG@3/@5

**Question:** Does the database-backed stack place evidence-rich chunks early?

Each chunk's gain is the number of distinct complete required evidence units it
contains, used linearly. Apply the shared rank discount and normalize against the
exact ideal over the entire eligible frozen corpus, identically for every pair.

```text
discounted evidence gain of the final ranking prefix
----------------------------------------------------
 discounted evidence gain of the ideal corpus ranking
```

**Example:** Ideal gains `[3,1,0]` have discounted gain about 3.631. Returned
gains `[1,3,0]` give about 2.893, so nDCG@3 is approximately `0.80`. If ranks
4/5 add zero in both rankings, nDCG@5 is also approximately `0.80`.

This secondary application measure exposes ordering effects beyond geometric
neighbour preservation. Different chunks containing the same evidence retain
their ordinary linear gain.

**Range and direction:** `[0,1]`; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Zero ideal gain in an otherwise valid frozen corpus | Every chunk fragments the required evidence. | `0` (`scored`), with upstream-fragmentation limitation. | Inherit the shared zero-ideal convention; the database cannot repair those boundaries. |

Shared availability and [integration integrity](#scoring-inputs-eligibility-and-result-status)
apply. Missing evidence annotations are not this zero-ideal case.

### Evidence-unit Recall@3/@5

**Question:** How much distinct required evidence reaches the selected context?

```text
distinct required evidence units fully contained in selected chunks
-----------------------------------------------------------------
                     required evidence units
```

**Example:** The question requires E1, E2 and E3. Top three chunks recover E1,
giving Recall@3 `1/3`; ranks 4/5 add E2, giving Recall@5 `2/3`.

Count each unit once regardless of repetition or position within the cutoff.
This measures coverage, not the rank discounts of nDCG or the exact vector IDs
of ANN recall. Eligible answerable questions have a positive gold denominator.

**Range and direction:** `[0,1]`; higher is better.

**Edge cases and availability:**

No additional empty-reference convention is needed for the eligible task.
[Shared availability](../metrics.md#shared-availability-rules) and
[integration integrity](#scoring-inputs-eligibility-and-result-status) apply.

### Evidence-token Precision@3/@5

**Question:** How much selected context consists of verified evidence?

Use canonical source text and `tiktoken:cl100k_base`, not the embedding/reranker
tokenizer. Count each returned token occurrence once against overlapping gold
intervals; occurrences repeated through different chunks count again.

```text
returned source-token occurrences overlapping verified evidence
--------------------------------------------------------------
                returned source-token occurrences
```

**Example:** Top three chunks contain 120 relevant token occurrences out of
300, giving Precision@3 `0.40`. If ranks 4/5 add 80 relevant occurrences out
of 200, Precision@5 is `200/500 = 0.40`. The other tokens are additional context,
not necessarily false statements.

This measures context concentration; a relevant fragment may contribute tokens
without containing an entire evidence unit.

**Range and direction:** `[0,1]`; higher is better.

**Edge cases and availability:**

The valid integration task has a positive source-token denominator. Token-free
or invalid rankings follow [output integrity](#scoring-inputs-eligibility-and-result-status),
not an artificial zero-denominator score. Shared availability rules apply.

## Diagnostics

### Candidate-pool Evidence-unit Recall@20

**Question:** How much required evidence is available before reranking?

Apply Evidence-unit Recall to the actual verified first-stage pool of the frozen
stack, after fusion for RRF but before learned reranking.

```text
distinct required evidence units contained anywhere in the candidate pool
-----------------------------------------------------------------------
                        required evidence units
```

**Example:** E1 and E2 are in the pool but E3 is absent: `2/3`. Reranking can
change which available chunks reach @3/@5, but cannot introduce E3.

Record the pool identity with this observation. It is not ANN Recall: E1 can be
available in a useful alternative chunk outside the exact nearest-neighbour set.

**Range and direction:** `[0,1]`; higher is a better evidence-availability ceiling.

**Edge cases and availability:**

Eligible questions have a positive gold denominator. A legitimately smaller
corpus uses its available pool; unexpected short integration pools follow
[integrity rules](#scoring-inputs-eligibility-and-result-status). Shared availability applies.

### Ranking Repeatability

**Question:** Does the database reproduce the same ordered retrieved IDs?

Apply the [shared all-pairs rule](../metrics.md#pairwise-repeatability) to the
database output before reranking: compare ordered IDs, not floating-point score
bit patterns. The scope is the requested top 20, or the actual valid returned
list when fewer records are returned; retain requested/actual counts.

```text
scheduled pairs with two valid identical ordered database outputs
----------------------------------------------------------------
                      scheduled pairs
```

**Example:** Lists A,A,B agree only in pair (1,2), giving `1/3`. Failure,A,A
also gives `1/3`, through pair (2,3). Three executed failures give zero, not
agreement. Average per-query pair rates within the declared condition.

This is diagnostic, not a perfect-agreement gate. Valid tied alternative pools
may have perfect ANN recall but less than perfect repeatability. This does not
measure reranked ordering, metric-value repeatability, or independent-index-build
reproducibility.

**Range and direction:** `[0,1]`; higher means more stable output, not more relevance.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Only one attempt deliberately scheduled | Smoke executes one request. | `null` (`inapplicable`). | No pairwise comparison task exists. |
| Valid empty results repeat | An empty-match filter returns `[]` three times. | `1` (`scored`). | Identical valid absence is stable; it does not establish nonempty-search recall. |

Failed and not-executed outcomes follow the shared pair rule, not new local
conventions. Legitimate short lists may agree without meeting integration needs.

## Reliability and filter correctness

Request Error Rates use submitted **logical operations**, not backend batch
counts. Include execution errors, timeouts and unusable response envelopes.
An ordinary low-recall/short ANN result is not an execution error. Completed
checkable filter violations have their own correctness outcome; they are not
relabeled as transport errors. Missing required verification remains unverified,
not an assumed success or failure. Deliberately invalid conformance probes and
warmup requests are outside these measured-operation denominators.

### Request Error Rate — Search

**Question:** What fraction of submitted search operations fail to execute successfully?

```text
failed submitted search operations / submitted search operations
```

**Example:** 10 timeouts among 1,000 submitted searches give `0.01`, or 1%.
Interpret with QPS and successful-request latency; a fast surviving subset does
not describe the failed requests.

**Range and direction:** `[0,1]`; lower is better.

**Edge cases and availability:**

No submitted searches gives `null`, not 0% errors. Deliberately absent search
work is inapplicable; missing required execution is incomplete under shared rules.

### Request Error Rate — Insert

**Question:** What fraction of submitted document insertions fail to complete?

```text
failed submitted document insertions / submitted document insertions
```

**Example:** Two of 40 insertions fail before their visibility barrier, giving
`2/40 = 0.05`. A partially inserted document is one failed logical operation,
not several successful API batches and one unrelated failure.

**Range and direction:** `[0,1]`; lower is better.

**Edge cases and availability:**

No submitted insertions gives `null`. Apply shared task-eligibility and incomplete
execution rules; an operation never submitted is not a failed insertion.

### Request Error Rate — Update

**Question:** What fraction of submitted document replacements fail to complete?

```text
failed submitted document replacements / submitted document replacements
```

**Example:** One of 20 replacements cannot remove obsolete chunks before its
deadline, giving `1/20 = 0.05`. Its acknowledged new chunks do not establish a
completed replacement.

**Range and direction:** `[0,1]`; lower is better.

**Edge cases and availability:**

No submitted updates gives `null`, with task-free versus incomplete status under
shared rules. Do not use insertion counts as this denominator.

### Request Error Rate — Delete

**Question:** What fraction of submitted document deletions fail to complete?

```text
failed submitted document deletions / submitted document deletions
```

**Example:** All 10 submitted deletions fail, giving `1`. This remains visible
even if 10,000 searches succeed; there is no global error rate hiding it.

**Range and direction:** `[0,1]`; lower is better.

**Edge cases and availability:**

No submitted deletions gives `null`, not proof that deletion works. Shared
availability and execution-accounting rules apply.

### Filter Violation Rate

**Question:** How often does a completed checkable filtered response contain
at least one record forbidden by the requested predicate?

Check each returned ID against trusted original metadata. Count responses, not
the number of forbidden records inside them.

```text
completed checkable filtered responses containing a forbidden record
------------------------------------------------------------------
                 completed checkable filtered responses
```

**Example:** Two of 100 checkable responses contain forbidden IDs, giving
`2/100 = 0.02`, even if one response contains five such IDs. Three additional
timeouts are recorded as search failures, not added as filter violations.

This metric measures observed filter leakage, not complete database security or
all filter correctness. Positive/compound-filter correctness probes remain separate.

**Range and direction:** `[0,1]`; lower is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Valid empty response | No returned IDs. | Violation event `0`. | No forbidden record was returned; positive-reference recall can still be poor. |
| Empty allowed subset but a known record is returned | Predicate permits nothing; output contains A. | Violation event `1`. | Every record is forbidden for this predicate. |
| No completed checkable responses | All submitted filtered requests time out. | `null` (`unavailable`). | There is no observed leakage denominator; failures remain visible separately. |

When no filtering task is scheduled, Filter Violation Rate is `null/inapplicable`.
Unknown IDs or missing checks are integrity/verification problems, not evidence
that filtering passed.

## Why the quality metrics are not interchangeable

| Observation | What it can mean |
|---|---|
| High ANN recall, low evidence recall | The database accurately reproduced neighbours that lack the needed evidence. |
| Lower ANN recall, high evidence recall | Useful alternative chunks supplied evidence despite missing some exact neighbours. |
| High pool recall, lower final recall | Evidence was available but did not reach the selected @3/@5 prefix. |
| High final recall, low token precision | Required evidence is present with substantial additional context. |
| Perfect ANN recall, lower repeatability | Different valid tied sets or orderings were returned. |

These are distinct questions, not competing definitions of one universal quality
score. nDCG adds rank sensitivity to evidence coverage; error rates describe
execution rather than nearest-neighbour or evidence correctness.

## Operational performance

Use [shared timing and availability rules](../metrics.md#latency-and-warm-latency).
Database request timers include adapter preparation, serialization, communication,
server work and decoded response. Upstream model inference, offline reference
calculation and quality scoring are excluded. Full-stack timers include the
frozen model-backed retrieval path. Build/startup/visibility boundaries are in
[execution methodology](methodology.md#execution-and-measurements).

Server-load and document-operation percentiles use individual completed logical
operations within their condition. Full-stack percentiles use each question's
median successful measured attempt. Apply the same
[percentile interpolation](../metrics.md#latency-percentiles) to those explicitly
different observation sets. Never average workload-cell percentiles or sum
component percentiles to invent a total. Failed elapsed times remain separate.
Here completed database observations mean the execution-success cohort defined
in [scoring inputs and result status](#scoring-inputs-eligibility-and-result-status), including
separately flagged checkable filter violations, not every decoded invalid response.

### Database Search Latency p50

**Question:** How long does a typical completed database request take?

Take p50 over completed request durations in one declared corpus/filter/load
condition. This is the median, not document-macro averaging.

**Example:** `2,3,4,5,11 ms` gives `4 ms`. A filtered condition has its own list.
This describes request service time at the client boundary, not model inference.

**Range and direction:** non-negative milliseconds/request; lower is better
at comparable quality and load.

**Edge cases and availability:**

Use [shared percentile and availability rules](../metrics.md#latency-percentiles):
one completed observation supplies the descriptive point; none gives unavailable
latency. Do not turn failed elapsed time into successful-completion latency.

### Database Search Latency p95

**Question:** How slow is the completed database-request tail under this load?

Take the interpolated 95th percentile of the same condition's individual completed
durations. It is not the mean of per-query p95 values.

**Example:** For `2,3,4,5,11 ms`, position `4 × 0.95 = 3.8` gives
`5 + 0.8 × (11 - 5) = 9.8 ms`. These five values illustrate arithmetic, not
sufficient evidence for a precise unseen-workload tail estimate.

**Range and direction:** non-negative milliseconds/request; lower is better.

**Edge cases and availability:**

Shared percentile rules apply. Weak independent support limits the confidence
claim, not the existence of a descriptive point. No completed durations is unavailable.

### Full-Stack Retrieval Latency p50

**Question:** How long is a typical complete question-to-retrieved-context wait?

Measure preparation/query embedding, database-backed source retrieval, any frozen
fusion, reranking and validated final chunks. Take each question's median successful
attempt duration, then p50 across those query observations.

**Example:** Three questions have median durations `20,30,100 ms`: p50 is `30 ms`.
This includes model work excluded from Database Search Latency, but no answer
generation. The shared estimator matches retrieval–reranking.

**Range and direction:** non-negative milliseconds/query; lower is better.

**Edge cases and availability:**

Shared completed-stage and percentile rules apply. A completed database search
followed by reranker failure supplies no successful full-stack duration; its
independent database timing remains available.

### Full-Stack Retrieval Latency p95

**Question:** How slow is the full-stack tail across benchmark questions?

Take p95 across the same per-question median successful durations.

**Example:** For `20,30,100 ms`, the interpolated p95 is `93 ms`.
It is the tail of typical per-question runtimes, not the percentile of every
individual model attempt or a sum of stage p95s.

**Range and direction:** non-negative milliseconds/query; lower is better.

**Edge cases and availability:**

Shared percentile/availability rules apply; contributing queries and independent
source support determine the reported scope and CI applicability.

### QPS

**Question:** How many successful search requests complete per wall-clock second?

```text
successful measured search completions / measured load-window wall seconds
```

One request contains one query vector. The denominator includes scheduling,
queuing, failed requests and the declared drain-to-completion/timeout period,
not warmup or setup. Concurrent request durations are not summed.

**Example:** 1,500 submitted searches, 30 failures and 10 measured seconds give
`1,470/10 = 147 QPS`. This is successful throughput at the stated concurrency,
not an automatically discovered server maximum.

**Range and direction:** non-negative queries/second; higher is better at
comparable quality, workload and resources.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No successes in a completely observed positive window | 50 requests fail over 5 seconds. | `0` (`scored`). | No successful work completed during the measured time. |
| No positive usable measured interval | Clock evidence is missing or duration is zero. | `null` (`unavailable`). | Do not divide by zero or fabricate an epsilon duration. |

An interrupted window retains completed work/time as partial diagnostics, not a
completed-window result, under the shared accounting rules. One window has no CI.

### Initial Build Time

**Question:** How long until the prepared corpus and configured indexes are ready?

```text
verified query/index readiness time - start of collection/index setup and ingestion
```

Start from a running empty server. Include schema/metadata index creation,
loading prepared records, required index construction, and readiness checking.
Exclude embedding, oracle computation, downloads and server startup.

**Example:** Setup starts at 10 seconds and all readiness checks finish at
35 seconds: Build Time is `25 seconds`. This captures ingestion readiness,
not merely acknowledgment or an index-only internal timer.

**Range and direction:** non-negative seconds; lower is better for the same corpus.

**Edge cases and availability:**

Use shared completed-phase rules: an unfinished build has no completed Build
Time; retain progress and elapsed-to-failure. One build receives no inferred CI.

### Insert Latency p50

**Question:** How long does a typical prepared document take to become available?

For each submitted document, time first submission through verified availability
of all intended records, including adapter batches and visibility polling.
Take p50 across completed logical insertions in the condition.

**Example:** Completed inserts take `20,30,40,50,100 ms`; p50 is `40 ms`.
A 40-chunk document is one operation, regardless of backend batch count.

**Range and direction:** non-negative milliseconds/document; lower is better.

**Edge cases and availability:**

Shared completed-phase/percentile rules apply. Partial acknowledgment is not a
completed insertion; no completed inserts gives unavailable latency.

### Insert Latency p95

**Question:** How slow are the completed document insertions in the tail?

Take p95 of the same completed logical-insertion durations.

**Example:** `20,30,40,50,100 ms` gives `90 ms`. Retain document chunk/byte sizes
so this is not mistaken for a per-vector insertion time.

**Range and direction:** non-negative milliseconds/document; lower is better.

**Edge cases and availability:**

Shared percentile/support rules apply. Do not give zero latency to failed inserts
or infer a precise tail from a small number of operations.

### Update Latency p50

**Question:** How long does a typical complete document replacement take?

Time first replacement work through verification that new records/metadata are
available and obsolete records are absent. Take p50 over completed replacements.
This is not merely changing one scalar metadata field.

**Example:** Replacing three documents takes `40,60,120 ms`; p50 is `60 ms`.
If a five-chunk document becomes a two-chunk document, both new availability and
removal of the obsolete chunks belong to that operation.

**Range and direction:** non-negative milliseconds/document; lower is better.

**Edge cases and availability:**

Shared completed-phase/percentile rules apply. An acknowledged but incomplete
replacement has no completed update latency; retain its partial state and error.

### Update Latency p95

**Question:** How slow is the tail of complete document replacements?

Take p95 across the same completed replacement durations.

**Example:** `40,60,120 ms` gives `114 ms`. Read it with the update error rate
and operation sizes, not as a guarantee that every replacement finishes in time.

**Range and direction:** non-negative milliseconds/document; lower is better.

**Edge cases and availability:**

Shared percentile/support rules apply. No completed replacement durations means
unavailable latency, not zero.

### Delete Latency p50

**Question:** How long does a typical document deletion take to become visible?

Time first deletion submission through verified absence of the document's records,
then take p50 across completed logical deletions.

**Example:** `5,8,20 ms` gives `8 ms`. Later physical disk reclamation is not
part of the logical deletion timer.

**Range and direction:** non-negative milliseconds/document; lower is better.

**Edge cases and availability:**

Shared completed-phase/percentile rules apply. An acknowledgment without verified
absence supplies no completed deletion latency.

### Delete Latency p95

**Question:** How slow is the tail of verified document deletions?

Take p95 across the same completed logical-deletion durations.

**Example:** `5,8,20 ms` gives `18.8 ms`. This describes deletion visibility,
not storage compaction or image/container removal.

**Range and direction:** non-negative milliseconds/document; lower is better.

**Edge cases and availability:**

Shared percentile/support rules apply. A task-free deletion condition is inapplicable;
a required task without completed durations is unavailable/incomplete as appropriate.

### Startup Time

**Question:** How long until an existing stopped database serves its preserved data?

```text
successful required query/state verification time - server start request time
```

Start with the populated server already stopped. Include readiness, preservation
and configured-index checks and the required query. Exclude the preceding stop,
image pulls and original corpus build. This does not imply an empty OS cache.

**Example:** Start at 0 seconds, health at 3, required state/query verification
at 4: Startup Time is `4 seconds`, not 3.

**Range and direction:** non-negative seconds; lower is better with verified state.

**Edge cases and availability:**

Shared completed-phase rules apply. A server that becomes healthy but never
serves the required preserved state has no completed Startup Time. One start has no CI.

### Peak Server RAM

**Question:** What is the highest observed server memory usage in the declared lifecycle?

Take the maximum usable memory sample under the frozen container/cgroup method.
For a declared multi-container server, sum simultaneous usage before taking the
peak. Client model memory and unrelated services are outside this server scope.

**Example:** Samples `600,900,750 MiB` give `900 MiB`, stored as
`943,718,400 bytes`. Do not sum each component's separately timed maximum.
Record cache accounting; raw Docker usage and cache-subtracted CLI memory are
not interchangeable measurements.

**Range and direction:** non-negative bytes; lower at comparable functionality
and workload. Sampling may miss shorter spikes.

**Edge cases and availability:**

Use [shared resource rules](../metrics.md#resource-measurements), including retained
partial peaks, missing telemetry and single-peak uncertainty. CPU-only server
execution does not acquire a zero-valued GPU-memory metric.

## Storage descriptors

### Persistent Storage Size

**Question:** How much persistent state does the completed database require?

Sum apparent bytes of dedicated database data, index, metadata and required log
files at the synchronized completed-build checkpoint. Exclude Docker images,
model snapshots, client memory and benchmark artifacts. Record measurement tool
and checkpoint; allocated disk blocks are a different quantity.

**Example:** `1,048,576,000 bytes` is `1,000 MiB` of apparent persistent state.
This exceeds raw vector payload size legitimately because database structures
also occupy space. It is not a reranker/embedding snapshot-size measurement.

**Range and direction:** non-negative bytes. This is a descriptor, not a quality
score; a smaller footprint is advantageous only at comparable functionality.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No trustworthy completed-build checkpoint | Interrupted loading leaves measurable partial files. | Completed size is `null` (`unavailable`); retain labelled partial footprint. | Partial bytes do not establish the space needed by the complete database. |

Missing telemetry follows shared availability rules. Logical deletion need not
reduce later physical footprint immediately; those later snapshots do not replace
the defined post-build measurement. One checkpoint has no inferred CI.

## Worked result interpretation

For one illustrative pair on aligned workloads:

```text
Unfiltered ANN Recall@20              = 0.98
Filtered ANN Recall@20                = 0.90
Candidate-pool Evidence-unit Recall@20 = 0.95
nDCG@5                               = 0.86
Evidence-unit Recall@5                = 0.80
Evidence-token Precision@5            = 0.44
Ranking Repeatability                = 0.90
Database Search Latency p95           = 12 ms
Full-Stack Retrieval Latency p95      = 90 ms
Search Request Error Rate            = 0.01
Filter Violation Rate                = 0
```

The search preserves most exact neighbours, with more loss under the stated
filter condition. Application pool recall is high, but not all evidence reaches
the final five chunks. Token precision describes context concentration, not
the truth of the remaining text. Repeatability exposes ordering changes without
making them integrity failures. The latency values cover different paths and
cannot be subtracted to calculate reranker p95. Zero observed filter violations
is conditional on completed checkable responses; the 1% search failure rate
remains visible. None of these values alone selects the database winner.

## Aggregation and confidence intervals

Use [shared uncertainty conventions](../metrics.md#confidence-intervals), preserving
metric-specific observations rather than applying quality's macro mean everywhere.

### Aggregation and completeness

| Metric family | Aggregate |
|---|---|
| ANN/Filtered ANN Recall | Mean first-response per-query scores within each condition; never pool different effective neighbour denominators. |
| nDCG, evidence-unit/token metrics, candidate-pool evidence recall | Mean eligible questions within each source document, then document-macro mean. |
| Database Ranking Repeatability | Mean scheduled-pair score within each query, then mean query scores within the condition. |
| Database load and document-operation latency | Named percentile of individual completed logical operations within the condition. |
| Full-stack latency | Named percentile of per-question median successful measured durations. |
| Request Error Rate | Failed submitted logical operations / submitted operations, separately by operation and condition. |
| Filter Violation Rate | Violating completed checkable responses / completed checkable filtered responses. |
| QPS | Successful completions / complete corresponding load-window wall time. |
| Build/startup/storage | Observed phase duration or checkpoint size; do not average unrelated corpus lifecycles. |
| Server RAM | Maximum usable sample over the declared monitoring window. |

**Example:** Evidence recalls `1,0` from one document and `1` from another give
document-macro `(0.5 + 1)/2 = 0.75`. ANN recalls `1,0.8,0.6` for three queries
give `0.8` within that workload. These answer different questions. Drawing source
groups for a latency CI does not turn the percentile into document-macro latency.

Report planned, eligible, contributing, failed, unavailable and independent-source
counts where applicable. Do not silently drop required missing quality from an
authoritative aggregate. Conditional successful latency/filter cohorts retain
their outcome coverage. Different workload conditions never become one pooled
database score.

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---|---|
| ANN quality and database repeatability | Conditional on sufficient independent support | Resample complete queries, or source groups when queries share a source; retain attempts and first-response identities. |
| Application evidence quality and pool recall | Conditional on sufficient independent sources | Resample complete source groups and recalculate document-macro scores. |
| Full-stack p50/p95 | Conditional | Resample source groups with their query medians, then recalculate percentiles. |
| Controlled-query Database Search Latency and search/filter rates | Conditional on sufficient independent queries/source groups | Retain each query's attempts and outcomes together; recalculate request-level percentiles or the declared count ratio, not per-query latency medians. |
| Server-load latency, QPS and request/filter rates | Conditional on repeated independent load windows or a reviewed dependence-aware design | Preserve concurrent request dependence; individual requests are not independent load experiments. |
| Standalone mutation latency/error rates | Conditional on independent operation/source groups and reviewed support | Keep related documents and repeated mutations together; preserve operation boundaries. |
| Smoke, fixed settings, inventories and check states | No authoritative interval | Wiring observations or fixed facts, not supported selection estimates. |
| One build/startup, lifecycle peak or storage checkpoint | No | One observation cannot establish execution-to-execution uncertainty. |

### Calculation

Apply the shared 10,000-resample, seed-42 percentile procedure offline. Quality
draws use whole independent query/source groups, retaining failed and unavailable
first observations and dependent attempts. Source grouping preserves dependence;
it does not change the metric's declared query-mean or document-macro weights.
Retain metadata distinguishing query-sampling uncertainty from variation between
independently rebuilt indexes, which one index does not measure.

For load uncertainty, resample independent complete load windows with their
requests, failures, work counts and wall durations. Recompute percentiles over
the sampled completed-request observations and QPS as total completions divided
by total sampled wall time. Do not bootstrap concurrent requests as if each were
a new experiment, or divide by summed per-request latency. Paired comparisons
use matched window schedules/conditions and aligned source draws across candidates.
If a condition lacks the reviewed window/support design, retain descriptive
points and null bounds rather than claiming precision from 10,000 resamples.

Freeze support requirements after [data review](../pending-data-review.md#vector-databases)
and before authoritative comparisons, including development, not merely before
choosing finalists. Shared undefined-draw and degeneracy rules apply. Repeated
identical queries add workload but not new independent sources. No nonzero-width
CI is fabricated when all defined bootstrap statistics agree.

### Interpretation

ANN Recall `0.94` with a supported 95% interval `[0.92,0.96]` describes uncertainty
in the condition's mean recall, not a guarantee that 95% of queries have those
scores. A p95 latency is a percentile of durations, not a confidence bound.
Intervals do not excuse incomplete required evidence or choose finalists automatically.
