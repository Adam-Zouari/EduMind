# Vector database metrics

[Shared metric conventions](../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../running.md)

On this page:

- [Reading the results](#reading-the-results)
- [Search and filter quality](#search-and-filter-quality)
- [Real-retrieval quality](#real-retrieval-quality)
- [Conformance validity gates](#conformance-validity-gates)
- [Performance and resources](#performance-and-resources)
- [Support, aggregation, and confidence intervals](#support-aggregation-and-confidence-intervals)
- [Current measurement limitations](#current-measurement-limitations)

NumPy exact cosine search is the oracle for neighbour preservation, not a
production candidate. Keep workload, dimension, K, filter selectivity, and
concurrency separate. Primary metrics are ANN Recall and Filtered ANN Recall
at 3, 5, and 10. Their rank-one versions and real-evidence retrieval scores
are secondary. Correctness gates cannot be traded against speed, and there
is no weighted winner score.

These definitions are the reporting contract. The implementation does not yet
satisfy every rule; [current limitations](#current-measurement-limitations)
identify the differences. All examples are illustrative arithmetic, not
measured results or server-selection evidence.

## Reading the results

Reference eligibility is fixed before execution. Distinguish defined numeric
observations (`scored`), absent reference tasks (`inapplicable`), unobtainable
measurements (`unavailable`), and missing scheduled execution (`incomplete`).
Inapplicable and unavailable fields retain null and a reason; missing requests
are not fabricated observed failures. Invalid vectors, predicates, or references
must be repaired before authoritative execution, not scored as poor quality.

A failed designated quality request is not a valid empty response: its
output-dependent quality is null/unavailable. Its executed failure still enters
the appropriate failure/gate observation. Do not substitute a later successful
or best output. Valid earlier quality, latency, and resource observations survive
a later failure with their actual lifecycle scope; they do not certify completion
of the rest of the run.

Retain scheduled, submitted, successful, failed, not-executed, reference-eligible,
scored, and unavailable counts as applicable. Omit unavailable MLflow numeric
scalars but keep their statuses/reasons in artifacts. A quality mean over
available outputs describes completed-output quality, not the whole workload.

## Search and filter quality

### ANN Recall@K

**Question:** Which exact neighbours remain in the first K server results?

Rank the complete normalized-vector corpus by exact cosine similarity. The
oracle breaks exact ties by ascending corpus index. For one query:

```text
distinct returned top-K IDs also in the exact top-K
-------------------------------------------------
             number of exact top-K IDs
```

The denominator is the smaller of K and eligible corpus size. Compare only
the first K returned IDs; ordering inside that prefix does not change recall.
Average query scores equally within a workload cell. Duplicate or unknown
returned IDs are invalid responses, not extra credit. Near-duplicate vectors
with different legitimate IDs remain different records.

**Examples by cutoff:** For exact order `A,B,C,D,E,F,G,H,I,J` and server
order `B,A,X,D,Y,F,G,Z,I,J`, where `X,Y,Z` are legitimate corpus IDs outside
the exact top ten:

- **ANN Recall@1:** `A` is absent at rank one: `0 / 1 = 0`. This secondary
  diagnostic is stricter than finding `A` later in the list.
- **ANN Recall@3:** `A,B` are preserved: `2 / 3 ≈ 0.67`.
- **ANN Recall@5:** `A,B,D` are preserved: `3 / 5 = 0.60`.
- **ANN Recall@10:** seven exact IDs are preserved: `7 / 10 = 0.70`.

High recall means exact-neighbour preservation, not relevance to an educational
question. Recall need not increase with K because both prefixes and denominators
change. If only two oracle records exist at @5, returning one earns
`1 / 2 = 0.50`, not 0.20.

**Range and direction:** all cutoffs lie in `[0,1]`; higher is better. A valid
empty response against a nonempty oracle earns zero; a failed request has
unavailable quality. An empty unfiltered corpus is not the declared workload.
Apply the query-level support and CI rules below to each cutoff separately.

### Filtered ANN Recall@K

**Question:** Which exact neighbours survive when the required predicate applies?

Apply the identical frozen predicate to the oracle corpus and server request
before selecting neighbours. Divide matching distinct top-K IDs by the number
of exact filtered top-K IDs. Average queries separately in each selectivity
band; an unfiltered oracle is not a valid denominator.

**Examples by cutoff:** For filtered oracle order `P,Q,R,S,T,U,V,W,X,Y`
and predicate-valid server order `P,Q,Z,S,N,U,V,M,X,Y`:

- **Filtered ANN Recall@1:** `P` is preserved: `1 / 1 = 1` (secondary).
- **Filtered ANN Recall@3:** `P,Q` are preserved: `2 / 3 ≈ 0.67`.
- **Filtered ANN Recall@5:** `P,Q,S` are preserved: `3 / 5 = 0.60`.
- **Filtered ANN Recall@10:** seven are preserved: `7 / 10 = 0.70`.

A missed neighbour need not be a filter violation: `Z` can satisfy the
predicate but be less similar than an omitted exact neighbour. If only two
records satisfy a predicate, retaining both at @5 earns `2 / 2 = 1`.

**Range and direction:** all cutoffs lie in `[0,1]`; higher is better. An
empty filtered oracle makes recall inapplicable; Empty-Filter Correctness
evaluates that case. Missing neighbours in a valid response earn zero credit;
a failed nonempty-filter request has unavailable quality. Calculate query-level
CIs within each band, not by pooling unlike predicates.

### Filter Correctness

**Question:** Does every returned record obey every requested predicate?

Assign one to a successful request only when all returned metadata satisfy
every predicate, including conjunctions; otherwise assign zero. The rate is
passed requests divided by all accounted-for requests in the filter cohort.
An executed request failure scores zero for this validity check, not for recall.

**Example:** Nine valid responses and one response leaking another document
among ten requests give `9 / 10 = 0.90`, failing conformance regardless of speed.
A successful empty response passes this leakage check vacuously; Filtered ANN
Recall exposes missing nonempty results.

**Range and direction:** `[0,1]`; must equal `1.0`. Report each filter cohort
and the required all-checks outcome. No requests means no rate; unexecuted
required checks mean incomplete, not a pass. Query-bootstrap intervals can
describe sample variation but cannot relax this gate.

### Empty-Filter Correctness

**Question:** Does a predicate with no possible match return nothing?

Count a pass only when a verified-empty predicate completes with zero records.
Divide passes by all accounted-for verified-empty requests. A request failure
or any returned record earns zero.

**Example:** Nineteen empty successes and one leaked record among twenty requests
give `19 / 20 = 0.95`, which fails. Twenty empty successes give one, but do not
create twenty perfect filtered-recall observations.

**Range and direction:** `[0,1]`; must equal `1.0`. No verified-empty requests
means no test evidence, not perfection. Missing required checks are incomplete.
Any query-bootstrap CI is descriptive and does not replace the pass requirement.

## Real-retrieval quality

Finalists receive the same canonical chunks, precomputed document vectors,
questions, evidence annotations, and frozen retrieval strategy. Databases never
create embeddings. These are evidence-retrieval scores, not generated-answer
scores or exact-neighbour recall. No context-token budget is applied.

Quality uses only the designated first measured ranking. Later attempts measure
timing and ranking agreement; they do not average quality or replace a failed
first ranking. Each metric below is calculated for answerable questions with
verified required evidence, averaged within each source document, then
macro-averaged over documents. Unanswerable questions are inapplicable; missing
required annotations are invalid inputs. Failed first rankings are unavailable,
not empty rankings. The complete retrieval contract requires the frozen top-20
pool; an unexpectedly empty/short ranking or invalid permutation is an integrity
failure, not a valid zero-quality response. Detailed evidence identities and
token accounting follow
the [retrieval metric reference](../rag/retrieval_reranking/metrics.md).

### Real-retrieval nDCG@3/@5

**Question:** Are chunks containing more required evidence ranked earlier?

A chunk's linear gain is the number of distinct required units it completely
covers. Divide each gain by `log2(rank + 1)`, add through K, then divide by
the ideal discounted sum at K. The ideal sorts gains from the complete frozen
corpus, not the returned pool. Repeated coverage in distinct chunks retains
ordinary nDCG gain.

**Example:** Corpus gains are `2,1,0,0,0`; returned gains are `1,2,0,0,0`.
At @3, the returned sum is `1 + 2 / log2(3) ≈ 2.262`; the ideal is
`2 + 1 / log2(3) ≈ 2.631`. nDCG@3 is about `0.860`. Here ranks 4 and 5
add zero, so nDCG@5 is also `0.860`. Moving the two-unit chunk first gives
one at both cutoffs.

**Range and direction:** `[0,1]`; higher is better. A valid ranking with no
complete evidence units scores zero. A zero ideal under the reference scorer
returns zero rather than dividing by zero; retain that corpus-coverage
limitation. Failed rankings are unavailable. Use document-level support and
CIs, not synthetic-query intervals.

### Real-retrieval Evidence-unit Recall@3/@5

**Question:** How much distinct required evidence appears anywhere in the prefix?

Divide required evidence-unit IDs completely covered by at least one returned
chunk through K by all required unit IDs. Each unit counts once regardless
of repetition or position inside the prefix.

**Example:** A question needs `A,B,C`. Top three chunks cover only `A`:
Recall@3 is `1 / 3 ≈ 0.33`. Ranks 4 and 5 add `B`: Recall@5 is
`2 / 3 ≈ 0.67`. Another copy of `A` adds no recall. High recall establishes
availability, not early placement or concise context.

**Range and direction:** `[0,1]`; higher is better. Partial unit fragments do
not earn full-unit credit. A valid complete ranking with no recovered unit
earns zero; failed or invalid retrieval is unavailable. Missing required-unit
annotations are not an empty task to score perfectly. Use document-macro support
and CIs.

### Real-retrieval Evidence-token Precision@3/@5

**Question:** What share of returned text is annotated evidence?

Using `tiktoken:cl100k_base` on canonical source text, divide returned token
occurrences inside verified evidence intervals by all returned token occurrences
through K. Overlapping chunks count a source token each time it consumes context;
overlapping evidence annotations do not multiply credit within one occurrence.

**Example:** At @3, 120 of 300 token occurrences are evidence:
`120 / 300 = 0.40`. Ranks 4 and 5 add 180 evidence tokens in 200 total tokens;
Precision@5 is `(120 + 180) / (300 + 200) = 0.60`.
Additional chunks can raise or lower concentration; extra text is not necessarily
false text.

**Range and direction:** `[0,1]`; higher is better. A valid complete ranking
with no evidence-bearing tokens scores zero. An unexpectedly empty/invalid
ranking is unavailable, not perfect precision or an invented empty result.
Use document-macro support and CIs; token occurrences are not independent samples.

## Conformance validity gates

Each controlled check below is binary: one means the observed check passes,
zero means it fails. A single check has no CI. All required checks must pass;
missing checks are incomplete, not assumed passes. Exceptions retain their reason
and cannot prove that a semantic operation worked. Performance may survive as
diagnostic evidence but cannot compensate for non-conformance. Filter rates
are explained above.

### Health Correctness

**Question:** Is the endpoint ready? Record one when the required adapter
health check succeeds, otherwise zero. An unreachable loopback endpoint fails;
a valid required health response passes. This proves connectivity, not stored
vector correctness. **Range:** `{0,1}`; must pass; no CI.

### Cosine Correctness

**Question:** Does a controlled cosine probe find its expected neighbour?
Record one when the known probe returns its expected nearest ID. Querying
`A`'s unique normalized vector should return `A` at rank one; `B` fails.
Tied vectors require the frozen tie rule to support a unique-ID probe.
This is a small semantic check, not corpus ANN recall.
**Range:** `{0,1}`; must pass; no CI.

### Dimension-Rejection Correctness

**Question:** Is a wrong-length vector rejected? Rejection passes; acceptance
fails. A 385-value vector in a 384-dimensional collection must be rejected.
An unrelated connection exception does not establish dimension enforcement.
**Range:** `{0,1}`; must pass; no CI.

### Compound-Filter Correctness

**Question:** Are all predicates enforced together? The positive controlled
probe must return matching records and no record outside the conjunction.
For `course=A AND page=2`, a `course=A,page=3` hit fails. Empty output also
fails this positive probe, unlike the general leakage metric.
**Range:** `{0,1}`; must pass; no CI.

### Duplicate-ID Correctness

**Question:** Does upserting an existing ID avoid creating a second record?
Pass when uniqueness and cardinality are preserved. Replacing ID `0` in a
40-record collection should leave 40 records, not 41. This alone does not prove
that updated values are visible; replacement has a separate contract.
**Range:** `{0,1}`; must pass; no CI.

### Replacement Correctness

**Question:** Are obsolete values/chunks removed when an ID/document is replaced?
Pass only when new vector/metadata values are visible and obsolete document chunks
are absent. Replacing document `D`'s five old chunks with one new chunk must
leave only the new ID under `D`'s filter, not six searchable records.
A count check alone cannot verify updated vector/metadata contents.
**Range:** `{0,1}`; must pass; no CI.

### Deletion Correctness

**Question:** Are deleted IDs/documents absent from later search and filtering?
Pass only after visibility verification. If `D:new` is the last chunk of
document `D`, deleting it must leave an empty `D` filter result.
A successful delete acknowledgment alone is insufficient.
**Range:** `{0,1}`; must pass; no CI.

### Restart-Persistence Correctness

**Question:** Does committed state survive restart? Pass when records and their
required vectors/metadata survive. Thirty-five retained records must remain
after restart; 34 fails. A matching count of 35 alone cannot prove that the same
contents survived. **Range:** `{0,1}`; must pass; no CI.

### ANN-Index Verification

**Question:** Is the configured ANN index available and used where required?
Verify index state before measurement and again after restart. A configured
pgvector HNSW index needs matching index/plan evidence, not a silent alternate
path for the required ANN probe. Qdrant's documented small-filter exact-search
exception remains unchanged. **Range:** `{0,1}` per check; both before/after
checks must pass; no CI.

### Target-Concurrency Success

**Question:** Does the highest declared concurrency finish without request errors?
Record one only when all submitted requests in that target cell succeed.
For 500 queries with three repeats, 1,500 successes pass; 1,499 successes fail.
Lower-concurrency error rates remain visible without changing this gate's scope.
**Range:** `{0,1}`; must pass; no CI for this observed cell state.

### Real-retrieval Ranking Agreement

**Question:** Do repeated complete rankings agree with the first ranking?
The retrieval contract divides matching later ordered-ID lists by expected
later comparisons. Three attempts `A,A,B` have one match among two comparisons:
`1 / 2 = 0.50`, failing the gate; `A,A,A` passes. Failed comparisons never
earn agreement. This is not extraction's three-pair Repeatability Success Rate,
and does not add that metric to synthetic ANN.
**Range:** `[0,1]`; must equal one. The current vector wrapper instead emits
an all-repetitions-equal flag per question; both representations fail this
example. A single observed flag has no CI; an average cannot excuse one failure.

## Performance and resources

The server-only timer surrounds the adapter call: serialization, loopback
transfer, server work, and complete response decoding are included. Thread-pool
queue waiting before the timer begins is not per-request latency, but remains
inside batch wall time. Embedding, reranking, exact-oracle computation, and
scoring are excluded; label full-stack real-retrieval timing separately.

Only successful completions enter latency percentiles. Retain failures and
coverage separately. The examples use NumPy linear quantiles: sort n values,
locate position `(n - 1) × percentile`, then interpolate adjacent values.
A percentile is not an exact percentage of a tiny observed sample.

### Unfiltered / Filtered Latency p50

**Question:** What does a typical successful search take? Take the median warm
request time separately for unfiltered and each filtered/concurrency cell.
For separately labeled complete-retrieval results, apply the same percentile
calculation to that full-stack timer, not to server-only observations.

**Example:** `2,3,4,5,11 ms` gives p50 `4 ms`; filtered
`3,4,6,8,15 ms` gives `6 ms`. Do not pool the two lists.
**Range and direction:** non-negative seconds/request; lower is better at equal
quality/correctness. No successes means unavailable, not zero. Query-cluster
CIs require adequate independent support.

### Unfiltered / Filtered Latency p95

**Question:** How slow is the warm search tail? Take the interpolated 95th
percentile of successful times in the declared cell, not the mean of query p95s.

**Example:** For `2,3,4,5,11 ms`, position `4 × 0.95 = 3.8` gives
`5 + 0.8 × (11 - 5) = 9.8 ms`. Filtered `3,4,6,8,15 ms` gives
`8 + 0.8 × (15 - 8) = 13.6 ms`. These tiny lists illustrate arithmetic,
not sufficient tail support.
**Range and direction:** non-negative seconds/request; lower is better.
No successes means unavailable. Report failures and query-cluster CIs only
when independent support suffices.

### Unfiltered / Filtered Latency p99

**Question:** How extreme is the successful-request tail? Take the interpolated
99th percentile separately in each declared cell.

**Example:** For `2,3,4,5,11 ms`, position `3.96` gives
`5 + 0.96 × (11 - 5) = 10.76 ms`; filtered `3,4,6,8,15 ms` gives
`14.72 ms`. Neither tiny list supports an authoritative p99.
**Range and direction:** non-negative seconds/request; lower is better.
Insufficient successful requests or independent query IDs means null with
`insufficient_requests`; no successes means unavailable. The support minimum
must be frozen after data review, before authoritative runs. Repeating one
query does not create independent support. The numeric minimum remains pending
that review.

### Query Throughput

**Question:** How many successful requests finish per wall-clock second?

```text
successful measured responses / complete measured batch seconds
```

Include dispatch, queue waiting, and failed-request time; exclude setup,
build, warmup, and offline scoring. Report every concurrency independently.
**Example:** 1,500 submissions, 30 failures, and a 10-second interval give
`1,470 / 10 = 147 successful queries/second`, not 150.
High throughput alone establishes neither correct filtering nor good recall.
**Range and direction:** non-negative successful queries/second; higher is
better. Zero successes over a complete positive interval gives zero; no batch
or non-positive/missing duration is unavailable. Partial batches are diagnostic,
not completed-phase evidence. One batch has no CI.

### Request Error Rate

**Question:** What fraction of submitted requests fail? Divide executed failed
requests by submitted measured requests within a concurrency cell. Count
timeouts, transport errors, and invalid responses, not unsubmitted work.

**Example:** 30 failures among 1,500 submissions give
`30 / 1,500 = 0.02`, or 2%. Read with throughput and success-only latency.
**Range and direction:** `[0,1]`; lower is better. No submissions means no
rate. An interruption also reports unsent work and incomplete status.
The current phase-level rate has no CI; query-cluster CIs would require
retained request outcomes and adequate independent support.

### Build Time

**Question:** How long until an empty ready server has a queryable index?
Subtract first-vector submission start from the time all vectors are queryable
and the index is ready. **Example:** submission at `10 s`, readiness at
`35 s`: `35 - 10 = 25 s`. Model preparation and server downloads are outside.
**Range and direction:** non-negative seconds; lower is better for equal corpus
and settings. Failed indexing has unavailable completed Build Time, with
elapsed-to-failure and partial counts retained. One build has no CI.

### Build Throughput

**Question:** At what average rate did initial indexing become ready? Divide
successfully indexed queryable vectors by Build Time.
**Example:** `100,000 / 25 = 4,000 vectors/second`. This is not query throughput
or a promise of constant insertion speed.
**Range and direction:** non-negative vectors/second; higher is better at equal
settings/corpus. Failed builds retain partial diagnostics, not a completed-build
rate. Zero/missing duration is unavailable, not division by an artificial small
number. One build has no CI.

### Incremental Upsert Throughput

**Question:** How quickly does a populated index make new mutations visible?
Divide successful upserted records by elapsed time through query visibility.
**Example:** 1,000 mutations visible in 2.5 seconds give
`1,000 / 2.5 = 400 upserts/second`. Record whether these insert new IDs or
replace existing IDs; they are different workloads.
**Range and direction:** non-negative records/second; higher is better.
Unverified/failed phases have unavailable completed throughput with partial
acknowledgments retained. Non-positive/missing timing is unavailable.
One phase has no CI; acknowledgment-only timing does not prove visibility.

### Incremental Delete Throughput

**Question:** How quickly does deletion remove records from queries? Divide
successfully deleted records by elapsed time through verified absence.
**Example:** 1,000 visible deletions in four seconds give
`1,000 / 4 = 250 deletions/second`, independently of upsert speed.
**Range and direction:** non-negative records/second; higher is better.
An undeclared deletion task is inapplicable; a required phase that never runs
is incomplete. Failed/unverified deletion is unavailable, not an invented zero
rate. Retain partial counts. Missing/non-positive duration
is unavailable. One phase has no CI.

### Restart Readiness

**Question:** How long before persisted indexed state is usable after restart?
Measure from restart request through health and persisted-index query
verification. **Example:** restart at zero, health at three seconds, verification
at four seconds gives `4 s`, not three.
**Range and direction:** non-negative seconds; lower is better only with passing
persistence/index gates. A timeout has unavailable completed latency and retained
elapsed-to-failure. One restart has no CI. The current timer ends at health,
before later index checks, and must be labeled as that narrower observation.

### First-Query Latency after Restart

**Question:** How long does the first workload query after readiness take?
Measure its adapter-call start through a decoded successful response, outside
warm percentiles. **Example:** start at `4.0 s`, completion at `4.08 s` gives
`0.08 s`; do not add restart duration. Readiness verification may already
issue queries, so this does not promise untouched server caches.
**Range and direction:** non-negative seconds/request; lower is better.
A failed first query has unavailable successful latency and retained failure
duration; do not rename a later success as the first. One probe has no CI.
The current probe uses the small conformance corpus, not the full workload.

### Peak Server RAM

**Question:** What is the largest observed server memory footprint? Take the
maximum sampled usage, not a sum of peaks from different times or the client.
The current collector uses Docker-reported container memory usage converted to
bytes, not a direct process-tree RSS sum. Record method and lifecycle scope.

**Example:** samples `600,900,750 MiB` give `900 MiB`, or
`943,718,400 bytes`. Sparse samples can miss a shorter spike.
**Range and direction:** non-negative bytes; lower at equal correctness/quality.
Missing monitoring is unavailable, not zero. Pre-failure peaks survive only as
partial-scope diagnostics. One peak has no CI. CPU server execution has no
applicable GPU-memory metric; CPU VRAM is null/inapplicable.

### Persistent Storage

**Question:** How much persistent server state remains at the recorded checkpoint?
Measure dedicated server data after declared writes are synchronized and before
teardown. Exclude image layers, model snapshots, and client memory. Record
whether the tool reports apparent bytes or allocated blocks.

**Example:** `du -sb` reporting `1,048,576,000 bytes` represents
`1,000 MiB` of apparent state. A volume inspector reporting
`1,024,000 KiB` converts to the same bytes but measures allocated usage;
equal units do not establish equal measurement semantics.
**Range and direction:** non-negative bytes; lower at equal correctness/quality.
Missing data is unavailable. A partial run's footprint names its checkpoint;
deleted records may leave unreclaimed storage. One snapshot has no CI.
The current final snapshot is not a synchronized post-build observation per
workload.

## Support, aggregation, and confidence intervals

Synthetic ANN quality is scored once per frozen query, then averaged equally
within each workload/filter cell. Scores `1,0.8,0.6` average to
`(1 + 0.8 + 0.6) / 3 = 0.8`; do not pool neighbour counts when denominators
differ. Real-retrieval quality instead averages questions within source
documents, then documents equally. Counts, settings, revisions, and binary
phase states are descriptors, not sampled quality estimates.

For eligible development, validation, and planned locked sample-based results,
use 10,000 bootstrap resamples with seed 42. Resample complete query IDs for
synthetic ANN, preserving compared servers, filter conditions, and dependent
repeats together. Real retrieval resamples independent source documents with
all their questions. Recalculate each aggregate; the 2.5th/97.5th percentiles
are the estimated 95% bounds. These are arithmetic resamples of saved results,
not extra inference or search requests.

Latency CIs recompute the declared percentile from successful observations in
each query-cluster draw. Repetitions do not increase independent support.
Full-stack real retrieval retains source-document clusters. Paired comparisons
use aligned values and the same sampled IDs, with their common scored cohort
and exclusions reported; success-only differences do not represent failures.

Draw from the reference-fixed eligible cohort, retaining unavailable records
rather than first filtering to successes. Aggregate defined values in each draw;
retain contributing counts and undefined-draw reasons. No eligible observations
gives no estimate or interval. Below a frozen independent support minimum, keep
a descriptive point estimate and null bounds with the reason. Do not invent a
threshold or a zero-width CI. Adequately supported identical observations can
produce coincident calculated bounds; flag a degenerate interval, not future
certainty. Missing required values remain incomplete evidence even if some
conditional bounds can be calculated.

For example, recall `0.94` with 95% bounds `[0.92,0.96]` describes uncertainty
in the aggregate, not a guarantee that 95% of individual queries score in that
range. Smoke has no authoritative CI. One-off batches, builds, mutation phases,
restarts, peaks, and storage snapshots have no CI. Independent operational repeats
could support one only with recorded repetition count, unit, and aggregation.

## Current measurement limitations

The implementation in `experiments/benchmarks/vectordb/` does not yet meet all
the reporting rules above:

- Dense ANN quality uses one serial request per query; the three measured
  repetitions apply to concurrent load traffic. Serial quality/filter latency
  precedes those load warmups. Filtered searches are not repeated at each
  concurrency; their serial values are not warm filtered load-cell measurements.
- Serial search exceptions abort a candidate without explicit unavailable
  per-query rows or complete retention of earlier evaluator-local observations.
  The recall helper deduplicates returned IDs rather than rejecting them;
  empty-oracle recall emits an empty-result indicator instead of inapplicability.
- Concurrent throughput divides submitted requests by elapsed time even after
  errors. That is offered request rate, not successful throughput when errors
  occur. Separate error rates and successful latencies lack the complete
  request ledger/support policy.
- p99 has no independent-support gate, and phase-level latency has no
  query-cluster CIs. Generic quality aggregation pools workload sample rows;
  filter-specific keys do not supply a complete per-cell report. Real-retrieval
  generic aggregation averages questions, not source-document macro values.
- The real-retrieval wrapper still calls the legacy overlap-graded scorer in
  `rag/evaluation.py`: its nDCG uses exponential overlap grades, and its context
  recall/precision are not Evidence-unit Recall/Evidence-token Precision. Those
  keys cannot be relabeled as the evidence-count metrics defined above. The
  current wrapper has not migrated to the retrieval benchmark's evidence scorer.
- Tuning broadly classifies trial exceptions as unsupported configurations;
  those labels alone do not distinguish genuine unsupported settings from
  infrastructure or request failures. Real-retrieval output validation also
  lacks the full checksummed-pool/permutation contract of the retrieval runner.
- Mutation timers end at adapter return without a uniform visibility barrier.
  Restart timing ends at health; persistence checks compare counts. The timed
  first-query probe uses the conformance corpus. Dimension rejection accepts
  any exception; replacement checks do not exhaustively verify updated values.
  These are limited probes, not proof of every required semantic guarantee.
- Docker memory monitoring spans tuning, conformance/restart, build, search,
  and mutation across the candidate's workloads. Storage is sampled at the end,
  with apparent-byte/allocated-block fallback methods and no uniform explicit
  synchronization checkpoint. Missing telemetry is omitted numerically without
  the full required status record.
- Gates are emitted as metrics, but the vector call does not supply a complete
  required-metric/gate validator to the common runner. A recorded `success`
  status alone does not prove all correctness values equal one; the
  real-retrieval `determinism` flag also needs explicit gate review.

These gaps qualify the evidence; they do not authorize new cutoffs, repeats,
candidate rosters, narrower gates, or automatic winner selection. See the
[methodology implementation status](methodology.md#implementation-status) for
frozen finalist settings and locked execution.
