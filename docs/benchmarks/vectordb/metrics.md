# Vector databases metrics

[Shared metric conventions](../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Search and filter quality](#search-and-filter-quality-1)
- [Conformance validity gates](#conformance-validity-gates)
- [Performance and resources](#performance-and-resources)
- [Eligibility, aggregation, and confidence intervals](#eligibility-aggregation-and-confidence-intervals)

The NumPy exact cosine search result is the oracle for ANN quality; it is not a
production candidate. Results are reported separately by `K`, filter
selectivity, concurrency, vector dimension, and workload profile rather than
pooling unlike conditions into one score.

## Metric summary

### Search and filter quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| ANN Recall@3/@5/@10 | Primary | How many exact nearest neighbours does the approximate server preserve at application-relevant depths? | Higher |
| Filtered ANN Recall@3/@5/@10 | Primary | Does the server preserve exact neighbours after applying the required metadata filter? | Higher |
| Filter Correctness | Validity gate | Do all returned records satisfy every requested predicate? | Must equal 1.0 |
| Empty-Filter Correctness | Validity gate | Does a filter with no valid match return an empty result? | Must equal 1.0 |
| ANN and Filtered ANN Recall@1 | Secondary | Does the server preserve the single nearest result? | Higher |
| Replacement, Deletion, Persistence, and ANN-Index Correctness | Validity gate | Does the server preserve required state semantics and actually use the configured ANN index? | Must pass |

### Performance and resource measurements

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Unfiltered/Filtered Latency p50 | Diagnostic | What does a typical successful request take? | Lower |
| Unfiltered/Filtered Latency p95/p99 | Operational | How slow is the warm request tail under each concurrency? | Lower |
| Query Throughput | Operational | How many requests complete successfully per wall-clock second? | Higher |
| Request Error Rate | Operational | What share of submitted requests fail? | Lower |
| Build Time and Build Throughput | Operational | How long does initial indexing take and how many vectors are indexed per second? | Lower / Higher |
| Incremental Upsert/Delete Throughput | Operational | How quickly can the ready server apply each mutation workload? | Higher |
| Restart Readiness | Operational | How long until persisted state is queryable after restart? | Lower |
| First-Query Latency after Restart | Diagnostic | How slow is the first successful query after readiness? | Lower |
| Peak Server RAM and Persistent Storage | Operational | What memory and disk footprint does the prepared server require? | Lower at equal correctness and quality |

## Search and filter quality

### ANN Recall@K

**Question:** How faithfully does approximate search preserve exact nearest
neighbours?

For each query, NumPy ranks the complete frozen vector corpus by exact cosine
similarity. ANN Recall compares the server's returned IDs with the first `K`
oracle IDs. Order inside the returned set does not change this metric; the
question is whether the exact neighbours remain available. If the corpus has
fewer than `K` eligible records, the denominator is the number the oracle can
actually return. Duplicate returned IDs invalidate the request rather than
earning repeated credit.

A failed server request is recorded with zero recall and also increments Request
Error Rate, so failures cannot disappear through eligibility filtering. Recall
is reported independently at `K=1`, `3`, `5`, and `10`; the `@1` value is
secondary and the other three are primary.

**Range and direction:** `[0, 1]`; higher is better.

### Filtered ANN Recall@K

**Question:** Does approximate search remain faithful after metadata filtering?

The evaluator applies exactly the same frozen predicate to the oracle corpus and
the server request, then compares their IDs as above. Results remain separated
by filter-selectivity band. A query whose exact filtered result is empty is not
eligible for Filtered ANN Recall; it is evaluated by Empty-Filter Correctness.
A failed non-empty filtered request receives zero recall and increments Request
Error Rate.

**Range and direction:** `[0, 1]`; higher is better.

### Filter Correctness

**Question:** Does every returned record obey the full requested predicate?

A filtered request receives `1` only when every returned record satisfies every
part of the predicate, including conjunctions. It receives `0` when any returned
record violates a predicate or the request fails. Returning too few otherwise
valid records does not reduce this metric because Filtered ANN Recall already
measures missing neighbours.

The aggregate is the mean of the request-level pass values, reported separately
for each filter-selectivity band.

**Range and direction:** `[0, 1]`; higher is better and `1.0` is required for a
conformant server.

### Empty-Filter Correctness

**Question:** Does the server correctly return nothing when no record matches?

Each verified-empty predicate receives `1` only when the request succeeds and
returns no records. A non-empty response or request failure receives `0`. The
aggregate is the mean over verified-empty requests.

**Range and direction:** `[0, 1]`; `1.0` is required.

## Conformance validity gates

Replacement checks require an upserted ID to expose only its new vector and
metadata. Deletion checks require removed IDs and complete removed documents to
be absent from later search and filtering. Persistence checks require committed
records and the configured ANN index to remain available after restart. Health,
cosine behavior, wrong-dimension rejection, compound filters, and real ANN-index
use are binary gates under the same rule: every required check must pass.

A failed gate makes the server profile non-conformant. Performance measurements
may remain available for diagnosis, but the profile cannot be selected by
trading a correctness failure against speed.

## Performance and resources

Warm latency starts before client serialization and ends after the complete
loopback response is decoded. Only successful requests have a latency value;
failures remain visible through Request Error Rate. p50 describes the typical
request, while p95 and p99 describe the tail. p99 is reported only for workload
cells with enough successful measured requests and independent query support
to estimate it; otherwise it is null with an `insufficient_requests` status.
The support threshold is frozen after data review, before authoritative runs.

Query Throughput counts successful responses over the complete measured wall
time at each concurrency. Request Error Rate uses every submitted request as its
denominator. These two values are always read together: failed traffic cannot
make throughput look successful.

Build Time starts when a ready empty server receives the first vector and ends
when all submitted vectors are queryable. Build Throughput uses successfully
indexed vectors over that same elapsed time. Incremental upsert and delete
throughput are measured separately on a ready populated index and include the
time until each mutation is visible to queries.

Restart Readiness starts when restart is requested and ends when health checks
pass and the persisted ANN index answers its verification query. First-Query
Latency times the first successful query after readiness and is not mixed into
warm latency percentiles.

Peak server RAM is the largest sampled resident-memory total for server
processes or containers during the measured phase. Persistent Storage is the
on-disk server state after synchronization and before teardown. Client resource
measurements are labeled separately and are never added to server peaks.

## Eligibility, aggregation, and confidence intervals

Search-quality metrics are calculated once per frozen query and then averaged
within each workload cell. Filtered results are also averaged independently per
selectivity band. Query identities and conditions remain aligned across servers.
Development, validation, and locked quality intervals use 10,000 bootstrap resamples of complete
query IDs with seed 42. A resampled query carries all of its compared server
results and filter conditions.

Latency percentiles use successful request observations within one fixed
workload cell. Bootstrap draws resample complete query IDs, preserving their
dependent repeated requests and aligned observations across servers; repetitions
are not extra independent samples. Build, mutation, restart, resource, and storage
values are observed phase-level measurements and receive no fabricated
confidence interval. Every table reports submitted, successful, failed, and
eligible request counts. Null means a defined eligibility condition was not met;
it never means zero.
