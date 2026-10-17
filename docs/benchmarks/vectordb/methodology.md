# Vector database methodology

[Shared methodology](../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../running.md)

On this page:

- [Servers](#servers)
- [Data and configurations](#data-and-configurations)
- [Execution](#execution)
- [Metrics and why they are used](#metrics-and-why-they-are-used)
- [Implementation status](#implementation-status)

Which networked vector server preserves nearest-neighbour and filter correctness
while providing the most useful latency, concurrency, ingestion, memory, and
storage trade-off?

This page defines the intended comparison and identifies the current execution
gaps below. Reviewed workloads, completed comparative runs, and a final server
decision are not claimed. The [metric reference](metrics.md) owns calculations,
worked examples, result statuses, support, and confidence intervals.

## Servers

| Server | Why it is included |
|---|---|
| Chroma server | Current provisional production baseline. |
| Qdrant server | Purpose-built vector server with payload filtering. |
| Weaviate | Independent vector-server architecture with structured filters. |
| PostgreSQL + pgvector | Transactional relational alternative with JSON metadata and HNSW. |

All servers receive identical precomputed vectors and metadata. They never create
embeddings internally.

## Data and configurations

The [shared data-validation workflow](../data-validation.md) checks synthetic
and selected-real workload shapes, finite vectors, metadata/filter cases,
reference inputs, seeds, and provenance before timed execution. Its proposed
standalone/report interface remains pending implementation. Database health,
index readiness, and adapter conformance are separate runtime/environment gates.

| Execution profile | Workload |
|---|---|
| Smoke | 1,000 vectors at dimension 384; 50 queries; concurrency 1 |
| Development | 100,000 vectors at dimensions 384 and 1,024; 500 queries; concurrency 1/8/32 |
| Validation | Selected real embeddings plus 1,000,000 clustered vectors; up to 1,000 queries; concurrency 1/8/32/64 |
| Locked (planned; not executable yet) | One selected server and frozen index settings on a held-out vector/query workload fixed before execution; reporting only |

Synthetic vectors contain clusters and 5% near-duplicates. Metadata creates
filters matching approximately 50%, 10%, 1%, and in validation 0.1% of records.

Development searches the supported HNSW combinations:

```text
m:                     16 or 32
construction breadth:  100 or 200
search breadth:         64 or 128
```

This makes sure one server is not compared with an unnecessarily weak default.
Unsupported settings are recorded rather than silently replaced.
Every supported configuration remains visible in MLflow; the runner does not
automatically choose the database winner.

The grid has eight declared combinations per server/dimension. Current tuning
evaluates up to 10,000 corpus vectors and 50 queries, records supported and
unsupported trials, and uses Recall@10 target `0.99`. Among trials meeting that
target it selects the lowest p95, breaking ties by `m`, search breadth, then
construction breadth. If none meets it, it retains the highest-recall supported
trial, with latency as the next criterion. This selects a configuration to
measure, not the winning server. Trials are recorded as values under the
server child run, not as eight independent finalist children.

The protocol uses Qdrant's minimum supported full-scan threshold, 10 KiB; the
pinned server rejects zero. Index readiness is verified before measurement.
Small filtered candidate sets may still use the server's exact-search path;
their measured recall and latency describe that server behavior.

A validation finalist is a server together with its exact development-tested
index settings. Validation rebuilds those indexes on the validation corpus and
compares only the recorded finalists without changing their settings or
reopening the grid search.

That frozen-settings validation rule is the contract, not a claim that the
current dense-ANN driver enforces it: its validation path still retunes.
At most three server finalists advance. Do not treat a newly tuned validation
configuration as the same frozen development-tested profile.

## Execution

The experiment has three steps.

### A. Conformance

The real server is checked for health, cosine behavior, wrong-dimension
rejection, compound filters, empty filters, duplicate-ID replacement, complete
document replacement, deletion, real ANN-index use, persistence after restart,
and index availability after restart. These are validity checks. A server that
fails one is reported as non-conformant rather than assigned a misleading
performance rank.

The small conformance corpus has 40 records. It is a controlled semantic probe,
not representative retrieval-quality data or a full-scale restart measurement.
Record each check and its reason; a missing check is incomplete. Current probe
coverage and gate-enforcement limitations are listed in the
[metric reference](metrics.md#current-measurement-limitations).

### B. Dense ANN performance

```text
NumPy computes exact top neighbours
→ server returns approximate neighbours
→ compare returned IDs with the exact IDs
→ repeat unfiltered and filtered queries
→ repeat at each configured concurrency
```

All servers receive the same normalized vectors, metadata, exact-oracle IDs,
query order, and conditions. Search-quality scores use the designated first
request; do not average later rankings or choose a later successful output.
A failed first quality request is unavailable, not a fabricated empty response
or zero-recall observation. A valid empty response against nonempty exact
neighbours really does score zero. Required filter and conformance checks keep
their separate failure rules.

The current protocol specifies two load warmups, one measured repetition for
smoke, and three for development/validation. In the current driver, serial
quality and filtered searches run once per query before load warmups; the
repeated concurrent workload is unfiltered. Thus the
diagram describes the intended condition coverage, not proof that filtered
load cells are already implemented. Synthetic ANN does not inherit extraction's
pairwise repeatability metric. Real retrieval retains its existing ranking
agreement gate and first-ranking quality rule.

Report each workload/dimension/K/filter/concurrency cell with actual coverage,
not only a pooled server mean. Warm successful-request latency excludes build,
warmup, and oracle scoring; batch wall time includes failed requests and
dispatch/queue overhead. Time to failure is not successful-response latency.
Keep earlier valid observations if a later phase fails, label partial diagnostic
scope, and mark missing scheduled execution incomplete. The current driver's
partial-retention gaps must not be mistaken for an intentional discard policy.

### C. Real retrieval

`vector-database-validation.json` selects one or more complete server/index
finalists from the completed development comparison, preserving their exact
tested settings. During validation, every finalist stores the same real chunks
and vectors and the complete selected retrieval strategy is rerun so database
ANN behavior is connected to actual RAG quality. Only after all validation
evidence is reviewed does `vector-database-locked.json` record the single
server/index profile approved for its own locked report and for Final RAG.
Planned locked execution evaluates only that server with its frozen index
settings and workload. It repeats the applicable conformance and performance
checks without reopening the HNSW search or server selection. Exact held-out
workload sizes and source provenance must be reviewed and recorded before the
locked run.

The complete-retrieval path freezes chunking, embedding, first-stage retrieval,
reranking, evidence annotations, and score cutoffs while varying only the server
and its recorded index profile. It uses three measured rankings per question;
the first supplies quality, later rankings supply agreement/timing. Questions
must remain grouped by source document for document-macro quality and uncertainty.
This is evidence retrieval, not generated-answer evaluation, and uses no LLM
judge. Current scoring and aggregation limitations are explicit in the metric
reference.

## Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Validity gate | Health, cosine behavior, dimension rejection, Filter Correctness, Empty-Filter Correctness, replacement, deletion, persistence, restart, and ANN-index verification | Determines whether results are trustworthy; these are not quality scores. |
| Primary | ANN Recall@3/@5/@10, Filtered ANN Recall@3/@5/@10 | Measures preservation of exact neighbours at application-relevant depths, with and without metadata filters. |
| Secondary | ANN and Filtered ANN Recall@1; real-retrieval nDCG, Evidence-unit Recall, and Evidence-token Precision at @3/@5 | Shows rank-one behavior and whether ANN results preserve real evidence retrieval. |
| Diagnostic | Unfiltered/filtered p50 latency, first query after restart | Explains typical and cold-query behavior. |
| Operational | Unfiltered/filtered p95/p99, throughput and error rate at each concurrency, build time and vectors/second, incremental upsert/delete throughput, restart readiness, peak server RAM, persistent storage | Measures tail latency, load handling, ingestion, restart, memory, and disk cost. |

The database report remains separate evidence. It shows which server should be
used by the final benchmark, but the benchmark never changes the current Chroma
production default automatically.

## Implementation status

Implemented entry points are dense ANN `smoke`, `development`, and `validation`,
plus the separate complete-retrieval `validation` entry point. Pinned image
verification, real network adapters, NumPy exact search, controlled conformance
probes, tuning-trial values, load measurements, server memory/storage collection,
and MLflow artifacts exist. The server-only workload is CPU-based; GPU-backed
embedding/reranking preparation in real retrieval is a separate model workload.
CPU VRAM is null/inapplicable, not zero.

The following are not yet implemented as the complete contract:

- Dense-ANN validation reruns the HNSW grid instead of enforcing the finalists'
  exact development-tested settings. Complete retrieval does read selected
  settings from the development source summary, so these two paths must not be
  assumed to use the same newly tuned validation profile.
- The vector protocol and CLI have no `locked` profile. A frozen winner decision
  is required before that reporting-only stage, but a decision file does not
  make a missing execution path available.
- The standalone validated-input/report interface is still planned for vector
  workloads; existing inline checks are not a sealed validation report.
- Filtered concurrency coverage, successful-throughput accounting on errors,
  latency support gates/cluster intervals, document-macro real-retrieval
  aggregation, complete per-request status/partial retention, uniform mutation
  visibility timing, and exhaustive gate enforcement need the qualifications
  listed in [metrics](metrics.md#current-measurement-limitations).

Current output is diagnostic where these requirements are unmet. A `success`
status or populated MLflow chart alone does not establish complete conformance
or an authoritative comparison. These gaps must be resolved before claiming the
corresponding frozen validation or locked evidence. The reporting contract keeps
the existing attempt counts, numeric settings, candidate roster, and gate scopes.
