# Vector databases methodology

[Shared methodology](../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../running.md)

On this page:

- [Servers](#servers)
- [Data and configurations](#data-and-configurations)
- [Execution](#execution)
- [Metrics and why they are used](#metrics-and-why-they-are-used)

Which networked vector server preserves nearest-neighbour and filter correctness
while providing the most useful latency, concurrency, ingestion, memory, and
storage trade-off?

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
| Locked | One selected server and frozen index settings on a held-out vector/query workload fixed before execution; reporting only |

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

A validation finalist is a server together with its exact development-tested
index settings. Validation rebuilds those indexes on the validation corpus and
compares only the recorded finalists without changing their settings or
reopening the grid search.

## Execution

The experiment has three steps.

### A. Conformance

The real server is checked for health, cosine behavior, wrong-dimension
rejection, compound filters, empty filters, duplicate-ID replacement, complete
document replacement, deletion, real ANN-index use, persistence after restart,
and index availability after restart. These are validity checks. A server that
fails one is reported as non-conformant rather than assigned a misleading
performance rank.

### B. Dense ANN performance

```text
NumPy computes exact top neighbours
→ server returns approximate neighbours
→ compare returned IDs with the exact IDs
→ repeat unfiltered and filtered queries
→ repeat at each configured concurrency
```

### C. Real retrieval

`vector-database-validation.json` selects one or more complete server/index
finalists from the completed development comparison, preserving their exact
tested settings. During validation, every finalist stores the same real chunks
and vectors and the complete selected retrieval strategy is rerun so database
ANN behavior is connected to actual RAG quality. Only after all validation
evidence is reviewed does `vector-database-locked.json` record the single
server/index profile approved for its own locked report and for Final RAG.
Locked execution evaluates only that server with its frozen index settings and
workload. It repeats the applicable conformance and performance checks without
reopening the HNSW search or server selection. Exact held-out workload sizes and
source provenance must be reviewed and recorded before the locked run.

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
