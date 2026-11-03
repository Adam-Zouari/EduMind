# Vector database methodology

[Shared methodology](../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../running.md)

On this page:

- [Purpose and scope](#purpose-and-scope)
- [Candidates and configurations](#candidates-and-configurations)
- [Data](#data)
- [Metrics and their purpose](#metrics-and-their-purpose)
- [Evaluation setup](#evaluation-setup)
- [Benchmark workflow and selection](#benchmark-workflow-and-selection)
- [Execution and measurements](#execution-and-measurements)
- [Results and MLflow structure](#results-and-mlflow-structure)

Which database and configuration preserve vector-search accuracy and useful
educational evidence while providing suitable search, document-maintenance,
startup, memory, and storage performance?

## Purpose and scope

One candidate is a **database–configuration pair**. The database identity alone
does not determine search accuracy, build cost, or memory use. One child run
records the exact configuration and all declared workload conditions it executes.

The benchmark has two measurement paths within the same comparison:

- **Database search:** prepared query vectors go directly to the server. Exact
  neighbour preservation, filtering, request latency, and load handling isolate
  database behaviour from model inference.
- **Full-stack retrieval:** questions pass through the frozen query embedding,
  retrieval strategy, and selected reranker to the final chunks. This checks
  application impact while varying only the database–configuration pair. It
  does not generate answers or use an LLM judge.

A **workload condition** identifies its corpus snapshot, query inventory,
filter, request payload, execution mode, and concurrency. Conditions have separate
results; an unfiltered result does not replace a filtered measurement.
Concurrency means outstanding client requests, not a database's advertised
worker limit or model batch size. Each search request contains one query vector.

One document insertion, replacement, or deletion is one **logical operation**,
even when its adapter needs several API batches. A document with 40 chunks is
not counted as 40 insert operations. Chunk counts and payload bytes explain
the work in the operation artifact.

### Implementation status

This page defines the revised benchmark contract. The existing vector drivers
have not yet been migrated to it. Their synthetic workloads, configuration
tuning, metric keys, agreement checks, and separate retrieval-validation entry
point must not be presented as execution of this revision. In particular, the
current driver does not implement the complete frozen-configuration validation
and locked workflow, tie-aware @20 scoring, or the reporting structure below.
The standalone validated-input/report interface also remains planned for this
suite. [The runbook](../running.md#9-vector-databases) distinguishes existing
diagnostic commands from the target lifecycle.

Reviewed data, configuration ranges, load conditions, and independent-support
requirements remain in [pending data review](../pending-data-review.md#vector-databases).
Historical runs retain their original protocols; they are not relabeled with
the revised metrics. A successful old driver exit does not establish conformance
with this contract.

## Candidates and configurations

| Database | Why it is included |
|---|---|
| Chroma server | Provisional application baseline. |
| Qdrant server | Purpose-built vector server with payload filtering. |
| Weaviate | Independent vector-server architecture with structured filters. |
| PostgreSQL + pgvector | Relational alternative with SQL metadata and vector indexes. |

Pinned product identities are in [model selection](../model-selection.md#vector-databases).
Image digests and deployment endpoints remain in deployment locks; benchmark
behaviour and the configuration roster belong in the versioned vector
`protocol.yaml`. Each configuration has a stable alias and resolved settings.
No parameter changes are hidden behind the same alias/checksum.

Development evaluates a small, declared configuration set per database. Search
breadth, graph connectivity, construction effort, and filter-planning settings
are useful dimensions when supported by the pinned backend. Freeze ranges and
tuning budgets before comparative runs; identical numeric settings across
implementations do not establish equivalent effort or accuracy. Unsupported
settings are distinguished from execution/infrastructure errors.

Each tested pair receives its own child, not a hidden trial inside a database
winner run. Examine recall, latency, QPS, build cost, and resources together.
Compare speed at reasonably comparable recall instead of rewarding configurations
that search less thoroughly. A recall target, if used to organize review, must
be declared in advance; there is no implicit 0.99 gate or automatic winner.

Native planning may select an exact scan for a small eligible set or another
documented path within the frozen configuration. This is a database comparison,
not a requirement that every query traverse HNSW. Record index readiness and
executed plans where observable. Apply this permission consistently to all
backends; an automatic plan change is not engineer retuning on held-out results.

## Data

Representative real chunks and the selected embedding model's vectors are the
core workload. Freeze a realistic initial corpus, plausible growth conditions,
held-out queries, and their source provenance. Synthetic stress conditions remain
separately labelled; their dimensions or easy self-neighbour queries do not
substitute for application-representative measurements.

| Profile | Data role |
|---|---|
| Smoke | Small fixtures exercising search, filters, writes, restart, and reporting. |
| Development | Reviewed representative and stress workloads for configuration comparison. |
| Validation | Unseen queries/source allocations and reviewed corpus conditions for frozen finalists. |
| Locked | Held-out workload for one selected pair; reporting only. |

Exact counts, growth sizes, synthetic seeds, query schedules, concurrency,
load-window lengths/repeats, and shortlist limits are frozen after data review.
They are not inherited automatically from the old driver's synthetic grid.
Source-related questions remain grouped across splits and uncertainty draws.
All candidates within a condition receive the same corpus and query inventory.

Filters refer to known metadata, such as a user-selected document's stable
internal ID. They do not infer a subject from an ambiguous filename. Freeze
eligible IDs/counts from trusted input metadata, including restrictive, compound,
and empty-match cases. Review both selectivity and its relationship to vector
clusters/query intent; equal matching percentages can create different workloads.

Answerable integration questions have reviewed gold evidence in their allowed
source scope. Shared evidence IDs/intervals and the `tiktoken:cl100k_base` scoring
tokenizer are unchanged. Unanswerable questions are not empty-reference retrieval
comparisons. Gold units outside a requested document filter indicate an invalid
question/filter pairing, not failure to recover forbidden evidence.

Full [data validation](../data-validation.md) runs separately during preparation.
It verifies corpus/query shapes, finite nonzero vectors, identities/checksums,
metadata, reference distances, evidence and split isolation. Each profile verifies
its inputs against that report before resource/timing windows. Database health,
generated output validity, index readiness, and write visibility remain runtime
checks, not claims made by the data validator.

## Metrics and their purpose

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | ANN Recall@20; Filtered ANN Recall@20 | Measures exact-neighbour preservation in each search condition. |
| Secondary | nDCG, Evidence-unit Recall, Evidence-token Precision at @3/@5 | Checks the frozen stack's useful evidence and ordering. |
| Diagnostic | Candidate-pool Evidence-unit Recall@20; Ranking Repeatability; per-operation Request Error Rates; Filter Violation Rate | Explains available evidence, database output stability, execution failures, and forbidden returns. |
| Operational | Database Search and Full-Stack Retrieval Latency p50/p95; QPS; Initial Build Time; Insert/Update/Delete Latency p50/p95; Startup Time; Peak Server RAM | Separates search/load, application wait, maintenance, readiness and memory costs. |
| Descriptor | Persistent Storage Size | Records persistent footprint at a comparable checkpoint. |

The [metric reference](metrics.md) owns questions, calculations, examples,
directions and genuine edge cases. Probe outcomes and workload inventories remain
validation evidence/parameters, not additional quality metrics.

## Evaluation setup

### Frozen upstream stack and exact reference

All candidates receive identical chunk IDs/text, vectors, metadata, similarity
definition, storage precision, query inputs, and equivalent returned payloads.
The server never generates embeddings. IDs-only and text-bearing responses are
different workload conditions, not interchangeable latency observations.

The full-stack path freezes the selected chunker, embedding, retriever, reranker,
top-20 pool depth, and @3/@5 scoring. Dense search uses the tested database.
For RRF, retain the same external BM25 implementation, source depth, and fusion
settings; do not substitute each server's native hybrid algorithm. A BM25-only
selected stack has no dense-search dependency: label that integrated path and
do not attribute identical BM25 results to ANN performance.

Compute exhaustive reference distances offline over the frozen vectors. Filtered
references use only the allowed subset. ANN Recall@20 uses the
[boundary-tie rule](metrics.md#exact-neighbours-and-boundary-ties): strictly closer
records receive individual credit; boundary alternatives receive credit only
up to the remaining slots. The common evaluator, not server-reported scores,
determines these classes. Exact equality is the default. A nonzero tolerance
requires numerical validation and a frozen protocol revision, never tuning
against candidate scores.

### Pool identity and integrity

Record each executed pool with query, attempt, condition, corpus/vector/metadata
checksums, configuration identity, ordered IDs, and scores. Its checksum verifies
that observation and provenance; it does not require different databases or
repeated searches to return identical pools.

The reranker receives the actual verified pool for that attempt and must return
exactly its IDs once each. Final @3/@5 results are ordered prefixes of that
validated ranking. The exact-retrieval benchmark's live-pool-equality gate is
not copied into this ANN benchmark. No reranked-output repeatability metric is
added here; only the database's pre-reranking output is compared across attempts.

Duplicate/unknown IDs, nonfinite required scores, forbidden records, and malformed
responses retain explicit integrity outcomes. Check filter eligibility against
trusted input metadata, not merely the values echoed by the server. A completed
usable response with known forbidden records supplies a Filter Violation Rate
event and retains its execution-success timing/QPS observation, but cannot
supply accepted ANN or downstream evidence quality. Reference eligibility is
established first: zero permitted records means no filtered-recall task, so its
status stays inapplicable regardless of the response. With permitted records,
a filter-violating result instead leaves recall unavailable. The metric reference
defines the [empty-subset outcomes](metrics.md#filtered-ann-recall20).
Malformed, duplicate-ID, unknown-ID and nonfinite-score responses are execution
errors instead. The
[search-outcome table](metrics.md#scoring-inputs-eligibility-and-result-status) fixes these
cohorts; operational observations do not excuse a failed correctness requirement.

A well-formed eligible response may contain fewer than the requested neighbours.
Preserve its ANN score, with credit only for recovered neighbours; do not pad or
silently refill it. Integration expects `min(20, eligible chunk count)` records
in each required source pool. An unexpectedly short pool leaves dependent
integration quality unavailable while retaining the database observation. A
truly smaller eligible corpus is not a short-pool failure. Unexpectedly empty or
token-free integration rankings follow shared evidence-scoring integrity rules.
Empty-filter probes are separate tasks, not answerable integration questions.

### Correctness probes

Use small controlled probes for vector/similarity semantics, wrong-dimension
rejection, compound and empty filters, unique-ID replacement, complete document
replacement, deletion, persistence, and configured-index readiness. Verify stored
identities/contents, not just counts. An unrelated connection error does not
prove correct rejection of a wrong dimension.

Store pass/fail/unverified outcomes and details in the validation artifact.
They are integrity evidence, not another family of quality percentages.
Do not add a perfect Ranking Repeatability or zero Request Error Rate selection
gate. Required correctness and verification remain necessary for trustworthy
evidence; performance trade-offs cannot excuse invalid results.

## Benchmark workflow and selection

| Profile | Candidate scope | Purpose and output |
|---|---|---|
| Smoke | Declared small fixtures for database paths | Wiring/conformance checks; no selection or authoritative CI. |
| Development | All declared database–configuration pairs | Compare the search space and review trade-offs; write the finalist decision. |
| Validation | Selected development pairs with exactly their recorded settings | Rebuild on the validation corpus without retuning; write one winner decision. |
| Locked | Exactly one selected validation pair | Report on held-out inputs without configuration search or reselection. |

Both database and applicable full-stack evidence belong to the pair's comparison.
Early server smoke/stress preparation may precede upstream selection; authoritative
application-impact measurements require the frozen stack. Changed upstream choices
require a new identified comparison, not mixed results. The two execution paths
are not independent ways to select conflicting winners. The database-only suite
has no CUDA smoke or GPU preflight; integration models retain their upstream checks.

Engineer decisions live in `data/benchmarks/decisions/`.
`vector-database-validation.json` references complete development candidate IDs,
exact configurations and source runs; `vector-database-locked.json` references
exactly one complete validation pair. Both retain reviewer, date, rationale, and
input/protocol provenance under the shared decision contract. Final RAG consumes
the same selected database configuration without new selection.

### Result completeness and advancement

Apply [shared execution, quality and integrity accounting](../metrics.md#execution-accounting-and-result-completeness).
Retain every scheduled outcome, including not-executed work. A failed first
request leaves applicable dependent quality unavailable; independent timings
and resources survive. Later failures do not erase completed first quality.
Ordinary ANN misses are scored search outcomes, not request errors.

Submitted-operation error rates and completed-response filter violations have
different denominators. An execution failure is not a checkable filter response.
Missing checks mean unverified, not pass. Partial required quality remains a
labelled diagnostic, not a complete comparison obtained by dropping failed queries.
Conditional successful latency still retains its failure coverage.

The engineer reviews complete applicable evidence and records finalists; no
weighted score, automatic winner, or extra zero-request-error rule is implied.
Protocol violations and missing required evidence block authoritative use of
affected results. No runner promotes application settings automatically.

## Execution and measurements

Run candidate servers sequentially with controlled CPU/RAM allocations, client
resources, connection reuse, request payloads, and pinned software. Unrelated
database workloads must not consume the same resources. Database server
measurements are CPU-only; full-stack models retain their frozen upstream
CUDA/dtype/batch-one settings without becoming database GPU candidates.

```text
verify inputs and deployment identities outside measurement
-> run isolated correctness probes and prepare a clean benchmark state
-> start server lifecycle monitoring
-> time initial ingestion and configured-index readiness
-> record the synchronized post-build storage checkpoint
-> run one representative query/operation or load-window warmup for each path
-> execute controlled query attempts and declared server-load windows
-> execute document operations with visibility checks and restored baselines
-> stop the database; time startup against the preserved workload state
-> stop monitoring; retain all outcomes and samples
-> score designated first outputs and all-pairs database Ranking Repeatability
-> aggregate and bootstrap offline
```

### Query attempts and load windows

Smoke has one measured attempt per controlled query. Development, validation,
and locked schedule three with frozen inputs/settings. The first supplies
quality; later outputs never replace it. Database Ranking Repeatability uses
all three unordered pairs before reranking. Valid alternative pools are not
failures. Repeated queries on one built index do not establish reproducibility
across independently rebuilt indexes.

Server load is a separately labelled workload mode within the same child. Freeze
its query schedule, concurrency, completion/timeout policy and independent window
repeats. Three repeatability attempts alone do not define its duration or support.
Mark the first scheduled occurrence of each query in a condition before execution;
only that occurrence supplies the condition's ANN quality, not the best or first
successful response. Where repeatability is claimed, identify its three scheduled
observations per query in advance. Additional traffic contributes to load
statistics, not extra quality samples. Retain each window's outputs and condition
identity; serial recall cannot substitute for a load-specific quality observation.

Use one warmup appropriate to each execution path: a representative complete
query/operation for controlled evaluation, or one untimed warmup window for a
server-load condition. The load warmup's duration and warm-state criterion are
frozen during workload review. Warmup never enters measured quality, repeatability,
error rates, latency, or QPS. Restore equal state after a mutation-path warmup.
Warmup is not configuration tuning; a startup measurement has no preceding warm
query against the stopped server.

Database Search Latency begins at the common client adapter operation and ends
at complete response decoding, including serialization, communication and server
work. Scheduling wait before that boundary is outside request latency but inside
the enclosing load window. Keep connection limits, timeout handling, retry policy,
and payloads explicit. Do not retry a failed measured operation and relabel its
successor as the designated first success. Deliberately enabled internal retries
remain inside the same logical operation and timer.

QPS divides successful measured completions by the complete load-window wall
time, including dispatch, queuing, failures, and draining submitted requests to
completion or fixed timeout. Setup, warmup, oracle work, scoring and bootstrap
are outside. A fixed-concurrency result is not a discovered universal capacity
limit. Do not sum concurrent request durations as the QPS denominator.

Full-Stack Retrieval Latency starts with the question request and ends with
validated final chunks, including preparation/embedding, required source searches,
fusion, reranking and synchronization. Model loading and corpus preparation are
outside this warm timer. Each question contributes its median successful measured
duration; p50/p95 then summarize questions. Server-load percentiles instead
summarize individual completed requests. Keep the different observations labelled.

### Document operations, build and startup

Initial Build Time starts from a running empty server at collection/schema/index
creation and prepared-data ingestion, and ends when all records are queryable and
configured indexes are ready. Include required metadata indexes and first-use
build work. Exclude downloads, extraction, chunking, embedding, oracle preparation,
and isolated conformance fixtures.

| Logical operation | Timed start | Verified finish |
|---|---|---|
| Insert | First submission of the prepared document records | Every intended record is available through the declared visibility check. |
| Update | First operation replacing an existing document | New vectors/metadata are available and obsolete chunks are absent. |
| Delete | First submission deleting the document | Its records are confirmed absent. |

Use the same original states and prepared replacements for every pair. Include
adapter batching and the visibility barrier; acknowledgment is not completion.
Verification establishes availability/absence, not an expectation that ANN will
retrieve every inserted record for an arbitrary query. Freeze verification queries
and polling intervals. Restore baseline state outside the timed logical operation;
retain restoration failures instead of continuing on a different corpus. Static
search, mutation, and any declared mixed-load conditions remain separate. Logical
deletion need not reclaim disk blocks immediately.

Startup Time measures a start command for an already stopped server holding the
preserved benchmark corpus through health, state/index verification, and a
successful required query. Stopping occurs before the timer. Do not substitute
an empty server or the small conformance corpus. This is not an OS-cache-cold
guarantee; freeze and record the restart/cache procedure.

### Resources and storage

Peak Server RAM follows [shared resource rules](../metrics.md#resource-measurements)
within the build/warmup/query/operation/restart lifecycle. Sample the server
container or declared container group with one consistent Docker/cgroup method;
record cache accounting and do not alternate raw API usage with cache-subtracted
CLI values. Sum simultaneous components, not independent maxima. Client model
memory is outside this metric. Stop/restart periods and telemetry gaps remain
explicit; unrelated services and offline scoring are excluded.

Persistent Storage Size is the apparent byte size of dedicated database data,
indexes, metadata, and required logs at the synchronized completed-build checkpoint,
before warm query/mutation workloads. Use identical measurement semantics, not
an apparent-size/allocated-block fallback. Exclude image layers, model snapshots
and benchmark artifacts. Later/partial footprints retain their checkpoint labels
but do not replace the completed-build scalar.

## Results and MLflow structure

Use `EduMind / Vector Database`, one parent per comparison and flat children per
database–configuration pair. Names omit dates and timestamps. These aliases
illustrate structure, not a frozen roster:

```text
EduMind / Vector Database
├── smoke
│   ├── chroma-config-01
│   └── ...
├── development
│   ├── chroma-config-01
│   ├── chroma-config-02
│   ├── qdrant-config-01
│   ├── weaviate-config-01
│   └── pgvector-config-01
├── validation
│   └── <selected development database–configuration pairs>
└── locked
    └── <one selected validation pair>
```

The parent retains plan, protocol source/resolved checksum, deployment lock,
input-validation identity, manifests, upstream decisions, completion accounting
and comparison tables. Children record configuration/profile, corpus/query/vector/
metadata checksums, resource allocations, payload and source-run provenance.
Query/attempt/window observations are artifact rows, not nested runs.

Keep workload conditions separate in scalar names. The following base keys use
`.condition.<condition_id>` for query/load/mutation results. The stable condition
ID resolves to its full specification in `workloads.json`. Build/startup/storage
facts instead use `.corpus.<corpus_id>` to identify their lifecycle; a whole-child
memory peak explicitly identifies that scope rather than claiming per-cell peaks.

```text
quality.ann_recall_at_20
quality.filtered_ann_recall_at_20
quality.ndcg_at_3                         # also @5
quality.evidence_unit_recall_at_3         # also @5
quality.evidence_token_precision_at_3     # also @5
diagnostic.candidate_pool_evidence_unit_recall_at_20
diagnostic.ranking_repeatability
diagnostic.request_error_rate.search
diagnostic.request_error_rate.insert
diagnostic.request_error_rate.update
diagnostic.request_error_rate.delete
diagnostic.filter_violation_rate
operational.database_search_latency_ms_p50       # also p95
operational.full_stack_retrieval_latency_ms_p50   # also p95
operational.qps
operational.initial_build_time_seconds
operational.insert_latency_ms_p50                # also p95
operational.update_latency_ms_p50                # also p95
operational.delete_latency_ms_p50                # also p95
operational.startup_time_seconds
operational.peak_server_ram_bytes
descriptor.persistent_storage_size_bytes
```

For example, `operational.database_search_latency_ms_p95.condition.dev-doc17-c4`
is not pooled with `...condition.dev-unfiltered-c1`. Matched condition IDs denote
matched observations across candidates. Metric contracts/status artifacts record
roles and observation units. Shared sample-count and supported-CI suffixes append
to each fully scoped key.

Planned workload counts are parameters. Actual support counters may use
`validity.<operation>.submitted_count`, `.successful_count`, `.failed_count`, and
`.not_executed_count`, scoped by condition. They reconcile to the ledger; they
are not a global error-rate metric or quantities to optimize. Integrity,
filter-checkability, quality and independent-source support stay explicit in
artifacts and metric-specific support fields.

| Artifact | Contents |
|---|---|
| `workloads.json` | Conditions, shapes, query/source IDs, filters/eligible counts, schedules, operation sizes, and checksums. |
| `candidate.json` | Resolved configuration, upstream identities, resource scope, lifecycle boundaries, and readiness status. |
| `requests.parquet` | Scheduled/submitted/completed/failed/not-executed outcomes, logical operation/query/attempt/window IDs, durations and errors. |
| `rankings.parquet` | Actual ordered database pools and downstream rankings, IDs/scores, provenance, expected/actual counts. |
| `query_metrics.parquet` | First-output ANN/evidence values, statuses, tie cutoff/slot/credit details, and source support. |
| `repeatability.parquet` | Three scheduled database-ranking pair comparisons per query when applicable. |
| `timings.parquet` | Build/startup/operation/full-stack boundaries, load-window wall time, and elapsed-to-failure. |
| `resources.parquet` | Timestamped server samples, monitoring coverage and storage checkpoints. |
| `validation_report.json` | Runtime conformance, visibility, provenance, permutation, filter and output-integrity evidence; separate from the input-validation seal. |
| `metric_statuses.json` | Scored/inapplicable/unavailable/incomplete statuses, units, counts, CI support, and reasons. |

Parent comparison tables preserve conditions and upstream identities. Paired
differences use aligned queries/source groups or independent load windows as
defined in [confidence intervals](metrics.md#aggregation-and-confidence-intervals).
Counts, configurations and descriptive storage facts do not acquire fabricated
winner directions or CIs. This reporting contract does not certify old artifacts.
