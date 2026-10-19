# Retrieval and reranking methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Implementation boundary](#implementation-boundary)
- [Candidate matrix](#candidate-matrix)
- [Frozen candidate pools and execution](#frozen-candidate-pools-and-execution)
- [Execution profiles and selection](#execution-profiles-and-selection)
- [Metrics and why they are used](#metrics-and-why-they-are-used)
- [MLflow result structure](#mlflow-result-structure)
- [Paired comparisons and advancement](#paired-comparisons-and-advancement)

Which complete first-stage retriever and reranker gives the best evidence
ordering for the selected chunker/embedding pair, and is any quality improvement
worth its latency and resource cost?

## Implementation boundary

This page and the metric reference define the approved reporting contract.
Retrieval–reranking implementation is being updated separately. Current
runner/protocol outputs still differ from this contract:

- Authoritative profiles specify two warmups rather than one, and alpha-nDCG
  eligibility filters out single-unit questions.
- The first-anchor Ranking Agreement validity gate must become the all-pairs
  diagnostic Ranking Repeatability. Pool integrity remains enforced, but its
  redundant `validity.pool_collection_match` scalar is not required reporting.
- Existing latency/storage/token keys and mandatory workload counters need the
  reporting layout below. Snapshot size belongs to preparation metadata, not
  the existing per-run reranker-snapshot-size metric.
- RRF may expose a combined dependency-build time and constant-zero incremental
  index bytes. Neither is a third index measurement; retain independent Dense
  and BM25 build observations and report their storage dependency breakdown.
- Expanded metric-specific status/support rules and small-corpus behavior need
  verification. The standalone validator/report-reuse interface remains planned.

These are implementation limitations, not alternative metric definitions or
permission to silently relabel old results.

## Candidate matrix

The selected chunker/embedding pair, corpus, questions, and evidence annotations
are frozen before this phase. A candidate is one complete
`<retriever>|<reranker>` stack. Every first-stage retriever is crossed with every
reranker option, declaring `3 × 5 = 15` stacks. Preflight tests this complete
roster; development evaluates exactly its qualified subset and retains the
hardware exclusions in provenance.

| First-stage retriever | What it tests |
|---|---|
| `dense` | Exact cosine ranking from the selected embedding. |
| `bm25` | Lexical ranking for exact terminology, names, identifiers, and codes. |
| `rrf` | Reciprocal-rank fusion of the Dense and BM25 top-20 lists without mixing their raw scores. |

| Reranker | What it tests |
|---|---|
| `none` | The first-stage order without learned reranking. |
| `gte-modernbert` | Weaker long-context GTE ModernBERT cross-encoder control. |
| `ettin-150m` | Compact learned reranker. |
| `ettin-400m` | Mid-size learned reranker. |
| `ettin-1b` | Larger learned reranker. |

The no-reranker rows are full candidates, not administrative baselines. Dense
and BM25 remain independent when selected directly. RRF reads their ranked
top-20 source lists, combines them using the frozen fusion settings, and returns
one deterministic top-20 list.

## Frozen candidate pools and execution

Each profile uses its own reviewed manifest and source split under the selected
chunker/embedding contract, with canonical chunks, evidence-unit IDs, and source
intervals. New validation and locked questions receive newly prepared pools for
their own corpus/index inputs; they do not reuse development question pools.
Within one profile, all compared stacks share its frozen corpus and annotations.

The [shared data-validation workflow](../../data-validation.md) checks those
prepared inputs and source isolation before execution; its standalone/report
interface is planned. Generated pool-collection, chunk-content, and index
checksums are separate runtime integrity gates, not additional data annotations.

The retrieval/reranking smoke fixture contains 30 frozen canonical chunks. This
leaves ten chunks outside each top-20 pool, so smoke execution can exercise pool
selection, exclusion, fusion, and permutation checks. The count is a smoke
fixture control; authoritative chunking strategies are not forced to produce
exactly 30 chunks because their natural chunk counts are benchmark outputs.

Quality comparisons use one checksummed top-20 pool per question and first-stage
retriever. During a full development comparison, the no-reranker candidates own
their corresponding collections:

| Pool owner | Shared by |
|---|---|
| `dense\|none` | All five Dense candidates. |
| `bm25\|none` | All five BM25 candidates. |
| `rrf\|none` | All five RRF candidates. |

A reranker receives exactly the corresponding frozen query pool for every
question. Its expected size is `min(20, chunk_count)`; a corpus
with fewer chunks uses all available chunks. It may only permute those IDs: it cannot add,
remove, duplicate, or retrieve chunks. The reranker child records the checksum
of the complete pool collection. It also records the owner's MLflow child-run ID
when an owner child exists. When the evaluated roster does not contain its
owner, the pool collection is prepared internally as an input; that case has
no owner child-run ID. A
mismatch or non-permutation invalidates the child and makes the parent comparison
incomplete.

Validation and locked shortlists need not contain `<retriever>|none`. Preparing
their required pools is normal input preparation, not an extra candidate or a
metric failure. The collection records its actual preparation provenance.
The same rule applies when preflight excludes a development baseline but leaves
another stack using that retriever qualified.

```text
load this profile's inputs and selected chunker/embedding contract
→ prepare the Dense/BM25 indexes required by the evaluated stacks
→ create each required retriever's top-20 query pool per question
→ checksum each collection and retain its owner/input-artifact provenance
→ score any evaluated no-reranker candidates at @3 and @5
→ execute the selected learned-reranker candidates on their verified pools
→ score every resulting order at @3 and @5
```

The top-20 pool supports reranking and diagnosis; it is not another quality
cutoff. No token-budget packing policy is applied in this phase. The first three
and first five canonical chunk texts are scored exactly as ranked.

Operational measurements do not reuse cached retrieval timings. Every reranked
candidate executes its first-stage method and reranker together so full-stack
latency and memory describe the deployable path. The frozen query pool is reused
only to guarantee a fair quality comparison: before reranking, the live ordered
IDs must match that question's frozen reference. A passing learned-reranker
execution therefore already establishes first-stage ordering consistency.
The no-reranker child measures the
same live first stage without a learned reranker.

The versioned `experiments/benchmarks/rag/retrieval_reranking/protocol.yaml` file is the
single source of truth for benchmark hyperparameters. Its strict schema rejects
missing or unknown fields. Algorithm definitions and deterministic tie-breaking
remain tested code invariants rather than configurable alternatives. The runner
checks the resolved protocol against the plan, records its checksum in every
child, logs the source file as an input, and writes `retrieval_protocol.json`
under the parent so a run can be reproduced without reconstructing settings
from code defaults.

The frozen controls include BM25
tokenization, variant, `k1`, `b`, and tie breaking; Dense similarity,
normalization, query formatting, and tie breaking; RRF source depth, fusion
constant, union/truncation behavior, and tie breaking; and each reranker's model
ID, revision, cache checksum, input template, native tokenizer, maximum input
length, batch size, score interpretation, device, dtype, and tie breaking.
Truncation is forbidden.

The target-hardware profile uses CUDA `float16`. The protocol fixes both
embedding and reranker inference to batch size `1`; learned rerankers therefore
score each actual query pool's `min(20, chunk_count)` pairs sequentially. Qualification includes
both components under the planned lifecycle and requires NVML-measured peak
device-total VRAM at or below 3,584 MiB under the shared hardware contract.
Per-candidate quantization, offload, fallback, or batch reduction is not an
allowed way to pass the gate.

The retrieval controls are fixed as follows. They are established, untuned
baselines rather than claims that one parameter set is optimal for every corpus:

- BM25 uses `BM25Okapi` with `k1=1.5`, `b=0.75`, and `epsilon=0.25`. The
  moderate `k1` gives repeated query terms diminishing returns, while `b=0.75`
  substantially normalizes variable chunk lengths without treating every
  length difference as irrelevant. In the selected implementation, `epsilon`
  floors negative IDF values for terms present in more than half the corpus.
  These are the explicit
  [`rank_bm25` defaults](https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py),
  so freezing them avoids adding a development-set tuning advantage to the
  lexical baseline.
- Dense retrieval uses exact cosine similarity over L2-normalized document and
  query vectors. Normalization prevents vector magnitude from affecting rank,
  and exact search isolates semantic retrieval from approximate-index error.
  Approximate nearest-neighbour behavior is evaluated separately in the vector
  database phase.
- RRF reads the Dense and BM25 top-20 lists, gives each source weight `1.0`,
  uses fusion constant `60`, joins by stable chunk ID, and keeps the highest
  scoring 20 unique chunks. Equal weights avoid an unvalidated preference for
  lexical or semantic retrieval, while rank fusion avoids normalizing their
  incomparable raw scores. The constant `60` is the frozen value used by the
  [original RRF study](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf).
  A missing source rank contributes zero. Ties resolve by best source rank, then
  combined source rank with missing ranks treated as `21`, then chunk ID.

Dense and BM25 own their respective searchable states. RRF builds no third
index: it references both checksummed indexes and creates only the fused ranked
pool. Its required storage is the unique sum of those dependencies, while its
component sizes remain separately inspectable. There is no incremental
fusion-index-size metric.

The exact ideal used to normalize linear nDCG sorts each chunk's complete
evidence-unit count over the complete frozen corpus for each question, never
over a candidate's retrieved pool. Alpha-nDCG uses a deterministic greedy
approximation from the same corpus and evidence-unit coverage, with the same
gain calculation, tie rule, and score cap as in chunking/embedding. Every
executed candidate therefore shares the same denominator for each metric on
a given question; the alpha-nDCG denominator is not a guaranteed maximum.

## Execution profiles and selection

```text
smoke:
minimal deterministic fixtures → wiring and artifact checks only

development:
all hardware-qualified members of the 15-candidate roster on the development manifest
→ engineer records up to three complete-stack finalists

validation:
the finalists on the unseen validation manifest
→ engineer records exactly one selected retrieval stack

locked test:
exactly one validation winner on the locked component split
→ no further retrieval or reranker tuning
```

Measured workers use one representative warmup before recorded requests.
Smoke uses one measured attempt; development, validation, and locked use three
measured attempts with seed 42. Preflight has no warmup and supplies hardware
qualification, not quality evidence. Loading, warmup, and measured attempts
follow the [shared lifecycle](../../methodology.md#preflight-and-measured-run-lifecycles).
This is the approved execution contract; the boundary above identifies the
current protocol's pending warmup migration.

A complete stack decision includes the frozen chunker, embedding, first-stage
retriever, and reranker. When a selected reranker needs its no-reranker owner's
pool, the runner prepares and verifies that pool as an input; the owner is not
added as another validation or locked candidate. No weighted score or automatic
winner rule is used.

## Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | **nDCG@3/@5**, **Evidence-unit Recall@3/@5**, **Evidence-token Precision@3/@5** | Separates evidence-richness ordering, evidence completeness, and concentration of retrieved text. |
| Diagnostic | alpha-nDCG@3/@5, candidate-pool Evidence-unit Recall@20, Ranking Repeatability | Diagnoses repeated evidence, first-stage pool limits, and stability of complete ordered outputs. |
| Operational | Full-stack warm p50/p95 latency, retrieval (first-stage) warm p50/p95 latency, reranker warm p50/p95 latency when applicable, cold initialization, peak process-tree RAM, peak device VRAM, separate Dense/BM25 index-build times | Separates retriever cost, incremental reranker cost, startup, memory, and one-time index preparation. |
| Workload descriptor | Mean/p95 Retrieved Context Tokens@3/@5, mean/p95 Reranker Input Tokens when applicable | Explains final context volume and native model input workload. |
| Storage descriptor | Index Size on each owner, RRF total with separate Dense/BM25 sizes | Describes searchable state without inventing a third RRF index. |

The three bold quality families are primary and are reviewed separately at both
cutoffs. Linear nDCG credits each chunk independently; repeated relevant evidence
does not reduce its gain. Alpha-nDCG adds a novelty preference and remains
diagnostic rather than treating repetition as a relevance-ranking error. It uses
`alpha=0.5` and uses the same answerable questions as the primary metrics,
including those with one gold evidence unit. Greedy-versus-exact normalization
is reviewed on development data as described in
[pending-data-review.md](../../pending-data-review.md).
Candidate-pool Recall@20 is recorded once on
each no-reranker pool owner because its four learned-reranker children receive
the same pool. Reranker-only latency and input-token metrics are omitted, not set to zero,
for the no-reranker option.

Eligible quality metrics are also reported separately for text, table, formula,
and mixed questions. These evidence types are reporting slices, not additional
metric roles.

The evidence-unit definitions, `tiktoken:cl100k_base` evaluation tokenizer,
answerable-question eligibility, document-macro aggregation, confidence
intervals, and text/table/formula/mixed slices are the same as in the
chunking/embedding phase. Exact definitions and comparison rules are in
[retrieval and reranking metrics](metrics.md).

Retrieval Warm Latency times Dense, BM25, or both source retrievals plus RRF fusion,
before learned reranking. Full-stack latency additionally includes the
reranker when present. All warm timers begin after the complete warmup; a
fresh-worker startup is reported separately as Cold Initialization Time.
That initialization counter excludes separately timed offline index construction
and input preflight. Dense and BM25 own their build-time observations; RRF
references those dependencies and has no separate third-index-build scalar.
The [metric reference](metrics.md#cold-initialization-time) gives the explicit
non-overlapping timing calculation rather than treating cold initialization
as the full elapsed readiness wait including corpus construction.

Quality uses the designated first measured ranking at both cutoffs. Later
measured attempts supply timing and diagnostic Ranking Repeatability, not
additional quality scores. A failed first ranking is unavailable; later success
does not replace it. A later failure does not erase valid first-ranking quality,
but still affects run-level validity. Required partial quality remains a labelled
diagnostic rather than a complete comparison result.

Failures and truncation are validity conditions rather than quality metrics.
Ranking Repeatability compares all three attempt pairs over the complete
ordered result, normally 20 IDs. Only identical valid rankings agree; executed
failures never agree. Valid differences are diagnostic, not an automatic
disqualification. Corpus inventory and pool-size checks stay in parameters and
artifacts rather than mandatory workload scalars. Optional Candidate-Pool
Tokens@20 is a per-query pool-artifact field; it does not duplicate final-context
token metrics.

## MLflow result structure

Retrieval/reranking uses `EduMind / Retrieval–Reranking` and one parent for each
fair comparison. Every planned
candidate is a direct child; the development parent therefore has up to 15
candidate children, with any definitive hardware exclusions retained in
qualification provenance. Pools and paired comparisons do not create
intermediate or nested runs.
Separate CPU/CUDA smoke parents and the preflight parent precede development.
Validation contains exactly the engineer-selected finalists; a required owner
pool is an input rather than an extra child. Locked contains exactly one
validation winner and is reporting-only.

```text
MLflow experiment: EduMind / Retrieval–Reranking
├── parent: smoke-cpu
├── parent: smoke-cuda
├── parent: preflight
│   └── one qualification child per declared stack
├── parent: development
│   ├── up to 5 Dense children
│   ├── up to 5 BM25 children
│   └── up to 5 RRF children
├── parent: validation
│   └── exactly the engineer-selected finalists
└── parent: locked
    └── one selected stack
```

The parent parameters identify the phase, profile, manifest and checksum,
selected chunker/embedding decision and fingerprint, the qualified subset of
the declared 15-candidate plan,
metric contract, protocol version and checksum, seed, warmups, repetitions,
model lock, Git/dependency
provenance, and hardware. Its direct metrics are completion counts only. Parent
input inventory/parameters also record verified corpus, question, and chunk
counts. Requested depth is a protocol parameter; observed pool sizes and optional
source-token totals belong to pool artifacts, with optional integrity summaries.
Parent artifacts are `plan.json`, `provenance.json`, `metric_contract.json`,
`retrieval_protocol.json`, `pool_collections.json`, `leaderboard.parquet`,
`retriever_comparisons.parquet`,
`retriever_comparisons.csv`, `reranker_comparisons.parquet`,
`reranker_comparisons.csv`, and `summary.json`. Validation additionally stores
`finalist_comparisons.parquet` and `finalist_comparisons.csv` when explicit
cross-stack finalist comparisons are requested.

Each child run name is its complete candidate ID. Its parameters contain the
resolved retrieval and reranking contracts, all model/configuration/checksum
fields, `requested_pool_size=20`, `quality_cutoffs=[3,5]`, evidence and tokenizer
rules, device, dtype, seed, warmups, repetitions, split, and manifest checksum.
A query pool is one question's ranked top-20 first-stage results. A pool
collection is the complete set of query pools for one retriever in this profile,
whether prepared by an evaluated owner child or as an input artifact.
A reranker child stores `pool_collection_checksum`, `chunks_sha256`, the
applicable `dense_index_sha256` and/or `bm25_index_sha256`, and the owner
candidate.
`pool_owner_child_run_id` is populated only when the pool collection was
produced by an actual MLflow owner child; an internally prepared collection
leaves it null rather than inventing a run ID. The pool-collection checksum
identifies every ranked row across every answerable question;
the other checksums identify every ordered chunk ID and its exact text plus the
concrete search indexes. Dense omits the BM25 field, BM25 omits the Dense field,
and RRF requires both. The chunking fingerprint remains separate provenance
explaining how those chunks were produced.

Under the approved reporting contract, a child logs applicable values under
these families. These are target keys, not aliases for unmodified legacy outputs:

```text
quality.overall.ndcg_at_3
quality.overall.ndcg_at_5
quality.overall.evidence_unit_recall_at_3
quality.overall.evidence_unit_recall_at_5
quality.overall.evidence_token_precision_at_3
quality.overall.evidence_token_precision_at_5
quality.overall.alpha_ndcg_at_3
quality.overall.alpha_ndcg_at_5
quality.<text|table|formula|mixed>.<metric>

diagnostic.pool_evidence_unit_recall_at_20   # pool owners only
diagnostic.ranking_repeatability

operational.full_stack_latency_ms_p50
operational.full_stack_latency_ms_p95
operational.retrieval_latency_ms_p50
operational.retrieval_latency_ms_p95
operational.reranker_latency_ms_p50          # reranked children only
operational.reranker_latency_ms_p95          # reranked children only
operational.cold_initialization_seconds
operational.peak_process_tree_ram_mb
operational.peak_vram_mb
operational.index_build_seconds              # Dense/BM25 index owners; no third RRF build

storage.index_size_bytes                     # Dense/BM25 owners; RRF dependency total
storage.dense_index_size_bytes               # RRF breakdown
storage.bm25_index_size_bytes                 # RRF breakdown

workload.retrieved_context_tokens_at_3_mean
workload.retrieved_context_tokens_at_3_p95
workload.retrieved_context_tokens_at_5_mean
workload.retrieved_context_tokens_at_5_p95
workload.reranker_input_tokens_per_query_mean # reranked children only
workload.reranker_input_tokens_per_query_p95  # reranked children only
```

Display names stay readable; scalar keys carry units because MLflow does not
infer them from a run name. Index sizes use bytes and build time uses seconds:

| Owner | Display name | Scalar key | Example value | Unit |
|---|---|---|---|---|
| Dense | Index Size | `storage.index_size_bytes` | 12,000,000 | bytes |
| BM25 | Index Size | `storage.index_size_bytes` | 3,000,000 | bytes |
| RRF | Index Size | `storage.index_size_bytes` | 15,000,000 | bytes |
| RRF | Dense Index Size | `storage.dense_index_size_bytes` | 12,000,000 | bytes |
| RRF | BM25 Index Size | `storage.bm25_index_size_bytes` | 3,000,000 | bytes |
| Dense | Index Build Time | `operational.index_build_seconds` | 12 | seconds |
| BM25 | Index Build Time | `operational.index_build_seconds` | 0.8 | seconds |

RRF references both separate build observations in `index_build.json`; it has
no third build-time scalar or copied component-build metrics. A verified size
or completed build survives later query failure. Incomplete sizes/builds retain
progress and elapsed-to-failure in artifacts, not fabricated completed scalars.
Observed partial resource peaks remain in `resources.parquet` and
`candidate.json` with partial lifecycle scope. If logged as an observed peak,
that scalar must retain the same scope/status in artifacts; it is not proof
that a full workload fits. Missing telemetry remains unavailable.

Eligibility counts and validity counters are logged under `validity.*`.
They are observed counts or integrity gates, not additional quality families:

| Key | Unit and meaning | Required complete-result interpretation |
|---|---|---|
| `validity.expected_query_count` | Unique scheduled answerable question IDs, not repetitions. | Equals the planned answerable workload. |
| `validity.processed_query_count` | Unique questions with produced query-result records, not necessarily successful attempts. | Must account for every planned question; complete attempt rows are checked separately. |
| `validity.failed_query_count` | Unique questions with at least one recorded failed measured attempt. | Current validity requires zero; one question failing twice still counts once here. |
| `validity.truncated_input_count` | Model inputs actually truncated, not native-token counts or unexecuted inputs. | Must be zero; rejected oversized inputs retain their separate failure report. |
| `validity.nonfinite_score_count` | Recorded attempt-level nonfinite-score errors in the current counter, not the number of bad scalar entries. | Must be zero; preserve errors rather than inferring individual bad-score counts. |
| `validity.exact_pool_permutation` | Binary check of exact output membership/uniqueness for each required pool. | One passes; zero fails; missing required checks are incomplete. |

The per-scope keys below use `overall`, `text`, `table`, `formula`, or
`mixed`. They describe reference eligibility, not successful execution:

| Scoped key suffix under `validity.quality.<scope>.` | Meaning |
|---|---|
| `eligible_question_count` | Eligible answerable questions for that scope. |
| `eligible_document_count` | Distinct source documents containing eligible questions. |
| `alpha_ndcg_eligible_question_count` | Alpha-nDCG-eligible questions; the approved contract uses the same questions as primary quality, including single-unit questions. |
| `alpha_ndcg_eligible_document_count` | Their distinct source documents; it must match primary eligibility under the approved rule. |

Current alpha eligibility differs as described in the implementation boundary.
Scored/unavailable support and independent source grouping remain separate from
these reference counts. A contributing `sample_count` is not permission to
discard planned questions or treat repetitions as independent documents.
Required checks include expected/processed/failed queries, zero truncation,
finite scores, pool provenance/live-order agreement, exact pool permutation,
the ordered top-K prefix, and verification completeness. These checks retain
detailed outcomes without a redundant pool-collection-match scalar. A failed
check stops the affected attempt's downstream evaluation; invalid shared inputs
stop dependent work. Independent completed measurements remain available, but
the affected result is not complete authoritative selection evidence. Absence
of recorded mismatches does not prove verification happened.
Every child stores `query_metrics.parquet`, `rankings.parquet`,
`evidence_matches.parquet`, `timings.parquet`, `resources.parquet`,
`candidate.json`, and `validation_report.json`. Pool-owner children additionally
store `pool_collection.parquet` and `index_build.json`. Raw documents and model
weights are never copied into child artifacts.

The parent's `pool_collections.json` identifies collections and their owners;
owner-child `pool_collection.parquet` stores ranked query rows. An RRF
`index_build.json` identifies both dependency indexes, not a separately built
fusion index. Internally prepared collections retain input-artifact provenance
without an invented owner child.

## Paired comparisons and advancement

Paired comparisons are analysis rows under the parent, not MLflow runs. They
are separated by the intervention being studied:

- `reranker_comparisons.parquet` and `reranker_comparisons.csv` contain up to 12
  reranker-effect comparisons, each evaluated learned reranker against the matching
  `<retriever>|none` candidate over the same checksummed pool collection;
- `retriever_comparisons.parquet` and `retriever_comparisons.csv` contain Dense
  versus BM25, Dense versus RRF, and BM25 versus RRF using the three
  evaluated no-reranker candidates (up to three comparisons); and
- validation uses `finalist_comparisons.parquet` and
  `finalist_comparisons.csv` only for explicitly declared cross-stack finalist
  comparisons.

A comparison requires both candidates to have completed the same evaluation.
Missing hardware-excluded candidates do not receive fabricated comparison rows;
qualification provenance explains their absence. An internally prepared pool is
not a substitute for a measured no-reranker baseline.

Parquet is authoritative and typed; CSV is the human-readable mirror. Each pair
is generated from one in-memory typed table. The writer reads both files back
with their declared schemas, sorts them by stable row keys, and verifies column
and cell equality with explicit null and floating-point handling. Matching row
counts alone is insufficient. Each file has its own SHA-256 and the pair records
one shared logical-table SHA-256.

The benchmark does not generate all 105 possible candidate pairs. Each stored
row represents one comparison, metric, evidence slice, and cutoff. It identifies
the applicable retriever or complete stacks, both run IDs and pool-collection
checksums,
metric direction, both values, raw `candidate - baseline` difference, whether
the result favors the candidate, eligible question/document counts, and
paired-bootstrap confidence bounds when supported.

The reranker-comparison schema is:

```text
comparison_id, retriever,
baseline_candidate, candidate, baseline_run_id, candidate_run_id,
shared_pool_collection_checksum, metric_name, evidence_slice, cutoff, direction,
baseline_value, candidate_value, candidate_minus_baseline, favors_candidate,
ci_lower, ci_upper, confidence_level, ci_status, eligible_questions,
eligible_documents, bootstrap_resamples, seed
```

The retriever-comparison schema replaces `retriever` and
`shared_pool_collection_checksum` with `baseline_retriever`,
`candidate_retriever`, `baseline_pool_collection_checksum`, and
`candidate_pool_collection_checksum`. Finalist rows identify both complete
stacks and both pool-collection checksums.

`candidate_minus_baseline` always preserves the raw arithmetic difference.
`favors_candidate` is populated only when the metric has an unconditional
direction: positive quality differences and negative operational differences
favor the candidate. For descriptive workload or storage differences,
`direction=descriptive` and `favors_candidate` is null.

Quality rows require `cutoff` and an `evidence_slice` such as `overall`, `text`,
`table`, `formula`, or `mixed`; non-quality rows set both fields to null.
Confidence bounds, `confidence_level`, bootstrap resamples, and seed are
populated only when a paired interval is available. Otherwise they are null and
`ci_status` states why, such as `smoke`, `insufficient_documents`, or
`not_supported`.

Reranker-effect rows compare all primary quality metrics, diagnostic alpha-nDCG,
full-stack latency, cold initialization, RAM, VRAM, and retrieved-token workload.
They do not pretend that shared pool Recall@20, shared index build, retrieval
latency, or reranker-only fields are reranker improvements. Retriever-effect rows
may compare quality, pool Recall@20, full-stack and retrieval latency, cold
initialization, RAM, VRAM, retrieved-context tokens, Index Build Time, and Index Size.
Index-build point differences require two defined independent-index observations,
such as Dense versus BM25. Do not invent a zero RRF build-time observation or
substitute an unmeasured sum; its required dependency storage is reported separately.
Ranking Repeatability remains diagnostic; its observed pair outcomes do not
replace integrity checks or quality comparisons.
Validity counters are gates, not winner metrics, and receive no comparison rows.

Quality differences use 10,000 paired bootstrap resamples of aligned source
documents with seed 42. One-off cold-load and peak-resource observations receive
point differences but no invented confidence interval. The engineer selects up
to three complete finalists for validation after jointly reviewing primary
quality, uncertainty, evidence slices, diagnostics, and operational feasibility,
then records exactly one validation winner. That stack receives its own locked
component report and is used unchanged by Final RAG. The versioned decision file
references the parent and child run IDs and all governing checksums.
