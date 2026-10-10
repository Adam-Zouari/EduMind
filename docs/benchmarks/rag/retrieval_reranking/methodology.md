# Retrieval and reranking methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Candidate matrix](#candidate-matrix)
- [Frozen candidate pools and execution](#frozen-candidate-pools-and-execution)
- [Execution profiles and selection](#execution-profiles-and-selection)
- [Metrics and why they are used](#metrics-and-why-they-are-used)
- [MLflow result structure](#mlflow-result-structure)
- [Paired comparisons and advancement](#paired-comparisons-and-advancement)

Which complete first-stage retriever and reranker gives the best evidence
ordering for the selected chunker/embedding pair, and is any quality improvement
worth its latency and resource cost?

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

The experiment uses the same frozen QASPER-plus-structured manifest, canonical
chunks, questions, evidence-unit IDs, and source intervals as the preceding
chunking/embedding decision.

The [shared data-validation workflow](../../data-validation.md) checks those
prepared inputs and source isolation before execution; its standalone/report
interface is planned. Generated pool-collection, chunk-content, and index
checksums are separate runtime integrity gates, not additional data annotations.

The retrieval/reranking smoke fixture contains 30 frozen canonical chunks. This
leaves ten chunks outside each top-20 pool, so smoke execution can exercise pool
selection, exclusion, fusion, and permutation checks. The count is a smoke
fixture control; authoritative chunking strategies are not forced to produce
exactly 30 chunks because their natural chunk counts are benchmark outputs.

Quality comparisons use one checksummed top-20 pool per first-stage retriever.
The three no-reranker candidates own those artifacts:

| Pool owner | Shared by |
|---|---|
| `dense\|none` | All five Dense candidates. |
| `bm25\|none` | All five BM25 candidates. |
| `rrf\|none` | All five RRF candidates. |

A reranker receives exactly the query pool owned by its matching no-reranker
child for every question. It may only permute those 20 chunk IDs: it cannot add,
remove, duplicate, or retrieve chunks. The reranker child records the checksum
of the complete pool collection. It also records the owner's MLflow child-run ID
when an owner child exists. When the evaluated roster does not contain its
owner, the pool collection is prepared internally as an input; that case has
no owner child-run ID. A
mismatch or non-permutation invalidates the child and makes the parent comparison
incomplete.

```text
load the frozen selected chunker/embedding pair and corpus
→ build Dense and BM25 indexes
→ create one Dense, BM25, and RRF top-20 query pool per question
→ checksum each retriever's complete pool collection under its no-reranker owner
→ score each no-reranker order at @3 and @5
→ rerank each matching frozen query pool with each of the four learned rerankers
→ score every resulting order at @3 and @5
```

The top-20 pool supports reranking and diagnosis; it is not another quality
cutoff. No token-budget packing policy is applied in this phase. The first three
and first five canonical chunk texts are scored exactly as ranked.

Operational measurements do not reuse cached retrieval timings. Every reranked
candidate executes its first-stage method and reranker together so full-stack
latency and memory describe the deployable path. The frozen query pool is reused
only to guarantee a fair quality comparison. The no-reranker child measures the
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
score the 20 query-passage pairs sequentially. Qualification includes
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
incremental fusion-index storage is zero.

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

A complete stack decision includes the frozen chunker, embedding, first-stage
retriever, and reranker. When a selected reranker needs its no-reranker owner's
pool, the runner prepares and verifies that pool as an input; the owner is not
added as another validation or locked candidate. No weighted score or automatic
winner rule is used.

## Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | **nDCG@3/@5**, **Evidence-unit Recall@3/@5**, **Evidence-token Precision@3/@5** | Separates evidence-richness ordering, evidence completeness, and concentration of retrieved text. |
| Diagnostic | alpha-nDCG@3/@5, candidate-pool Evidence-unit Recall@20 | Diagnoses repeated evidence and first-stage pool limits without replacing the primary decision. |
| Validity gate | Ranking Agreement | Requires repeated inference to return the same complete ordering before the run may be used as evidence. |
| Operational | Full-stack warm p50/p95 latency, first-stage warm p50/p95 latency, reranker warm p50/p95 latency when applicable, cold initialization, peak process-tree RAM, peak device VRAM, index-build time | Separates retriever cost, incremental reranker cost, startup, memory, and one-time index preparation. |
| Workload descriptor | Corpus/query/chunk/pool counts, retrieved tokens at @3/@5, reranker input tokens when applicable | Explains the amount of work behind quality and operational results. |
| Storage descriptor | Dense/BM25 index bytes, RRF required and incremental index bytes, reranker snapshot bytes | Describes the local searchable state and model storage each candidate requires. |

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

Failures and truncation are validity conditions rather than quality metrics.

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
├── parent: rag-retrieval-reranking-smoke-cpu-<timestamp>
├── parent: rag-retrieval-reranking-smoke-cuda-<timestamp>
├── parent: retrieval-reranking-preflight-<timestamp>
│   └── one qualification child per declared stack
├── parent: rag-retrieval-reranking-development-<timestamp>
│   ├── up to 5 Dense children
│   ├── up to 5 BM25 children
│   └── up to 5 RRF children
├── parent: rag-retrieval-reranking-validation-<timestamp>
│   └── exactly the engineer-selected finalists
└── parent: rag-retrieval-reranking-locked-<timestamp>
    └── one selected stack
```

The parent parameters identify the phase, profile, manifest and checksum,
selected chunker/embedding decision and fingerprint, the qualified subset of
the declared 15-candidate plan,
metric contract, protocol version and checksum, seed, warmups, repetitions,
model lock, Git/dependency
provenance, and hardware. Its direct metrics are completion counts only. Parent
artifacts are `plan.json`, `provenance.json`, `metric_contract.json`,
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
collection is the complete set of query pools produced by one retriever child.
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

A valid child logs the applicable values under these families:

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
validity.ranking_agreement

operational.full_stack_latency_ms_p50
operational.full_stack_latency_ms_p95
operational.first_stage_latency_ms_p50
operational.first_stage_latency_ms_p95
operational.reranker_latency_ms_p50          # reranked children only
operational.reranker_latency_ms_p95          # reranked children only
operational.cold_initialization_seconds
operational.peak_process_tree_ram_mb
operational.peak_vram_mb
operational.index_build_seconds              # pool owners only

storage.index_bytes                          # Dense and BM25 owners
storage.required_index_bytes                 # RRF owner: unique Dense + BM25 bytes
storage.incremental_index_bytes              # RRF owner: zero
storage.reranker_snapshot_bytes              # reranked children only

workload.corpus_document_count
workload.answerable_query_count
workload.chunk_count
workload.requested_pool_size
workload.actual_pool_size_mean
workload.actual_pool_size_min
workload.short_pool_query_count
workload.retrieved_tokens_at_3_mean
workload.retrieved_tokens_at_3_p95
workload.retrieved_tokens_at_5_mean
workload.retrieved_tokens_at_5_p95
workload.reranker_input_tokens_per_query_mean # reranked children only
workload.reranker_input_tokens_per_query_p95  # reranked children only
```

Eligibility counts and validity counters are logged under `validity.*`.
Required checks include expected/processed/failed queries, zero truncation,
finite scores, pool-collection agreement, exact pool permutation, and deterministic
rank agreement. Every child stores `query_metrics.parquet`, `rankings.parquet`,
`evidence_matches.parquet`, `timings.parquet`, `resources.parquet`,
`candidate.json`, and `validation_report.json`. Pool-owner children additionally
store `pool_collection.parquet` and `index_build.json`. Raw documents and model
weights are never copied into child artifacts.

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
They do not pretend that shared pool Recall@20, shared index build, first-stage
latency, or reranker-only fields are reranker improvements. Retriever-effect rows
may compare quality, pool Recall@20, full-stack and first-stage latency, cold
initialization, RAM, VRAM, retrieved tokens, index-build time, and index bytes.
Validity counters are gates, not winner metrics, and receive no comparison rows.

Quality differences use 10,000 paired bootstrap resamples of aligned source
documents with seed 42. One-off cold-load and peak-resource observations receive
point differences but no invented confidence interval. The engineer selects up
to three complete finalists for validation after jointly reviewing primary
quality, uncertainty, evidence slices, diagnostics, and operational feasibility,
then records exactly one validation winner. That stack receives its own locked
component report and is used unchanged by Final RAG. The versioned decision file
references the parent and child run IDs and all governing checksums.
