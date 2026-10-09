# Chunking and embedding methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Terminology and unit of comparison](#terminology-and-unit-of-comparison)
- [Candidate pairs](#candidate-pairs)
- [Data](#data)
- [Common input and output rules](#common-input-and-output-rules)
- [Per-candidate execution](#per-candidate-execution)
- [Development, validation, and locked test](#development-validation-and-locked-test)
- [Metrics and why they are used](#metrics-and-why-they-are-used)
- [MLflow result structure](#mlflow-result-structure)

Which complete chunker/embedding pair retrieves verified educational evidence
best?

## Terminology and unit of comparison

Chunking and embedding are evaluated together. The candidate is the complete
`chunker|embedding` pair: a chunker cannot be scored until its chunks are
represented and ranked, and an embedding cannot be judged independently of the
passages it embeds. This phase isolates that pair by using exact cosine search;
it does not test BM25, rank fusion, a reranker, a vector database, a generator,
or final answer quality.

Each pair produces:

- one chunk manifest with stable IDs and source offsets;
- one ranked retrieval list for every answerable question;
- evidence matches and per-question quality values; and
- timing, resource, warning, and validity records.

Quality is evaluated on answerable questions with verified evidence. The
answerability label remains in the manifest for downstream generation and final
RAG evaluation, but this phase does not ask a dense retriever to abstain. An
unanswerable question has no gold evidence and therefore cannot receive a
fabricated retrieval-quality score.

One answerable question is one scoring sample. Because questions from the same
paper are related, the source document—not the individual question—is the
independent resampling unit. Repeated query executions improve timing
measurement but do not create additional quality samples.

## Candidate pairs

The declared matrix crosses every chunking strategy with every embedding model.
Development evaluates its preflight-qualified members. The result belongs to
the complete pair.

### Chunking strategies

| Strategy | What it tests |
|---|---|
| Recursive character 1000/200 | Cheap textual separators without tokenizer dependence. |
| Token 256/32 | Short, focused passages with modest overlap. |
| Token 384/64 | Middle fixed-token trade-off. |
| Token 512/64 | More local context per chunk. |
| Sentence 8/2 | Linguistic boundaries instead of fixed token windows. |
| Semantic | Topic-change boundaries based on adjacent-sentence embedding similarity. |
| Section-aware 512/64 | Preserves authored Markdown sections before splitting oversized sections. |
| Structure-aware 512/64 | Preserves headings, tables, formulas, and table-row boundaries where possible. |

### Embedding models

| Model | Role in the comparison |
|---|---|
| GTE ModernBERT base | 149M lightweight production control with an 8,192-token input limit. |
| Snowflake Arctic Embed M v2 | Strong small-model retrieval candidate with a long input limit. |
| F2LLM v2 0.6B | Alternative 0.6B retrieval architecture. |
| Octen Embedding 0.6B | Strong 0.6B retrieval challenger. |
| Qwen3 Embedding 0.6B | Strong directly comparable 0.6B candidate using documented last-token pooling. |
| Nemotron 3 Embed 1B | Larger nearby-size retrieval candidate. |

Full public evidence and exact revisions are in
[model-selection rationale](../../model-selection.md).

## Data

The RAG corpus uses pinned QASPER papers plus EduMind's additional verified
structured-evidence set. The counts below describe QASPER papers only:

| Split | QASPER papers | Used for |
|---|---:|---|
| Development | 100 | Development candidate comparison |
| Validation | 40 | Validation finalist comparison |
| Locked test | 40 | One selected chunking–embedding pair; reporting only |

Each question stores answerability, accepted answers, evidence type, and exact
half-open evidence offsets. The structured supplement contains table, formula,
and mixed-evidence questions because QASPER is primarily text.

## Common input and output rules

Every pair receives the same frozen documents, answerable questions, and
evidence annotations. Candidate-specific text repair, query rewriting, reranking,
and approximate vector search are not allowed in this phase.

Each answerable question contains one or more independently useful **gold
evidence units**. A unit has a stable ID, one mutually exclusive evidence type,
and one or more exact half-open source intervals. Units must be atomic enough to
fit inside a valid candidate chunk. A table unit includes any row/column headers
needed to interpret its selected cells, and a formula unit includes the complete
expression. Evidence that genuinely requires different modalities or separated
support is represented by multiple units and the question is labelled `mixed`.

For the rank-aware and unit-recall metrics, chunk `j` covers evidence unit `i`
only when that single chunk contains every required source interval for the
unit. Partial overlap does not turn a fragment into complete evidence. This
rule makes boundary fragmentation visible and avoids candidate-specific fuzzy
matching. The metric roles, counting behavior, and edge cases are summarized in
[chunking and embedding metrics](metrics.md).

Evidence-token Precision uses the frozen evaluation tokenizer
`tiktoken:cl100k_base` for every candidate. It never uses each embedding model's
tokenizer for scoring, because changing the counting unit between candidates
would make their precision values incomparable. Model-specific tokenizers are
used separately to prepare inference inputs and validate their exact lengths.
The fixed tokenizer is a protocol control, not another benchmark candidate. A
future multilingual protocol may evaluate tokenizer sensitivity separately
before freezing a replacement.

Every declared pair is checked against the embedding model's native input
contract, including query/document prefixes and required special tokens.
Preflight records native-token counts on reviewed development/stress inputs;
each evaluation also validates all chunks and answerable queries in its own
manifest. Validation and locked content are never inspected by preflight.
Execution never silently truncates, changes a strategy's size, or splits chunks
again for one model. An oversized input is reported as `input_length_exceeded`:
a preflight input-contract failure blocks development until resolved, while an
evaluation child with that failure makes its parent comparison incomplete.

## Per-candidate execution

One child run executes one planned pair in a fresh operating-system process.
The requested device, dtype, model and tokenizer revisions, query/document
prefixes, pooling, normalization, seed, warmups, and repetitions are fixed and
recorded. Silent device fallback or unrecorded truncation invalidates the child.
Authoritative development, validation, and locked comparisons use the target RTX 3050 through
CUDA with `float16` and embedding batch size `1`. Peak Device VRAM must remain
at or below 3,584 MiB under the shared whole-device contract.
The same settings apply to every pair; a candidate cannot receive a smaller
batch or a different precision to avoid an out-of-memory result.

An eligible pair performs:

```text
split documents into chunks with exact source offsets
-> validate every chunk with the candidate model tokenizer
-> embed every valid chunk
-> embed each answerable question
-> rank with exact NumPy cosine search and deterministic tie-breaking
-> retain the top 20 as the auditable retrieval artifact
-> score the first three and first five against verified evidence units and source spans
```

Exact NumPy search removes vector-database approximation from this experiment.
For semantic chunking, the tested embedding also creates the boundaries; that
result intentionally represents the complete semantic-chunker/embedding pair.
The search scores the complete candidate corpus, retains the top 20 only for
near-miss diagnosis, and scores both `@3` and `@5`. These cutoffs match the two
downstream Final-RAG context counts; this phase adds no token budget or context
curve.

Corpus-build timing starts after model and tokenizer loading and includes
chunking, document embedding, and searchable-matrix/metadata assembly. Query
latency includes query tokenization, query embedding, exact cosine scoring, and
deterministic top-20 selection. Each query's median measured repetition is its
warm-latency observation. Model downloads, data downloads, and environment setup
are excluded.

A successful child must account for every expected document and answerable
query, produce the expected number and dimension of finite nonzero vectors, use
zero truncated inputs, and reproduce the same ordered top-five IDs for repeated
identical queries. An evaluation-time input-limit violation, crash, OOM, or
malformed output retains its validation report and makes the child failed and
comparison incomplete. Definitive hardware exclusions recorded by preflight
are different: those candidates never enter the development comparison, and
their reasons remain in its qualification provenance.

## Development, validation, and locked test

```text
smoke:
declared chunking and embedding paths on tiny committed fixtures
-> verify compatibility checks, embedding, ranking, scoring, and artifacts

development:
8 chunkers × 6 embeddings = 48 declared pairs
-> qualify all 48 using reviewed development/stress inputs
-> evaluate exactly the qualified subset on 100 papers
-> retain all hardware exclusions in qualification provenance
-> engineer selects up to three complete finalist pairs

validation:
selected finalists on 40 unseen papers
-> engineer records exactly one selected chunker/embedding pair

locked test:
the selected pair runs in its own component benchmark on 40 locked papers
-> reporting only, with no further component tuning
```

Smoke validates wiring only. Development compares the qualified matrix members, and
validation checks the finalists on unseen papers. A failed pair makes its parent
comparison incomplete. The component locked report evaluates the selected pair
in isolation and is not another chunking/embedding selection round.

## Metrics and why they are used

| Category | Metrics | Why they are needed |
|---|---|---|
| Retrieval quality | **nDCG@3/@5**, **Evidence-unit Recall@3/@5**, **Evidence-token Precision@3/@5**; alpha-nDCG@3/@5 diagnostic | Measures evidence-richness ordering, evidence completeness, and context concentration; alpha-nDCG diagnoses novelty on the same answerable questions. |
| Evidence slices | Retrieval metrics repeated for text, table, formula, and mixed questions | Reveals a pair that performs well only on the majority evidence type. |
| Operational | Corpus-build time and source-token throughput, p50/p95 warm query latency, peak process-tree RAM, peak device VRAM | Measures the observed preparation, query, and hardware cost of the complete pair. |
| Workload and storage | Corpus counts, source and indexed-token counts, chunk count and lengths, embedding dimension and dtype, matrix bytes | Explains how much work and storage each pair creates without treating those values as quality. |

The three bold metric families are primary and are reviewed separately at both
cutoffs. nDCG uses each chunk's complete evidence-unit count directly as its
linear gain, without discounting evidence already supplied by another chunk.
Its exact ideal sorts all chunk gains in the pair's complete corpus, not only
the retrieved top 20. This evaluates relevance ranking; Evidence-unit Recall
separately measures distinct coverage.

Alpha-nDCG is diagnostic with fixed `alpha=0.5`. Its denominator uses a
deterministic greedy approximation over the same complete corpus, choosing the
unused chunk with the greatest remaining alpha gain at each rank and resolving
ties by frozen corpus order. Scores are capped at one. It describes novelty,
not a relevance failure or demonstrated harm from repeated evidence. The
[data-review checklist](../../pending-data-review.md) tracks whether exact
normalization is practical; any change is frozen before held-out evaluation.
All four retrieval-quality metrics use every answerable question with at least
one gold evidence unit, including single-unit questions. No weighted overall
score is created.
First-hit metrics and chunk-level precision/recall are omitted because they add
little coverage information or use denominators changed by the chunker. Exact
definitions, examples, directions, and confidence-interval rules are in
[chunking and embedding metrics](metrics.md).

The four retrieval metric families are reported overall and for mutually
exclusive `text`, `table`, `formula`, and `mixed` slices. A mixed question
requires at least two evidence types. Each slice reports its contributing
question and document counts, shared by all four metrics. Unanswerable questions
remain a workload count and do not enter retrieval-quality aggregates.

## MLflow result structure

Chunking/embedding uses `EduMind / Chunking–Embedding` and one parent per fair
comparison. Independent CPU/CUDA smoke parents precede preflight; the final
reporting-only parent contains exactly one validation winner:

```text
MLflow experiment: EduMind / Chunking–Embedding
├── parent: rag-chunking-embedding-smoke-cpu-<timestamp>
│   └── one child per smoke-tested pair on CPU
├── parent: rag-chunking-embedding-smoke-cuda-<timestamp>
│   └── one child per smoke-tested pair on CUDA
├── parent: chunking-embedding-preflight-<timestamp>
│   └── one qualification child per declared pair
├── parent: rag-chunking-embedding-development-<timestamp>
│   └── up to 48 children: one per hardware-qualified planned pair
├── parent: rag-chunking-embedding-validation-<timestamp>
│   └── up to three child runs: one per engineer-selected finalist pair
└── parent: rag-chunking-embedding-locked-<timestamp>
    └── one child for the selected pair
```

Each parent stores the phase, dataset and checksum, candidate plan, seed,
repetitions, metric contract, model lock, Git and
dependency provenance, hardware, and any engineer-decision file. Its direct
metrics contain completion counts only: planned, successful, and failed pairs.
`plan.json`, `provenance.json`, `metric_contract.json`, `leaderboard.parquet`,
paired comparisons, and `summary.json` are parent artifacts. The chosen pair's
locked result remains in this component experiment and cannot reopen selection.

Each child is one planned pair. Its run name is the complete pair identifier,
`<chunker_id>|<embedding_id>`. Parameters contain the resolved chunker and
embedding contracts, model and tokenizer revisions and checksums, device, dtype,
prefixes, pooling, normalization, evaluation tokenizer, evidence rule,
`quality_cutoffs=[3,5]`, `artifact_top_k=20`, `alpha=0.5`, seed, warmups,
repetitions, split, and manifest checksum.
The child has no child runs for documents, questions, repetitions, or metrics.

A valid child logs:

```text
quality.overall.ndcg_at_3
quality.overall.ndcg_at_5
quality.overall.evidence_unit_recall_at_3
quality.overall.evidence_unit_recall_at_5
quality.overall.evidence_token_precision_at_3
quality.overall.evidence_token_precision_at_5
quality.overall.alpha_ndcg_at_3
quality.overall.alpha_ndcg_at_5

quality.text.<metric>
quality.table.<metric>
quality.formula.<metric>
quality.mixed.<metric>

operational.corpus_build_seconds
operational.corpus_build_source_tokens_per_second
operational.query_latency_ms_p50
operational.query_latency_ms_p95
operational.peak_process_tree_ram_mb
operational.peak_vram_mb

workload.source_tokens
workload.indexed_token_occurrences
workload.document_count
workload.answerable_query_count
workload.unanswerable_query_count
workload.chunk_count
workload.chunk_tokens_mean
workload.chunk_tokens_p95
workload.embedding_dimension
workload.embedding_matrix_bytes
```

Primary, diagnostic, operational, and descriptor labels are documentation
roles; they are not repeated as MLflow prefixes. The prefixes above describe
metric families and evidence slices. Eligible uncertainty bounds use the shared
`.ci_lower` and `.ci_upper` suffixes.
Validity counters such as expected/processed documents and queries, truncated
inputs, failed cases, nonfinite/zero-norm vectors, dimension mismatches, and
determinism mismatches are logged under `validity.*`. Retrieval-eligible question
and document counts are shared by all four quality metrics and logged per
evidence scope. The final decision is also stored as a
`benchmark.valid` tag and a `validation.status` tag whose value is `passed` or
`failed`. MLflow scalar values do not replace the
detailed `validation_report.json` artifact. Stored dtype is a parameter in the
resolved embedding contract rather than a numeric metric.

Each successful child stores:

| Artifact | Contents and purpose |
|---|---|
| `chunk_manifest.parquet` | One row per chunk with source document, exact offsets, strategy metadata, and token counts. |
| `query_metrics.parquet` | One row per answerable question with all four retrieval-quality families at @3/@5 and its evidence slice. |
| `retrievals.parquet` | Ordered top-20 chunk IDs and scores for near-miss diagnosis; the first three and first five are scored. |
| `evidence_matches.parquet` | Trace from each question and evidence unit to the chunks that recovered it. |
| `timings.parquet` | One row per query and measured repetition with warm latency and success state. |
| `resources.parquet` | Timestamped process-tree RAM and raw assigned-device VRAM samples, with measurement identity. |
| `candidate.json` | Resolved contracts, status, fingerprint, aggregates, confidence intervals, operational values, and artifact references. |
| `validation_report.json` | Input-length and validity checks, counts, limits, checksums, and any errors. |

Raw documents, model weights, and a full embedding matrix are not duplicated
into every child; frozen inputs, offsets, revisions, and checksums make the rows
reproducible. Any failed child logs its reason and available validation evidence,
remains visible with MLflow status `FAILED`, and makes the parent comparison
incomplete.

Aggregation and uncertainty follow the
[chunking and embedding confidence-interval contract](metrics.md#chunking-and-embedding-confidence-intervals).
Evidence slices use the classifications above and report their contributing
question and document counts.

The engineer approves up to three complete chunker/embedding pairs. No separate
embedding winner or chunker winner is required. Advancement jointly reviews
nDCG, Evidence-unit Recall, and Evidence-token Precision at `@3` and `@5` as the
three primary quality dimensions. Diagnostic alpha-nDCG, uncertainty, evidence
slices, and operational feasibility explain or constrain that judgment; they
are not steps in an automatic lexicographic ranking. The decision and rationale
are stored in a versioned engineer-decision file that references the
parent/child MLflow run IDs and all governing checksums. The benchmark never
promotes a candidate automatically.
