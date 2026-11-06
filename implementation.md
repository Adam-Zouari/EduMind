# Implementation audit and repair handoff

Audit date: **2026-11-06**  
Baseline commit: **e04fa1e7be8332175c4498d0130432c489157f90**

This is a repair brief for the next implementation pass, not a record of completed
fixes. The audit changed no implementation, test, protocol, or benchmark guide.
The former follow-up checklist was consolidated into this handoff and removed.

## Reading this handoff

The highest-risk findings concern lost evidence, incorrectly accepted results,
unverified model/evaluator identity, a broken Final RAG execution path, and
application rollback. Address these before cosmetic cleanup. There are **57
numbered work items**, including verified defects, approved migrations and one
explicit legacy-removal item; they are not 57 independent newly discovered bugs.

- [Shared infrastructure](#1-shared-infrastructure)
- [Preparation](#2-model-evaluator-and-dataset-preparation)
- [Document](#3-document-extraction) and [ASR/video](#4-asr-and-video-extraction)
- [Chunking–embedding and retrieval–reranking](#5-chunkingembedding-and-retrievalreranking)
- [Generation](#6-generation), [Final RAG](#7-final-rag), and [Vector Database](#8-vector-database)
- [Application findings](#9-application-issues-found-while-tracing-benchmark-consumers)
- [Cleanup/tests](#10-justified-cleanup-and-test-changes),
  [confirmation items](#11-items-requiring-confirmation-not-automatic-changes),
  and [data/environment boundaries](#12-dataenvironment-boundaries)
- [Documentation completion](#13-documentation-completion-checklist-for-the-fixing-pass)
  and [implementation order](#14-implementation-order-and-completion-gates)

### Contract references

Read [shared methodology](docs/benchmarks/methodology.md),
[shared metrics](docs/benchmarks/metrics.md), and
[data validation](docs/benchmarks/data-validation.md), then the applicable pair:

| Benchmark | Methodology | Metrics |
|---|---|---|
| Document | [Guide](docs/benchmarks/extraction/document/methodology.md) | [Reference](docs/benchmarks/extraction/document/metrics.md) |
| ASR | [Guide](docs/benchmarks/extraction/audio/methodology.md) | [Reference](docs/benchmarks/extraction/audio/metrics.md) |
| Video | [Guide](docs/benchmarks/extraction/video/methodology.md) | [Reference](docs/benchmarks/extraction/video/metrics.md) |
| Chunking–embedding | [Guide](docs/benchmarks/rag/chunking_embedding/methodology.md) | [Reference](docs/benchmarks/rag/chunking_embedding/metrics.md) |
| Retrieval–reranking | [Guide](docs/benchmarks/rag/retrieval_reranking/methodology.md) | [Reference](docs/benchmarks/rag/retrieval_reranking/metrics.md) |
| Generation | [Guide](docs/benchmarks/rag/generation/methodology.md) | [Reference](docs/benchmarks/rag/generation/metrics.md) |
| Vector Database | [Guide](docs/benchmarks/vectordb/methodology.md) | [Reference](docs/benchmarks/vectordb/metrics.md) |
| Final RAG | [Guide](docs/benchmarks/rag/final/methodology.md) | [Reference](docs/benchmarks/rag/final/metrics.md) |

## Instructions for the implementing agent

Implement the verified fixes and approved migrations below, using the current
benchmark methodology and metric references as the intended contract. Read the
relevant shared and local documentation before editing each subsystem.

1. Recheck the working tree and cited symbols: line numbers refer to the baseline
   above and will move. Preserve unrelated user changes.
2. Work in dependency order. Add focused behavioral regressions, fix the behavior,
   then remove superseded code and tests whose contracts are genuinely retired.
3. Do not restore an obsolete metric, classifier, profile, or human-review stage
   just to make an old caller run. Migrate the caller to the approved contract.
4. Keep application policy separate from benchmark selection policy. Application
   findings are explicitly separated below; they are not benchmark promotions.
5. Reuse existing helpers when their semantics match. Do not create a universal
   framework, split files cosmetically, or delete validation solely to reduce lines.
6. Do not invent authoritative datasets, winners, calibration results, judge
   identities, configuration ranges still awaiting review, or CI-support minima.
   Distinguish implementation completion from data/environment readiness.
7. Preserve raw predictions, failed attempts, partial measurements, source
   identities, and historical artifacts. Evaluation recovery must not rerun
   locked inference or replace a designated first attempt.
8. After each verified change, update the corresponding implementation-status
   statements and executable commands in the docs. Do not leave a completed
   feature described as unimplemented; do not remove genuine readiness warnings.
9. Re-audit code, callers, tests, artifact contracts, and documentation after
   implementing. Report remaining blockers honestly rather than declaring
   completion because the existing tests pass.

This brief replaces the former follow-up checklist. The linked benchmark guides
remain authoritative for definitions. Version changed scoring definitions and
preserve historical evidence rather than silently relabeling old results.

### Classification

- **P1:** high-priority execution, data-loss, provenance, or evaluation-integrity issue.
- **P2:** narrower correctness, accounting, or reproducibility issue.
- **Migration:** an approved design not yet implemented; existing docs often
  correctly acknowledge it. This is not an accusation that those docs claim it works.
- **Cleanup:** a demonstrated simplification or retired contract, not speculative
  deletion based on a missing direct import.
- **Confirmation:** a concern needing a contract decision or live verification;
  not a proven defect and not permission to change methodology.

## Audit coverage and baseline checks

The review covered the eight benchmark families, their shared execution,
statistics, validation, preparation, tracking and worker boundaries, application
consumers, configuration/dependency files, and related tests. Six independent
subsystem reviews were consolidated with a shared-infrastructure review.
CLI entry points, registry factories, subprocess module launches, Docker-mounted
workers, and dynamic hooks were considered when checking reachability.

| Check | Result |
|---|---|
| Complete pytest suite | **1,214 passed**, one third-party Authlib deprecation warning, 111.78 seconds |
| Ruff check: src, experiments, tests | Passed |
| Ruff format --check | 175 files already formatted |
| AST parsing | All 175 Python files parsed successfully |
| Documentation heading-link scan | No suspected unresolved local heading fragments found; heuristic supplementary check |
| Focused failure reproductions | Read-only/in-memory or mocked; observations described below |
| Real model inference, GPU qualification, Docker services, live judge | **Not run in this audit** |

The initial sandboxed pytest run failed because pytest could not access its own
temporary directories. It was stopped and rerun with approval outside that
restriction. Those sandbox errors are not recorded as project defects.
Passing baseline tests do not establish compliance with subsequently approved
methodology: several tests only exercise legacy interfaces or omit these failures.

## 1. Shared infrastructure

### C01 — P1: preserve completed evidence when resource reporting fails

**Locations:** `experiments/benchmarks/common/runner.py:634`
(`_run_candidate`); `extraction/audio/worker.py:215`;
`extraction/video/visual_worker.py:298`, under the same benchmark root.
Related scope bug: `extraction/document/runner.py:283` and
`extraction/document/worker.py:513`.

**Evidence:** The common runner calls `resources.metrics()` before unpacking the
completed evaluator return. A mocked post-evaluation telemetry error produced a
failed candidate with zero samples, empty quality/operational dictionaries, and
lost predictions, despite completed evaluation. Audio replaces observed peaks
with null and only nests the original evidence in an error; visual also loses
that exception evidence. Document labels any execution error as a partial
measurement window, including errors occurring after monitoring completed.

**Fix:** Preserve the evaluator return and independently observed resource
evidence before resource validation. Carry monitoring completeness separately
from inference/integrity success. Keep valid observed peaks with their actual
window scope; missing observations stay null. Do not substitute supervisor
process-tree RAM for the worker's measurement or weaken qualification failures.

**Acceptance:** Completed predictions survive late telemetry failure; observed
RAM/VRAM peaks survive interruption; zero observations remain null; CPU VRAM
remains inapplicable; post-monitor placement failure preserves a complete window
while still failing integrity. Verify persisted artifacts and MLflow, not only
returned dictionaries. Update the contrary scope assertion in
`tests/test_document_benchmark.py:149`.

**Contract:** [Partial resource measurements](docs/benchmarks/metrics.md#missing-or-partial-resource-measurements)
and [completed stages](docs/benchmarks/metrics.md#completed-stages-and-failed-requests).

### C02 — P1: count-based F1 rounding changes threshold eligibility

**Locations:** `common/metrics.py:163` (`precision_recall_f1`),
`extraction/document/metrics.py:1172`, `extraction/audio/evaluate.py:819`.
Paths in this section are relative to `experiments/benchmarks/`.

**Evidence:** Eight reference tokens, twenty predicted tokens and seven shared
tokens give mathematically exact F1 = 0.5, but the current precision/recall
harmonic computation returns 0.4999999999999999. Document matching and ASR span
alignment reject the pair at the frozen 0.5 threshold. Frozen-video ASR shares
the scorer.

**Fix:** Compute count-based F1 directly from counts, preserving each caller's
explicit empty-set convention. Keep threshold semantics; do not loosen the
threshold with an arbitrary tolerance. Reuse the shared arithmetic rather than
adding another duplicate scorer.

**Acceptance:** Both token-length orientations match exactly at 0.5; a genuinely
lower score does not; document empty comparisons remain unchanged; audio and
frozen-video timestamp coverage/MAE use the corrected matched cohort.

### C03 — P2: worker protocol envelopes do not validate their schema

**Location:** `common/protocol.py:75` (`ProtocolMetadata.validate_worker_payload`).

**Evidence:** Document `protocol_from_worker` accepts an outer schema version of
999, true, or a missing schema field when resolved settings/version/checksum
remain unchanged. Only the inner resolved protocol schema is validated.

**Fix:** Strictly validate required envelope fields and outer schema version,
including rejecting booleans as versions. Retain version/checksum/resolved-content
validation. Apply through the common boundary, not duplicated domain checks.

**Acceptance:** Parameterized worker-envelope tests across every implemented suite
reject missing/unsupported/mistyped versions and unknown fields before loading
models, while valid round trips pass.

### C04 — P2 / migration dependency: retire the misleading generic CI fallback

**Locations:** `common/statistics.py:12`, `common/runner.py:683`,
`common/metrics.py:188`. Live legacy consumers include Vector Database and Final RAG.

**Evidence:** `aggregate_samples` removes null records before resampling,
bootstraps individual values, assumes two values establish independent support,
and lacks the complete support/undefined-draw/degeneracy contract. Two questions
from one source plus an unavailable record from another produce bounds and
`sample_count=2` without any source-aware support check.

**Fix:** Migrate remaining consumers to their declared estimators, eligible source
inventories, and reviewed support policies. Do not silently use this fallback for
pooled ratios, percentiles, document-macro quality, or load-window statistics.
Once no production caller needs it, remove the fallback and corresponding
superseded tests; keep reusable arithmetic that still has valid consumers.

**Acceptance:** One source with many questions cannot create independent support;
unavailable records remain in the resampling inventory; undefined draws are
counted; unfrozen support produces null bounds; equal supported bounds retain
degeneracy metadata. Existing modern domain aggregators must not regress.

**Contract:** [Shared confidence intervals](docs/benchmarks/metrics.md#confidence-intervals).

### C05 — P2: generation preflight logs an empty dataset checksum

**Locations:** `common/preflight.py:331` and `:379`;
`rag/generation/run.py:144` and `:306`.

**Evidence:** Qualification context contains
`input_envelope.stress_manifest_checksum`, but preflight parent/child tags read
`context.stress_manifest_checksum`. Generation does not add the redundant
top-level field; the fingerprint retains the identity but MLflow tags are empty.

**Fix:** Use one canonical validated context representation for execution,
fingerprinting and logging. Avoid another logging-only duplicate field.

**Acceptance:** Every suite's preflight parent and children have the correct
dataset identity, matching the saved context/report and explicit/local discovery.

## 2. Model, evaluator and dataset preparation

Paths below are relative to `experiments/benchmarks/` unless fully qualified.

### P01 — P1: dry-run dispatch can perform real preparation

**Locations:** `prepare.py:96`, `:108`, `:117`, `:162`.

**Evidence:** `qasper`, `rag-selection`, `assets` and `vectordb` dispatch ignore
`--dry-run`. Mocked dispatch confirms the mutating helpers are called.

**Fix:** Provide a non-mutating plan for every target, or reject unsupported
dry-run combinations before side effects.

**Acceptance:** Parameterize all preparation targets: dry-run makes no network,
Docker, filesystem-write or model-preparation calls. Do not run the currently
unsafe dry-run commands against real services as a test.

### P02 — P1: old snapshot bytes can be relabelled as a new revision

**Location:** `preparation/models.py:106`, `:117`, `:127`.

**Evidence:** Local directories are named by repository, not revision, and the
requested revision is recorded without validating the returned snapshot identity.
The installed Hub downloader can return existing local content when unavailable
remotely. An explicitly offline request for an unrelated commit returned the
existing GTE directory; no download was performed.

**Fix:** Prepare revision-specific completed snapshots and verify their immutable
inventory before publishing the lock. Reject unverifiable fallback, mixed revisions
and stale excluded exports. Do not erase existing downloads as an implicit repair.

**Acceptance:** An unavailable new revision cannot reuse/relabel old bytes;
interrupted preparation cannot publish a completed entry; resumed preparation
retains verified provenance and measured snapshot size.

### P03 — P1: Docling package pin is not a component-model pin

**Location:** `preparation/models.py:391`, `:420`, `:422`.

**Evidence:** Generic Docling preparation records package version 2.117.0 as
revision. The installed layout downloader uses a model revision of `main`.
A checksum authenticates the bytes already downloaded, but does not reproduce
the same bytes on a fresh preparation. Preparing a smaller component subset
also replaces the declared component inventory.

**Fix:** Record and request immutable repository revisions/checksums for each
prepared component, separately from package version. Preserve already verified
component/system provenance when augmenting a bundle.

**Acceptance:** Fresh preparation explicitly requests pinned component identities;
app preparation after benchmark preparation does not discard verified components;
all executed parser components are represented.

### P04 — P2: model-lock validation and schema migration have holes

**Locations:** `preparation/models.py:275`, `:337`, `:362`.

**Evidence:** A schema-3 entry missing both model_path and checksum passes because
checks depend on a truthy path. Separately, merging one prepared model upgrades
the whole lock to schema 3 while preserving untouched legacy entries without
their required checksums, making those entries unusable.

**Fix:** Validate supported schemas, mandatory path/directory/integrity fields and
types. Migrate old entries explicitly and atomically, or reject before publication
with an actionable migration instruction. Preserve stage-scoped validation of
only requested models; do not confuse optional size metadata with integrity.

**Acceptance:** Missing/null/empty paths and checksums are rejected. Partial
preparation preserves untouched usable entries or fails without publishing a
mixed invalid lock. Unrelated unrequested snapshots remain out of scope.

### P05 — P1: application preparation is coupled to benchmark eligibility

**Locations:** `preparation/models.py:79` and `:339`;
`config/base.yaml:42`; `experiments/benchmarks/selection_evidence.csv:44`.

**Evidence:** The documented app generator Qwen/Qwen3-1.7B is excluded from the
benchmark roster. `app-models --dry-run` rejects it, and lock merging can remove
an existing entry because it is not benchmark-included.

**Fix:** Separate pinned application controls from benchmark selection eligibility.
Do not add this model back into the benchmark roster to repair app setup. Scope
lock consumers to the models they actually need.

**Acceptance:** App preparation succeeds with the documented controls; later
benchmark preparation preserves those entries; benchmark candidate counts and
eligibility do not change.

### P06 — P1: evaluator source integrity is only a marker-file assertion

**Locations:** `preparation/evaluators.py:27`;
`extraction/document/official_metrics.py:30` and `:72`;
`common/data_validation.py:79`.

**Evidence:** A revision marker is trusted, then mutable local evaluator source
is bind-mounted into Docker. Changing table_metric.py while leaving the marker
and image lock intact does not change the claimed evaluator identity or sealed
reference-validation dependencies.

**Fix:** Verify a prepared source-content manifest covering executed evaluator
code/dependencies, or use verified source inside the immutable image. Bind the
actual identity into validation seals and scoring provenance.

**Acceptance:** Changed/deleted evaluator source is rejected before candidate
execution/scoring despite unchanged markers. Scorer corruption is an evaluator
failure, not a candidate quality zero. Coordinate D06/D07 below.

### P07 — P2: verified Tesseract installation is not bound to execution

**Locations:** `preparation/models.py:591` and `:602`;
`src/edumind/extraction/extractors/document.py:121`.

**Evidence:** Preparation verifies a specific executable and English traineddata.
Runtime constructs Tesseract options without those paths, leaving PATH and
default tessdata resolution able to select different files.

**Fix:** Pass and verify the exact prepared executable/language-data identities
using supported runtime options. Retain effective runtime provenance.

**Acceptance:** Changing PATH or tessdata environment cannot silently substitute
another installation while reporting the original lock.

### P08 — P2: vector preparation derives project paths from the lock destination

**Location:** `preparation/vectordb.py:52` and `:59`.

**Evidence:** `output_path.parents[3]` determines the Compose .env location.
A permitted relative output such as lock.json raises IndexError after preparation
and the JSON write; deep custom paths can place .env outside the project.

**Fix:** Resolve the Compose environment from its owned project location,
independently of the chosen lock output. Validate destinations before side effects.

**Acceptance:** Relative and absolute custom outputs work without changing .env
ownership; invalid destinations fail before Docker calls or partial publication.

**Documentation for P01–P08:** Update the relevant preparation commands,
immutable-lock/component guarantees, and verification instructions in
[installation](docs/setup/installation.md), [datasets](docs/benchmarks/datasets.md),
[data validation](docs/benchmarks/data-validation.md), and the
[runbook](docs/benchmarks/running.md) only after the corresponding fix is tested.

## 3. Document extraction

Paths below are relative to `experiments/benchmarks/extraction/document/`.
Also implement C01/C02 and P03/P06/P07; do not duplicate those fixes.

### D01 — P1: a malformed prediction can destroy all worker outcomes

**Locations:** `worker.py:427` (`extract_payload`);
`experiments/benchmarks/common/process.py:397`.

**Evidence:** The unvalidated produced_document is retained before canonical
validation. An object() in first-attempt metadata produces outcomes
failed/success/success, but final JSON serialization raises TypeError, losing the
entire worker ledger and later successful evidence.

**Fix:** Retain a bounded JSON-safe diagnostic representation of rejected native
payloads without accepting them as canonical results. Handle unsupported objects,
cycles, invalid keys and nonfinite values; sanitization must not invent valid text.

**Acceptance:** Each malformed case remains inspectable, every scheduled outcome
serializes, later attempts run, and canonical validation still rejects the bad
prediction. Extend the existing recoverable-failure serialization test.

### D02 — P1: Paddle effective batch settings are not frozen

**Locations:** `adapters.py:226` (`_paddle_runtime`); `protocol.py:314`.

**Evidence:** No batch overrides reach the constructor. The installed PaddleX
configuration has page batch 64, layout batch 8 and VL batch -1. Sequential
document calls do not enforce single-input inference inside a multipage pipeline.

**Fix:** Configure supported page/layout/VL batching from the execution contract
and validate initialized effective settings. The installed wrapper supports
paddlex_config; use supported APIs, not invented constructor keywords. Video
consumers of the same adapter must execute/log the same settings.

**Acceptance:** A mocked multipage initialization proves every exposed inference
batch control is enforced, and effective parameter artifacts agree. Follow with
an authorized real smoke when the runtime is available.

### D03 — P2: reference coercion changes claimed annotations

**Locations:** `metrics.py:808` (`_reference_from_mapping`), `:1438` (`_box`),
`:1461` (`_optional_string`).

**Evidence:** Authoritative validation accepts text:null as the text "None",
parent_id:"" as an explicitly annotated null root, and a malformed
three-coordinate table box as absent geometry, permitting text fallback.

**Fix:** Validate raw claimed fields before coercion. Distinguish omitted, null,
empty and invalid values; preserve valid blanks/negatives and unclaimed fields.

**Acceptance:** Reject the three examples without breaking blank-source and
capability-conditional tests. Update validation identity and regenerate affected
seals after the implementation changes.

### D04 — P2: page-only content is missed by Unexpected Empty Output Rate

**Location:** `metrics.py:549` (`expected_content`).

**Evidence:** A valid pages-only reference containing text plus a successful empty
prediction gets zero page recovery but Unexpected Empty Output Rate = 0.

**Fix:** Include usable content in claimed page annotations when deciding whether
empty output was unexpected. Do not infer obligations from unclaimed annotations.

**Acceptance:** Nonempty page-only reference/empty success → 1; verified blank
pages/empty success → 0; a failed attempt remains a separate failure outcome.

### D05 — P2: recoverable Paddle conversion loses rejected raw evidence

**Locations:** `adapters.py:171` and `:363`.

**Evidence:** Malformed table HTML followed by a valid block retains the valid
block and a generic warning but loses the rejected HTML/text. The worker cannot
retain native evidence it never receives.

**Fix:** Retain rejected block fragments, native page/block identity and conversion
errors in prediction evidence, separate from scored canonical content and
structured-output repeatability projections.

**Acceptance:** A unique marker in rejected HTML survives artifact serialization,
later blocks still score, and malformed content is not silently counted as valid.
Preserve allocation-guard tests.

### D06 — P1 migration: implement symmetric official TEDS normalization

**Location:** `omnidocbench_worker.py:35`.

**Evidence:** The wrapper adds an HTML/body envelope only. Inspection of the
pinned official table scorer confirms raw th nodes lose text/span information;
header text/colspan perturbations can remain indistinguishable.

**Fix:** Use the complete pinned normalized_html_table transformation on both
reference and prediction. Retain raw and scorer-facing HTML plus normalization,
wrapper, upstream revision and image identities.

**Acceptance:** Identical/equivalent-wrapper cases; changed header wording lowers
TEDS but not structural TEDS-S; changed spans lower both. Exercise the actual
nonempty official interface, then the pinned Docker evaluator.

**Contract:** [Document metric reference](docs/benchmarks/extraction/document/metrics.md),
“Table-scorer normalization”; [evaluation setup](docs/benchmarks/extraction/document/methodology.md#evaluation-setup).

### D07 — P1 migration: derive CDM exactness from raw counts

**Locations:** `omnidocbench_worker.py:126`; `metrics.py:438`.

**Evidence:** The pinned evaluator supplies tp, gt_tokens, pred_tokens and rounded
F1_score; the wrapper retains only F1. Counts (1000,1000,1001) round to 1.000 and
are incorrectly treated as exact.

**Fix:** Transport, validate and retain counts/status per formula. Apply the
approved cdm_exact_counts_v1 rule: matched = reference = predicted > 0. Preserve
independently available similarity when exactness is unavailable. Missed/invalid
prediction zero conventions must not invent successful evaluation counts.

**Acceptance:** Rounded near-perfect case is not exact; true equality is exact;
missing/malformed counts, missed predictions and evaluator infrastructure failure
remain distinct. Preserve document-macro aggregation.

**Documentation after D06/D07:** Remove completed-migration wording only after
nonempty official acceptance passes, including Document methodology/metrics,
data-validation evaluator-conformance notes and runbook implementation status.
Historical results retain their original scoring definitions.

## 4. ASR and video extraction

The matching-boundary and telemetry fixes are C01/C02 above; apply them to
standalone ASR and video frozen-ASR consumers.

### AV01 — P2: tolerated decoded audio tails can be silently omitted

**Locations:** `experiments/benchmarks/extraction/video/frozen_asr_worker.py:602`
(`_transcribe_video`) and `:750` (`_window_records`); related validation/reuse in
`video/validation.py` and `video/frozen_asr.py`.

**Evidence:** Manifest duration 30.0 seconds and valid canonical PCM duration
30.05 pass the duration tolerance, but only [0,30] is scheduled. The artifact
records the longer canonical duration and success despite missing real audio.
The current shorter-duration test does not cover this direction.

**Fix:** Build the canonical audio window inventory from the sealed decoded PCM
duration. Keep video/annotation duration distinct. Coordinate scheduled counts,
reviewed nonspeech-window IDs and frozen-artifact reuse; version incompatible
schedule/artifact identities.

**Acceptance:** Shorter/longer tolerated durations, exact/near-30-second boundaries,
complete PCM coverage without padding, consistent failed-window denominators and
rejection of stale reused schedules.

**Contract:** Video “Shared audio preparation,” “Frozen-ASR diagnostics,” and
“Audio Attempt Failure Rate.” Update implementation-status/runbook claims after
verification, not authoritative data-readiness claims.

## 5. Chunking–embedding and retrieval–reranking

Paths below are relative to `experiments/benchmarks/rag/` unless stated otherwise.

### R01 — P1: later retrieval integrity violations can pass candidate validation

**Location:** `retrieval_reranking/benchmark.py:790` (`_attempt`).

**Evidence:** OutputIntegrityError and query InputCompatibilityError can be treated
as ordinary later-attempt execution failures through string/type heuristics.
Mocked shape, zero-norm and input-limit failures returned status=passed and
integrity_complete=true.

**Fix:** Classify explicit typed integrity/input exceptions. Preserve completed
designated-first quality as evidence, but fail integrity when any scheduled
attempt violates its output/input contract.

**Acceptance:** Later-attempt shape, zero-norm, length and numerical violations
cannot pass integrity. Ordinary recoverable runtime failures retain their separate
failure accounting and do not get reclassified merely by message wording.

### R02 — P1: RRF hides invalid source rankings

**Location:** `retrieval_reranking/benchmark.py:1099` (`_first_stage`).

**Evidence:** BM25 scores/cardinality are not validated before fusion discards
scores. A NaN-scored source and a one-row source both produced an accepted finite
20-row fused result.

**Fix:** Validate each dense/lexical source against its frozen cardinality, known
unique positions and finite-score contract before fusion. Validate the fused
output separately.

**Acceptance:** Nonfinite, short, duplicate and out-of-range source outputs are
rejected before fusion; legitimate small-corpus behavior follows the declared
profile contract rather than a fabricated pool.

### R03 — P1: retrieval artifacts cannot independently reconstruct quality ideals

**Locations:** `retrieval_reranking/benchmark.py:630` and `:1368`
(`_index_identity`).

**Evidence:** A 30-chunk fixture retained only its retrieved 20 chunks, not a full
chunk inventory. nDCG/alpha-nDCG denominators depend on all corpus chunks. A hash
cannot reconstruct omitted semantic chunk boundaries. This does not prove the
current in-memory score wrong; it prevents the documented independent rescoring.

**Fix:** Retain one checksummed complete canonical chunk inventory, offsets/order
and reference identities, owned once and referenced by relevant children.
Avoid copying the entire corpus into every reranker artifact.

**Acceptance:** Rescore saved rankings and recompute full-corpus ideals with all
model access disabled, including semantic chunking. Detect stale/changed inventory.

### R04 — P2: interrupted index-build measurements are dropped

**Locations:** `retrieval_reranking/benchmark.py:138` and `:654`.

**Evidence:** index_build starts as an empty dictionary, but recovery only runs
when it is None. Dense completed/BM25 failed produces empty build evidence and
no retained completed storage observation. The helper-only test does not reach
this evaluator wiring.

**Fix:** Correct the sentinel/recovery flow, retaining independently completed
component builds and labelled partial work without promoting an incomplete RRF
build to a completed total.

**Acceptance:** Evaluator-level failure injection after dense and during BM25
preserves completed dense time/bytes, partial BM25 scope and overall incompleteness.
Extend `tests/test_retrieval_reranking_benchmark.py:1184`.

### R05 — P2: malformed reranker output is lost before artifact retention

**Locations:** `methods.py:208`; `retrieval_reranking/benchmark.py:753`.

**Evidence:** Predictor NaN output raises before scores are attached to the
attempt; retrieval output survives but no reranker fragment does.

**Fix:** Attach bounded JSON-safe native scores/shape to a typed output exception
and persist it as unscored diagnostic evidence. Coordinate D01's serialization
principle without treating diagnostic serialization as canonical validation.

**Acceptance:** Mock the actual adapter's predictor to return malformed/nonfinite
scores; verify the resulting persisted attempt retains the rejected fragment and
still fails integrity.

### R06 — P2: embedding execution and reported batch settings can diverge

**Locations:** `evaluation.py:136`;
`retrieval_reranking/benchmark.py:1539`.

**Evidence:** Execution takes the chunking–embedding protocol batch size while
retrieval parameters report the retrieval protocol's value. Individually accepted
custom protocols can specify 1 and 2; defaults currently agree.

**Fix:** Cross-validate composed controls before loading and derive execution,
qualification and logged parameters from the same resolved settings. Preserve
the current single-input contract.

**Acceptance:** Conflicting protocols fail early; matched settings reach the
actual embedder constructor and agree with parameter/fingerprint artifacts.

### R07 — P2: tokenizer initialization can perform an unplanned runtime download

**Location:** `src/edumind/rag/tokenizers.py:27`.

**Evidence:** A .ready marker is trusted before a network-capable tiktoken loader.
With missing cache content mocked, fetching was attempted and intercepted before
network/write activity.

**Fix:** Verify prepared pinned tokenizer assets and fail closed when absent or
corrupt. Require the preparation repair path; do not silently download during
measured execution.

**Acceptance:** A stale marker or corrupt/missing encoding fails with actionable
guidance and no downloader call. Valid prepared operation remains offline.

### R08 — P1 migration: pool recall must use the designated first attempt

**Locations:** `retrieval_reranking/benchmark.py:242` and `:338`.

**Evidence:** The no-reranker owner retains its first successful pool and can score
pool recall from a later attempt after the first fails.

**Fix:** Separate the designated-first quality observation from any later pool
retained for diagnostic/integrity evidence. Never substitute later success for
first-attempt pool recall. Preserve the documented ownership and dependent
execution rules.

**Acceptance:** Failed first/successful later attempts leave first-pool quality
unavailable with the full schedule retained; successful first remains the quality
source. Test both owner and learned-reranker consumers.

### R09 — P2 migration: remove duplicate alpha eligibility scalars

**Locations:** `retrieval_reranking/metrics.py:69` and `:123`.

**Fix:** Reuse the common reference eligibility population rather than duplicating
identical alpha counters. Retain metric-specific contributing, unavailable and
independent-source support, including single-evidence-unit questions.

**Acceptance:** No duplicate reference scalar families; per-metric support remains
correct after failures or unavailable scoring.

### CE01 — P2 migration: put fixed workload and build-throughput evidence in the intended places

**Locations:** `chunking_embedding/benchmark.py:342`, `:361` and `:626`.

**Fix:** Move fixed corpus document/question/source-token counts to parent workload
metadata; keep verified embedding dimension and dtype as child parameters; retain derived
Corpus-Build Throughput only in build artifacts. Keep candidate-dependent
descriptors, actual build/query timings and all completeness/support evidence.

**Acceptance:** Parent inventory matches the validated manifest; dimension matches
the produced matrix; throughput is derivable from retained build evidence but not
emitted as a comparison scalar. Update the obsolete scalar assertion at
`tests/test_chunking_embedding_benchmark.py:1024`.

**Documentation after R01–CE01:** Reconcile CE/RR implementation-status and
reporting-migration notes in their methodology/metrics and runbook sections 1/7/8.
Retain the pending reviewed CI-support requirements. Tokenizer offline guarantees
also affect installation/preparation guidance.

## 6. Generation

Paths below are relative to `experiments/benchmarks/rag/generation/`.

### G01 — P1: rescoring can declare missing execution complete

**Locations:** `evaluate.py:455` (`score_execution`);
`metrics.py:273`, `:403`, `:532`.

**Evidence:** Development rescoring can return result_status=complete and report
seeds [42,43,44] when only seed 42 remains, all records are labelled seed 42,
one question is omitted, or records is empty. The expected schedule is inferred
from survivors and vacuous all() checks succeed.

**Fix:** Validate retained evidence against an independently retained frozen
question/seed schedule. Reject duplicates/unexpected IDs; missing scheduled
slots remain incomplete. Bind recovery to manifest, candidate/model, protocol,
prediction and evaluator identities.

**Acceptance:** All four reproductions fail completeness; valid one-seed smoke
passes; stale or changed evidence cannot be rescored as the original execution.
Do not merely add a nonempty-list check.

### G02 — P2: infrastructure and input failures are counted as generator crashes

**Locations:** `evaluate.py:708` and `:357`.

**Evidence:** WorkerMeasurementError during an in-flight request becomes
runtime_error=1. Prompt-token drift detected before backend inference also becomes
a measured runtime error while lifecycle phase is still finished/error=None.

**Fix:** Preserve typed cause and failure stage. Monitoring/infrastructure
interruptions, input-integrity rejection, actual model errors and skipped work
must remain distinct. A supervisor-wide deadline is not automatically a
per-request generation timeout.

**Acceptance:** Fault-inject each category, retain earlier complete outputs and
timings, account for interrupted/unexecuted slots, and verify Runtime Error Rate
uses its documented eligible attempts.

### G03 — P2: delayed backend return misclassifies context exhaustion

**Location:** `models.py:258`.

**Evidence:** Capacity is exhausted by a token at 0.5s, deadline is 1s, backend
returns at 1.5s: the current elapsed-time branch reports timeout.

**Fix:** Determine the first observed terminal event using token/stop timing,
including capacity exhaustion, rather than final return time alone.

**Acceptance:** Context-before-deadline with delayed return → context_limit;
deadline-before-capacity → timeout; timely EOS in the last slot → normal completion;
backend exception → runtime error.

### G04 — P1 migration: automatic judging and durable evaluation-only recovery

**Locations:** `run.py:241`; `judge.py:158`; `evaluate.py:455`.

**Evidence:** CLI requires already obtained judge replies but still performs
generation. Python rescoring is not a safe recovery command. Judge requests/replies
accumulate in memory rather than being independently checkpointed.

**Fix:** Verify the reviewed judge contract, generate and durably save the complete
schedule, end generator monitoring/release resources, then judge saved outputs.
Provide configured transport without inventing the judge/provider. Journal
request identity, raw replies, retry outcomes and accepted labels durably.
Recovery reuses successful evaluations and scores missing ones without model
loading, replacement seeds or another locked invocation. Implement G01 first.

**Acceptance:** Interrupt judging mid-run, resume without inference, reuse
accepted replies, reject mismatched replay/provenance, retain malformed/exhausted
replies and evaluator failures without changing model outcome.

### G05 — P1 migration: Citation Support labels and metric are absent

**Locations:** `judge.py:21`, `:212`, `:302`; `metrics.py:13`.

**Fix:** Add separately validated cited-subset support/justifications on the same
claim inventory as full-context faithfulness. Resolve only valid supplied
citations; evaluator-only references cannot establish either context-based
support label. Enforce cited-supported ⇒ full-context-supported through the frozen
judge error/retry policy, not clamping. Add diagnostic quality.citation_support
to scoring, slices, completeness, support policy, CIs and artifacts. Preserve the
four approved primary metrics.

Version label/calibration schemas and require distinct full-context/cited-subset
calibration evidence. **Keep development human judge calibration.**
Keep semantic pair comparison in its own rubric, separate from the response
rubric's evidence scopes.

**Acceptance:** Uncited supporting evidence, wrong/unknown/empty citations,
claim-free answers, evaluator-only correct facts and contradictory labels.
Keep claim inventories identical between the support evaluations.

**Documentation after G01–G05:** Generation implementation status, semantic judge
contract, measured lifecycle, metric reference, runbook judge commands/recovery,
and shared validation availability. Calibration and data-review notes remain
pending until actual reviewed evidence exists.

## 7. Final RAG

### F01 — P1: current Final RAG fails before integrated execution

**Locations:** `experiments/benchmarks/rag/final/run.py:282` and `:332`;
`experiments/benchmarks/rag/generation/evaluate.py:796`.

**Evidence:** Mocked CPU smoke raises AttributeError for the removed
GenerationProtocol.faithfulness_model. Removing that access alone does not fix
it: the generation evaluator explicitly rejects the final_index integration that
Final passes.

**Fix:** Implement F02, not a compatibility restoration of the retired classifier
or a bypass of the integration rejection.

**Acceptance:** Mocked smoke must execute the selected integrated server path and
persist request evidence. Import/--help tests are not execution tests.

### F02 — P1 migration: replace the legacy finalist workflow with one frozen system

**Locations:** `experiments/benchmarks/rag/final/run.py:78`, `:185`, `:237`,
`:257`, `:317`, `:358`, `:401`; adjacent `protocol.py` and `protocol.yaml`.

Implement the current [Final methodology](docs/benchmarks/rag/final/methodology.md)
and [metrics](docs/benchmarks/rag/final/metrics.md), specifically:

1. Resolve one reviewed final-rag-locked.json composition for smoke, preflight and
   locked: selected chunking/embedding, retrieval/reranker, database/configuration,
   generator/mode, one top-K of 3 or 5, prompts/packing/decoder and residency policy.
   Remove Final development/validation selection and finalist Cartesian products.
2. Provide separate CPU/CUDA smoke and integrated CUDA preflight on nonlocked
   stress inputs. Smoke has one warmup and one measured attempt per question;
   preflight has zero warmups. Require matching integrated qualification for locked.
3. Prepare the phase corpus/index outside online timing; query the selected
   server and frozen strategy live. Do not substitute ExactIndex or cached
   quality pools. Validate actual pool identity, filters, permutation and prefix,
   while permitting valid ANN variation.
4. Retain request-local citation labels mapped to stable corpus/chunk/offset/content
   identities. Standalone generation's sufficient-gold-context validator cannot
   be reused unchanged: Final must allow retrieval misses and must not inject gold.
5. Use the first measured ranking for retrieval quality and all three scheduled
   generation seed outputs for answer quality/pairs. Compare stable citation
   identities, not literal E1/E2 labels across varying retrieved contexts.
6. Keep request failure separate from generation eligibility/skips. Preserve
   completed upstream stages and timings. Measure server calls, retrieval,
   packing/residency-switching-inclusive generation and total request on one live
   path; retain generator-call TTFT separately.
7. Reserve locked consumption atomically before held-out execution. The current
   marker written only after result.complete allows failed inference to be rerun.
   Recovery must be evaluation-only, preserving the original prediction identity.
8. Implement Final standalone validation/sealed report reuse, including separate
   nonlocked preflight inputs. Compose all applicable protocols including the
   selected vector configuration; report one parent and one selected-system child.
9. Export the complete scheduled request/attempt evidence, statuses, four primary
   metrics, diagnostic Citation Support and equivalent CSV/Parquet tables.

**Acceptance:** Stale composition/qualification, duplicate locked invocation,
first-ranking failure/later success, upstream skip versus model exception,
valid ANN variation, changed local labels, missing stable citation mapping,
preparation/late-stage failure, preserved earlier measurements, and judging-only
recovery. Use mocked integration until reviewed data and real services are ready.

### F03 — Cleanup: retire the obsolete human-review and extraction-confirmation paths

**Locations:** `experiments/benchmarks/review.py`;
`experiments/benchmarks/rag/final/review.py` and `confirm_extraction.py`;
`experiments/benchmarks/common/tracking.py:38`.

These are reachable legacy CLIs, not dead because of missing imports.
confirm_extraction also references the removed classifier and rejected integration.
Remove their Final prerequisites, obsolete protocol fields, CLI dispatch and
entry-point test cases together after replacing needed consumers.
The extraction-confirmation routing alias is only needed if an explicitly
supported historical reader still consumes it; do not rewrite historical runs.
**This removal does not apply to human calibration of the LLM judge.**

Replace tests accepting Final top-K [2,4] and development/validation profiles
(`tests/test_benchmark_protocols.py:257`, `:294`) with the approved single-system
contract. Remove obsolete review/confirmation --help entries from
`tests/test_benchmark_completion.py:131` and `:139`.

## 8. Vector Database

Paths below are relative to `experiments/benchmarks/vectordb/`.
The existing docs already acknowledge a substantial migration; complete V10
without carrying the concrete defects V01–V09 into the replacement.

### V01 — P1: failed requests inflate QPS

**Locations:** `run.py:638` (`_concurrent` at `:662`);
`retrieval_run.py:303`.

**Evidence:** Submitted requests, including failures, form the numerator.
One failed request over two seconds reports 0.5 QPS and error rate 1.

**Fix:** Count execution-success completions after response validation over the
actual dispatch-to-drain interval. Preserve failure counts separately. A truly
zero-duration window is unavailable, not divided by epsilon.

**Acceptance:** All failures/positive time → zero QPS; mixed outcomes use successes
only; malformed responses do not succeed. A checkable known-ID filter violation
remains an execution success with separate filter-quality failure.

### V02 — P1: response integrity and filtering trust invalid backend output

**Locations:** `run.py:567`, `:663`; `retrieval_run.py:213`;
`adapters/weaviate.py:112`.

**Evidence:** A forbidden known ID with forged echoed metadata can pass filter
correctness. Duplicate unknown IDs with NaN scores are scored. Load searches
discard returned hits. Missing Weaviate distance becomes similarity 1 through
distance-or-zero. Integration does not verify complete reranker permutation.

**Fix:** Validate responses against trusted corpus identity/metadata, unique known
IDs and required finite scores. Preserve separate execution, filtering and quality
outcomes. Verify reranker output against its actual observed pool.

**Acceptance:** Forged metadata, duplicates, unknown IDs, missing/nonfinite scores,
dropped/duplicated reranker positions and short pools. Preserve the documented
distinction between a valid short ANN response and an integration pool too short
for its downstream contract; empty eligible filters stay inapplicable for recall.

### V03 — P1: configuration selection hides failures and cannot recover valid selections

**Locations:** `run.py:464` (`_select_config`);
`retrieval_run.py:106` and `:373`; `protocol.yaml:9`.

**Evidence:** Almost any configuration-trial exception becomes “unsupported,”
discarding its cause. Integration then infers the selected configuration from
dimension substrings in workload names. Default development has 384/1024 dimensions,
so a valid 768-dimensional selected embedding cannot recover its configuration.

**Fix:** Consume explicit database–configuration identities/settings, with
engineer-owned advancement. Distinguish unsupported settings from infrastructure
or execution errors and retain every trial outcome. Do not infer settings from
workload-name substrings or arbitrary first matches.

**Acceptance:** Timeout/outage cannot become unsupported; attributable unsupported
setting can. Valid 384/768/1024 corpora consume the selected pair, or fail a real
compatibility check before runtime work.

### V04 — P2: protocol accepts invalid filters and duplicate workload identities

**Locations:** `protocol.py:116`, `:231`, `:267`;
`run.py:555` and `:202`.

**Evidence:** An unknown development filter is accepted then crashes on missing
metadata. Duplicate workload names create repeated sample IDs and overwrite
name-keyed configuration/results.

**Fix/acceptance:** Validate unique condition identities and supported trusted
filter fields during protocol loading. Reject both cases before server calls;
preserve legitimate source-ID filtering.

### V05 — P2: readiness checks do not prove the requested index/corpus identity

**Locations:** `adapters/qdrant.py:135`, `pgvector.py:122`,
`chroma.py:112`, `weaviate.py:132`.

**Evidence:** Qdrant accepts green status/counts without verifying requested
dimension/metric/HNSW settings or expected corpus count. PostgreSQL checks HNSW
presence and a forced sample plan, not complete configuration identity; it also
ignores Config.collection in favor of a fixed table. Other adapters verify settings
but still need expected corpus visibility checks.

**Fix:** Verify backend-specific effective settings and trusted expected inventory,
record native plans without inventing an HNSW-only eligibility rule, and represent
unverifiable checks explicitly. Honor or reject unsupported collection overrides.

**Acceptance:** Wrong settings/dimensions, missing/wrong corpus, unready index and
alternate collection name. Preserve optimizer-resume and existing configuration
verification tests.

### V06 — P2: correctness probes can pass on unrelated failures or unchanged counts

**Location:** `conformance.py:58`, `:91`, `:131`.

**Evidence:** ConnectionError can count as dimension rejection; no-op replacement
passes if count is unchanged; same-count corruption can pass persistence.
Adapter-side rejection is also not automatically proof of server rejection.

**Fix:** Require attributable error categories and verify IDs/vectors/text/metadata
with controlled fixtures. Distinguish passed/failed/unverified and adapter/server
scope.

**Acceptance:** No-op replacement, equal-count corruption and outages cannot pass.
Genuine rejection/replacement/persistence still pass.

### V07 — P2: integration runs unused retrieval branches and pools wrong latency observations

**Location:** `retrieval_run.py:126`, `:209`, `:264`.

**Evidence:** Dense embedding/search and BM25 both execute regardless of selected
strategy. BM25 becomes dependent on vector-server availability; dense includes
irrelevant work. Latency summaries pool attempts despite calculating query medians.

**Fix:** Execute only sources required by the frozen strategy. Full-stack latency
includes its actual required work/validation and aggregates successful
per-question medians.

**Acceptance:** BM25 succeeds with failing dense/embedder stubs; dense never calls
BM25; RRF calls both. Unequal attempt durations distinguish pooled and
median-per-query percentiles.

### V08 — P2: storage fallback changes units and checkpoint semantics

**Location:** `docker_metrics.py:121` and `:153`; `run.py:235`.

**Evidence:** du -sb measures apparent bytes; fallback du -sk measures allocated
blocks. Storage is measured after workloads/mutations rather than at each
completed-build checkpoint.

**Fix:** Use one apparent-byte definition and synchronized post-build checkpoint.
Retain later/partial/unavailable observations with their actual scope.

**Acceptance:** A sparse-file fixture distinguishes apparent from allocated size;
fallbacks agree on quantity; checkpoint precedes measured queries/mutations.

### V09 — P2: deployment/transport provenance is only partially enforced

**Locations:** `docker_metrics.py:218`; `adapters/qdrant.py:18` and
`adapters/weaviate.py:18`.

**Evidence:** image_lock reports recorded client versions without comparing
installed distributions and accepts unvalidated digest strings. Local versions
currently match; no current mismatch is alleged. connection_pool_limit is enforced
by Chroma/PostgreSQL but not supplied to Qdrant/Weaviate clients.

**Fix:** Validate deployment-lock schema/digests and actual client identities.
Resolve and record backend-specific supported transport limits; do not claim
unused Config fields control execution.

**Acceptance:** Missing/changed clients, malformed digests and changed transport
settings cannot silently reuse a verified configuration identity.

### V10 — P1 migration: implement the approved vector evaluation contract

**Locations:** `run.py`, `retrieval_run.py`, `workload.py:97`,
`protocol.py`/`protocol.yaml`; shared `rag/evaluation.py:595`.

Use the [vector methodology](docs/benchmarks/vectordb/methodology.md) and
[metric reference](docs/benchmarks/vectordb/metrics.md) as the complete contract:

| Area | Required replacement |
|---|---|
| Candidates/lifecycle | Declared database–configuration children; frozen development/validation decisions and one reporting-only locked pair; no hidden validation retuning |
| ANN/filtered ANN | Tie-aware @20 strict-neighbour plus boundary-slot credit; effective eligible denominator; trusted filters; no perfect recall for an absent task |
| Evidence quality | Shared canonical evidence-unit/token scoring at @3/@5, full eligible-corpus ideal, candidate-pool recall @20; retire the old overlapping retrieval scorer after consumers migrate |
| Repeatability | Scheduled pre-rerank database all-pairs ordering diagnostic, not all-equal reranked outputs; retain failed/missing attempts; smoke inapplicable |
| Reliability | Search/insertion/update/deletion request-error rates from logical operation ledgers; separate filter-violation denominator |
| Search/load | Frozen conditions/windows, appropriate warmups, retained validated responses, designated-first quality and successful-request timing/QPS |
| Build | Required setup through verified inventory/index readiness, not only post-reset insertion |
| Mutations | Document-level insertion/replacement/deletion latency, visibility barriers, expected outcomes and restored baseline; not just incremental-record throughput |
| Startup | Already-stopped preserved benchmark corpus to verified required query; exclude shutdown and tiny unrelated conformance-corpus timing |
| Resources/storage | Correct declared windows/checkpoints, timestamped server samples and coverage/gap evidence; exclude tuning/oracle computation from candidate windows |
| Data identity | Standalone validator/sealed report; hash actual vectors, queries, canonical IDs/text/metadata and reviewed provenance, not just counts/selections; no synthetic metadata masquerading as real filters |
| Aggregation/CIs | Separate workload conditions; query ANN means, source-document evidence macro, declared load-window/operation support, explicit unavailable bounds; coordinate C04 |
| Artifacts | Complete condition, request/operation, ranking, status, configuration and comparison evidence; equivalent CSV/Parquet where required |

The current tie oracle deterministically chooses IDs, but equivalent boundary
alternatives can receive zero recall. Keep deterministic ordering as a useful
property; do not use it as the tie-aware scoring definition.

**Acceptance:** Documented boundary-tie example, excess boundary alternatives
unable to replace missing strictly closer IDs, empty/short filtered sets,
failed-first/later-success cases, incomplete schedules, all-failure load windows,
logical-operation visibility, separate conditions and selected-server integration.
Freeze data-dependent choices through review; do not invent them during migration.

## 9. Application issues found while tracing benchmark consumers

These are application fixes, not benchmark metric changes or automatic promotion.

### A01 — P1: model integrity is missing from application loading/cache identity

**Locations:** `src/edumind/common/models.py:43`;
`src/edumind/extraction/pipeline.py:275`; `src/edumind/extraction/cache.py:20`.

**Evidence:** require_model accepts an existing directory without verifying recorded
checksums. Resolved extraction options carry paths rather than verified artifact
identity; different digests at the same path/revision yield the same cache key.

**Fix:** Share structural/integrity checks with preparation without importing
benchmark eligibility into production. Include verified bundle identity in
extraction provenance/cache keys.

**Acceptance:** Tampered bytes fail before inference/cache reuse; an intentionally
changed verified bundle invalidates the cache. Coordinate P04/P05.

### A02 — P1: indexing drops extraction source-location provenance

**Locations:** `src/edumind/rag/pipeline.py:301`;
`src/edumind/rag/text_chunker.py:85`; `src/edumind/ui/streamlit_app.py:119`.

**Evidence:** Normalization retains document text/metadata but discards segments.
A two-page fixture produces chunks without page/timestamp locations.

**Fix:** Retain segment spans and project overlapping location/structure provenance
onto chunks, including multi-segment/page chunks. Context truncation must retain
truthful provenance for the displayed span.

**Acceptance:** PDF and timed-media provenance survives ingestion, retrieval,
context construction and UI display; offsets still identify exact source text.

### A03 — P2: failed uploads cannot be retried and deduplication ignores logical identity

**Locations:** `src/edumind/ui/controller.py:40`;
`src/edumind/ui/streamlit_app.py:75`.

**Evidence:** Any recorded checksum is skipped, including FAILED records; UI says
already indexed. Replacing document B with bytes already recorded for A also skips
B's required replacement.

**Fix:** Deduplicate successful ingestion by logical document and content/version;
allow explicit retries after failures and accurately label outcomes.

**Acceptance:** Failure then retry succeeds, a successful same-document duplicate
is skipped, and replacing B with A's bytes does not leave stale B chunks.

### A04 — P2: production Whisper can silently lose or accept malformed text

**Locations:** `src/edumind/extraction/extractors/audio.py:94` and `:170`.

**Evidence:** Native text "alpha beta" with only an "alpha" chunk becomes "alpha";
a non-object native response becomes a clean empty transcript.

**Fix:** Validate native schema and lexical completeness. Preserve unmatched
transcript text or explicit recoverable diagnostics rather than silently
discarding it. Keep production's intentional handling of missing time boundaries;
do not blindly impose the stricter benchmark timestamp contract.

**Acceptance:** Partial/malformed chunks, conflicting transcript, non-object output,
legitimate silence and the existing untimed-tail preservation test.

### A05 — P2: production video discards visual warnings and structure

**Location:** `src/edumind/extraction/extractors/video.py:150` and `:192`.

**Evidence:** Frame documents become plain text immediately; a visual heading's
partial-conversion warning disappears, and audio/visual segments lose their
distinct kinds/native structure during composition.

**Fix:** Retain frame documents through composition, propagate contextual warnings,
preserve modality/native provenance and rebuild exact text offsets.

**Acceptance:** Warned partial output stays warned; audio/visual identity and
metadata survive; every segment offset still slices its text correctly.

### A06 — P1: failed document replacement can lose the previous index contents

**Location:** `src/edumind/rag/vector_store.py:102` and `:116`.

**Evidence:** Normal writes batch, but rollback restores all previous records in
one request. A mocked 6,000-record document against a 5,461-record server limit
cannot be restored after replacement failure; the restore error masks the original.

**Fix:** Batch both writes and rollback against actual supported limits, preserve
IDs/text/metadata/NumPy embeddings and report both original/restoration failure.

**Acceptance:** Failure in a later batch restores the complete old document.
Separate restoration-failure tests retain both causes. Extend the existing NumPy
rollback test rather than deleting it.

### A07 — P2: a production streamer timeout leaves inference running

**Location:** `src/edumind/rag/llm_generator.py:165`.

**Evidence:** A streamer queue.Empty exits before worker.join(); the background
model thread remains active with no cancellation/finally ownership boundary.

**Fix:** Stop or isolate inference on timeout/error and complete cleanup before
releasing lifecycle ownership. A daemon thread alone is not cancellation.

**Acceptance:** Stalled generation/streamer timeout leaves no orphan inference;
subsequent unload/request cannot overlap it; original backend errors survive.

### A08 — P2: reasoning-only/interrupted generation can be exposed as the answer

**Location:** `src/edumind/rag/llm_generator.py:279`.

**Evidence:** A closed reasoning-only stream is restored as visible text by the
empty-visible fallback. Prompt-opened or unclosed reasoning is not safely separated.

**Fix:** Use model/template-aware reasoning boundaries. Never fall back to the
reasoning body when no visible answer was generated.

**Acceptance:** Closed reasoning-only, prompt-opened reasoning, interruption and
normal direct-answer cases. Preserve the documented distinction between provisional
application generation and benchmark generation.

**Application documentation:** Reconcile verified behavior with
`docs/architecture/overview.md`, `extraction.md`, `rag.md`, `application.md` and
`ui.md` as applicable. Do not claim production equivalence to a selected benchmark
winner merely because these bugs are fixed.

## 10. Justified cleanup and test changes

| Target | Action and reason |
|---|---|
| Final review/confirmation CLIs and fields | Retire together under F03; they are obsolete but currently reachable |
| Legacy vector/RAG scoring and configuration inference | Replace consumers under V10/F02 before deleting old helpers |
| Generic aggregation fallback | Resolve remaining consumers under C04, then delete only if unused |
| Duplicate vector Config construction | Consolidate run._config and retrieval_run._config around the explicit selected configuration |
| Duplicate vector load accounting loops | Share request/outcome accounting while preserving database-only versus full-stack measurement boundaries |
| generation/metrics.py:536 | Compute question_statistics once per record rather than twice |
| retrieval_reranking/worker.py:7 | Remove unread worker_started_at plumbing and its wrapper, not the subprocess entry-point module |
| rag/evaluation.py:373, rank | No repository execution route was found through imports, CLIs, registries or tests; remove after rechecking callers, without confusing it with the live Reranker.rank method |
| retrieval_reranking/run.py:107, :143, :458 | Consolidate repeated annotation/report work; keep full standalone validation distinct from lightweight sealed-report verification |
| CE _interval / RR source_interval | Share genuinely identical resampling primitives only if grouping, status, support and estimators remain explicit; avoid a new universal statistics framework |
| test_video_benchmark.py:328 | Replace absence-of-two-source-strings check with a behavioral visual-worker test that fails if ASR loading/transcription occurs |
| test_video_benchmark.py:318 | Replace retired phase="hybrid" fixture with supported development phase; retain missing-threshold rejection |
| Document post-inference scope test | Correct the obsolete “any failure means partial window” assertion under C01 |
| Official worker import test | Empty payload does not import either scorer; supplement with nonempty interface/acceptance coverage |
| Final protocol/entry-point tests | Replace retired positive contracts with smoke/preflight/locked single-system execution tests |
| Selection/preparation tests | Add app-model preparation, dry-run, revision fallback and partial-lock-migration behavior; roster-string assertions alone miss these failures |

Do not delete worker modules or minimal run.py entry points because they lack
ordinary imports. The following are live: fresh-process workers, the
Docker-mounted OmniDocBench worker, registry factories, image-pipeline
initialization hooks, all four vector adapters, PostgreSQL finish_index via
dynamic dispatch, and official evaluator startup checks.

Keep negative-path tests for bad inputs, checksum tampering, pool permutation,
model placement, rollback, optimizer readiness, transport cleanup and interrupted
execution. A large test file is not evidence that its tests are useless.
Strengthen helper-only interrupted-build, one-query-per-document bootstrap,
empty-comparison-hook and row-count-only serialization tests with evaluator-level,
unequal-group and CSV/Parquet content-equality checks; passing a shallow test is
not a reason to delete its meaningful underlying contract.
Production measurement hooks without tracked callers require an external-API
check before deletion; they must not be reused as benchmark token-event timing.

## 11. Items requiring confirmation, not automatic changes

- Video's selected scene-threshold source-run ID is checked syntactically but not
  resolved against the preliminary comparison. The docs prescribe a manual
  protocol edit; confirm whether machine validation is required before adding
  another decision workflow or rejecting the intended protocol-version change.
- Document hierarchy-level consistency and multi-page Docling element provenance
  need their precise annotation/segmentation meaning confirmed before stricter
  validation or splitting. Do not invent a universal hierarchy-depth rule.
- Verify Weaviate exact-string filter semantics and large-document delete behavior
  against the pinned live server; neither is established as a runtime defect here.
- Confirm the public production source-ID filter vocabulary before translating
  source_id to the store's internal __source_id. Also clarify overlapping
  replacement concurrency requirements before adding locks/transactions.
- Extraction readiness currently checks preparation rather than all native
  binaries/providers. Clarify that API's guarantee before broadening it.
- Ruff is documented as a development check but has no corresponding dependency
  pin. Address through the existing developer dependency policy if reproducible
  tooling is intended. Do not restore the deliberately removed CI workflow merely
  because it is absent.

## 12. Data/environment boundaries

At audit time, the authoritative extraction/RAG manifests and reviewed decisions
needed for complete comparisons were absent. Generation judge identity/calibration
and reviewed independent-source CI support remain pending. The local model lock
does not include the application generator or the approved generation checkpoints.
Video scene selection and nonspeech allocation still require reviewed evidence.

These are not requests to fabricate data or choose winners. Code/schema/mocked
workflow work can proceed; real authoritative runs require the appropriate inputs,
snapshots, services, GPU/runtime support and permission.

The 17 locally prepared primary snapshot sizes matched the current size helper.
No snapshot-size bug was found. This is file inventory size, not model VRAM or
deployment memory. A complete real-model integrity/inference sweep was not run.

## 13. Documentation completion checklist for the fixing pass

Use this matrix to update accurate status statements after verification, not to
erase them before the implementation exists.

| Completed work | Documentation to revisit |
|---|---|
| Shared evidence/resources/schema/statistics | Shared methodology/metrics; runbook; relevant local implementation claims |
| Preparation/provenance | Installation, datasets, data-validation identity guarantees, architecture model provenance |
| TEDS/CDM migration | Document methodology Evaluation setup; Document metrics normalization/ExpRate notes; evaluator validation guidance; runbook implementation status |
| ASR/video defects | Their methodology implementation summaries, applicable resource/audio-schedule explanations and runbook |
| Chunking–embedding reporting migration | Local methodology/metrics target-reporting notes; runbook implementation status |
| Retrieval pool/eligibility migration | Local methodology/metrics pending-reporting notes; runbook implementation status |
| Generation judge/Citation Support/recovery | Local implementation status, judge contract, metric reference, recovery commands and shared data-validation availability |
| Vector migration | Local implementation status/metric boundary, runbook diagnostic versus target commands, data-validation planned validator, datasets, model-selection/setup references |
| Final migration | Local methodology/metrics, runbook, shared Final readiness references, planned validator notes, README/docs index where applicable |
| Application corrections | Applicable architecture/UI/setup behavior descriptions |

Search at the end for “not implemented,” “not yet,” “planned,” “pending,”
“legacy,” “implementation status,” “target keys,” and “migration.” Resolve each
hit by meaning, not mass deletion. Keep data/calibration/hardware blockers and
historical limitations that remain true. Add a concise changelog entry for
verified behavior changes; do not rewrite historical audit/run claims.

## 14. Implementation order and completion gates

1. Preparation/integrity fixes and shared evidence/schema/numeric fixes.
2. Document scorer migrations and extraction regressions.
3. Chunking–embedding/retrieval reporting and correctness fixes.
4. Generation schedule validation, failure classification, judge/recovery and
   Citation Support.
5. Vector configuration/response/accounting migration and adapter verification.
6. Final single-system integration using the corrected upstream interfaces.
7. Application corrections, coordinated with shared model/storage/runtime work.
8. Remove retired code/tests, reconcile documentation status, then repeat audits.

For each finding, record: fixed files/symbols, regression result, artifact/schema
impact, documentation update, and any remaining real-runtime verification.

Run at minimum from the prepared environment:

~~~powershell
python -m ruff check src experiments tests --no-cache
python -m ruff format --check src experiments tests --no-cache
python -m pytest -q
~~~

Also perform:

- syntax/compilation checks without altering tracked files;
- mocked complete CLI workflows, not only --help/imports;
- temporary MLflow parent/child workflows validating null/status/support evidence,
  failure retention, exact provenance and equivalent CSV/Parquet content;
- real official nonempty TEDS/TEDS-S/CDM acceptance when Docker is available;
- controlled vector-service checks for visibility, index settings, filtering,
  mutations, restart and storage when the pinned services are available;
- available real model smoke/qualification only with the correct prepared
  snapshots and reviewed inputs, no fallback or fabricated success;
- call-site searches after every removal, including -m launches, registries and
  runtime hooks;
- final source-versus-documentation trace for metric eligibility, direction,
  aggregation, attempt schedule, boundaries, failure status and artifact fields.

An implementation can be unit-tested yet still awaiting real inference,
calibration or data review. Report those states separately. This audit is a
verified set of findings at the recorded baseline, not a guarantee that no
additional issue can be discovered during repair.
