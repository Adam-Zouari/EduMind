# Final RAG methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Purpose and scope](#purpose-and-scope)
- [Frozen system](#frozen-system)
- [Data](#data)
- [Metrics and their purpose](#metrics-and-their-purpose)
- [Evaluation setup](#evaluation-setup)
- [Benchmark workflow and selection](#benchmark-workflow-and-selection)
- [Execution and measurements](#execution-and-measurements)
- [Results and MLflow structure](#results-and-mlflow-structure)

How well does the already-selected RAG stack answer questions over a prepared,
reviewed canonical corpus?

## Purpose and scope

This benchmark reports one already-selected complete RAG system. It does not
compare new component combinations or reopen component selection. Questions
supply retrieval and generation observations, source documents define quality
aggregation, and reviewed independent-source groups define uncertainty support.
The calibrated LLM judge supplies automated semantic evaluation, as in the
Generation benchmark. Measured requests begin with frozen chunks, embeddings,
and indexes ready. Document, ASR, and video extraction are outside measured scope
and quality evaluation. This is RAG over prepared content, not raw-upload-to-answer
application performance; no additional extraction integration benchmark is
required by this contract.

### Implementation status

This page defines the approved single-system design: CPU/CUDA smoke, integrated
preflight, and one locked evaluation. The current Final RAG runner/protocol expose
smoke/development/validation/locked profiles, omit integrated preflight, and cross
component finalists/top-K values into multiple systems. Their smoke path does not
yet establish the complete selected-server workflow described here. The current
locked command also requires a legacy human-review artifact; removing that
dependency and its obsolete review workflow is pending implementation. Those
paths must not be presented as evidence of this approved single-system contract.

Judge calibration is pending as described in the
[generation methodology](../generation/methodology.md#semantic-judge-contract).
Final RAG's standalone data-validator/report-reuse interface is also pending.
A matching server-backed execution trace, required metric coverage, and
retained evaluation artifacts must be verified before claiming the complete
contract is implemented. No authoritative manifest, winner, judge identity,
or calibration result is supplied by these examples.
The documented design does not authorize the existing multi-system workflow as
another Final RAG selection stage.

Citation Support, stage-eligible failure accounting, stable citation identities,
the attempt-table exports, and the live timing boundaries below are approved
requirements, not claims of implemented behavior. Commands and their current
limitations belong to the [runbook](../../running.md); this documentation change
does not modify the runner or protocol.

## Frozen system

Final RAG uses `smoke` and `preflight` for technical readiness, followed by
`locked` for authoritative quality reporting. It has no development or validation
selection stage. All three use the same frozen composition. The engineer records
one complete system
assembled from the selected chunking–embedding pair, retrieval–reranking stack,
vector-server profile, and generator model-mode configuration. Top-K is one
frozen value, either `3` or `5`,
chosen from component development and validation evidence before any applicable
locked evaluation. Offline annotation review is separate from candidate selection.
Freeze component/model identities and revisions, database/index configuration,
prompts, context packing, decoding, refusal policy, and model-loading/residency
behavior. Required per-request switching/loading belongs to measured requests;
it cannot be silently treated as one-time startup.

## Data

Validate the held-out inputs and frozen composition's requirements through the
[shared data-validation workflow](../../data-validation.md) before execution.
The proposed standalone/report interface remains pending implementation. Final
RAG uses tiny smoke fixtures, separate reviewed development/stress inputs for
integrated preflight, and untouched locked evaluation inputs. Data validation
does not introduce another candidate comparison or authorize reviewing held-out
answers during selection.

The locked profile uses an untouched held-out manifest over a frozen, reviewed
canonical corpus. Retain source identities, text checksums, reviewed evidence
offsets, required claims, answerability, and independent-source groups. Prepared
chunks, embeddings, and indexes derive only from that version of the corpus.
Smoke and preflight never use the locked questions or their held-out answers.
Their prepared indexes derive from their respective input corpus; reuse means
the same selected index configuration, not an index from the wrong dataset.

## Metrics and their purpose

The following quality and uncertainty contract applies to the locked evaluation.
Smoke supplies wiring evidence; preflight supplies integrated hardware evidence.
Neither supplies quality-selection evidence or authoritative confidence intervals.

For `top_k=3`, retrieval quality is reported at 3. For `top_k=5`, it is reported
at 3 and 5.

Final RAG applies the [shared retrieval evidence definitions](../../metrics.md#retrieval-evidence-quality)
to nDCG, Evidence-unit Recall, and Evidence-token Precision at the available
cutoff, plus alpha-nDCG and Candidate-pool Evidence-unit Recall@20. All retrieval
quality is diagnostic in Final RAG, using the designated first attempt; it is
not a second vector-database comparison. The four primary answer-quality metrics
remain Faithfulness, Factual Correctness F1, Answer Relevancy, and Citation F1,
using all three scheduled generation outputs. Citation Support is diagnostic.
Final RAG also reports behavioral validity, reliability, and
repeatability diagnostics from the
[Generation benchmark](../generation/metrics.md), plus retrieval, generation,
server-call, and complete end-to-end p50/p95 latency. RAM and VRAM remain
operational measurements.

End-to-End Request Failure Rate records failed complete requests. Generator
Runtime Error, Timeout, and Context-Limit-Reached rates retain distinct events
and generation-stage eligibility, so a skip after retrieval failure is not a
generator crash. Roles add no automatic winner, qualification gate, or weighted
overall score. See the [failure denominators](metrics.md#request-failures-and-generation-stage-eligibility).

## Evaluation setup

### Output and evidence trace

Reuse the [generation response schema](../generation/methodology.md#response-schema)
without adding model-visible gold answers or evaluator-only source material.
Number only the retrieved blocks and retain the mapping from each ID to chunk
ID, rank, source document, exact offsets, content identity, and covered gold units.
For [citation repeatability](metrics.md#citation-repeatability-on-retrieved-evidence),
resolve known request-local labels to stable chunk/source-content identities
before comparing outputs; unknown-label sentinels remain in a separate namespace.
A response-level citation array does not assert a separately emitted
claim-to-citation map.

For example, a question asks for enrollment and duration. Retrieved `E1` states
"The trial enrolled 500 participants" and `E2` states "Follow-up lasted six
months." A schema-valid example is:

```json
{
  "status": "answered",
  "answer": "The trial enrolled 500 participants and follow-up lasted six months.",
  "citations": ["E1", "E2"]
}
```

Legal IDs alone do not establish correctness or semantic citation support. If
retrieval misses the duration, that required fact/unit remains in the evaluator's
denominator; do not insert its gold evidence to rescue generation. Refusal uses
the exact frozen generation refusal object, not an invented runtime-error status.
Reasoning is separate from the visible answer where exposed. Failed retrieval,
unexecuted generation, malformed response, and judge failure retain distinct
outcomes and null/status rules from [Final RAG metrics](metrics.md#scoring-inputs-eligibility-and-result-status).

Reuse the [single structured response rubric and separate pairwise rubric](../generation/methodology.md#semantic-judge-contract).
Faithfulness uses all supplied context; Citation Support uses only combined
valid cited passages and the same deduplicated atomic claim inventory. Neither
support label may be rescued with gold text. The judge identity and calibration
remain pending review and must be frozen before authoritative evaluation.

### Retrieval integrity and frozen-index checks

Keep the prepared index fixed across measured attempts. ANN is approximate,
not necessarily nondeterministic; test its observed stability on nonlocked
inputs without rebuilding the index between searches. Rebuild variability is
a different task, not three new ANN-quality evaluations in Final RAG.

Follow the [database pool-integrity contract](../../vectordb/methodology.md#pool-identity-and-integrity)
for server-backed search. Validate IDs, contents/provenance, filter eligibility,
pool size, and required finite scores. A reranker must permute exactly the actual
verified pool for that attempt; selected top-K must be its ordered prefix.
A valid ANN pool may differ from the exact-search reference or another attempt.
Do not transplant the exact-retrieval live-pool-equality gate into this path.
Integrity checks are not a requirement for perfect ranking repeatability.

## Benchmark workflow and selection

`final-rag-locked.json` identifies this composition and its upstream reviewed
decisions. Smoke, integrated preflight, and locked execution all resolve that
same system decision; advancing does not require new finalist files. Final RAG
does not cross component finalists, tune settings, or select a winner. Readiness
checks occur before the one locked benchmark invocation. That invocation includes
the frozen measured repetitions and generation seeds; those are not additional
selection rounds.

The sequence is:

```text
component development/validation decisions -> freeze one complete composition
-> smoke-cpu -> smoke-cuda -> integrated preflight
-> one locked Final RAG invocation -> automated evaluation and report
```

Smoke checks loading, interfaces, evidence offsets, and the complete request
path on both devices without fallback. Integrated preflight checks the actual
selected server, retrieval/reranking, packing, and generator lifecycle on CUDA,
including model residency transitions. Separate component qualification does
not guarantee that this composition fits or runs correctly together.

The component decisions, not readiness checks, establish the
composition. Failed or unverified required readiness blocks locked execution.
Do not silently reduce top-K, truncate context, offload, or substitute a model
to make the run fit. Any intentional correction creates a new version of the
affected configuration and repeats the applicable readiness checks. Qualification
reuse requires the exact composition, protocols, model/software/hardware
identities, and supported input envelope to match the
[shared preflight contract](../../methodology.md#preflight-and-measured-run-lifecycles).

If component and Final RAG reports share locked data, freeze and verify this
composition before the first applicable component locked report. Otherwise use
a separate untouched Final RAG holdout. Readiness is not permission to tune from
locked results or create another finalist competition.

## Execution and measurements

### Readiness execution

The target `smoke` command runs separate `smoke-cpu` and `smoke-cuda` executions
of the selected system on the same fixtures. Each initializes the stack, runs
one warmup, and records one measured attempt per question. A device-specific
debug run does not establish that the other device works. These smoke records
test the pipeline and reporting, not the selected system's quality on real data.

Integrated preflight starts process-tree and device monitoring before model
construction, inspects placement, runs the reviewed demanding complete request
workload without a warmup, inspects placement again, and retains its qualification
report. Test the frozen residency/switching policy, not an artificial requirement
to keep all models loaded together. Include the server and selected CPU work,
such as BM25, while GPU
model stages retain their frozen placement and resource requirements. Normal
CPU database execution and tokenization are not model-weight offloading. Offline
LLM judging is outside this measured path. The reviewed stress envelope and
runtime requirements must be frozen before qualification; do not infer coverage
of larger inputs from a small smoke success.

### Locked execution

Prepare the corpus and indexes once as recorded preparation work, separate from warm requests:

```text
frozen reviewed canonical corpus -> selected chunks/embeddings -> required indexes

for every answerable AND unanswerable question and each measured attempt:
query Dense through the approved server, BM25, or both for RRF
→ fuse rankings only for RRF and apply the selected reranker, if any
→ select the ranked top 3 or top 5 chunks
→ number the evidence blocks
→ pack/check the actual prompt and perform required per-request model switching
→ generate the answer and citations
→ retain output/outcome, evidence identities, and stage/complete-request timing
```

Retrieval quality uses the designated first measured ranking on answerable
questions. Generation quality uses all three scheduled seeds `42`, `43`, and
`44`; its repeatability retains the three unordered response pairs. Those are
different metric observation rules, not interchangeable ways of selecting an
answer. A later success does not replace an earlier failure. Retain retrieval
integrity and generation's diagnostic agreement scopes separately.

Use the [shared measured-run lifecycle](../../methodology.md#preflight-and-measured-run-lifecycles):
initialize the frozen stack and run one representative complete-request warmup
before the three measured attempts, not one warmup per seed. Warm latency
excludes offline corpus/index preparation, one-time initialization, that warmup,
and judging, but includes every selected per-request residency transition and
context-packing operation. Generator cold loading is not full-system startup;
generator TTFT is not user-request or visible-answer TTFT. Capture server-call,
retrieval, generation, packing, switching, and the outer timer on the same live
path. Preserve detailed subintervals as timing artifacts, not new metrics.
Stage support and complete/partial resource windows remain explicit after
failure. Save all three outputs/outcomes before automatic offline judging;
evaluation-only recovery follows the
[shared saved-evidence rule](../../metrics.md#prediction-evidence-and-rescoring),
without re-inference or another locked execution. The exact
[metric boundaries and examples](metrics.md#complete-system-operations) prevent
double-counting server time or adding unrelated component p95 values.

## Results and MLflow structure

### Run hierarchy

Final RAG uses `EduMind / Final RAG`. Each phase has one parent and one
frozen-system child, not another comparison of component combinations:

```text
MLflow experiment: EduMind / Final RAG
├── parent: smoke-cpu
│   └── child: selected-system
├── parent: smoke-cuda
│   └── child: selected-system
├── parent: preflight
│   └── child: selected-system
└── parent: locked
    └── child: selected-system
```

`selected-system` is the concise display name; the exact composition and its
fingerprint belong in the recorded parameters/provenance. Run IDs distinguish
records without adding timestamps to display names. There are no child runs
for questions, attempts, components, or the LLM judge.
Component smoke, preflight, development, validation, and locked evidence stays
in the component experiments and is linked through the frozen decisions.

This is the approved target layout. The
[implementation boundary](#implementation-status) still applies to its runner
and artifacts.

### Parent and child records

The parent stores the executed profile/phase, plan, frozen system decision,
manifest checksum, composed protocol/model identities, judge contract and
calibration identity where applicable, hardware/software provenance, linked
upstream run IDs, and completion summary. Its direct metrics describe
execution/comparison completeness, not a
second copy of the system's answer-quality metrics. It has one planned system,
not a finalist leaderboard or paired system comparisons.

The child records the complete resolved composition: chunker, embedding,
retriever, reranker, vector server/configuration, generator/model mode, top-K,
prompts, context packing, residency policy, device/dtype, batch size, the phase's
warmup/attempt settings, and measured seeds where applicable. Keep the model,
protocol, manifest, judge, and decision identities available when the child is
viewed alone. Its metrics and artifacts
cover all questions and attempts; a question is a row, not another MLflow run.

### Child metrics

On the locked child, automated metrics retain the component definitions and
family names, with Final RAG's documented integrated timing and stage-eligibility
scopes:

| Metric family | What appears on the frozen-system child |
|---|---|
| Retrieval diagnostics | First-attempt nDCG, unit recall, token precision, and alpha-nDCG at the supported frozen cutoffs, plus pool recall at 20. |
| Answer quality | The four primary metrics, their precision/recall diagnostics, and diagnostic `quality.citation_support`. Semantic labels come from the LLM judge; citation-ID coverage is calculated deterministically. |
| Behavioral validity and reliability | Response/Refusal Validity, Malformed Output Rate, End-to-End Request Failure Rate, and the three generation-stage event rates with their own denominators. |
| Repeatability | Status, citation, and semantic agreement over scheduled response pairs. |
| Operations | Server-call, retrieval, generation, and complete-request p50/p95; generator TTFT, decode throughput, cold model load, and scoped RAM/VRAM peaks. |
| Workload and support | Applicable token/context/storage descriptors and eligible, contributing, failed, unavailable, and independent-source support. |

The [metric catalogue](metrics.md#metric-summary) owns roles, eligibility, units,
and definitions. Record the four primary names and directions explicitly.
Shared [MLflow suffix rules](../../methodology.md#shared-mlflow-metric-convention)
govern applicable support and confidence bounds. Undefined scalars are omitted,
while their statuses, reasons, and records remain in artifacts.

Keep fixed corpus/question/source-token inventories as workload metadata or
artifacts, and embedding dimension/dtype as parameters checked against prepared
vectors. Retain selected chunk counts, vector/index bytes, retrieved-context
sizes, and generated-token summaries under their declared descriptor scopes.
These do not add selection metrics or a new corpus-build throughput scalar.

### Artifacts

Smoke retains its fixture outputs, checks, timings, and device evidence without
authoritative intervals. Preflight attaches `preflight_report.json` to its parent
and retains child placement checks, resource samples, stress-input identity, and
qualification outcome. Locked execution links that exact qualification evidence.
Neither readiness phase creates a winner.

The locked report retains:

| Location | Retained evidence and purpose |
|---|---|
| Parent: plan, provenance, and summary artifacts | Frozen composition/decisions, input and protocol identities, upstream run links, judge/calibration identity, workload inventory, and execution/evaluation completeness. |
| Child: `attempts.csv` and `attempts.parquet` | Equivalent content with one row per scheduled question/attempt, including successes, failures, and explicitly unexecuted work. Include question/source ID, attempt, seed, request status, failure stage, error type, generation eligibility, downstream status, and output/timing references. |
| Child: prediction and evidence records | First-ranking and per-attempt pool/ranking IDs, supplied context, stable citation mappings, raw generated streams, visible responses, schema/ID checks, and native token counts. |
| Child: LLM evaluation records | Frozen rubric/request identities, raw judge replies, claim inventories, separate support/correctness labels, evidence justifications, pair labels, and retry/resume outcomes. |
| Child: measurement and metric-status records | Stage/outer timings, resource samples and monitoring scope, input/placement/integrity outcomes, metric values/statuses, denominators, and CI support. |

Reuse the existing outcome records and artifact references rather than creating
a separate file or run for every exception. Detailed causes remain artifacts,
not one metric per exception or schema violation. Automated evaluation consumes
the saved locked outputs under the frozen judge contract; evaluation recovery
does not repeat inference or authorize changing the judge or system.

Apply the [shared availability and artifact-reporting rules](../../metrics.md#shared-availability-rules)
to component evaluation, attempt records, and completed stage measurements.
Missing required component evidence prevents a complete locked report.
Independent support, bootstrap units, and single-observation exceptions follow
[Final RAG uncertainty rules](metrics.md#aggregation-and-confidence-intervals).
