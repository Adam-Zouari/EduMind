# Final RAG and human review methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Implementation boundary](#implementation-boundary)
- [Frozen system and execution profile](#frozen-system-and-execution-profile)
- [Data and execution](#data-and-execution)
- [Output and evidence trace](#output-and-evidence-trace)
- [Human review](#human-review)
- [Extraction-to-RAG confirmation](#extraction-to-rag-confirmation)
- [Result and MLflow structure](#result-and-mlflow-structure)

How well does the already-selected complete system produce evidence-backed
answers when every component runs together?

## Implementation boundary

This page defines the approved locked-only design. The current Final RAG
runner/protocol still expose smoke/development/validation profiles and cross
component finalists/top-K values into multiple systems. The current review
exporter consumes development finalists, and the locked command requires that
earlier review. Those paths do not implement the one-system, post-locked,
reporting-only review specified here and must not be presented as evidence of it.

The calibrated generation judge and standalone data-validator/report-reuse
interface are also pending. A matching server-backed execution trace, required
metric coverage, and review/confirmation artifacts must be verified before
claiming the complete contract is implemented. No authoritative manifest,
winner, judge identity, or completed human review is supplied by these examples.
The documented design does not authorize the existing multi-system workflow as
another Final RAG selection stage.

## Frozen system and execution profile

Final RAG has only the `locked` profile. The engineer records one complete system
assembled from the selected extraction routes, ASR/video policy where applicable,
chunking–embedding pair, retrieval–reranking stack, vector-server profile, and
generator model-mode configuration. Top-K is one frozen value, either `3` or `5`,
chosen from component development and validation evidence before any applicable
locked evaluation. Offline annotation review is separate from candidate selection.
The prompt, context packing, and refusal policy are also frozen.

`final-rag-locked.json` identifies this composition and its upstream reviewed
decisions. Final RAG does not cross component finalists, tune settings, or select
a winner. Non-locked integration checks and extraction confirmation occur before
the one locked benchmark invocation. It includes the frozen measured
repetitions and generation seeds; those are not additional selection rounds.

The sequence is:

```text
component development/validation decisions -> freeze one complete composition
-> non-locked integration checks and separate extraction confirmation
-> one locked Final RAG invocation -> blinded reporting-only human review
```

Integration checks diagnose loading, interfaces, evidence offsets, model
residency, and runtime behavior without creating Final RAG finalist rounds.
The component decisions, not those checks or human ratings, establish the
composition. Separate component qualification does not guarantee integrated
memory fit. Record the selected lifecycle/residency and ordinary runtime gates;
do not silently reduce top-K, truncate context, offload, or substitute a model
to make the locked run fit. Extraction retains its own uncapped isolated
resource contract; a RAG-family cap is not applied retroactively to it.

## Data and execution

Validate the held-out inputs and frozen composition's requirements through the
[shared data-validation workflow](../../data-validation.md) before execution.
The proposed standalone/report interface remains pending implementation. Final
RAG has only locked evaluation inputs; validation does not introduce another
candidate comparison or authorize reviewing held-out answers during selection.

The locked profile uses the held-out locked-test manifest. Prepare the corpus
and indexes once as recorded preparation work, separate from warm requests:

```text
selected source routes -> canonical chunks -> selected stack's required indexes

for every answerable AND unanswerable question:
query Dense through the approved server, BM25, or both for RRF
→ fuse rankings only for RRF and apply the selected reranker, if any
→ select the ranked top 3 or top 5 chunks
→ number the evidence blocks
→ generate the answer and citations
```

For `top_k=3`, retrieval quality is reported at 3. For `top_k=5`, it is reported
at 3 and 5.

Final RAG applies the [shared retrieval evidence definitions](../../metrics.md#retrieval-evidence-quality)
to nDCG, Evidence-unit Recall, and Evidence-token Precision at the available
cutoff; alpha-nDCG remains a diagnostic. It also reports the
primary generation-quality metrics, behavioral validity, reliability, and
repeatability diagnostics from the
[Generation benchmark](../generation/metrics.md), plus retrieval, generation,
server-call, and complete end-to-end p50/p95 latency. RAM and VRAM remain
operational measurements.

Retrieval quality uses the designated first measured ranking on answerable
questions. Generation quality uses all three scheduled seeds `42`, `43`, and
`44`; its repeatability retains the three unordered response pairs. Those are
different metric observation rules, not interchangeable ways of selecting an
answer. Preserve the selected retrieval stack's own ranking-consistency gate
and generation's diagnostic agreement scopes; neither is extended to extraction
or replaced by the other's checks.

Warm request latency excludes ingestion/build, loading, warmup, and offline
judging. Capture server-call, retrieval, generation, context-packing overhead,
and an outer complete-request timer on the same live path. Stage support and
complete/partial resource windows remain explicit after failure. The exact
[metric boundaries and examples](metrics.md#complete-system-operations) prevent
double-counting server time or adding unrelated component p95 values.

## Output and evidence trace

Reuse the [generation response schema](../generation/methodology.md#response-schema)
without adding model-visible gold answers or evaluator-only source material.
Number only the retrieved blocks and retain the mapping from each ID to chunk
ID, rank, source document, exact offsets, and covered gold units. A response-level
citation array does not assert a separately emitted claim-to-citation map.

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
outcomes and null/status rules from [Final RAG metrics](metrics.md#eligibility-and-result-status).

## Human review

Human review provides a separate qualitative report on the frozen system; it
does not select another system. The exporter selects 20 locked questions using
a selection rule frozen before answers are inspected and creates:

```text
20 questions × 1 anonymous system = 20 anonymous answer items
```

One reviewer scores all 20 answer items while model identity remains hidden.
The reviewer sees the question, answer/status/citations, accepted answer,
required facts, supplied evidence, and frozen evaluator-only verification
material. That verification material is never added to the generator's prompt.
Faithfulness, Answer Correctness,
Completeness, and Citation Accuracy each use `0–2`; Answerability Correctness
uses `0–1`. Their separate questions, rubric levels, examples, means, and
missing-value rules are defined in [human review metrics](metrics.md#human-review-metrics).

Before export, freeze the question-sampling rule, response/seed-selection rule,
anonymization and shuffle identity, and refusal-specific rubric instructions.
The response-selection and refusal-scoring details remain unresolved in the
current design; do not choose a best response or infer favorable refusal scores.
Keep the identity mapping outside the reviewer's blinded packet, retaining locked
run IDs, question/response IDs, and evidence identity for import validation.
An item with no available answer must remain visible as such, not an invented
empty response or a later successful substitution.

This is single-reviewer evidence, so the report does not claim inter-reviewer
reliability. Ratings are imported and validated as reporting-only evidence;
they cannot reopen selection after the locked test. A future
multi-reviewer study must define overlap, agreement, and adjudication separately.

## Extraction-to-RAG confirmation

How much does real extraction reduce the quality of the selected RAG system?

### Execution

Two versions of the same documents and questions are compared:

```text
verified reference text → frozen selected RAG
selected extracted text → the same frozen selected RAG
```

Question IDs, document IDs, questions, model profiles, prompt, and retrieval
strategy remain identical. The selected parser, ASR, and vector server are part
of the extracted-text path. Each text version keeps its own evidence offsets
because extraction can change length and layout. Gold evidence-unit IDs,
required claims, verification material, and reference-fixed eligibility remain
the same in both branches. Required-unit and required-fact inventories do not
shrink when extraction loses content. Output-dependent denominators, such as
returned tokens and generated claims, can differ; each branch also computes
[shared rank normalization](../../metrics.md#retrieval-evidence-quality) from its
own complete chunk corpus. A unit lost or corrupted by extraction remains a missed
required unit; it is not removed from the extracted branch's reference.
Record unrecoverable units explicitly rather than inventing empty or unrelated
source spans. Review and freeze the variant-specific mappings before scoring
outputs.

Build each branch's chunks/indexes from its own text using the same selected
settings; do not reuse reference-text vectors in the extracted-text branch.
Preserve source-level pairing and aligned generation seeds. Extraction can
change boundaries, token counts, and chunk rankings; those are measured effects,
not reasons to retune a branch. Explicit unrecoverable unit mappings must be
distinguished from missing or invalid reference annotations during validation.

This comparison uses a separate frozen confirmation manifest derived without
locked-test questions. It may describe deployment risk, but it cannot reopen
component selection after the system has been frozen.

### Metrics

The experiment reports the paired extracted-minus-reference difference for:

- **Retrieval:** nDCG, Evidence-unit Recall, and Evidence-token Precision at the
  system's actual top-K, plus alpha-nDCG as a diagnostic on the same answerable
  questions.
- **Generation quality:** Faithfulness, Factual Correctness F1, Answer Relevancy,
  and answerable-only Citation Precision/Recall/F1.
- **Validity and diagnostics:** Response Validity Rate, Refusal Validity Rate,
  Malformed Output Rate, reliability rates, repeatability metrics, and
  per-question error inspection.
- **Operational:** server-call, retrieval, generation, and complete p50/p95
  latency.

This experiment does not select the extractor again. It quantifies the downstream
cost of extraction after component selection.

The difference is always `extracted - reference`: negative quality differences
indicate loss, while positive failure-rate or latency differences indicate
regression. Preserve both branch values and aligned support. The
[paired-difference contract](metrics.md#extraction-to-rag-paired-differences)
defines examples, null behavior, percentile differences, and paired bootstrap
intervals. The confirmation report remains separate from the locked-test report;
its non-locked data do not become another selection profile.

## Result and MLflow structure

The approved result layout is one `EduMind / Final RAG` locked parent and one
frozen complete-system child, not a child per component finalist combination.
The parent retains the frozen composition/decisions, component protocol and
model identities, manifest checksum, completion state, and linked upstream run
IDs. The child retains the resolved system, measured seeds, first-ranking IDs,
per-question/per-attempt outputs and outcomes, evidence/citation traces,
semantic labels, stage timing, resource samples, and metric-specific support.
These are required evidence contents, not a claim that the current exporter
already creates the corresponding approved artifacts.

Automated metrics retain the component definitions and family names. Human
ratings are separate review artifacts linked to the same locked answers;
confirmation keeps both branch records and paired analysis on its own manifest.
Neither human review nor confirmation creates an extra candidate/winner child.

Apply the [shared availability and artifact-reporting rules](../../metrics.md#shared-availability-rules)
to component evaluation, attempt records, and completed stage measurements.
Missing required component evidence prevents a complete locked report.
Independent support, bootstrap units, and single-observation exceptions follow
[Final RAG uncertainty rules](metrics.md#support-and-confidence-intervals).
