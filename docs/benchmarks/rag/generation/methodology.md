# Generation methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Implementation boundary](#implementation-boundary)
- [Models and modes](#models-and-modes)
- [Data](#data)
- [Observation and aggregation](#observation-and-aggregation)
- [Frozen decoding and output contract](#frozen-decoding-and-output-contract)
- [Response schema](#response-schema)
- [Semantic judge contract](#semantic-judge-contract)
- [Execution profiles and selection](#execution-profiles-and-selection)
- [Per-candidate execution](#per-candidate-execution)
- [Metrics and why they are used](#metrics-and-why-they-are-used)
- [MLflow result structure](#mlflow-result-structure)

Which local Hugging Face generator produces the best grounded, cited answer when
every model receives exactly the same verified evidence?

## Implementation boundary

This page specifies the approved generation design. The current runner and
`protocol.yaml` have not yet been aligned with all of it: they retain a
reasoning-only roster, greedy decoding with a separate answer-length cap,
a development screen, two warmups, and lexical/local-classifier evaluation.
Those paths do not establish the seven model-mode, full-question-set,
one-warmup, calibrated-semantic-judge comparison described below. The
standalone validation/report-reuse interface also remains planned.

Judge identity is pending calibration, not selected by this document. No
authoritative claim-level score or complete comparison is established until
the implementation, resolved settings, reference data, and calibration artifact
meet this contract. The existing local faithfulness classifier is not a
calibrated replacement for all semantic responsibilities.

## Models and modes

| Model/profile | Modes evaluated | Why it is included |
|---|---|---|
| Falcon-H1-Tiny-R-90M control | Reasoning only | Independent weak control that establishes a low-resource quality floor using its documented profile. |
| Qwen3 0.6B | Direct and reasoning | Small established candidate with an official hard thinking switch. |
| Qwen3.5 0.8B | Direct and reasoning | Newer intermediate candidate with official mode-specific generation settings. |
| MiniCPM5 1B | Direct and reasoning | Highest recorded public screening score among the three shortlisted checkpoints and provides an official thinking switch; GPU fit requires preflight. |

The three switchable checkpoints therefore create six model-mode configurations;
Falcon contributes one reasoning control. A model-mode pair is one candidate
configuration because mode and decoding can change quality, token workload, and
latency. No trustworthy public benchmark compares all seven configurations under
EduMind's grounded QA, citation, refusal, faithfulness, and local-hardware
protocol. Public evidence made the shortlist; this experiment makes the
configurations directly comparable.

## Data

Smoke uses committed wiring fixtures. Every hardware-qualified generator
configuration receives the complete frozen development question set, including
answerable and unanswerable questions from QASPER and the verified structured
supplement. Validation evaluates only engineer-selected generator finalists on
the complete frozen validation question set. The selected generator
configuration receives its own locked test on frozen evidence, separately from
the end-to-end Final RAG report.

- For an answerable question, the generator receives the question and verified
  numbered evidence blocks. The evaluator separately receives accepted answers,
  required gold claims, gold evidence-unit IDs, and frozen verification material;
  gold answers are never included in the generator's prompt.
- Unanswerable questions receive text from their document that does not answer
  the question.
- Retrieval is not run in this stage.

Using frozen evidence prevents a good generator from being penalized by a poor
retriever.

The [shared data-validation workflow](../../data-validation.md) checks accepted
answers, required facts, supplied evidence, answerability, context provenance,
and source isolation before execution; its standalone/report interface is
planned. Judge calibration and output-schema validation remain separate gates.
Preparation checks do not enter generation latency or model resource windows.

Check frozen prompts for each model-mode's actual native input compatibility
before measured generation, following the
[reference/input-compatibility contract](../../data-validation.md#generation-references-and-model-input-compatibility).
This gate does not shorten evidence, change the context capacity, or omit
questions for an individual candidate. Its report and MLflow status are
separate from data-validation reports, hardware qualification, and quality.

## Observation and aggregation

Development, validation, and locked execution use three sequential sampled
responses per question, with aligned seeds `42`, `43`, and `44`. Score each
evaluable output, average within its question, then within its source document,
then macro-average documents. All three responses contribute quality evidence;
none is selected as the best response or used to replace a failed observation.
The repeated draws reduce dependence on a single sampled response, while
repeatability separately measures agreement of status, citations, and meaning.

Generation uses this all-seed quality policy because the frozen decoder samples
responses. It does not change the designated-first-quality policy of extraction,
chunking–embedding, or retrieval–reranking. The shared
[partial-evidence](../../metrics.md#dependent-metrics-and-partial-observations)
and [confidence-interval](../../metrics.md#confidence-intervals) rules apply to
the observation set declared here. The independent unit is the source document,
not a generation seed. Detailed denominators, conditional observations, and CI
applicability belong to the [metric reference](metrics.md#eligibility-aggregation-and-confidence-intervals).

## Frozen decoding and output contract

Every generator uses its exact pinned local snapshot, official chat template,
official mode switch, and documented mode-specific decoding:

| Configuration | Frozen decoding |
|---|---|
| Qwen3 direct | `enable_thinking=false`, sampling, temperature `0.7`, top-p `0.8`, top-k `20`, min-p `0` |
| Qwen3 reasoning | `enable_thinking=true`, sampling, temperature `0.6`, top-p `0.95`, top-k `20`, min-p `0` |
| Qwen3.5 direct | `enable_thinking=false`, sampling, temperature `1.0`, top-p `1.0`, top-k `20`, min-p `0`, presence penalty `2.0`, repetition penalty `1.0` |
| Qwen3.5 reasoning | `enable_thinking=true`, sampling, temperature `1.0`, top-p `0.95`, top-k `20`, min-p `0`, presence penalty `1.5`, repetition penalty `1.0` |
| MiniCPM5 direct | `enable_thinking=false`, sampling, temperature `0.7`, top-p `0.95` |
| MiniCPM5 reasoning | `enable_thinking=true`, sampling, temperature `0.9`, top-p `0.95` |
| Falcon control | Reasoning-only configuration resolved from its pinned official generation configuration |

The mode switches and settings follow the pinned official model cards for
[Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md),
[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md),
and [MiniCPM5](https://huggingface.co/openbmb/MiniCPM5-1B/blob/87179e5c1f455ef22e6223592d2d61351b525bfc/README.md).

Development, validation, generation locked, and Final RAG reporting use an
8,192-token model context, the same CUDA device, `float16`, and batch size `1`. CPU smoke uses
`float32`. No configuration receives hidden quantization, CPU/GPU offload,
automatic device splitting, or a candidate-specific batch size.

Output capacity is the context remaining after the complete prompt is tokenized.
When the runtime requires `max_new_tokens`, the runner supplies that remaining
capacity. Generation stops at EOS, the context boundary, or the frozen timeout.
Every attempt records prompt, reasoning, visible-answer, and total output token
counts when the runtime exposes them, plus the exact finish reason.

## Response schema

Every visible model response contains exactly three fields:

| Field | Meaning |
|---|---|
| `status` | A string: either `answered` or `insufficient_evidence`. |
| `answer` | A string containing the visible answer or the frozen refusal sentence. |
| `citations` | An ordered array of string evidence-block IDs; supplied-ID resolution is checked separately. |

An answerable response uses `status="answered"`, contains a non-empty substantive
answer, and cites the supplied evidence blocks used by that answer. Only IDs
from evidence blocks supplied with the current question are valid; this isolated
generation benchmark does not provide conversation history. For this illustrative
request, block `E1` states "The trial included 500 participants." A valid answer is:

```json
{
  "status": "answered",
  "answer": "The trial included 500 participants.",
  "citations": ["E1"]
}
```

An unanswerable response uses the frozen refusal representation:

```json
{
  "status": "insufficient_evidence",
  "answer": "The provided evidence is insufficient to answer the question.",
  "citations": []
}
```

Reasoning text is captured separately when the runtime exposes it and is never
placed inside `answer`. Structural validation checks JSON parsing, the exact
field set, field types, and the two allowed statuses. Missing/additional fields,
wrong types, unsupported statuses, and unparsable output are malformed. Normally
completed empty text also fails that schema. Runtime errors, timeouts, and output
context exhaustion are attempt outcomes rather than invented model statuses;
their partial text does not become a completed response merely because it parses.

Schema validity and required behavior are different. An otherwise well-formed
`answered` object with `citations: []` fails Response Validity and citation
quality but is not malformed solely for having no citations. A correctly formed
refusal on an answerable question is likewise a behavioral/quality failure,
not a parsing error. A citation such as `E9` is legal only if `E9` was supplied;
legal but irrelevant evidence lowers citation quality, while an unknown ID
fails ID validity and Response Validity without changing structural validity.
Unknown IDs remain in the citation-precision denominator and earn no recall
credit; they do not automatically invalidate the visible answer's semantic
scores. Repeated IDs are deduplicated for citation scoring, with the original
list retained. A limitation statement alongside a partial answer is assessed
for behavior and meaning, not classified as malformed by wording. Accepted
answers and evaluator-only verification material never become extra model-visible
evidence blocks.

## Semantic judge contract

Authoritative generation will use one pinned LLM judge after its identity has
passed calibration and been frozen. That judge supplies every semantic label
needed for Faithfulness, Factual Correctness, Answer Relevancy, and Repeat
Semantic Agreement. It uses one structured per-response rubric for claim
extraction, context support, source-verified correctness, gold-claim matching,
and relevancy, plus one pairwise rubric for repeated-answer semantic equivalence.
Deterministic
benchmark code converts those labels into metric values and aggregates; the
judge never calculates citation-ID coverage, validity, reliability, or
operational metrics.

The claim and evidence rules are frozen before calibration:

- Score only the visible answer, not hidden reasoning. Extract independently
  checkable atomic claims, splitting compound statements while preserving
  negation, quantities, units, dates, scope, and uncertainty.
- Interpret factual assertions in question context, so a short answer such as
  "500" can express a claim. The exact frozen refusal is a behavioral response,
  not an asserted answer fact; its answer-claim inventory is empty.
- Semantically deduplicate generated claims within a response before calculating
  claim ratios; retain their original spans for audit. Repeating a correct fact
  cannot inflate precision, faithfulness, or recall.
- Use human-reviewed, atomic, non-duplicate gold claims to define the facts
  required for a complete answer. Semantic equivalents count; a partially
  covered compound statement must first be separated into atomic facts. Each
  required fact can receive recall credit only once.
- Verify every distinct generated factual claim against frozen authoritative
  source/reference material. A correct extra fact absent from the required gold
  list receives precision credit if that material verifies it, but no extra
  recall credit. Unverified additions receive no correctness credit. Relevancy
  independently assesses whether additions help answer the question.
- Judge Faithfulness solely against the evidence supplied to this generator
  request. Broader verification material may establish factual correctness but
  cannot rescue a claim unsupported by the supplied context. When those two
  reference sets coincide, support and correctness can overlap; gold recall
  still measures completeness.
- Retain the source span or evidence justification for each support,
  correctness, and gold-matching label. The judge's own memory or an unrecorded
  lookup is not an authoritative verification source.
- For repeat semantic agreement, compare valid completed responses with matching
  statuses for equivalent material meaning, including factual qualifiers.
  Paraphrases can agree; incorrect but equivalent answers can also agree.

The frozen verification material and required gold claims are separate inputs:
the former establishes correctness, while the latter establishes completeness.
This is EduMind's source-verified claim contract, rather than treating every
generated claim absent from a reference-answer list as incorrect.

The exact judge version, decoding, prompts, rubric checksums, schema, verification
material identity, finite retry policy, and calibration artifact are frozen
before authoritative execution. Candidate identity is hidden from the judge,
raw judge outputs are retained, and a judge
failure makes evaluation incomplete rather than lowering the candidate's score.
The judge must first pass a human-labeled development calibration set with
recorded acceptance criteria for claim extraction, support, correctness,
gold matching, relevancy, and semantic equivalence separately. Unresolved judge
errors make evaluation incomplete after the frozen retry policy is exhausted.
Its latency, cost, and resources are excluded from generator operational
measurements.

## Execution profiles and selection

```text
smoke:
all seven model-mode configurations on tiny committed fixtures, independently on CPU and CUDA
-> verify loading, mode switching, decoding, output parsing, scoring, artifacts, and cleanup

preflight:
all declared configurations in fresh CUDA workers on frozen stress inputs
-> qualify placement, offloading, VRAM, first-inference behavior, and the supported input envelope

development:
all hardware-qualified configurations on the complete frozen development question set
-> engineer records up to three generator finalists

validation:
only the recorded finalists on the complete unseen validation question set
-> engineer records exactly one selected model-mode configuration

locked test:
the one selected generator runs on frozen evidence in its own benchmark
-> reporting only, with no further generator tuning
```

Smoke supplies wiring evidence only. Preflight supplies hardware eligibility
only and has zero warmups. Development is the stage for introducing candidate
configurations and tuning their settings. Validation compares only the frozen
finalists named in the reviewed development decision on unseen data; it does not
introduce configurations or retune their settings. The resulting
`generation-locked.json` records exactly one successful model-mode configuration
selected from validation. Its
own locked run measures generation on verified evidence; Final RAG uses the
same selection but measures answers on the complete retrieval path. Neither
locked report can reopen selection.

## Per-candidate execution

Development, validation, and locked profiles follow the sequence below. Smoke
uses one measured seed per device, and preflight uses the shared qualification
lifecycle with no warmup; neither reports three-seed comparison evidence.

```text
verify the dataset report and frozen model-mode prompt compatibility
-> unload previous generator
-> start a fresh worker and measure process-cold model load
-> run one representative warmup with seed 0
-> give every candidate the same question and numbered evidence
-> generate sequentially with aligned seeds 42, 43, and 44
-> separate reasoning from the visible answer where supported
-> record termination, validate structural schema and supplied citation IDs
-> end generator resource monitoring after the last measured request
-> apply the frozen semantic judge after generator timing is complete
-> derive answer/refusal validity from recorded checks and required claim labels
-> aggregate quality, validity, repeatability, workload, and operational results
```

The warmup output and latency are excluded from quality and warm-latency
statistics, but a warmup failure remains fatal. Cold Model-Load Time ends when
the model is ready and CUDA is synchronized; it is separate from Time to First
Token and total warm latency. The three fixed measured seeds sample the frozen
decoder reproducibly and support semantic, status, and citation repeatability
diagnostics.

All scheduled outcomes remain recorded. Runtime/timeout/context-interrupted
responses and required structural-schema failures have unavailable dependent
quality, not artificial zero-quality answers. Normally completed responses with
wrong behavior use the metric's own scoring or known-empty convention. Missing
judging affects required labels, not independently completed ID checks or
operational measurements. Apply the [shared availability rules](../../metrics.md#shared-availability-rules).

Repeatability adopts the shared scheduled-pair rule with local status, Jaccard
citation, and semantic comparison functions. Operational observations keep
their own cohorts: latency uses its declared per-question observations, decode
throughput pools completed-stream work/time, and memory records the largest
valid lifecycle sample even if it came from a failed attempt. Raw counts and
monitoring coverage remain in artifacts; they do not change quality weighting.

## Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary quality | **Faithfulness**, **Factual Correctness F1**, **Answer Relevancy**, **Citation F1** | Separates support by supplied context, correctness and completeness against required facts, relevance to the question, and explicit evidence selection. |
| Quality diagnostic | Factual Correctness Precision/Recall, Citation Precision/Recall | Explains whether an F1 loss comes from incorrect/unverified claims, omitted required facts, or incorrect/missing citations. |
| Behavioral validity | **Response Validity Rate**, **Refusal Validity Rate**, **Malformed Output Rate** | Measures valid answer behavior on answerable questions, the exact refusal contract on unanswerable questions, and general schema integrity. |
| Reliability | Runtime Error Rate, Timeout Rate, Context-Limit-Reached Rate | Distinguishes measured inference errors, deadline stops, and output-context exhaustion. |
| Repeatability diagnostic | Repeat Status Agreement, Repeat Citation Agreement, Repeat Semantic Agreement | Measures stability of answer/refusal decisions, selected evidence, and meaning across seeds without requiring identical wording. |
| Operational | Cold Model-Load Time, Time to First Token p50/p95, End-to-End Latency p50/p95, Decode Throughput, peak process-tree RAM, peak device VRAM | Separates startup, initial responsiveness, complete warm-request latency, decoding rate, and memory. |
| Workload descriptor | Prompt, context, reasoning, visible-answer, and total output tokens | Records how much native-tokenizer work produced the observed quality and latency. |

Malformed Output Rate is minimized; Response and Refusal Validity Rates are
maximized. Malformed Output Rate is conditional on normally completed outputs;
the behavior and reliability rates retain their own scheduled denominators.
Citation metrics are deterministic: a correct citation is a supplied
evidence-block ID covering required gold evidence. The judge supplies semantic
claim labels, while code calculates ratios, F1, aggregates, and intervals.
Citation/claim counts and a compact finish-reason inventory remain per-response
or run artifacts, not extra MLflow summary metrics.

The engineer selects validation finalists, then one validation winner, after
inspecting the primary metrics, validity and reliability metrics, repeatability, latency,
workload, and resources.
Exact metric eligibility, failure behavior, and aggregation are defined in
[generation metrics](metrics.md).

## MLflow result structure

Generation uses `EduMind / Generation`:

```text
MLflow experiment: EduMind / Generation
|- parent: smoke-cpu
|  `- one child per model-mode configuration
|- parent: smoke-cuda
|  `- one child per model-mode configuration
|- parent: preflight
|  `- one qualification child per model-mode configuration
|- parent: development
|  `- one child per hardware-qualified configuration
|- parent: validation
|  `- one child per engineer-selected finalist
`- parent: locked
   `- one selected model-mode child
```

The parent stores the manifest, protocol and judge identities, aligned seed
list, candidate order, qualification or decision provenance, completion state,
and aggregate comparison artifacts. Each child stores the model and mode,
resolved decoder, device, dtype, batch size, warmup count, cold-load and warm
operational measurements, primary-metric contract, judge rubric checksums, and
per-question outputs and labels. The judge does not receive a nested candidate
run and its resource use is never attributed to the generator.

Generation quality, validity, reliability, repeatability, and operational base
keys use these metric-family namespaces. Token descriptors retain the
per-attempt counts and descriptive-summary contract in the metric reference:

```text
quality.faithfulness
quality.factual_correctness_f1
quality.answer_relevancy
quality.citation_f1
quality.factual_correctness_precision
quality.factual_correctness_recall
quality.citation_precision
quality.citation_recall
validity.response_validity_rate
validity.refusal_validity_rate
validity.malformed_output_rate
reliability.runtime_error_rate
reliability.timeout_rate
reliability.context_limit_reached_rate
repeatability.status_agreement
repeatability.citation_agreement
repeatability.semantic_agreement
operational.cold_model_load_seconds
operational.ttft_p50_seconds
operational.ttft_p95_seconds
operational.latency_p50_seconds
operational.latency_p95_seconds
operational.decode_tokens_per_second
operational.peak_ram_mib
operational.peak_vram_mib
```

The parent and every child also store the four primary metric names and their
directions in the metric contract so the main selection evidence is visible and
validated consistently.

Input compatibility is visible on the same candidate child. For example, a
frozen prompt of 8,300 native tokens with context capacity 8,192 records
`input_compatibility=failed`, `result_status=incomplete`, and the detailed
`input_compatibility.json` report. No generation occurred, so no zero quality,
Runtime Error Rate, or Context-Limit-Reached Rate is emitted. A passed gate
records `input_compatibility=passed`; subsequent output-context terminations
contribute to the existing context-limit metric and per-attempt outcomes.
There is no separate Input Context Overflow Rate.

Artifacts retain each question/seed's raw stream and visible response, native
token counts, citation/claim counts and labels, structural and ID checks,
behavioral checks, outcome/finish reason, and timing. A finish-reason count/share
table is a diagnostic artifact, not another scalar family. Keep monitoring
coverage and request outcomes alongside RAM/VRAM samples and their observed
peaks. Undefined scalar keys are omitted according to the shared rules, while
the records, reasons, and partial/incomplete status remain visible.
