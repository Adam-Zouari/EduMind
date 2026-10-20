# Shared benchmark metric conventions

[Benchmark program](../README.md#experiments) · [Experiment sequence and rationale](methodology.md) ·
[Benchmark runbook](running.md)

This page defines the shared measurement and reporting conventions for EduMind's
benchmarks. The benchmark pages below define **what each value means and how it
is calculated**; their methodology pages explain **where each metric is used and why**.
A development, validation, or locked result is authoritative only when its runner
implements the applicable contract exactly and records every required value.
Each metric declares its preferred direction; descriptive values have no
automatic better direction.

## Benchmark metrics

The shared rules on this page apply alongside each benchmark's specific contract.

| Benchmark | Methodology | Metrics |
|---|---|---|
| Document extraction | [Methodology](extraction/document/methodology.md) | [Metrics](extraction/document/metrics.md) |
| Audio extraction | [Methodology](extraction/audio/methodology.md) | [Metrics](extraction/audio/metrics.md) |
| Video extraction | [Methodology](extraction/video/methodology.md) | [Metrics](extraction/video/metrics.md) |
| Chunking and embedding | [Methodology](rag/chunking_embedding/methodology.md) | [Metrics](rag/chunking_embedding/metrics.md) |
| Retrieval and reranking | [Methodology](rag/retrieval_reranking/methodology.md) | [Metrics](rag/retrieval_reranking/metrics.md) |
| Generation | [Methodology](rag/generation/methodology.md) | [Metrics](rag/generation/metrics.md) |
| Vector databases | [Methodology](vectordb/methodology.md) | [Metrics](vectordb/metrics.md) |
| Final RAG and human review | [Methodology](rag/final/methodology.md) | [Metrics](rag/final/metrics.md) |

## Shared conventions

### Reading an individual metric

Each benchmark's metric reference gives the question, counted or timed units,
calculation, a worked example, interpretation, range/direction, and
availability rules. A word formula names its actual denominator: reference
objects, generated claims, returned tokens, completed requests, and scheduled
attempts are not interchangeable.

The Document, ASR, Video, Chunking–Embedding, Retrieval–Reranking, and Generation references
use the following order for each individual metric:

1. **Question:** what the metric answers.
2. **Calculation:** exactly what is counted, timed, compared, or averaged.
3. **Example and interpretation:** what a concrete result means.
4. **Range and direction:** units, valid range, and preferred direction.
5. **Edge cases and availability:** genuine metric-specific exceptions and the
   value/status each receives.

The final block appears directly under the metric, not in one catch-all section
at the end of the page. When needed, its table uses
`Edge case | Example | Value/status | Explanation`. A metric without additional
exceptions links the shared availability rules instead of inventing cases.

This page owns genuinely identical cross-benchmark rules. A benchmark page
identifies the adopted rule and keeps its own units, denominator, eligibility,
measurement window, aggregation, examples, and exceptions. A local edge-case
table may show an applied value with a link to the shared convention; it must
not establish a second competing definition. Equal metric names do not imply
equal aggregation or eligibility.

Shared definitions on this page:

- [Availability and result status](#shared-availability-rules)
- [Execution accounting and result completeness](#execution-accounting-and-result-completeness)
- [Dependent metrics and partial observations](#dependent-metrics-and-partial-observations)
- [Prediction evidence and rescoring](#prediction-evidence-and-rescoring)
- [Reporting slices](#reporting-slices)
- [Prose scoring projection](#prose-scoring-projection)
- [Recognition error rates and empty references](#recognition-error-rates-and-empty-references)
- [Spoken timestamp alignment](#spoken-timestamp-alignment)
- [Content and detection empty comparisons](#content-and-detection-empty-comparisons)
- [Pairwise repeatability](#pairwise-repeatability)
- [Retrieval evidence quality](#retrieval-evidence-quality)
- [Retrieved-context workload](#retrieved-context-workload)
- [Latency percentiles](#latency-percentiles)
- [Ratio-of-totals throughput](#ratio-of-totals-throughput)
- [Resource measurements](#resource-measurements)
- [Confidence intervals](#confidence-intervals)

For example, recovering two of three required evidence units gives Recall
`2 / 3`. Finding evidence in 240 of 600 returned token occurrences gives
token Precision `240 / 600`. The first measures completeness; the second
measures concentration. Their denominators answer different questions even
when both come from the same ranking.

A **sample** is an input such as a document, clip, or question. An **attempt**
is one measured execution on that input. A **candidate** is one complete
configuration being evaluated. Repeated attempts remain attached to their
sample; they are not new independent sources.

The metric's preferred direction applies only to defined comparable values.
Null is not a score below zero, and a workload descriptor has no automatic
better direction. Metric roles describe the evidence an engineer reviews;
they do not assign a weighted winner score.

### What belongs in a metric's edge-case block

A metric-specific edge case arises when valid inputs and the metric's ordinary
calculation need an explicit convention: for example, both compared sets are
empty, a denominator is zero, or a conditional task has no eligible objects.
It can also concern that metric's own measurement or evaluator, such as an
undefined rate interval or failed official table scorer.

Ordinary outcomes are not edge cases. Zero recall after a valid irrelevant
ranking, WER above one, a slow request, and a low-quality but valid extraction
belong in calculations, examples, and interpretation. Do not relabel them as
exceptions just because the result is undesirable.

Invalid manifests, missing required annotations, contradictory reference labels,
and invalid source intervals belong to
[data validation](data-validation.md#data-validity-and-metric-applicability).
Malformed required output envelopes, prohibited truncation, nonfinite vectors,
incorrect pool provenance, and invalid ranking permutations belong to runtime
integrity checks.
They are not alternative numeric cases of nDCG, recall, or precision. Their
effect on available evidence follows the shared rules below and the benchmark's
execution contract.

### Shared availability rules

Metric-specific blocks add to these rules; they do not repeat every generic
failure under every metric:

| Condition | Value/status and treatment |
|---|---|
| A measured request fails or violates its required output-integrity contract | Dependent quality is `null/unavailable`; retain the actual outcome and any raw output. Do not score a fabricated empty prediction. |
| A designated-first-quality suite has a failed first request | Do not replace it with a later successful output; the request-failure rule above applies. |
| A later attempt fails | Preserve independently valid first-attempt quality and completed measurements. Record the failed outcome for the suite's reliability/consistency rules. |
| Required execution or records are missing | Mark affected evidence `incomplete`; never assume missing attempts succeeded or failed. |
| No applicable task exists | Record `null/inapplicable` with zero applicable task-unit support, retaining the source inventory. A missing required annotation is invalid input, not this case. |
| A required measurement or evaluation cannot be obtained | Record `null/unavailable` and its reason; do not report zero or infinity. A nonfinite or impossible measured value is a measurement-integrity problem. |
| An independent operation completed before another failed | Preserve its defined observation. Mark interrupted resource windows as partial; partial peaks or progress do not establish a completed workload. |
| There are no contributing values for an applicable aggregate | Record `null/unavailable`, retaining eligible and failed/unavailable support. |
| A defined point lacks sufficient independent support for a confidence interval | Retain the point, with null bounds and a reason. This is uncertainty support, not a new candidate-quality score. |

Result artifacts use these shared status meanings:

| Status | Meaning |
|---|---|
| `scored` | A defined numeric observation under the metric's own contract, including an explicit valid-empty convention. |
| `inapplicable` | No task is eligible or scheduled for this metric. |
| `unavailable` | A required observation or evaluation could not be obtained. |
| `incomplete` | Required execution or records are absent; preserve known observations without inventing the rest. |
| `invalid_reference` | Inputs violate their contract and must be repaired before evaluation; input rejection does not create measured candidate failures. |

`null` is a storage value, not an explanation. Artifacts retain the reason and
applicable scheduled, eligible, contributing, failed, unavailable, and unexecuted
support, including independent-source counts. Reference eligibility is fixed
before inference and is the same for compared candidates; actual contributing
cohorts and failures remain visible separately.
Undefined values are excluded only from the affected numeric calculation, not
from saved rows or other defined metrics. MLflow omits null scalar keys; local
and uploaded artifacts preserve the status. Partial required quality is a
diagnostic, not complete authoritative selection evidence.

Loading, warmup, input-validation, infrastructure, and offline evaluator failures
are separate from measured-request failures. Continue scheduled work after
recoverable request errors; if execution cannot continue, retain explicit
unexecuted outcomes. Do not manufacture a failed request for every input that
was never executed. A complete set of executed failures can still supply the
suite's failure/repeatability observations.

A completed timed phase retains its observation after an independent later
failure. A phase that never completes has no successful-completion latency;
retain elapsed-to-failure separately. Partial successful cohorts remain
labelled with their support. A valid empty comparison follows its explicit
metric convention, not the execution-failure rule.
Likewise, a valid output may contain an incorrect or unreconstructed object:
apply the metric's object-level penalty rather than treating every wrong table
or formula as a failed whole request.

### Execution accounting and result completeness

Report execution accounting, execution coverage, quality coverage, and integrity
separately. A complete outcome ledger is not a claim that every request succeeded
or that every required quality value is available.

| Condition | Accounting and coverage | Metric consequence |
|---|---|---|
| Every scheduled attempt executes and its outcome is recorded | Complete accounting and execution coverage, including observed failures. | Calculate each metric from its own valid observations and failure conventions. |
| A scheduled attempt has an explicit `not_executed` outcome and reason | Accounted for, but required execution coverage is incomplete. | Do not invent a failed request, success, or completed repeatability/failure-rate aggregate. |
| A required attempt has no outcome record | Accounting and execution evidence are incomplete. | Preserve known observations; absence of a record proves neither execution nor failure. |
| The protocol deliberately schedules no task | No missing required execution; the task is inapplicable. | For example, one-attempt smoke has no repeatability pair task. |

For three scheduled attempts, `A,A,failure` can retain complete first-output
quality, repeatability `1/3`, and failure fraction `1/3`. The same fractions for
`failure,A,A` do not supply first-output quality. Three executed failures have
complete failure evidence but no completed quality outputs. An unexecuted third
attempt instead leaves pairwise and scheduled failure-rate evidence incomplete,
even when its reason is recorded. Never silently replace a scheduled denominator
with the number of surviving requests.

Measured request errors do not by themselves erase earlier measurements or
create an implicit zero-error acceptance rule. Preserve declared hardware,
placement, telemetry, input, and output-integrity requirements. Missing required
quality remains partial evidence, not a complete quality comparison. Engineer
advancement decisions review these separate facts; any stricter exclusion must
be an explicit justified execution contract, not an inference from an MLflow
status or an unavailable scalar.

### Dependent metrics and partial observations

A derived value requires valid inputs from the same declared observation and
scope. Do not combine precision from one response with recall from another,
or complete-workload output with an interrupted operation's duration. If a
required component is unavailable, the dependent value is unavailable unless
the metric explicitly defines that known situation without the component.
A documented valid-empty convention is not permission to replace unknown
measurements with zero.

Calculate nonlinear scores, such as F1, at their declared observation level
before applying the benchmark's aggregation. F1 from averaged precision and
recall is not the average of individual F1 scores. A metric explicitly defined
from pooled counts retains that different estimator.

An observed partial quantity remains real evidence, with its scope. A derived
calculation using it cannot establish a complete-workload result. Preserve
independently complete operations, such as a finished cold load before a later
request fails. Missing required evaluations instead leave surviving quality
aggregates partial/diagnostic; do not promote their points or confidence bounds
to complete comparison evidence.

Conditional eligibility is not missing execution. Successful-completion latency
deliberately uses successful requests, while still retaining every outcome and
its support. Similarly, a fully evaluated zero-denominator case follows the
metric's own convention and reason. It is not an unresolved evaluator error.

### Prediction evidence and rescoring

Retain the prediction evidence and immutable input references needed to inspect
and rescore the declared metrics. Hashes supplement that evidence; they cannot
reconstruct a transcript, timestamp sequence, parsed table, or ranked result.

Keep sample and attempt IDs, executed outcomes, produced outputs or recorded
failure fragments, source identities/checksums, and the protocol, conversion,
normalization, and evaluator identities needed to interpret them. Retain every
executed attempt's relevant output representation for repeatability, while
quality still uses only the observations declared by its suite. A saved later
output is not a replacement first-quality observation.

Reuse the existing output records and artifact references rather than copying
the same prediction into every table. Document durable local paths/checksums and
whether the evidence is uploaded, referenced, or retained under controlled access.
Temporary worker files deleted after execution are not durable retained evidence.
Raw media, model weights, and full embedding matrices need not be duplicated
when immutable inputs and sufficient prediction records already exist.

Rescoring under a repaired evaluator preserves the original outputs and creates
versioned evaluation provenance; it does not silently rewrite a historical run.
Changed definitions require newly identified results, not relabelled old scores.

### Reporting slices

A slice is a reviewed subset of the same evaluation records, for example a
source type, text/table/formula/mixed evidence, or answerable/unanswerable
questions. It is not another inference run. Apply each metric's existing
eligibility, observation, and aggregation rules inside the subset, and report
eligible and contributing sample/source counts. No eligible task gives
`null/inapplicable`; insufficient CI support does not erase a defined point.

### Latency and warm latency

**Latency** is the elapsed time of an operation, measured between the start and
end boundaries defined by its benchmark. **Warm latency** is latency measured
after the candidate is ready and its warmup has finished. The word "warm"
describes the runtime state; it does not add another duration to the request.

Display names use **Time** for initialization and complete corpus-preparation
operations, and **Latency** for a query/request or a component inside it. Both
measure elapsed duration; seconds versus milliseconds and the presence of
inference do not determine the name. The search algorithm belongs in the
execution contract: Vector Search Latency can measure exact cosine search
without making "exact" part of its display name.

For example, a model takes 8 seconds to load, its warmup request takes
200 milliseconds, and a subsequent measured request takes 50 milliseconds.
The measured request's warm latency is **50 milliseconds**, not 8.25 seconds.
Cold loading is measured separately. The warmup prepares reusable inference
state; its latency and output do not enter measured-request aggregates.

Each successful measured request contributes its own duration under the suite's
observation rules. A p50 or p95 then summarizes those observations; it is not
the duration of model loading or warmup. Component latency and complete-request
latency can have different boundaries, so their names must identify what is
timed. See the [shared lifecycle](methodology.md#preflight-and-measured-run-lifecycles)
for monitoring, synchronization, loading, and warmup order.

Component timers have explicit, non-overlapping boundaries inside the same live
execution. Required asynchronous work is synchronized at its measured boundary.
Keep orchestration/assembly work in the enclosing timer rather than hiding it
or counting one operation in two components. Observe the complete interval
directly; neither component percentiles nor different successful cohorts can be
combined to invent a total. Timing records retain full precision and phase
outcomes, including elapsed-to-failure separately from completed durations.

### Calculation and result rules

- Source and evidence spans are half-open intervals: `[start, end)`.
- Empty denominators use the explicit behavior stated in each benchmark's metric contract; they never produce
  fabricated zero-quality observations.
- A valid empty comparison can have the contract's best or worst numeric value.
  A failed request is not an observed empty output. Extraction,
  chunking–embedding, and retrieval–reranking quality use the designated first
  measured attempt under the [shared availability rules](#shared-availability-rules).
  Failure/repeatability metrics retain their own definitions. Invalid references
  fail [data validation](data-validation.md) before candidate execution.
- Development, validation, and locked runs retain one row per sample before aggregation.
- Percentile calculations follow [Latency percentiles](#latency-percentiles).
  Each throughput metric specifies its own completed work and timed interval.
- Eligible intervals follow [Confidence intervals](#confidence-intervals),
  with independent units and applicability declared by each benchmark.
- Normalized precision, recall, F1, accuracy, coverage, nDCG, and correctness
  values lie in `[0, 1]`. CER and WER are non-negative and can exceed 1 when
  insertions outnumber reference units. Human rubric scores use their stated
  `0–2` or `0–1` scales. Time, memory, storage, and throughput are non-negative
  and have no fixed upper bound.

### Prose scoring projection

Prose comparisons use one symmetric, evaluation-only projection: Unicode NFC,
case-folding, replacement of Unicode punctuation with spaces, and whitespace
collapse. Prose Content metrics use the resulting whitespace-separated tokens;
WER uses those words and CER uses characters of the same projected strings,
including remaining spaces. These lexical units are not model-tokenizer IDs.

Emptiness is assessed after projection. Different raw punctuation-only strings
can therefore compare as empty. Keep raw outputs unchanged in artifacts. The
projection does not dehyphenate words, correct spelling, rewrite numbers,
remove headers, deduplicate content, or modify formulas, code, layout trees,
and table trees. This projection is not source-token counting for retrieval;
the retrieval metrics declare their fixed tokenizer separately.

### Recognition error rates and empty references

Document recognition, standalone ASR, and frozen-video ASR use the same
word/character edit convention after the prose projection. Align each complete
reference/prediction pair separately before applying the metric's local
aggregation. Never align across unrelated source boundaries.

For a positive reference length, WER or CER is substitutions plus deletions
plus insertions divided by reference length. The three word-error components
divide their respective counts by the same contributing reference-word count.
They sum to WER. A valid empty prediction against a nonempty reference requires
deleting every reference unit, giving WER/CER one under the ordinary calculation.
Insertions can make WER, CER, or Word Insertion Rate exceed one.

Use the established [JiWER empty-reference convention](https://jitsi.github.io/jiwer/#a-note-on-empty-references)
for WER/CER. EduMind extends the same edit-count behavior to its three word-error
component rates:

| Contributing reference length | Prediction | WER/CER | Word components, when reported |
|---|---|---|---|
| Zero | Empty | `0` | Substitution, deletion, and insertion rates are `0`. |
| Zero | Nonempty | Total word/character insertions, not division by zero | Word Substitution/Deletion Rates are `0`; Word Insertion Rate is the word insertion count. |

The convention applies at the declared calculation level. Document metrics
calculate document scores before macro-averaging; ASR and frozen-video ASR pool
edits and reference lengths from their contributing completed pairs before
division. Do not average per-clip zero-reference values into a positive-reference
corpus ratio. Preserve raw integer edit counts for reconciliation and rescoring.
An empty contributing cohort is unavailable, not a valid empty comparison.
Speech-corpus quotas and transcript/timing annotations remain
[data-validation requirements](data-validation.md), not scoring conventions
reapplied to bootstrap draws.

### Spoken timestamp alignment

Standalone ASR and frozen-video ASR share this alignment contract. For each
reference segment, enumerate contiguous spans of predicted timestamp units with
normalized token Content F1 at least `0.5`. Dynamic programming selects ordered,
non-overlapping matches by maximum total similarity, then match count, then
earliest spans. A reference segment matches at most one span and predicted units
cannot be reused. The matched span contributes its minimum start and maximum end.
Matching uses content/order, not boundary-error minimization.

**Timestamp Boundary MAE** divides total absolute start/end errors by matched
boundaries, two per matched segment. **Timestamp Alignment Coverage** divides
matched reference segments by required reference segments from valid completed
eligible outputs, not by predicted segments. Pool real errors/counts at the
benchmark's declared cohort level rather than averaging sample MAEs.

No reference timing task makes both metrics inapplicable. An eligible completed
output with no matches gives coverage zero and unavailable MAE: there is no
observed boundary error. Failed complete outputs do not become scored misses;
retain their planned reference counts separately. Read error and coverage
together so a precise small matched subset cannot imply complete recovery.

Each suite keeps its timestamp timeline, observation selection, and task
eligibility local: designated first clip outputs for ASR, complete stitched
video outputs for frozen ASR. Visual appearance timing uses a different matching
contract and does not inherit these spoken-segment rules.

### Content and detection empty comparisons

The document Content and Detection families and video's distinct Visual Content
family adopt the following empty-comparison convention. It applies only after
their local eligibility, units, and matching rules have been satisfied:

| Compared reference and prediction units | Precision | Recall | F1 |
|---|---:|---:|---:|
| Both empty | `1` | `1` | `1` |
| Exactly one empty | `0` | `0` | `0` |

For nonempty comparisons, use each metric's normal calculation. If precision
and recall are both zero, define F1 as zero rather than dividing by zero.
This convention changes neither multiset versus distinct-unit counting nor
the benchmark's aggregation.

Reference-based reconstruction still needs a reference object. For a recovered
expected table, the convention can compare projected cell text. A missed table,
even with blank cells, receives the declared missed-object score before that
inner text comparison; no reference tables means no reconstruction task.
Likewise, formula reconstruction, conditional timestamp errors, retrieval
evidence metrics, and generation correctness/citation metrics do not acquire
perfect empty scores from this rule. Their local contracts govern eligibility.

### Pairwise repeatability

Document and visual-video Repeatability Success Rate, ASR Transcript
Repeatability Success Rate, and Chunking–Embedding/Retrieval–Reranking Ranking
Repeatability share one pair-counting rule. Each benchmark freezes its own output
representation and comparison depth. These metrics are diagnostic: a different
valid output does not itself fail integrity. Invalid IDs, incorrect provenance,
prohibited truncation, and non-permutation outputs retain their separate gates.
Generation keeps its local semantic/citation comparison functions. Deferred
Vector Database and Final RAG revisions retain their explicitly documented
implementation boundaries; a shared definition does not certify those runners.

Compare every unordered pair of scheduled measured attempts. A pair succeeds
only when both outputs are valid and identical in the benchmark's chosen
representation. Divide successful pairs by scheduled pairs. Three attempts
give pairs `(1,2)`, `(1,3)`, `(2,3)`: `A,A,A` gives `1`, `A,A,B` or
`failure,A,A` gives `1/3`, and three executed failures give `0`. Identical
errors never agree; identical valid empty outputs can agree without establishing
good quality.

One deliberately scheduled attempt has no pair task: `null/inapplicable`,
`repeatability_not_measured` where that reason is used. Missing required outcomes
make the evidence incomplete, not a favorable surviving-pair statistic or
invented disagreement. Representation, aggregation, CI eligibility, and metric
role stay in the benchmark reference.

Generation's status, citation, and semantic agreements adopt the same scheduled
pair denominator and zero contribution from pairs containing executed failures,
interrupted outputs, or structurally invalid outputs. Their local comparison
functions assign binary or fractional scores to otherwise evaluable pairs;
sum those scores over scheduled pairs. An evaluator error is unavailable
evaluation, not an executed candidate disagreement.

Where the suite lists repeatability intervals, resample complete independent
sources under [Confidence intervals](#confidence-intervals), retaining every
scheduled attempt and pair. The recorded pair comparison is exact; the interval
describes variation of its aggregate across sampled sources, not uncertainty
about whether two saved outputs match. Diagnostic status does not remove that
source-sampling question or create an automatic selection threshold.

### Retrieval evidence quality

Chunking–Embedding and Retrieval–Reranking share the following definitions.
Their local pages identify which complete chunk corpus supplies the baseline:
the corpus produced by a candidate pair or the frozen corpus shared by stacks.
This is not a pool-local normalization. Other benchmarks reuse this contract
only when they explicitly declare it.

**Evidence coverage:** an answerable question has reviewed gold units with
stable IDs and half-open source intervals. One chunk covers a unit only when
it contains all that unit's required intervals in the same canonical document.
Partial overlap can contribute evidence tokens without recovering a whole unit.
Missing required annotations fail data validation; unanswerable questions are
not empty-reference retrieval comparisons. Valid rankings and native model
inputs retain their separate runtime-integrity checks.

| Metric | Shared calculation |
|---|---|
| nDCG@K | A chunk's gain is its count of distinct completely covered gold units, used linearly. Sum `gain / log2(rank + 1)` through K and divide by the exact ideal obtained by sorting gains over the complete candidate corpus. Repeated evidence keeps its gain in distinct chunks. |
| Evidence-unit Recall@K | Distinct gold units completely covered by at least one selected chunk, divided by required units. Each unit counts once, regardless of rank or repetition inside K. |
| Evidence-token Precision@K | Returned source-token occurrences overlapping verified evidence, divided by all returned source-token occurrences. Use canonical text and `tiktoken:cl100k_base`, not the prose projection or a model's native tokenizer. Count an occurrence once despite overlapping gold intervals; returned occurrences in different chunks count again. Native inference prefixes/special tokens do not count. |
| alpha-nDCG@K | Sum discounted novelty gains through K: each complete unit earns `0.5` raised to its earlier coverage count, followed by the same rank discount. Normalize by a deterministic greedy baseline over the complete corpus; at each step choose the unused chunk with maximum remaining novelty gain, breaking ties by frozen corpus order. |

The greedy alpha baseline is approximate, not a guaranteed maximum. Cap the
reported alpha-nDCG ratio at one; retain any exceeded uncapped ratio and flag.
An exceedance is not itself a scoring failure. For nDCG and alpha-nDCG, a valid
corpus with no chunk covering any complete required unit receives zero when
the normalization gain is zero, retaining its fragmentation limitation.
All answerable questions, including single-unit questions, are alpha eligible.
The [original alpha-nDCG paper](https://plg.uwaterloo.ca/~gvcormac/novelty.pdf)
describes this practical greedy approximation; exact-normalization review
remains in the [data-review checklist](pending-data-review.md).

Recall has a positive reference denominator and token precision has a positive
valid-output denominator. A token-free or invalid ranking is unavailable
under the integrity contract, not a scored empty comparison. Valid rankings
recovering no evidence receive ordinary zero scores. Each local page keeps
its @3/@5 examples, metric roles, pool diagnostics, and aggregation/CI rules.

### Retrieved-context workload

Chunking–Embedding and Retrieval–Reranking count selected context from the valid
designated first ranking, after reranking when that stage exists. At each cutoff
`K=3` and `K=5`, sum `tiktoken:cl100k_base` lengths of the selected canonical chunk
texts. Count each chunk separately, including overlapping/repeated source text;
exclude model-native prefixes and inference special tokens. A valid corpus
smaller than K uses its available chunks without padding or rescaling.

Mean Retrieved Context Tokens is the sum of these per-query totals divided by
contributing queries. p95 Retrieved Context Tokens is their interpolated 95th
percentile. These are query-weighted workload descriptors, not document-macro
quality, token budgets, or native-generator compatibility checks. The mean
describes average returned volume; p95 describes its upper tail. Neither has an
automatic better direction or a workload-descriptor confidence interval.

Require a valid first ranking and complete token counting for each query total.
Known totals survive later independent evaluator failures; later rankings never
replace a failed first observation. If eligible queries exist but no complete
total contributes, the aggregate is `null/unavailable`, not an observed zero.
No eligible task in an optional slice is instead inapplicable; missing required
corpus/query inputs still fail data validation. An incomplete collection of
required totals remains a labelled partial workload diagnostic with
planned/contributing counts. Shared percentile
rules handle one observation and fractional interpolated counts; invalid empty
rankings remain output-integrity failures, not a new empty-context convention.

### Latency percentiles

For p50, p95, or p99, sort the benchmark's eligible observations and locate
zero-based position `(n - 1) × p`, where `p` is `0.50`, `0.95`, or `0.99`.
Interpolate linearly between adjacent values at fractional positions. With one
observation, every percentile equals it; with none, the percentile is unavailable.
The same interpolation applies to a descriptor explicitly reporting p95, such
as chunk lengths; the descriptor retains its own observation set and units.

The benchmark defines the observation before this calculation: for example,
a document-normalized latency, per-query median successful duration, or a server
request in a workload cell. Do not silently pool repetitions or replace the
declared observations with per-document averages. Failed elapsed times are not
successful-latency observations.

A percentile is not an arithmetic mean or a confidence bound. Small valid
samples retain descriptive points without establishing reliable tail inference.
Component percentiles cannot be added or subtracted to derive an end-to-end
percentile; that path must be measured and summarized separately.

### Ratio-of-totals throughput

For a metric defined as total work divided by total time, sum its declared work
units and corresponding timed intervals before division. Do not substitute the
arithmetic mean of individual work-per-time rates. Two operations completing
60 units in one and three seconds yield `120 / 4 = 30 units/second`, not the
equal-operation mean `(60 + 20) / 2 = 40`. The former measures pooled work per
second; the latter answers a different question about request-level rates.

Each benchmark owns the contributing cohort, work units, timing boundary, and
treatment of failures. Decode-only intervals, elapsed request time, and an
entire batch window are not interchangeable. A throughput CI resamples the
declared independent units, carries their work/time contributions, and
recalculates the ratio on each draw. No positive contributing interval gives
an unavailable aggregate, not zero or infinity.

### Resource measurements

Each resource metric declares its monitored process/device and lifecycle window.
Model-backed workers normally include loading, warmup, and measured requests;
offline data validation, quality scoring, and bootstrap calculations are outside
that window. A suite may have additional measured work such as corpus build or
media decoding; its local window remains explicit.

Usable samples are required. Missing required telemetry is unavailable, never
an observed zero. Retain timestamped samples, method, scope, and failure reason.
Valid peaks measured before interruption remain observed partial-window peaks,
not proof of a completed lifecycle or of future hardware fit. A hardware gate
does not clamp or erase real observations. One lifecycle peak has no inferred CI.

The reported observed peak is the maximum across all usable samples in the
declared window, including loading, warmup, and failed/interrupted attempts
when those phases belong to it. Do not select only successful attempts or
average their separate peaks. Completed/partial attempt peaks of 2,400, 3,400,
and 2,600 MiB therefore report an observed 3,400 MiB, retaining the interrupted
attempt's outcome. A request failure does not itself make monitoring incomplete
if usable monitoring continued through the complete scheduled lifecycle.
Execution outcomes and monitoring coverage are separate recorded facts.

#### Peak Process-Tree RAM

At each sample, sum resident memory for the worker and its living descendants,
including children and grandchildren. Report the largest simultaneous sum,
not the sum of separate process peaks. Units are MiB; unrelated processes are
not added to this host-process measurement. Docker server memory follows its
own declared container/cgroup method, not this worker RSS definition.

#### Peak Device VRAM

Use the largest raw NVML `memory.used` sample on the assigned GPU. This is total
device occupancy, including driver, desktop, and other process allocations,
not per-process bytes or a baseline-subtracted increase. Record
`nvml-device-total`, GPU identity, idle baseline, total/free memory, and samples.
The baseline is explanatory and is not deducted. Keep background GPU activity
controlled across candidates. This measurement does not establish model placement.

CPU execution records `null/inapplicable`; missing required CUDA telemetry
records `null/unavailable`. Hardware caps and offload gates are execution
contracts, not alternative memory estimators. Each suite identifies its own
monitoring window, including video's separate audio/visual lifecycles.

### Confidence intervals

Eligible development, validation, and locked sample-based aggregates use the
shared percentile-bootstrap procedure, calculated offline from retained records:

1. Resample complete independent units with replacement 10,000 times, using
   seed `42`. The benchmark declares its unit and eligible inventory: source
   documents, source-grouped clips/videos, or queries within a vector workload
   cell. Retain all dependent pages, questions, outputs, attempts, failures,
   and windows inside that unit; do not prefilter to successful records.
2. Recalculate the metric's declared statistic on each draw: its macro-average,
   pooled-count ratio, percentile, or other defined aggregate. A unit drawn
   twice contributes twice; do not deduplicate drawn copies by ID.
3. Take the 2.5th and 97.5th percentiles of defined draw statistics as 95% bounds.
   Paired comparisons use the same aligned source draws for both candidates
   and recalculate the difference on each draw.

Record contributing and original independent-source support, not the number of
repeated requests or resampled copies. More draws stabilize arithmetic but add
no independent evidence or inference. A CI describes source-sampling uncertainty,
not the range of individual scores, a latency percentile, or certainty on future
inputs. One source repeatedly resampled cannot establish new support.

Repeated attempts remain grouped within the source on every draw. Recalculate
the estimator from those records, not by averaging separately calculated
attempt-specific confidence bounds. Bootstrap arithmetic uses saved outputs,
labels, and timing records; it launches no new model inference or judging.
Intervals describe the declared estimator under the frozen execution protocol,
not all possible future generation seeds or configurations.

Freeze minimum CI support by metric/slice after data review and before an
authoritative comparison, including development. Earlier exploratory runs can
retain descriptive points but cannot claim authoritative intervals from an
unfrozen support policy. Keep a defined point with null bounds and a reason when support is
insufficient. Record defined/undefined draw counts; an undefined draw is excluded
only from that metric's bounds, not from other defined metrics or retained rows.
With sufficient support, equal calculated bounds such as `[0,0]` remain valid
observed bounds with a degeneracy flag, not a claim of future certainty. Artifacts
keep null bounds/reasons; MLflow omits the corresponding scalar keys.

Support policy identifies the independent source groups, required attempt and
evaluation coverage, and minimum independent eligible/contributing sources for
each listed metric or slice. Forty independent papers with ten questions and
three attempts each still supply forty independent papers, not 1,200 sources.
Reviewed grouping can reduce that count further. Numeric minima remain pending
where the protocol/review checklist says so; no universal threshold is implied.
Source count alone does not establish representative coverage. Freeze the
policy before inspecting authoritative comparison results, not to obtain
favorable bounds after the fact.

Smoke, fixed identifiers/configuration counts, and one cold-load, index-build,
or resource-peak observation have no authoritative interval. Each benchmark's
CI applicability section or table identifies which sample-based aggregates qualify;
this procedure does not add intervals to unlisted descriptors or diagnostics.

## Aggregation and interpretation

Aggregate metrics never replace sample rows. Development, validation, and locked
reports retain each metric's defined aggregate: a macro-average, pooled-count
ratio, named percentile, or other specified statistic. Report contributing and
independent sample counts, failures, result status, and a 95% interval when
eligible and sufficiently supported. Inapplicable or unavailable values retain
their reasons; they do not silently become numeric zeros. Smoke values, counts,
statuses, fixed identifiers, and single operational observations do not have
authoritative intervals. An engineer reviews the complete evidence; EduMind
does not combine unrelated metrics into a weighted overall score or promote a
candidate automatically.

### A missing value is not a missing run

| Case | Stored result | What remains visible |
|---|---|---|
| A valid result recovers none of the required evidence | A normal numeric zero for recall | The valid ranking and its evidence matches. |
| The measured request fails before a valid ranking exists | Null/unavailable quality | Scheduled input, attempt failure, error, and any completed independent measurements. |
| An optional formula slice has no formula questions | Null/inapplicable slice aggregate | Zero eligible support and the slice's reason. |
| Required attempts were never executed | Incomplete execution | Actual completed outcomes and explicit missing/unexecuted records. |

Artifacts preserve the rows and reasons in every case. A numeric aggregate
uses only values defined under that metric's contract, while support counts
still account for all scheduled and eligible work. Partial required quality is
not a complete selection result merely because its surviving average is high.
One metric becoming unavailable does not erase another independently completed
measurement, such as cold loading or sampled RAM.
