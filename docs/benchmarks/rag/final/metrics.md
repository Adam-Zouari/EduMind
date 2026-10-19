# Final RAG and human review metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Automated metric map](#automated-metric-map)
- [Eligibility and result status](#eligibility-and-result-status)
- [Complete-system operations](#complete-system-operations)
- [Human review metrics](#human-review-metrics)
- [Extraction-to-RAG paired differences](#extraction-to-rag-paired-differences)
- [Support and confidence intervals](#support-and-confidence-intervals)

Final RAG reports one frozen complete system, not another candidate comparison.
These are the approved definitions; the [implementation boundary](methodology.md#implementation-boundary)
identifies runner/review gaps. Examples are illustrative, not measured results.

## Automated metric map

Reuse the component definitions, examples, and edge rules, rather than
implementing a second version of the same metrics.

Retrieval adopts the [shared evidence-quality rules](../../metrics.md#retrieval-evidence-quality).
Its normalization source is the selected system's complete frozen chunk corpus,
not the retrieved pool. Component links below provide applications and examples.

- [nDCG@3/@5](../chunking_embedding/metrics.md#ndcg35): earlier placement of
  chunks covering more required evidence.
- [Evidence-unit Recall@3/@5](../chunking_embedding/metrics.md#evidence-unit-recall35):
  coverage against the full required-unit inventory, not only retrieved units.
- [Evidence-token Precision@3/@5](../chunking_embedding/metrics.md#evidence-token-precision35):
  evidence concentration in returned `tiktoken:cl100k_base` token occurrences,
  not model-native generation tokens.
- [Alpha-nDCG@3/@5](../chunking_embedding/metrics.md#alpha-ndcg35): diagnostic
  novelty with `alpha=0.5` under the shared normalization and capping rules.
- [Semantic quality](../generation/metrics.md#semantic-quality-and-judge-responsibilities):
  Faithfulness, Factual Correctness Precision/Recall/F1, and Answer Relevancy.
- [Citation quality](../generation/metrics.md#citation-quality): Citation
  Precision/Recall/F1. ID/unit coverage is not claim-level citation entailment.
- [Behavioral validity](../generation/metrics.md#response-refusal-and-output-validity):
  Response Validity, Refusal Validity, and Malformed Output Rates.
- [Reliability](../generation/metrics.md#generation-reliability): Runtime Error,
  Timeout, and Context-Limit-Reached Rates.
- [Repeatability](../generation/metrics.md#repeatability-1): Repeat Status,
  Citation, and Semantic Agreement over three scheduled generation pairs.
- [Generation operations](../generation/metrics.md#operational-measurement)
  and [workload descriptors](../generation/metrics.md#workload-descriptors):
  cold load, TTFT, decode throughput, and native token descriptors. Per-response
  claim/citation counts and finish-reason inventories remain diagnostic artifacts,
  not extra MLflow scalar families. Integrated latency boundaries are defined below.

Retrieval reports `@3` for frozen `top_k=3`, and `@3` and `@5` for frozen
`top_k=5`. These are cutoffs of one ranking, not separate systems or selection
rounds. Generation receives only the frozen top-K blocks; a three-block system
has no unexecuted `@5` result. The primary retrieval and generation dimensions
remain separate, with no weighted universal score.

## Eligibility and result status

Retrieval quality, semantic answer quality, citation quality, and Repeat Citation
Agreement use reference-answerable questions. The complete path also executes
unanswerable questions: Refusal Validity uses that slice; general reliability
and status/semantic repeatability use both. Gold answerability, required claims,
and required evidence remain fixed even when retrieval misses evidence.
Do not reduce citation-recall denominators to the units actually retrieved.

Retrieval quality uses the designated first measured ranking. Generation
quality averages all three scheduled seeds `42`, `43`, and `44` under its own
contract. Later rankings do not replace or average into first-ranking quality.
Apply the [shared availability rules](../../metrics.md#shared-availability-rules)
to each stage and measured seed, retaining the component metric's own known-empty
conventions and eligible/contributing support.

| Case | Required reporting |
|---|---|
| Valid first ranking misses all complete units | Normal numeric retrieval scores, including zero unit recall; partial token evidence can still give nonzero token precision. |
| Generator error, timeout, output-context termination, or structural schema failure | Dependent generation quality is unavailable. Preserve actual outcome, partial output, and independently completed measurements. |
| Valid refusal on an answerable question | Apply generation's evaluated claim-free and missed-fact/citation rules, not runtime-failure treatment. |

Unknown citation IDs remain reference-validity failures, not malformed output;
inherit the generator's deterministic ID scoring and behavioral checks. Native
prompt compatibility is checked on the actual packed retrieved context under
the [shared input contract](../../data-validation.md#generation-references-and-model-input-compatibility).

A retrieval failure can prevent generation from executing. An unexecuted
generation is not an observed generator crash or refusal. Valid earlier answers,
completed build/load times, observed TTFT, and scoped resource samples survive
later failure under the shared retention rules. Required semantic labels, review
ratings, and integrated instrumentation are not optional reporting fields.

## Complete-system operations

Warm request timing begins with the searchable corpus ready. Ingestion,
corpus/index preparation, loading, warmup, offline scoring, and human review
are separate costs, not hidden warm latency. An additional source-ingestion-to-answer
timer must identify its different boundary explicitly.

For each latency family, take each question's median successful measured stage
duration, then p50 and p95 across contributing questions using the
[shared percentile calculation](../../metrics.md#latency-percentiles).
Keep stage support: a completed stage duration can survive a later stage's
failure, while complete-system latency requires complete-system success.
Component p95 values must not be added to obtain system p95.

Resource monitoring starts before selected-component loading and includes
initialization, warmup, and measured requests, ending before offline judging
and bootstrap work. Record all participating processes and complete/partial
lifecycle scope, including preparation work executed inside that window.
Resource availability and partial-window reporting follow the
[shared resource rules](../../metrics.md#resource-measurements).

### Server-Call Latency p50/p95

**Question:** How long does the caller wait for the approved vector server?

Time dispatch through receipt/validation of the vector-search response, including
transport/serialization, but not query embedding or later reranking. This is
inside retrieval, not a cost to add again. Retain call-level traces if several
calls are required; their request-level scope must be explicit before aggregation.

**Example:** A `12 ms` server call within a `40 ms` retrieval remains `12 ms`,
not `52 ms` retrieval. Question observations `10,20,30,40,50 ms` give p50
`30 ms` and p95 `40 + 0.8 × (50 - 40) = 48 ms`.

**Range and direction:** Non-negative time; lower is better. BM25-only execution
with no server call is null/inapplicable, not zero. A failed call has no successful
completion time; retain elapsed-to-failure. Server-internal timing is separate.

### Retrieval Latency p50/p95

**Question:** How long does the live selected stack take to return its ranking?

Time required query preparation/embedding, Dense and/or BM25 requests, RRF when
selected, learned reranking when selected, and validated top-K selection. Server
time is included; cached quality pools cannot replace the live operational path.
This is the complete retrieval stack: the retrieval–reranking component page
calls the corresponding boundary Full-Stack Warm Latency, while its shorter
Retrieval Warm Latency isolates only the first-stage retriever.

**Example:** Sequential `8 ms` preparation, `12 ms` server call, `5 ms` ranking,
and `15 ms` reranking give `40 ms`. Question observations `20,30,40,50,100 ms`
give p50 `40 ms` and p95 `90 ms`, exposing the tail hidden by the median.

**Range and direction:** Non-negative time; lower is better. Failed retrieval
has no successful-completion value. Completed retrieval remains measured if
generation later fails, with that outcome retained.

### Generation Latency p50/p95

**Question:** After evidence is ready, how long does a complete generator request take?

Use the [generation warm-request boundary](../generation/metrics.md#end-to-end-latency-p50p95):
prompt preparation, tokenization/prefill, reasoning and visible decoding, parsing,
and citation validation. Account for context packing/numbering in the complete
trace rather than losing it between stage timers.

**Example:** Question observations `2,3,4,5,10 s` give p50 `4 s` and p95
`5 + 0.8 × (10 - 5) = 9 s`. A timeout after emitting tokens can supply TTFT
but not successful generation latency.

**Range and direction:** Non-negative seconds; lower is better. No successful
completions gives null/unavailable. Retain failure/termination times separately.
Judge and human-review time are excluded.

### Complete End-to-End Latency p50/p95

**Question:** How long does the caller wait for a complete answer to a warm question?

Use an outer wall timer from question-request entry through retrieval, context
packing/numbering, generation, parsing, and citation validation, including dispatch
overhead and required synchronization. This is not a sum of stage percentiles.

**Example:** `0.04 s` retrieval, `0.01 s` packing, `4.00 s` generation, and
`0.02 s` other dispatch work give `4.07 s` for one request. Complete question
observations `2,3,4,5,10 s` give p50 `4 s` and p95 `9 s`.

**Range and direction:** Non-negative seconds; lower is better. No complete
successes means null/unavailable. Fast failure is not fast completed answering;
report successful coverage alongside reliability.

### Peak Process-Tree RAM

**Question:** What peak host memory is observed for integrated execution?

Use the [shared process-tree RSS peak](../../metrics.md#peak-process-tree-ram)
for the worker and its children. Identify separately managed vector-server
processes and their memory scope rather than silently excluding them or
double-counting them as children.

**Example:** Simultaneous process-tree totals `2,000,3,500,3,000 MiB` report
`3,500 MiB`. Adding a separately managed server's independent peak would not
measure a simultaneous system peak.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
Apply the shared resource rules to the integrated monitoring window above.

### Peak Device VRAM

**Question:** How much total assigned-GPU memory is occupied at the integrated peak?

Use the [shared NVML device-total peak](../../metrics.md#peak-device-vram) across
the recorded integrated lifecycle with resident selected components. Do not sum
isolated component peaks. Placement is checked separately.

**Example:** Samples `700,3,200,2,900 MiB` report `3,200 MiB`. Independent
component qualification does not prove that concurrent residency fits.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
Applicable runtime gates still apply; this creates neither Final RAG preflight
nor an extraction VRAM cap. Single-peak uncertainty follows the shared resource rules.

## Human review metrics

One blinded reviewer rates 20 anonymous answer items from the one frozen system.
The reviewer sees the question, answer/status/citations, accepted answer, required
facts, supplied evidence, and frozen evaluator-only verification material, with
system identity hidden. Human review is
reporting-only and cannot reopen selection.

Each criterion retains its own item ratings and report mean:

```text
criterion mean = sum of that criterion's item ratings / number of rated items
```

Keep rating counts and rows. Do not combine criteria or normalize the four
`0–2` means into automated `[0,1]` quality scores. A Faithfulness sum of `30`
across 20 fully rated items means `1.5 / 2`, not `75% of claims supported`.

### Human Faithfulness

**Question:** Are the material factual claims supported by the supplied evidence?

Assign `0` when the support requirement is not met, `1` when partly met, and `2`
when fully met. Background knowledge cannot substitute for context support.

**Example:** Evidence states only enrollment of 500. "500 participants, followed
for six months" includes an unsupported duration, illustrating `1`. Stating only
the supported enrollment meets faithfulness (`2`) even if duration is required
for a complete answer.

**Range and direction:** Item ratings `{0,1,2}`, mean `[0,2]`; higher is better.
This ordinal rating is not an automated supported-claim fraction. Refusal and
missing-review cases follow the common rules below.

### Human Answer Correctness

**Question:** Is the answer factually correct against the accepted answer and
frozen verification material?

Assign `0` for not met, `1` for partly met, and `2` for fully met. Correctness
is distinct from supplied-context support and essential-part completeness.

**Example:** Required facts are 500 participants and six months. "500 participants
over eight months" is partly correct (`1`); "800 participants over eight months"
meets neither required fact (`0`).

**Range and direction:** Item ratings `{0,1,2}`, mean `[0,2]`; higher is better.
Missing verification material is a review-data problem, not an invented rating
or permission for unrecorded lookup.

### Human Completeness

**Question:** Does the answer cover every essential part of the question?

Compare with frozen required facts: `0` for no required coverage, `1` for partial
coverage, and `2` for full coverage. Verbosity and repeated facts add no coverage.

**Example:** Asked for enrollment and duration, "500 participants" receives `1`;
"500 participants over six months" covers both and receives `2`. A faithful
short answer can still be incomplete.

**Range and direction:** Item ratings `{0,1,2}`, mean `[0,2]`; higher is better.
Missing required-fact annotations prevent complete review; they do not make
every answer complete by default.

### Human Citation Accuracy

**Question:** Do cited evidence blocks support the claims for which they are offered?

Assign `0` for not met, `1` for partly met, and `2` for fully met. The schema
contains a response-level citation array, not a claim-to-ID map. Review that
array and the visible claims without inventing inline citation attachments.
This adds semantic support assessment beyond automated ID/unit coverage.

**Example:** `E1` verifies enrollment and `E2` duration. An answer giving both
facts with `citations: ["E1"]` offers partial citation support (`1`); citing
both appropriate blocks meets the requirement (`2`). Legal IDs alone do not
establish support.

**Range and direction:** Item ratings `{0,1,2}`, mean `[0,2]`; higher is better.
No citations cannot establish citation support for a substantive answer requiring
evidence. An empty refusal citation list is not automatically a perfect rating.

### Human Answerability Correctness

**Question:** Did the system answer or refuse according to frozen answerability?

Assign `1` for correct answer/refusal behavior and `0` otherwise. Its mean is
`correct behavior items / rated items`, not balanced accuracy or factual accuracy.

**Example:** A substantive answer on an answerable question can have correct
answer behavior even if a fact is wrong; correctness is separate. A required
valid refusal on an unanswerable question earns `1`. Correct behavior on 16 of
20 items gives `16 / 20 = 0.80`.

**Range and direction:** Item ratings `{0,1}`, mean `[0,1]`; higher is better.
A retrieval miss does not change source answerability. No response is not an
invented human-reviewed refusal.

### Review completeness and edge rules

Freeze which measured response becomes each anonymous item before inspecting
quality. The design does not yet specify that response-selection rule. Do not
guess it, choose a best seed, silently drop an item, or replace failure with a
later success. The general `0–2` rubric also needs explicit instructions for
the four substantive criteria on correct refusals before review. A correct
refusal does not imply four perfect scores or permission to shrink denominators.
Until these rules are frozen, the human review contract is not ready for a
complete authoritative report.

Missing/duplicate/unknown item IDs, missing/non-integer/out-of-range ratings, or
mismatched evidence identity prevent a complete review. Retain available rows
as partial diagnostics with null missing ratings, not zeros. A provisional mean
on 18 of 20 items describes those 18 only. No items in an optional slice means
inapplicable, not zero quality. One reviewer supports no inter-reviewer agreement
claim. Human means are descriptive; 20 items alone do not establish a calibrated
judge, universal score, or human confidence interval.

## Extraction-to-RAG paired differences

**Question:** How does replacing verified reference text with selected extracted
text change the already-frozen system's result?

For every metric in the [confirmation methodology](methodology.md#extraction-to-rag-confirmation),
retain both branch values and calculate:

```text
paired difference = extracted-text result - verified-reference-text result
```

With complete aligned support, subtract the document-macro quality/rate results
(equivalently average their paired document differences). For p50/p95, subtract
branch percentiles on aligned observations; this is not the percentile of
per-question differences.

**Examples:** Recall `0.70 - 0.80 = -0.10` is a loss of 0.10 points, not a
10% relative decrease (that decrease is 12.5%). Failure Rate `0.06 - 0.02 = +0.04`
is worse, while latency `6 s - 5 s = +1 s` means slower extracted-text execution.

**Range and direction:** Differences of `[0,1]` metrics lie in `[-1,1]`; time
differences have the underlying unit and no fixed bound. Positive favors
extraction only for higher-is-better metrics; negative favors it for errors/time.
Zero means no measured change in that metric, not identical outputs or proven
absence of extraction harm.

Both branches retain the same required evidence IDs/claims and seeds. A lost
unit remains required, with an explicit unrecoverable source mapping, not an
invented empty span. A completed branch missing a unit can score zero recovery;
a failed/invalid retrieval ranking has unavailable retrieval quality, not an
observed zero. Generation's explicit known-empty conventions and independently
completed stage measurements keep their own contracts; a branch failure does
not erase them. An unavailable or inapplicable branch value makes that
metric's difference null/unavailable. Keep exclusions/common support
and label partial differences diagnostic rather than complete confirmation.

## Support and confidence intervals

Automated quality/rates inherit question-then-document macro weighting,
generation's scheduled-seed/pair means, and each stage's latency observation
unit. Keep answerable/unanswerable and evidence-type eligible/contributing counts
plus failed, unavailable, and unexecuted records. Do not pool all questions or
claims into an alternative overall quality percentage.

Eligible locked automated quality/rate/repeatability aggregates receive 95%
intervals with sufficient independent support. Warm p50/p95 intervals are
conditional on source support. Apply the
[shared bootstrap and confidence-interval rules](../../metrics.md#confidence-intervals)
with source documents as the independent units, retaining all questions, seeds,
failures, and pair records. Quality/rates retain their document-macro statistic;
latency uses the per-question stage observations defined above, not
document-average times.

Generation Decode Throughput inherits its pooled work/time estimator and
conditional interval eligibility. Recalculate that ratio on each aligned source
draw rather than applying quality's document-macro averaging to stream rates.

Confirmation resamples identical source IDs for both branches and recalculates
the extracted-minus-reference statistic per draw. Do not subtract independent
CI endpoints. A recall difference `-0.10` with paired bounds `[-0.15,-0.04]`
describes uncertainty in that difference, not individual-question loss ranges.

The shared support and degeneracy rules apply to source-document support;
calculated bounds such as `[1,1]` do not establish certainty. Cold load, resource
peaks, workload counts, and fixed settings use the shared interval exclusions.
Human ratings remain descriptive under the [review rules](#review-completeness-and-edge-rules),
not automatically interval-eligible automated scores.
