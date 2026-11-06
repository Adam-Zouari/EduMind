# Final RAG metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Scoring inputs, eligibility, and result status](#scoring-inputs-eligibility-and-result-status)
- [Request failures and generation-stage eligibility](#request-failures-and-generation-stage-eligibility)
- [Citation repeatability on retrieved evidence](#citation-repeatability-on-retrieved-evidence)
- [Complete-system operations](#complete-system-operations)
- [Worked result interpretation](#worked-result-interpretation)
- [Aggregation and confidence intervals](#aggregation-and-confidence-intervals)

Final RAG reports one frozen RAG system over a reviewed canonical corpus, not
another candidate comparison or raw-upload-to-answer performance. Extraction
and offline corpus/index preparation are outside request scope.
These are the approved definitions; the [implementation boundary](methodology.md#implementation-status)
identifies runner gaps. Examples are illustrative, not measured results.

The catalogue describes the locked evaluation. CPU/CUDA smoke checks the same
selected system on fixtures without authoritative intervals; integrated preflight
records hardware qualification, not answer-quality estimates. Neither phase
introduces another candidate comparison. Their evidence is described in the
[run hierarchy](methodology.md#results-and-mlflow-structure).

Automated semantic evaluation uses the frozen, calibrated LLM judge from
Generation; benchmark code calculates the resulting metrics and aggregates.

## Metric summary

### Retrieval quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| nDCG@3/@5 | Diagnostic | Does the selected system rank evidence-rich chunks early? | Higher |
| Evidence-unit Recall@3/@5 | Diagnostic | How much required evidence reaches the selected context? | Higher |
| Evidence-token Precision@3/@5 | Diagnostic | How concentrated is that context in verified evidence? | Higher |
| Alpha-nDCG@3/@5 | Diagnostic | How early does the ranking provide distinct evidence? | Higher |
| Candidate-pool Evidence-unit Recall@20 | Diagnostic | How much required evidence was available before reranking? | Higher |

Report final-ranking quality only at cutoffs supported by the frozen top-K;
pool recall uses its separate frozen top-20 depth. Detailed definitions are in
the component references linked below; they are not redefined here.

### Generation quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Faithfulness | Primary | Are the generated factual claims supported by the supplied evidence? | Higher |
| Factual Correctness F1 | Primary | Does the answer include the required facts without introducing incorrect ones? | Higher |
| Answer Relevancy | Primary | Does the response directly address the question that was asked? | Higher |
| Citation F1 | Primary | Are the explicit citations both correct and complete? | Higher |
| Factual Correctness Precision | Diagnostic | What share of the generated factual claims are correct? | Higher |
| Factual Correctness Recall | Diagnostic | What share of the required gold claims are correctly included? | Higher |
| Citation Precision | Diagnostic | What share of the produced citations identify required evidence? | Higher |
| Citation Recall | Diagnostic | What share of the required gold evidence is covered by valid citations? | Higher |
| Citation Support | Diagnostic | What share of generated factual claims is supported by the combined valid cited passages? | Higher |

### Behavioral validity and reliability

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Response Validity Rate | Diagnostic | How often does an answerable question receive the required valid answer behavior? | Higher |
| Refusal Validity Rate | Diagnostic | How often does an unanswerable question receive the exact required refusal behavior? | Higher |
| Malformed Output Rate | Diagnostic | How often does a normally completed generation produce structurally invalid output? | Lower |
| End-to-End Request Failure Rate | Diagnostic | How often does an executed complete request fail to produce a structurally usable response? | Lower |
| Runtime Error Rate | Diagnostic | How often does measured inference encounter a runtime execution error? | Lower |
| Timeout Rate | Diagnostic | How often does generation reach the frozen wall-clock timeout? | Lower |
| Context-Limit-Reached Rate | Diagnostic | How often does generation exhaust the available model context before EOS? | Lower |

Behavioral validity and reliability are metric families, not automatic
qualification gates. Their diagnostic rates explain the one frozen system for
reporting; the required execution and measurement gates are separate. Generator
rates use stage eligibility below, not all complete-request slots indiscriminately.

### Repeatability

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Repeat Status Agreement | Diagnostic | Across fixed sampling seeds, does the model consistently answer or refuse? | Higher |
| Repeat Citation Agreement | Diagnostic | Across fixed sampling seeds, does the model select the same evidence? | Higher |
| Repeat Semantic Agreement | Diagnostic | Across fixed sampling seeds, do the visible answers preserve the same meaning? | Higher |

### Generation operations and workload

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Time to First Token p50/p95 | Operational | From the generation call, how long until the first model token, possibly reasoning? | Lower |
| Decode Throughput | Operational | How many native-tokenizer output tokens are produced per measured decoding second? | Higher |
| Cold Model-Load Time | Operational | How long does a fresh worker need to make the generator ready? | Lower |
| Prompt Tokens | Workload descriptor | How much input does the complete rendered request consume? | Descriptive |
| Context Tokens | Workload descriptor | How much supplied evidence text accompanies the question? | Descriptive |
| Reasoning Tokens | Workload descriptor | How much exposed reasoning did the model generate? | Descriptive |
| Visible-Answer Tokens | Workload descriptor | How long is the visible answer string? | Descriptive |
| Total Output Tokens | Workload descriptor | How many generated-token events did the request produce? | Descriptive |

### Integrated operations

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Server-Call Latency p50/p95 | Operational | How long does the caller wait for the selected vector server? | Lower |
| Retrieval Latency p50/p95 | Operational | How long does the complete selected retrieval/reranking path take? | Lower |
| Generation Latency p50/p95 | Operational | From the validated ranking, how long do packing, generator switching, and generation take? | Lower |
| Complete End-to-End Latency p50/p95 | Operational | How long does a request over the prepared corpus take to deliver its complete response? | Lower |
| Peak Process-Tree RAM | Operational | What peak host memory does the integrated lifecycle use? | Lower at equal quality and scope |
| Peak Device VRAM | Operational | What peak assigned-device memory does the integrated lifecycle use? | Lower at equal quality and scope |

The integrated resource windows are defined locally. They are not sums of
component peaks or additional copies of isolated generation resource metrics.

### Inherited metric definitions

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
- [Candidate-pool Evidence-unit Recall@20](../../vectordb/metrics.md#candidate-pool-evidence-unit-recall20):
  required-unit coverage in the actual verified first-attempt pool before
  reranking, after fusion when selected. It is not exact-neighbour ANN recall.
- [Semantic quality](../generation/metrics.md#semantic-quality-and-judge-responsibilities):
  Faithfulness, Factual Correctness Precision/Recall/F1, and Answer Relevancy.
- [Citation quality](../generation/metrics.md#citation-quality): Citation
  Precision/Recall/F1. ID/unit coverage is not claim-level citation entailment.
- [Citation Support](../generation/metrics.md#citation-support): the same
  deduplicated atomic generated claims assessed against only combined valid
  cited supplied passages; key `quality.citation_support`. Its no-claim,
  unanswerable, unknown-ID, unavailable-evaluation, and subset-consistency rules
  apply unchanged. Gold verification text is not injected into support evidence.
- [Behavioral validity](../generation/metrics.md#response-refusal-and-output-validity):
  Response Validity, Refusal Validity, and Malformed Output Rates.
- [Reliability](../generation/metrics.md#generation-reliability): Runtime Error,
  Timeout, and Context-Limit-Reached event definitions, with Final's local
  [stage-eligible denominators](#request-failures-and-generation-stage-eligibility).
- [Repeatability](../generation/metrics.md#repeatability-1): Repeat Status,
  Citation, and Semantic Agreement over three scheduled request/output pairs.
  Citation comparison requires [stable identities](#citation-repeatability-on-retrieved-evidence),
  not raw request-local labels.
- [Generation operations](../generation/metrics.md#operational-measurement)
  and [workload descriptors](../generation/metrics.md#workload-descriptors):
  cold load, TTFT, decode throughput, and native token descriptors. Per-response
  claim/citation counts and finish-reason inventories remain diagnostic artifacts,
  not extra MLflow scalar families. Integrated latency boundaries are defined below.

Retrieval reports `@3` for frozen `top_k=3`, and `@3` and `@5` for frozen
`top_k=5`. These are cutoffs of one ranking, not separate systems or selection
rounds. Generation receives only the frozen top-K blocks; a three-block system
has no unexecuted `@5` result. Pool recall uses the frozen top-20 pool, not the
selected top-K context. Retrieval diagnostics and the four primary answer
metrics remain separate, with no weighted universal score or new database
comparison. Metric roles do not create automatic selection or qualification gates.

## Scoring inputs, eligibility, and result status

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
| Valid first pool exists but reranking fails | Preserve pool recall; final-ranking and dependent generation quality are unavailable. |
| Retrieval or live prompt preparation fails before generation | Record a failed complete request and an explicit downstream skip, not a generator crash or refusal. |
| Generator error, timeout, output-context termination, or structural schema failure | Dependent generation quality is unavailable. Preserve actual outcome, partial output, and independently completed measurements. |
| Valid refusal on an answerable question | Apply generation's evaluated claim-free and missed-fact/citation rules, not runtime-failure treatment. |

Unknown citation IDs remain citation-ID validity failures, not malformed output;
inherit the generator's deterministic ID scoring and behavioral checks. Native
prompt compatibility is checked on the actual packed retrieved context under
the [shared input contract](../../data-validation.md#generation-references-and-model-input-compatibility).

A retrieval failure can prevent generation from executing. An unexecuted
generation is not an observed generator crash or refusal. Valid earlier answers,
completed build/load times, observed TTFT, and scoped resource samples survive
later failure under the shared retention rules. Required semantic labels and
integrated instrumentation are not optional reporting fields.

Use the [frozen-index and integrity contract](methodology.md#retrieval-integrity-and-frozen-index-checks).
Valid ANN pool differences are not failures of equality to an exact-search
reference. IDs, provenance, pool size, reranker permutation, and selected-prefix
checks still apply. Later attempts supply timing and response repeatability,
not additional retrieval-quality samples or replacement first observations.

## Request failures and generation-stage eligibility

### End-to-End Request Failure Rate

**Question:** How often does a complete request fail operationally over the
prepared corpus?

For a question with its complete three-attempt schedule executed and recorded:

```text
question End-to-End Request Failure Rate = failed complete requests / scheduled requests
```

A complete-request failure includes retrieval/integration-integrity failure,
live packing or native-input rejection, per-request model-transition failure,
generator runtime error, timeout, output-context termination, or a normally
completed but structurally unusable response. Record the actual failing stage
and downstream execution status, not a separate metric for each cause. An
incorrect but schema-usable answer, invalid citation ID, or wrong refusal is a
quality/behavior result, not by itself an operational request failure. Offline
judge failure leaves evaluation incomplete without changing request status.
This success boundary also governs complete-request latency.
A request that executes and fails upstream is an observed request failure even
when generation is skipped; a wholly unexecuted scheduled request is not an
observed failure.

**Example:** One retrieval failure, one generator timeout, and one structurally
usable answer yield `2 / 3` request failures. The retrieval failure records a
generation skip; only the timeout is a generator timeout. Average question
rates within each document, then documents; do not substitute a pooled corpus
failure fraction for this document-macro estimator.

**Range and direction:** `[0, 1]`; lower is better. All three executed requests
failing gives `1`, not missing failure evidence or three generator crashes.

**Edge cases and availability:**

There is no additional metric-specific exception. Apply the
[shared execution-accounting](../../metrics.md#execution-accounting-and-result-completeness)
and [availability rules](../../metrics.md#shared-availability-rules), without
shrinking the scheduled denominator to surviving requests.

### Generation-stage rates

**Question:** Among slots eligible for generation, how often does the generator
encounter an execution error, timeout, or output-context termination?

Retain the [generation event definitions, examples, and metric-specific edges](../generation/metrics.md#generation-reliability),
but freeze their Final RAG denominator as follows. A generation slot is eligible
when that request has completed valid retrieval, evidence packing/native-input
checks, and required generator preparation, and requires the generation call.
Count a call that raises immediately as executed generation. Explicit upstream
or preparation failures exclude that slot from generator reliability denominators
and remain complete-request failures. Missing stage/execution records never
establish an upstream skip or a successful generation.

```text
question Runtime Error Rate = generation execution errors / generation-eligible slots
question Timeout Rate = generation timeouts / generation-eligible slots
question Context-Limit-Reached Rate = output-context terminations / generation-eligible slots
```

Every eligible slot must execute and have its outcome recorded for a complete
question rate. Apply question-then-document macro aggregation to the defined
rates, retaining eligible, executed, skipped, failed, and unavailable counts.
These are conditional generator rates, not complete-system failure percentages.

**Example:** The three requests in the example above have two eligible generator
slots: one timeout and one normal completion. Generator Timeout Rate is `1 / 2`,
Runtime Error Rate `0 / 2`, and Context-Limit-Reached Rate `0 / 2`, while complete
request failure is `2 / 3`. If instead a generator OOM replaces the timeout,
Runtime Error Rate is `1 / 2` and Timeout Rate `0 / 2`.

**Range and direction:** Each rate is `[0, 1]`; lower is better when defined.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No generator slot is eligible because all requests have recorded upstream failures | Three retrieval failures and three explicit generation skips | Generator rates `null/inapplicable`, zero eligible support | Complete-request failure is still `1`; no generator execution was observed. |

Other missing outcomes or eligibility evidence follow the
[shared execution-accounting](../../metrics.md#execution-accounting-and-result-completeness)
and [availability rules](../../metrics.md#shared-availability-rules); absence of
records cannot establish the known zero-eligible case above.

Response/Refusal Validity retain the full scheduled complete-request denominator
on their answerable/unanswerable slices. A known executed request failure earns
no valid answer/refusal behavior, even when generation is skipped. Malformed
Output Rate remains conditional on normally completed generator outputs; a skip
is not malformed output. Scheduled response pairs containing an executed request
failure contribute zero under the [shared pairwise rule](../../metrics.md#pairwise-repeatability);
missing requests or required judge labels remain incomplete/unavailable rather
than manufactured disagreements. No stage denominator turns missing required
answer quality into complete evidence.

## Citation repeatability on retrieved evidence

**Question:** Do repeated outputs cite the same evidence content despite any
changes in request-local evidence labels?

Reuse [Repeat Citation Agreement](../generation/metrics.md#repeat-citation-agreement)'s
Jaccard calculation, answerable eligibility, scheduled three-pair denominator,
and empty/failure conventions. Change only the
comparison representation required by retrieved contexts:

1. Resolve each known citation label through that attempt's retained supplied
   evidence mapping to a stable identity: frozen corpus version, canonical
   document/chunk ID, source offsets, and content checksum.
2. Deduplicate these resolved identities. Evidence rank and local labels such
   as `E1` are not identity; identical chunks remain the same evidence if reordered.
3. Represent each unknown well-typed label separately as an unknown-label
   sentinel containing its literal value. It must not collide with any valid
   chunk identity, and unknown labels are not silently discarded.
4. Calculate Jaccard over the resulting sets, then aggregate the scheduled pair
   scores within question and source document as for generation.

**Example:** Attempt 1 maps `E1` to chunk A and `E2` to chunk B; attempt 2 reverses
the labels. Citing A as `[E1]` then `[E2]` gives agreement `1`, not `0`.
Conversely `[E1]` then `[E1]` cites A then B and gives `0`, not `1`. Matching
unknown-label sentinels can agree as repeated errors, but supply no valid
evidence, Citation Support, or citation-quality credit.

**Range and direction:** `[0, 1]`; higher is better. Agreement measures repeated
selection, not correctness, gold coverage, or semantic support.

**Edge cases and availability:** Both evaluated empty sets still give pair
agreement `1`, not good citation quality. A missing required mapping for a
known label is unavailable comparison evidence, not an unknown-label sentinel
or an empty set. Unanswerable questions have no citation-repeatability task.
The [shared availability](../../metrics.md#shared-availability-rules) and
[pairwise rules](../../metrics.md#pairwise-repeatability) apply.

## Complete-system operations

Warm request timing begins with the frozen searchable corpus ready. Offline
corpus/index preparation, one-time system initialization, the shared one
complete-request warmup and offline scoring are outside measured
requests. All required per-request switching/loading and context packing remain
inside request timing, even if the residency policy reloads a model each time.
This contract does not add an ingestion-to-answer measurement.

Generation's [Cold Model-Load Time](../generation/metrics.md#cold-model-load-time)
measures only initial generator readiness, not full-system startup or index
build. Its [Time to First Token](../generation/metrics.md#time-to-first-token-p50p95)
inherits exactly the standalone model-generation-call boundary, after prompt
rendering, native tokenization, input validation, and input-device transfer.
Retrieval, packing, and generator switching also precede that call. These costs
remain in their enclosing stage and complete-request timers, not generator TTFT;
it is not user-request TTFT. An internal reasoning token does not establish a
visible answer token. Keep startup and switching traces as artifacts without
inventing additional scalar metrics.

For each latency family, take each question's median successful measured stage
duration, then p50 and p95 across contributing questions using the
[shared percentile calculation](../../metrics.md#latency-percentiles).
Keep stage support: a completed stage duration can survive a later stage's
failure, while complete-system latency requires complete-system success.
Component p95 values must not be added to obtain system p95.

Resource monitoring starts before selected-component loading and includes
initialization, warmup, required per-request transitions, and measured requests,
ending before offline judging and bootstrap work. Record all participating
processes and complete/partial lifecycle scope. Corpus/index preparation precedes
this online lifecycle; keep
its separate preparation records rather than folding build resource use into
the request-lifecycle peak.
Resource availability and partial-window reporting follow the
[shared resource rules](../../metrics.md#resource-measurements).

### Server-Call Latency p50/p95

**Question:** How long does the caller wait for the approved vector server?

Measure the logical dense-query interval for each request attempt: start at
dispatch to the selected vector server and stop after receipt and validation of
the last reply required by that frozen query. Include transport/serialization,
but not query embedding or later reranking. Ordinarily this is one physical
search call. If the selected configuration requires multiple physical calls
for that logical query, retain every call trace and measure the enclosing
wall-clock interval; do not sum overlapping durations. This defines accounting
for the frozen call plan, not permission to add calls or retries. The interval
is inside retrieval, not a cost to add again.

**Example:** A `12 ms` server call within a `40 ms` retrieval remains `12 ms`,
not `52 ms` retrieval. Question observations `10,20,30,40,50 ms` give p50
`30 ms` and p95 `40 + 0.8 × (50 - 40) = 48 ms`.

**Range and direction:** Non-negative time; lower is better.

**Edge cases and availability:**

BM25-only execution with no logical vector-server query is null/inapplicable,
not zero. A failed logical query has no successful completion time; retain
elapsed-to-failure and its physical-call outcomes.
Server-internal timing is separate.

The [shared percentile rules](../../metrics.md#latency-percentiles) and
[shared availability rules](../../metrics.md#shared-availability-rules) apply.

### Retrieval Latency p50/p95

**Question:** How long does the live selected stack take to return its ranking?

Time required query preparation/embedding, Dense and/or BM25 requests, RRF when
selected, learned reranking when selected, and validated top-K selection. Server
time and any per-request embedding/reranker residency transitions are included;
cached quality pools cannot replace the live operational path. Start at retrieval
dispatch before such preparation, and stop at the validated ranked top-K.
This is the complete retrieval stack: the retrieval–reranking component page
calls the corresponding boundary Full-Stack Warm Latency, while its shorter
Retrieval Warm Latency isolates only the first-stage retriever.

**Example:** Sequential `8 ms` preparation, `12 ms` server call, `5 ms` ranking,
and `15 ms` reranking give `40 ms`. Question observations `20,30,40,50,100 ms`
give p50 `40 ms` and p95 `90 ms`, exposing the tail hidden by the median.

**Range and direction:** Non-negative time; lower is better.

**Edge cases and availability:**

Failed retrieval has no successful-completion value. Completed retrieval
remains measured if generation later fails, with that outcome retained.

The [shared percentile rules](../../metrics.md#latency-percentiles) and
[shared availability rules](../../metrics.md#shared-availability-rules) apply.

### Generation Latency p50/p95

**Question:** From the validated retrieved ranking, how long does preparing and
completing generation take?

Start immediately after validated top-K selection. Include evidence
packing/numbering, required switching into generator residency, prompt
preparation/native-input checks, tokenization/prefill, reasoning and visible
decoding, parsing, citation validation, and synchronization. Stop when the
structurally usable response is ready. This extends the
[standalone generation boundary](../generation/metrics.md#end-to-end-latency-p50p95)
with Final's live packing and residency work. Keep non-overlapping subintervals
in artifacts so packing/switching are neither lost nor counted twice.

**Example:** `0.01 s` packing, `0.80 s` required generator switching, and
`4.00 s` prompt/inference/response processing give generation latency `4.81 s`.
Question observations `2,3,4,5,10 s` give p50 `4 s` and p95
`5 + 0.8 × (10 - 5) = 9 s`. A timeout after emitting tokens can supply TTFT
but not successful generation latency.

**Range and direction:** Non-negative seconds; lower is better.

**Edge cases and availability:**

No successful completions gives null/unavailable. Retain failure/termination
times separately. Judge time is excluded.

The [shared percentile rules](../../metrics.md#latency-percentiles) and
[shared availability rules](../../metrics.md#shared-availability-rules) apply.

### Complete End-to-End Latency p50/p95

**Question:** How long does the caller wait for a complete answer to a warm question?

Use an outer wall timer from question-request entry through retrieval, context
packing/numbering, generation, parsing, and citation validation, including dispatch
overhead, every selected per-request model transition, and required synchronization.
Stop at complete structurally usable response delivery. Correctness and valid
citation-ID behavior are evaluated separately; an ordinary wrong but usable
response can complete. This is not a sum of stage percentiles.

**Example:** `0.04 s` retrieval, `4.81 s` generation (including the packing and
switching above), and `0.02 s` other dispatch work give `4.87 s` for one request.
Do not add packing/switching again or remove them as cold loading. Complete question
observations `2,3,4,5,10 s` give p50 `4 s` and p95 `9 s`.

**Range and direction:** Non-negative seconds; lower is better.

**Edge cases and availability:**

No complete successes means null/unavailable. Fast failure is not fast
completed answering; report successful coverage alongside reliability.

The [shared percentile rules](../../metrics.md#latency-percentiles) and
[shared availability rules](../../metrics.md#shared-availability-rules) apply.

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

**Edge cases and availability:**

Apply the [shared resource rules](../../metrics.md#resource-measurements) to the integrated monitoring window above.

### Peak Device VRAM

**Question:** How much total assigned-GPU memory is occupied at the integrated peak?

Use the [shared NVML device-total peak](../../metrics.md#peak-device-vram) across
the recorded integrated lifecycle with resident selected components. Do not sum
isolated component peaks. Placement is checked separately.

**Example:** Samples `700,3,200,2,900 MiB` report `3,200 MiB`. Independent
component qualification does not prove that concurrent residency fits.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
Applicable runtime gates still apply during locked execution, independently of
the preceding integrated preflight qualification.
Single-peak uncertainty follows the shared resource rules.

**Edge cases and availability:**

CPU-only execution is null/inapplicable, not zero VRAM. Missing required CUDA
telemetry is null/unavailable. Complete and partial monitoring windows follow
the [shared resource rules](../../metrics.md#resource-measurements); retain
measured peaks without representing a partial window as a complete one.

## Worked result interpretation

A frozen system can retrieve evidence yet generate an incomplete answer.
Illustrative document-macro Evidence-unit Recall `0.90` and Factual Correctness
Recall `0.60` are averages under their distinct question/document estimators,
not pooled percentages of all evidence units or facts. For example, one question
in each of two documents has correctness recall `1 / 1 = 1.00` and
`1 / 5 = 0.20` on every measured response. The macro recall is
`(1.00 + 0.20) / 2 = 0.60`, while pooled fact coverage would be
`(1 + 1) / (1 + 5) = 1 / 3`, not 60%. Inspect missed evidence and the use of
retrieved content before attributing a retrieval/answer-quality gap to generation.
Faithfulness `1.00` over evaluated claims establishes support, not completeness;
Citation Support can be lower if their supporting passages were not cited.

A `40 ms` retrieval stage inside a `4.87 s` complete request is already part
of that request's latency. It is not added to the total again. Inspect
failure/support records before interpreting conditional quality or latency.

None of these results selects a replacement system after the locked evaluation.
These values illustrate interpretation, not measured results.

## Aggregation and confidence intervals

Automated quality/rates use question-then-document macro weighting,
first-attempt retrieval quality, generation's all-three-output/pair means,
the local stage-eligible failure denominators, and each stage's latency observation
unit. Keep answerable/unanswerable and evidence-type eligible/contributing counts
plus failed, unavailable, and unexecuted records. Do not pool all questions or
claims into an alternative overall quality percentage.

Request failure rates retain all scheduled request slots; generator rates retain
only their known eligible slots, with upstream skips visible separately. The
[shared partial-evidence rules](../../metrics.md#dependent-metrics-and-partial-observations)
apply: evaluated no-claim/inapplicable ratios are not missing judging, but missing
required outputs or labels leave surviving quality diagnostic, not complete.
Keep each source's questions, three attempts, and pair records together; three
outputs do not create three independent questions. Citation Support uses the
same response/question/document estimator and CI eligibility as the other
semantic quality ratios, not a pooled generated-claim fraction.

Eligible locked automated quality/rate/repeatability aggregates receive 95%
intervals with sufficient independent support. Warm p50/p95 intervals are
conditional on source support. Apply the
[shared bootstrap and confidence-interval rules](../../metrics.md#confidence-intervals)
with reviewed independent-source groups as the resampling units, retaining all
member documents, questions, seeds, failures, and pair records. A document is
independent only when the reviewed grouping establishes it. Quality/rates retain
their document-macro statistic; latency uses the per-question stage observations defined above, not
document-average times.

Generation Decode Throughput inherits its pooled work/time estimator and
conditional interval eligibility. Recalculate that ratio on each aligned source
draw rather than applying quality's document-macro averaging to stream rates.

The shared support and degeneracy rules apply to reviewed independent-source support;
calculated bounds such as `[1,1]` do not establish certainty. Cold load, resource
peaks, workload counts, and fixed settings use the shared interval exclusions.
Numeric independent-source minima remain pending reviewed data. Freeze them
at the shared deadlines; this document supplies no thresholds or calibration
result.
