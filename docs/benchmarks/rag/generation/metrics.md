# Generation metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Implementation boundary and result status](#implementation-boundary-and-result-status)
- [Semantic quality and judge responsibilities](#semantic-quality-and-judge-responsibilities)
- [Citation quality](#citation-quality)
- [Response, refusal, and output validity](#response-refusal-and-output-validity)
- [Generation reliability](#generation-reliability)
- [Repeatability](#repeatability-1)
- [Operational measurement](#operational-measurement)
- [Workload descriptors](#workload-descriptors)
- [Worked candidate interpretation](#worked-candidate-interpretation)
- [Eligibility, aggregation, and confidence intervals](#eligibility-aggregation-and-confidence-intervals)

This page covers generation on frozen evidence and the same automated metrics
when generation is embedded in Final RAG. Once selected, calibrated, and frozen,
one pinned LLM judge supplies structured semantic labels; deterministic
benchmark code calculates the metric values and aggregates. Blinded human review
provides separate reporting-only evidence about the selected complete system.

## Implementation boundary and result status

This is the approved metric contract, not a claim that the current runner
implements it. The existing generation evaluator still exposes lexical scores
and a local sequence-classification faithfulness scorer; neither substitutes
for the claim-level calibrated judge defined here. Judge identity, calibration
acceptance evidence, prompts, and retry policy must be frozen before an
authoritative semantic report. No judge or authoritative result is designated
by these examples. See the [methodology's implementation boundary](methodology.md#implementation-boundary).

Generation deliberately differs from extraction and chunking/embedding:
quality is scored across all three scheduled seeds, not only attempt 1.
An observed candidate failure on an answerable question receives zero under
this suite's failed-quality rule, but its absent response is still stored as
null with the actual outcome, not fabricated as a completed empty answer.

| Case | Treatment |
|---|---|
| Normal, schema-valid answered output | Score answerable quality, behavioral validity, and applicable diagnostics. A valid but incorrect answer can score poorly without being malformed. |
| Observed refusal, malformed output, execution error, timeout, or context-limit termination on an answerable question | Apply the documented zero-quality rule; retain the actual output if any and distinct outcome. Runtime errors do not become malformed outputs. |
| No questions in a metric's reference-fixed eligibility slice | Null/inapplicable, with zero eligible support. |
| Judge error, missing instrumentation, or missing scheduled record | Null/unavailable or incomplete with a reason; do not fabricate candidate zero, a refusal, or agreement. |

Required attempts, labels, fields, and measurements must be accounted for before
calling a report complete. Valid observations survive later failures: an
observed TTFT, completed cold load, earlier answer score, or partial resource
peak is not erased. Artifacts retain scheduled, eligible, observed/scored,
failed, unavailable, and unexecuted counts as applicable. MLflow omits null
scalar keys; a missing key never means a measured zero.

## Metric summary

### Quality

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

### Behavioral validity and reliability

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Response Validity Rate | Diagnostic | How often does an answerable question receive the required valid answer behavior? | Higher |
| Refusal Validity Rate | Diagnostic | How often does an unanswerable question receive the exact required refusal behavior? | Higher |
| Malformed Output Rate | Diagnostic | How often does an attempted generation violate the required response schema? | Lower |
| Generation Failure Rate | Diagnostic | How often does model execution fail before producing an evaluable output? | Lower |
| Timeout Rate | Diagnostic | How often does generation reach the frozen wall-clock timeout? | Lower |
| Context-Limit-Reached Rate | Diagnostic | How often does generation exhaust the available model context before EOS? | Lower |

Behavioral validity and reliability are metric families, not automatic
qualification gates. Their diagnostic rates remain selection evidence for
engineer review; the required execution and measurement gates are separate.

### Repeatability

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Repeat Status Agreement | Diagnostic | Across fixed sampling seeds, does the model consistently answer or refuse? | Higher |
| Repeat Citation Agreement | Diagnostic | Across fixed sampling seeds, does the model select the same evidence? | Higher |
| Repeat Semantic Agreement | Diagnostic | Across fixed sampling seeds, do the visible answers preserve the same meaning? | Higher |

### Operational measurements and descriptors

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Time to First Token p50/p95 | Operational | How long does a warm request wait before generation begins? | Lower |
| End-to-End Latency p50/p95 | Operational | How long does the complete warm generation request usually take, and how slow is its tail? | Lower |
| Decode Throughput | Operational | How many native-tokenizer output tokens are produced per measured decoding second? | Higher |
| Cold Model-Load Time | Operational | How long does a fresh worker need to make the generator ready? | Lower |
| Peak Process-Tree RAM and Peak Device VRAM | Operational | What peak host-process and total device memory accompany the profile? | Lower at equal quality |
| Prompt, Context, Reasoning, Visible-Answer, and Total Output Tokens | Workload descriptor | How much native-tokenizer input and output produced the quality and latency results? | Descriptive |
| Citation Count, Generated-Claim Count, and Finish-Reason Distribution | Workload descriptor | What evidence and factual workload did the model produce, and why did generation stop? | Descriptive |

## Semantic quality and judge responsibilities

Once selected, calibrated, and frozen, the judge returns structured claim-level
labels. It extracts distinct atomic factual claims from the visible answer,
checks support against the supplied context, verifies correctness against frozen
authoritative reference material, matches required gold claims, and applies the
Answer Relevancy rubric. Benchmark code retains those labels and calculates all
ratios, F1 values, aggregates, and intervals.

An atomic claim expresses one independently checkable fact. Compound statements
are split; repeated or semantically equivalent claims count once per response.
Qualifiers such as numbers, units, dates, negation, and population must be
preserved. The required gold claims define answer completeness, not every
correct fact the response is permitted to contain. The exact extraction,
verification, and matching rules are frozen in the
[semantic judge contract](methodology.md#semantic-judge-contract).

These metrics use answerable questions. A refusal, malformed response, timeout,
context-limited response, or execution failure on an answerable question receives
zero, so difficult questions cannot disappear from the quality denominator.

### Faithfulness

**Question:** What share of the factual claims made by the answer are supported
by the evidence supplied to the generator?

```text
Faithfulness =
supported generated factual claims
----------------------------------
all generated factual claims
```

A claim is supported only when the supplied evidence entails it. General world
knowledge does not compensate for missing support in the provided context.

**Example:** An answer makes four atomic factual claims and the judge finds
support for three. Faithfulness is `3 / 4 = 0.75`.

**Range and direction:** `[0, 1]`; higher is better. An answerable response with
no evaluable factual claim receives zero.

### Factual Correctness Precision

**Question:** What share of the generated factual claims are correct according
to the frozen authoritative reference material?

```text
Factual Correctness Precision =
correct generated factual claims
--------------------------------
  all generated factual claims
```

**Example:** If four claims are generated and three are correct, precision is
`3 / 4 = 0.75`. The incorrect additional claim lowers precision even when the
three remaining claims are correct.

A fact absent from the required gold list still receives correctness credit
when the frozen verification material establishes it. It does not earn extra
recall credit or compensate for a missing required fact. An unverified addition
receives no correctness credit; the judge's background knowledge is not a
verification source. Answer Relevancy separately evaluates unrelated additions.
This source-verified precision is EduMind's frozen claim contract, not a rule
that every claim absent from a reference-answer list is automatically false.

**Range and direction:** `[0, 1]`; higher is better. An answerable response with
no evaluable factual claim receives zero.

### Factual Correctness Recall

**Question:** What share of the required gold claims are correctly included in
the answer?

```text
Factual Correctness Recall =
required gold claims correctly included
----------------------------------------
         all required gold claims
```

**Example:** If the reference requires five atomic claims and the response
correctly includes three, recall is `3 / 5 = 0.60`.

Semantic paraphrases receive credit, but each required claim counts at most once.
Repeating a fact or adding a different correct fact cannot increase recall.

**Range and direction:** `[0, 1]`; higher is better.

Every answerable reference must have at least one required atomic gold claim.
An absent or empty required-claim list is invalid reference data, not a
perfect empty-recall case. A valid substantive response covering none receives
zero; absent judge labels are unavailable evaluation, not zero matching.

### Factual Correctness F1

**Question:** Does the response balance avoiding incorrect claims with covering
the facts required for a complete answer?

```text
Factual Correctness F1 =
2 x precision x recall
----------------------
  precision + recall
```

**Example:** Precision `0.75` and recall `0.60` produce F1 approximately `0.67`.
A response that is correct but incomplete has high precision and lower recall;
a detailed response containing false additions has lower precision.

For example, the required facts are "500 participants" and "six months". The
verification source also establishes that participants were adults. An answer
stating "500 adult participants" has two correct generated claims: precision
is `1`, recall is `1 / 2`, and F1 is approximately `0.67`. The correct extra fact
is accepted, while the missing duration still lowers completeness.

**Range and direction:** `[0, 1]`; higher is better. F1 is zero when precision
or recall is zero.

### Answer Relevancy

**Question:** Does the visible response directly address the question without
being evasive or dominated by unrelated information?

The judge assigns one frozen rubric level:

| Level | Meaning |
|---:|---|
| `0` | The response does not address the question. |
| `1` | The response addresses the question only partly or contains substantial unrelated material. |
| `2` | The response directly addresses the question with no substantial unrelated material. |

Benchmark code divides the rubric level by `2` to report Answer Relevancy on
`[0, 1]`.

**Example:** A response that answers one part of a two-part question but avoids
the other receives level `1`, reported as `0.50`. Its factual completeness is
still measured separately by Factual Correctness Recall.

**Range and direction:** `[0, 1]`; higher is better.

The exact judge model version, decoding settings, prompts, rubric checksums,
schema, and retry policy must be frozen before authoritative use. Candidate
identities are hidden from the judge. Its labels must satisfy the recorded agreement
criteria for each responsibility on a human-labeled development calibration set.
A judge failure makes evaluation
incomplete. Judge latency and resources are excluded from candidate operational
metrics.

## Citation quality

Citation Precision, Recall, and F1 are calculated only for answerable questions,
which always have one or more verified gold evidence units. Unanswerable
questions are evaluated by Refusal Validity Rate and the applicable reliability
metrics rather than receiving artificially perfect citation scores.

A citation is correct when it is a valid supplied evidence-block ID and that
block completely covers at least one required gold evidence unit. Repeated IDs
count once. An unknown ID makes the response malformed, so all three reported
citation scores are zero under the malformed-output rule below. Retain the
individual citation checks in artifacts for diagnosis.

### Citation Precision

**Question:** When the model cites an evidence block, how often is that block
part of the verified evidence required for the answer?

```text
Citation Precision =
distinct correct citation IDs
-----------------------------
all distinct produced citation IDs
```

**Example:** Citations `[E1, E2, E9]` contain two correct IDs. `E9` is a valid
supplied evidence block, but it does not cover any required gold evidence unit.
Citation Precision is `2 / 3`, approximately `0.67`.

**Range and direction:** `[0, 1]`; higher is better. An answerable response with
no citations receives zero.

### Citation Recall

**Question:** How much of the required gold evidence is covered by at least one
correct citation?

```text
Citation Recall =
required gold evidence units covered by correct citations
---------------------------------------------------------
           all required gold evidence units
```

**Example:** If an answer requires evidence units `G1`, `G2`, and `G3`, and the
correct citations cover `G1` and `G2`, Citation Recall is `2 / 3`, approximately
`0.67`.

**Range and direction:** `[0, 1]`; higher is better.

No correct citations gives zero. The denominator remains the question's full
required evidence inventory, including units missing from Final RAG's retrieved
context. Restricting it to retrieved units would hide retrieval misses. Missing
gold annotations are invalid reference data, not an empty citation task.

### Citation F1

**Question:** Does the response balance citing only correct evidence with citing
all evidence required for the answer?

```text
Citation F1 =
2 x precision x recall
----------------------
  precision + recall
```

**Example:** Citation Precision `0.80` and Citation Recall `0.50` produce
Citation F1 approximately `0.62`.

**Range and direction:** `[0, 1]`; higher is better. Citation F1 is defined as
zero when both precision and recall are zero. A malformed answerable response
receives zero for all three citation metrics.

## Response, refusal, and output validity

The rate formulas below are calculated within each question over its scheduled
seeds. Question rates are then averaged within each document and macro-averaged
across documents, as defined in the aggregation section. Raw event and attempt
counts are retained alongside the reported rate; their corpus-wide ratio is not
the document-macro result.

### Response Validity Rate

**Question:** How often does an answerable generation attempt exhibit the
required answer behavior?

A response passes when generation completes normally and the output is
schema-valid, uses `status="answered"`, contains a non-empty substantive
answer, and contains at least one valid citation ID from the evidence blocks
supplied with the current question. Parseable JSON with wrong field types,
extra fields, or unknown citation IDs is not schema-valid.

```text
Response Validity Rate =
valid answered attempts for this question
-----------------------------------------------
all scheduled seeds for this answerable question
```

**Example:** Two of an answerable question's three scheduled seeds produce
valid answered responses; the third produces a parseable refusal. That
question's Response Validity Rate is `2 / 3`, approximately `0.67`. The refusal
is well formed but fails the required answer behavior.

**Range and direction:** `[0, 1]`; higher is better. It is null when the split
contains no answerable questions (inapplicable). Passing this metric does not
prove correctness or that citations entail the answer; it establishes valid
answer behavior and legal evidence IDs.

### Refusal Validity Rate

**Question:** How often does an unanswerable generation attempt exhibit the
exact required refusal behavior?

A refusal passes when generation completes normally and the output is
schema-valid, uses `status="insufficient_evidence"`, contains the frozen refusal
text, and has an empty citation list.

```text
Refusal Validity Rate =
valid refusal attempts for this question
-------------------------------------------------
all scheduled seeds for this unanswerable question
```

**Example:** Two of an unanswerable question's three scheduled seeds produce
the required refusal and one produces a substantive answer. That question's
Refusal Validity Rate is `2 / 3`, approximately `0.67`.

**Range and direction:** `[0, 1]`; higher is better. It is null when the split
contains no unanswerable questions (inapplicable). On this slice a timeout or
runtime failure is not a valid refusal, even if partial text resembles one.

### Malformed Output Rate

**Question:** How often does an attempted generation violate the common response
schema?

Malformed output includes unparsable JSON, missing or additional top-level
fields, fields with the wrong type, unsupported statuses, unknown citation IDs,
and responses that mix refusal text with a substantive answer. Repeated citation
IDs are deduplicated before citation scoring and do not make an otherwise valid
response malformed.

```text
Malformed Output Rate =
malformed outputs for this question
----------------------------------
all scheduled seeds for this question
```

**Example:** One malformed output among a question's three scheduled seeds
gives that question a Malformed Output Rate of `1 / 3`. A runtime failure with
no output contributes to Generation Failure Rate instead; it is not also
labeled malformed.

**Range and direction:** `[0, 1]`; lower is better.

The three validity metrics have different scopes. A malformed answerable output
increments Malformed Output Rate and fails Response Validity Rate. A parseable
refusal on an answerable question does not increment Malformed Output Rate but
still fails Response Validity Rate. Likewise, an otherwise valid `answered`
response with an empty citation list is not malformed, but it fails Response
Validity Rate and receives zero for Citation Precision, Recall, and F1. Store
the numerator, denominator, and rate for each metric.

## Generation reliability

### Generation Failure Rate

**Question:** How often does model or runtime execution fail before an evaluable
response exists?

```text
Generation Failure Rate =
failed generation attempts for this question
-------------------------------------------
all scheduled seeds for this question
```

**Example:** One CUDA error among a question's three scheduled seeds gives
that question a Generation Failure Rate of `1 / 3`.

This category counts execution errors. Timeout and context-boundary termination
have their own finish reasons and rates below; they are not counted again as
execution errors. Each scheduled attempt retains its exact outcome.

**Range and direction:** `[0, 1]`; lower is better. No execution errors means
zero only when scheduled outcomes are fully accounted for. Unexecuted requests
and setup failures are not invented per-question runtime errors.

### Timeout Rate

**Question:** How often does the frozen wall-clock timeout stop generation?

```text
Timeout Rate = timed-out attempts for this question / its scheduled seeds
```

**Example:** With one timeout and two normal completions, the question's rate
is `1 / 3 ≈ 0.33`. A timeout is counted even if tokens were emitted before the
boundary; observed TTFT can survive but successful-completion latency cannot.

**Range and direction:** `[0, 1]`; lower is better. It is not also an execution
error. A timed-out answerable attempt receives failed-quality treatment.
Missing finish-reason records make evaluation incomplete, not timeout-free.

### Context-Limit-Reached Rate

**Question:** How often does output exhaust the model context before EOS?

```text
Context-Limit-Reached Rate = context-boundary terminations for this question
                            / its scheduled seeds
```

**Example:** If one seed times out, one reaches the context boundary, and one
finishes at EOS, the question's Context-Limit-Reached Rate and Timeout Rate are
each `1 / 3`. They describe different recorded termination causes.

**Range and direction:** `[0, 1]`; lower is better. A context-limited answerable
attempt receives failed-quality treatment even if its partial output parses.
The available output capacity is the frozen context minus the complete prompt,
not an independent answer-token budget. A judge or benchmark-infrastructure
failure makes evaluation incomplete rather than becoming any candidate
reliability event. All three reliability rates follow document-macro aggregation.

## Repeatability

Generation development, validation, and locked reporting, and Final RAG locked
reporting, generate each question with seeds `42`, `43`, and `44`. The same seed
list is used by every candidate. The three outputs create three unordered response pairs, and pair
values are averaged within the question. Smoke and preflight use one seed and do
not report repeatability metrics.

All three scheduled pairs remain in the question's denominator. Any pair
containing an execution failure, malformed output, timeout, or context-limited
output receives zero for each applicable agreement metric, including a pair of
two failed outputs. These events are not successful agreement. An unresolved
judge error makes evaluation incomplete rather than assigning candidate zero.

### Repeat Status Agreement

**Question:** Does the model consistently decide to answer or refuse?

Each valid, completed response pair receives `1` when both statuses match and
`0` otherwise. Invalid or interrupted pairs follow the zero rule above.

```text
Question Repeat Status Agreement = matching valid-status pairs / 3 scheduled pairs
```

**Example:** Statuses `answered`, `answered`, and `insufficient_evidence` create
three pairs. One pair agrees, so Repeat Status Agreement is `1 / 3`,
approximately `0.33`.

**Range and direction:** `[0, 1]`; higher is better.

Three identical refusals on an answerable question can agree perfectly while
Response Validity and answerable quality are zero. Agreement is not correctness.

### Repeat Citation Agreement

**Question:** Does the model repeatedly select the same evidence blocks?

Each valid, completed answerable response pair uses Jaccard agreement over
distinct valid supplied evidence IDs:

```text
citations present in both outputs
---------------------------------
distinct citations in either output
```

**Example:** Citation sets `{E1, E2}`, `{E1, E2}`, and `{E1}` have pair values
`1.0`, `0.5`, and `0.5`. Their mean Repeat Citation Agreement is approximately
`0.67`.

**Range and direction:** `[0, 1]`; higher is better. Two empty citation sets on
an answerable question receive zero because required evidence was not selected.
Unanswerable questions are inapplicable to this metric.

### Repeat Semantic Agreement

**Question:** Do repeated visible answers preserve the same meaning even when
their wording differs?

The frozen judge labels each valid, completed response pair with matching
statuses as semantically equivalent or not.
Equivalent pairs receive `1`; non-equivalent pairs, status disagreements, and
failed attempts receive `0`.

```text
Question Repeat Semantic Agreement = semantically equivalent valid pairs
                                     / 3 scheduled pairs
```

**Example:** If two answers state that 500 people participated and the third
states that 800 participated, only the first pair agrees. Repeat Semantic
Agreement is `1 / 3`, approximately `0.33`.

Paraphrases and two valid refusals can agree. Two equally incorrect but
semantically equivalent answers can also agree: repeatability measures stability,
while factual correctness and faithfulness measure quality.

**Range and direction:** `[0, 1]`; higher is better.

## Operational measurement

### Time to First Token p50/p95

**Question:** After a warm request begins, how long does the generator take to
produce its first token?

Time to First Token is measured for every query and measured seed. It starts
immediately before the generation call and ends when the first model-generated
token becomes available. For reasoning configurations, that token may belong to
the reasoning stream; the metric measures model start responsiveness rather than
time to the completed visible answer.

This matters because two models can have similar total latency while one leaves
the caller waiting much longer before generation starts. Prompt prefill and
initial generation work are visible in TTFT.

For each question, the median of available measured TTFT values becomes its warm
observation. A request that emits no token has no TTFT; a later timeout does not
erase an already-observed first token. Report scheduled attempts, attempts with
observed TTFT, and contributing questions alongside the percentiles. p50
summarizes a typical contributing question and p95 its slow tail.

**Example:** Token waits of `0.2`, `0.4`, and no token give that question an
observed median TTFT of `0.3 seconds`. Across question medians
`0.1, 0.2, 0.3, 0.4, 0.5`, p50 is `0.3` and p95 is
`0.4 + 0.8 × (0.5 - 0.4) = 0.48 seconds`. Thus p50 describes typical
responsiveness and p95 the slow tail, not an uncertainty bound.

**Range and direction:** Non-negative seconds; lower is better. TTFT percentiles
are null when no measured request emits a generated token.

### End-to-End Latency p50/p95

**Question:** How long does a complete warm generation request take?

The interval includes prompt preparation, tokenization, model prefill, reasoning
and visible-answer decoding, output parsing, and citation validation. The median
latency of successfully completed requests for each question becomes its warm
observation; p50 and p95 are then calculated across contributing questions.
Every attempt also retains its elapsed time through termination and outcome in
the timing artifact, including failed and interrupted attempts. These elapsed
times are not treated as successful completion latency. Report scheduled and
successful attempt counts and contributing question counts, and read latency
alongside the failure and termination rates.

**Example:** Successful attempts of `4` and `6 seconds` plus one timeout give
that question a completion observation of `(4 + 6) / 2 = 5 seconds`.
Question observations `2, 3, 4, 5, 10` have p50 `4` and interpolated p95
`5 + 0.8 × (10 - 5) = 9 seconds`. This is not a guarantee that exactly 95%
of a small sample completes within that value; the timeout stays visible
separately.

For Final RAG, retrieval, reranking, context packing, generation, and complete
system latency are also reported separately.

**Range and direction:** Non-negative seconds; lower is better. End-to-End
Latency percentiles are null when no measured request completes successfully.

### Decode Throughput

**Question:** Once decoding begins, how quickly does the model produce output
tokens?

```text
Decode Throughput =
N - 1 generated tokens after the first token
------------------------------------------
time of last counted token - time of first token
```

`N` is the number of model-generated token events in the measured output,
including reasoning, visible output, and emitted control/termination tokens.
Prompt tokens and padding are not counted. The interval starts when the first
generated token is available and ends when the last counted token is available.
The first token starts the clock and is therefore not counted as work performed
inside this interval. Parsing and cleanup after the last token are excluded.

**Example:** Four tokens arrive at `0.0`, `0.1`, `0.2`, and `0.3` seconds.
Decode Throughput is `(4 - 1) / 0.3 = 10 tokens/second`.

**Range and direction:** Non-negative tokens/second; higher is better. It is null
when fewer than two tokens are generated or no positive inter-token interval
can be measured. Missing required timing instrumentation makes measurement
incomplete. Throughput has no upper bound of one.

Retain the request-level numerator and interval, including a partial stream's
outcome if token timing was observed before termination. Any run summary must
identify its contributing requests and whether it averages request rates or
divides pooled tokens by pooled intervals; those are not interchangeable.
The approved contract does not yet name that summary estimator. It must be
frozen and implemented before a run-level throughput scalar is comparable.

### Cold Model-Load Time

**Question:** How long does a fresh worker need to make the generator ready?

The timer starts before model construction and ends after weights are loaded,
device placement is complete, and CUDA is synchronized. It ends before the
warmup request. The model is absent from the new process and device at the start;
the operating-system disk cache is left in its normal state.

**Example:** If loading starts at `0.0 seconds` and the synchronized CUDA model is
ready at `5.4 seconds`, Cold Model-Load Time is `5.4 seconds`.

**Range and direction:** Non-negative seconds; lower is better. One cold-load
observation receives no confidence interval.

An interrupted load has null/unavailable completed load time, with its
elapsed-to-failure retained separately. A completed load remains known if
warmup or a later measured request fails; a setup error does not create
per-question failed responses for requests that never ran.

### Peak Process-Tree RAM

**Question:** What peak host memory does the generator worker require?

At each sample, sum resident memory for the worker and its child processes;
report the largest simultaneous sum. Do not add independent process peaks.
The monitoring window includes loading, warmup, and measured generation, not
offline judging or bootstrap calculations.

**Example:** Simultaneous worker/child totals of `2,000 + 600 = 2,600 MiB`
and `1,900 + 1,200 = 3,100 MiB` give a peak of `3,100 MiB`, not
`2,000 + 1,200 = 3,200 MiB`.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
No samples means null/unavailable, not zero. Samples before a failure remain
valid partial-window observations with start/end scope; they do not establish
the memory needed for the unexecuted workload. Missing required instrumentation
makes the report incomplete. An observed peak receives no confidence interval.

### Peak Device VRAM

**Question:** How much total assigned-GPU memory is occupied at the observed peak?

Take the maximum raw NVML `memory.used` sample from before loading through
the last measured request. Include desktop/driver allocations; the idle baseline
is explanatory and is not subtracted. This device-total value does not prove
correct model placement, which is checked separately.

**Example:** Device samples `700, 2,850, 2,500 MiB` report `2,850 MiB`,
not `2,150 MiB` above the baseline. A peak retained before an OOM is partial
evidence, not proof that the complete workload fits.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
CPU execution is null/inapplicable; missing required CUDA telemetry is
null/unavailable and makes measurement incomplete. Preserve observed values
and their scope after failure; do not clamp them to a hardware threshold.
The existing generation resource limit is an execution gate, not this metric's
upper range, and does not impose a cap on extraction. One peak has no CI.

## Workload descriptors

These are per-attempt workload observations, not higher-is-better quality
metrics. Keep question/seed identity and outcome, and identify the observed
attempts and weighting of each descriptive summary. Token counts use the
candidate's native tokenizer, not retrieval's fixed evidence-scoring tokenizer;
equal token counts across different models need not represent equal text or work.
Missing observations remain null, while genuine empty components can count zero.
These descriptors receive no sample-based quality confidence interval.

### Prompt Tokens

**Question:** How much input does the complete model request consume?

Count the fully rendered native chat-template input, including instructions,
question, evidence, and special input tokens. Do not count padding as real input.

**Example:** A rendered prompt containing `1,200` native input tokens in an
`8,192`-token context leaves `8,192 - 1,200 = 6,992` tokens of output capacity.
It is this complete count, not the evidence count alone, that consumes context.

**Range and direction:** Non-negative integer tokens; descriptive. A missing
tokenization record is unavailable, not an empty prompt. An overlength input is
an input-contract problem, not permission to truncate or alter context settings.

### Context Tokens

**Question:** How much supplied evidence text accompanies this question?

Count the supplied evidence component with the same native tokenizer, retaining
its text and counting boundary. It is part of, not additional to, Prompt Tokens.
Separate component tokenizations need not sum to the complete chat tokenization.

**Example:** The `1,200`-token prompt above may contain evidence measured as
`900` native tokens. Reporting both distinguishes evidence workload from the
complete input cost without asserting that every remaining token is an instruction.

**Range and direction:** Non-negative integer tokens; descriptive. A known
empty supplied context counts zero; an unrecorded context is unavailable.
Retrieved-token descriptors using `cl100k_base` are separate measurements.

### Reasoning Tokens

**Question:** How much output belongs to the exposed reasoning stream?

Count generated tokens in the runtime's reliably separated reasoning component.
Hidden or unexposed reasoning is not inferred from elapsed time or the answer.

**Example:** An exposed reasoning span of `30` tokens records `30`. A runtime
that returns only a visible answer without a reliable reasoning boundary records
null, not an assumption of zero reasoning.

**Range and direction:** Non-negative integer tokens when observed; descriptive.
Verified absence can count zero. Boundary uncertainty is null/unavailable.

### Visible-Answer Tokens

**Question:** How much visible answer text did the model deliver?

Count the `answer` string with the native tokenizer, excluding separated
reasoning and the JSON wrapper/citation fields. This describes answer length,
not the number of required facts recovered.

**Example:** A valid response whose answer string encodes to `20` native tokens
records `20`, whether those tokens express correct facts or unsupported claims.

**Range and direction:** Non-negative integer tokens; descriptive. An absent
response or unparseable answer field has no defined answer-string count.
Retain any available raw generated tokens separately rather than inventing an
empty answer. A known empty string counts zero even if it fails answer behavior.

### Total Output Tokens

**Question:** How many native generated-token events did the request produce?

Count all generated events, including reasoning, visible output, JSON/control
syntax, and emitted termination tokens; exclude input/padding. This is the `N`
used for Decode Throughput, not just the retokenized visible answer string.

**Example:** A request producing `60` output tokens can have `30` reasoning
tokens and a visible answer counted as `20` tokens, with the rest belonging to
other output syntax/control events. Do not require these separately measured
components to sum to `60` when boundaries or retokenizations differ.

**Range and direction:** Non-negative integer tokens; descriptive. A measured
attempt known to emit no tokens counts zero. Partial tokens before timeout remain
observed workload with that outcome; a missing event record is unavailable.

### Citation Count

**Question:** How many distinct evidence IDs does the response select?

Count distinct citation IDs after the scoring deduplication; retain the original
ordered list and invalid-ID diagnostics. A large count is not automatically
better citation quality.

**Example:** `["E1", "E1", "E2"]` has a distinct Citation Count of `2`,
not `3`. If `E2` is legal but irrelevant, count is still `2` while Citation
Precision is only `1 / 2`.

**Range and direction:** Non-negative integer IDs; descriptive. A valid empty
list counts zero. Missing/unparseable citation fields are unavailable rather
than fabricated empty lists; unknown IDs remain malformed-output evidence.

### Generated-Claim Count

**Question:** How many distinct factual assertions must be verified?

Count atomic, semantically deduplicated visible-answer claims from the calibrated
judge's extraction labels, not sentences or required gold facts.

**Example:** "The trial enrolled 500 adults and lasted six months" yields
three claims: enrollment count, adult population, and duration. Repeating the
duration does not create a fourth distinct claim.

**Range and direction:** Non-negative integer claims; descriptive. Completed
claim extraction finding none gives zero. Missing judge labels give
null/unavailable and incomplete semantic evaluation, not a claim-free answer.

### Finish-Reason Distribution

**Question:** Why did scheduled generation attempts terminate?

Retain counts for the exact recorded finish reasons and their observed/scheduled
support. A descriptive observed share is `reason count / attempts with a recorded
termination`. It is not a replacement for the document-macro reliability rates.

**Example:** Recorded reasons `EOS, EOS, timeout` give counts `2, 1` and
observed shares `2 / 3, 1 / 3`. If one required attempt never ran, keep it
unexecuted and the run incomplete, not an invented EOS or timeout.

**Range and direction:** Counts are non-negative integers, shares are `[0, 1]`;
descriptive. No recorded terminations means no defined shares, not all-zero
successful behavior. These frequencies do not receive confidence intervals.

## Worked candidate interpretation

Suppose one generator configuration produces:

```text
Faithfulness                         = 0.92
Factual Correctness Precision        = 0.88
Factual Correctness Recall           = 0.70
Factual Correctness F1               = 0.78
Answer Relevancy                     = 0.95
Citation F1                          = 0.84
Response Validity Rate               = 0.96
Refusal Validity Rate                = 0.85
Malformed Output Rate                = 0.02
Repeat Semantic Agreement            = 0.90
TTFT p95                              = 0.90 seconds
End-to-End Latency p95                = 8.40 seconds
```

These are document-macro quality/rate estimates, not pooled percentages of all
claims or attempts. Calculate F1 per response before averaging, not from the
displayed aggregate precision and recall. The answers are usually well supported
and directly relevant. Required-fact coverage is weaker than generated-claim
precision; the different denominators do not establish corpus-wide error counts.
Refusal Validity shows
that unanswerable questions remain a distinct weakness. The non-zero malformed
rate identifies a schema issue separate from answer/refusal behavior, while
repeatability and latency describe stability and runtime cost.


## Eligibility, aggregation, and confidence intervals

Quality, validity, and reliability indicators are calculated for every scheduled
seed and first averaged within each question. Question values are then averaged
within each source document and macro-averaged across documents. Repeatability
first averages all scheduled pairs within a question, then uses the same
document-macro calculation. Each metric uses its declared eligibility slice.
Raw numerators, scheduled denominators, contributing counts, and failure rows
remain available even when a conditional value is null.

For example, document A has four question values `1, 1, 0, 0` and document B
has one question value `1`. Their document means are `0.50` and `1.00`; the
reported document-macro value is `0.75`. The pooled question average would be
`3 / 5 = 0.60`, giving the longer document four times the influence. Equal
document weight is used consistently for generation quality and rates.

Development, validation, and locked runs use
10,000 bootstrap resamples of complete documents with bootstrap seed `42`; that
seed is independent of generation seeds. The 2.5th and 97.5th percentiles are
the 95% confidence bounds. Answerable-only, unanswerable-only, and evidence-type
results always report eligible question and document counts. Substantive-answer-only
diagnostics, if reported, do not replace primary answerable results that
include refusals and failed attempts.

Warm latency and token-workload summaries use the same fixed question set and
report their observation counts. Cold load and observed RAM/VRAM peaks are
single-run measurements and receive no fabricated interval. An explicitly
inapplicable metric or an operational value with no valid observation is null,
with its reason and observation counts retained. For example, a recorded request
that emits no token has no TTFT observation; its failure or completion outcome
still contributes to the applicable reliability metrics. This differs from a
missing required attempt record, artifact, metric field, or timing instrument:
those omissions make the evaluation incomplete, not a valid null observation.

### Interval support and calculation

Quality, validity, reliability, and repeatability aggregates receive 95% bounds
only with sufficient independent source support. Warm TTFT and completion-latency
p50/p95 receive conditional intervals under the same frozen support requirement.
One cold load, a resource peak, token/citation/claim counts, and finish-reason
inventories do not acquire intervals merely because they are numeric. A decode
throughput summary needs its estimator defined before its uncertainty can be
specified.

For each of the 10,000 draws, resample complete reference-eligible source
documents with replacement, retaining questions, all scheduled seeds, pair
labels, failures, and unavailable evaluations. Recalculate the same
seed-then-question-then-document aggregate. A source drawn twice contributes
twice. Do not prefilter to successful answers: observed candidate zeros remain,
and unavailable evaluations retain their coverage/incomplete status. For latency,
recalculate percentiles from the sampled documents' per-question observations,
not from document-average times. Paired comparisons use the same source draw
for both configurations.

Report eligible and contributing independent-source counts, not `3 × questions`
as independent support. If no eligible questions exist, the result is
inapplicable; if required labels or attempts are missing, any surviving-output
estimate is explicitly incomplete/diagnostic. Do not turn it into a normal
complete comparison by omitting missing rows.

Below the support minimum frozen after data review and before held-out use,
keep a defined descriptive point estimate with null bounds and a reason. No
numeric support minimum is set until that review. Record defined/undefined bootstrap draws for each
conditional metric. Identical resamples can yield bounds such as `[1, 1]`;
retain them with a degenerate-interval flag, not a claim of future certainty.

For example, `Faithfulness = 0.90, 95% CI [0.86, 0.94]` describes uncertainty
in the document-macro estimate, not a range of individual answers. Bootstrap
resampling performs arithmetic on saved labels; it launches neither extra
generation nor extra judge calls.
