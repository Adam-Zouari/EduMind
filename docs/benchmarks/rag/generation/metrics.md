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
- [Per-response diagnostic artifacts](#per-response-diagnostic-artifacts)
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

Apply the [shared availability and support rules](../../metrics.md#shared-availability-rules)
to every measured seed. Generation quality uses all three scheduled seed
outputs, not only attempt 1; this observation policy does not turn failed
requests into numeric quality scores.

| Case | Treatment |
|---|---|
| Normally completed, schema-valid answered response | Score each applicable metric; incorrect content is not a structural failure. |
| Well-formed refusal on an answerable question | Apply the evaluated claim-free, missed-fact, citation, and behavioral rules rather than treating refusal as an inference crash. |
| Schema-valid response cites an unknown string ID | Apply citation-ID scoring and Response Validity separately; the ID does not make the JSON malformed or overwrite semantic scores. |
| Successful claim extraction finds no factual claims | Apply the explicit claim-free conventions below. This is an evaluated empty denominator, not missing judging. |

Only normally completed, structurally evaluable responses supply answer quality.
An interrupted fragment is not a completed response merely because it parses.
The [shared dependency and partial-observation rules](../../metrics.md#dependent-metrics-and-partial-observations)
govern derived scores and partial aggregates. Retain per-question/seed outcome,
reference eligibility, contributing support, schema/ID checks, and evaluator
status separately. A defined null convention, such as `no_generated_claims`,
does not itself make completed evaluation incomplete; missing required labels
or execution does.

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
| Malformed Output Rate | Diagnostic | How often does a normally completed generation produce structurally invalid output? | Lower |
| Runtime Error Rate | Diagnostic | How often does measured inference encounter a runtime execution error? | Lower |
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
| Peak Process-Tree RAM | Operational | What maximum simultaneous host-process memory was observed in the monitored lifecycle? | Lower at equal quality and scope |
| Peak Device VRAM | Operational | What maximum total assigned-device memory was observed in the monitored lifecycle? | Lower at equal quality and scope |
| Prompt Tokens | Workload descriptor | How much input does the complete rendered request consume? | Descriptive |
| Context Tokens | Workload descriptor | How much supplied evidence text accompanies the question? | Descriptive |
| Reasoning Tokens | Workload descriptor | How much exposed reasoning did the model generate? | Descriptive |
| Visible-Answer Tokens | Workload descriptor | How long is the visible answer string? | Descriptive |
| Total Output Tokens | Workload descriptor | How many generated-token events did the request produce? | Descriptive |

Citation counts, generated-claim counts, and finish-reason inventories remain
[diagnostic artifacts](#per-response-diagnostic-artifacts), not additional
MLflow summary metrics.

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

All five semantic quality metrics use reference-answerable questions;
unanswerable questions are `null/inapplicable`. A known claim-free response,
including an exact refusal when an answer was required, leaves Faithfulness
and Factual Correctness Precision undefined, but has zero required-fact recall
and Factual Correctness F1. Answer Relevancy follows its rubric. A failed request
or unresolved evaluation instead follows the shared availability rules, not
these known-empty conventions. Keep reference-eligible and actually contributing
counts distinct.

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

**Range and direction:** `[0, 1]`; higher is better when defined.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Completed claim extraction finds no factual assertions | A visible answer contains only "Thank you!" | `null` (`inapplicable`, `no_generated_claims`) | There is no generated-claim support ratio. This is not evidence of hallucination or a failed extraction. |
| Required claim/support labels cannot be obtained | The judge fails after its frozen retries | `null` (`unavailable`); evaluation incomplete | Unknown labels cannot establish that there were no claims or that they were unsupported. |

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

**Range and direction:** `[0, 1]`; higher is better when defined.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Completed extraction finds no generated factual claims | The response is a greeting rather than a factual answer | `null` (`inapplicable`, `no_generated_claims`) | There are no correctness judgments to divide by. Required-fact omission remains visible in recall and F1. |
| Required correctness labels cannot be obtained | Claims were extracted, but verification fails | `null` (`unavailable`); evaluation incomplete | Missing verification is not a set of incorrect claims. |

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

The positive gold denominator is established by
[reference validation](../../data-validation.md#generation-references-and-model-input-compatibility).
A completed answer recovering none, including a known claim-free response,
receives ordinary zero recall.

**Range and direction:** `[0, 1]`; higher is better.

**Edge cases and availability:**

There is no additional valid empty-reference convention. Missing required
gold-matching labels follow the shared evaluator-availability rules, not zero
matching.

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

**Range and direction:** `[0, 1]`; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Both defined components are zero | Claims exist, but no correctness or required-fact credit is earned | `0` (`scored`) | Resolve the harmonic mean's zero denominator explicitly. If either defined component is zero and the other is positive, the ordinary formula also gives zero. |
| A completed response is known to contain no factual claims | Claim extraction succeeds with an empty list; required facts exist | `0` (`scored`, known-empty convention) | The combined correctness/completeness task recovered none of the required facts. This explicit convention does not substitute zero for null precision. |
| A required evaluation is unavailable | Gold matching or claim verification is missing after judge failure | `null` (`unavailable`); evaluation incomplete | The known-empty convention cannot be applied to unknown claims or judgments. |

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

**Edge cases and availability:**

There is no division-by-zero convention: every successfully assigned rubric
level maps to a defined score. Missing or unusable rubric labels are unavailable
under the shared rules. Judge calibration, label validity, and retry requirements
belong to the [semantic judge contract](methodology.md#semantic-judge-contract),
not additional numeric cases of this metric.

## Citation quality

Citation Precision, Recall, and F1 are calculated only for answerable questions,
which always have one or more verified gold evidence units. Unanswerable
questions are evaluated by Refusal Validity Rate and the applicable reliability
metrics rather than receiving artificially perfect citation scores.

A citation is correct when it is a valid supplied evidence-block ID and that
block completely covers at least one required gold evidence unit. Repeated IDs
count once. Unknown IDs remain in the distinct produced-ID denominator, earn
no correctness or evidence-coverage credit, and fail Response Validity. They
are reference-validity failures, not structural schema failures, and do not
overwrite independently evaluable semantic scores. Retain each ID check.

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

**Range and direction:** `[0, 1]`; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| A completed answerable response has an empty citation list | `citations: []` | `0` (`scored`) | Required evidence exists, but no evidence IDs were selected; define the empty produced-ID denominator as zero precision. |
| No citation-quality task exists | An unanswerable question receives the required refusal | `null` (`inapplicable`) | Correct absence of citations is assessed by Refusal Validity, not a perfect citation score. |

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
context. Restricting it to retrieved units would hide retrieval misses.

**Edge cases and availability:**

Only answerable questions have this task; other questions are
`null/inapplicable`. Its reviewed reference denominator is positive, so there
is no additional valid empty-denominator case. A valid empty list covers zero
units normally; input annotation requirements belong to reference validation.

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

**Range and direction:** `[0, 1]`; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Both defined components are zero | The answerable response has no correct citations | `0` (`scored`) | Resolve the harmonic mean's `0/0` without rewarding missing evidence. |
| Citation quality is inapplicable | An unanswerable question | `null` (`inapplicable`) | F1 inherits its components' reference eligibility. |

Unavailable required citation fields follow the shared dependency rules. An
unknown but well-typed ID is instead a scored incorrect citation: `[E1, UNKNOWN]`
has precision `1/2` and recall `1` if E1 covers the only required unit, giving
F1 approximately `0.67`, not a blanket zero for the entire response.

## Response, refusal, and output validity

Calculate each rate within its question, then average contributing question
rates within each source document and macro-average documents. Response and
Refusal Validity use scheduled seeds; Malformed Output Rate uses normally
completed generations. Retain each numerator/denominator and all outcomes;
their pooled corpus ratio is not the document-macro result. Undefined question
rates retain their reasons and support under the shared rules.

### Response Validity Rate

**Question:** How often does an answerable generation attempt exhibit the
required answer behavior?

A response passes when generation completes normally and the output is
schema-valid, uses `status="answered"`, contains a non-empty substantive
answer, and contains at least one citation, with every cited ID resolving to an
evidence block supplied with the current question. Substantive factual content
means at least one atomic claim in the existing claim-extraction labels; code
derives this check without another judge call. A known failed check gives zero.
If mechanical checks pass but the required claim labels are unavailable, the
validity observation is unavailable rather than assumed substantive. Structural
checks, citation-ID validity, and substantive-answer checks remain separate.

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

**Range and direction:** `[0, 1]`; higher is better. Passing does not establish
correctness, relevance, or that a citation entails the generated claims.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| The reporting slice has no answerable questions | An unanswerable-only slice | `null` (`inapplicable`) | There is no required answered behavior to assess. |

Missing required outcomes or judgments follow the shared availability rules.
Known runtime failures count as unsuccessful answer behavior; unexecuted
requests are not fabricated validity events.

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

**Range and direction:** `[0, 1]`; higher is better. A timeout or runtime failure
is not a valid refusal, even if partial text resembles the refusal sentence.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| The reporting slice has no unanswerable questions | An answerable-only slice | `null` (`inapplicable`) | No refusal behavior was required. |

Missing required execution follows the shared availability rules; it must not
shrink the scheduled denominator to surviving attempts.

### Malformed Output Rate

**Question:** How often does a normally completed generation violate the
structural response schema?

Malformed output includes unparsable JSON, missing or additional top-level
fields, fields with the wrong type, and unsupported statuses. Normally completed
empty text is a malformed output, not a missing observation. Unknown citation
IDs, repeated IDs, factual errors, and semantic mixtures of answering and
refusing do not change structural validity.

```text
Malformed Output Rate =
structurally malformed normally completed outputs for this question
-----------------------------------------------------------------
all normally completed generations for this question
```

**Example:** One valid output, one malformed output, and one runtime error give
Malformed Output Rate `1 / 2 = 0.50` and Runtime Error Rate `1 / 3`. Two valid
outputs and one crash give malformed rate `0 / 2 = 0`, not proof of complete
execution success. Timed-out and context-limited fragments are not normally
completed outputs, even if they parse.

**Range and direction:** `[0, 1]`; lower is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No generation completes normally | All three executed attempts crash or are interrupted | `null` (`unavailable`, no completed outputs) | No complete output exists whose structure can be counted; reliability and scheduled support remain visible. |

The three validity metrics have different scopes. A malformed answerable output
increments Malformed Output Rate and fails Response Validity Rate. A parseable
refusal on an answerable question does not increment Malformed Output Rate but
still fails Response Validity Rate. Likewise, an otherwise valid `answered`
response with an empty citation list is not malformed, but it fails Response
Validity Rate and receives zero for Citation Precision, Recall, and F1. Store
the numerator, denominator, and rate for each metric.

## Generation reliability

### Runtime Error Rate

**Question:** How often does a measured inference attempt encounter a model or
runtime execution error?

```text
Runtime Error Rate =
inference attempts with execution errors for this question
--------------------------------------------------------
all scheduled seeds for this question
```

**Example:** One CUDA error among a question's three scheduled seeds gives
that question a Runtime Error Rate of `1 / 3`.

This category includes inference OOM and backend exceptions, not only Python's
specific `RuntimeError` class. Loading/warmup, judging, monitoring, and input
compatibility failures are not measured-inference errors. Malformed completed
outputs, timeouts, and output-context termination have separate outcomes and
are not counted again here. A fully executed schedule with every inference
attempt failing has rate one.

**Range and direction:** `[0, 1]`; lower is better. Zero errors is an ordinary
result only when scheduled outcomes are fully accounted for.

**Edge cases and availability:**

There is no additional arithmetic exception for a nonempty, accounted schedule.
Other availability follows the shared rules rather than inventing measured
events for requests that did not execute.

### Timeout Rate

**Question:** How often does the frozen wall-clock timeout stop generation?

```text
Timeout Rate = timed-out attempts for this question / its scheduled seeds
```

**Example:** With one timeout and two normal completions, the question's rate
is `1 / 3 ≈ 0.33`. A timeout is counted even if tokens were emitted before the
boundary; observed TTFT can survive but successful-completion latency cannot.

The generation timeout window starts immediately before the generation call
and ends at normal decoding completion or the stopping boundary. Record the
first actual terminal cause. EOS before the deadline is normal completion even
if parsing/cleanup ends later; a timeout after tokens were emitted still counts.

**Range and direction:** `[0, 1]`; lower is better.

**Edge cases and availability:**

No additional numeric exception is needed for accounted attempt outcomes.
Missing termination records follow the shared availability rules, not an
assumption of zero timeouts. Observed TTFT can survive an interrupted response;
quality and successful-completion latency use their own availability rules.

### Context-Limit-Reached Rate

**Question:** How often does output exhaust the model context before EOS?

```text
Context-Limit-Reached Rate = context-boundary terminations for this question
                            / its scheduled seeds
```

**Example:** If one seed times out, one reaches the context boundary, and one
finishes at EOS, the question's Context-Limit-Reached Rate and Timeout Rate are
each `1 / 3`. They describe different recorded termination causes.

EOS at the final permitted token is normal completion. Without EOS, exhaustion
of output capacity remains context termination even if the fragment parses.
The available output capacity is the frozen context minus the complete prompt,
not an independent answer-token budget. Input context overflow is a separate
[input-compatibility outcome](../../data-validation.md#generation-references-and-model-input-compatibility)
before generation, not an output-exhaustion event or another summary metric.

**Range and direction:** `[0, 1]`; lower is better.

**Edge cases and availability:**

No additional empty-denominator convention applies to an accounted, nonempty
generation schedule. Other availability follows the shared rules, not an
assumption of zero context-exhaustion events.

## Repeatability

Generation development, validation, and locked reporting, and Final RAG locked
reporting, generate each question with seeds `42`, `43`, and `44`. The same seed
list is used by every candidate. The three outputs create three unordered response pairs, and pair
values are averaged within the question. Smoke and preflight use one seed and do
not report repeatability metrics.

Adopt the [shared scheduled-pair denominator, failure handling, and availability rules](../../metrics.md#pairwise-repeatability).
The three comparison functions below determine pair values; none measures
answer correctness. A schema-valid response with wrong behavior or incorrect
citation IDs can be consistently wrong without becoming a failed execution.

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
Response Validity, required-fact recall/F1, and citation quality are zero;
claim-based precision and faithfulness have their claim-free null convention.
Agreement is not correctness.

**Edge cases and availability:**

Use the shared pair rules: executed failures contribute zero, while a one-seed
profile is inapplicable and missing required outcomes are incomplete. No
additional status-comparison exception is needed.

### Repeat Citation Agreement

**Question:** Does the model repeatedly select the same evidence blocks?

Each valid, completed answerable response pair uses Jaccard agreement over
distinct produced citation IDs. Preserve unknown IDs in these sets: filtering
them away could manufacture agreement between different erroneous choices.
Their correctness is measured by citation quality, not agreement:

```text
citations present in both outputs
---------------------------------
distinct citations in either output
```

**Example:** Citation sets `{E1, E2}`, `{E1, E2}`, and `{E1}` have pair values
`1.0`, `0.5`, and `0.5`. Their mean Repeat Citation Agreement is approximately
`0.67`.

**Range and direction:** `[0, 1]`; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Both valid citation arrays are empty | `[]` compared with `[]` on an answerable question | Pair value `1` (`scored`) | They make the same selection. Citation quality and Response Validity separately expose omitted evidence. |
| No citation-repeatability task exists | An unanswerable question, or a deliberately one-seed profile | `null` (`inapplicable`) | Neither correct refusal nor absent pairs earns artificial agreement. |

Failed or structurally invalid response pairs follow the shared zero rule; a
missing citation field is not an empty array. Matching erroneous ID sets can
agree, just as matching incorrect answers can agree semantically.

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

**Edge cases and availability:**

Required pair labels that cannot be obtained follow shared evaluator-availability
rules, not a zero disagreement. The shared scheduled-pair rule still applies to
executed candidate failures. A one-seed profile has no pair task. Two valid
refusals can agree on meaning even when refusing was the wrong behavior.

## Operational measurement

Use the [shared latency-percentile calculation](../../metrics.md#latency-percentiles)
on the per-question observations defined below; TTFT and successful-completion
latency deliberately have different observation eligibility.

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

**Range and direction:** Non-negative seconds; lower is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No first-token event is observed for a request | Inference fails before emitting a token | Request TTFT `null` (`unavailable`) | No elapsed-to-first-token observation exists; it is not instantaneous response. |
| A first token is observed before interruption | First token at `0.4 s`, followed by timeout | Preserve `0.4 s` (`scored`) | The first-token measurement completed independently of response completion. |
| No question contributes a TTFT observation | All measured requests emit no tokens | p50/p95 `null` (`unavailable`) | There is no observation set to summarize. |

Single-observation percentiles and missing instrumentation follow the shared
percentile and availability rules; do not substitute full-response time for TTFT.

### End-to-End Latency p50/p95

**Question:** How long does a complete warm generation request take?

The interval includes prompt preparation, tokenization, model prefill, reasoning
and visible-answer decoding, output parsing, and citation validation. The median
latency of successfully completed requests for each question becomes its warm
observation; p50 and p95 are then calculated across contributing questions.
Successful execution here means normal termination and a usable structural
response schema, not a correct or behaviorally valid answer. Citation-ID checks
are timed, but a completed check finding an unknown ID is not a runtime crash.
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

**Range and direction:** Non-negative seconds; lower is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No request completes successfully for a question | All three attempts terminate early | Question completion-latency observation `null` (`unavailable`) | Interrupted elapsed times remain in artifacts, not the successful-completion distribution. |
| No question contributes successful completion latency | Every request is interrupted or structurally unusable | p50/p95 `null` (`unavailable`) | No successful-completion observation set exists. |

Use the shared percentile rule for a single observation, retaining support rather
than inventing another request. An ordinary wrong answer is still a completed
request; correctness is not a latency eligibility filter.

### Decode Throughput

**Question:** Once decoding begins, how quickly does the model produce output
tokens?

```text
Aggregate Decode Throughput =
sum of counted decoding tokens from contributing streams
-------------------------------------------------------
sum of their corresponding first-to-last-token intervals

For each stream:
counted decoding tokens = N - 1
decoding interval = last-token time - first-token time
```

`N` is the number of model-generated token events in the measured output,
including reasoning, visible output, and emitted control/termination tokens.
Prompt tokens and padding are not counted. The interval starts when the first
generated token is available and ends when the last counted token is available.
The first token starts the clock and is therefore not counted as work performed
inside this interval. Parsing and cleanup after the last token are excluded.

Contributing streams are normally completed measured generations with at least
two recorded token events and a positive measured decoding interval. Completion
of decoding does not require correct facts, valid citations, or parseable JSON;
do not filter slow or incorrect answers from the physical decoding measurement.
Exclude loading, warmup, interrupted streams, and offline judging. Retain
interrupted-stream counts, times, rates when measurable, and outcomes separately.
Apply the [shared ratio-of-totals estimator](../../metrics.md#ratio-of-totals-throughput)
to all contributing questions/seeds, not a document-macro average of rates.

**Example:** Four tokens arrive at `0.0`, `0.1`, `0.2`, and `0.3` seconds.
The stream contributes `3` decoding tokens and `0.3 seconds`.
If another contributes `6` tokens over `0.6 seconds`, aggregate throughput is
`(3 + 6) / (0.3 + 0.6) = 10 tokens/second`. The scalar describes observed decoding
work per second on this workload, not an input-independent model property or
end-to-end server throughput.

**Range and direction:** Non-negative tokens/second; higher is better. There is
no upper bound of one.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Fewer than two token events | No token, or only immediate EOS | Stream rate `null` (`unavailable`) | There is no measured interval containing a subsequent token. |
| No positive interval can be resolved | Several tokens receive the same timestamp | Stream rate `null` (`unavailable`) | Do not report an infinite speed or invent elapsed decoding time. |
| No completed stream contributes valid work/time | Every stream is interrupted or lacks a usable interval | Aggregate `null` (`unavailable`) | No positive contributing decoding interval exists. |

Missing required instrumentation follows the shared availability rules. The
main scalar's cohort excludes interrupted streams even when their diagnostic
partial-stream rates are defined; their known work remains in artifacts.

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

**Edge cases and availability:**

Apply the [shared completed-phase and partial-observation rules](../../metrics.md#dependent-metrics-and-partial-observations):
interrupted loading has unavailable completed load time, while completed loading
survives a later warmup or inference failure. Retain elapsed-to-failure separately.

### Peak Process-Tree RAM

**Question:** What peak host memory does the generator worker require?

Use the [shared simultaneous process-tree RSS calculation](../../metrics.md#peak-process-tree-ram)
for the generator worker and its children.
The monitoring window includes loading, warmup, and measured generation, not
offline judging or bootstrap calculations.

**Example:** Simultaneous worker/child totals of `2,000 + 600 = 2,600 MiB`
and `1,900 + 1,200 = 3,100 MiB` give a peak of `3,100 MiB`, not
`2,000 + 1,200 = 3,200 MiB`.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

**Edge cases and availability:**

Availability, partial-window reporting, and single-peak uncertainty follow the
[shared resource rules](../../metrics.md#resource-measurements). The candidate
has one continuous monitored lifecycle, not three independent peak experiments.

### Peak Device VRAM

**Question:** How much total assigned-GPU memory is occupied at the observed peak?

Use the [shared NVML device-total peak](../../metrics.md#peak-device-vram) from
before loading through the last measured request. This device-total value does
not prove correct model placement, which is checked separately.

**Example:** Device samples `700, 2,850, 2,500 MiB` report `2,850 MiB`,
not `2,150 MiB` above the baseline. A peak retained before an OOM is partial
evidence, not proof that the complete workload fits.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

**Edge cases and availability:**

Apply the [shared CPU/CUDA resource availability, partial-window, and scope rules](../../metrics.md#resource-measurements).
Missing required telemetry is not an observed allocation of zero.

## Workload descriptors

These are per-attempt workload observations, not higher-is-better quality
metrics. Keep question/seed identity and outcome. If an MLflow descriptive
summary is emitted, its value is the document-macro mean of defined per-attempt
counts: first within question, then document, then documents. Retain underlying
counts and support; means may be fractional even though per-attempt counts are
integers. Do not reweight quality by answer length. Token counts use the
candidate's native tokenizer, not retrieval's fixed evidence-scoring tokenizer;
equal token counts across different models need not represent equal text or work.
Missing observations remain null, while genuine empty components can count zero.
These descriptors receive no sample-based quality confidence interval.

### Prompt Tokens

**Question:** How much input does the complete model request consume?

Count the fully rendered native chat-template input, including instructions,
question, evidence, and special input tokens. Do not count padding as real input.
Native input compatibility is checked before generation under the input contract,
not by truncating the prompt to obtain a preferred count.

**Example:** A rendered prompt containing `1,200` native input tokens in an
`8,192`-token context leaves `8,192 - 1,200 = 6,992` tokens of output capacity.
It is this complete count, not the evidence count alone, that consumes context.

**Range and direction:** Non-negative integer tokens; descriptive.

**Edge cases and availability:**

There is no additional arithmetic exception for an observed integer count.
Missing tokenization measurements follow the shared availability rules.

### Context Tokens

**Question:** How much supplied evidence text accompanies this question?

Count the supplied evidence component with the same native tokenizer, retaining
its text and counting boundary. It is part of, not additional to, Prompt Tokens.
Separate component tokenizations need not sum to the complete chat tokenization.

**Example:** The `1,200`-token prompt above may contain evidence measured as
`900` native tokens. Reporting both distinguishes evidence workload from the
complete input cost without asserting that every remaining token is an instruction.

**Range and direction:** Non-negative integer tokens; descriptive.

**Edge cases and availability:**

A known empty evidence component, where the request contract permits it, has
count zero. An unrecorded component is unavailable under the shared rules, not
an empty component. Retrieved-token counts using `cl100k_base` are separate.

### Reasoning Tokens

**Question:** How much output belongs to the exposed reasoning stream?

Count generated tokens in the runtime's reliably separated reasoning component.
Hidden or unexposed reasoning is not inferred from elapsed time or the answer.

**Example:** An exposed reasoning span of `30` tokens records `30`. A runtime
that returns only a visible answer without a reliable reasoning boundary records
null, not an assumption of zero reasoning.

**Range and direction:** Non-negative integer tokens when observed; descriptive.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Reasoning is not reliably separated or exposed | Only a visible response is returned | `null` (`unavailable`) | Do not infer hidden reasoning length from latency or answer text. |
| Reasoning absence is verified | A supported direct mode exposes no reasoning events | `0` (`scored`) | Known absence is different from missing visibility. |

### Visible-Answer Tokens

**Question:** How much visible answer text did the model deliver?

Count the `answer` string with the native tokenizer, excluding separated
reasoning and the JSON wrapper/citation fields. This describes answer length,
not the number of required facts recovered.

**Example:** A valid response whose answer string encodes to `20` native tokens
records `20`, whether those tokens express correct facts or unsupported claims.

**Range and direction:** Non-negative integer tokens; descriptive.

**Edge cases and availability:**

A known empty `answer` string counts zero even if it fails answer behavior.
An absent/unusable field follows the shared unavailable-observation rule, not
zero answer length. Preserve raw generated-token counts independently.

### Total Output Tokens

**Question:** How many native generated-token events did the request produce?

Count all generated events, including reasoning, visible output, JSON/control
syntax, and emitted termination tokens; exclude input/padding. This is the `N`
used for Decode Throughput, not just the retokenized visible answer string.

**Example:** A request producing `60` output tokens can have `30` reasoning
tokens and a visible answer counted as `20` tokens, with the rest belonging to
other output syntax/control events. Do not require these separately measured
components to sum to `60` when boundaries or retokenizations differ.

**Range and direction:** Non-negative integer tokens; descriptive.

**Edge cases and availability:**

An observed stream known to emit no token events has count zero. Missing token
records are unavailable, not zero. Partial streams follow the shared scoped
observation rules; their real token counts do not establish successful completion.

## Per-response diagnostic artifacts

The values below support audit and rescoring without adding MLflow summary
metrics or changing question/document weighting. They are artifact fields and
inventories, not another selection-metric family.

### Citation Count

**Question:** How many distinct evidence IDs does the response select?

Count distinct citation IDs after the scoring deduplication; retain the original
ordered list and invalid-ID diagnostics. A large count is not automatically
better citation quality.

**Example:** `["E1", "E1", "E2"]` has a distinct Citation Count of `2`,
not `3`. If `E2` is legal but irrelevant, count is still `2` while Citation
Precision is only `1 / 2`.

**Range and direction:** Non-negative integer IDs; descriptive. A valid empty
list counts zero; unknown string IDs remain counted and separately flagged.

**Edge cases and availability:**

Missing or unusable citation fields follow shared availability rules rather than
becoming empty lists. ID-reference validity is separate from structural validity.

### Generated-Claim Count

**Question:** How many distinct factual assertions must be verified?

Count atomic, semantically deduplicated visible-answer claims from the calibrated
judge's extraction labels, not sentences or required gold facts.

**Example:** "The trial enrolled 500 adults and lasted six months" yields
three claims: enrollment count, adult population, and duration. Repeating the
duration does not create a fourth distinct claim.

**Range and direction:** Non-negative integer claims; descriptive.

**Edge cases and availability:**

Completed extraction finding no claims gives zero. Missing labels follow the
shared unavailable-evaluation rule, not a fallback claim-free answer.

### Finish-reason inventory

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

**Edge cases and availability:**

Apply shared missing-record rules: no recorded terminations means undefined
shares, with missing/unexecuted support retained. Keep runtime outcome, decoding
finish reason, schema validity, and ID validity in separate fields. For example,
EOS with malformed JSON is normal decoding completion but structural failure;
an input-context rejection has no decoding finish reason because it never began.

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

The quality/rate values are document-macro estimates, not pooled percentages of
all claims or attempts; the two latency values are named percentiles. Calculate
F1 per response before averaging, not from the displayed aggregate precision
and recall. The answers are usually well supported
and directly relevant. Required-fact coverage is weaker than generated-claim
precision; the different denominators do not establish corpus-wide error counts.
Refusal Validity shows
that unanswerable questions remain a distinct weakness. The non-zero malformed
rate identifies a schema issue separate from answer/refusal behavior, while
repeatability and latency describe stability and runtime cost. Malformed Output
Rate describes normally completed outputs, not all scheduled attempts. Judge
coverage, defined claim ratios, completion counts, and every metric's support
remain necessary to interpret the displayed estimates.


## Eligibility, aggregation, and confidence intervals

Quality is evaluated on each of the three scheduled outputs. For each metric,
average defined output scores within a question, then contributing question
scores within a source document, then documents equally. Calculate F1 per output
before those averages. Never select a best response or replace an unavailable
score with a later favorable output. Every seed's execution and evaluator outcome
remains recorded even when its score is null.

Apply the [shared reporting-slice](../../metrics.md#reporting-slices) and
[partial-evidence rules](../../metrics.md#dependent-metrics-and-partial-observations).
Known claim-free null ratios are explicit evaluated conventions, not missing
judge labels; they retain reference eligibility and `no_generated_claims`
support. Missing required evaluations or unavailable outputs leave surviving
quality partial/diagnostic, not a complete comparison. If every reference-eligible
output was evaluated and every claim ratio has `no_generated_claims`, its
aggregate is `null/inapplicable` with zero generated claim support and original
reference-eligible question/source counts intact. If no scores are defined
because outputs or evaluations were unavailable, the aggregate is unavailable
instead; do not disguise missing evidence as a known absence of claims.

Response/Refusal Validity and reliability calculate question-level rates over
their eligible scheduled seeds. Malformed Output Rate instead divides malformed
normally completed outputs by normally completed generations within the question;
no completed output gives a null question rate. Average each family's defined
question rates within documents, then documents. Repeatability averages its
three scheduled pair contributions within each question before the same
document-macro aggregation. Do not replace these means with pooled event ratios.

For example, document A has four question values `1, 1, 0, 0` and document B
has one question value `1`. Their document means are `0.50` and `1.00`; the
reported document-macro value is `0.75`. The pooled question average would be
`3 / 5 = 0.60`, giving the longer document four times the influence. Equal
document weight is used consistently for generation quality and rates.

Development, validation, and locked runs apply the
[shared bootstrap and confidence-interval rules](../../metrics.md#confidence-intervals)
to complete source documents; the bootstrap seed is independent of generation
seeds. Answerable-only, unanswerable-only, and evidence-type
results always report reference-eligible, executed, evaluated, contributing,
claim-free, failed, and unavailable support as applicable. A refusal on an
answerable question still has known missed-fact and citation scores; it does
not remove the question from all quality calculations. Runtime failures and
unavailable judging remain visible instead of numeric imputation.

Operational metrics keep their own estimators over this fixed question set:
TTFT and successful-completion latency use per-question medians followed by
cross-question percentiles; Decode Throughput pools eligible work/time across
all questions/seeds; resource peaks take the maximum valid lifecycle sample.
Token descriptors retain their declared descriptive aggregation and support.
These are not document-macro quality scores. A missing required record or
instrument makes affected evidence incomplete under the shared rules.

### Interval support and calculation

| Value | CI applicability | Statistic recalculated on each source draw |
|---|---|---|
| Quality, behavioral validity, reliability, and repeatability | When defined and sufficiently supported | The metric's output/pair/question/document aggregation, with its own denominator and availability conventions. |
| TTFT and completion-latency p50/p95 | Conditional on observed eligible durations and sufficient support | Percentiles of the sampled documents' per-question observations. |
| Decode Throughput | Conditional on normally completed, instrumented streams and sufficient support | Sum eligible counted decoding tokens divided by sum corresponding decoding intervals. |
| One cold load or observed lifecycle memory peak | No | One lifecycle observation is not an independent-sample distribution. |
| Token descriptors, citation/claim counts, and finish-reason inventories | No | Descriptive counts/inventories are not additional quality estimates. |

The independent bootstrap unit is the complete reference-eligible source
document, including questions, all three seed outputs, pair labels, failures,
and unavailable evaluations. Follow the shared resampling procedure rather
than calculating and averaging three seed-specific intervals. The table's
statistic is recomputed from the retained records on every draw. This estimates
source-sampling uncertainty under the frozen three-seed protocol, not every
possible future generation seed.

Report eligible and contributing independent-source counts, not `3 × questions`
as independent support. Missing required judge labels or scheduled seeds keep
any surviving-output estimate incomplete/diagnostic under the shared availability
rules, not a complete comparison obtained by omitting missing rows.

The numeric independent-source support minimum remains pending data review.
Its freeze follows the [shared CI-support deadline](../../metrics.md#confidence-intervals),
before any authoritative comparison, including development. Earlier exploratory
points remain descriptive, with null bounds and a reason. Apply the shared
undefined-draw and degeneracy rules;
calculated bounds such as `[1, 1]` are not a claim of future certainty.

For example, `Faithfulness = 0.90, 95% CI [0.86, 0.94]` describes uncertainty
in the document-macro estimate, not a range of individual answers. Bootstrap
resampling performs arithmetic on saved labels; it launches neither extra
generation nor extra judge calls.
