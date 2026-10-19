# Audio extraction metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Scoring inputs and result status](#scoring-inputs-and-result-status)
- [Recognition quality](#recognition-quality-1)
- [WER diagnostics](#wer-diagnostics-1)
- [Timestamp quality](#timestamp-quality-1)
- [Reliability and failure behavior](#reliability-and-failure-behavior-1)
- [Operational performance](#operational-performance-1)
- [Worked candidate interpretation](#worked-candidate-interpretation)
- [Audio confidence intervals](#audio-confidence-intervals)

The ASR benchmark evaluates the complete ordered transcript, its timestamps,
catastrophic output behavior, and the cost of the recorded runtime profile. It
does not report Content F1 or document Reading Order NED as transcript-quality
metrics: unlike a two-dimensional page, audio already defines one chronological
sequence. Content F1 is used internally to match timestamp spans, as described
under [Timestamp quality](#timestamp-quality-1).

## Metric summary

### Recognition quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Corpus Word Error Rate (WER) | Primary | How wrong is the complete ordered word transcript? | Lower |
| Corpus Character Error Rate (CER) | Secondary | How severe are character-level spelling, name, and number errors? | Lower |

### WER diagnostics

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Word Substitution Rate | Diagnostic | How often is a spoken word recognized as a different word? | Lower |
| Word Deletion Rate | Diagnostic | How much spoken content is omitted? | Lower |
| Word Insertion Rate | Diagnostic | How much unsupported word content is added? | Lower |

### Timestamp quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Timestamp Boundary MAE | Primary | How far are aligned segment starts and ends from the reference boundaries? | Lower |
| Timestamp Alignment Coverage | Primary | What proportion of timed reference segments received a valid alignment? | Higher |

### Reliability and failure behavior

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Unexpected Empty Transcript Rate | Diagnostic | How often does a completed first speech attempt return empty when lexical text is required? | Lower |
| Nonspeech False-Transcription Rate | Diagnostic | How often does a completed first nonspeech attempt invent lexical text? | Lower |
| Transcript Repeatability Success Rate | Diagnostic | How often do two scheduled attempts both return the same valid projected transcript? | Higher |
| Attempt Failure Rate | Diagnostic | How often does a measured speech or control request fail to deliver a valid output? | Lower |

### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Complete-Pipeline Real-Time Factor | Operational | How much processing time is required relative to audio duration? | Lower |
| p50 Warm Clip Latency | Operational | What is normal warm processing latency? | Lower |
| p95 Warm Clip Latency | Operational | What is slow-case warm processing latency? | Lower |
| Cold Model-Load Time | Operational | How long does initial model loading take? | Lower |
| Peak Process-Tree RAM | Operational | How much total system memory does the candidate require? | Lower |
| Peak Device VRAM | Operational | How much total GPU memory is occupied during candidate execution? | Lower |

These 17 metrics define the implemented ASR evaluation contract.
Diarization metrics remain out of scope until speaker identification becomes a
product requirement.

## Scoring inputs and result status

Apply the [shared first-output and availability rules](../../metrics.md#shared-availability-rules)
to the designated first measured attempt after warmup. Later attempts supply
timing, repeatability, and failure evidence, not additional quality scores.

Corpus WER/CER and the three word-error rates pool counts from valid completed
first speech outputs. A recorded failed first attempt has unavailable edit
diagnostics, not an invented empty transcript or arbitrary maximum error.
Applicable timestamp metrics are also unavailable after a failed first attempt.
Retain its known reference denominators as planned support, not scored recovery failures.
Always report scheduled and contributing clips/source groups, failed first
attempts, and planned versus contributing reference words/characters. The
conditional recognition result cannot stand in for workload-wide reliability.

Result statuses, retained rows/reasons, and omitted MLflow null scalars follow
the [shared availability rules](../../metrics.md#shared-availability-rules).
ASR artifacts include null values and reasons in `candidate.json` and
`summary.json`; matched-boundary availability is specified under Timestamp
Boundary MAE. Invalid references follow
[data validation](../../data-validation.md#data-validity-and-metric-applicability).

An empty transcript with an empty timestamp sequence is a valid completed
output. Nonempty lexical text without required timestamps, empty text with
lexical timestamp segments, or invalid required boundaries/schema are failed
attempts. Continue remaining attempts after recoverable failures; missing
records after a fatal setup/process failure make the execution incomplete.
Reference eligibility is fixed before inference and identical for all candidates.
See [data validation](../../data-validation.md#asr-references-and-controls) for
preparation checks, not prediction scoring. Inputs require positive audio
duration, explicit reviewed transcripts and timing annotations, some normalized
reference words and eligible timed segments in each full authoritative speech
split, and reviewed controls from all four nonspeech categories. Legitimate
individual empty projections and deliberately task-free diagnostic slices do
not waive those authoritative corpus requirements. Bootstrap draws are not new
manifests subject to the corpus quota.

Attempt Failure Rate uses the scheduled speech/control requests defined below,
not setup outcomes. Its completeness and fully recorded failure evidence follow
the shared rules, as does speech-repeatability availability.

All individual availability blocks also use the
[shared availability rules](../../metrics.md#shared-availability-rules).
The [edge-case classification](../../metrics.md#what-belongs-in-a-metrics-edge-case-block)
separates metric conventions from input rejection and runtime integrity.

## Recognition quality

Recognition applies the [shared prose projection](../../metrics.md#prose-scoring-projection)
and [recognition error/empty-reference conventions](../../metrics.md#recognition-error-rates-and-empty-references).
The tables below apply those conventions to ASR's pooled completed speech pairs,
not to nonspeech controls or failed requests.

### Corpus Word Error Rate

**Question:** How wrong is the complete ordered word transcript?

Corpus WER uses the [shared projected words](../../metrics.md#prose-scoring-projection),
not model-tokenizer subwords.
Align each clip separately, then pool edit counts across valid completed first
speech outputs before
division; the benchmark does not average clip error rates. Raw transcripts are
not rewritten; controls are excluded from this speech aggregate.

```text
Corpus WER =
all word substitutions + deletions + insertions
------------------------------------------------
        reference words in those completed pairs
```

**Example:** Across the complete corpus, 1,000 reference words with 40
substitutions, 20 deletions, and 10 insertions produce WER `70 / 1,000 = 0.07`.

WER is primary because ASR must reproduce the ordered spoken sequence in one
value. The component rates below explain its failure type rather than replacing
it.

For a valid nonempty reference with an empty prediction, all words are deleted,
so WER is `1`. Pool raw edit counts before applying a zero-reference convention;
do not average individual empty-reference clip rates into a corpus with
reference content.

**Range and direction:** WER is non-negative and lower is better. It can exceed
`1` when insertions outnumber reference words.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Zero reference words and no predicted words in valid contributing pairs | Projected reference and prediction are both empty | `0` (`scored`) | No required or invented lexical words; this is a valid empty comparison. |
| Zero reference words with predicted words in valid contributing pairs | Empty reference, prediction `go now` | Total word insertions (`2` here) (`scored`) | Use the total word insertion count, not division by zero. The same rule applies to a pooled zero-reference subset. |

Shared availability rules apply: [first-output and aggregate availability](../../metrics.md#shared-availability-rules).

### Corpus Character Error Rate

**Question:** How severe are character-level transcription errors that may be
hidden by whole-word scoring?

Corpus CER uses the same valid first-output pairs and pooled calculation over
characters of the projected strings, including their remaining spaces:

```text
Corpus CER =
all character substitutions + deletions + insertions
-----------------------------------------------------
          reference characters in those completed pairs
```

**Example:** Fifty character edits across 5,000 reference characters produce
CER `50 / 5,000 = 0.01`.

CER supports WER by exposing small spelling, number, abbreviation, and name
errors that a whole-word error treats as one event.

For a valid nonempty reference with an empty prediction, all characters are
deleted, so CER is `1`. As with WER, pool raw character edits before applying
the zero-reference convention rather than averaging clip rates.

**Range and direction:** CER is non-negative and lower is better. It can exceed
`1` when insertions outnumber reference characters.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Zero reference characters and no predicted characters in valid contributing pairs | Projected reference and prediction are both empty | `0` (`scored`) | No required or invented lexical characters; this is a valid empty comparison. |
| Zero reference characters with predicted characters in valid contributing pairs | Empty reference, prediction `go now` | Total character insertions (`6` here) (`scored`) | Use the total character insertion count, including the space. The same rule applies to a pooled zero-reference subset. |

Shared availability rules apply: [first-output and aggregate availability](../../metrics.md#shared-availability-rules).

## WER diagnostics

The three diagnostic rates reuse the exact word alignment and pooled reference
word count used by Corpus WER. Together they add up to WER, but each answers a
different failure question.

For a positive pooled reference-word count, divide each pooled edit count by
that same count. Raw integer substitutions, deletions, insertions, and reference
lengths remain in `samples.parquet`; the three reported metrics are rates.
The [shared ASR component-rate extension](../../metrics.md#recognition-error-rates-and-empty-references)
preserves the component sum as WER for zero-reference comparisons; the tables
below apply it to the same pooled speech pairs.

### Word Substitution Rate

**Question:** How often is a spoken reference word replaced by a different
predicted word?

```text
Word Substitution Rate =
substituted reference words in valid completed first speech outputs
------------------------------------------------------------------
          reference words in those same completed pairs
```

**Example:** Forty substitutions among 1,000 reference words produce `0.04`.

This distinguishes word confusion from omitted or invented speech.

**Range and direction:** Non-negative and lower is better. It cannot exceed `1`
because each reference word can be substituted at most once.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Zero reference words in valid contributing pairs | Empty reference with either an empty prediction or `go now` | `0` (`scored`) | With no reference word to substitute, the benchmark defines the rate as zero. Added words are insertions. |

Shared availability rules apply: [first-output and aggregate availability](../../metrics.md#shared-availability-rules).

### Word Deletion Rate

**Question:** How much spoken reference content is missing from the transcript?

```text
Word Deletion Rate =
deleted reference words in valid completed first speech outputs
--------------------------------------------------------------
        reference words in those same completed pairs
```

**Example:** Twenty deletions among 1,000 reference words produce `0.02`.

This exposes omissions that are especially harmful when lectures contain
definitions, negations, or instructions.
For a valid nonempty reference with an empty prediction, every reference word
is deleted and the rate is `1`.

**Range and direction:** Lies in `[0, 1]`; lower is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Zero reference words in valid contributing pairs | Empty reference with either an empty prediction or `go now` | `0` (`scored`) | With no reference word to delete, the benchmark defines the rate as zero. |

Shared availability rules apply: [first-output and aggregate availability](../../metrics.md#shared-availability-rules).

### Word Insertion Rate

**Question:** How much unsupported word content did the model add?

```text
Word Insertion Rate =
inserted predicted words in valid completed first speech outputs
---------------------------------------------------------------
         reference words in those same completed pairs
```

**Example:** Ten insertions against 1,000 reference words produce `0.01`.

This exposes invented speech that is not present in the recording.

**Range and direction:** Non-negative and lower is better. It can exceed `1`
when the model inserts more words than the entire reference contains.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Zero reference words and no insertions in valid contributing pairs | Empty reference and empty prediction | `0` (`scored`) | The benchmark's zero-reference insertion-count convention gives zero. |
| Zero reference words with insertions in valid contributing pairs | Empty reference, prediction `go now` | Total insertions (`2` here) (`scored`) | Use the total insertion count, preserving the component sum as WER without dividing by zero. |

Shared availability rules apply: [first-output and aggregate availability](../../metrics.md#shared-availability-rules).

## Timestamp quality

Apply the [shared spoken-timestamp alignment](../../metrics.md#spoken-timestamp-alignment)
to valid completed first clip outputs. It supports segment-level references
and word-level model timestamps without allowing one prediction to cover
multiple required segments.

Boundary MAE alone can look excellent when only easy segments align. Coverage
shows how much of the timed reference actually contributed. The two metrics are
therefore interpreted together: low MAE and high coverage. When no segments
align, their distinct denominator rules below keep missing boundary measurements
separate from missed alignment.

Pool actual matched boundary-error totals for MAE. Pool matched and required
segment counts from valid completed first outputs for coverage, including zero
matches from valid empty outputs. A failed first attempt contributes neither a
fabricated boundary error nor a scored zero-coverage observation. Keep its known
reference count in planned support and report contributing, failed, and unavailable
clips alongside both metrics. Coverage diagnoses missed alignment among completed
outputs; Attempt Failure Rate diagnoses execution failures.

### Timestamp Boundary MAE

**Question:** How far are aligned spoken start/end boundaries from their
reviewed times?

```text
Timestamp Boundary MAE = sum of absolute start and end errors / matched boundaries
```

Take both errors per aligned segment and pool their totals across valid first
outputs; do not average clip MAEs equally. A matched span has two boundaries.

**Example:** Reference `[2.0, 4.0]` seconds and predicted span `[2.3, 3.9]`
give errors `0.3` and `0.1`, hence MAE `(0.3 + 0.1) / 2 = 0.20 s`.
Across 18 matched segments with `7.2 s` total error, MAE is likewise
`7.2 / 36 = 0.20 s`.

Low MAE means aligned segments are precisely timed, not that all speech aligned.
Read it with Alignment Coverage to expose a small easy matched subset.

**Range and direction:** Non-negative seconds; lower is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No reference timing task | A reviewed empty projected clip has an explicit empty reference-segment list | `null` (`inapplicable`) | No spoken boundaries are required, so there is no timing-error task. |
| A valid completed timing task has no matched boundaries | Twenty reference segments, zero aligned spans, including a valid empty output | `null` (`unavailable`) | The matched-boundary denominator is zero; do not invent perfect or infinitely bad boundary errors. |

Shared availability rules apply: [first-output and aggregate availability](../../metrics.md#shared-availability-rules).
Keep matched-boundary and independent-source support for the
[conditional confidence interval](#audio-confidence-intervals).

### Timestamp Alignment Coverage

**Question:** How much of the required spoken timing task received an alignment?

```text
Timestamp Alignment Coverage = matched reference segments / required reference segments
```

Pool segment counts from valid completed first outputs, including valid outputs
with zero matches. Do not divide by predicted segments or include the planned
denominators of failed first attempts as observed misses.
A valid completed timing task with no matches, including a valid empty output,
receives zero coverage because its required-segment denominator is positive.

**Example:** Eighteen matches among twenty required segments give
`18 / 20 = 0.90`. If another valid clip has ten required segments and no matches,
pooled coverage is `18 / 30 = 0.60`, not the mean of `0.90` and `0`.

High coverage means most required segments align; it does not establish accurate
boundaries or penalize every extra predicted segment. Boundary MAE and WER supply
those complementary views.

**Range and direction:** `[0, 1]`; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No reference timing task | A reviewed empty projected clip has an explicit empty reference-segment list | `null` (`inapplicable`) | With no required segments, there is no alignment task; this is not zero recovery of real speech. |

Shared availability rules apply: [first-output and aggregate availability](../../metrics.md#shared-availability-rules).
Planned and contributing segment support remains visible.

## Reliability and failure behavior

### Unexpected Empty Transcript Rate

**Question:** Among completed first speech attempts that require lexical text,
how often does transcription return empty?

Divide valid empty first outputs by valid completed first outputs for speech
clips with nonempty projected references. A completed output need not be correct:
empty text plus empty timestamps is valid execution and counts as an event.
Read this conditional rate with Attempt Failure Rate and the
[output-validity rules](#scoring-inputs-and-result-status).

**Example:** Of 100 eligible first speech attempts, 10 fail and 90 return valid
outputs; four of those are empty. The rate is `4 / 90`, approximately `0.044`.
The 10 failures remain in failure and support artifacts, not hidden as empties.
No empty completed outputs gives `0`; all eligible completed outputs empty
gives `1`.

This denominator intentionally differs from document Unexpected Empty Output
Rate, whose denominator is all scheduled documents. They answer different
questions; neither rate silently changes to the other's denominator.

**Range and direction:** `[0, 1]`; lower is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| No expected lexical task in a diagnostic slice | All reviewed projected speech references in the slice are empty | `null` (`inapplicable`) | No clip is eligible to return an unexpectedly empty lexical transcript. |
| Eligible lexical tasks but no valid completed first outputs | Positive eligible count, zero completed first-output denominator | `null` (`unavailable`); `no_successful_first_outputs` | No completed-output event rate can be measured; this is not an observed zero-empty rate. |

Shared availability rules apply: [execution and record completeness](../../metrics.md#shared-availability-rules).

### Nonspeech False-Transcription Rate

**Question:** How often does the model invent lexical speech on verified
nonspeech audio?

For the fixed silence, music-without-lyrics, background-noise, and environmental
sound controls:

```text
Nonspeech False-Transcription Rate =
valid completed first control outputs producing lexical text
------------------------------------------------------------
         valid completed first control outputs
```

**Example:** Three controls producing text among 20 valid completed first
control outputs produce `3 / 20 = 0.15`. If another two scheduled controls fail,
they remain failure records; the event denominator is still 20, not 22.

Empty-reference WER can count insertions under the JiWER convention, but these
controls are deliberately excluded from speech Corpus WER/CER. The dedicated
rate measures how often verified nonspeech produces invented lexical speech,
without weighting it by the speech corpus's reference-word count.
When none of the completed controls invents text, the rate is `0`; when all do,
it is `1`.

**Range and direction:** `[0, 1]`; lower is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| A deliberately control-free diagnostic slice | A speech-only diagnostic slice declares no nonspeech controls | `null` (`inapplicable`) | No false-transcription control task was scheduled; authoritative control requirements still apply. |
| Eligible controls but no valid completed first outputs | Positive eligible count, zero completed first-control-output denominator | `null` (`unavailable`); `no_successful_first_outputs` | The conditional event denominator is empty, not evidence of no invented speech. |

Shared availability rules apply: [execution and record completeness](../../metrics.md#shared-availability-rules).

### Transcript Repeatability Success Rate

**Question:** How often does the same model profile return the same transcript
in two scheduled attempts on the same clip?

Apply [shared pairwise repeatability](../../metrics.md#pairwise-repeatability)
to each speech clip's three measured attempts. Agreement means identical
projected transcripts, not identical timestamp arrays. Divide successful pairs
by the three scheduled pairs, then average clip scores. Nonspeech controls do
not enter this speech-repeatability aggregate.

**Example:** Distinct valid projected transcripts are denoted A/B/C. Outcomes
`A, A, A` give `3 / 3 = 1`; `A, A, B` or `A, A, failure` give `1 / 3`.
Outcomes `A, B, C`, `A, B, failure`, `A, failure, failure`, or three recorded
failures give `0 / 3 = 0`. Clip scores `1`, `1/3`, and `0` average to
approximately `0.444`.

Three valid empty transcripts also score `1`; their recognition quality is
assessed independently.
This diagnoses repeatable delivery, not correctness: identical wrong answers
can repeat perfectly. It uses the already collected measured attempts, with
the same seed/settings, and adds no inference or quality rescoring.

**Range and direction:** `[0, 1]`; higher is better.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| Repeatability is not measured | Smoke schedules one measured attempt per clip | `null` (`inapplicable`); `repeatability_not_measured` | The profile deliberately supplies no attempt pairs; this is not disagreement. |

Shared availability rules apply: [execution and record completeness](../../metrics.md#shared-availability-rules).

### Attempt Failure Rate

**Question:** How often does a measured request fail to deliver a valid output?

Divide recorded failed measured attempts by all scheduled measured attempts,
including speech and nonspeech controls. Authoritative speech clips have three
attempts; controls have one. Keep separate speech/control breakdowns and the
actual attempt counts in artifacts. Later speech successes do not repair the
first attempt's quality result.

**Example:** Two speech clips schedule six attempts and two controls schedule
two. If two requests fail, the total rate is `2 / 8 = 0.25`.
All scheduled attempts succeeding gives `0`; all executing and failing gives
`1`. This is measured reliability, not an automatic candidate-selection rule.

**Range and direction:** `[0, 1]`; lower is better.

**Edge cases and availability:**

Shared availability rules apply. See [shared availability](../../metrics.md#shared-availability-rules)
and the [audio execution-status rules](#scoring-inputs-and-result-status).

## Operational performance

Timing validity, completed-phase availability, and failure diagnostics follow
the [shared availability rules](../../metrics.md#shared-availability-rules).
RAM/VRAM instrumentation and partial-window scope follow
[shared resource measurements](../../metrics.md#resource-measurements).
The ASR monitoring window includes loading, warmup, and measured inference;
it excludes data preparation/validation and offline scoring. Interval eligibility
and warm-latency support follow [audio confidence intervals](#audio-confidence-intervals).

Every result records its explicit CPU/GPU profile; silent fallback is invalid.
Extraction has no artificial VRAM cap, and observed memory use alone does not
exclude a profile. The [execution prerequisites](methodology.md#per-candidate-execution)
remain separate from metric edge cases.

### Complete-Pipeline Real-Time Factor

**Question:** How much complete processing time is required relative to audio
duration?

```text
Complete-Pipeline Real-Time Factor =
processing time of successful measured speech attempts
------------------------------------------------------
audio duration of those same successful attempts
```

**Example:** Processing 60 minutes of audio in 15 minutes produces RTF `15 / 60
= 0.25`.

An RTF below `1` means processing is faster than audio playback; above `1`
means slower than playback.

The request timer covers candidate input reading/preprocessing, native features,
transcription, required timestamp/alignment work, output conversion/validation,
and device synchronization. Dataset acquisition, reference review, and offline
metric scoring are not transcription requests.

Count each completed repetition's audio duration once: three successful
30-second attempts contribute 90 seconds, not 30. Valid empty outputs count as
completed execution. Failed-attempt elapsed times remain artifacts but do not
describe completed transcription speed. Controls, cold loading, warmup, data
validation, and offline scoring are outside this RTF.

**Range and direction:** Non-negative, unbounded; lower is better.

**Edge cases and availability:**

Shared availability rules apply. See [shared availability](../../metrics.md#shared-availability-rules)
and [operational measurement availability](#operational-performance-1).
No completed timed speech attempt gives `null/unavailable`, not zero RTF.

### p50 and p95 Warm Clip Latency

For each speech clip, take the median latency of its successful measured warm
attempts, then report p50/p95 across those clip medians. Report contributing and
failed-attempt counts; a clip with no completed timed attempt contributes no
completed latency. Retain failed elapsed times separately.

Apply the [shared latency-percentile calculation](../../metrics.md#latency-percentiles)
to these clip medians. They describe observed clips; source support for an
authoritative tail estimate follows the local interval rules.

#### p50 Warm Clip Latency

**Question:** What is the typical successful warm transcription wait per clip?

Take the median of the clip-level successful-attempt medians.

**Example:** Clip observations `2, 3, 4, 5, 11 s` have p50 `4 s`.
One clip's attempts `3, 4, failure` supply median `3.5 s`, not the failure time.

The median is a typical wait, not an arithmetic mean or duration-normalized speed.

**Range and direction:** Non-negative seconds/clip; lower is better.

**Edge cases and availability:**

Shared availability rules apply. See [shared availability](../../metrics.md#shared-availability-rules)
and [operational measurement availability](#operational-performance-1).
No completed timed clips gives `null/unavailable`, not a zero-second median.
Independent-source interval support follows [audio confidence intervals](#audio-confidence-intervals).

#### p95 Warm Clip Latency

**Question:** What is the slow-tail successful warm transcription wait per clip?

Take the linearly interpolated 95th percentile of the same clip observations.

**Example:** For `2, 3, 4, 5, 11 s`, position `4 × 0.95 = 3.8` lies 80% of
the way from `5` to `11`; p95 is `5 + 0.8 × (11 − 5) = 9.8 s`.

This is not a maximum, nor does it imply exactly 95% of five observations falls
below `9.8 s`.

**Range and direction:** Non-negative seconds/clip; lower is better.

**Edge cases and availability:**

Shared availability rules apply. See [shared availability](../../metrics.md#shared-availability-rules)
and [operational measurement availability](#operational-performance-1).
No completed timed clips gives `null/unavailable`, not a zero-second tail latency.
Independent-source interval support follows [audio confidence intervals](#audio-confidence-intervals).

### Cold Model-Load Time

**Question:** How long does a fresh worker need to make the ASR model ready?

Measure from the start of model construction until the complete ASR profile is
ready, before warmups or transcription. Any component needed before the first
request belongs to cold load; work performed later belongs to the measured
complete-pipeline latency and cannot be hidden from both measurements.

**Example:** Construction starts at `0.4 s` and full readiness at `5.4 s`:
cold load is `5.4 − 0.4 = 5.0 s`. This is initialization cost, not warm latency
or first-request latency including warmup.

**Range and direction:** Non-negative seconds; lower is better.

**Edge cases and availability:**

Shared availability rules apply. See [shared availability](../../metrics.md#shared-availability-rules)
and [operational measurement availability](#operational-performance-1).

### Peak Process-Tree RAM

**Question:** How much host memory does the complete ASR worker require?

Apply [shared Peak Process-Tree RAM](../../metrics.md#peak-process-tree-ram)
to the ASR worker and its descendants over the monitoring window above.

**Example:** At two samples, worker/child RAM is `2,000 + 1,200 = 3,200 MiB`
and `2,400 + 500 = 2,900 MiB`. The peak is `3,200 MiB`, not the sum of the
processes' separate peaks (`3,600 MiB`). This estimates simultaneous host demand.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

**Edge cases and availability:**

Shared availability rules apply. See [shared availability](../../metrics.md#shared-availability-rules)
and [resource-window and measurement availability](#operational-performance-1).

### Peak Device VRAM

**Question:** How much total GPU memory is occupied during the ASR lifecycle?

Apply [shared Peak Device VRAM](../../metrics.md#peak-device-vram)
over the ASR monitoring window, including loading and first-use allocations.

**Example:** Device samples `700, 2,100, 1,900 MiB` give `2,100 MiB`, not a
baseline-subtracted `1,400 MiB`. This is observed device demand, not model size.

**Range and direction:** Non-negative MiB; lower is better at equal quality.

**Edge cases and availability:**

| Edge case | Example | Value/status | Explanation |
|---|---|---|---|
| CPU-only profile | An explicitly requested CPU smoke run | `null` (`inapplicable`) | No GPU-memory measurement task exists; this is not zero observed CUDA use. |

Shared availability rules apply: [CUDA telemetry availability](../../metrics.md#shared-availability-rules)
and [resource-window measurement scope](#operational-performance-1).

## Worked candidate interpretation

Suppose one ASR profile produces:

```text
Corpus WER                         = 0.08
Word Deletion Rate                 = 0.04
Timestamp Boundary MAE             = 0.32 seconds
Timestamp Alignment Coverage       = 0.94
Nonspeech False-Transcription Rate = 0.10
Complete-Pipeline RTF              = 0.40
Peak Device VRAM                   = 2,100 MiB
```

The profile transcribes faster than playback and aligns most timed segments,
but half of its scored word errors come from omitted speech and it invents text
on 10% of valid completed first nonspeech outputs. Check the contributing and
failure counts before interpreting those conditional quality values. Those
failure modes remain visible even though the total
WER is relatively low. No weighted overall score combines these values.

## Audio confidence intervals

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Development, validation, and locked WER, CER, and substitution/deletion/insertion rates | When defined and sufficiently supported | Resample all declared speech units, then recalculate pooled valid-first-output counts. |
| Development, validation, and locked Timestamp Alignment Coverage | When defined and sufficiently supported | Recalculate matched/required counts from valid completed first outputs; retain failures and planned denominators separately. |
| Development, validation, and locked Timestamp Boundary MAE | When defined and sufficiently supported | Resample all speech units, then use their actual matched boundary totals. |
| Development, validation, and locked Unexpected Empty Transcript Rate | When defined and sufficiently supported | Recalculate the valid completed eligible first-output denominator in each draw. |
| Development, validation, and locked Nonspeech False-Transcription Rate | When defined and sufficiently supported | Resample controls separately and recalculate their completed first-output denominator. |
| Development, validation, and locked Transcript Repeatability Success Rate | When complete and sufficiently supported | Retain all scheduled attempt outcomes and average pairwise clip scores. |
| Development, validation, and locked Attempt Failure Rate | When complete and sufficiently supported | Resample speech and control source units, preserving their scheduled/failed attempt totals. |
| Development, validation, and locked Real-Time Factor | When defined and sufficiently supported | Recalculate successful measured time divided by the same attempts' audio durations. |
| Development, validation, and locked p50/p95 warm latency | Conditional | Report only when enough independent clips support the percentile estimate. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| One cold-load measurement | No | One observation cannot estimate uncertainty. |
| One observed peak RAM or VRAM value | No | Report the observed peak without invented bounds. |

### Calculation

Use the [shared bootstrap algorithm, parameters, and support rules](../../metrics.md#confidence-intervals)
with the independent source IDs frozen in the manifest. Related clips from one
declared source stay together with all attempts, first-output statuses, and
reference denominators. Resample nonspeech control units separately for their
event rate and their contribution to combined Attempt Failure Rate, preserving
the speech/control allocation. Recalculate the local pooled/conditional
statistics; do not prefilter failed or unmatched clips before drawing.

A draw with valid completed timed outputs but no matches contributes zero
Alignment Coverage; its Boundary MAE is undefined. A draw with no valid completed
timed outputs has unavailable coverage as well as MAE. A draw with no valid first
outputs cannot contribute a recognition or corresponding event-rate value,
but still contributes defined failure/repeatability evidence. Legitimate
zero-reference contributing subsets use the empty-reference conventions;
bootstrap draws do not rerun the corpus-validation quota.

Both support minima are currently null in the protocol pending data review;
the calculation is implemented, but this does not supply an authoritative
support threshold. See [pending data review](../../pending-data-review.md#audio-extraction).

Conditional availability follows the individual
[Boundary MAE](#timestamp-boundary-mae) and
[Alignment Coverage](#timestamp-alignment-coverage) blocks. A fully recorded
execution with an undefined conditional value is not a missing-record failure;
retain it under the [result-status rules](#scoring-inputs-and-result-status).

### Interpretation

Corpus WER `0.08` with a 95% CI of `[0.07, 0.10]` means the observed aggregate
error rate on valid completed first outputs is 8%, while independent-source
resampling estimates a plausible range of 7% to
10%. A narrower interval indicates a more precise corpus estimate; it does not
mean that every individual clip has an error rate inside that range.
