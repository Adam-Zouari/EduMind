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

These 17 metrics define the approved ASR evaluation contract. Runner alignment
with the revised failure, empty-output, and repeatability rules remains pending;
documentation alone does not implement them. Technical-Term
Accuracy is excluded because EduMind has no fixed subject vocabulary;
diarization metrics remain out of scope until speaker identification becomes a
product requirement.

## Scoring inputs and result status

Quality uses only the designated first measured attempt, after the warmup. Its
index is fixed before execution. A failed first attempt is never replaced by a
successful later one: doing so would select a favorable output rather than
evaluate the same request for every candidate. Later attempts supply timing,
repeatability, and failure evidence, not additional quality scores.

Corpus WER/CER and the three word-error rates pool counts from valid completed
first speech outputs. A recorded failed first attempt has unavailable edit
diagnostics, not an invented empty transcript or arbitrary maximum error.
Timestamp metrics are also unavailable after a failed first attempt. Retain its
known reference denominators as planned support, not scored recovery failures.
Always report scheduled and contributing clips/source groups, failed first
attempts, and planned versus contributing reference words/characters. The
conditional recognition result cannot stand in for workload-wide reliability.

| Status | Value and treatment |
|---|---|
| `scored` | Numeric result, including an explicit completed-output empty-denominator convention. |
| `inapplicable` | No reference task exists; null with its eligibility reason. |
| `unavailable` | A required value cannot be measured, such as boundary MAE with no matches or WER after a failed first attempt; null with reason and support counts. |
| `incomplete` | Scheduled execution or required records are missing; no fabricated observation. |
| `invalid_reference` | Inputs violate the data contract; reject before candidate execution. |

Keep every row and status in artifacts. Nulls are excluded only from the
undefined metric's numeric calculation, not from failure records or other
defined metrics. No valid first outputs makes recognition aggregates null;
no matched boundaries makes MAE null. Neither produces a zero-error claim.
MLflow omits null scalar keys; JSON/Parquet retain the fields and reasons.

An empty transcript with an empty timestamp sequence is a valid completed
output. Nonempty lexical text without required timestamps, empty text with
lexical timestamp segments, or invalid required boundaries/schema are failed
attempts. Continue remaining attempts after recoverable failures; missing
records after a fatal setup/process failure make the execution incomplete.
Reference eligibility is fixed before inference and identical for all candidates.
See [data validation](../../data-validation.md) for preparation checks, not
prediction scoring.

## Recognition quality

### Corpus Word Error Rate

**Question:** How wrong is the complete ordered word transcript?

Corpus WER uses the fixed symmetric prose projection in the shared conventions:
NFC, case-folding, punctuation replacement with spaces, and whitespace collapse.
Words are whitespace-separated lexical units, not model-tokenizer subwords.
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

**Range and direction:** WER is non-negative and lower is better. It can exceed
`1` when insertions outnumber reference words. The full reviewed speech split
must have reference words, but a legitimate individual projection or contributing
subset may be empty. The empty-reference conventions below apply to those pairs
and subsets, not to missing annotations or failed attempts.

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

**Range and direction:** CER is non-negative and lower is better. It can exceed
`1` when insertions outnumber reference characters.

### Empty projected strings

These are completed-output rules from [JiWER's empty-reference convention](https://jitsi.github.io/jiwer/#a-note-on-empty-references).
They apply after the shared projection; different raw punctuation can project
to the same empty string without constituting missing data.

| Projected reference | Projected prediction | WER | CER | Explanation |
|---|---|---:|---:|---|
| Empty | Empty | 0 | 0 | No required lexical content and no invented lexical output. |
| Empty | `go now` | 2 | 6 | Every predicted word/character is an insertion; CER includes the space. |
| `go now` | Empty | 1 | 1 | All reference words/characters are deleted. |
| Nonempty | Nonempty | Normal edit ratio | Normal edit ratio | Insertions can make either value exceed one. |
| Any valid reference | Failed first attempt | null | null | There is no valid transcript pair to align; retain the failure. |

For a pooled contributing subset with zero reference words or characters, use
the corresponding total insertion count (zero when predictions are also empty).
Do not turn each empty-reference pair into a rate before pooling a corpus that
has reference content. No contributing pairs is unavailable, not an empty/empty
comparison. Reject an entirely empty authoritative speech split in data
preparation; bootstrap draws are not new manifests subject to that rejection.

## WER diagnostics

The three diagnostic rates reuse the exact word alignment and pooled reference
word count used by Corpus WER. Together they add up to WER, but each answers a
different failure question.

For a positive pooled reference-word count, divide each pooled edit count by
that same count. Raw integer substitutions, deletions, insertions, and reference
lengths remain in `samples.parquet`; the three reported metrics are rates.
For zero reference words, the benchmark extends the JiWER convention explicitly:
substitution and deletion rates are zero, and insertion rate is the insertion
count. This keeps the three components summing to WER, including empty-reference
cases. JiWER supplies the edit counts and empty WER rule; the component-rate
extension is this benchmark's reporting convention.

| Reference / prediction | Substitution Rate | Deletion Rate | Insertion Rate | Raw S / D / I counts |
|---|---:|---:|---:|---|
| Empty / empty | 0 | 0 | 0 | 0 / 0 / 0 |
| Empty / `go now` | 0 | 0 | 2 | 0 / 0 / 2 |
| `go now` / empty | 0 | 1 | 0 | 0 / 2 / 0 |
| Failed first attempt | null | null | null | Unavailable, not fabricated deletions |

### Word Substitution Rate

**Question:** How often is a spoken reference word replaced by a different
predicted word?

```text
Word Substitution Rate =
substituted reference words
---------------------------
all reference words
```

**Example:** Forty substitutions among 1,000 reference words produce `0.04`.

This distinguishes word confusion from omitted or invented speech.

**Range and direction:** Non-negative and lower is better. It cannot exceed `1`
because each reference word can be substituted at most once.

### Word Deletion Rate

**Question:** How much spoken reference content is missing from the transcript?

```text
Word Deletion Rate =
deleted reference words
-----------------------
all reference words
```

**Example:** Twenty deletions among 1,000 reference words produce `0.02`.

This exposes omissions that are especially harmful when lectures contain
definitions, negations, or instructions.

**Range and direction:** Lies in `[0, 1]`; lower is better.

### Word Insertion Rate

**Question:** How much unsupported word content did the model add?

```text
Word Insertion Rate =
inserted predicted words
------------------------
all reference words
```

**Example:** Ten insertions against 1,000 reference words produce `0.01`.

This exposes invented speech that is not present in the recording.

**Range and direction:** Non-negative and lower is better. It can exceed `1`
when the model inserts more words than the entire reference contains.

## Timestamp quality

### Timestamp Boundary MAE and Timestamp Alignment Coverage

**Question:** How accurate are the predicted start/end times, and how much of
the timed reference could actually be aligned?

For each reference segment, the evaluator enumerates contiguous spans of
predicted timestamp units and keeps spans whose normalized token Content F1 is
at least `0.5`. Dynamic programming selects ordered, non-overlapping one-to-one
matches by maximum total similarity, then match count, then earliest spans.
No predicted unit can be reused. This lets a segment-level reference match
several word-level predictions while preventing one broad prediction from
covering multiple references. The span envelope supplies its minimum start and
maximum end. Timestamp Boundary MAE is then calculated as follows:

1. Find the absolute start-time error for every aligned segment.
2. Find the absolute end-time error for every aligned segment.
3. Average all start and end errors.

```text
Timestamp Alignment Coverage =
reference segments successfully aligned
---------------------------------------
reference segments with timestamps
```

**Example:** If 18 of 20 reference segments align, coverage is `18 / 20 =
0.90`. If their 36 start/end boundaries have 7.2 seconds of total absolute
error, Timestamp Boundary MAE is `7.2 / 36 = 0.20 seconds`.

Boundary MAE alone can look excellent when only easy segments align. Coverage
shows how much of the timed reference actually contributed. The two metrics are
therefore interpreted together: low MAE and high coverage. When no segments
align, MAE is undefined rather than fabricated as zero, while coverage is zero.

Pool actual matched boundary-error totals for MAE. Pool matched and required
segment counts from valid completed first outputs for coverage, including zero
matches from valid empty outputs. A failed first attempt contributes neither a
fabricated boundary error nor a scored zero-coverage observation. Keep its known
reference count in planned support and report contributing, failed, and unavailable
clips alongside both metrics. Coverage diagnoses missed alignment among completed
outputs; Attempt Failure Rate diagnoses execution failures.

| Case | Boundary MAE | Alignment Coverage | Explanation |
|---|---|---|---|
| Two required segments, one aligns | Mean of its two boundary errors | 0.5 | Accuracy of the match and recovery of the task are separate. |
| Required segments, no matches or valid empty output | null, unavailable | 0 | There is no measured boundary error, but required segments were not recovered. |
| Required segments, first attempt fails | null, unavailable | null, unavailable | No valid prediction exists; retain the failure and planned reference count. |
| Legitimately no required timed segments | null, inapplicable | null, inapplicable | There is no spoken timing task, unlike an empty prediction for a real task. |
| Claimed required timing annotation missing | No scoring | No scoring | Reject invalid data before execution. |

**Range and direction:** Boundary MAE is a non-negative number of seconds and
lower is better. Alignment Coverage lies in `[0, 1]` and higher is better.

## Reliability and failure behavior

### Unexpected Empty Transcript Rate

**Question:** Among completed first speech attempts that require lexical text,
how often does transcription return empty?

Divide valid empty first outputs by valid completed first outputs for speech
clips with nonempty projected references. A completed output need not be correct:
empty text plus empty timestamps is valid execution and counts as an event.
Crashes and invalid required output schemas are failed attempts, not observed
empty transcripts. Read this conditional rate with Attempt Failure Rate.

**Example:** Of 100 eligible first speech attempts, 10 fail and 90 return valid
outputs; four of those are empty. The rate is `4 / 90`, approximately `0.044`.
The 10 failures remain in failure and support artifacts, not hidden as empties.

**Range and direction:** `[0, 1]`; lower is better. No empty completed outputs
gives `0`; all eligible completed outputs empty gives `1`. With no successful
eligible first outputs, retain null/unavailable and
`no_successful_first_outputs`. A diagnostic slice with no expected lexical task
is inapplicable; a full authoritative speech split with no lexical references
fails preparation. Missing scheduled records make the execution incomplete.

This denominator intentionally differs from document Unexpected Empty Output
Rate, whose denominator is all scheduled documents. They answer different
questions; neither rate silently changes to the other's denominator.

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

**Range and direction:** `[0, 1]`; lower is better. No completed control invents
text gives `0`; all do gives `1`. No valid completed first control outputs gives
null/unavailable with `no_successful_first_outputs`, not a perfect zero.
Authoritative inputs require reviewed controls in all four categories: their
absence is invalid data. A deliberately control-free diagnostic slice is
inapplicable; missing scheduled records make execution incomplete.

### Transcript Repeatability Success Rate

**Question:** How often does the same model profile return the same transcript
in two scheduled attempts on the same clip?

For each speech clip, compare all three pairs of its three scheduled measured
attempts. A pair receives one only when both attempts deliver valid outputs with
identical projected transcripts; otherwise it receives zero. Divide by three,
then average the clip scores. Errors do not agree merely because their messages
are identical. This includes failures rather than rewarding a candidate for
the one surviving output.

| Three outcomes | Successful agreeing pairs / scheduled pairs | Clip score |
|---|---|---:|
| A, A, A | 3 / 3 | 1 |
| A, A, B | 1 / 3 | 0.333… |
| A, B, C | 0 / 3 | 0 |
| A, A, failure | 1 / 3 | 0.333… |
| A, B, failure | 0 / 3 | 0 |
| A, failure, failure | 0 / 3 | 0 |
| failure, failure, failure | 0 / 3 | 0 |

A/B/C stand for distinct valid projected transcripts. Three valid empty
transcripts also score `1`; their recognition quality is assessed independently.

**Example:** Clip scores `1`, `1/3`, and `0` average to approximately `0.444`.
This diagnoses repeatable delivery, not correctness: identical wrong answers
can repeat perfectly. It uses the already collected measured attempts, with
the same seed/settings, and adds no inference or quality rescoring.

**Range and direction:** `[0, 1]`; higher is better. Complete recorded failures
give zero, not null. Smoke has one measured attempt, so the rate is null with
status `inapplicable` and reason `repeatability_not_measured`. Unexecuted attempts
or missing required records make execution
incomplete, not a fabricated three-attempt score.

### Attempt Failure Rate

**Question:** How often does a measured request fail to deliver a valid output?

Divide recorded failed measured attempts by all scheduled measured attempts,
including speech and nonspeech controls. Authoritative speech clips have three
attempts; controls have one. Keep separate speech/control breakdowns and the
actual attempt counts in artifacts. Later speech successes do not repair the
first attempt's quality result.

**Example:** Two speech clips schedule six attempts and two controls schedule
two. If two requests fail, the total rate is `2 / 8 = 0.25`.

**Range and direction:** `[0, 1]`; lower is better. All succeed gives `0`; all
scheduled attempts execute and fail gives `1`. Loading, warmup, data-validation,
and measurement-infrastructure failures have separate setup/run statuses, not
invented measured attempts. No execution or missing scheduled outcome records
makes the rate unavailable/incomplete; keep any observed failure counts without
publishing a falsely complete rate. This is measured reliability, not an
automatic candidate-selection rule.

## Operational performance

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

An RTF below `1` means processing is faster than audio playback.

The request timer covers candidate input reading/preprocessing, native features,
transcription, required timestamp/alignment work, output conversion/validation,
and device synchronization. Dataset acquisition, reference review, and offline
metric scoring are not transcription requests.

Count each completed repetition's audio duration once: three successful
30-second attempts contribute 90 seconds, not 30. Valid empty outputs count as
completed execution. Failed-attempt elapsed times remain artifacts but do not
describe completed transcription speed. Controls, cold loading, warmup, data
validation, and offline scoring are outside this RTF.

**Range and direction:** Non-negative, unbounded; lower is better. RTF above
one simply means slower than playback. No completed timed speech attempt gives
null/unavailable. Zero/negative source duration is invalid data; missing or
non-finite required timing is unavailable and blocks a complete timing report.

### p50 and p95 Warm Clip Latency

**Question:** What are the typical and slow-tail steady-state transcription
times per clip?

For each speech clip, take the median latency of its successful measured warm
attempts, then report p50/p95 across those clip medians. Report contributing and
failed-attempt counts; a clip with no completed timed attempt contributes no
completed latency. Retain failed elapsed times separately. No valid clip
latencies gives null/unavailable, not zero. Missing timing records are incomplete.

**Example:** A p95 of `5.2 seconds` estimates the slow 95th-percentile warm
request, rather than the typical clip. An interpolated percentile does not
require exactly 95% of a small observed sample to fall below that value.

**Range and direction:** Non-negative seconds per clip; lower is better.

### Cold Model-Load Time

**Question:** How long does a fresh worker need to make the ASR model ready?

Measure from the start of model construction until the complete ASR profile is
ready, before warmups or transcription. Any component needed before the first
request belongs to cold load; work performed later belongs to the measured
complete-pipeline latency and cannot be hidden from both measurements.

A loading failure has no completed cold-load observation: retain null and the
elapsed time to setup failure separately. An initialized model with later
inference failures still has a measured cold-load time.

**Range and direction:** Non-negative seconds; lower is better.

### Peak Process-Tree RAM and Peak Device VRAM

**Question:** What is the worker's peak host-memory use and the assigned GPU's
peak total memory use during complete candidate execution?

Peak Process-Tree RAM is the largest sampled resident-memory total for the
worker, children, and grandchildren alive at each sample, then the maximum of
those simultaneous totals. Do not add each process's independent peak. Peak
Device VRAM follows the shared raw NVML
device-total contract, including loading and first-use allocations.
Every result records the explicit CPU or GPU
profile; silent device fallback is invalid.

**Example:** Process-tree samples peaking at `3,200 MiB` and GPU samples peaking
at `2,100 MiB` produce those two reported peaks.

**Range and direction:** Non-negative MiB; lower is better at equal quality. A
CPU-only profile records VRAM as null/inapplicable; missing CUDA measurement is
null/unavailable, never zero. Retained samples after a failure may establish an observed
partial-window peak, explicitly labelled with its scope; they cannot establish
a missing completed-window measurement. Candidate monitoring excludes data
preparation/validation and includes loading, warmup, and measured inference.

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

Development, validation, and locked runs use 10,000 bootstrap resamples with seed 42:

1. Use the independent source IDs frozen in the manifest. A clip is a unit only
   when it is independent; related clips from one declared source travel together.
2. Resample those complete units with replacement; resample the nonspeech
   control units separately for their reliability rate and their contribution
   to the combined Attempt Failure Rate. Keep the speech/control allocation.
3. Retain every unit's first-attempt status, reference denominators, and complete
   attempt records. Recalculate each aggregate from those sampled units using
   its own eligibility and completed-output rules. Do not prefilter failed or
   unmatched clips before drawing, or treat repetitions as independent clips.
4. Use the 2.5th and 97.5th percentiles as the 95% bounds.

A draw with valid completed timed outputs but no matches contributes zero
Alignment Coverage; its Boundary MAE is undefined. A draw with no valid completed
timed outputs has unavailable coverage as well as MAE. A draw with no valid first
outputs cannot contribute a recognition or corresponding event-rate value,
but still contributes defined failure/repeatability evidence. Legitimate
zero-reference contributing subsets use the empty-reference conventions;
bootstrap draws do not rerun the corpus-validation quota.

Record defined and undefined draw counts separately for every conditional
metric. An undefined value is excluded only from that metric's interval, not
from retained records or other defined calculations. Freeze minimum independent
support after data review and before evaluation; insufficient support retains
the point estimate with null bounds and a reason. Degenerate calculated bounds
are flagged, not interpreted as proof of certainty.

If valid completed first outputs contain a timed reference task but no segment
aligns, Timestamp Boundary MAE is null and Alignment Coverage is `0`. If no valid
completed first timed output exists, both values are unavailable. Complete execution with those outcomes is not a
missing-record failure. This reports the absence of a measurable boundary without
inventing either a perfect or infinitely bad time error. The null is preserved
in `candidate.json` and `summary.json`; its MLflow scalar key is absent because
MLflow scalar metrics cannot represent null.

### Interpretation

Corpus WER `0.08` with a 95% CI of `[0.07, 0.10]` means the observed aggregate
error rate on valid completed first outputs is 8%, while independent-source
resampling estimates a plausible range of 7% to
10%. A narrower interval indicates a more precise corpus estimate; it does not
mean that every individual clip has an error rate inside that range.
