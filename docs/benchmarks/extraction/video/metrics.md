# Video extraction metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Scoring inputs and result status](#scoring-inputs-and-result-status)
- [Visual-content quality](#visual-content-quality-1)
- [Visual timestamp quality](#visual-timestamp-quality-1)
- [Reliability and failure behavior](#reliability-and-failure-behavior-1)
- [Visual operational performance](#visual-operational-performance-1)
- [Workload descriptor](#workload-descriptor-1)
- [Frozen-ASR diagnostics](#frozen-asr-diagnostics-1)
- [Worked candidate interpretation](#worked-candidate-interpretation)
- [Video confidence intervals](#video-confidence-intervals)

The video experiment compares keyframe configurations while holding the image
parser and ASR fixed. Visual quality measures useful on-screen text and its
capture time. Frozen-ASR diagnostics describe the shared audio input separately;
they never contribute to visual scores or rank keyframe policies.

The runner implements occurrence-aware duplication, first-attempt failure and
repeatability accounting, frozen-ASR diagnostics, and grouped MLflow runs.
See the [runbook](../../running.md#1-prepare-the-environment).

## Metric summary

### Visual-content quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Visual Content F1 | Primary | Does the configuration balance correct visible text with complete recovery? | Higher |
| Visual Content Precision | Secondary | How much extracted visible text is supported by the reference? | Higher |
| Visual Content Recall | Secondary | How much verified visible text was recovered? | Higher |

### Visual timestamp quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Mean Visual First-Detection Delay | Primary | How long after text appears is it first captured? | Lower |
| Timed Visual Occurrence Coverage | Primary | What proportion of verified appearances were captured while visible? | Higher |

### Reliability and failure behavior

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Duplicate Visual Text Rate | Diagnostic | How much supported output repeats identical text within the same visibility occurrence? | Lower |
| Repeatability Success Rate | Diagnostic | How often do two scheduled attempts deliver the same valid visual output? | Higher |
| Attempt Failure Rate | Diagnostic | How often does a measured visual request fail to deliver a valid output? | Lower |

### Visual operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Visual Real-Time Factor | Operational | How much complete visual-processing time is required relative to video duration? | Lower |
| p50 Warm Visual Latency | Operational | What is typical warm visual-processing time per video? | Lower |
| p95 Warm Visual Latency | Operational | What is slow-tail warm visual-processing time per video? | Lower |
| Cold Visual-Pipeline Load Time | Operational | How long does a fresh worker take to make the visual pipeline ready? | Lower |
| Peak Visual Process-Tree RAM | Operational | How much system memory does the visual worker and its children require? | Lower |
| Peak Visual Device VRAM | Operational | What peak total memory is occupied on the assigned GPU? | Lower |

### Workload descriptor

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Mean Selected Frames per Video | Descriptor | How many frames does the policy select, regardless of later parser success? | Descriptive |

### Frozen-ASR diagnostics

These values belong only to the shared frozen-ASR child. Their quality roles
are diagnostic in this benchmark, even when the standalone ASR benchmark uses
the same calculation as a primary metric.

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Transcript WER | Diagnostic | How wrong is the complete stitched word transcript? | Lower |
| Transcript CER | Diagnostic | How severe are character-level transcription errors? | Lower |
| Word Substitution Rate | Diagnostic | How much spoken content was replaced by different words? | Lower |
| Word Deletion Rate | Diagnostic | How much spoken content was omitted? | Lower |
| Word Insertion Rate | Diagnostic | How much unsupported word content was added? | Lower |
| Timestamp Boundary MAE | Diagnostic | How accurate are aligned spoken boundaries? | Lower |
| Timestamp Alignment Coverage | Diagnostic | How much reviewed spoken timing received a valid alignment? | Higher |
| Unexpected Empty Transcript Rate | Diagnostic | How often does a completed speech result unexpectedly contain no lexical text? | Lower |
| Nonspeech False-Transcription Rate | Diagnostic | How often does completed reviewed nonspeech input produce lexical text? | Lower |
| Attempt Failure Rate | Diagnostic | How often does a scheduled ASR window request fail? | Lower |
| Complete Audio-Pipeline Real-Time Factor | Operational | How much complete audio-processing time is required relative to original duration? | Lower |
| p50 Warm Audio-Processing Latency | Operational | What is typical complete audio-processing time per video? | Lower |
| p95 Warm Audio-Processing Latency | Operational | What is slow-tail complete audio-processing time per video? | Lower |
| Cold ASR Model-Load Time | Operational | How long does the fresh worker take to initialize the selected ASR? | Lower |
| Peak Audio Process-Tree RAM | Operational | How much system memory does the audio worker and its children require? | Lower |
| Peak Audio Device VRAM | Operational | What peak total GPU memory is occupied during the audio lifecycle? | Lower |

## Scoring inputs and result status

Visual quality and duplication use the designated first measured attempt only.
Development, validation, and locked schedule three measured attempts per video
after one warmup; attempts 2 and 3 supply timing, failure, and repeatability
evidence, not extra quality scores. Smoke schedules one measured attempt.

A failed first attempt has unavailable output-dependent metrics. Never invent
an empty prediction, assign quality penalties, or substitute a later success.
A valid completed empty output is scored normally. Selection counts, completed
cold loading, and sampled resource peaks remain available when their own
measurement completed, even if output subsequently fails; record their scope.

| Status | Value and treatment |
|---|---|
| `scored` | Defined numeric result, including an explicit completed-output empty convention. |
| `inapplicable` | No relevant reference task exists; null with reason. |
| `unavailable` | A required output or measurement could not be obtained; null with reason. |
| `incomplete` | Required execution or records are missing; no fabricated observation. |
| `invalid_reference` | Input annotations violate the contract; reject before inference. |

Keep every sample and attempt in artifacts, including nulls and reasons.
MLflow receives only numeric scalars. Every aggregate records scheduled,
reference-eligible, contributing, failed, unavailable, and independent-source
counts. Reference eligibility is identical across candidates; actual completed
output cohorts can differ and must remain visible beside quality.

Visual content, coverage, delay, duplication, and repeatability use video-macro
aggregation: calculate each defined video value, then average contributing
videos equally. F1 is calculated per video, not from averaged precision/recall.
RTF, latency percentiles, selected-frame counts, and resource peaks retain the
operational definitions below.

Data validation checks annotation presence, positive duration, unitization,
visibility intervals, reviewed reappearances, and source isolation. It does
not certify model outputs. See [data validation](../../data-validation.md).

## Visual-content quality

Under `normalized_lines_distinct_v1`, apply the shared prose projection to each
visible line and discard projected empty lines. Compare distinct normalized
predicted/reference line sets by exact equality. Repeated copies count once
for content, but remain separate detections for timing and duplication.
These units are lines, not model-tokenizer IDs.

| Edge case | Example | Value | Explanation |
|---|---|---|---|
| Both projected sets empty | Text-free video; valid empty output | P/R/F1 = 1/1/1 | Correct absence is established. |
| Empty reference, nonempty output | Text-free video; invented title | P/R/F1 = 0/0/0 | Unsupported output; empty-reference recall uses the explicit convention. |
| Nonempty reference, empty output | A title is required but nothing is extracted | P/R/F1 = 0/0/0 | Required content was missed. |
| First attempt fails | Parser crash or invalid complete output | null/unavailable | No valid prediction exists. |

Different raw strings that project to empty compare as empty; raw text is retained.
Each metric is calculated per video before video-macro averaging. The following
three definitions share the empty-output and failure rules above.

### Visual Content Precision

**Question:** How much extracted visible text is supported by the reference?

```text
Visual Content Precision = matched distinct lines / predicted distinct lines
```

**Example:** Eighty exact line matches among ninety predicted distinct lines give
`80 / 90 ≈ 0.89`; ten predicted lines are unsupported. An extra copy of an
already predicted line does not change this set-based score.

High precision means few unsupported lines, not that every required line was
captured or that the capture occurred while the line was visible.
**Range and direction:** `[0, 1]`; higher is better.

### Visual Content Recall

**Question:** How much verified visible text was recovered?

```text
Visual Content Recall = matched distinct lines / reference distinct lines
```

**Example:** Eighty matches among one hundred distinct reference lines give
`80 / 100 = 0.80`. Capturing a title once can satisfy content recall even if a
later reappearance is missed; occurrence coverage measures that separate task.

High recall means broad distinct-line recovery. Extra unsupported lines do not
lower recall, so inspect precision too.
**Range and direction:** `[0, 1]`; higher is better.

### Visual Content F1

**Question:** Does visible-text extraction balance support and completeness?

```text
Visual Content F1 = 2 × precision × recall / (precision + recall)
```

Use each video's precision and recall, with zero when both are zero; do not
calculate F1 from averaged precision/recall.

**Example:** Eighty matches, ninety predicted lines, and one hundred reference
lines give `2 × 80 / (90 + 100) = 160 / 190 ≈ 0.84`.
If a second video's F1 is `0.50`, macro F1 is `(160 / 190 + 0.50) / 2 ≈ 0.67`.

High F1 balances extra and missing text but does not establish timing, layout,
or successful recovery on failed first attempts.
**Range and direction:** `[0, 1]`; higher is better.

## Visual timestamp quality

Every reviewed occurrence has a stable ID, normalized nonempty text, and a
visibility interval `[start, end)`. Reappearance after disappearance is a new
occurrence even when its text is identical. Eligible detections must come from
frames inside that interval; timestamp tolerance is zero.

Occurrence matching uses token Content F1 of at least `0.5`. Tokens here are
whitespace-separated units of the shared prose projection. This approximate
eligibility test is separate from exact distinct-line content scoring.

Match reference occurrences and frame text detections one-to-one. Maximize
match count first, then minimize total detection delay; remaining ties use
stable occurrence IDs and chronological frame/unit order. A detection cannot
cover two occurrences. The same fixed assignment supplies both metrics below;
do not optimize a different assignment for each displayed score.

| Edge case | Example | Coverage / delay | Explanation |
|---|---|---|---|
| No reference appearances | Verified text-free video | null/inapplicable for both | No timed recovery task exists, although content absence can be correct. |
| Valid output covers none | Required title never captured | 0 / null-unavailable | Recovery failed; there is no observed delay. |
| First attempt fails | Visual worker crashes | null/unavailable for both | No valid detection output exists. |
| Required annotation missing | Visible text without its claimed interval | No scoring | Reject during data validation. |

### Mean Visual First-Detection Delay

**Question:** How soon after an appearance starts is it first captured?

Subtract appearance start from its assigned frame time. Average covered
occurrence delays within the video, then average defined video values equally.

```text
Video mean delay = sum of assigned in-interval detection delays / covered occurrences
```

**Example:** An appearance starts at `10 s` and its assigned frame is at `12 s`,
so delay is `2 s`. Eight covered appearances with `12 s` total delay give
video mean `12 / 8 = 1.5 s`.

Low delay describes captured appearances only. It can look good when difficult
appearances are missed, so read it with coverage.
**Range and direction:** Non-negative seconds; lower is better. Each delay is
less than its own `[start, end)` interval length. No covered occurrence makes
delay unavailable; no timing task is inapplicable. A failed first output is
unavailable. Misses receive no fabricated delay penalty or zero.

### Timed Visual Occurrence Coverage

**Question:** How much of the reviewed appearance task was captured while visible?

```text
Video occurrence coverage = matched reference occurrences / required occurrences
```

Calculate each video's ratio before averaging; do not pool corpus appearances.

**Example:** Eight captures among ten appearances give `8 / 10 = 0.80`.
A second video with one capture out of two gives macro coverage
`(0.80 + 0.50) / 2 = 0.65`, not the pooled ratio `9 / 12 = 0.75`.
A frame exactly at an appearance's end receives no credit because the interval
excludes its end; a genuine reappearance is a new required occurrence.

High coverage means required appearances were captured, not that every word or
boundary is exact. **Range and direction:** `[0, 1]`; higher is better. A valid
output with no captures of required appearances receives zero. No appearance
task is inapplicable; failed first output is unavailable, not zero coverage.

## Reliability and failure behavior

### Duplicate Visual Text Rate

**Question:** How much supported output repeats identical text within the same
continuous visibility occurrence?

Assign each detection to at most one reference occurrence using the same
in-interval and Content F1 ≥ `0.5` eligibility rules. For ambiguous assignments,
choose highest Content F1, then earliest occurrence start, then stable ID.
Unlike coverage, an occurrence may receive multiple detections for this diagnostic.

Process detections chronologically. Within each assigned occurrence, count
every exact normalized predicted-text copy after its first copy as a duplicate.
The denominator is all assigned detections. Average the defined per-video rates.
Do not use approximate Content F1 between predictions as the duplication test.

**Example:** For one continuous reference occurrence of `Chapter 1`:

| Frame | Extracted text | Assigned? | Duplicate? |
|---|---|---|---|
| 10 s | `Chapter` | Yes; Content F1 is 2/3 | No |
| 12 s | `Chapter 1` | Yes | No |
| 14 s | `Chapter 1` | Yes | Yes |

The rate is `1 / 3`, not `2 / 3`. A genuine later reappearance receives a new
occurrence ID, so its first copy is not a duplicate. Unsupported detections
affect content quality; they are outside this supported-duplication denominator.
Retain assigned/unassigned and duplicate counts.

| Edge case | Example | Value | Explanation |
|---|---|---|---|
| No assigned detections | Empty output, or only unsupported text | null/unavailable | The denominator is zero; this is not evidence of efficient recovery. |
| No reference occurrence task | Text-free reference video | null/inapplicable | There is no supported visibility occurrence to assign. |
| First attempt fails | Parser failure | null/unavailable | No valid output exists. |

**Range and direction:** `[0, 1]`; lower is better. One assigned detection, or
multiple assigned detections with no exact repeats, gives zero.

### Repeatability Success Rate

**Question:** How often do two scheduled attempts deliver the same valid visual
output?

Compare all three pairs of the three measured attempts for each video. A pair
earns one only when both attempts return valid identical canonical visual
outputs; otherwise it earns zero. Divide successful agreeing pairs by three,
then average video values. Compare selected frame times and canonical frame/text
outputs; exclude runtime IDs and timing/resource metadata. Normalization,
timestamp/box precision, ordering, and stable ID mapping are frozen and recorded.
Each attempt executes extraction rather than returning a cached prior output.

| Three outcomes | Repeatability | Attempt Failure Rate |
|---|---:|---:|
| A, A, A | 1 | 0 |
| A, A, B | 1/3 | 0 |
| A, B, C | 0 | 0 |
| A, A, failure | 1/3 | 1/3 |
| failure, A, A | 1/3 | 1/3 |
| A, failure, failure | 0 | 2/3 |
| failure, failure, failure | 0 | 1 |

**Example:** For `A, A, failure`, only pair `(1,2)` agrees successfully, so
repeatability is `1 / 3`. Video scores `1`, `1/3`, and `0` macro-average to
`(1 + 1/3 + 0) / 3 ≈ 0.444`.

A/B/C are distinct valid outputs. Three valid empty outputs agree perfectly;
their quality is scored independently. For `failure, A, A`, first-attempt
quality stays unavailable. Identical error messages do not agree.

**Range and direction:** `[0, 1]`; higher is better. Smoke has one attempt:
null/inapplicable, `repeatability_not_measured`. Missing required attempts make
the measurement incomplete, not an invented zero or a favorable surviving pair.

### Attempt Failure Rate

**Question:** How often does a measured visual request fail to deliver valid output?

For each video, divide recorded failed attempts by its scheduled measured
attempts, then macro-average video fractions. With three attempts per video,
this equals total recorded failures divided by total scheduled attempts.
A request covers selection through validated visual output. A crash, timeout,
selector violation, or invalid complete output is a failure. A completed valid
empty output is not. Recoverable block warnings affect quality without
automatically failing the whole request.

**Example:** One failure among three attempts gives `1/3`. Continue remaining
scheduled attempts after recoverable errors.

**Range and direction:** `[0, 1]`; lower is better. All recorded attempts
succeed gives zero; all execute and fail gives one. Missing/not-executed required
outcomes make the rate incomplete. Setup, telemetry, and offline evaluator errors
remain separate, not fabricated measured-request failures. This rate is evidence
for engineer review, not an automatic quality-based exclusion rule.

## Visual operational performance

### Visual Real-Time Factor

**Question:** How much complete visual-processing time is required relative to
video duration?

Divide the sum of successful measured visual-request times by the sum of those
same attempts' verified video durations. Include FFmpeg selection/extraction,
required frame preparation, parsing, conversion, output validation, and device
synchronization. Count duration again for each successful repetition.
Exclude frozen ASR, cold load, warmup, input verification, and offline scoring.

**Example:** A complete four-minute video processed in one minute gives
RTF `60 / 240 = 0.25`. A value above one simply means slower than playback.

**Range and direction:** Non-negative and unbounded; lower is better. No
complete timed success gives null/unavailable. Missing/non-finite required
timings are unavailable; non-positive source duration is invalid data. Failure
elapsed times remain in artifacts, not successful-speed aggregates.

### p50 and p95 Warm Visual Latency

For each video, take the median latency of its successful measured warm attempts.
Calculate p50/p95 across these video medians using shared linear interpolation.
The timer covers exactly the complete visual request listed under RTF.
No successful video observation gives null/unavailable. A video with failed
attempt 1 can still contribute later successful timings without supplying
quality. Report completion counts. Limited independent support keeps descriptive
point estimates, flags p95 limitations, and withholds unsupported CI bounds.

#### p50 Warm Visual Latency

**Question:** What is the typical successful complete visual-processing wait?

Take the median of the video-level successful-attempt medians.

**Example:** Video observations `10, 12, 14, 16, 18, 20 s` give p50
`(14 + 16) / 2 = 15 s`. An attempt sequence `12, 14, failure` contributes
median `13 s`, not the time taken to fail.

This summarizes a typical complete-video wait, not average frame speed.
**Range and direction:** Non-negative seconds/video; lower is better. The shared
completion and independent-source support rules above apply.

#### p95 Warm Visual Latency

**Question:** What is the slow-tail successful complete visual-processing wait?

Interpolate the 95th percentile of the same sorted video observations, at
zero-based position `(n − 1) × 0.95`.

**Example:** For `10, 12, 14, 16, 18, 20 s`, position `5 × 0.95 = 4.75` gives
`18 + 0.75 × (20 − 18) = 19.5 s`.

An interpolated p95 need not equal an observation or put exactly 95% of a small
sample below it. **Range and direction:** Non-negative seconds/video; lower is
better. Unsupported tail/CI estimates remain descriptive with null bounds;
repeated attempts do not add independent videos.

### Cold Visual-Pipeline Load Time

**Question:** How long does a fresh worker take to make the visual pipeline ready?

Time runtime construction, required model loading/device placement, explicit
pipeline initialization, and synchronization, ending before warmup. Docling
uses `initialize_pipeline(InputFormat.IMAGE)`; Paddle constructor initialization
receives the equivalent readiness boundary. Do not move lazy model loading into
an unmeasured gap. Operating-system disk caches are not forcibly cleared.

**Example:** Construction starts at `0.4 s` and readiness at `5.4 s`:
cold load is `5.0 s`.

**Range and direction:** Non-negative seconds; lower is better. Failed readiness
gives null; retain elapsed time to setup failure. Successful readiness remains
measured if a later inference fails. One observed load has no confidence interval.

### Peak Visual Process-Tree RAM

**Question:** How much host memory does the complete visual lifecycle require?

At each sample, sum resident RAM of the worker and its living children/
grandchildren, including media subprocesses; report the largest simultaneous
total. Do not sum independent process peaks.
Monitor from before loading through warmup and final measured inference.
Offline scoring and input validation are outside candidate monitoring.
Frozen-ASR models are unloaded before the fresh visual worker.

**Example:** Simultaneous worker/subprocess samples of `4,000 + 1,600 MiB` and
`4,500 + 800 MiB` give totals `5,600` and `5,300 MiB`, hence peak `5,600 MiB`.
Adding the separate process peaks would incorrectly give `6,100 MiB`.

This measures simultaneous host demand of the parser and its media subprocesses.
**Range and direction:** Non-negative MiB; lower is better at equal quality.
Missing RAM telemetry is unavailable. Retain real partial-window samples after
failure with their scope; one observed peak has no CI.

### Peak Visual Device VRAM

**Question:** What total GPU memory is occupied during visual extraction?

Take the largest raw NVML device-total sample from before initialization through
warmup and the final measured visual request. Do not use per-process memory or
subtract an idle baseline. Frozen ASR is unloaded before visual monitoring;
input validation and offline scoring are outside this window.

**Example:** Device samples `700, 2,900, 2,400 MiB` give peak `2,900 MiB`,
not the baseline-subtracted `2,200 MiB`. This is device occupancy, not model size.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
CPU VRAM is null/inapplicable; missing CUDA telemetry is null/unavailable.
Retain real samples after failure with partial/full lifecycle scope. One observed
peak has no CI. There is no artificial extraction VRAM cap; actual OOM, placement,
and required telemetry gates still apply.

## Workload descriptor

### Mean Selected Frames per Video

**Question:** How many frames did the strategy actually select?

Use each video's completed first-attempt selection count, regardless of later
parser acceptance or failure, then average videos with known completed counts.
Keep all per-attempt counts in timing records. Do not replace an unavailable
first selection with a later favorable one.

In plain words, sum known completed first-selection counts and divide by the
number of videos contributing those counts.

**Example:** If 360 frames are selected across 30 videos, the mean is
`360 / 30 = 12 frames/video`.
If selection yields 12 frames and parsing later fails, retain `12`, not the
number parsed successfully.

**Range and direction:** Non-negative frames/video; descriptive, not inherently
better when smaller. Failed/incomplete selection with unknown final count gives
null/unavailable. Zero selected frames violates mandatory frame zero: retain the
observed zero and selector failure, not a successful empty visual output.
No completed selection counts gives an unavailable aggregate.

## Frozen-ASR diagnostics

Decode each unique video once, transcribe each required 30-second/2-second-overlap
window once, shift timestamps, and stitch its transcript. Compare the complete
stitched result to that video's reviewed spoken reference. Do not score repeated
overlap words twice or copy results to visual children.

Use the [ASR recognition contract](../audio/metrics.md#recognition-quality-1):
align each valid completed video separately, then pool word/character edit
counts and contributing reference lengths. CER includes spaces left by the
shared projection. Substitution/deletion/insertion rates reuse pooled word
counts; raw integer counts stay in artifacts.

| Edge case | Example | Value | Explanation |
|---|---|---|---|
| Empty/empty projected pair | Silent video; empty transcript | WER/CER and S/D/I rates = 0 | No lexical edits. |
| Empty/nonempty pair | Silent video; `go now` | WER 2; CER 6; S/D/I rates 0/0/2 | Insertions only; CER includes the space. |
| Nonempty/empty pair | `go now`; valid empty transcript | WER/CER 1; S/D/I rates 0/1/0 | All required content is deleted. |
| Required window fails | Only part of a video transcribed | null/unavailable video recognition | Partial output is not a complete transcript. |
| No completed video pairs | All video results unavailable | null/unavailable aggregate | No observed comparison exists. |

For a contributing subset with zero reference words/characters, use total
insertions under the JiWER convention; component substitution/deletion rates
are zero and insertion rate is the word insertion count. WER/CER/insertion rate
can exceed one. Missing annotation is invalid data, not silence. Genuinely silent
videos are valid: the standalone speech-corpus positive-word quota does not apply.

These shared recognition conventions apply to each of the five metrics below.
Report planned and contributing videos, reference lengths, failed videos, and
independent-source counts. A failed required window makes the complete video
pair unavailable, not an empty transcript; later visual attempts cannot replace
it. Their CIs resample complete source videos and recalculate pooled counts.

### Transcript WER

**Question:** How wrong is the complete stitched ordered word transcript?

```text
Transcript WER = pooled word substitutions + deletions + insertions
                 divided by contributing reference words
```

Align each completed video's projected words separately before pooling.
**Example:** `40 + 20 + 10` edits across `1,000` reference words give
`70 / 1,000 = 0.07`.

This diagnoses the fixed ASR input; it cannot distinguish visual policies.
**Range and direction:** Non-negative, unbounded; lower is better. Insertions can
make WER exceed one; the shared empty-reference and unavailable rules above apply.

### Transcript CER

**Question:** How severe are character-level errors in the stitched transcript?

Divide pooled character substitutions, deletions, and insertions by contributing
projected reference characters, including remaining spaces.

**Example:** `50` character edits across `5,000` reference characters give
`50 / 5,000 = 0.01`. A single wrong character can count as one complete word
error in WER, so the two metrics expose different error scales.

**Range and direction:** Non-negative, unbounded; lower is better. Empty
reference uses character insertion count; failed/partial video pairs are
unavailable, not deletions. Apply the shared recognition support/CI rules.

### Word Substitution Rate

**Question:** How much required spoken content became different words?

Use the same word alignment as WER: pooled substitutions divided by contributing
reference words. **Example:** `40 / 1,000 = 0.04` means four substitutions per
hundred reference words, not that four percent of videos failed.

This isolates word confusion from omission and invention.
**Range and direction:** `[0, 1]`; lower is better. Zero reference words gives
zero substitutions under the explicit convention. Failed video pairs remain
unavailable; use recognition support and CIs, not window-level resampling.

### Word Deletion Rate

**Question:** How much required speech was omitted from the complete transcript?

Divide pooled word deletions by the same contributing reference-word count.
**Example:** `20 / 1,000 = 0.02` exposes two missing words per hundred reference
words, even when the retained words are accurate.

**Range and direction:** `[0, 1]`; lower is better. Valid empty output against
nonempty reference gives one; empty/empty gives zero. A failed window does not
create invented deletions for its video. Shared recognition support/CIs apply.

### Word Insertion Rate

**Question:** How much unsupported speech was added?

Divide pooled word insertions by contributing reference words.
**Example:** `10 / 1,000 = 0.01`. These counts reuse WER's alignment, so rates
`0.04 + 0.02 + 0.01 = 0.07` reconcile with WER, not with CER.

**Range and direction:** Non-negative, unbounded; lower is better. With zero
reference words, use insertion count: empty reference versus `go now` gives two.
No completed pairs is unavailable, not zero. Shared recognition support/CIs apply.

### Spoken timestamp matching

Reuse [ASR span alignment](../audio/metrics.md#timestamp-quality-1) on the stitched
video timeline: Content F1 ≥ `0.5`, ordered non-overlapping predicted spans,
no reuse, and the matched span's minimum start/maximum end. Pool real boundary
errors for MAE and matched/required segment counts from valid completed eligible
videos for coverage.

No reviewed spoken timing task gives null/inapplicable; visual appearance
timestamps are not spoken timestamps. A claimed but missing spoken annotation
fails data validation. Valid output with required timings but no matches gives
coverage zero and MAE null/unavailable. Failed video transcription makes both
unavailable. Retain planned/contributing reference and matched-boundary counts.

### Timestamp Boundary MAE

**Question:** How far are aligned spoken starts/ends from reviewed boundaries?

Sum absolute start and end errors and divide by the number of matched boundaries,
pooling real errors across completed videos rather than averaging video MAEs.

**Example:** Eighteen matched segments have 36 boundaries. Total absolute error
`7.2 s` gives `7.2 / 36 = 0.20 s` MAE.

Small MAE means precise matched boundaries, not complete alignment.
**Range and direction:** Non-negative seconds; lower is better. No matches is
unavailable; no reviewed timing task is inapplicable. Failed video transcription
is unavailable. CI draws retain source videos and use actual matched errors.

### Timestamp Alignment Coverage

**Question:** How much of the reviewed spoken timing task received an alignment?

Divide pooled matched reference segments by pooled required segments from valid
completed eligible videos. **Example:** `18 / 20 = 0.90`; adding a valid video
with ten required segments and no matches gives `18 / 30 = 0.60`.

Coverage exposes missing alignment that a small MAE could hide.
**Range and direction:** `[0, 1]`; higher is better. Completed output with no
matches receives zero; no task is inapplicable. Failed transcripts are
unavailable and retain their planned, not scored, denominators. Resample full
source videos, including those failures, for conditional coverage CIs.

### Unexpected Empty Transcript Rate

**Question:** How often does completed video speech unexpectedly vanish?

Divide valid empty complete video transcripts by valid completed video
transcripts requiring nonempty projected spoken text.

**Example:** Of 100 eligible videos, 10 fail complete transcription and four of
the 90 completed transcripts are empty: `4 / 90 ≈ 0.044`.

This measures empty successful execution, not failed windows. Read it with ASR
Attempt Failure Rate and complete-video failure counts.
**Range and direction:** `[0, 1]`; lower is better. No completed eligible videos
is unavailable; no lexical reference task is inapplicable. Zero events with a
positive completed denominator gives zero. CI draws recalculate that denominator.

### Nonspeech False-Transcription Rate

**Question:** How often does reviewed nonspeech produce invented lexical text?

Divide valid completed reviewed nonspeech units producing lexical text by all
valid completed reviewed nonspeech units.

The protocol must freeze one nonspeech unit type before evaluation: whole
nonspeech videos or explicitly reviewed actual ASR windows, including overlap.
Do not mix types, infer nonspeech from missing text, add separate inference
repetitions, or impose the standalone control-category allocation on video.
Require the reviewed units declared for the video corpus and retain their labels.

**Example:** Three invented outputs among twenty completed reviewed units give
`3 / 20 = 0.15`. Two other failed scheduled units remain failures, not events;
the completed-output denominator remains twenty.

**Range and direction:** `[0, 1]`; lower is better. No events
with a positive completed denominator gives zero; all give one. No valid
completed eligible units gives null/unavailable. A legitimate slice without
the task is null/inapplicable; absent required reviewed inputs fail preparation.
For CIs, windows remain dependent parts of their source video, not independent
control samples. The unit type/allocation are currently null pending data review.

### Audio Attempt Failure Rate

**Question:** How often does a scheduled ASR window request fail?

Divide recorded failed window requests by all scheduled measured window
requests, keeping per-video completion separately. Windows are request units,
not independent bootstrap units. One failed required window makes the complete
video transcript unavailable. Continue recoverable scheduled work and retain
partial outputs and elapsed times as diagnostics only.

**Example:** Two failures among 40 fully recorded requests give `2 / 40 = 0.05`.

**Range and direction:** `[0, 1]`; lower is better. All recorded requests
succeed gives zero, all execute and fail gives one. Missing/not-executed outcomes
give incomplete measurement, not an invented denominator. Loading, warmup, or
decoding failure before window inference remains a setup/preprocessing outcome,
with affected video status, not fabricated failed ASR requests.
There is no audio repeatability score: each window is transcribed once.

### Shared audio operational measurements

One warmup follows cold ASR readiness. Time each complete video from audio
decoding through window preparation, native features, transcription, required
alignment, timestamp shifting, stitching, output conversion/validation, and
synchronization. Exclude cold loading, warmup, input verification, and offline
scoring. Record window timings, but p50/p95 use one completed warm
audio-processing latency per video, with shared linear interpolation.

Successful video timings survive failures of other videos. Partial outputs and
elapsed-to-failure diagnostics are retained but do not become complete-video
speed observations. All measurements live on the frozen-ASR child, not on every
visual child; component percentiles and resource peaks cannot be added to infer
an integrated audio/visual measurement.

### Complete Audio-Pipeline Real-Time Factor

**Question:** How much complete audio-processing time is needed per unit duration?

Divide summed successful complete-video processing time by those videos'
original audio durations. Count original duration once, not each overlapping
window duration. **Example:** `60 s / 240 s = 0.25` means four times playback
speed for that completed video, including decoding and stitching.

**Range and direction:** Non-negative, unbounded; lower is better. Above one
means slower than playback. No complete timed success is unavailable; nonpositive
duration is invalid reference data. CIs recalculate the time/duration ratio from
full source units, not means of per-window RTFs.

### p50 Warm Audio-Processing Latency

**Question:** What is the typical complete audio-processing wait per video?

Take the median of completed videos' warm audio latencies; there is one measured
preparation per video, not three ASR repetitions.
**Example:** `10, 12, 14, 16, 18, 20 s` gives `(14 + 16) / 2 = 15 s`.

This describes a typical full preparation, not window latency.
**Range and direction:** Non-negative seconds/video; lower is better. No complete
timings is unavailable. Independent-source support governs its conditional CI.

### p95 Warm Audio-Processing Latency

**Question:** What is the slow-tail complete audio-processing wait per video?

Use linear interpolation across completed video latencies at position
`(n − 1) × 0.95`. **Example:** For `10, 12, 14, 16, 18, 20 s`, position `4.75`
gives `18 + 0.75 × (20 − 18) = 19.5 s`.

This is a percentile estimate, not a maximum or a guarantee for unseen videos.
**Range and direction:** Non-negative seconds/video; lower is better. No complete
timings is unavailable; limited tail support keeps a descriptive point with null
CI bounds and reason. Overlapping windows do not increase independent support.

### Cold ASR Model-Load Time

**Question:** How long does initialization take before audio warmup?

Subtract model/profile construction start from synchronized full readiness,
including device placement and required components, ending before warmup.
**Example:** Start `0.4 s`, readiness `5.4 s`: cold load is `5.0 s`.

This is a one-time initialization cost, not per-video preparation latency.
**Range and direction:** Non-negative seconds; lower is better. Failed readiness
is unavailable with elapsed-to-failure retained. Completed load survives later
transcription failure. One load observation has no CI.

### Peak Audio Process-Tree RAM

**Question:** How much simultaneous host memory does audio preparation require?

At each sample sum resident RAM of the worker and its living descendants,
including decoding subprocesses, then take the maximum. Monitor loading,
warmup, and measured audio work, excluding data validation, scoring, and visuals.
**Example:** Simultaneous totals `2,000 + 1,200 = 3,200 MiB` and
`2,400 + 500 = 2,900 MiB` give peak `3,200 MiB`, not `3,600 MiB` from separate peaks.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
Missing instrumentation is unavailable. Retain partial-window observed peaks
with scope after failure; one observed peak has no CI.

### Peak Audio Device VRAM

**Question:** What total GPU memory is occupied during audio preparation?

Take the maximum raw NVML device-total sample over loading, warmup, and measured
audio preparation, ending before scoring and visual execution.
**Example:** Device samples `700, 2,100, 1,900 MiB` give `2,100 MiB`; idle use
is not subtracted. This measures occupancy, not isolated model-weight bytes.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
CPU execution is null/inapplicable; missing CUDA telemetry is null/unavailable.
Real partial-window peaks survive failure with scope. No artificial VRAM cap
applies; OOM/placement/telemetry gates still apply. One observed peak has no CI.

## Worked candidate interpretation

Suppose a completed development child with 18 videos reports these rounded
values (quality/repeatability use video-macro aggregation):

```text
Visual Content Precision          = 0.92
Visual Content Recall             = 0.76
Visual Content F1                 = 0.83
Timed Visual Occurrence Coverage  = 0.80
Mean Visual First-Detection Delay = 1.5 seconds
Duplicate Visual Text Rate        = 0.18
Repeatability Success Rate        = 0.89
Attempt Failure Rate              = 0.04
Mean Selected Frames per Video    = 14
Visual Real-Time Factor           = 0.35
```

On average across contributing videos, content precision is high but recovery
is incomplete. Coverage is 80% on average across defined videos, not necessarily
80% of all corpus occurrences. Covered appearances are found quickly.
Duplication is 18% on average among assigned detections within their occurrences;
genuine reappearances are not penalized. Repeatability and failures describe
delivery separately from first-output correctness. Frame count and RTF show cost.
Inspect support counts before comparing conditional scores. Frozen-ASR values
appear on its own child, not in this visual result.

## Video confidence intervals

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---|---|
| Visual content, coverage, defined delay/duplication | When defined and sufficiently supported | Recalculate video-macro results from complete independent source groups. |
| Visual repeatability and Attempt Failure Rate | When complete and sufficiently supported | Preserve every scheduled attempt inside its video. |
| Visual RTF and Mean Selected Frames per Video | When defined and sufficiently supported | Recalculate their own ratio/mean from saved measurements. |
| Frozen-ASR WER/CER/components, spoken timestamps, event/failure rates, and audio RTF | When defined and sufficiently supported | Recalculate their pooled/conditional statistics; keep windows inside source videos. |
| Visual and audio p50/p95 latency | Conditional | Enough representative independent videos must support the percentile and interval. |
| Smoke values | No authoritative interval | Wiring evidence only. |
| One cold-load observation or observed RAM/VRAM peak | No | Report the observation without invented bounds. |

### Calculation

After execution, use 10,000 bootstrap resamples with seed 42:

1. Resample complete independent source groups with replacement. Related excerpts
   travel together with all their reference tasks, failures, windows, and attempts.
2. Recalculate each metric using its own eligibility and aggregation. Do not
   prefilter failed/unmatched videos before drawing.
3. Use the 2.5th and 97.5th percentiles of defined draws as the 95% bounds.
4. Retain contributing/independent counts, defined/undefined draw counts,
   support limitations, and degenerate-interval flags.

A valid output with required appearances but none recovered supplies coverage
zero and undefined delay. No timed task, or no completed eligible outputs in
a draw, does not supply an invented zero coverage. Apply the same distinction
to spoken timing and event rates.

Support requirements are frozen after data review, before authoritative
evaluation; [pending-data-review](../../pending-data-review.md#video-extraction)
records unresolved data-dependent support and annotation choices. Insufficient
support keeps point estimates and null bounds with reasons. Equal calculated
bounds are flagged, not treated as certainty about future videos.

### Interpretation

Visual Content F1 `0.84` with a 95% CI `[0.79, 0.88]` reports the observed
aggregate and estimated source-sample uncertainty, not the range of individual
video scores. Bootstrap performs arithmetic on saved outputs, not extra inference.

Eighteen measured videos remain eighteen observations after 10,000 resamples.
p50/p95 are measured latency percentiles; their confidence bounds describe
uncertainty in those estimates. A p95 latency and a 95% confidence interval are
different quantities. More resamples stabilize the calculation but do not add
independent videos or establish reliable tail behavior.
