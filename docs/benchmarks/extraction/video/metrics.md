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

### Visual Content Precision, Recall, and F1

**Question:** How much recovered visible text is supported, how much required
text was recovered, and does the configuration balance both?

Under `normalized_lines_distinct_v1`, apply the shared prose projection to each
visible line and discard projected empty lines. Compare distinct normalized
predicted/reference line sets by exact equality. Repeated copies count once
for content, but remain separate detections for timing and duplication.
These units are lines, not model-tokenizer IDs.

Precision is matched distinct lines divided by predicted distinct lines.
Recall is matched distinct lines divided by reference distinct lines.
F1 is their harmonic mean; zero precision and recall give zero F1.

**Example:** One video has 100 reference lines and 90 predicted lines, with 80
matches. Precision is `80 / 90 = 0.89`, recall is `80 / 100 = 0.80`, and F1
is approximately `0.84`. Average each metric's video values separately.

| Edge case | Example | Value | Explanation |
|---|---|---|---|
| Both projected sets empty | Text-free video; valid empty output | P/R/F1 = 1/1/1 | Correct absence is established. |
| Empty reference, nonempty output | Text-free video; invented title | P/R/F1 = 0/0/0 | Unsupported output; empty-reference recall uses the explicit convention. |
| Nonempty reference, empty output | A title is required but nothing is extracted | P/R/F1 = 0/0/0 | Required content was missed. |
| First attempt fails | Parser crash or invalid complete output | null/unavailable | No valid prediction exists. |

**Range and direction:** All three values lie in `[0, 1]`; higher is better.
Different raw strings that project to empty compare as empty; raw text is retained.

## Visual timestamp quality

### Mean Visual First-Detection Delay and Timed Visual Occurrence Coverage

**Question:** Was each appearance captured while visible, and how quickly?

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
cover two occurrences. Coverage is matched occurrences divided by required
occurrences. Delay is the assigned frame time minus appearance start; average
covered delays within each video, then average defined video values.

**Example:** Eight of ten appearances are captured, giving coverage `0.80`.
Their delays total 12 seconds, giving that video's mean delay `12 / 8 = 1.5 s`.
Low delay must be read with coverage: detecting only an easy subset quickly
does not establish complete recovery.

| Edge case | Example | Coverage / delay | Explanation |
|---|---|---|---|
| No reference appearances | Verified text-free video | null/inapplicable for both | No timed recovery task exists, although content absence can be correct. |
| Valid output covers none | Required title never captured | 0 / null-unavailable | Recovery failed; there is no observed delay. |
| First attempt fails | Visual worker crashes | null/unavailable for both | No valid detection output exists. |
| Required annotation missing | Visible text without its claimed interval | No scoring | Reject during data validation. |

**Range and direction:** Coverage lies in `[0, 1]`, higher is better. Delay is
non-negative seconds, lower is better. A missed occurrence does not receive an
invented zero delay or an arbitrary maximum delay.

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

**Question:** What are typical and slow-tail complete visual-processing times?

For each video, take the median latency of its successful measured warm attempts.
Calculate p50/p95 across these video medians using shared linear interpolation.
The timer covers exactly the complete visual request listed under RTF.

**Example:** Video observations `10, 12, 14, 16, 18, 20` seconds give p50
`15 s` and p95 `19.5 s`. An interpolated percentile need not equal a measured
latency or put exactly 95% of a small observed sample below its value.

**Range and direction:** Non-negative seconds/video; lower is better. No
successful video observation gives null/unavailable. A video with failed
attempt 1 can still contribute later successful timings without supplying
quality. Report completion counts. Limited independent support keeps descriptive
point estimates, flags p95 limitations, and withholds unsupported CI bounds.
p50 can be the main summary but is not an arithmetic mean.

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

### Peak Visual Process-Tree RAM and Peak Visual Device VRAM

**Question:** What host and GPU memory does the complete visual lifecycle require?

At each sample, sum resident RAM of the worker and its living children/
grandchildren, including media subprocesses; report the largest simultaneous
total. Do not sum independent process peaks. CUDA VRAM is the largest raw NVML
device-total sample, not per-process memory or a baseline-subtracted increase.
Monitor from before loading through warmup and final measured inference.
Offline scoring and input validation are outside candidate monitoring.
Frozen-ASR models are unloaded before the fresh visual worker.

**Example:** Peaks of `5,600 MiB` RAM and `2,900 MiB` device VRAM are reported
separately.

**Range and direction:** Non-negative MiB; lower is better at equal quality.
CPU VRAM is null/inapplicable; missing CUDA telemetry is null/unavailable.
Missing RAM telemetry is unavailable. Retain real samples after failure with
partial/full lifecycle scope. One observed peak has no confidence interval.

## Workload descriptor

### Mean Selected Frames per Video

**Question:** How many frames did the strategy actually select?

Use each video's completed first-attempt selection count, regardless of later
parser acceptance or failure, then average videos with known completed counts.
Keep all per-attempt counts in timing records. Do not replace an unavailable
first selection with a later favorable one.

**Example:** If 360 frames are selected across 30 videos, the mean is `12`.
If selection yields 12 frames and parsing later fails, retain `12`, not the
number parsed successfully.

**Range and direction:** Non-negative frames/video; descriptive, not inherently
better when smaller. Failed/incomplete selection with unknown final count gives
null/unavailable. Zero selected frames violates mandatory frame zero: retain the
observed zero and selector failure, not a successful empty visual output.
No completed selection counts gives an unavailable aggregate.

## Frozen-ASR diagnostics

### Transcripts and word-error components

**Question:** What recognition quality does the shared audio result provide?

Decode each unique video once, transcribe each required 30-second/2-second-overlap
window once, shift timestamps, and stitch its transcript. Compare the complete
stitched result to that video's reviewed spoken reference. Do not score repeated
overlap words twice or copy results to visual children.

Use the [ASR recognition contract](../audio/metrics.md#recognition-quality-1):
align each valid completed video separately, then pool word/character edit
counts and contributing reference lengths. CER includes spaces left by the
shared projection. Substitution/deletion/insertion rates reuse pooled word
counts; raw integer counts stay in artifacts.

**Example:** 40 substitutions, 20 deletions, and 10 insertions across 1,000
contributing reference words give WER `0.07` and component rates `0.04`,
`0.02`, and `0.01`.

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

### Spoken timestamps

**Question:** Are the frozen transcript's spoken boundaries accurate and usable?

Reuse [ASR span alignment](../audio/metrics.md#timestamp-quality-1) on the stitched
video timeline: Content F1 ≥ `0.5`, ordered non-overlapping predicted spans,
no reuse, and the matched span's minimum start/maximum end. Pool real boundary
errors for MAE and matched/required segment counts from valid completed eligible
videos for coverage.

**Example:** 18 of 20 reviewed spoken segments align with 7.2 seconds total
error across their 36 boundaries: coverage `0.90`, MAE `0.20 s`.

No reviewed spoken timing task gives null/inapplicable; visual appearance
timestamps are not spoken timestamps. A claimed but missing spoken annotation
fails data validation. Valid output with required timings but no matches gives
coverage zero and MAE null/unavailable. Failed video transcription makes both
unavailable. Retain planned/contributing reference and matched-boundary counts.

### Empty and nonspeech behavior

**Question:** Does completed speech unexpectedly vanish, or does nonspeech
produce invented lexical text?

Unexpected Empty Transcript Rate divides valid empty complete video transcripts
by valid completed videos requiring nonempty projected spoken text.
Nonspeech False-Transcription Rate divides valid completed reviewed nonspeech
units producing lexical text by all valid completed reviewed nonspeech units.

The protocol must freeze one nonspeech unit type before evaluation: whole
nonspeech videos or explicitly reviewed actual ASR windows, including overlap.
Do not mix types, infer nonspeech from missing text, add separate inference
repetitions, or impose the standalone control-category allocation on video.
Require the reviewed units declared for the video corpus and retain their labels.

**Example:** Four unexpected empties among 90 completed eligible speech videos
give `4 / 90`; three invented outputs among 20 completed nonspeech units give
`3 / 20`. Failed outputs are not empty/false-transcription events.

**Range and direction:** Both rates lie in `[0, 1]`; lower is better. No events
with a positive completed denominator gives zero; all give one. No valid
completed eligible units gives null/unavailable. A legitimate slice without
the task is null/inapplicable; absent required reviewed inputs fail preparation.

### Audio attempt failures

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

### Audio latency, RTF, loading, and resources

**Question:** What cost did producing the shared complete audio results require?

One warmup follows cold ASR readiness. Time each complete video from audio
decoding through window preparation, native features, transcription, required
alignment, timestamp shifting, stitching, output conversion/validation, and
synchronization. Exclude cold loading, warmup, input verification, and offline
scoring. Record window timings, but p50/p95 use one completed warm
audio-processing latency per video, with shared linear interpolation.

Complete Audio-Pipeline RTF is summed successful complete-video processing time
divided by those videos' original audio durations. Count each original duration
once, not overlapping window durations.

**Example:** A four-minute video completed in one minute gives RTF `0.25`.
Complete video latencies `10, 12, 14, 16, 18, 20` seconds give p50 `15 s`
and p95 `19.5 s`.

Cold ASR loading ends at full model/profile readiness before warmup. Peak Audio
Process-Tree RAM sums simultaneous worker/subprocess resident memory; Peak Audio
Device VRAM follows raw NVML device-total sampling. Both monitor loading,
warmup, and all measured audio work, then stop before scoring/visual processing.

**Range and direction:** Latency/load are non-negative seconds, RTF is
non-negative and unbounded, and peaks are non-negative MiB; lower is better at
equal quality. No completed video timing gives null/unavailable latency and RTF.
Incomplete loading has no completed cold-load value. Valid prior measurements
survive later failure with scope. CPU VRAM is null/inapplicable; missing required
telemetry is null/unavailable. Limited support retains descriptive p50/p95,
not fabricated confidence bounds. Cold load and observed peaks have no interval.

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
