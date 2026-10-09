# Video extraction metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Visual-content quality](#visual-content-quality-1)
- [Visual timestamp quality](#visual-timestamp-quality-1)
- [Reliability and failure behavior](#reliability-and-failure-behavior-1)
- [Frozen-input diagnostic](#frozen-input-diagnostic-1)
- [Operational performance](#operational-performance-1)
- [Worked candidate interpretation](#worked-candidate-interpretation)
- [Video confidence intervals](#video-confidence-intervals)

The video experiment compares keyframe configurations while holding the
document parser and ASR fixed. Its quality metrics therefore answer two focused
questions: did the selected frames recover the useful visible content, and did
they place that content at a useful time? Spoken and visible tokens are not
combined into one score because the much longer transcript would dominate it.

## Metric summary

### Visual-content quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Visual Content F1 | Primary | Does the configuration balance correct visible text with complete visible-text recovery? | Higher |
| Visual Content Precision | Secondary | How much extracted visible text is supported by the reference? | Higher |
| Visual Content Recall | Secondary | How much verified visible text was recovered from selected frames? | Higher |

### Visual timestamp quality

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Mean Visual First-Detection Delay | Primary | After visible text first appears, how long does the strategy take to capture it? | Lower |
| Timed Visual Occurrence Coverage | Primary | What proportion of verified timed visible-text occurrences were captured at least once while visible? | Higher |

### Reliability and failure behavior

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Duplicate Visual Text Rate | Diagnostic | How much extracted visible content was repeated because similar frames were selected repeatedly? | Lower |

### Frozen-input diagnostic

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Frozen-ASR Transcript WER | Diagnostic | Is the frozen audio transcript sufficiently understood when interpreting the later combined pipeline? | Lower |

### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Visual Real-Time Factor | Operational | How much keyframe-selection and visual-parsing time is required relative to video duration? | Lower |
| p50 Warm Visual Latency | Operational | What is normal warm visual-processing time for one video? | Lower |
| p95 Warm Visual Latency | Operational | What is slow-case warm visual-processing time? | Lower |
| Cold Visual-Pipeline Load Time | Operational | How long does initial loading of the keyframe and document-parser path take? | Lower |
| Peak Visual Process-Tree RAM | Operational | How much system memory does the visual path require? | Lower |
| Peak Visual Device VRAM | Operational | How much total GPU memory is occupied while the visual path runs? | Lower |

### Workload descriptor

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Mean Selected Frames per Video | Workload descriptor | How many frames does the configuration send to the document parser on average? | Descriptive |

## Visual-content quality

### Visual Content Precision, Recall, and F1

**Question:** How much extracted visible content is supported, how much required
visible content was recovered, and does the configuration balance both?

The frozen `normalized_lines_distinct_v1` representation uses nonempty visible
text lines as content units, not individual words or model-tokenizer IDs.
Apply the shared prose projection independently to each line; equal normalized
lines count once within a video. Compare the distinct predicted and reference
line sets for exact matches. Repeated copies of the same line do not earn
additional content matches; their occurrences remain available for Duplicate
Visual Text Rate and timed occurrence matching.

```text
Visual Content Precision =
matched extracted visible-content units
---------------------------------------
all extracted visible-content units

Visual Content Recall =
matched reference visible-content units
---------------------------------------
all reference visible-content units

Visual Content F1 =
the balance between Visual Content Precision and Visual Content Recall
```

**Example:** The reference contains 100 visible-content units. A configuration
extracts 90 units, 80 of which match the reference. Precision is `80 / 90 =
0.89`, recall is `80 / 100 = 0.80`, and F1 is approximately `0.84`.

F1 is primary because it prevents a configuration from looking good by
extracting almost nothing or by extracting large amounts of unsupported text.
Precision and recall are supporting diagnostics that explain whether a lower F1
comes from extra text or missing text.

Corpus aggregates are recomputed from the summed per-video distinct-reference,
distinct-prediction, and matched-unit counts. They are not an unweighted mean of
the per-video ratios. Timed occurrence coverage and Duplicate Visual Text Rate
are pooled from their corresponding occurrence counts for the same reason.

**Range and direction:** All three values lie in `[0, 1]`; higher is better. If
both reference and prediction contain no visible content, the sample is not
eligible rather than being assigned perfect quality.

## Visual timestamp quality

### Mean Visual First-Detection Delay and Timed Visual Occurrence Coverage

**Question:** Did the strategy capture each timed visible-text occurrence while
it was on screen, and how long after appearance did the first capture happen?

Each verified visible-text occurrence has text plus an interval during which it
is visible. A reference occurrence is covered when a selected frame inside that
interval yields matching text. Its delay is the first matching frame time minus
the reference start time.

The versioned video protocol supplies the normalized Content F1 eligibility
threshold. Here Content F1 compares whitespace-token occurrences within the
reference and predicted line; it is separate from the exact distinct-line
content counts above. The protocol freezes frame-time tolerance at zero, so a
frame outside the verified interval
is never eligible. Matching is one-to-one: each reference occurrence and
each normalized text unit from a selected frame can be used at most once.
Cardinality is maximized first, then the earliest eligible detections are chosen,
so repeated frames cannot inflate coverage and delay really is first detection.

```text
Timed Visual Occurrence Coverage =
timed reference occurrences captured while visible
--------------------------------------------------
          all timed reference occurrences

Mean Visual First-Detection Delay =
sum of first matching frame time minus reference start time
-----------------------------------------------------------
                covered reference occurrences
```

**Example:** Suppose 8 of 10 timed occurrences are captured while visible, so
coverage is `0.80`. If their first matching frames arrive a total of 12 seconds
after their verified starts, mean first-detection delay is `12 / 8 = 1.5
seconds`.

The two values must be read together. Low delay with low coverage means the
strategy quickly captured only an easy subset. A useful result has high
coverage and low delay. If nothing is covered, coverage is zero and delay is
undefined rather than reported as a false zero.

**Range and direction:** Coverage lies in `[0, 1]` and higher is better. Delay
is a non-negative number of seconds and lower is better; it is null when no
timed occurrence is covered.

## Reliability and failure behavior

### Duplicate Visual Text Rate

**Question:** How much output repeats content already recovered from an
unchanged or repeatedly selected frame?

```text
Duplicate Visual Text Rate =
repeated normalized visible-content occurrences after their first occurrence
-------------------------------------------------------------------------
all extracted visible-content occurrences
```

**Example:** If a configuration extracts 50 visible-text occurrences and 15 are
repeated copies of content already recovered from unchanged frames, the rate is
`15 / 50 = 0.30`.

This metric exposes wasted document-parser work and repeated downstream context.
It is separate from content precision so duplicates cannot obscure whether the
text itself is supported by the video.

**Range and direction:** The rate lies in `[0, 1]`; lower is better. An output
with no visible-content occurrences is undefined for duplication and is handled
by the visual-content metrics instead.

## Frozen-input diagnostic

### Frozen-ASR Transcript WER

**Question:** How accurately does the already selected ASR transcribe the audio
of this video corpus?

The ASR profile and its audio output are frozen before frame selection begins.
Transcript WER is therefore calculated once for the shared ASR output and
stored on the phase's frozen-ASR child run. It is a diagnostic for understanding the final
combined extraction, not a metric for ranking keyframe configurations.

Its calculation, range, and direction use the
[Corpus WER contract](../audio/metrics.md#corpus-word-error-rate). It is calculated once per video phase rather than repeated for every
keyframe configuration.

## Operational performance

### Visual Real-Time Factor

**Question:** How much keyframe-selection and visual-parsing time is required
relative to video duration?

```text
Visual Real-Time Factor =
keyframe-selection and visual-parsing time
------------------------------------------
               video duration
```

**Example:** Extracting a four-minute video in one minute produces an RTF of
`1 / 4 = 0.25`. A value below 1 means extraction is faster than playback.

**Range and direction:** RTF is non-negative and lower is better.

### p50 and p95 Warm Visual Latency

**Question:** What are the typical and slow-tail visual-processing times after
the document parser is loaded?

For each video, take the median latency of its measured warm repetitions. The
benchmark reports p50 across videos as typical latency and p95 as the slow tail.

**Example:** A p95 of `48 seconds` estimates the slow 95th-percentile warm
video request. It does not mean exactly 95% of a small observed sample must
fall below an interpolated value.

**Range and direction:** Non-negative seconds per video; lower is better.

### Cold Visual-Pipeline Load Time

**Question:** How long does a fresh worker need to load the keyframe and frozen
document-parser path required by the video configuration?

Measure model construction and loading before warmups or video extraction. It
is kept separate from warm latency so startup does not distort steady-state
performance.

**Range and direction:** Non-negative seconds; lower is better.

### Peak Visual Process-Tree RAM and Peak Visual Device VRAM

**Question:** What peak host memory does the visual worker use, and what peak
total memory is occupied on its assigned GPU?

Peak RAM is the largest sampled resident-memory total across the worker and its
child processes. Peak Visual Device VRAM follows the shared raw NVML device-total
contract during the fresh visual worker's lifecycle. Frozen ASR runs separately;
its models and allocations are not kept resident during visual measurement.

**Example:** If process-tree RAM peaks at `5,600 MiB` and GPU memory peaks at
`2,900 MiB`, those are the reported resource values.

**Range and direction:** Non-negative MiB; lower is better at equal quality. A
confirmed CPU-only profile reports zero VRAM; unavailable measurement is not
converted to zero.

### Mean Selected Frames per Video

**Question:** How many frames does the strategy send to the frozen document
parser, on average?

```text
Mean Selected Frames per Video =
total selected frames
---------------------
videos evaluated
```

**Example:** Selecting 360 frames across 30 videos produces a mean of `12`
frames per video.

**Range and direction:** Non-negative frames per video. Lower is preferable
only when visual-content and timestamp quality are preserved; it is a cost
diagnostic rather than a standalone selection objective.

## Worked candidate interpretation

Suppose one keyframe configuration produces:

```text
Visual Content Precision           = 0.92
Visual Content Recall              = 0.76
Visual Content F1                  = 0.83
Mean Visual First-Detection Delay  = 1.5 seconds
Timed Visual Occurrence Coverage   = 0.80
Duplicate Visual Text Rate         = 0.18
Mean Selected Frames per Video     = 14
Visual Real-Time Factor            = 0.35
```

Most extracted visible content is supported by the reference, but approximately
one quarter of the required content is still missed. The matched content is
captured quickly when found, although 20% of timed occurrences were never
captured while visible. Eighteen
percent duplicate output suggests that the strategy still selects some
unchanged frames. The frame count and RTF show the processing cost that produced
this quality. No weighted overall score combines these values.

## Video confidence intervals

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Development, validation, and locked Visual Content Precision/Recall/F1 | Yes | Resample complete videos and recalculate each aggregate. |
| Development, validation, and locked Timed Visual Occurrence Coverage | Yes | Resample complete videos with their timed visible references. |
| Development, validation, and locked Mean Visual First-Detection Delay | Yes, when defined | Use resampled videos containing covered timed occurrences. |
| Development, validation, and locked Duplicate Visual Text Rate | Yes | Resample complete videos with their duplicate counts. |
| Development, validation, and locked Frozen-ASR Transcript WER | Yes | Resample the shared per-video ASR outputs; report the result on the frozen-ASR child run. |
| Development, validation, and locked Visual Real-Time Factor and Mean Selected Frames per Video | Yes | Resample complete videos with their duration, visual-processing time, and frame counts. |
| Development, validation, and locked p50/p95 warm latency | Conditional | Report only when enough independent videos support the percentile estimate. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| One cold-load measurement | No | One observation cannot estimate uncertainty. |
| One observed peak RAM or VRAM value | No | Report the observed peak without invented bounds. |

### Calculation

Development, validation, and locked runs use 10,000 bootstrap resamples with seed 42:

1. Use the independent source IDs frozen in the manifest. A complete video is
   a unit only when independent; excerpts from one declared source travel together.
2. Resample those complete units with replacement.
3. Recalculate each eligible aggregate from the sampled visual matches, timed
   occurrences, duplicates, durations, latencies, and frame counts.
4. Use the 2.5th and 97.5th percentiles as the 95% bounds.

A resample with no covered timed occurrence still contributes zero Timed Visual
Occurrence Coverage and contributes normally to the other defined metrics. Its
undefined First-Detection Delay does not enter the delay percentile calculation.

### Interpretation

Visual Content F1 `0.84` with a 95% CI of `[0.79, 0.88]` means the point
estimate summarizes the observed videos, while video resampling estimates the
uncertainty around it. A narrow interval indicates a more precise corpus-level
estimate; it does not describe the range of individual-video F1 values.
