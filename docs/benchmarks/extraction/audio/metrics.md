# Audio extraction metrics

[Shared metric conventions](../../metrics.md) · [Benchmark methodology](methodology.md) · [Run commands](../../running.md)

On this page:

- [Metric summary](#metric-summary)
- [Recognition quality](#recognition-quality-1)
- [WER diagnostics](#wer-diagnostics-1)
- [Timestamp quality](#timestamp-quality-1)
- [Reliability and failure behavior](#reliability-and-failure-behavior-1)
- [Operational performance](#operational-performance-1)
- [Worked candidate interpretation](#worked-candidate-interpretation)
- [Audio confidence intervals](#audio-confidence-intervals)

The ASR benchmark evaluates the complete ordered transcript, its timestamps,
catastrophic output behavior, and the cost of the recorded runtime profile. It
does not use Content F1 or document Reading Order NED: unlike a two-dimensional
page, audio already defines one chronological sequence.

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
| Empty Transcript Rate | Diagnostic | How often does speech-containing audio produce no lexical text? | Lower |
| Nonspeech False-Transcription Rate | Diagnostic | How often does verified nonspeech audio produce lexical text? | Lower |
| Repeat Transcript Agreement Rate | Diagnostic | How often do repeated measured runs return the same scored transcript? | Higher |

### Operational performance

| Metric | Role | Question answered | Direction |
|---|---|---|---|
| Complete-Pipeline Real-Time Factor | Operational | How much processing time is required relative to audio duration? | Lower |
| p50 Warm Clip Latency | Operational | What is normal warm processing latency? | Lower |
| p95 Warm Clip Latency | Operational | What is slow-case warm processing latency? | Lower |
| Cold Model-Load Time | Operational | How long does initial model loading take? | Lower |
| Peak Process-Tree RAM | Operational | How much total system memory does the candidate require? | Lower |
| Peak Device VRAM | Operational | How much total GPU memory is occupied during candidate execution? | Lower |

These 16 metrics are the frozen ASR evaluation contract. Technical-Term
Accuracy is excluded because EduMind has no fixed subject vocabulary;
diarization metrics remain out of scope until speaker identification becomes a
product requirement.

## Recognition quality

### Corpus Word Error Rate

**Question:** How wrong is the complete ordered word transcript?

Corpus WER uses the fixed ASR evaluator normalization described in the shared
conventions. Edit counts are pooled across clips before division; the benchmark
does not average independently calculated clip error rates.

```text
Corpus WER =
all word substitutions + deletions + insertions
------------------------------------------------
        all words in the reference clips
```

**Example:** Across the complete corpus, 1,000 reference words with 40
substitutions, 20 deletions, and 10 insertions produce WER `70 / 1,000 = 0.07`.

WER is primary because ASR must reproduce the ordered spoken sequence in one
value. The component rates below explain its failure type rather than replacing
it.

**Range and direction:** WER is non-negative and lower is better. It can exceed
`1` when insertions outnumber reference words. A speech set with no reference
words is invalid.

### Corpus Character Error Rate

**Question:** How severe are character-level transcription errors that may be
hidden by whole-word scoring?

Corpus CER uses the same pooled calculation over characters:

```text
Corpus CER =
all character substitutions + deletions + insertions
-----------------------------------------------------
          all characters in the reference clips
```

**Example:** Fifty character edits across 5,000 reference characters produce
CER `50 / 5,000 = 0.01`.

CER supports WER by exposing small spelling, number, abbreviation, and name
errors that a whole-word error treats as one event.

**Range and direction:** CER is non-negative and lower is better. It can exceed
`1` when insertions outnumber reference characters. A speech set with no
reference characters is invalid.

## WER diagnostics

The three diagnostic rates reuse the exact word alignment and pooled reference
word count used by Corpus WER. Together they add up to WER, but each answers a
different failure question.

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

**Range and direction:** Boundary MAE is a non-negative number of seconds and
lower is better. Alignment Coverage lies in `[0, 1]` and higher is better.

## Reliability and failure behavior

### Empty Transcript Rate

**Question:** How often does a valid speech clip produce no lexical transcript?

For speech-containing clips:

```text
Empty Transcript Rate =
speech clips producing no lexical text
--------------------------------------
       speech clips evaluated
```

**Example:** Four empty transcripts from 100 speech clips produce a rate of
`4 / 100 = 0.04`.

This exposes complete transcription failures that can be diluted inside corpus
WER. A process crash is a failed candidate run, not an empty transcript.

An empty transcript with an empty timestamp sequence remains a scoreable speech
sample: all reference words and characters are deletions, timestamp coverage is
zero, Boundary MAE is null, and this rate increments. Non-empty transcript text
without timestamps and empty text with lexical timestamp segments are
contradictory fatal outputs.

**Range and direction:** The rate lies in `[0, 1]`; lower is better. A dataset
without speech clips is invalid for this metric.

### Nonspeech False-Transcription Rate

**Question:** How often does the model invent lexical speech on verified
nonspeech audio?

For the fixed silence, music-without-lyrics, background-noise, and environmental
sound controls:

```text
Nonspeech False-Transcription Rate =
nonspeech clips producing lexical text
---------------------------------------
       nonspeech clips evaluated
```

**Example:** Three controls producing text among 20 nonspeech clips produce a
rate of `3 / 20 = 0.15`.

Empty-reference WER can count insertions under the JiWER convention, but these
controls are deliberately excluded from speech Corpus WER/CER. The dedicated
rate measures how often verified nonspeech produces invented lexical speech,
without weighting it by the speech corpus's reference-word count.

**Range and direction:** The rate lies in `[0, 1]`; lower is better. It is
undefined when no reliability controls were evaluated. Authoritative runs
require the reviewed reliability subset, so its absence makes the report
incomplete; a debug artifact retains the unavailable value and reason.

### Repeat Transcript Agreement Rate

**Question:** How often does the same model profile return the same transcript
when it processes the same clip repeatedly?

For each speech clip, the three measured transcripts receive the same prose
projection used for WER. The clip receives `1` only when all three projected
transcripts are identical and `0` otherwise. The benchmark then averages those
clip values:

```text
Repeat Transcript Agreement Rate =
speech clips with identical repeated transcripts
-------------------------------------------------
             speech clips evaluated
```

**Example:** If 47 of 50 clips have identical transcripts in all three measured
runs, agreement is `47 / 50 = 0.94`.

This is a determinism diagnostic, not a correctness score. A model can repeat
the same wrong transcript perfectly, so the value must be read with WER. The
metric uses the repetitions already collected for latency and adds no model
inference.

**Range and direction:** `[0, 1]`; higher is better. Smoke uses one measured
run and therefore cannot provide meaningful determinism evidence.

## Operational performance

### Complete-Pipeline Real-Time Factor

**Question:** How much complete processing time is required relative to audio
duration?

```text
Complete-Pipeline Real-Time Factor =
total transcription-and-timestamp processing time
--------------------------------------
          total audio duration
```

**Example:** Processing 60 minutes of audio in 15 minutes produces RTF `15 / 60
= 0.25`.

An RTF below `1` means processing is faster than audio playback.

**Range and direction:** RTF is non-negative and lower is better.

### p50 and p95 Warm Clip Latency

**Question:** What are the typical and slow-tail steady-state transcription
times per clip?

For each clip, the benchmark takes the median latency of its measured warm
repetitions. It then reports p50 across clips as typical latency and p95 as the
slow tail.

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

**Range and direction:** Non-negative seconds; lower is better.

### Peak Process-Tree RAM and Peak Device VRAM

**Question:** What is the worker's peak host-memory use and the assigned GPU's
peak total memory use during complete candidate execution?

Peak Process-Tree RAM is the largest sampled resident-memory total for the
worker and its child processes. Peak Device VRAM follows the shared raw NVML
device-total contract, including loading and first-use allocations.
Every result records the explicit CPU or GPU
profile; silent device fallback is invalid.

**Example:** Process-tree samples peaking at `3,200 MiB` and GPU samples peaking
at `2,100 MiB` produce those two reported peaks.

**Range and direction:** Non-negative MiB; lower is better at equal quality. A
confirmed CPU-only profile reports zero VRAM; unavailable measurement is not
converted to zero.

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
but half of its word errors come from omitted speech and it invents text on 10%
of nonspeech controls. Those failure modes remain visible even though the total
WER is relatively low. No weighted overall score combines these values.

## Audio confidence intervals

### Which values receive an interval

| Value | 95% confidence interval? | Rule |
|---|---:|---|
| Development, validation, and locked WER, CER, and substitution/deletion/insertion rates | Yes | Resample complete speech clips and recalculate the pooled counts. |
| Development, validation, and locked Timestamp Alignment Coverage | Yes | Resample complete timed speech clips. |
| Development, validation, and locked Timestamp Boundary MAE | Yes, when defined | Use resampled clips containing valid aligned boundaries. |
| Development, validation, and locked Empty Transcript Rate | Yes | Resample speech clips. |
| Development, validation, and locked Nonspeech False-Transcription Rate | Yes | Resample the separate nonspeech controls. |
| Development, validation, and locked Repeat Transcript Agreement Rate | Yes | Resample complete speech clips with their already-computed agreement flags. |
| Development, validation, and locked Real-Time Factor | Yes | Resample complete speech clips with their measured processing time and duration. |
| Development, validation, and locked p50/p95 warm latency | Conditional | Report only when enough independent clips support the percentile estimate. |
| Smoke metrics | No authoritative interval | Smoke validates execution and is too small for selection claims. |
| One cold-load measurement | No | One observation cannot estimate uncertainty. |
| One observed peak RAM or VRAM value | No | Report the observed peak without invented bounds. |

### Calculation

Development, validation, and locked runs use 10,000 bootstrap resamples with seed 42:

1. Use the independent source IDs frozen in the manifest. A clip is a unit only
   when it is independent; related clips from one declared source travel together.
2. Resample those complete units with replacement; resample the nonspeech
   control units separately for their reliability rate.
3. Recalculate each eligible aggregate from the sampled counts, timestamps,
   durations, and latencies.
4. Use the 2.5th and 97.5th percentiles as the 95% bounds.

A resample with no timestamp match still contributes to recognition,
reliability, latency, and zero Alignment Coverage. Its undefined Boundary MAE
does not enter the MAE percentile calculation.

If no reference segment aligns anywhere in the complete candidate run,
Timestamp Boundary MAE is stored as null, Alignment Coverage is `0`, and the run
remains successful. This reports the absence of a measurable boundary without
inventing either a perfect or infinitely bad time error. The null is preserved
in `candidate.json` and `summary.json`; its MLflow scalar key is absent because
MLflow scalar metrics cannot represent null.

### Interpretation

Corpus WER `0.08` with a 95% CI of `[0.07, 0.10]` means the observed aggregate
error rate is 8%, while clip resampling estimates a plausible range of 7% to
10%. A narrower interval indicates a more precise corpus estimate; it does not
mean that every individual clip has an error rate inside that range.
