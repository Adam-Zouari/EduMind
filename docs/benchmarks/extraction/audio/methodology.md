# Audio extraction methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Terminology and unit of comparison](#terminology-and-unit-of-comparison)
- [ASR profiles](#asr-profiles)
- [Data](#data)
- [Common input and output rules](#common-input-and-output-rules)
- [Per-candidate execution](#per-candidate-execution)
- [Development, validation, and locked test](#development-validation-and-locked-test)
- [Metrics and why they are used](#metrics-and-why-they-are-used)
- [MLflow result structure](#mlflow-result-structure)

Which English speech-to-text profile produces the most accurate educational
transcript with timestamps suitable for navigation and citations?

## Terminology and unit of comparison

An **ASR profile** is one complete executable transcription configuration. It
includes the model revision, decoder settings, device, numeric precision,
audio preprocessing, timestamp method, and any required aligner. The benchmark
runner calls it a `candidate`, but the result belongs to the complete profile,
not only to the model weights.

Each speech clip produces:

- one ordered transcript;
- timestamped segments in the common benchmark representation;
- warnings and timing information; and
- the counts required to reproduce Corpus WER and CER.

Audio already defines chronological order, so the benchmark evaluates the final
ordered transcript with WER. It does not split recognition into Content F1 and
document Reading Order NED for two-dimensional element sequences.

One audio clip supplies a quality observation. Independent clips are resampling
units; related clips from a reviewed common source travel together under the
manifest's group ID. Three measured attempts improve timing and repeatability
diagnosis but never become three independent quality samples.

The revised ASR contract below is approved documentation; executable alignment
of its failure accounting and repeatability remains pending. See the
[runbook's implementation status](../../running.md#1-prepare-the-environment).

## ASR profiles

| Model/profile | Why it is included |
|---|---|
| Whisper `small.en` | Established English control with timestamp output. |
| Canary 180M | Compact timestamp-capable challenger. |
| Parakeet TDT 0.6B v2 | Mid-size profile with word, segment, and character timestamps. |
| MOSS Transcribe-Diarize | Larger timestamp-capable transcription challenger. Diarization is not scored because speaker identification is not currently an EduMind requirement. |

## Data

Smoke uses two committed audio clips. The authoritative corpus contains 90
English clips split 54 development, 18 validation, and 18 locked. It includes:

- verified transcripts;
- complete audio duration;
- segment or word timestamps;
- clean and noisy speech;
- accents and multiple speakers;
- technical and educational vocabulary.

The [dataset guide](../../datasets.md) defines the LibriSpeech, M³AV, EdAcc, and AMI
source pools and the recommended allocation. The exact recordings are
fixed by the manifests, which record source, revision, selected clip interval,
license, checksum, duration, and speaker/document family. Public leaderboard
scores are screening context; they do not replace this frozen corpus.

A small fixed reliability set contains verified silence, music without lyrics,
background noise, and other nonspeech audio. These controls are separate from the
90 speech clips. They measure false transcription on audio with no spoken
reference and contribute their request outcomes to Attempt Failure Rate, not
speech Corpus WER/CER or speech repeatability. The reliability manifest labels its development,
validation, and locked-test controls so each phase uses only its own subset.

Every speech sample records at least:

```text
sample ID
source, license, revision, and checksum
split and document/speaker family
duration and audio-condition labels
verified transcript
verified timestamped reference segments
```

Each sample stores a `conditions` list. It contains exactly one acoustic label,
`clean` or `noisy`, and may also contain `accented` and/or `multi_speaker`.
Every authoritative split covers all four labels. They remain in the per-sample
artifact for diagnosis and do not create extra required MLflow metric namespaces
or a larger metric contract.

Prepare and review each split through the shared
[data-validation workflow](../../data-validation.md). It distinguishes an
explicit reviewed empty transcript from a missing annotation, checks duration,
required timestamps, and control categories, and rejects a full speech split
with no normalized reference words or eligible timed segments. Individual
legitimate empty projections remain valid. No data check runs inside measured
transcription or candidate resource monitoring. Reusable validation reports and
the standalone commands are the planned interface, not yet implemented.

## Common input and output rules

Every candidate receives the same canonical audio: mono, 16 kHz, signed 16-bit
PCM before model-native feature extraction, with no
candidate-specific denoising, volume repair, prompting, or vocabulary hints.
Model-native feature extraction and the documented deterministic decoder remain
part of the ASR profile and are recorded. Candidate output receives only the
fixed evaluator normalization used by WER and CER; the evaluator does not repair
misspellings, remove repetitions, or rewrite transcripts.

All timestamp outputs are converted to ordered benchmark segments containing
text, start time, and end time. For each reference segment, the evaluator
considers contiguous predicted spans whose normalized token Content F1 is at
least `0.5`. Dynamic programming chooses ordered, non-overlapping one-to-one
matches by maximum total similarity, then match count, then earliest span.
Predicted timestamp units cannot be reused by another reference segment. This
supports word timestamps and broader segment timestamps without truncating
unequal arrays. Boundary MAE uses the matched span's minimum start and maximum
end; Alignment Coverage records how much of the timed reference aligned.

An empty transcript with no timestamp segments is a valid completed output.
Against required speech, all reference words/characters are deletions, coverage
is zero, Boundary MAE is unavailable, and Unexpected Empty Transcript Rate
records an event. Against a legitimately empty projected reference, WER/CER and
the three error rates are zero. Timestamp metrics are inapplicable when no
reference boundaries are required; this is not a failed alignment. Nonempty
lexical text without required timestamps,
empty text with lexical timestamp segments, or invalid required boundaries/schema
are failed attempts. Recoverable failures do not cancel later scheduled attempts.

## Per-candidate execution

One child run executes one ASR profile in a fresh operating-system process. A
CPU process has CUDA hidden before any model runtime is imported; a CUDA process
must provide working NVML VRAM measurement. No profile may change device or use
CPU/GPU offloading silently.

For development, validation, and locked, the process performs:

```text
load the exact pinned model
→ record cold model-load time
→ run one warmup
→ execute three measured warm attempts for every deterministically shuffled speech clip
→ execute one measured attempt per corresponding nonspeech reliability control
→ aggregate quality, timestamp, reliability, and operational results
→ unload the model and release resources
```

Smoke uses the same one-warmup lifecycle with one measured attempt per speech
fixture and control, so it cannot estimate repeatability. Preflight is separate:
zero warmups and the retained reviewed stress inference under monitoring, with
no quality scoring. All scheduled speech clips supply repeatability in the
three-attempt profiles; it is not a selected portion of the corpus or another
benchmark invocation.

The quality result for a clip comes from its designated first measured output.
Repeated executions preserve timing and outcome records but are not averaged
into additional quality samples. The first attempt is selected by index, not
success; never replace a failed first output with a successful later one.
Transcript Repeatability Success Rate scores all three scheduled pairs: both
attempts must return valid identical projected transcripts for pair credit.
Attempt Failure Rate accounts for all scheduled measured speech/control requests.
Three attempts provide three comparison pairs and distinguish full agreement,
partial agreement, and no agreement; they do not establish statistical precision.

Authoritative comparisons use the
[frozen CUDA hardware profile](../../methodology.md#shared-mlflow-lifecycle):
batch size `1`, the stage's supported 16-bit dtype, one whole model on the GPU,
and no fallback, offload, device splitting, or quantization. Device, dtype,
decoder, timestamp path, and runtime versions are recorded. Smoke and debugging
may explicitly request CPU or CUDA but cannot support candidate selection.
Recorded measured-attempt failures are reliability evidence, not automatically
an incomplete comparison. Missing scheduled records, fatal setup failures, or
unresolved required measurement/evaluation failures make it incomplete. A fully
accounted execution can report unavailable conditional quality values alongside
its failure rate; it must not be presented as successful transcription of the
entire workload. The engineer reviews evidence and decides which candidates
advance; no measured quality threshold selects a winner automatically.

Behavior-changing settings are part of each child artifact: batch size one;
Whisper word timestamps and deterministic generation; Canary beam size one,
punctuation, capitalization, and timestamps; Parakeet greedy-batch decoding and
timestamp level; and MOSS `max_new_tokens=2048` with `do_sample=false`. Runtime
versions, model paths, revisions, cache-manifest checksums, dtype, and device are
recorded with them.

## Development, validation, and locked test

```text
smoke:
separate CPU and CUDA runs on tiny committed speech and nonspeech fixtures
→ verify loading, transcription, timestamps, scoring, artifacts, and cleanup

preflight:
fresh CUDA worker per declared ASR profile → reviewed demanding development/stress input
→ exact hardware qualification, with no warmup or quality ranking

development:
all hardware-qualified members of the four-profile ASR roster on 54 speech clips
and development reliability controls
→ engineer reviews MLflow and records finalists

validation:
engineer-selected finalists on 18 unseen speech clips and validation controls
→ engineer records exactly one selected ASR profile

locked test:
one benchmark invocation for the selected profile on 18 locked speech clips
and locked controls, using the configured measured repetitions
→ final unbiased ASR report; no further tuning in this benchmark version
```

Smoke validates wiring only. Development is where all candidates that passed
the exact matching preflight are compared; exclusions remain recorded in the
parent provenance.
Validation checks whether the chosen finalists retain their behavior on unseen
recordings. The locked split is used only after the engineer has selected one
profile. MLflow records evidence throughout but never advances a candidate or
changes application configuration.

## Metrics and why they are used

| Category | Metrics | Why they are needed |
|---|---|---|
| Recognition | **Corpus WER** (primary), Corpus CER | WER measures the complete ordered word transcript; CER exposes character-level spelling, name, and number errors. |
| WER diagnostics | Word Substitution Rate, Word Deletion Rate, Word Insertion Rate | Shows whether WER comes mainly from confused, omitted, or unsupported words. These explain WER but do not replace it. |
| Timestamps | **Timestamp Boundary MAE**, **Timestamp Alignment Coverage** | MAE measures the accuracy of aligned start/end boundaries; coverage prevents a candidate from looking accurate after aligning only easy segments. |
| Reliability | Unexpected Empty Transcript Rate, Nonspeech False-Transcription Rate, Transcript Repeatability Success Rate, Attempt Failure Rate | Distinguishes valid empty/invented first outputs from execution failures and measures repeatable delivery across scheduled speech attempts. |
| Operational | **Complete-Pipeline Real-Time Factor**, p50/p95 warm clip latency, cold model-load time, peak process-tree RAM, peak device VRAM | Measures the complete transcription and alignment cost of the frozen CUDA profile. |

Content F1 and document Reading Order NED are not ASR metrics in this benchmark.
Audio already supplies chronological order, so Corpus WER evaluates the required
ordered transcript. Technical-Term Accuracy is also excluded because EduMind is
not restricted to a stable subject vocabulary. Diarization is not scored unless
speaker identification becomes a product requirement.

The exact calculations, examples, ranges, directions, and confidence-interval
rules are defined in [audio metrics](metrics.md). Corpus WER, CER, and the three
WER components pool valid completed first-output edit counts before division; they are
not averages of independently calculated clip error rates. Timestamp Boundary
MAE and Alignment Coverage are interpreted together. Reliability controls are
excluded from speech Corpus WER/CER and evaluated by their own false-transcription
rate. Every authoritative split requires reviewed controls from all four
nonspeech categories; absent required controls fail data preparation. The two
output-event rates use successful eligible first outputs, with Attempt Failure
Rate and contributing counts beside them. Recognition diagnostics exclude
failed first outputs rather than inventing deletions; both timestamp metrics
are unavailable for those failed outputs. Planned reference counts stay in
artifacts, while scored timestamp coverage uses valid completed first outputs. These denominators answer different
questions and are frozen explicitly in [audio metrics](metrics.md).

## MLflow result structure

Audio uses its own MLflow experiment. Display names omit timestamps and redundant
benchmark prefixes; MLflow run IDs and recorded timestamps identify individual
invocations. Smoke creates separate CPU and CUDA parents, and preflight creates
one qualification parent before development:

```text
MLflow experiment: EduMind / ASR
├── parent: smoke-cpu
│   └── one child per smoke-tested ASR profile on CPU
├── parent: smoke-cuda
│   └── one child per smoke-tested ASR profile on CUDA
├── parent: preflight
│   └── one qualification child per declared ASR profile
├── parent: development
│   └── one child per hardware-qualified ASR profile
├── parent: validation
│   └── one child per engineer-selected finalist
└── parent: locked
    └── one child for the selected ASR profile
```

The parent is the comparison run. It stores the phase, profile, dataset and
reliability-manifest checksums, candidate order, seed, Git state, hardware,
dependency locks, model revisions, runtime plan, and any engineer-decision file.
Its artifacts are `plan.json`, `provenance.json`, the frozen manifests, and
`summary.json`. Its only direct metrics describe completion: whether the entire
comparison completed and how many candidates succeeded or failed.

Each child is one ASR profile. Its parameters contain the complete resolved
runtime profile even when some values repeat the parent plan: candidate and
submodel revisions and paths, device, dtype, language, decoder, timestamp
method, seed, FFmpeg version, canonical audio format, duration limit, warmups,
repetitions, data split, and manifest checksums. This makes a child interpretable
when viewed or exported alone. It has no child runs for individual clips,
repetitions, or metrics.

ASR child metrics use descriptive flat names because the run already has
`stage=audio` and no metric names collide inside it:

```text
word_error_rate
character_error_rate
word_substitution_rate
word_deletion_rate
word_insertion_rate

timestamp_boundary_mae_seconds
timestamp_alignment_coverage

unexpected_empty_transcript_rate
nonspeech_false_transcription_rate
transcript_repeatability_success_rate
attempt_failure_rate

real_time_factor
p50_warm_clip_latency_seconds
p95_warm_clip_latency_seconds
cold_model_load_seconds
peak_process_tree_ram_mb
peak_vram_mb
```

The recognition, timestamp, reliability, and operational labels remain useful
documentation categories, but they are not repeated as MLflow prefixes.
Applicable development, validation, and locked uncertainty bounds use the shared MLflow suffix
[metric convention](../../methodology.md#shared-mlflow-metric-convention).

Eligible defined recognition, timestamp, reliability, RTF, and sufficiently
supported warm-latency values receive 10,000 independent-source bootstrap
resamples with seed 42 and percentile 95% bounds. Each draw preserves clips and
all their attempts, then recalculates the same pooled/conditional statistic;
controls are resampled separately. Confidence intervals describe source-sample
uncertainty, not variation across the three attempts. Loading and observed
resource peaks have no fabricated intervals. Undefined conditional draws retain
defined/undefined counts; insufficient support keeps null bounds with a reason.
The complete procedure is in [audio metrics](metrics.md#audio-confidence-intervals).

Each candidate retains these artifacts, including observed records when execution
cannot finish:

| Artifact | Contents and purpose |
|---|---|
| `samples.parquet` | One row per speech/control sample with source/group IDs, labels, duration, designated-first-attempt status, known reference lengths, raw word/character edit counts when available, timestamp denominators/matches/errors, event eligibility/flags, repeatability score, and warnings. Unavailable predicted edit counts remain null. |
| `timings.parquet` | One row per scheduled speech/control measured attempt: index, success/failure/not-executed status, error, elapsed time, duration, valid completed latency where available, and device. Canonical transcript hashes allow pairwise agreement audit without uploading raw predictions. |
| `candidate.json` | Resolved runtime parameters, execution status, fingerprint, all metric fields with value/status/reason/support counts, intervals, operational values, and artifact references. |

The fixed artifact schema uses null plus status/reason for inapplicable or
unavailable values, not omitted required fields or fabricated zeros.
Raw audio and candidate predictions are not uploaded to MLflow. The frozen
speech manifest is uploaded and contains the verified reference transcripts,
source identifiers, and checksums needed to reproduce scoring.

Every completed development, validation, or locked child retains all 17 aggregate
metric fields in JSON, even when a conditional value is null. Required artifact
fields are not omitted to hide failures. MLflow receives only numeric scalars;
artifacts retain nulls, reasons, scheduled/contributing/failure counts, and
planned/contributing reference lengths. Historical runs preserve their recorded
metric names and protocol versions, not a reinterpretation under this contract.
A CPU profile records VRAM as null/inapplicable. Missing CUDA instrumentation
records null/unavailable, not zero.
CUDA children record `vram_measurement_method="nvml-device-total"` and the
assigned device's raw peak memory use under the shared hardware contract.
Per-process byte availability on Windows WDDM is not required. Device placement
is verified independently; a missing device-memory measurement fails the CUDA
child rather than producing a fabricated zero.
An observed request crash or invalid required timestamp output is an attempt
failure. If a worker cannot continue, retain observed outcomes and mark the
remaining requests not executed; do not invent failure records or a complete
repeatability score. Missing required records/artifacts or unresolved telemetry
make the comparison incomplete. Fully recorded attempt failures remain visible
reliability evidence rather than silently excluding difficult clips.

The engineer reviews the completed child runs and per-sample artifacts. No
weighted overall score is calculated. Finalist and winner choices are written
to explicit engineer-decision files containing the source parent run and
selected child profiles. The selected ASR is then frozen for video extraction.
