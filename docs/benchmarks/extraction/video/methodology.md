# Video extraction methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Candidates](#candidates)
- [Data](#data)
- [Execution](#execution)
- [Metrics and why they are used](#metrics-and-why-they-are-used)
- [MLflow result structure](#mlflow-result-structure)

With the image parser and ASR fixed, which keyframe policy recovers useful
on-screen text at useful times without unnecessary processing and duplicate output?

The runner implements the metric contract, grouped run structure, and execution
flow below. Reviewed authoritative datasets and data-derived protocol selections
remain separate prerequisites.

## Candidates

Development compares nine configurations of three strategies:

| Strategy | Configurations | Question answered |
|---|---|---|
| Fixed interval | One frame every 5, 10, or 20 seconds | What visual recovery is gained by more frequent sampling, and at what cost? |
| Scene change | FFmpeg scene threshold 0.30, 0.40, or 0.50 | How sensitive should transition detection be? |
| Hybrid | Selected scene threshold plus maximum gap of 5, 10, or 20 seconds | How often must fallback sampling capture gradual or static changes? |

Every configuration includes frame zero and uses variable-frame-rate FFmpeg
output. The hybrid configurations share the scene threshold chosen from a
preliminary development scene comparison. Testing all thresholds with all gaps
would produce 15 configurations; that interaction grid is not the current design.
The staged choice does not establish that every untested hybrid threshold would
perform worse.

The intervals are development search points, not universal constants: 5 seconds
is the more frequent setting, 20 seconds the less frequent setting, and 10 seconds
the midpoint. FFmpeg's [scene-filter guidance](https://ffmpeg.org/pipermail/ffmpeg-cvslog/2012-June/051105.html)
describes approximately 0.3–0.5 as a practical starting range. Hybrid gaps reuse
the fixed intervals for direct cost comparison.

## Data

Smoke uses two committed videos. The reviewed authoritative allocation is 30
educational videos: 18 development, 6 validation, and 6 locked. Sources cover
slides, screen recordings, presenter video, gradual changes, and genuine repeated
appearances. The [dataset guide](../../datasets.md#video-extraction) defines the
SlideSpeech, AVLectures, and EduMind-owned allocation.

Each video has a verified spoken transcript, positive duration, complete
visible-text annotation (including explicit text-free negatives), occurrence IDs,
and visibility intervals. A genuine reappearance has a new occurrence ID.
Spoken timing metrics additionally require reviewed spoken segments; visual
appearance intervals cannot stand in for them. Reviewed nonspeech videos or
actual ASR windows support Nonspeech False-Transcription Rate. Their consistent
unit type and allocation are frozen after development data review, not inferred
from missing transcripts or invented before review.

Manifests record exact assets, source intervals, revisions, licenses, checksums,
split assignments, and independent source-group IDs. Related excerpts stay in
one split and travel together in bootstrap draws. Public subtitles/OCR are
annotation seeds and must be reviewed against the video.

Use the [data-validation workflow](../../data-validation.md) during preparation.
Automatic validated-input verification checks the matching report before each
invocation. Both finish before candidate loading, monitoring, or timing.
Run `python -m experiments.benchmarks.validate video --profile <profile>` after
preparing the inputs. Frozen-artifact identity checks are separate runtime
integrity gates.

## Execution

The video protocol owns frame-selection ranges, frame-zero/VFR rules, windows,
overlap/stitching, text unitization, matching, dataset requirements, and execution
profiles. Frozen audio composes the video and ASR protocols; visual execution
composes the video and document protocols and the selected image-parser decision.
Workers verify their resolved identities before loading models.

### Shared audio preparation

The frozen-ASR child performs cold initialization and one separate representative
warmup, then prepares every unique video in the phase. Warmup output is discarded;
it is not an artifact result or a measured transcription. Measured preparation
follows this sequence for each video:

```text
For each video:
decode audio once
→ deterministic windows ≤30 seconds, with 2-second overlap
→ selected ASR once per required window
→ shift window-local timestamps to the video timeline
→ stitch the normalized suffix/prefix overlap once, retaining native timestamps
→ order retained timestamp units chronologically (stable window order on ties)
→ retain transcript completion status and every window outcome
```

The checksummed `FrozenASRArtifact` records the producing run ID, selected-model
decision fingerprint, model-lock identity, manifest and composed protocol
checksums, executing/scoring code and software-lock hashes, installed runtime
versions, FFmpeg version/commands, per-video transcripts/segments, window outcomes,
latencies, edit counts, and resource observations. Artifact reuse requires
matching identities and exact sample membership. Any changed composed protocol
requires regeneration. Changed executing code or dependencies also require a
new artifact; do not reinterpret old artifacts under new definitions.

A failed required window makes that video's complete transcript unavailable.
Keep partial output and failure evidence without scoring it as a complete video.
Missing required records make preparation incomplete. A fully accounted set of
observed request failures remains reliability evidence, not automatically a
quality-based disqualification. Required setup, artifact-integrity, placement,
and measurement gates remain separate.

After preparation, unload the ASR and release its allocations. Visual candidates
run in fresh workers, reference the artifact/run ID, and never invoke ASR.
One frozen-ASR child covers all phase videos; videos/windows are artifact rows,
not nested candidate runs.

### Visual worker lifecycle

For each candidate, start monitoring before initialization, record cold readiness,
run one complete representative warmup, then process every phase video with the
same frozen settings and seed. Development, validation, and locked schedule three
measured attempts per video; smoke schedules one.

Attempt 1 alone supplies quality and duplication. Attempts 2/3 supply timing,
failure, and repeatability evidence. Do not replace a failed first attempt with
a later success, and do not erase valid first quality after a later failure.
Recoverable errors do not cancel remaining attempts. Missing required outcomes
make execution incomplete.

Warm processing includes FFmpeg selection/extraction, frame preparation, image
parsing, conversion, output validation, and device synchronization. Cold load,
warmup, frozen ASR, input verification, and offline scoring are excluded from warm
latency/RTF. Resource monitoring includes loading, warmup, and measured visual
work, then ends before scoring. Docling's explicit
`initialize_pipeline(InputFormat.IMAGE)` and Paddle's equivalent constructor
readiness define cold visual initialization.

Repeatability compares canonical selected-frame times and structured visual
outputs across the three scheduled pairs. Numeric precision, normalization,
ordering, and stable ID mapping are frozen. Every attempt executes extraction,
not a cached previous result.

### Development, validation, and locked

1. Run smoke on CPU and CUDA separately, then qualify the selected frozen ASR
   and executable fixed/scene policies using reviewed development/stress inputs.
2. Run `development-scene-selection`: the shared audio preparation and qualified
   scene thresholds 0.30, 0.40, and 0.50 on development videos.
3. Review that comparison. Write one declared threshold and its source run ID
   into `selected_scene_threshold` and `selected_scene_source_run_id` in video
   `protocol.yaml`, then increment the version. The runner never writes them.
4. Refresh preflight and regenerate frozen ASR because the checksum changed.
   Run the main `development` comparison with all qualified members of the final
   nine-policy roster. Fixed/scene results are fresh under the same checksum as
   hybrid results; do not mix preliminary scores into this comparison.
5. Record finalists in `data/benchmarks/decisions/video-validation.json`.
   Evaluate only those finalists on unseen validation videos, with validation's
   own shared audio preparation.
6. Record exactly one winner in `video-locked.json`. Run one reporting-only
   locked invocation on unseen locked videos, including its own audio preparation
   and the configured three visual attempts per video. Locked evidence does not
   reopen selection.

Until threshold freezing, the authoritative threshold/source run ID are null
and hybrids cannot execute. Smoke's 0.40 threshold is fixture-only evidence.
Qualification before freezing covers the executable roster; afterward it covers
the final declared roster under the new checksum. Definitive hardware exclusions
stay visible. Missing preparation/readiness evidence blocks execution rather than
silently changing the selected ASR or parser.

## Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | Visual Content F1; Timed Visual Occurrence Coverage; Mean Visual First-Detection Delay | Separates useful-content recovery, recovery while visible, and capture delay. Interpret delay with coverage. |
| Secondary | Visual Content Precision/Recall | Separates unsupported visible text from omissions. |
| Diagnostic | Duplicate Visual Text Rate | Counts identical repeated output within one visibility occurrence, not genuine reappearance. |
| Diagnostic | Repeatability Success Rate; Attempt Failure Rate | Separates output agreement and successful delivery from first-output correctness. |
| Frozen-input diagnostic | WER/CER, word-error rates, spoken timestamps, empty/nonspeech behavior, and ASR failures | Describes the shared audio result; does not compare visual policies. |
| Operational | Separate visual and audio RTF, p50/p95 latency, cold load, RAM, and VRAM | Keeps audio preparation cost separate from candidate-dependent visual cost. |
| Descriptor | Mean Selected Frames per Video | Counts actual selected frames even when later parsing fails. |

The [metric definitions](metrics.md) give calculations, examples, edge cases,
and interval eligibility. Visual quality uses video-macro aggregation; audio
recognition pools valid complete-video edit counts. Latency/RTF/resource metrics
retain their own definitions. Null output-dependent quality after failure is read
with retained failure and support counts, not mistaken for zero error.

Spoken and visible content are separate. Combined audio/visual usefulness is
checked through the non-locked
[extraction-to-RAG confirmation](../../rag/final/methodology.md#extraction-to-rag-confirmation),
not a transcript-dominated combined recall score. An integrated confirmation
measures actual full-path latency/RTF/resources; adding component percentiles or
peaks cannot reconstruct that measurement.

## MLflow result structure

Video uses `EduMind / Video`. One parent groups audio preparation and the visual
comparison. Display names omit timestamps and benchmark prefixes; unique run IDs,
recorded timestamps, protocol identities, and artifacts retain provenance.

```text
EduMind / Video
├── smoke-cpu
│   ├── frozen-asr — <selected candidate>
│   └── visual smoke candidates
├── smoke-cuda
│   ├── frozen-asr — <selected candidate>
│   └── visual smoke candidates
├── preflight
│   ├── frozen-ASR qualification
│   └── visual-policy qualification children
├── development-scene-selection
│   ├── frozen-asr — <selected candidate>
│   ├── scene-0.30
│   ├── scene-0.40
│   └── scene-0.50
├── development
│   ├── frozen-asr — <selected candidate>
│   ├── fixed-5s
│   ├── fixed-10s
│   ├── fixed-20s
│   ├── scene-0.30
│   ├── scene-0.40
│   ├── scene-0.50
│   ├── hybrid-0.40-5s
│   ├── hybrid-0.40-10s
│   └── hybrid-0.40-20s
├── validation
│   ├── frozen-asr — <selected candidate>
│   └── selected visual finalists
└── locked
    ├── frozen-asr — <selected candidate>
    └── one selected visual configuration
```

The hybrid 0.40 is illustrative, not an authoritative winner. Only qualified
policies appear. Refreshed preflight uses another parent with the same concise
name and a different run ID/checksum; names are not lookup keys.

The frozen-ASR child is tagged `run_type=input_preparation`, not a visual
competitor. Visual candidates are direct siblings; a locked parent has one
visual competitor despite its preparation child. The parent separates preparation
readiness from visual candidate counts in its completion summary.
`development-scene-selection` uses `profile=development` with a distinct
comparison-purpose tag. No extra fixed/scene/hybrid comparison parents are needed
for the main shared-corpus comparison.

Every child records complete executed settings, model/component identities,
device/dtype, seed, warmups/repetitions, manifest/protocol checksums, qualification,
and decision provenance. Visual display aliases such as `fixed-5s` do not rename
canonical IDs such as `video-fixed-5s`; decisions use canonical IDs.

Visual children log:

```text
visual_content_precision
visual_content_recall
visual_content_f1
timed_visual_occurrence_coverage
mean_visual_first_detection_delay_seconds
duplicate_visual_text_rate
repeatability_success_rate
attempt_failure_rate

visual_real_time_factor
p50_warm_visual_latency_seconds
p95_warm_visual_latency_seconds
cold_visual_pipeline_load_seconds
peak_visual_process_tree_ram_mb
peak_visual_vram_mb
mean_selected_frames_per_video
```

Frozen-ASR child metrics are separate:

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
attempt_failure_rate

real_time_factor
p50_warm_audio_processing_latency_seconds
p95_warm_audio_processing_latency_seconds
cold_model_load_seconds
peak_process_tree_ram_mb
peak_vram_mb
```

Each visual child stores `samples.parquet` with one row per scheduled video,
`timings.parquet` with every scheduled attempt, and `candidate.json` with
resolved settings, values/statuses/reasons, support, and interval fields. Retain
per-attempt frame counts and canonical output fingerprints. Quality is not
rescored from later outputs. The frozen-ASR child retains its checksummed
artifact, per-video completion/failure records, raw edits/reference counts,
reviewed nonspeech-unit records, window outcomes/timings, and resource samples.
Do not create a separate child run for each video, window, metric, or repetition.

Applicable sample-based aggregates use the
[shared metric suffix convention](../../methodology.md#shared-mlflow-metric-convention).
Bootstrap complete source groups 10,000 times with seed 42; windows and repeated
visual attempts are not extra independent samples. Supported defined visual
quality/reliability and frozen-ASR aggregates receive intervals. p50/p95 support
is checked separately; insufficient support retains descriptive values and null
bounds. Cold load and observed resource peaks do not receive intervals.
All nulls remain in JSON/Parquet even when MLflow scalar keys are absent.

Every decoding/selection step records the FFmpeg version and exact argument
vector. Raw source videos are not duplicated into candidate artifacts.
Historical runs keep their original names, definitions, and protocol identity.
