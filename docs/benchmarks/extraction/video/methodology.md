# Video extraction methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Candidates](#candidates)
- [Data](#data)
- [Execution](#execution)
- [Metrics and why they are used](#metrics-and-why-they-are-used)
- [MLflow result structure](#mlflow-result-structure)

With the document parser and ASR fixed, which keyframe policy recovers useful
on-screen text without processing too many duplicate frames?

## Candidates

The experiment compares three frame-selection strategies, but it does not test
only one arbitrary setting for each strategy. Development produces nine
configurations:

| Strategy | Development configurations | Question answered |
|---|---|---|
| Fixed interval | One frame every 5, 10, or 20 seconds | How much visual coverage is gained by sampling more frequently, and what does that coverage cost? |
| Scene change | FFmpeg scene threshold 0.30, 0.40, or 0.50 | How sensitive should transition detection be before extra frames become mostly redundant? |
| Hybrid | The selected scene threshold plus a maximum gap of 5, 10, or 20 seconds | How frequently must the fallback sample gradual or static scenes that never produce a strong transition? |

Every configuration includes the first frame. This protects titles, opening
slides, and initial screen state even when no early scene transition occurs.

These are nine configurations of three strategies, not nine unrelated
strategies. The hybrid configurations use the scene threshold selected from the
scene-change comparison. Testing every scene threshold with every maximum gap
would produce 15 configurations and answer an additional interaction question
that is not required in the first benchmark.

## Data

Smoke uses two committed videos. The authoritative set contains 30 educational
videos split 18 development, 6 validation, and 6 locked. Every video has a
verified transcript, duration, visible text, and visual timestamps. The set
includes slides, screen recordings, presenter video, gradual text changes, and
repeated scenes. Exact sources, revisions, licenses, clip intervals, and checksums
are frozen in the manifests. The [dataset guide](../../datasets.md) defines the
SlideSpeech, AVLectures, and EduMind-owned allocation and explains why public
subtitles and OCR must be manually corrected rather than accepted as ground
truth.

Use the [shared data-validation workflow](../../data-validation.md) for reviewed
transcripts, visual units/intervals, duration, source isolation, and development
stress inputs before execution. The proposed validator/report interface remains
pending implementation. Frozen-ASR artifact checksum validation is a separate
runtime integrity gate; a validated manifest alone cannot certify that artifact.

## Execution

The dedicated runner supports smoke, development, validation, and locked
profiles. Its normal `protocol.yaml` freezes the candidate ranges, frame-zero
and VFR rules, ASR windows and overlap, deterministic normalized suffix/prefix
stitching, visible-text unitization, occurrence matching, dataset counts, and
execution profiles. Manifests remain ordinary data inputs whose checksums are
recorded in run provenance and in the frozen-ASR artifact; there is no separate
manifest-bound video lock.

The frozen-ASR phase composes the video and ASR protocols. Each visual phase
also composes the document protocol so the selected image parser receives the
same resolved parser options and candidate-factor validation used by the
document benchmark. The visual worker verifies both protocol payloads before
initializing the parser.

The authoritative hybrid threshold and its development source run ID begin as
`null`. Fixed and scene development comparisons can run in that state. After
reviewing the scene comparison, an engineer writes the selected declared
threshold and source run ID into `protocol.yaml` and increments the protocol
version. Hybrid execution rejects missing or out-of-range values. Because this
changes the protocol checksum, the development frozen-ASR artifact must be
regenerated and hardware qualification refreshed before the final combined
comparison. Before threshold freezing, preflight covers the frozen ASR and
executable fixed/scene policies. After freezing, it covers the complete declared
nine-policy roster under the new checksum. Development compares the qualified
policies; hardware exclusions remain in provenance. The runner never writes the
threshold or application configuration itself.

```text
video frozen-ASR phase
└─ FFmpeg mono 16 kHz windows → selected ASR once → FrozenASRArtifact

video visual phase
└─ candidate FFmpeg keyframes → frozen selected document parser
   → visible-text and timing metrics
   → reference the FrozenASRArtifact checksum; never invoke ASR
```

Only the keyframe configuration changes. The selected ASR transcribes each
window once in the frozen-ASR phase; stitched per-video outputs and measurements
are frozen
as an upstream artifact. Every keyframe child references that artifact by run ID
and checksum instead of retranscribing the videos. Reopening parser or ASR
selection here would make it unclear whether a difference came from frame
selection, visual parsing, or speech recognition.

Video audio may be longer than the ASR benchmark's 30-second single-clip limit.
The selected ASR therefore receives deterministic windows no longer than 30
seconds. Window-local timestamps are shifted back onto the video timeline and
overlapping text is stitched once. The exact overlap and stitching rule are
frozen before the video comparison. This qualifies the serving policy of the
already selected ASR; it does not reopen the four-profile ASR comparison.

Development proceeds in this order:

1. Run the fixed-interval configurations at 5, 10, and 20 seconds.
2. Run the scene-change configurations at thresholds 0.30, 0.40, and 0.50.
3. Review visual quality and processing cost, then record one scene threshold
   and its source run ID in `protocol.yaml`; increment the protocol version and
   refresh preflight and regenerate the frozen-ASR artifact.
4. Combine that threshold with maximum gaps of 5, 10, and 20 seconds.
5. Run all qualified members of the nine-configuration development comparison
   under this final protocol checksum, including fresh fixed/scene results
   alongside the hybrids.
   The earlier scene run supplies the threshold-freezing evidence, not mixed-
   protocol comparison scores. Record finalists from the completed comparison;
   do not calculate an automatic overall score.
6. Run only the engineer-selected finalists on validation. Run one selected
   configuration once on the locked test.

The numerical settings are initial development search points, not universal
constants. Five seconds is the high-coverage/high-cost interval, 20 seconds is
the low-cost/low-coverage interval, and 10 seconds is the midpoint. FFmpeg's
[scene-filter guidance](https://ffmpeg.org/pipermail/ffmpeg-cvslog/2012-June/051105.html)
defines the score on a 0-to-1 scale and identifies roughly 0.3 to 0.5 as a
practical range; 0.30, 0.40, and 0.50 sample that range without a large grid.
The hybrid gaps reuse the fixed-interval values so the fallback cost can be
compared directly with the fixed strategy. The selected values are valid only
for the recorded educational-video corpus.

## Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | Visual Content F1; Mean Visual First-Detection Delay with Timed Visual Occurrence Coverage | Measures whether useful on-screen text was recovered, whether it was captured while visible, and how quickly it was first captured. Delay and coverage must be interpreted together. |
| Secondary | Visual Content Precision/Recall | Explains whether a low F1 came from unsupported extracted text or missed visible text. |
| Diagnostic | Duplicate Visual Text Rate | Shows whether repeatedly selected unchanged frames duplicate the same content. |
| Diagnostic | Frozen-ASR Transcript WER, recorded once for the shared ASR output | Confirms the audio input to every policy; it is not used to compare keyframe policies because it is constant. |
| Operational | Visual Real-Time Factor, p50/p95 warm visual latency, cold visual-pipeline load time, peak visual process-tree RAM, peak visual device VRAM | Measures the keyframe and visual-parser cost that differs between configurations. |
| Workload descriptor | Mean selected frames per video | Records how much visual input each policy sends to the parser without treating fewer frames as inherently better. |

Spoken and visible tokens remain separate. A video's transcript usually
contains far more words than its frames, so one combined recall value would be
dominated by audio and could hide a poor keyframe policy. The isolated video
comparison therefore scores visible content and its timing. Downstream
usefulness of combined audio and visual content requires the separate
[extraction-to-RAG confirmation](../../rag/final/methodology.md#extraction-to-rag-confirmation)
on paired, reviewed material; it is not measured by the keyframe comparison.

## MLflow result structure

Video uses `EduMind / Video`. Smoke creates separate CPU and CUDA parents.
Initial preflight and frozen ASR support the fixed/scene development screen.
After the scene threshold is frozen, the new protocol checksum requires fresh
qualification and frozen ASR. One final comparison then evaluates all qualified
members of the nine-policy roster under that checksum:

```text
MLflow experiment: EduMind / Video
├── parents: video-frozen-asr-smoke-cpu/cuda-<timestamp>
├── parents: video-visual-smoke-cpu/cuda-<timestamp>
├── parent: video-preflight-<timestamp>
│   └── frozen-ASR qualification plus one child per visual policy
├── parent: video-frozen-asr-development-<timestamp>
│   └── child: <selected-asr-profile-across-all-development-videos>
├── parent: video-visual-development-fixed-<timestamp>
│   ├── child: video-fixed-5s
│   ├── child: video-fixed-10s
│   └── child: video-fixed-20s
├── parent: video-visual-development-scene-<timestamp>
│   ├── child: video-scene-0.30
│   ├── child: video-scene-0.40
│   └── child: video-scene-0.50
├── parent: video-preflight-<timestamp> [threshold frozen; new checksum]
│   └── frozen-ASR qualification plus the nine declared visual policies
├── parent: video-frozen-asr-development-<timestamp> [new checksum]
│   └── child: <selected-asr-profile-across-all-development-videos>
├── parent: video-visual-development-all-<timestamp> [new checksum]
│   └── up to nine children: qualified fixed, scene, and hybrid configurations
├── parent: video-visual-validation-<timestamp>
│   └── one child per engineer-selected finalist
└── parent: video-visual-locked-<timestamp>
    └── one child for the selected configuration
```

Each child is one complete keyframe configuration evaluated on every video in
that phase. It is not split into child runs for individual videos or metrics.
Each child records its complete resolved runtime profile: strategy and numerical
settings, parser revision and settings, device, seed, FFmpeg version and command,
warmups and repetitions, dataset checksum, plus the frozen ASR run ID and
artifact checksum. Values shared with the parent are repeated deliberately so
the child remains interpretable when exported alone. Validation and
locked phases follow the same pattern: one phase-specific frozen-ASR input run,
then the visual-policy comparison run.
The child logs these aggregate metrics:

```text
visual_content_precision
visual_content_recall
visual_content_f1
mean_visual_first_detection_delay_seconds
timed_visual_occurrence_coverage
duplicate_visual_text_rate

visual_real_time_factor
p50_warm_visual_latency_seconds
p95_warm_visual_latency_seconds
cold_visual_pipeline_load_seconds
peak_visual_process_tree_ram_mb
peak_visual_vram_mb
mean_selected_frames_per_video
```

The frozen-ASR child logs `word_error_rate` and the ASR component's
actual per-video latency, RTF, RAM, and VRAM measurements. These values describe
shared upstream audio work and are not copied into every visual child. Each
visual child stores per-video
quality and timing rows in `samples.parquet`, per-repetition timings in
`timings.parquet`, and the complete aggregate result in `candidate.json`.
In development, validation, and locked runs, Visual Content Precision/Recall/F1, Timed
Visual Occurrence Coverage, Duplicate Visual Text Rate, Visual Real-Time Factor,
and Mean Selected Frames per Video receive video-bootstrap intervals. Mean
Visual First-Detection Delay receives an interval over videos with covered timed
occurrences. Warm p50
and p95 latency receive intervals when enough independent videos support the
percentiles. The frozen-ASR child's Transcript WER receives an interval from
the same videos. Cold load and observed RAM/VRAM peaks do not receive fabricated
intervals.

Frozen-ASR operational measurements and visual-child measurements remain
separate because adding aggregate percentiles or memory peaks would not recreate
a real pipeline measurement. After the keyframe policy is selected, one
integrated confirmation run measures actual end-to-end latency, RTF, RAM, and
VRAM for the complete video path.

For every audio or video decoding step, the run records the FFmpeg version and
the exact argument vector used. This belongs in the run plan/provenance
artifacts, not only in documentation, because installed codecs and command
options can change the decoded input.

The selected keyframe policy joins the selected parser and ASR as the provisional
video-extraction profile.
