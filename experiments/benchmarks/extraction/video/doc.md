# Video-to-text benchmark

## Question and candidates

Which deterministic keyframe policy adds useful visible text to ASR at acceptable cost? With FFmpeg and the selected ASR/OCR profiles fixed, compare fixed-interval, scene-change, and scene-change plus maximum-interval fallback.

## Data and procedure

The target 30-video corpus is split 18/6/6 by video and pins transcript, speech timestamps, visible text/timestamps, licenses, checksums, preprocessing, and duration. Seed 42 fixes order. Standard/full use one cold invocation, two warmups, and three repetitions. The production extractor labels audio and visual segment counts, enabling separate transcript and visible-text scoring; temporary media is automatically cleaned.

## Metrics

Transcript WER is computed only from audio segments. Visual Text Precision/Recall/F1 use normalized token overlap against `reference_visual_text`. Complete Content Recall scores the combined extracted document. Duplicate Visual Text Rate measures repeated non-empty visual lines. Timestamp MAE and Timestamp Alignment Coverage assess available speech annotations; Audio/Visual Alignment MAE is emitted only for equal-length visual timestamp annotations. All quality rates range 0-1 and are higher-better except duplicate rate; timing and WER are lower-better. Real-Time Factor, cold invocation, p50/p95 latency, videos/minute, RAM, and VRAM are operational.

Missing annotations cause the dependent metric to be omitted, not replaced with zero. Standard/full retain per-video observations and 95% bootstrap intervals; selection uses separate Pareto objectives.

## Commands and limits

```powershell
python experiments/benchmarks/extraction/video/run.py --profile smoke
python experiments/benchmarks/extraction/video/run.py --profile standard --image-summary IMAGE_SUMMARY_JSON --audio-summary AUDIO_SUMMARY_JSON
python experiments/benchmarks/extraction/video/run.py --profile full --shortlist SUMMARY_JSON --image-summary IMAGE_SUMMARY_JSON --audio-summary AUDIO_SUMMARY_JSON
```

This does not evaluate diagrams, objects, formulas, general visual understanding, or non-English speech.

Artifacts are plan/provenance JSON, candidate JSON, per-video Parquet, summary intervals/comparisons, and local MLflow runs. Example: a 0.03 visual-recall gain can be dominated when duplicate visual text and p95 latency both rise materially.
