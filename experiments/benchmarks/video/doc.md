# Video-to-text experiment

## Question, candidates, and control

Which deterministic keyframe policy adds useful visible text to the selected ASR transcript at acceptable cost? With FFmpeg fixed, compare fixed-interval frames (control), scene-change frames, and scene-change plus maximum-interval fallback. Each uses the promoted ASR and image profiles.

## Dataset, splits, and procedure

The licensed 30-video educational corpus is split 18/6/6 by video and stores verified transcripts, speech timestamps, visible text/timestamps, licenses, revisions, checksums, and preprocessing version. Candidate order, FFmpeg options, scene threshold, maximum interval, and component revisions are frozen. Cold load, warmups, repetitions, temporary disk, and cleanup are measured through the production video extractor.

## Metrics and rationale

Transcript WER is word edit distance/reference words (0 upward, lower). Timestamp MAE and Audio/Visual Timeline Alignment Error are mean absolute seconds (lower). Visual Text Precision/Recall/F1 use matched visible-text units (0–1, higher). Duplicate Visual Text Rate is repeated recovered units/all recovered units (0–1, lower). Complete Content Recall is recovered annotated audio-or-visual facts/all facts (0–1, higher). Real-Time Factor, p50/p95 latency, memory, and temporary disk are operational.

## Statistics, promotion, and artifacts

Per-video paired bootstrap intervals and resource gates precede Pareto selection. The hybrid policy advances only when visual/complete recall improvement justifies duplication and latency. Artifacts contain plan/provenance, per-video observations, candidate aggregates/intervals, Pareto set, success marker, and optional nested MLflow runs.

## Commands, example, and limitations

```powershell
edumind benchmark --profile smoke extraction video
edumind benchmark --profile standard extraction video
edumind benchmark --profile full extraction video
```

Example: 0.05 extra visual recall with 0.30 duplicate rate and doubled p95 may be dominated. This does not evaluate object recognition, diagrams, formulas, arbitrary visual understanding, or non-English speech.
