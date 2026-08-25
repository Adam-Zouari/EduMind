# Audio extraction benchmark

[Benchmark overview](../overview.md) · [Benchmark manual](../methodology.md) ·
[Candidate selection](../model-selection.md) · [Preparation guide](../../setup/installation.md)

## Question and candidates

Which approved English ASR profile best transcribes educational recordings while preserving useful timestamps? Compare Whisper `small.en` (control), Canary 180M, Parakeet TDT 0.6B v2, MOSS Transcribe-Diarize, and Qwen3 ASR 1.7B plus the official 0.6B ForcedAligner.

The shortlist uses the pinned Open ASR English short-form WER table plus verified timestamp support. That public screen does not decide the winner because EduMind needs longer educational speech, citations, resource measurements, and stable timestamps.

## Procedure

The same manifest, device, preprocessing, question order, and repetitions apply to every model. Qwen ASR is unloaded before its forced aligner loads, preventing simultaneous residency. Its sample latency still covers transcription plus alignment, and peak resources cover the complete sequential operation. Canary and Parakeet use their pinned NeMo checkpoints; MOSS uses its official transcription/diarization helper; Whisper uses the pinned Transformers snapshot.

## Metrics

Word Error Rate and Character Error Rate are lower-better edit distances after common normalization. Timestamp Mean Absolute Error measures absolute timestamp deviation in seconds. Segment Boundary MAE measures start/end error. Missing and hallucinated speech rates separate deletions from unsupported output. Timestamp Alignment Coverage prevents a model from appearing accurate by returning only a few timestamps. Operational metrics are Real-Time Factor, p50/p95 clip latency, throughput, cold load, RAM, and VRAM.

```powershell
python experiments/benchmarks/extraction/audio/run.py --profile smoke --device cpu
python experiments/benchmarks/extraction/audio/run.py --profile standard --device cuda
python experiments/benchmarks/extraction/audio/run.py --profile full --shortlist SUMMARY_JSON --device cuda
```
