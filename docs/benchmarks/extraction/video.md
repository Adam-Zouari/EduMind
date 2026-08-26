# Video extraction benchmark

[Benchmark overview](../overview.md) · [Benchmark manual](../methodology.md) ·
[Candidate selection](../model-selection.md) · [Preparation guide](../../setup/installation.md)

Which deterministic keyframe policy adds useful visible text to a frozen ASR transcript at acceptable cost? Compare fixed interval, scene change, and scene change plus a maximum-interval fallback. The engineer-selected document parser and ASR are supplied through prior decision files; video does not reopen those candidate choices.

The 30-video target corpus is split 18/6/6 by video and pins transcript, speech timestamps, visible text/timestamps, licenses, checksums, preprocessing, and duration. FFmpeg extracts media into temporary paths that are always cleaned.

Primary metrics are Transcript WER, Visual Text Precision/Recall/F1, Audio/Visual Alignment MAE, and Complete Content Recall. Diagnostics include duplicate visual text and timestamp coverage. Operational metrics are Real-Time Factor, cold load, p50/p95 video latency, throughput, RAM, VRAM, and temporary disk.

Complete Content Recall is the main metric in MLflow; it is a navigation aid, not an automatic winner rule.

```powershell
python experiments/benchmarks/extraction/video/run.py --profile smoke
python experiments/benchmarks/extraction/video/run.py --profile standard --document-selection DOCUMENT_DECISION_JSON --audio-selection AUDIO_DECISION_JSON
python experiments/benchmarks/extraction/video/run.py --profile full --shortlist VIDEO_DECISION_JSON --document-selection DOCUMENT_DECISION_JSON --audio-selection AUDIO_DECISION_JSON
```

The experiment measures visible text, not general diagram or object understanding.
