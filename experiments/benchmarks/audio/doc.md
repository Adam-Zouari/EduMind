# Audio-to-text experiment

## Question, candidates, and control

Which local English ASR profile minimizes transcription/timestamp error while meeting memory and latency gates? Compare OpenAI Whisper `small.en` with faster-whisper `tiny.en`/`base.en`/`small.en` int8, `small.en` float16, and `turbo` int8. The reference Whisper implementation is the control. Turbo is a quality ceiling until it proves deployment feasibility. VAD off/on is a second factor only for shortlisted profiles.

## Dataset, splits, and procedure

The licensed 90-clip manifest is split 54/18/18 by recording and stratified for clean speech, noise, accents, technical vocabulary, and multiple speakers. It freezes IDs, source/license/revision, checksums, transcript/timestamps, preprocessing, and split seed. Candidate order is seeded; each profile gets a cold-load measurement, two warmups, and measured repetitions through the production audio extractor. Decoding parameters and model revision are fixed.

## Metrics and rationale

Primary quality is Word Error Rate `word edit distance / reference words` and Character Error Rate `character edit distance / reference characters`; both range from 0 upward and lower is better. Timestamp MAE is `mean(abs(predicted-aligned_reference))` seconds, and Segment Boundary Error is mean absolute start/end displacement; lower is better. Missing/Hallucinated Speech Rate are missing or unsupported aligned units divided by reference or prediction units. Real-Time Factor is processing seconds/audio seconds; below 1 is faster than real time. p50/p95 clip latency, cold load, RAM, and VRAM are operational diagnostics, not quality votes.

## Statistics, promotion, and artifacts

Per-clip results are persisted before aggregation. The harness reports fixed-seed 95% bootstrap intervals and Pareto candidates after crash, memory, and determinism gates. Overlapping quality intervals prefer lower p95, memory, then storage. `plan.json`, provenance, candidate samples, `summary.json`, `_SUCCESS.json`, and optional nested MLflow runs are written under the run directory.

## Commands, example, and limitations

```powershell
edumind benchmark --profile smoke extraction audio
edumind benchmark --profile standard extraction audio
edumind benchmark --profile full extraction audio
```

Example: RTF 0.6 does not compensate for a statistically meaningful WER regression from 0.10 to 0.18. Smoke only validates execution. The result is English-only and cannot establish diarization, translation, semantic correctness, or production quality for unrepresented microphones.
