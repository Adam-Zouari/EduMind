# Audio-to-text benchmark

## Question and candidates

Which local English ASR profile minimizes transcription and timing error at acceptable cost? Compare OpenAI Whisper `small.en` with faster-whisper tiny/base/small/turbo candidates. Turbo is a quality ceiling until it passes resource and latency gates.

## Data and procedure

The target corpus is 90 clips split 54/18/18 by recording across clean speech, noise, accents, technical terms, and multiple speakers. Manifests pin transcript, timestamps, license, source revision, checksum, duration, and seed. Candidate order uses seed 42. The first call measures lazy load, two calls warm the runtime, and standard/full measure each clip three times with fixed decoding/model settings.

Tiny/base int8 run on CPU; OpenAI small, faster-whisper small, and turbo run on CUDA. A manifest may explicitly override `device`, which becomes part of the frozen plan. After standard selection, the full runner expands every shortlisted profile into VAD off/on candidates instead of hiding VAD in an undocumented option.

## Metrics

- WER and CER are Levenshtein error counts divided by reference word/character counts; lower is better.
- Missing and hallucinated speech use normalized unmatched transcript tokens; lower is better.
- Timestamp MAE is mean absolute start-time error in seconds when one-to-one timestamp annotations exist. Timestamp Alignment Coverage reports matched count / larger count when counts differ.
- Segment Boundary MAE is the mean absolute start/end error only when the manifest supplies aligned boundaries and the predicted count matches. It is omitted otherwise, not fabricated.
- Real-Time Factor = measured processing seconds / annotated audio seconds. Below 1 is faster than real time.
- Operations: cold invocation, p50/p95 clip latency, clips/minute, process RAM, and VRAM when available. Determinism records identical transcripts across repetitions.

Standard/full store per-clip results and 10,000-resample 95% intervals before Pareto selection. Smoke is path validation only.

## Commands and limits

```powershell
python experiments/benchmarks/extraction/audio/run.py --profile smoke
python experiments/benchmarks/extraction/audio/run.py --profile standard
python experiments/benchmarks/extraction/audio/run.py --profile full --shortlist SUMMARY_JSON
```

This does not evaluate diarization, translation, semantic correctness, or unrepresented microphones/languages.

Artifacts are plan/provenance JSON, candidate JSON, per-clip Parquet, summary intervals/comparisons, and local MLflow runs. Example: RTF 0.6 cannot compensate for a clearly worse non-overlapping WER interval.
