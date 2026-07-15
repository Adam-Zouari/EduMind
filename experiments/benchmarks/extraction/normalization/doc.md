# Text-normalization benchmark

## Question and candidates

How much repair can be applied without damaging legitimate content? Compare minimal Unicode/line-ending normalization, conservative whitespace/dehyphenation repair, and aggressive cleanup.

## Data and procedure

At least 200 frozen corruption and preservation cases are required for standard/full. Each contains `observed` and verified `reference` text, provenance, checksum, and split family. Seed 42 fixes order. Standard/full run every case three times; repeat equality is determinism. This experiment needs no media asset or model download.

## Metrics

Content Preservation Recall is normalized reference-token recall after normalization. Let `B = edit_distance(reference, observed)`, `A = edit_distance(reference, normalized)`, `C = edit_distance(observed, normalized)`, and `R = max(0, B - A)`. Corruption Removal Recall is `R/B` (1 when `B=0`), precision is `R/C` (1 only when no corruption and no change), and F1 is their harmonic mean. These edit-repair metrics are bounded 0-1 and higher is better. CER/WER, content precision/recall/F1, missing/hallucinated text, repeated-line rate, determinism, p50/p95 latency, and cases/minute are also stored.

The formula is intentionally conservative: changes that do not reduce reference edit distance receive no credit. It does not label individual semantic deletions, so no separate accidental-deletion or merge metric is claimed without span annotations.

Standard/full produce per-case 95% bootstrap intervals and use preservation/determinism gates before Pareto selection.

## Commands and limits

```powershell
python experiments/benchmarks/extraction/normalization/run.py --profile smoke --no-mlflow
python experiments/benchmarks/extraction/normalization/run.py --profile standard
python experiments/benchmarks/extraction/normalization/run.py --profile full --shortlist SUMMARY_JSON
```

Normalization quality alone does not prove downstream retrieval or answer improvement.

Artifacts are plan/provenance JSON, candidate JSON, per-case Parquet, summary intervals/comparisons, and local MLflow runs. Example: aggressive cleanup that removes more corruption but lowers preservation recall is not automatically preferred.
