# Normalization experiment

## Question, candidates, and control

How much repair can be applied without deleting or merging legitimate content? Compare minimal Unicode/line-ending normalization (control), conservative encoding/whitespace/dehyphenation/header-footer repair, and aggressive cleanup.

## Dataset, splits, and procedure

At least 200 deterministic cases annotate required content and corrupt spans. Cases cover corruption and preservation counterexamples and are frozen by manifest/checksum; source families remain isolated across development, validation, and locked test. Each profile runs the production normalizer repeatedly in randomized order. Byte-identical repeat output is required.

## Metrics and rationale

Content Preservation Recall is retained required units/reference required units (0–1, higher). Corruption Removal Precision is correctly removed corrupt units/all removed units; recall is correctly removed/all corrupt units; F1 is their harmonic mean (0–1, higher). Accidental Deletion Rate is legitimate units deleted/reference units, and Merge Rate is unintended merges/eligible boundaries (0–1, lower). Determinism is repeated identical outputs/runs (must be 1); latency is operational. Preservation is primary because a cleaner-looking destructive output is unacceptable.

## Statistics, promotion, and artifacts

Per-case results receive fixed-seed bootstrap intervals. Determinism and preservation are hard gates; aggressive cleanup advances only if its preservation interval does not regress from conservative. Remaining candidates use Pareto selection. The run stores plan/provenance, samples, intervals, Pareto results, success marker, and optional nested MLflow tracking.

## Commands, example, and limitations

```powershell
edumind benchmark --profile smoke extraction normalization
edumind benchmark --profile standard extraction normalization
edumind benchmark --profile full extraction normalization
```

Example: deleting a repeated-looking lesson title is an accidental deletion even if output appears cleaner. This component result alone cannot establish better retrieval or answers; extraction-to-RAG confirmation measures downstream effects.
