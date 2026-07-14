# DOCX-to-text experiment

## Question, candidates, and control

Which production DOCX adapter best preserves readable content and simple document order? Compare python-docx (control), Mammoth, Docling, and Unstructured. All candidates return the common `ExtractedDocument`; no experiment-only parser is allowed.

## Dataset, splits, and procedure

The licensed 45-document manifest is split 27/9/9 by source document and covers paragraphs, headings, lists, captions, images, and flattened table/formula text. IDs, licenses, revisions, checksums, annotations, preprocessing, and seed are frozen. Candidate order is randomized with seed 42; imports/cold load, warmups, repeated extraction, normalization, and resource measurements follow the same production path.

## Metrics and rationale

Normalized Exact Match is 1 for equal normalized strings and 0 otherwise. Content Precision is matched annotated units/predicted units; Content Recall is matched/reference units; F1 is their harmonic mean. All range 0–1 and higher is better. Heading/List Structure F1 use their annotated units; Caption and Embedded Image Recall are recovered/reference items; Reading Order Accuracy is correctly ordered annotated pairs/all pairs. These are primary, distinct quality objectives. Empty/Duplicate Output Rate (0–1, lower), p50/p95 latency, and peak memory are diagnostics and gates.

## Statistics, promotion, and artifacts

The harness persists per-document metrics and 95% bootstrap intervals. Failures, nondeterminism, or unacceptable content recall reject a candidate before Pareto selection. Interval ties use p95, memory, then storage. Artifacts are the frozen plan/provenance, candidate samples, summary/Pareto set, success marker, and optional parent/child MLflow runs.

## Commands, example, and limitations

```powershell
edumind benchmark --profile smoke extraction docx
edumind benchmark --profile standard extraction docx
edumind benchmark --profile full extraction docx
```

Example: better heading F1 with missing captions is a tradeoff, not a universal win. Flattened table/formula text is not structural reconstruction. The benchmark does not justify editing fidelity, styling preservation, macros, tracked changes, or non-English support.
