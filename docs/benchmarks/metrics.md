# Shared benchmark metric conventions

[Benchmark program](../README.md#experiments) · [Experiment sequence and rationale](methodology.md) ·
[Benchmark runbook](running.md)

This page defines the shared measurement and reporting conventions for EduMind's
benchmarks. The benchmark pages below define **what each value means and how it
is calculated**; their methodology pages explain **where each metric is used and why**.
A development, validation, or locked result is authoritative only when its runner implements the applicable contract
exactly and records every required value. Higher is better unless a metric is
marked lower-is-better.

## Benchmark metrics

The shared rules on this page apply alongside each benchmark's specific contract.

| Benchmark | Methodology | Metrics |
|---|---|---|
| Document extraction | [Methodology](extraction/document/methodology.md) | [Metrics](extraction/document/metrics.md) |
| Audio extraction | [Methodology](extraction/audio/methodology.md) | [Metrics](extraction/audio/metrics.md) |
| Video extraction | [Methodology](extraction/video/methodology.md) | [Metrics](extraction/video/metrics.md) |
| Chunking and embedding | [Methodology](rag/chunking_embedding/methodology.md) | [Metrics](rag/chunking_embedding/metrics.md) |
| Retrieval and reranking | [Methodology](rag/retrieval_reranking/methodology.md) | [Metrics](rag/retrieval_reranking/metrics.md) |
| Generation | [Methodology](rag/generation/methodology.md) | [Metrics](rag/generation/metrics.md) |
| Vector databases | [Methodology](vectordb/methodology.md) | [Metrics](vectordb/metrics.md) |
| Final RAG and human review | [Methodology](rag/final/methodology.md) | [Metrics](rag/final/metrics.md) |

## Shared conventions

- Prose comparison uses one symmetric, evaluation-only projection on both the
  reference and prediction: Unicode NFC, case-folding, replacement of Unicode
  punctuation with spaces, and whitespace collapse. The resulting whitespace-
  separated tokens are used by prose Content metrics; prose CER and WER
  operate on the same projected strings. Raw outputs remain unchanged in
  artifacts. The projection does not dehyphenate words,
  correct spelling, rewrite numbers, remove headers, or alter formulas, code,
  layout trees, or table trees.
- Source and evidence spans are half-open intervals: `[start, end)`.
- Empty denominators use the explicit behavior stated in each benchmark's metric contract; they never produce
  fabricated zero-quality observations.
- A valid empty comparison can have the contract's best or worst numeric value.
  A failed request is not an observed empty output. `null` is a storage value:
  retain `inapplicable`, `unavailable`, or `incomplete` status and its reason.
  Exclude an undefined value only from that metric's numeric calculation, never
  from retained sample/attempt records or other defined metrics. Report scheduled,
  eligible, contributing, failed, and unavailable counts. MLflow omits null scalar
  keys; artifacts preserve them. Invalid references fail
  [data validation](data-validation.md) before candidate execution.
- Development, validation, and locked runs retain one row per sample before aggregation.
- p50, p95, and p99 are latency percentiles. Each throughput metric specifies
  which completed work and timed interval it uses.
- Eligible development, validation, and locked sample-based aggregates use 10,000
  bootstrap resamples with seed 42 and 95% confidence intervals. Counts,
  statuses, fixed identifiers, and single operational observations do not
  receive intervals.
- A bootstrap draw preserves each independent unit with all its dependent
  observations: a document and its pages, a speech clip, a video, a source paper
  and its questions, or a vector-database query within its workload cell.
  Related clips or videos from one original source are grouped when the reviewed
  manifest declares that source as the independent unit. Repetitions do not
  increase the independent sample count. Each draw recalculates the metric's
  own aggregate, preserving its documented macro-average, pooled-count ratio,
  percentile, or other definition rather than substituting another statistic.
  Paired comparisons use the same sampled units for both candidates.
- Conditional intervals record defined and undefined resample counts. A draw
  undefined for one metric still contributes to other defined metrics. If the
  frozen minimum independent count is not met, retain the point estimate and
  record null confidence bounds and their reason in artifacts; MLflow scalar
  bounds are absent. The minimum is fixed after data review, before evaluation.
- Normalized precision, recall, F1, accuracy, coverage, nDCG, and correctness
  values lie in `[0, 1]`. CER and WER are non-negative and can exceed 1 when
  insertions outnumber reference units. Human rubric scores use their stated
  `0–2` or `0–1` scales. Time, memory, storage, and throughput are non-negative
  and have no fixed upper bound.

CUDA **Peak Device VRAM** is the largest sampled total `memory.used` reported by
NVML for the assigned GPU, from before candidate loading through the final
inference. It includes driver, desktop, and other process allocations; it is
neither a per-process estimate nor a baseline-subtracted increase. Record the
measurement method (`nvml-device-total`), GPU identity, idle baseline, total/free
memory, and resource samples. The baseline is explanatory and is not deducted.
GPU background activity must remain controlled across candidates. Missing
required CUDA telemetry invalidates the measurement rather than producing zero.
A confirmed CPU-only zero-VRAM report is an execution marker, not a measurement
of the machine's GPU. Peak Process-Tree RAM continues to measure the worker and
its children.

## Aggregation and interpretation

Aggregate metrics never replace sample rows. Development, validation, and locked
reports retain each metric's defined aggregate: a macro-average, pooled-count
ratio, named percentile, or other specified statistic. Report contributing and
independent sample counts, failures, result status, and a 95% interval when
eligible and sufficiently supported. Inapplicable or unavailable values retain
their reasons; they do not silently become numeric zeros. Smoke values, counts,
statuses, fixed identifiers, and single operational observations do not have
authoritative intervals. An engineer reviews the complete evidence; EduMind
does not combine unrelated metrics into a weighted overall score or promote a
candidate automatically.
