# EduMind benchmark program

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Run commands](running.md)

EduMind benchmarks components before anyone changes the provisional application
defaults. Every run uses explicit candidates, frozen data, per-sample results,
and MLflow provenance. The code records evidence; an engineer chooses what to
advance.

## Read the benchmark documentation by purpose

| Question | Authority |
|---|---|
| What runs first, what each stage compares, which data it uses, and why each metric matters | [Experiment methodology](methodology.md) |
| What each metric means and its exact calculation | [Metric reference](metrics.md) |
| Why each model or server entered the candidate list | [Model-selection rationale](model-selection.md) |
| Which commands to run and what files they consume or produce | [Benchmark runbook](running.md) |
| How to obtain and describe benchmark data | [Benchmark dataset guide](datasets.md) |
| Which model decisions and revisions are machine-readable | [`selection_evidence.csv`](../../experiments/benchmarks/selection_evidence.csv) |
| What was repaired and independently re-audited for document, ASR, and video | [Repair audit](repair-audit.md) |

These documents have deliberately separate roles. The methodology does not
repeat formulas or shell commands, and the runbook does not repeat candidate
rationale.

## Experiment sequence

```text
document parser ─┐
audio ASR ───────┴─> video extraction

chunking × embedding -> retrieval/reranking -> real vector-server retrieval

generation on frozen evidence

selected server + retrieval + generator -> Final RAG -> blinded review
                                           -> extraction-impact confirmation
                                           -> one locked-test evaluation
```

Document extraction, audio, chunking/embedding, vector-server ANN checks, and
generation can begin independently. Downstream experiments use
engineer-authored decision files so an earlier choice is frozen rather than
silently reselected.

## Evidence rules

- `smoke` proves only that a small path executes.
- `development` compares the stage's registered candidates on development data.
- `validation` runs explicit engineer-selected finalists on validation data.
- `locked` runs exactly one frozen selection on locked-test data and is never
  used for tuning.
- Every planned candidate and required metric must complete for a comparison to
  be usable.
- Development, validation, and locked runs retain per-sample rows and report 95% confidence intervals for
  eligible sample-based aggregates. Counts, statuses, fixed identifiers, and
  one-off operational observations do not receive artificial intervals.
- Decisions that require inspecting downloaded corpora are tracked in the
  [temporary data-review checklist](pending-data-review.md), not guessed in the
  benchmark contract.
- No weighted overall score or automatic production promotion is used.
- Performance results apply to the hardware and software environment recorded
  with that run.

Current implementation limitations that affect whether a run is authoritative
are recorded beside the relevant commands in the [runbook](running.md), rather
than being hidden in stage-specific pages.
