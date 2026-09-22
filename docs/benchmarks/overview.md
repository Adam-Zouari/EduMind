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
- One smoke command runs independent CPU and CUDA paths where supported; neither
  path may fall back to the other device.
- `preflight` qualifies every declared candidate on the target CUDA hardware.
  It produces hardware evidence, not quality evidence.
- `development` compares only candidates qualified by the exact matching
  preflight. Vector Database, Generation, and Final RAG follow their documented
  exceptions.
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

New runs are grouped into one MLflow experiment per benchmark rather than one
broad extraction or RAG experiment. Each phase has a comparison parent and one
direct child per candidate; qualification reports and decision fingerprints
make every transition traceable.

## Configuration ownership

Every independently executable benchmark has one strict, versioned
`protocol.yaml` beside its runner. The protocol is the only editable source for
settings that can change output, eligibility, timing, memory, or failure status.
Candidate rosters are defined in the protocol where configurable or generated
from the supported adapters in code; no separate candidate registry is used.
`data/benchmarks/models/selected.json` contains immutable model revisions, local
snapshot paths, and checksums; manifests contain data and data provenance; and
`config/base.yaml` contains provisional application settings. Benchmark runners
never rewrite any of these inputs or promote a result automatically.

Each parent fingerprint includes every protocol it composes. The parent stores
the source YAML and a resolved `<name>_protocol.json`; parent and child MLflow
runs record protocol version and checksum. A changed protocol therefore creates
a different run identity and invalidates worker payloads or frozen artifacts
created with the old checksum.

Current implementation limitations that affect whether a run is authoritative
are recorded beside the relevant commands in the [runbook](running.md), rather
than being hidden in stage-specific pages.
