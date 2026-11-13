# Changelog

## Unreleased

### Implementation repair — 2026-11-13

- Preserved completed predictions and partial resource evidence across telemetry,
  placement, cleanup, and worker interruptions; tightened protocol envelopes and
  corrected count-based F1 threshold arithmetic.
- Hardened snapshot/component/evaluator identity, offline tokenizer loading,
  preparation planning, stage-scoped locks, and prepared Tesseract execution.
- Repaired document scorer normalization/count transport, ASR/video evidence
  retention and decoded-PCM schedules, and chunking/retrieval integrity and reporting.
- Added durable generation judging, Citation Support, sealed evaluation-only
  recovery, explicit vector configurations and workload ledgers, and the single
  selected-server Final RAG smoke/preflight/locked workflow.
- Fixed application source provenance, failed-upload retries, extraction warning
  retention, replacement rollback, and generator cancellation/reasoning boundaries.
- Added behavioral and mocked workflow regressions and reconciled implementation
  status in the guides. Real services, reviewed data, calibrated judging, and
  component re-preparation remain separate readiness requirements.
- Completed live pinned-Docker scorer and four-server smoke checks. Repaired
  Qdrant's conflicting transport options and Weaviate's unset metadata and
  server-bounded document deletion; preserved the previous evaluator source
  before publishing a separately verified pinned copy.

### Earlier documentation and architecture changes

- Revised the vector-database documentation contract around explicit database
  configurations, tie-aware ANN Recall@20, application evidence quality,
  operation-specific reliability, visibility-based maintenance timing, and
  scoped MLflow/uncertainty reporting. This was the documentation-stage contract;
  implementation follow-up is recorded above.
- Expanded vector-database metric explanations and documented current runner
  limitations; clarified setup, architecture, and contribution guidance without
  changing benchmark implementation or methodology settings.
- Changed the project license to PolyForm Strict License 1.0.0; third-party
  dependencies, model weights, and datasets retain their upstream licenses.
- Reworked the documentation hierarchy around project purpose, architecture,
  setup, experiment methodology, exact metrics, and one benchmark runbook;
  removed duplicate stage pages and command copies.
- Aligned production with Docling Standard, Whisper `small.en`, GTE ModernBERT, Chroma HTTP,
  and direct pinned Hugging Face Qwen3 1.7B controls.
- Kept experiment implementation, candidates, metrics, and MLflow logging under
  `experiments/benchmarks`, with all reader documentation centralized in `docs/`.
- Moved Streamlit and end-to-end orchestration into the installed `edumind`
  package and separated experiment-only model and strategy implementations.
- Removed the API/service layer, packaged benchmark framework, embedded vector databases, duplicate application modules, recommendation machinery, and broad build/lint/coverage workflows.
- Split application and benchmark dependencies into separate lock files.
- Added direct document and audio extraction benchmarks, RAG benchmarks, and a
  four-server vector-database benchmark with per-sample artifacts and confidence
  intervals. The video strategy code is implemented; authoritative data and
  selection runs remain pending.
- Made the approved selection evidence authoritative for model preparation and removed
  retired candidate registries, model locks, and serving-specific generation code.
