# Temporary benchmark data-review checklist

[Benchmark overview](overview.md) · [Dataset guide](datasets.md) ·
[Methodology](methodology.md)

This is a temporary checklist for decisions that cannot be made honestly before
the candidate datasets are downloaded and inspected. It is not an experiment
contract and must not be used to justify a result. Resolve each item by updating
the authoritative dataset manifests and methodology, then remove the item.

## Shared confidence intervals

- After the frozen manifests exist, set a documented minimum number of
  independent eligible samples for reporting confidence intervals, especially
  for conditional table, formula, reliability, timestamp, and p95-latency
  results. Repeated measurements of one document, audio clip, or video are not
  additional independent samples.

## Document extraction

- Verify that every conditional metric has enough eligible samples: pages with
  page references, layouts with element boxes/types/hierarchy, tables with cell
  structure, and formulas with detection and transcription references.
- Confirm whether document rows need multiple source/capability labels and make
  those labels consistent across datasets.
- Publish one complete authoritative manifest example after real assets and
  checksums exist.
- Verify the PureDoc source snapshot and checksum before treating it as frozen
  benchmark data.

## Audio extraction

- Verify the reviewed `conditions` lists: every speech clip must be either
  `clean` or `noisy`, while `accented` and `multi_speaker` may coexist with that
  acoustic label.
- Determine the available count of independent silence, music, background-noise,
  and environmental-sound controls per split. Do not promise reliability
  confidence intervals until the counts support them.
- Publish one complete speech-manifest row and one reliability-manifest row with
  real source revisions, clip boundaries, and checksums.

## Video extraction

- Confirm that the reviewed video durations and annotations are compatible with
  the frozen 30-second windows, 2-second overlap, normalized suffix/prefix
  stitching, and one-to-one occurrence threshold in
  `experiments/benchmarks/extraction/video/protocol.yaml`. Any required revision
  must be justified from development data and receive a new protocol version;
  validation and locked-test results cannot tune it.
- After the development scene comparison, write the selected declared scene
  threshold and its source development run ID into video `protocol.yaml`, bump
  the protocol version, and regenerate the frozen-ASR artifact.
- Confirm that SlideSpeech and the other selected sources are downloadable under
  the recorded terms and that the chosen assets can be checksum-pinned.
- Verify how many independent validation and locked videos are available before
  treating p95 latency or bootstrap intervals as stable evidence.
- Define the annotation labels needed to diagnose slide, screen-recording,
  presenter, gradual-change, and repeated-scene behavior from the real corpus.
- Publish one complete video-manifest row after the source interval, visible-text
  timestamps, transcript, license, and checksum have been verified.

## Chunking, embedding, retrieval, and generation

- Inspect the pinned QASPER splits and verify that the prepared 100/40/40 paper
  allocation preserves source-paper isolation and useful answerable/unanswerable
  coverage.
- Build and review the structured supplement independently for development,
  validation, and locked test. Confirm at least ten answerable questions with
  verified evidence for each of `table`, `formula`, and `mixed` in every split.
- Verify every canonical document, accepted answer, answerability label,
  required atomic gold claim, evidence-unit ID, evidence type, and half-open
  source interval before combining a structured manifest with QASPER.
- Build a development-only semantic-judge calibration set with human labels for
  claim support, gold-claim matching, answer relevancy, and pairwise semantic
  equivalence. Freeze the exact judge version and rubric only after it passes the
  recorded agreement criteria; do not use validation or locked-test questions.
- Inspect the resulting question and document counts by evidence type before
  deciding which slice confidence intervals are sufficiently supported.
- Publish one complete authoritative RAG document row and answerable,
  unanswerable, table, formula, and mixed question examples after the real
  manifests and checksums exist.

## Resolution rule

For each item, record the inspected corpus revision, observed counts, the chosen
rule, and the reason. Dataset reality may change the manifest design or reporting
slices; it must not silently change model outputs, metric definitions, or prior
results.
