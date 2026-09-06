# Temporary benchmark data-review checklist

[Benchmark overview](overview.md) · [Dataset guide](datasets.md) ·
[Methodology](methodology.md)

This is a temporary checklist for decisions that cannot be made honestly before
the candidate datasets are downloaded and inspected. It is not an experiment
contract and must not be used to justify a result. Resolve each item by updating
the authoritative dataset manifests and methodology, then remove the item.

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

- Inspect the chosen clips before deciding whether condition labels such as
  `clean`, `noisy`, `accented`, and `multi_speaker` are single-valued or
  multi-label. A noisy accented clip must not be forced into a false category.
- Determine the available count of independent silence, music, background-noise,
  and environmental-sound controls per split. Do not promise reliability
  confidence intervals until the counts support them.
- Publish one complete speech-manifest row and one reliability-manifest row with
  real source revisions, clip boundaries, and checksums.

## Video extraction

- Inspect actual video durations before freezing the long-video ASR window
  overlap and transcript-stitching policy.
- Freeze the one-to-one text-matching rule used to decide whether a selected
  frame captured a timed visible-text occurrence. Inspect the annotations first
  so the rule is neither too permissive for short labels nor too strict for long
  slide text; do not tune it on validation or locked-test results.
- Confirm that SlideSpeech and the other selected sources are downloadable under
  the recorded terms and that the chosen assets can be checksum-pinned.
- Verify how many independent validation and locked videos are available before
  treating p95 latency or bootstrap intervals as stable evidence.
- Define the annotation labels needed to diagnose slide, screen-recording,
  presenter, gradual-change, and repeated-scene behavior from the real corpus.
- Publish one complete video-manifest row after the source interval, visible-text
  timestamps, transcript, license, and checksum have been verified.

## Resolution rule

For each item, record the inspected corpus revision, observed counts, the chosen
rule, and the reason. Dataset reality may change the manifest design or reporting
slices; it must not silently change model outputs, metric definitions, or prior
results.
