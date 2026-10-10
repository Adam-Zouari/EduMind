# Temporary benchmark data-review checklist

[Benchmark program](../README.md#experiments) · [Dataset guide](datasets.md) ·
[Methodology](methodology.md)

This is a temporary checklist for decisions that cannot be made honestly before
the candidate datasets are downloaded and inspected. It is not an experiment
contract and must not be used to justify a result. Resolve each item by updating
the authoritative dataset manifests and methodology, then remove the item.

## Shared confidence intervals

- After the frozen manifests exist, set a documented minimum number of
  independent eligible samples for reporting confidence intervals, especially
  for reference-eligible table/formula tasks, reliability, timestamp, and p95-latency
  results. Repeated measurements of one document, audio clip, or video are not
  additional independent samples.
- Identify dependent pages, capture variants, and excerpts from a common
  original source. Freeze the independent resampling/group IDs in the reviewed
  manifests so all dependent observations travel together in bootstrap draws
  and paired comparisons.

## Document extraction

- Verify that every reference-eligible metric has enough independent sources: pages with
  page references, layouts with element boxes/types/hierarchy, tables with cell
  structure, and formulas with detection and transcription references.
- Verify blank-source and table/formula-negative coverage within the existing
  corpus allocation. Empty reference fields must be verified negatives, not
  substitutes for missing annotations; publish positive/negative and eligible
  document/source counts for every reported metric and slice.
- Freeze compatible element granularity and eligible types across source
  converters, complete reference reading-order sequences, canonical parent-ID
  mapping, box-free non-text object identities, symbol-preserving formula/code
  unitization, coordinate units, and numeric fingerprint precision before
  authoritative execution. Record the canonicalization version; do not adjust
  matching or fingerprint tolerances after observing held-out outputs.
- Check the number of independent sources after grouping related pages and
  clean/degraded captures. Confirm support for document-macro quality,
  three-attempt repeatability/failure intervals, and successful-latency intervals;
  more bootstrap resamples or repeated attempts do not replace more sources.
- Inspect degenerate bootstrap cases, including all-zero/all-one reliability
  values, and ensure equal calculated bounds are flagged rather than described
  as proof of zero uncertainty on future documents.
- Confirm whether document rows need multiple source/capability labels and make
  those labels consistent across datasets.
- Publish one complete authoritative manifest example after real assets and
  checksums exist.
- Verify the PureDoc source snapshot and checksum before treating it as frozen
  benchmark data.

## Audio extraction

- Verify transcript-field presence separately from reviewed lexical emptiness.
  Freeze independent source groups, timed-segment coverage, positive normalized
  reference-word totals, and planned/contributing speech/control counts per
  split. A fully empty speech split is invalid, but legitimate individual empty
  projections remain valid; human review must detect falsely empty annotations.
- Check support for conditional first-output recognition/event rates and boundary
  MAE, reference-denominator timestamp coverage, three-attempt pairwise
  repeatability, and scheduled Attempt Failure Rate. Freeze interval support
  before evaluation; retain undefined-draw and failure counts rather than
  treating missing measurements as zero.

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
  the protocol version, refresh preflight, regenerate the frozen-ASR artifact,
  and run the complete nine-configuration comparison under the final checksum.
- Confirm that SlideSpeech and the other selected sources are downloadable under
  the recorded terms and that the chosen assets can be checksum-pinned.
- Verify how many independent validation and locked videos are available before
  treating p95 latency or bootstrap intervals as stable evidence.
- Freeze consistent visible-line segmentation for reference text and timed
  occurrences under `normalized_lines_distinct_v1`; each timed unit must have
  nonempty comparison text and a verified visibility interval. Do not convert
  reference lines and predicted lines at different granularities.
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
- Review and checksum the authoritative verification material used for factual
  correctness. Keep the required gold claims atomic and non-duplicate; they
  define completeness, while that verification material can establish additional
  correct facts. Neither is supplied as an answer to the generator.
- Review alpha-nDCG ideal-ranking normalization for chunking–embedding and
  retrieval–reranking using development data. Deterministic greedy normalization
  remains the current method. Compare it with exact maximization at `@3` and
  `@5`, with `alpha=0.5`, over the complete frozen chunk corpus rather than only
  the retrieved top-20 pool.
- Record the number of evidence units and distinct evidence-coverage patterns,
  exact computation time with reuse of identical chunk corpora, and score or
  ranking differences between the two methods. Check whether clipping scores at
  one under the greedy denominator hides differences. Use these findings to
  decide whether exact normalization is practical; freeze one method in the
  metric contract and versioned protocol before validation or locked execution.
  Do not mix normalization methods within a comparison or tune the choice on
  held-out results.
- Defer semantic-judge model selection until representative development data has
  been collected and reviewed. Compare a small, cost-aware shortlist against the
  same human-labeled calibration set; neither a model's price nor its general
  benchmark ranking establishes its suitability as a judge. Estimate full-run
  judging costs across generator candidates, modes, repetitions, and semantic
  tasks using observed input/output usage, including billable reasoning and
  retries. Record agreement results, estimated costs, and the selection rationale
  in the calibration artifact. Choose an affordable judge that meets the reviewed
  acceptance criteria, rather than assuming a frontier model is required.
- Build a development-only semantic-judge calibration set with human labels for
  claim extraction, context support, source-verified correctness, gold-claim
  matching, answer relevancy, and pairwise semantic equivalence. Include repeated
  and compound claims, qualifiers, correct extra facts, unverified additions,
  and missed required facts. Freeze the exact judge version, decoding, prompts,
  rubric/schema, finite retry policy, and calibration checksum only after it
  passes recorded criteria for each responsibility; do not use validation or
  locked-test questions. Do not fabricate acceptance thresholds before review.
- Inspect the resulting question and document counts by evidence type before
  deciding which slice confidence intervals are sufficiently supported.
- Publish one complete authoritative RAG document row and answerable,
  unanswerable, table, formula, and mixed question examples after the real
  manifests and checksums exist.

## Vector databases

- Freeze the independent synthetic corpus/query seeds and selected-real-corpus
  allocation for validation and locked reporting, including exact workload sizes,
  dimensions, filters, concurrency, and source provenance. Locked requests must
  not reuse inputs observed during server/index selection.
- Set minimum successful-request and independent-query support for p95/p99
  reporting per workload cell. Record submitted, successful, and failed counts;
  many failed or repeated requests do not establish tail-latency support.

## Resolution rule

For each item, record the inspected corpus revision, observed counts, the chosen
rule, and the reason. Dataset reality may change the manifest design or reporting
slices; it must not silently change model outputs, metric definitions, or prior
results.
