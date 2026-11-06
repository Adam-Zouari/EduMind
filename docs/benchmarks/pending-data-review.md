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
- Freeze minimum independent eligible/contributing-source support separately
  from observation counts. State required attempt/evaluation coverage and keep
  descriptive points with null bounds when support is insufficient. Diagnostic
  repeatability can have source-sampling uncertainty; repeated pairs do not add
  sources, and no universal numerical minimum has been adopted.

## Document extraction

- Review whether the largest-file-per-format preflight inputs exercise demanding
  page counts, rendered resolutions, and content density. File bytes alone do
  not establish an upper workload boundary; freeze any revised stress-selection
  or supported-envelope contract from development data before qualification.
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
- Check support for completed-first-output recognition/event rates and boundary
  MAE, completed-output reference-denominator timestamp coverage, three-attempt pairwise
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
- Verify independent video/source-group counts separately from window/attempt
  counts. Freeze CI support requirements for visual quality/reliability, frozen-ASR
  recognition/timestamps/event/failure rates, audio/visual RTF, and frame counts.
  Review p50/p95 support separately, especially on the six-video validation and
  locked splits. Limited support keeps descriptive point estimates with null
  bounds/reasons; p50 can remain the main summary, not an arithmetic mean.
  Ten thousand bootstrap resamples do not add independent evidence.
- Freeze consistent visible-line segmentation under `normalized_lines_distinct_v1`.
  Each timed unit has nonempty projected text, a stable occurrence ID, and a
  verified `[start, end)` interval. Review explicit text-free negatives and
  genuine reappearances. Do not convert reference/prediction at different
  granularities or count a first capture of a new appearance as duplication.
- Freeze canonical visual repeatability representation: selected-frame times,
  frame/text ordering, normalized numeric precision, and stable ID mapping.
  Identical errors never agree; every measured attempt must execute extraction.
- Review spoken reference timestamps independently of visual intervals. Freeze
  which videos claim the spoken timing task, retaining legitimate silent videos.
  Freeze `datasets.nonspeech_unit_type` (`whole_video` or `asr_window`) and
  `datasets.required_nonspeech_units` in video `protocol.yaml`: one nonspeech
  event unit type and required per-split reviewed allocation
  using whole nonspeech videos or actual ASR windows including overlap. Do not
  fabricate counts, infer labels from missing text, or import standalone speech
  quotas/control-category requirements into video.
- Define the annotation labels needed to diagnose slide, screen-recording,
  presenter, gradual-change, and repeated-scene behavior from the real corpus.
- Publish one complete video-manifest row after the source interval, visible-text
  timestamps, transcript, license, and checksum have been verified.

## Chunking, embedding, retrieval, and generation

- Inspect the pinned QASPER splits and verify that the prepared 100/40/40 paper
  allocation preserves source-paper isolation and useful answerable/unanswerable
  coverage.
- Freeze answerable-query and independent-document counts for chunking–embedding
  overall and per evidence slice. Unanswerable coverage belongs to generation;
  those shared-manifest queries are not retrieval/timing requests in this phase.
  Review atomic units and token/source spans so complete-unit coverage and token
  overlap describe the intended evidence without candidate-specific repairs.
  Freeze quality/slice, diagnostic top-five Ranking Repeatability, and complete-
  query/query-embedding/vector-search p50/p95 CI support requirements under the
  [shared support policy](metrics.md#confidence-intervals) before authoritative comparisons;
  insufficient support retains defined point estimates with null bounds/reasons.
- Freeze corresponding source support for retrieval–reranking's complete-order
  Ranking Repeatability. Preserve source-document grouping and every scheduled
  pair/failure inside each draw; do not infer support from the number of queries
  or repeated outputs. Candidate-roster and index-selection changes are not
  part of this statistical-support review.
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
- Retain uncapped alpha-nDCG ratios and greedy-baseline-exceeded flags during
  that review, including single-unit questions. Inspect degenerate quality and
  latency bootstrap results: retain supported calculated equal bounds with a
  degeneracy flag rather than treating them as certainty on future papers.
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
  claim extraction, full-context support, cited-passage support, source-verified correctness, gold-claim
  matching, answer relevancy, and pairwise semantic equivalence. Include repeated
  and compound claims, qualifiers, correct extra facts, unverified additions,
  missed required facts, and genuinely claim-free responses. Freeze the exact judge version, decoding, prompts,
  rubric/schema, finite retry policy, and calibration checksum only after it
  passes recorded criteria for each responsibility; do not use validation or
  locked-test questions. Do not fabricate acceptance thresholds before review.
- Calibrate the single structured per-response rubric with contrasting evidence
  scopes: support in cited passages, support elsewhere in supplied context,
  correctness established only by evaluator references, and no verified support.
  Include valid-but-irrelevant citations, unknown IDs, true extra facts, and
  missing required facts. Check that gold references do not leak into either
  support label and that cited-subset support implies full-context support.
  Pairwise semantic equivalence remains a separate calibrated rubric; keep
  claim extraction and per-claim justifications inspectable.
- Inspect the resulting question and document counts by evidence type before
  deciding which slice confidence intervals are sufficiently supported.
- Review generation token-event instrumentation and independent-source support
  for [Decode Throughput](rag/generation/metrics.md#decode-throughput). Its
  completed-stream work/time estimator and interrupted-stream exclusion are
  frozen; data review determines support for the conditional interval, not a
  choice between averaging rates and pooling work/time.
- Publish one complete authoritative RAG document row and answerable,
  unanswerable, table, formula, and mixed question examples after the real
  manifests and checksums exist.

## Final RAG readiness

- Freeze reviewed integrated-preflight inputs and their supported envelope:
  demanding retrieval/reranker inputs, packed generator prompts, generation
  limits, and selected model-residency transitions. Check the complete selected
  server-backed path, not only independently qualified component models. Keep
  these inputs and CPU/CUDA smoke fixtures separate from locked questions.
- On those nonlocked readiness inputs, verify stable chunk/source/content identity
  despite reordered context and request-local citation labels. Verify that the
  frozen ANN index permits valid pool variation without weakening corpus,
  filter, ranking/permutation, or citation-resolution checks. This is readiness
  testing, not another finalist search or a required exact-pool equality gate.

## Vector databases

- Inspect representative selected-model vectors and queries; freeze the initial
  corpus, plausible growth sizes, source allocations and held-out validation/locked
  workloads. Record dimensions/dtype, corpus/query checksums, related-source
  grouping, and separately labelled synthetic stress seeds. The old synthetic
  100k/1M grid is not automatically the reviewed real-data scale.
- Freeze each database's bounded configuration roster and tuning budget before
  comparative runs, including supported native planning. Record shortlist limits;
  validation and locked reuse complete database–configuration identities without
  retuning. Any recall target used for review must be declared, not inferred from
  a driver's old automatic selector.
- Review document-ID filters, exact eligible sets, restrictive/compound/empty
  cases, and correlation between filter membership and query/vector clusters.
  Integration evidence must be available inside each question's allowed scope.
- Freeze exact-distance evaluator precision and identity. Keep boundary-tie
  equality exact by default; any nonzero tolerance requires numerical evidence
  and a protocol revision before candidate comparison. The boundary-slot credit
  rule itself is fixed, not selected from benchmark outcomes.
- Freeze request payloads, concurrency, schedules, representative and load warmup,
  measured window duration/count, connection/retry/timeout policies, and drain
  boundaries. Identify first quality and three repeatability observations before
  execution. Keep static search and document mutations separate unless an explicit
  mixed-load condition is needed.
- Select representative logical document insertion/replacement/deletion sizes
  and independent source groups. Freeze starting/restored states, visibility
  probes and polling intervals, restart/cache procedure, server memory/cache
  accounting, and the synchronized apparent-byte post-build storage checkpoint.
- Set independent query/source support for ANN/evidence/repeatability intervals,
  and independent load-window or operation-group support for p50/p95, QPS and
  rates before authoritative comparisons, including development. Preserve
  concurrent dependence. Record submitted, completed, failed,
  not-executed and checkable counts; 10,000 bootstrap draws do not create new
  independent data or load executions. Unsupported bounds remain null while
  descriptive points retain their actual scope.

## Resolution rule

For each item, record the inspected corpus revision, observed counts, the chosen
rule, and the reason. Dataset reality may change the manifest design or reporting
slices; it must not silently change model outputs, metric definitions, or prior
results.
