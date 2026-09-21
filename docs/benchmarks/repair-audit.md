# Document, ASR, and video benchmark repair audit

This file preserves dated verification evidence. A result applies to the
repository state named by its section and must not be read as a rolling claim
about the current candidate set.

## Historical audit snapshot — 2026-09-11

At the close of this audit, the target-scope status was complete. The document,
ASR, and video benchmark code implemented the then-frozen methodology and metric
contracts. The independent second pass found no unresolved correctness or
methodology mismatch in those three systems.

Application behavior, the end-to-end RAG pipeline, and authoritative dataset
population were not part of this repair.

The `VideoProtocolLock` references below describe the historical 2026-09-11
repository state. They were superseded by the centralized benchmark-protocol
migration: current video runs use
`experiments/benchmarks/extraction/video/protocol.yaml`, normal manifest
provenance, and frozen artifacts bound to both video and audio protocol
checksums. Historical runs and their recorded lock artifacts are preserved and
must not be reinterpreted as current-format runs.

## Implemented contract

### Document

- PaddleOCR-VL page indices, table HTML, canonical rows/cells, plain table text,
  original HTML, exact formula LaTeX, and recoverable conversion warnings are
  preserved by the native-output adapter.
- Authoritative references require explicit capabilities. Smoke references may
  infer them. Table and formula negatives remain explicit through `has_table`
  and `has_formula`.
- Visual layout, table, and formula matching is page-restricted, one-to-one,
  and thresholded at 0.5. Page attribution remains a separate content-first
  match. Layout boxes affect matching only when `layout_boxes` is claimed.
- Every configured repetition is attempted and recorded. Any failed measured
  attempt produces empty quality, failure rate 1, and determinism 0 for that
  sample. Throughput uses pages from every successful attempt divided by all
  measured attempt time.
- Primary models, Paddle layout components, Docling parser components, OCR
  assets, system executables, revisions, options, package versions, and cache
  manifests are recorded and verified.

### ASR

- At audit time, the now-retired Qwen profile used
  `Qwen/Qwen3-ASR-1.7B-hf` at
  `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` and the token-classification
  `Qwen/Qwen3-ForcedAligner-0.6B-hf` at
  `c07281df297b9905d24a508279258cccf987a064`.
- Empty transcript plus empty timestamps is a valid zero-quality observation.
  Lexical text without timestamps and empty text with lexical timestamp units
  are fatal contradictions.
- Timestamp alignment uses deterministic dynamic programming over ordered,
  non-overlapping, non-reusable contiguous predicted spans. Eligibility is
  normalized token Content F1 at 0.5; optimization and tie-breaking follow the
  documented total-similarity, match-count, and earliest-span order.
- Every model's decoder, generation, timestamp, language, batch, device, dtype,
  revision, path, package version, and cache checksum settings are frozen and
  persisted.

### Video

- Video has a dedicated runner and a fresh visual worker. The obsolete generic
  document/video benchmark route has been removed.
- The executable grid is fixed 5/10/20 seconds, scene 0.30/0.40/0.50, and hybrid
  selected-threshold plus 5/10/20-second maximum gaps. Every selector includes
  frame zero and uses FFmpeg variable-frame-rate output.
- Smoke, development, validation, and locked profiles enforce their phase/selection
  contracts. Locked execution requires exactly one validated configuration.
- At that historical revision, a versioned `VideoProtocolLock` bound the manifest, ASR windows, overlap,
  deterministic stitching, visible-text units, occurrence thresholds,
  reviewer/date, and hybrid scene threshold. Authoritative runs reject smoke,
  missing, or checksum-mismatched locks.
- Frozen ASR is created once in a separate process and checksummed. Visual
  children validate and reference it and do not import or invoke ASR. WER is
  emitted only by the frozen-ASR child.
- Visual metrics use distinct units and one-to-one timed occurrences. Coverage
  is maximized first and total raw first-detection delay in seconds is minimized
  second. Delay and duplication are null in their documented empty cases.
- Visual RTF, warm p50/p95, explicit parser cold load, process-tree RAM, VRAM,
  and mean selected frames exclude audio work.

## Independent second-pass findings

The second audit was performed after the initial repair and real-model smoke
runs. It found and fixed five edge cases:

1. A single unsupported extra page was incorrectly counted as a duplicate page.
2. Document throughput reused the first successful repetition's page count
   instead of summing the actual successful attempt counts.
3. Video occurrence assignment minimized delay relative to interval length
   instead of the documented raw seconds.
4. A non-object Paddle block was skipped without a recoverable-conversion
   warning.
5. Unclaimed layout boxes could influence content matching for references that
   declared only types, hierarchy, or reading order.

Each correction has a regression test. Re-audit searches also confirmed that
the legacy complete-content recall, per-visual-candidate transcript WER, and
visual-text-prefixed metric names are absent, the old generic video execution
path is absent, and no stale non-`-hf` Qwen checkpoint identifier remains.

## Historical verification results — 2026-09-11

| Check | Result |
|---|---|
| Full test suite | 86 passed |
| Ruff over document/ASR/video and their shared edited code | Passed |
| Python bytecode compilation | Passed |
| `pip check` | No broken requirements |
| `git diff --check` | Passed; line-ending notices only |
| Generated selected-model lock | All 8 entries requested by the 2026-09-11 candidate set and all cache/component checksums verified |
| Pinned OmniDocBench Docker identity preflight | TEDS, TEDS-S, and CDM passed |

A repository-wide Ruff run reports three unrelated existing findings in RAG
generation and vector-database code. They are outside this audit's requested
scope and do not occur in the repaired benchmark systems.

## Historical real-model smoke evidence — 2026-09-11

The smoke inputs are wiring fixtures, so their quality numbers are not model
selection evidence.

| System | Candidate | Device | Result | Artifact run |
|---|---|---:|---|---|
| ASR | Whisper small.en | CUDA | Success; WER 0.0 | `20260911-010514-a5b73b32` |
| ASR | Canary 180M | CUDA | Success; WER 0.0 | `20260911-005508-759626bc` |
| ASR | Parakeet TDT 0.6B v2 | CUDA | Success; WER 0.0 | `20260911-005545-a190a095` |
| ASR | MOSS Transcribe-Diarize | CUDA | Success; WER 0.0 | `20260911-005626-b12dd683` |
| ASR (retired) | Qwen3 ASR plus exact forced aligner | CUDA | Historical success; WER 0.0 and timestamp coverage 1.0 | `20260911-005723-891fafba` |
| Document | Docling Standard | CUDA | Success | `20260911-005859-529d84ee` |
| Document | Granite Docling 258M | CUDA | Success | `20260911-005949-8a22cef7` |
| Document | PaddleOCR-VL 1.6 | CPU | Success, including native table conversion | `20260911-002347-8e9d35e3` |
| Video ASR child | Frozen Whisper artifact | CPU | Success; one ASR artifact for two videos | `20260911-003307-a748b7fb` |
| Video visual child | Docling fixed 5/10/20-second candidates | CUDA | All three successful; no ASR execution | `20260911-010343-f1babf9e` |

The Paddle table smoke additionally produced three canonical rows, nine cells,
plain cell text, original HTML, page 1, and no conversion warnings in
`artifacts/benchmarks/real-smoke/paddle-table.json` (a local ignored artifact).

## Post-audit selection change — 2026-09-14

Qwen3-ASR was removed from the runnable shortlist because its BF16 weights
exceed the target GPU memory gate before activations. Its runtime code and local
snapshots were removed. The active ASR benchmark now contains four profiles:
Whisper `small.en`, Canary 180M, Parakeet TDT 0.6B v2, and MOSS
Transcribe-Diarize.

The retired Qwen contract and smoke row above remain solely as evidence of what
was tested on 2026-09-11. The “8 entries” model-lock result also belongs only to
that historical snapshot. Candidate documentation and selection records were
rechecked after removal, but the complete 2026-09-11 real-model smoke suite was
not rerun as part of this selection change.

Documentation-only verification on 2026-09-15 confirmed that active benchmark
documents list the four-profile ASR set, contain no active Qwen3-ASR or forced-
aligner references, and resolve all local Markdown file links. This check does
not supersede or restate the historical code, model-lock, or real-inference
results above.

## Residual limitations outside benchmark code

- Authoritative manifests and the data-reviewed authoritative video protocol
  values do not yet exist. The committed video protocol therefore leaves
  `selected_scene_threshold` and `selected_scene_source_run_id` null. Fixed and
  scene development work can run, while authoritative hybrid work intentionally
  rejects the unresolved selection.
- The installed pinned PaddlePaddle 3.3.1 Windows wheel is not compiled with
  CUDA. Paddle completed real CPU inference, while the CUDA request failed
  explicitly and did not fall back silently.
- This Windows WDDM driver exposes target GPU processes but not per-process
  memory bytes. CUDA artifacts therefore label the fallback
  `nvml-device-delta-wddm`, which measures peak device-memory increase from a
  pre-run baseline after observing the target process.
- Prepared model snapshots and generated benchmark run artifacts are local,
  ignored artifacts rather than source-controlled data.

## Centralized protocol migration and re-audit — 2026-09-21

Every independently executable benchmark now owns a strict, versioned
`protocol.yaml`: document, ASR, video, chunking–embedding,
retrieval–reranking, generation, Final RAG, and vector database. Resolved
protocol checksums participate in run fingerprints and provenance; parent and
child runs persist the exact settings they execute. Candidate registries retain
identities and phase membership, while immutable model revisions and snapshot
checksums remain in the selected-model lock.

The independent post-migration audit found and fixed these execution drifts:

1. Document execution did not reject a manifest from the wrong split.
2. Shared audio preparation still had hidden 16 kHz/mono/sample-width defaults.
3. Several protocol fields accepted values unsupported by the actual adapters.
4. Video occurrence settings were copied into a second worker field instead of
   being read directly from the verified protocol.
5. Generation reconstructed model dtype independently from the executed
   protocol configuration.
6. Vector concurrency, adapter timeouts, and confidence levels were not all
   consumed from the resolved protocol.
7. Sequential fresh-process candidates reused Python's cached first temporary
   directory after that directory had been deleted.
8. Windows WDDM VRAM monitoring rebased each child on transient memory retained
   from its predecessor, and monitor failures could hide the underlying
   candidate error.
9. The vector HNSW protocol label omitted the final `ef_construction`
   tie-break that the selector executes.
10. Factorless Docling-standard paths could fall through to library OCR/table
    defaults instead of the protocol's declared baseline configuration.

Regression coverage was added for the corrected paths. A final search found no
active `VideoProtocolLock`, `--protocol-lock`, `--scene-selection`, implicit
cross-benchmark protocol load, duplicated execution-only parameter dictionary,
or behavior-changing candidate-registry value outside the documented ownership
rules. Historical lock references above remain intentionally dated evidence.

### Verification results

| Check | Result |
|---|---|
| Full test suite | 162 passed |
| Repository Ruff check | Passed |
| Python bytecode compilation | Passed |
| `pip check` | No broken requirements |
| `git diff --check` | Passed; line-ending notices only |
| Protocol parser/behavior/worker identity tests | Passed for all eight suites |
| CUDA retrieval/reranking matrix | All 15 stacks completed |

### Real smoke evidence

Smoke fixtures verify wiring and reproducibility; their scores are not model
selection evidence.

| System | Device | Result | Artifact run |
|---|---:|---|---|
| Chunking–embedding, GTE control | CUDA FP16 | Complete; normalized finite 768-dimensional vectors | `20260921-045230-4986c952` |
| Document PDF | CUDA | Complete | `20260921-045258-c15ed291` |
| Document image | CUDA | Complete | `20260921-045335-b404667c` |
| Document DOCX | CUDA | Complete after explicit protocol-baseline routing | `20260921-052845-164db78b` |
| ASR, all four active candidates | CUDA | Complete; 4/4 successful | `20260921-045426-a1ce9c9f` |
| Video frozen-ASR phase | CUDA | Complete; regenerated artifact uses current video and audio checksums | `20260921-045640-28b59fb0` |
| Video fixed visual candidates | CUDA | Complete; fixed 5/10/20-second candidates | `20260921-045651-05ed5108` |
| Retrieval–reranking | CUDA FP16 | Complete; 15/15 stacks successful | `20260921-051356-1881454c` |

Docker vector smoke could not run because the local Docker Desktop daemon was
not available. Generation and Final-RAG real smoke could not run because their
selected model snapshots are not prepared in the current model lock. These are
environment/preparation limitations, not silent fallbacks; their protocol and
mocked execution tests pass. Any pre-migration frozen video-ASR artifact remains
intentionally incompatible and must be regenerated before reuse.
