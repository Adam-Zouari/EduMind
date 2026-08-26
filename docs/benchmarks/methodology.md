# EduMind benchmark manual

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Benchmark overview and commands](overview.md) ·
[Candidate-selection rationale](model-selection.md)

Read this document when you need to judge whether an EduMind experiment supports
a conclusion. For installation and downloads, use the
[installation guide](../setup/installation.md). For one stage's candidates and
metrics, use its page under `docs/benchmarks/`.

## 1. Purpose

EduMind uses experiments to produce evidence for an engineer to choose components; public leaderboards only choose which candidates are worth running. A complete result applies only to the recorded dataset, software revisions, hardware, device, and protocol. Benchmarks never choose a winner or edit production configuration.

The selection package has two files:

- `docs/benchmarks/model-selection.md` explains candidate screening and the included strategies.
- `experiments/benchmarks/selection_evidence.csv` records every model/vector-server decision and immutable runnable revision.

Executable YAML registries contain settings and included candidate IDs. `prepare.py` resolves those IDs into project-local snapshots and writes `data/benchmarks/models/selected.json`. Excluded evidence rows cannot become executable through preparation.

## 2. Profiles

`smoke` runs tiny real paths with one repetition. It detects missing dependencies and broken contracts but provides no comparative evidence. `standard` runs every approved candidate on development or validation data, includes warmups, and measures three repetitions. `full` runs engineer-selected finalists on larger or locked workloads; it requires `--shortlist DECISION_JSON` where a previous decision is expected.

## 3. Reproducibility

Each run records:

- dataset name, split, checksum, source revision, license, preprocessing version, and selected IDs;
- candidate settings and exact model revisions or server digests;
- Git state, dependency versions, seed 42, hardware, and devices;
- candidate/query ordering, warmups, repetitions, and failures;
- per-sample output before aggregation.

Documents or papers, not chunks/questions, are assigned to splits. Exact and near-duplicate checks prevent leakage. Evidence spans are verified against normalized text and represented by half-open offsets `[start, end)`.

MLflow creates one parent run per invocation and one child per candidate. Every candidate's scalar aggregates and confidence intervals are MLflow metrics; plans, provenance, summaries, candidate errors, and per-sample Parquet are artifacts. The parent logs whether the comparison is complete. Partial evidence remains visible but cannot be used as the source of a downstream decision.

## 4. Completion, statistics, and engineering decisions

Standard/full compute 95% confidence intervals using 10,000 paired bootstrap resamples with seed 42. The document, paper, recording, or query is the resampling unit. Holm correction is applied only when a report makes formal claims across several pairwise tests.

Every stage declares one **main metric** to make its central question obvious in MLflow and reports all other required quality, correctness, and operational metrics beside it. The main metric is not an automatic score or winner rule.

A comparison is complete only when every planned candidate finishes, returns the same unique sample IDs, and supplies every finite metric declared in that stage's metric contract. A low quality score, slow latency, or failed correctness check is still a successfully measured result. A crash, missing sample, duplicate sample ID, missing metric, or non-finite metric makes the whole comparison incomplete. Whatever was produced remains in MLflow and the artifact directory for diagnosis.

The runner calculates confidence intervals and paired differences but produces no gate result, eligibility label, Pareto set, ranking, recommendation, or promotion. An engineer reviews the complete standard/full run in MLflow and records any advancement in a separate JSON file:

```json
{
  "schema_version": 1,
  "source_summary": "../artifacts/benchmarks/rag/chunking-embedding/RUN/summary.json",
  "source_run_id": "RUN",
  "selected_candidates": ["token-256-32|MODEL"],
  "selected_by": "engineer name",
  "selected_date": "2026-08-26",
  "reason": "Why this trade-off fits the next experiment."
}
```

The loader verifies that the source is a complete standard/full run and that every selected ID succeeded in it. Smoke runs cannot be selected. `--shortlist`, `--embedding-selection`, and the other `--*-selection` arguments consume these engineer-authored files. Each downstream parent run logs the decision file under `engineer-decisions/` and records its SHA-256 in provenance.

## 5. Document extraction

Images, PDFs, and DOCX are evaluated through complete document parsers and one canonical structured output. It retains normalized text, pages, reading order, tables, formulas, bounding boxes where available, source identity, parser revision, warnings, and provenance.

The development phase evaluates exactly 24 Docling Standard configurations:

```text
OCR engine {RapidOCR, Tesseract, EasyOCR}
× OCR mode {PDF-aware regions, full page}
× TableFormer {fast, accurate}
× formula enrichment {off, on}
```

OCR engine tests recognizer integration; OCR mode tests preservation of native PDF text versus complete raster recognition; TableFormer tests table quality/cost; formula enrichment tests CodeFormula value/cost. Their Cartesian product is necessary because these options interact.

Docling 2.117.0, English, OCR scale 3.0, table-cell matching, canonical output, code enrichment off, and native DOCX ingestion stay fixed. They define the experimental environment or output requirements rather than a current product question.

The engineer selects Standard configurations for the architecture comparison with Granite Docling 258M and PaddleOCR-VL-1.6. The latter two test independent visual-parser architectures, not isolated OCR recognition.

The target corpus covers clean/noisy scans, phone photos, digital/scanned/mixed PDFs, broken encodings, slides, academic layouts, headings, lists, captions, embedded images, tables, and formulas. Split membership is document-family isolated. Required annotations depend on claimed metrics.

Primary quality includes CER, WER, Content F1, Reading Order Accuracy, Page Coverage/Attribution, Block Structure F1, Table Detection/Content/Structure F1, Formula Detection F1, and formula similarity. Operational output includes cold load, p50/p95 page/document latency, throughput, RAM, VRAM, and temporary disk.

## 6. Audio extraction

Candidates are Whisper `small.en` control, Canary 180M, Parakeet TDT 0.6B v2, MOSS Transcribe-Diarize, and Qwen3 ASR 1.7B with Qwen3 ForcedAligner 0.6B. All execute exact local snapshots on a common requested device.

Qwen transcription and forced alignment execute sequentially. The ASR is released before loading the aligner, but complete latency and peak resources cover the entire operation. This prevents the composite profile from hiding alignment cost.

The corpus covers clean and noisy English speech, accents, technical vocabulary, multiple speakers, and longer educational recordings. Primary metrics are WER, CER, Timestamp MAE, Segment Boundary MAE, missing/hallucinated speech, and timestamp coverage. Real-Time Factor, p50/p95 latency, cold load, throughput, RAM, and VRAM measure operation.

## 7. Video extraction

Video freezes the selected ASR and visual parser, then compares fixed-interval keyframes, scene-change keyframes, and scene change with maximum-interval fallback. This isolates frame selection.

Metrics are Transcript WER, Visual Text Precision/Recall/F1, duplicate visual text, Audio/Visual Alignment MAE, Complete Content Recall, Real-Time Factor, latency, resource use, and temporary disk. Missing timestamp annotations omit the dependent metric rather than fabricating zero.

## 8. Normalization

Minimal normalization changes Unicode, line endings, and whitespace. Conservative normalization repairs common extraction artifacts with preservation priority. Aggressive normalization tests stronger cleanup and its deletion/merge risk.

For reference `R`, observed text `O`, and normalized text `N`, the benchmark compares edit distance before and after. Content Preservation Recall and Corruption Removal Precision/Recall/F1 are primary. CER/WER, content precision/recall/F1, repeated text, determinism, and latency are diagnostic. At least 200 document-family-isolated cases are required for standard/full.

## 9. Chunking and embeddings

The 64-pair matrix crosses:

- recursive character;
- token 256/32, 384/64, and 512/64;
- sentence 8/2;
- semantic;
- section-aware 512/64;
- structure-aware 512/64;

with MiniLM, Snowflake Arctic Embed M v2, F2LLM v2 0.6B, Octen 0.6B/4B, Qwen3 Embedding 0.6B/4B, and Nemotron Embed 1B.

Each embedding profile freezes its query/document interface or prefix, tokenizer limit, checkpoint pooling behavior, vector normalization, dimension, and cosine similarity. Semantic chunking uses the tested embedding both to detect topic boundaries and retrieve chunks, so it is intentionally evaluated as a complete pair.

Exact NumPy retrieval removes vector-server ANN error. Chunk/evidence overlap is clamped non-negative, and overlapping retrieved intervals are merged before Context Recall.

The main metric is nDCG@5. The remaining required metrics are nDCG@3, Context Recall@3/@5, rank-aware Context Precision@3/@5, Context Recall under 2,048 tokens, Precision/Recall/Hit Rate at 1/3/5/10, MAP@3/@5/@10, MRR, nDCG@10, chunk statistics, indexing time, latency, memory, and storage. An engineer may select up to three pairs for retrieval testing.

## 10. Retrieval and reranking

The retrieval stage freezes engineer-selected chunker–embedding pairs and compares dense retrieval, BM25, RRF of dense and BM25 ranks, and RRF followed by each approved reranker. RRF fuses ranks rather than incompatible raw scores. Rerankers rescore the top 20 passages with the query.

The rerankers are MiniLM control, Ettin 150M/400M/1B, and Qwen3 Reranker 4B. Their exact scoring interfaces and tokenizer limits are part of the candidate profile. Quality metrics match the chunking stage; reranker latency and memory are added. An engineer may select up to three retrieval stacks for final RAG.

## 11. Vector databases

The first server comparison contains Chroma, Qdrant, Weaviate, and PostgreSQL/pgvector. NumPy is only the exact-neighbor oracle. Identical precomputed normalized float32 vectors, IDs, metadata, filters, query order, and cosine contracts are supplied to every server.

Conformance checks health, dimension rejection, compound and empty filters, replacement without stale chunks, deletion, persistence after restart, schema incompatibility, and actual ANN-index existence/use. Each check is reported as evidence. A zero score does not make the run incomplete unless the server failure prevents the remaining measurements from running.

Smoke uses 1,000 vectors and 50 queries at concurrency 1. Standard uses 100,000 vectors at 384 and 1,024 dimensions, 500 queries, concurrency 1/8/32, and filter selectivity near 50%/10%/1%. Full uses selected real vectors plus 1,000,000 synthetic clustered vectors, 1,000 queries, concurrency 1/8/32/64, and filters down to 0.1%.

The main metric is ANN Recall@10. Filtered and unfiltered ANN Recall@1/@3/@5/@10, filter and lifecycle correctness, p50/p95/p99 latency, throughput/error rate, build and incremental ingestion throughput, memory, storage, and restart/readiness time are also required. The engineer judges the measured trade-offs; the runner applies no pass threshold.

## 12. Generation

The frozen-context screen compares Qwen3 1.7B thinking-off control, MiniCPM5 1B reasoning, G9v3 3B reasoning, and Qwen3.5 4B reasoning. All run through direct Hugging Face loaders with official chat templates, native checkpoint dtype, no quantization or offload, temperature 0, seed 42, 8,192 context tokens, 256 answer tokens, and the same whole-model device.

Human Faithfulness, Answer Correctness, Completeness, Citation Precision/Recall/F1, and Answerability Balanced Accuracy are primary. Exact Match, Token F1, ROUGE-L, HHEM faithfulness, refusal behavior, unsupported answers, malformed output, and determinism are diagnostic. HHEM is not an authoritative evaluator.

Operational metrics include load time, TTFT, prompt evaluation, total p50/p95, answer tokens, tokens/second, answers/minute, RAM, and VRAM. Models are unloaded and memory is cleaned between candidates.

## 13. Final RAG and human review

Only engineer-selected components enter the complete-system comparison. Each request includes retrieval, optional rank fusion/reranking, 2,048-token context packing, prompting, and direct generation. Component and end-to-end timings remain separately visible.

Three anonymous systems produce answers for 20 stratified questions, yielding 60 blinded judgments. Reviewers score Faithfulness, Answer Correctness, Completeness, Citation Accuracy, and answerability on the documented rubric. Judgments are imported and validated before system identities are revealed.

One selected stack is evaluated once on the locked test. Any subsequent tuning creates a new benchmark version.

## 14. Extraction-to-RAG confirmation

After extraction and RAG selection, the same verified raw documents are run through:

```text
verified reference text → selected RAG
selected extracted text → selected RAG
```

The report contains deltas in retrieval quality, human answer quality, citation F1, and end-to-end latency. This measures extraction-induced degradation without using downstream results to retroactively tune the locked component comparisons.

## 15. Invalid conclusions

Do not claim that smoke results establish quality or speed; a public leaderboard winner is the EduMind winner; different public metrics are numerically comparable; a component winner is automatically the best complete stack; automated faithfulness replaces human review; a result generalizes to unrecorded hardware; or a benchmark changes production without approval.
