# Benchmark candidate selection

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Benchmark overview](overview.md) · [Benchmark manual](methodology.md) ·
[Selection evidence](../../experiments/benchmarks/selection_evidence.csv)

Status: **public-evidence shortlist; EduMind's local benchmarks inform the engineer's final decisions**

Selection package: **benchmark-candidates**

Evidence reviewed: **2026-08-25**

## How candidates are chosen

Candidate selection follows the same practical sequence for each component:

1. **Define the job.** Retrieval models are screened on retrieval quality, rerankers on reranking quality, ASR models on English transcription plus timestamps, and generators on their suitability for grounded answering.
2. **Inspect relevant public evidence.** Prefer a common task-specific benchmark such as MTEB Retrieval, a common reranker comparison, OmniDocBench, or the Open ASR Leaderboard. General-capability evidence is used only when no current task-specific comparison covers the candidate set.
3. **Apply basic eligibility checks.** A candidate must be downloadable, self-hostable, usable for the intended EduMind deployment, and expose the capability required by the experiment.
4. **Keep different resource scales.** After reviewing the available models, they are organized into approximate parameter-size groups so the local benchmark compares compact, middle, and higher-quality options. These groups describe the reviewed shortlist; they were not fixed before the search.
5. **Prefer comparable evidence.** When candidates were tested under the same public protocol, the strongest eligible representatives are kept. When promising models use incompatible protocols, both may be kept and compared locally instead of comparing unlike public scores.
6. **Keep a control.** The current or established baseline is always included so the experiment can measure whether changing the component is worthwhile.
7. **Decide locally.** Public evidence creates the shortlist. An engineer reviews EduMind's frozen-dataset quality, latency, and resource measurements to make the final decision.

Scores from different benchmarks are never combined or compared numerically. An MTEB Retrieval score, an RTEB score, an Artificial Analysis score, and WER answer different questions.

### What the approximate size groups mean

The ranges in this document were added after inspecting the candidate sizes. They are a readable way to preserve resource diversity, not preregistered thresholds and not memory guarantees.

## What is in the selection package

| File | Purpose | Example |
|---|---|---|
| `docs/benchmarks/model-selection.md` | Reader-facing explanation of the method, candidates, strategies, and evidence. | The embedding table explains why Qwen, Octen, F2LLM, Nemotron, and Snowflake are included. |
| `selection_evidence.csv` | One machine-readable row for every model checkpoint or vector-database product that received an explicit include/exclude decision. | The Qwen3-Embedding-0.6B row records its public screening result, exact source, candidate revision, and decision. |

### `selection_evidence.csv` column reference

| Column | Meaning | Example from the file |
|---|---|---|
| `component` | Component being selected. | `embedding` |
| `candidate` | Exact model or product identifier. | `Qwen/Qwen3-Embedding-0.6B` |
| `purpose` | Why the row participates in the selection package. | `candidate`, `control`, or `diagnostic` |
| `decision` | Whether the row is included in the benchmark shortlist. | `include` or `exclude` |
| `approx_params_b` | Approximate parameter count in billions; blank for non-model products. | `0.595776512` |
| `public_benchmark` | Public benchmark used to screen the candidate. | `MTEB English v2` |
| `public_metric` | Metric represented by `public_score`. | `Retrieval` or `avg WER` |
| `public_score` | Public screening value; blank when no numerical public result is claimed. | `61.83` |
| `benchmark_source_url` | Exact page or result file containing the public score. | A revision-pinned model card or result table. |
| `benchmark_source_revision` | Exact commit, dataset revision, or benchmark version used for the score. | `d43997c8...` |
| `candidate_source_url` | Official model/product page for the candidate that will be executed. | A pinned Hugging Face model page. |
| `candidate_revision` | Exact checkpoint, composite profile revisions, or selected server versions. | `d43997c8...` |
| `license` | Recorded published license or terms label. | `Apache-2.0` |
| `reviewed_date` | Date the evidence and candidate information were reviewed. | `2026-08-23` |
| `reason` | Concise explanation for the decision and any important evidence limitation. | Qwen is kept as a strong candidate for direct local comparison. |

Blank values mean that the field does not apply or the information is unavailable. They are not replaced with `N/A`, zero, or another invented value.

### Purpose and decision

`purpose` uses only three values:

| Purpose | Meaning |
|---|---|
| `candidate` | A possible component to evaluate. It may be included or excluded from the runnable shortlist. |
| `control` | The current or established baseline used to measure improvement. |
| `diagnostic` | A supporting evaluator, such as HHEM, that cannot replace authoritative human evaluation. |

`decision=include` means **run this row in the relevant EduMind benchmark**. It does not mean that the candidate has been promoted into production. `decision=exclude` means that the reviewed row is not part of the current runnable shortlist; `reason` explains why.

### How to interpret public evidence

`public_benchmark`, `public_metric`, and `public_score` must be read together. A score is meaningful only under the dataset, metric, and evaluation protocol that produced it. Public scores are compared only when all of those details and the benchmark version are identical. Results from different MTEB/RTEB tables or other protocols are evidence that a candidate is worth testing, not proof that one candidate is better than another.

Every shortlisted candidate is therefore evaluated again through EduMind's frozen local benchmark. Those common local results make the final comparison.

`benchmark_source_url` and `benchmark_source_revision` identify where a public score came from. `candidate_source_url` and `candidate_revision` identify the model, composite profile, or server EduMind intends to execute. These may point to different artifacts: an ASR leaderboard can establish WER while the model page establishes timestamp support and the exact checkpoint.

Candidate-specific links appear in the relevant table row. A benchmark shared by every row is linked once immediately below the table. A live link is convenient for browsing newer submissions; a revision-pinned link identifies the state used for a recorded value.

## Required controls

| Component | Control | Purpose |
|---|---|---|
| Chunking | Token 256/32 | Current fixed-window chunking. |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` at `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` | Current lightweight embedding. |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` at `233902d25c440f23af6f7d6e94d2946bac0bee0a` | Established cross-encoder baseline. |
| Generation | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B/tree/b9352fbb8ce704292730cf54b3b1dceb2a808738), thinking disabled | Small direct-checkpoint control executed through the same Hugging Face runtime as the candidates. |
| Document extraction | Docling Standard baseline configuration | Current unified-parser reference. |
| ASR | `openai/whisper-small.en` at `e8727524f962ee844a7319d92be39ac1bd25655a` | Established English ASR reference. |
| Vector database | Chroma server | Current server baseline. |

Controls are run alongside the candidates; being a control does not make a component the final choice.

## Chunking and embeddings

### Embedding candidates

Primary public quality metric: retrieval-specific **nDCG@10 / Retrieval score**, not generic overall MTEB average.

Approximate size groups observed in the reviewed shortlist:

- **≤350M**
- **>350M–800M**
- **>800M–1.5B**
- **>1.5B–4.5B**

| Approximate size | Included candidates | Why they are included | Evidence |
|---|---|---|---|
| ≤350M | `Snowflake/snowflake-arctic-embed-m-v2.0` | Highest eligible English Retrieval result in the frozen ≤350M comparison: **58.4**. | [MTEB record](https://leaderboard.mteb.org/models/Snowflake/snowflake-arctic-embed-m-v2.0); [pinned comparison](https://github.com/ibm-granite/granite-embedding-models/tree/250b8522ad2a7ea0c1e26f089d3de212390f614b); [pinned model card](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/blob/95c2741480856aa9666782eb4afe11959938017f/README.md) |
| >350M–800M | `Qwen/Qwen3-Embedding-0.6B`; `Octen/Octen-Embedding-0.6B`; `codefuse-ai/F2LLM-v2-0.6B` | Qwen leads the directly comparable MTEB screen (**61.83 Retrieval**). Octen has strong RTEB evidence and F2LLM has official MTEB task results, but their available aggregates are not directly comparable to Qwen's frozen result. | [Qwen MTEB](https://leaderboard.mteb.org/models/Qwen/Qwen3-Embedding-0.6B); [Octen/RTEB](https://leaderboard.mteb.org/benchmark/RTEB%28beta%29); [F2LLM MTEB](https://leaderboard.mteb.org/models/codefuse-ai/F2LLM-v2-0.6B) |
| >800M–1.5B | `nvidia/Nemotron-3-Embed-1B-BF16` | NVIDIA's common RTEB-16 table reports **72.38 average nDCG@10**, above the reviewed nearby-size models in that table. | [MTEB record](https://leaderboard.mteb.org/models/nvidia/Nemotron-3-Embed-1B-BF16); [pinned RTEB table](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16/blame/c932836c54f75b7df5da0b0f519ea4cfd276a8e4/README.md) |
| >1.5B–4.5B | `Qwen/Qwen3-Embedding-4B`; `Octen/Octen-Embedding-4B` | Qwen reports **68.46 MTEB English-v2 Retrieval** and Octen reports **0.7747 RTEB public mean**. The protocols differ, so both proceed to the same local benchmark. | [Qwen MTEB](https://leaderboard.mteb.org/models/Qwen/Qwen3-Embedding-4B); [Octen/RTEB](https://leaderboard.mteb.org/benchmark/RTEB%28beta%29) |

The MiniLM control is evaluated with every compatible chunker. Public evidence sources and exact candidate revisions are recorded in `selection_evidence.csv`.

### Chunking candidates

| Strategy | What it does | Why it is included |
|---|---|---|
| Recursive character | Splits at headings, paragraphs, sentences, spaces, then characters as needed. | General boundary-aware strategy without requiring document-specific structure. |
| Token 256/32 | 256-token chunks with 32-token overlap. | Short fixed-token candidate with low context waste. |
| Token 384/64 | 384-token chunks with 64-token overlap. | Middle fixed-token candidate. |
| Token 512/64 | 512-token chunks with 64-token overlap. | Larger-context fixed-token candidate. |
| Sentence 8/2 | Groups eight sentences and overlaps two. | Linguistic-boundary alternative to token windows. |
| Semantic | Uses embedding similarity to split at topic changes. | Tests topic-aware boundaries as a complete chunker–embedding configuration. |
| Section-aware 512/64 | Respects headings, paragraphs, and lists before applying the token limit. | Tests whether document organization improves retrieval. |
| Structure-aware 512/64 | Protects tables/formulas and splits large tables at row boundaries. | Tests preservation of educational structured content. |

Semantic chunking is intentionally evaluated as a **chunker–embedding pair** because the embedding determines both the topic boundaries and retrieval vectors. This is the deployable configuration EduMind actually needs to choose. Pair results should not be interpreted as proving the isolated effect of the semantic chunker or embedding alone.

## Retrieval and reranking

### Retrieval candidates

| Strategy | What it tests |
|---|---|
| Dense | Pure semantic retrieval using the tested embedding/chunker pair. |
| BM25 | Lexical retrieval for exact terminology, names, and identifiers. |
| RRF | Fusion of dense and BM25 ranks. |
| RRF + reranker | Whether re-scoring the fused candidates improves evidence ordering enough to justify the extra cost. |

### Reranker candidates

A reranker receives the query and the top passages returned by the first retrieval stage, then assigns new relevance scores. It can improve ordering without embedding every document again, but it adds query latency and memory use.

The public screen uses a 23-model comparison published by the Ettin authors. “Author-run” means the model authors chose and executed the comparison rather than an independent leaderboard operator. The common protocol makes its rows useful for shortlisting, but author bias remains possible through candidate selection, implementation details, and tuning. EduMind therefore reproduces the comparison on its own data before selecting a reranker.

| Approximate size | Included candidate | Public result | Why it is included |
|---|---|---:|---|
| ≤200M | [`cross-encoder/ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1/tree/3b3282e9bca7a60211a8b99e2936479703151a4f) | **0.5994** mean nDCG@10 | Highest value in this approximate size group in the common table. |
| >200M–700M | [`cross-encoder/ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1/tree/5dca36282a5d85f368d2544002513a29159b4c9e) | **0.6091** | Highest value in this approximate size group. |
| >700M–1.5B | [`cross-encoder/ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1/tree/7d20e9baad17016fdf5549c08f69a2d7ca3e60c3) | **0.6114** | Highest value in this approximate size group. |
| >1.5B–4.5B | [`Qwen/Qwen3-Reranker-4B`](https://huggingface.co/Qwen/Qwen3-Reranker-4B/tree/22e683669bc0f0bd69640a1354a6d0aebcfeede5) | **0.6367** | Highest value overall in the common table. |
| Control | [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a) | **0.5082** | Established low-cost baseline. |

Shared evidence: [published comparison](https://huggingface.co/blog/ettin-reranker), [pinned comparison source](https://github.com/huggingface/blog/blob/8dc6a4f4bcdd9fe5ac2a107895b0515377691a17/ettin-reranker.md), and [MTEB English-v2 Retrieval](https://leaderboard.mteb.org/benchmark/MTEB%28eng%2C%20v2%29).

## Generation and faithfulness

No public benchmark currently compares the selected compact models under one protocol for all of EduMind's target behavior: grounded correctness, faithfulness, citations, answerability, completeness, refusal, and local latency. [ALCE](https://github.com/princeton-nlp/ALCE), [FaithJudge](https://github.com/vectara/FaithJudge), [ChatRAG-Bench](https://huggingface.co/datasets/nvidia/ChatRAG-Bench), and [FACTS Grounding](https://www.kaggle.com/benchmarks/google/facts-grounding) are relevant, but do not publish a common result for these current checkpoints.

Artificial Analysis is therefore used only to choose plausible compact quality points. Its estimated Intelligence scores do not select the final generator.

| Approximate profile | Candidate and mode | Public screening evidence | Why it is included |
|---|---|---:|---|
| ~1B | [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B/tree/87179e5c1f455ef22e6223592d2d61351b525bfc), reasoning | [Estimated AA score **12**](https://artificialanalysis.ai/models/minicpm5-1b) | Compact reasoning candidate; its quality, generated-token cost, and latency are measured locally. |
| ~3B | [`ai9stars/G9v3-3B`](https://huggingface.co/ai9stars/G9v3-3B/tree/d9553445ff92dbce667381954c9699fbcbc924f9), reasoning | [Estimated AA score **16**](https://artificialanalysis.ai/models/g9v3-3b) | Middle-size quality point and strongest scored model in the reviewed ≤4B set. |
| ~5B | [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a), reasoning | [Estimated AA score **20**](https://artificialanalysis.ai/models/qwen3-5-4b) | Upper compact quality point at approximately 4.7B total parameters. |
| Control | [`Qwen/Qwen3-1.7B`](https://huggingface.co/Qwen/Qwen3-1.7B/tree/b9352fbb8ce704292730cf54b3b1dceb2a808738), thinking disabled | Direct Hugging Face control | Smallest generator profile and common-runtime baseline. |

### Automated faithfulness diagnostic

[`vectara/hallucination_evaluation_model`](https://huggingface.co/vectara/hallucination_evaluation_model/blob/d3924deeff88f76f9203ae18d11432c400c07f41/README.md) is included as an automated diagnostic. Its model card reports **74.28% balanced accuracy** and **60.00% F1** on RAGTruth-QA. It does not replace blinded human faithfulness evaluation.

## Document extraction

Document extraction should benchmark configurations, not assume that one Docling setup is optimal. The experiment is staged so configuration choices are measured before comparing complete parser architectures.

### Docling Standard configuration matrix

All **24 combinations** of the following factors are run on the development split:

| Factor | Values | What the comparison answers |
|---|---|---|
| OCR engine | RapidOCR with ONNX Runtime; Tesseract CLI; EasyOCR | Which OCR implementation integrates best with Docling's layout and page attribution on EduMind documents? |
| OCR mode | `pdf_aware_layout_regions`; `full_page` | Is preserving usable native PDF text better than rasterizing and OCRing the complete page? |
| TableFormer mode | `fast`; `accurate` | Is the table-structure quality gain worth the additional latency? |
| Formula enrichment | off; on | Does CodeFormulaV2 improve formula recovery enough to justify its cost? |

The matrix is `3 × 2 × 2 × 2 = 24`. Testing the combinations preserves interactions—for example, an OCR engine may behave differently under full-page and PDF-aware modes.

The four factors vary because each represents a real production choice that can materially change extraction quality or cost:

- **OCR engine:** RapidOCR is the existing ONNX-based control and provides a portable neural OCR path. Tesseract is the established classical CPU baseline, showing whether the neural alternatives provide a worthwhile improvement. EasyOCR is a separate neural implementation with GPU support. Running all three inside Docling measures not only standalone recognition but also how each engine's text and bounding boxes interact with Docling's layout, page-attribution, and table processing.
- **OCR mode:** `pdf_aware_layout_regions` preserves usable native PDF text and applies OCR only to regions that require it. It is expected to suit digital and mixed PDFs while avoiding unnecessary recognition errors. `full_page` rasterizes and OCRs the complete page, which may recover scanned pages and broken text encodings but can be slower and can damage or duplicate correct native text. Both are required because EduMind handles digital, scanned, mixed, and broken-encoding PDFs.
- **TableFormer mode:** `fast` is the lower-cost table-structure path; `accurate` spends more computation on reconstructing rows, columns, and cell relationships. Comparing them establishes whether any gain in table detection, content, and structure justifies the additional latency and memory.
- **Formula enrichment:** the off configuration measures Docling's base extraction and avoids loading a specialized formula model. The on configuration uses CodeFormulaV2 to recover mathematical expressions. Both are tested because educational documents may contain formulas, but the extra model should be enabled only when formula accuracy improves enough to justify its resource cost.

These factors are evaluated together rather than one at a time because their effects can interact. Full-page OCR can change the text and coordinates supplied to TableFormer, one OCR engine may perform differently under region-based and full-page processing, and formula enrichment adds different value across document types. The full matrix identifies complete configurations rather than assuming that the best isolated setting remains best in every combination.

The following settings stay fixed:

| Setting | Fixed value | Reason |
|---|---|---|
| Docling | v2.117.0, commit `f2683c0b5aa14a53b74373b0640260891cdbc1b0` | Keeps the implementation constant. |
| OCR language | English | Matches the initial EduMind scope. |
| OCR scale | `3.0` | Uses one higher-resolution 216-DPI render for every OCR engine so rendering resolution does not confound their comparison. |
| Table cell matching | enabled | Required to map recognized cells back to the document structure. |
| Code enrichment | disabled | Code-specific extraction is not a current benchmark requirement. |
| Output | canonical `DoclingDocument` JSON | Preserves text, structure, pages, tables, formulas, and provenance in one comparable representation. |
| DOCX | native Docling ingestion | Rasterizing DOCX would confound native document parsing with PDF/image conversion. |

These settings are fixed because they define the common experimental environment or are requirements of the evaluated output, rather than useful candidate strategies:

- **Docling version:** changing the release could change models, defaults, parsing behavior, and output schemas. Pinning one commit ensures that measured differences come from the four tested factors.
- **OCR language:** the initial EduMind corpus is English. Testing additional languages would require representative multilingual data and a separate evaluation.
- **OCR scale:** `3.0` provides a common rendering resolution for every OCR engine. Adding multiple scales would turn the 24 configurations into a much larger resolution-tuning experiment. Scale should be revisited only if the results identify small-text resolution as a material failure mode.
- **Table cell matching:** this connects recognized content to table cells and is necessary for evaluating table structure. Disabling it would deliberately remove information required by the task rather than provide a credible production configuration.
- **Code enrichment:** the current corpus has no dedicated source-code requirement or annotations for indentation, symbols, and code-block correctness. It can become an on/off factor later if programming documents and suitable metrics are added.
- **Canonical output:** every configuration must produce the same `DoclingDocument` representation so text, structure, pages, tables, formulas, and provenance are compared consistently. Output format is an evaluation contract, not a quality candidate.
- **Native DOCX ingestion:** DOCX already contains machine-readable text and structure. Rasterizing it would discard that information, introduce OCR and rendering errors, and test a different extraction architecture. A rendered-DOCX fallback must be evaluated separately if corrupted DOCX files become a requirement.

The fixed values are controls for this experiment, not claims that they are universally optimal. A focused follow-up should vary one of them only when the first-stage results or a new product requirement provide a concrete reason.

### Complete architecture comparison

The Docling Standard configurations selected by the engineer advance to an end-to-end comparison:

| Candidate | Configuration | Why it is included | Evidence |
|---|---|---|---|
| Docling Standard finalist | Best measured Standard configuration from the 24-combination screen | Conventional layout/OCR/table pipeline with optional targeted formula enrichment. | [Pinned Docling release](https://github.com/docling-project/docling/releases/tag/v2.117.0); [pipeline options](https://github.com/docling-project/docling/blob/f2683c0b5aa14a53b74373b0640260891cdbc1b0/docling/datamodel/pipeline_options.py) |
| Docling VLM | `VlmPipeline` with [`ibm-granite/granite-docling-258M`](https://huggingface.co/ibm-granite/granite-docling-258M/tree/982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe) | Tests Docling's full-page visual parsing architecture rather than only changing Standard-pipeline options. | [Docling VLM documentation](https://docling-project.github.io/docling/usage/vision_models/); [model catalog](https://docling-project.github.io/docling/usage/model_catalog/) |
| PaddleOCR-VL-1.6 | [`PaddlePaddle/PaddleOCR-VL-1.6`](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.en.md), weights `c5630abae1d940eafe0697512a0325494b02ab42` | Independent 0.9B document parser and the strongest compact numerical row in the pinned OmniDocBench v1.6 table. | [Pinned OmniDocBench table](https://github.com/opendatalab/OmniDocBench/blob/193627ae9e97d89188468ed1ee3b7a856ff76044/README.md) |

PaddleOCR-VL should remain. Without it, the architecture comparison would contain only two configurations from the same Docling project. It is promoted only if EduMind's local document quality and operational results justify it.

Every architecture is normalized into the same extracted-document contract and evaluated on the same text, reading-order, page-attribution, table, formula, latency, RAM, and VRAM metrics.

## Audio extraction

The public screen uses **`avg` WER** from the pinned Open ASR English short-form results; lower is better. Because EduMind needs cited timestamps, a model also needs a verified timestamp path. The size groups below summarize the reviewed candidates and preserve different resource scales.

| Approximate size | Candidate | Public `avg` WER | Timestamp path and reason |
|---|---|---:|---|
| ≤200M | [`nvidia/canary-180m-flash`](https://huggingface.co/nvidia/canary-180m-flash/tree/b12ab418510d093e83890178fd0e8b0d0f7918a6) | **5.6914** | Compact candidate with documented word and segment timestamps. |
| >200M–800M | [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/blob/dcb0e1db8b2220830fecb8f60df74a88a34cb128/README.md) | **4.8186** | Best reviewed WER in this approximate group among models with documented timestamp output. |
| >800M–1.5B | [`OpenMOSS-Team/MOSS-Transcribe-Diarize`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/0844c4adb24300bc7c6cd91e379bc790f939f2d6/README.md) | **4.7429** | Segment timestamps and diarization are part of the documented output. |
| >1.5B–3B | [`Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/tree/bcd2b5b7f32b480ab5790554cfa8347f246a14f3) plus [`Qwen3-ForcedAligner-0.6B`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/tree/c7cbfc2048c462b0d63a45797104fc9db3ad62b7) | **4.4257** for the ASR model | Lowest reviewed WER in this group; the official aligner supplies timestamps. |
| Control | [`openai/whisper-small.en`](https://huggingface.co/openai/whisper-small.en/tree/e8727524f962ee844a7319d92be39ac1bd25655a) | Local control | Established reference implementation with timestamp output. |

Shared quality source: [Open ASR methodology](https://github.com/huggingface/open_asr_leaderboard) and the [revision-pinned English short-form result file](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results/blob/a0c08d3ac1ef99ea7148666061839b853cbfa89a/english_short_latest.csv).

The Qwen profile contains a 2.04B ASR model and a 0.6B forced aligner, or 2.64B parameters across both components. They run sequentially: transcription completes and the ASR is unloaded before alignment starts. The benchmark still measures the complete transcription-plus-alignment latency and peak resources.

Short-form public WER only creates the shortlist. Final ASR selection uses complete educational recordings and measures long-form WER, missing and hallucinated speech, timestamp MAE, real-time factor, latency, RAM, and VRAM.

## Video extraction

| Strategy | Why it is included |
|---|---|
| Fixed interval | Deterministic coverage and predictable processing cost. |
| Scene change | Reduces redundant frames by sampling visual transitions. |
| Scene change + maximum interval | Adds fallback coverage when content changes gradually without a sharp transition. |

The selected ASR and visual parser are held fixed while frame-selection strategies are compared.

## Vector database servers

The benchmark compares self-hosted network servers with the same vectors, metadata, filters, schema, query order, and client-visible latency. Vendor benchmark numbers are not used to rank them because those numbers do not hold EduMind's workload and environment constant.

| Server | Benchmark profile | Why it is included | Evidence |
|---|---|---|---|
| Chroma | `chromadb/chroma:1.5.9`; client `1.5.9` | Current HTTP-server baseline. | [Docker deployment](https://docs.trychroma.com/guides/deploy/docker) |
| Qdrant | `qdrant/qdrant:v1.17.0`; client `1.18.0` | Purpose-built HNSW server with payload indexes and filtered search. | [Installation](https://qdrant.tech/documentation/installation/); [filtering](https://qdrant.tech/documentation/guides/) |
| Weaviate | `cr.weaviate.io/semitechnologies/weaviate:1.38.2`; client `4.22.0` | Independent purpose-built HNSW server with structured filtering. | [Docker deployment](https://docs.weaviate.io/deploy/installation-guides/docker-installation) |
| PostgreSQL + pgvector | `pgvector/pgvector:0.8.2-pg17-bookworm`; Psycopg `3.3.4` | Relational and transactional design point with SQL metadata and HNSW cosine search. | [pgvector documentation](https://github.com/pgvector/pgvector/tree/v0.8.2) |
