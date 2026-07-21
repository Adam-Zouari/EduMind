# Benchmark candidate decisions

Status: **approved candidate-selection specification; executable registries synchronized**

Evidence reviewed: **2026-07-21**

This document answers two questions only:

1. Is a candidate included in the EduMind benchmark?
2. Why is it included or excluded?

Every decision is `INCLUDE` or `EXCLUDE`. This document is the approved target list. Candidate files, downloaders, dependency guidance, typed extraction output, and implemented metrics are kept synchronized with it. Application defaults remain provisional and never change automatically.

## How candidates are selected

A candidate is included when all of these are true:

- It is relevant to EduMind's actual task.
- There is credible public evidence that it is worth testing.
- It can run locally or on a self-hosted server.
- Its license permits it to become the project default.
- It answers a different question from the other candidates: quality, efficiency, architecture, or a necessary baseline.

A candidate is excluded when it has an incompatible license, duplicates another candidate without stronger evidence, is obsolete, or evaluates a feature outside EduMind's scope.

Public benchmark rank is a filter, not the final decision. A leaderboard tells us which candidates deserve an EduMind run. EduMind's frozen data decides which one works best for EduMind.

## Are the internet benchmarks reliable?

They are reliable enough to choose candidates, but none is sufficient to choose EduMind's final system.

| Public benchmark | What is reliable about it | What it does not answer for EduMind |
|---|---|---|
| [MTEB](https://huggingface.co/spaces/mteb/leaderboard) | Standard datasets, task definitions, metrics, and published result files make embedding comparisons reproducible. | Its overall score mixes retrieval with unrelated tasks. Public test sets may influence model development. It does not test EduMind's chunks, QASPER questions, evidence offsets, or 2,048-token context budget. |
| [RTEB](https://huggingface.co/blog/rteb) | Retrieval-only evaluation and private datasets make it more relevant and harder to optimize against directly. | It still does not reproduce EduMind's scientific papers, chunk boundaries, prompts, or operational environment. |
| [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) | Public evaluation code, a common normalizer, and multiple English speech datasets make WER comparisons meaningful. | It does not represent EduMind's exact educational recordings, accents, technical terms, noise, timestamp needs, or local runtime. |
| [OmniDocBench](https://github.com/opendatalab/OmniDocBench) | A peer-reviewed, annotated document benchmark with official evaluation code for text, layout, tables, formulas, and reading order. | Its combined score is not EduMind's selection rule. It does not test EduMind's image mix, provenance contract, normalization, table/formula serialization for RAG, downstream questions, or runtime. |
| [olmOCR-Bench](https://github.com/allenai/olmocr) | More than 7,000 reproducible tests over roughly 1,400 documents cover old scans, headers, multiple columns, tiny text, math, and tables. | It measures PDF linearization using category-specific tests, not EduMind's complete typed output, cell/formula provenance, downstream RAG quality, or operational environment. |
| [VectorDBBench](https://github.com/zilliztech/vectordbbench) | Public workloads cover ingestion, search, filters, concurrency, and recall. | Results depend strongly on hardware, index settings, client/server versions, and filters. It is also sponsored by Zilliz, which develops Milvus. |
| Official LLM model cards | They establish model size, license, supported inference modes, and broad capability evidence. | General knowledge and reasoning scores do not measure grounded answers, refusal, citation correctness, or the 30-second EduMind limit. |

The answer is therefore:

- We do not manually repeat public leaderboards.
- We use them to make a credible shortlist.
- We benchmark the shortlist because EduMind has a different dataset, output contract, metrics, context budget, preprocessing pipeline, and execution environment.
- If a public benchmark exactly matched all of those conditions, repeating it would be unnecessary. None currently does.

MTEB's own [leaderboard guidance](https://huggingface.co/blog/Samoed/mteb-v3-leaderboard) also recommends selecting tasks that match the use case instead of following one global rank.

## Chunking and embedding benchmark

### Embedding candidates

| Candidate | Decision | Why |
|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | **INCLUDE** | It is the current production baseline, is small, and produces 384-dimensional vectors. Its [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) documents a 256-wordpiece limit. Keeping it tells us how much newer models improve over the current inexpensive option. |
| `infgrad/Jasper-Token-Compression-600M` | **INCLUDE** | It tests a genuinely different efficiency technique: internal token compression. The model shortens the hidden token sequence before later transformer computation; it does not shorten the source text or replace chunking. Its [MIT-licensed model card](https://huggingface.co/infgrad/Jasper-Token-Compression-600M) and [technical report](https://arxiv.org/abs/2511.14405) disclose the method. The author also discloses a remaining retrieval gap and degradation beyond 1,024 tokens, which is exactly why EduMind must measure quality and speed instead of accepting the headline. |
| `Qwen/Qwen3-Embedding-0.6B` | **INCLUDE** | It is an established modern compact retrieval model and the uncompressed model-family comparison for Jasper. Qwen's [official family results](https://huggingface.co/Qwen/Qwen3-Embedding-4B) report strong MTEB English v2 retrieval and define the required query instructions. |
| `nvidia/Nemotron-3-Embed-1B-BF16` | **INCLUDE** | It is a current retrieval-focused model near the practical 1B class. NVIDIA reports RTEB 72.4 and MMTEB retrieval 71.0 in its [release report](https://huggingface.co/blog/nvidia/nemotron-3-embed-wins-rteb). Its [model card](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) defines 2,048-dimensional normalized embeddings and mandatory `query:` and `passage:` prefixes. |
| `Qwen/Qwen3-Embedding-4B` | **INCLUDE** | It is a larger, established quality candidate. Qwen's published MTEB English v2 retrieval result is 68.46, above the 0.6B member in the same evaluation. It tests whether additional embedding capacity improves EduMind enough to justify the cost. |
| `nvidia/Nemotron-3-Embed-8B-BF16` | **INCLUDE** | It is the current open retrieval quality ceiling in this list. NVIDIA reports RTEB 78.5 and MMTEB retrieval 75.5 at release, and its [model card](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16) is linked to the MTEB leaderboard. It is included to measure the maximum available quality, not because the provider result is accepted as EduMind evidence. |
| `jinaai/jina-embeddings-v5-text-small` | **EXCLUDE** | Technically it is a strong candidate: its [model card](https://huggingface.co/jinaai/jina-embeddings-v5-text-small) reports MTEB English v2 71.7 with 677M parameters. However, the weights are CC BY-NC 4.0. A benchmark winner must be usable as an unrestricted production default, so a non-commercial model is excluded. |
| `google/embeddinggemma-300m` | **INCLUDE** | It is the current small-model quality candidate rather than merely another old baseline. Google's [model card](https://huggingface.co/google/embeddinggemma-300m) reports MTEB English v2 mean-task 69.67 at 768 dimensions and supports smaller Matryoshka dimensions. It tests whether a current 300M model gives a better quality/size trade-off than MiniLM and the 600M candidates. Its Gemma license acceptance must be recorded. |
| `BAAI/bge-base-en-v1.5` | **EXCLUDE** | It is an older baseline whose role is covered by MiniLM and whose quality role is covered by the newer Qwen and Nemotron candidates. |
| `nomic-ai/nomic-embed-text-v1.5` | **EXCLUDE** | Its long-context advantage is not important for EduMind's 256-512-token chunks, and newer included candidates provide stronger retrieval evidence. |

### What each chunking technique does

#### Recursive character

The splitter tries separators in order, normally headings, paragraph breaks, newlines, sentences, spaces, and finally characters. It keeps natural boundaries when possible and falls back to smaller boundaries only when a block is too large.

Why include it: it is a robust, widely used baseline and Chroma's [chunking study](https://www.trychroma.com/research/evaluating-chunking) found that a correctly configured recursive splitter can be competitive.

#### Token 256/32

The tokenizer counts actual model tokens. Each chunk contains at most 256 tokens, and the next chunk repeats the final 32 tokens.

Why include it: it is the current default, provides precise size control, and favors short factual passages. Overlap protects facts split at a boundary but increases chunk count and duplicate context.

#### Token 384/64

This is the same sliding-token technique with larger chunks and overlap.

Why include it: it is a middle point between precise short chunks and more complete passages. Five chunks are close to the 2,048-token context budget after prompt overhead.

#### Token 512/64

Chunks hold more surrounding explanation while retaining a limited overlap.

Why include it: NVIDIA's [multi-dataset chunking study](https://developer.nvidia.com/blog/finding-the-best-chunking-strategy-for-accurate-ai-responses/) found that 256-512 tokens often work well for factoid retrieval, while larger context can help analytical questions.

#### Sentence 8/2

The splitter groups eight detected sentences and repeats the final two sentences in the next chunk.

Why include it: it never intentionally cuts a sentence in half and tests whether linguistic boundaries are more useful than fixed token boundaries. Its chunk sizes vary because sentence lengths vary.

#### Semantic

The splitter embeds consecutive sentences, measures similarity, and creates a boundary when the topic changes substantially. A fixed maximum token size prevents unbounded chunks.

Why include it: it tests whether topic boundaries improve retrieval. The boundary embedding model and threshold must be frozen for every candidate so the comparison remains about chunking rather than an undocumented model change.

#### Section-aware 512/64

The splitter uses extracted headings, paragraphs, and lists first, then applies a 512-token limit within those sections. If reliable section structure is absent, it falls back deterministically to recursive/token splitting. Tables and formulas receive no special treatment beyond their position in the section.

Why include it: it is the direct control for structure-aware chunking. Comparing the two with the same size, overlap, documents, and embedding isolates the effect of special table/formula handling instead of confusing that effect with ordinary section boundaries.

#### Structure-aware 512/64

The splitter uses extracted headings, paragraphs, lists, tables, and formulas before applying a 512-token target. A table remains atomic where the context budget permits; otherwise Markdown tables split only at row boundaries. The exact-offset contract prevents synthesizing a repeated header into a non-contiguous chunk, so header repetition is evaluated later as a representation variant instead of being hidden inside this chunker. A displayed formula remains with its identifier and nearby explanatory text when the token ceiling permits. Ordinary prose uses section boundaries and the 512/64 token rule. If reliable structure is absent, the splitter falls back deterministically to recursive/token splitting.

Why include it: educational and academic documents have meaningful sections, tables, and equations. Flattening or cutting those structures arbitrarily can preserve most characters while destroying their meaning.

### Why run the full chunker by embedder matrix?

The earlier statement about avoiding a Cartesian explosion was too strong. Screening embeddings with only one chunker can miss interactions. For example, a model with a 256-token limit may look good with 256-token chunks and worse with 512-token chunks, while a long-context model may show the opposite pattern.

The authoritative standard benchmark will therefore run all:

```text
8 chunkers x 7 included embedding models = 56 combinations
```

This is more reliable because it measures the interaction between chunk boundaries and embedding behavior. Every combination receives the same documents, questions, exact NumPy search, metrics, and paired bootstrap analysis. The full profile may run only the non-dominated combinations on larger data, but the standard selection run does not pre-eliminate combinations.

## Retrieval strategy benchmark

### What the retrieval strategies do

#### Dense retrieval

The query and chunks are converted to embedding vectors. Cosine similarity ranks chunks whose meanings are close, even when they do not share exact words.

Strength: paraphrases and semantic similarity.

Weakness: it can miss exact names, formulas written as text, identifiers, and rare technical terms.

Decision: **INCLUDE** as the semantic baseline.

#### BM25

BM25 is lexical retrieval. It ranks chunks using matching terms, term frequency, document frequency, and length normalization.

Strength: exact terminology, names, acronyms, and unusual words.

Weakness: it does not understand paraphrases well.

Decision: **INCLUDE** as the lexical baseline.

#### Reciprocal-rank fusion

Dense and BM25 each return a ranked list. RRF gives an item a score based on its rank in each list:

```text
RRF score = sum(1 / (constant + rank))
```

It combines ranks instead of adding incompatible cosine and BM25 scores.

Strength: benefits from semantic and exact-word retrieval without score calibration.

Decision: **INCLUDE**.

#### RRF followed by reranking

RRF retrieves a broad top 20. A reranker then reorders those 20 and the context packer selects the final chunks.

Strength: a more accurate model spends compute only on a small candidate set.

Weakness: additional latency and memory.

Decision: **INCLUDE**.

### What is a reranker?

An embedding model encodes the query and each chunk separately. This is fast because chunk vectors can be stored in advance. A cross-encoder reranker reads the query and one candidate chunk together, allowing every query token to interact with every chunk token. This usually gives a better relevance judgment, but it must run again for every query-candidate pair.

The reranker is not another database and stores no index. It only reorders the top results returned by dense/BM25/RRF retrieval.

### Reranker candidates

| Candidate | Decision | Why |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | **INCLUDE** | Small, established, and fast. It provides the latency baseline for cross-encoder reranking. |
| `BAAI/bge-reranker-v2-m3` | **INCLUDE** | A modern conventional 0.6B cross-encoder from a different family. Its [model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) describes a simple relevance-score interface and Apache-2.0 license. |
| `Qwen/Qwen3-Reranker-0.6B` | **INCLUDE** | Compact current quality candidate. Qwen's [provider-run table](https://huggingface.co/Qwen/Qwen3-Reranker-4B) reports MTEB-R 65.80 versus 57.03 for BGE v2-m3 under the same retrieval setup. EduMind will verify that result independently. |
| `Qwen/Qwen3-Reranker-4B` | **INCLUDE** | Quality candidate. The same Qwen table reports MTEB-R 69.76, providing a direct small-versus-large comparison within one family. |
| `Qwen/Qwen3-Reranker-8B` | **EXCLUDE** | Qwen's own table reports MTEB-R 69.02, below its 4B model's 69.76, while requiring more compute. |
| `jinaai/jina-reranker-v3` | **EXCLUDE** | Its non-commercial weight license prevents it from becoming the general production default. |

The retrieval strategies are therefore dense, BM25, RRF, and RRF followed separately by each of the four included rerankers.

## Generation benchmark

### Why these Ollama models?

Generation is selected on EduMind's grounded QA task, not by a general chat leaderboard. The list covers:

- The current baseline.
- A small and medium model from one strong family, with and without thinking.
- Independent model families to avoid selecting only Qwen.
- A larger reasoning ceiling.

| Ollama model/profile | Decision | Why |
|---|---|---|
| `qwen3:1.7b` | **INCLUDE** | It is installed and is the current production baseline. Without it, improvement cannot be measured. |
| `qwen3.5:4b` direct | **INCLUDE** | Current compact Qwen model and likely latency-oriented contender. Ollama publishes the exact [4B tag](https://ollama.com/library/qwen3.5/tags). |
| `qwen3.5:4b` thinking | **INCLUDE** | Same weights with reasoning enabled. Comparing it with direct mode isolates whether reasoning improves faithfulness/completeness enough to justify extra latency. |
| `qwen3.5:9b` direct | **INCLUDE** | Tests the effect of more capacity in the same family without reasoning overhead. |
| `qwen3.5:9b` thinking | **INCLUDE** | Tests the combined effect of more capacity and reasoning. The [official Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B) provides broad capability evidence, but EduMind decides RAG quality. |
| `gemma4:12b-it-q4_K_M` | **INCLUDE** | Current independent Google family and replacement for the older Gemma 3 candidates. Google's [model card](https://huggingface.co/google/gemma-4-12B) reports current instruction/reasoning evaluations, and Ollama publishes the exact [Q4 tag](https://ollama.com/library/gemma4%3A12b-it-q4_K_M). |
| `ministral-3:8b-instruct-2512-q4_K_M` | **INCLUDE** | Independent Mistral architecture in the middle-size range. Mistral's [model card](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) describes an Apache-2.0 edge-focused instruct model, and Ollama provides the [exact tag](https://ollama.com/library/ministral-3/tags). |
| `gpt-oss:20b` low reasoning | **INCLUDE** | Independent large open-weight quality ceiling with lower reasoning cost. See the [official model card](https://huggingface.co/openai/gpt-oss-20b) and [Ollama tag](https://registry.ollama.com/library/gpt-oss/tags). |
| `gpt-oss:20b` medium reasoning | **INCLUDE** | Same model with more reasoning effort, isolating the reasoning-quality/latency trade-off. |
| Gemma 3 4B and 12B | **EXCLUDE** | Gemma 4 is their current successor and provides the same family comparison without maintaining obsolete profiles. |
| Ministral 3 14B | **EXCLUDE** | The 8B Ministral already provides family diversity, while Gemma 12B and GPT-OSS 20B cover the larger-quality question. |
| Qwen 3.6 27B/35B | **EXCLUDE** | They duplicate the Qwen family and large-model role without answering a new EduMind hypothesis. |

Every included generation profile receives the same frozen contexts, prompt, output limit, seed, warmups, and question order. Generic benchmark scores only justify inclusion. Human grounded-answer evaluation chooses the winner.

## Automated faithfulness diagnostic

Faithfulness asks:

> Is every factual claim in the generated answer supported by the retrieved context?

It is not the same as answer correctness:

- A faithful answer can repeat incorrect information that exists in the source.
- A correct answer can be unfaithful if it uses outside knowledge not present in the supplied context.
- A complete answer must cover all required parts; faithfulness alone does not check that.
- Citation F1 checks whether citations point to relevant contexts; faithfulness checks the claims themselves.

The automated diagnostic receives a context as the premise and an answer or answer claim as the hypothesis. It returns a support score between 0 and 1. EduMind will use `vectara/hallucination_evaluation_model` (HHEM-2.1-Open) because its [model card](https://huggingface.co/vectara/hallucination_evaluation_model) evaluates RAG factual consistency on RAGTruth QA and summarization data.

| Evaluator | Decision | Why |
|---|---|---|
| HHEM-2.1-Open | **INCLUDE** | It is local, Apache-2.0, small, and trained specifically for context-supported generation. |
| Generic `cross-encoder/nli-deberta-v3-base` | **EXCLUDE** | General NLI is less directly matched to RAG hallucination detection, so it does not justify a separate authoritative metric. |
| Hosted or paid LLM judge | **EXCLUDE** | It violates the local/no-paid-judge boundary and introduces an external model whose version may change. |

Automated faithfulness remains secondary because the detector can make mistakes and may behave differently on scientific text. Its threshold must be calibrated against the blinded human labels, and the report must show agreement with humans. Human Faithfulness remains the primary metric.

## Image and complete-document extraction benchmark

### What OmniDocBench measures

[OmniDocBench](https://github.com/opendatalab/OmniDocBench) contains diverse real document pages such as academic papers, textbooks, newspapers, reports, and handwritten notes. It annotates text, layout regions, tables, formulas, and reading order and provides official end-to-end and component evaluation code. Its methodology was published at [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Ouyang_OmniDocBench_Benchmarking_Diverse_PDF_Document_Parsing_with_Comprehensive_Annotations_CVPR_2025_paper.html).

It is reliable for comparing document parsers under its dataset and scoring rules. Its overall score is not EduMind's selection score: EduMind retains text, table, formula, layout, provenance, downstream RAG, and operational measurements as separate dimensions instead of averaging them into one number.

### What olmOCR-Bench measures

[olmOCR-Bench](https://github.com/allenai/olmocr) uses more than 7,000 tests across roughly 1,400 documents. Its categories include:

- ArXiv pages.
- Old scans and old scans containing math.
- Multi-column layouts.
- Headers and footers.
- Long tiny text.
- Tables and a general base category.

It is reliable for checking whether a parser linearizes difficult PDF content. It is not a substitute for EduMind's typed table/formula output, cell and expression provenance, CER/WER, page coverage, phone-photo evaluation, downstream RAG evaluation, or local operational measurements.

### Why benchmark OCR again?

EduMind needs answers that these public benchmarks do not provide:

- Separate English text, table, and formula quality so one content type cannot hide another's failures.
- Clean scans, noisy scans, phone photos, perspective distortion, and low resolution.
- Exact page, text-block, table-cell, and formula provenance used later for citations.
- Behavior after EduMind preprocessing and normalization.
- Local latency, RAM, VRAM, cold load, and failure rates.
- Downstream degradation when extracted text, tables, and formulas are serialized and used by RAG.

### Image candidates

| Candidate | Decision | Why |
|---|---|---|
| Tesseract 5 LSTM | **INCLUDE** | Mature classic OCR baseline. It is inexpensive, transparent, and shows whether larger neural systems provide meaningful improvement. See the [official repository](https://github.com/tesseract-ocr/tesseract). |
| PP-OCRv5 English mobile detection + recognition | **INCLUDE** | Compact conventional neural OCR and the efficiency member of the Paddle family. Paddle's [release notes](https://github.com/PaddlePaddle/PaddleOCR/releases) document the English model and Windows/local support. |
| PP-OCRv5 English server detection + recognition | **INCLUDE** | Higher-capacity conventional OCR from the same family. Comparing mobile/server isolates the quality/latency trade-off without changing the overall OCR design. |
| Docling | **INCLUDE** | Structured non-generative parser that represents text, tables, equations, hierarchy, and reading order in one document object. It tests a different architecture from conventional OCR and end-to-end VLMs. See the [official project](https://github.com/docling-project/docling). |
| PP-StructureV3 | **INCLUDE** | Modular pipeline combining layout analysis, OCR, table recognition, formula recognition, chart understanding, reading-order restoration, and Markdown export. It answers whether specialized components beat a single document VLM. See the [official documentation](https://www.paddleocr.ai/main/en/version3.x/algorithm/PP-StructureV3/PP-StructureV3.html). |
| `PaddlePaddle/PaddleOCR-VL-1.6` | **INCLUDE** | Compact complete-document VLM. Its [model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) reports separate text, formula, table, and overall OmniDocBench v1.6 results and provides local inference. Provider results justify inclusion but do not decide EduMind's winner. |
| `zai-org/GLM-OCR` | **INCLUDE** | Compact MIT-licensed document VLM whose official SDK combines layout analysis with text, table, and formula recognition and structured output. Its [model card](https://huggingface.co/zai-org/GLM-OCR) provides OmniDocBench evidence and local/self-hosted inference paths. |
| `opendatalab/MinerU2.5-Pro-2605-1.2B` | **INCLUDE** | Complete document parser supporting images, PDFs, and DOCX with ordered Markdown/JSON, table-to-HTML, and formula-to-LaTeX output. It supplies a distinct end-to-end parsing pipeline. See the [official repository](https://github.com/opendatalab/MinerU) and [model card](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B). |
| `allenai/olmOCR-2-7B-1025` | **INCLUDE** | Larger Apache-2.0 parser trained further on equations, tables, and difficult OCR cases. Its [model card](https://huggingface.co/allenai/olmOCR-2-7B-1025) reports category-specific olmOCR-Bench results. It is the large quality/compute comparison. |
| `deepseek-ai/DeepSeek-OCR` | **EXCLUDE** | After adding PaddleOCR-VL, GLM-OCR, MinerU, and olmOCR, it duplicates the end-to-end document-VLM role without stronger current complete-document evidence or a unique production hypothesis. |
| docTR `fast_base` + PARSeq | **EXCLUDE** | PP-OCRv5 already supplies conventional neural OCR baselines, while the included complete parsers cover structure. docTR no longer answers a unique question. |
| Chandra 2 | **EXCLUDE** | It reports strong results, but its [repository](https://github.com/datalab-to/chandra) states that the weights have commercial self-hosting restrictions. |
| Hosted OCR APIs | **EXCLUDE** | EduMind requires local, unpaid inference. |

For Tesseract and PP-OCRv5, run raw, document, and photo preprocessing. Complete parsers use their official page pipeline and supported preprocessing; blindly applying the conventional OCR preprocessing factorial to VLMs would create an invalid comparison. Tesseract and PP-OCRv5 are valid text controls but cannot be promoted as complete defaults unless a complete candidate configuration adds structure recovery and passes the table/formula gates.

## PDF complete-document benchmark

The candidates are based on distinct PDF failure modes, not popularity:

- Digital PDFs may already contain correct text.
- Layout-aware native extraction may repair ordering.
- Mixed/scanned PDFs require OCR.
- Tables and formulas require structure-aware output rather than character recovery alone.
- Complex pages may benefit from modular document pipelines or document VLMs.
- A router may use different methods per page.

| Candidate | Decision | Why |
|---|---|---|
| pypdf native extraction | **INCLUDE** | Minimal, fast native-text baseline for digital PDFs. |
| pdfplumber native/layout extraction | **INCLUDE** | Tests whether character positions and layout-aware processing improve order and coverage over basic extraction. |
| Docling | **INCLUDE** | Complete local structured parser with page layout, reading order, table structure, formulas, and a lossless document representation. |
| PP-StructureV3 | **INCLUDE** | Modular structured pipeline and control for specialized layout, OCR, table, and formula components. |
| PaddleOCR-VL-1.6 | **INCLUDE** | Compact complete-document VLM with current OmniDocBench text/table/formula evidence and local inference. |
| `zai-org/GLM-OCR` | **INCLUDE** | Independent compact document VLM and complete SDK with structured output. |
| `opendatalab/MinerU2.5-Pro-2605-1.2B` | **INCLUDE** | End-to-end PDF parser with ordered Markdown/JSON, tables, formulas, scanned-page detection, and OCR routing. |
| `allenai/olmOCR-2-7B-1025` | **INCLUDE** | Larger PDF-parsing quality candidate. Its [model card](https://huggingface.co/allenai/olmOCR-2-7B-1025) reports 82.3 +/- 1.1 on olmOCR-Bench using the documented toolkit. |
| Page-level native/OCR hybrid | **INCLUDE** | Tests whether native extraction on usable pages and OCR on bad/scanned pages beats always-native or always-OCR processing. |
| DeepSeek-OCR | **EXCLUDE** | It duplicates the end-to-end VLM category after adding three stronger compact structured candidates and olmOCR's large quality comparison. |
| Marker | **EXCLUDE** | It adds licensing and deployment constraints without a necessary role after Docling, PP-StructureV3, PaddleOCR-VL, GLM-OCR, MinerU, and olmOCR are included. |
| Chandra 2 | **EXCLUDE** | Commercial self-hosting restrictions prevent unrestricted promotion. |

The PDF benchmark scores the complete output while keeping dimensions separate:

- Text: CER, WER, coverage, block accuracy, and reading order.
- Tables: detection, cell content, row/column/header relations, TEDS/TEDS-S, and deterministic Markdown/HTML serialization.
- Formulas: detection, normalized LaTeX match/edit distance, symbol and expression-structure accuracy, and CDM where the reproducible evaluator is available.
- Document integrity: page attribution, bounding boxes, source offsets, missing/duplicate elements, and deterministic ordering.
- Downstream use: retrieval and answer quality stratified by text-, table-, formula-, and mixed-evidence questions.
- Operations: latency, throughput, cold load, RAM, VRAM, temporary disk, and failure rate.

EduMind does not use one weighted extraction score. A complete default must pass minimum gates for text, table, formula, provenance, and operational correctness; Pareto comparison follows only after those gates.

## DOCX complete-document benchmark

There is no accepted public leaderboard matching EduMind's DOCX output contract, so selection is based on distinct parsing approaches.

| Candidate | Decision | Why |
|---|---|---|
| python-docx | **INCLUDE** | Direct OOXML baseline with little hidden processing. |
| Mammoth | **INCLUDE** | Converts DOCX semantics to HTML/text and may preserve headings/lists better than raw paragraph traversal. |
| Docling | **INCLUDE** | Unified structured representation that preserves tables, equations, hierarchy, and reading order across DOCX and PDF. |
| MinerU | **INCLUDE** | Native DOCX-capable complete parser with ordered structured output, tables, and formulas; it supplies an independent end-to-end comparison. |
| Unstructured DOCX partitioner | **EXCLUDE** | Once table and formula fidelity are required, Docling and MinerU cover the element-oriented complete-parser role with stronger structured-output evidence. |

EduMind's verified DOCX dataset is the final evidence because a relevant public leaderboard does not exist. DOCX evaluation uses native OOXML as the reference and includes table-cell relations and Office Math expressions; candidates must not gain apparent accuracy by rendering every DOCX to an image unless that rendering is explicitly part of the measured candidate.

### Effect on the RAG candidate list

Adding tables and formulas changes the data and representation benchmark before it changes the embedding or generation shortlist:

- The extraction contract must retain typed text, table, and formula elements with page, order, bounding boxes, and source provenance.
- The chunking benchmark includes the structure-aware strategy defined above and evaluates Markdown tables, HTML tables, row-oriented text, normalized LaTeX, and formula-plus-explanation representations.
- The RAG questions must be stratified into text, table, formula, and mixed-evidence cases so aggregate retrieval scores cannot hide a failure category.
- Every existing embedding candidate receives the same representation variants. A specialized mathematical or multimodal embedding is added only if the text/LaTeX candidates demonstrably fail and the new model answers that specific hypothesis.
- The generation shortlist remains unchanged, but complete-RAG evaluation adds table reasoning, formula preservation, structured citation correctness, and appropriate refusal cases.
- Automated faithfulness remains diagnostic because a generic detector may not reliably judge mathematical equivalence or table relationships; blinded human review and deterministic structured checks remain primary.

This preserves a clean causal experiment: first select extraction and serialization, then measure retrieval, then measure grounded generation. It avoids attributing a broken table representation to the embedding model or LLM.

## Audio-to-text benchmark

The [Open ASR Leaderboard](https://github.com/huggingface/open_asr_leaderboard) supplies the initial evidence. All included models still run on EduMind's English educational audio because technical terms, noise, accents, timestamps, hallucinated speech, and local speed are application-specific.

| Candidate | Decision | Why |
|---|---|---|
| OpenAI Whisper `small.en` reference | **INCLUDE** | Reference implementation needed to separate model behavior from faster-whisper runtime behavior. |
| faster-whisper `tiny.en` int8 | **INCLUDE** | Lowest-cost Whisper point. |
| faster-whisper `base.en` int8 | **INCLUDE** | Current application baseline and middle efficiency point. |
| faster-whisper `small.en` int8 | **INCLUDE** | Stronger Whisper model under memory-saving quantization. |
| faster-whisper `small.en` float16 | **INCLUDE** | Same model family without int8, isolating precision effects. |
| faster-whisper `turbo` int8 | **INCLUDE** | Whisper-family quality/speed ceiling. |
| `distil-whisper/distil-large-v3.5` | **INCLUDE** | Distilled Whisper efficiency candidate. Its [model card](https://huggingface.co/distil-whisper/distil-large-v3.5) reports Open ASR mean WER 7.21 and its documented speed/accuracy comparison. |
| `nvidia/parakeet-tdt-0.6b-v3` | **INCLUDE** | Non-Whisper FastConformer/TDT architecture, preventing a Whisper-only conclusion. Its [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) includes Open ASR evaluation. |
| `nvidia/canary-qwen-2.5b` | **INCLUDE** | Open ASR quality candidate. Its [model card](https://huggingface.co/nvidia/canary-qwen-2.5b) reports mean WER 5.63 with the leaderboard normalizer. |
| Whisper large-v3 | **EXCLUDE** | Turbo, Distil-Whisper, and Canary already cover high-quality ASR while providing more useful efficiency or architecture comparisons. |

VAD off/on is tested only after model selection because VAD is a decoding/preprocessing strategy, not a separate ASR model.

## Video-to-text benchmark

Video extraction combines audio transcription with visual text extraction.

| Candidate | Decision | Why |
|---|---|---|
| Fixed-interval keyframes | **INCLUDE** | Simple deterministic baseline; it can catch text even when there is no large visual scene change. |
| Scene-change keyframes | **INCLUDE** | Avoids processing nearly identical frames and focuses OCR on visual transitions. |
| Scene-change plus maximum-interval fallback | **INCLUDE** | Prevents missing slowly changing slides or persistent text while retaining scene-change efficiency. |

The selected OCR and ASR models from their own benchmarks are used here. Re-running every OCR model against every ASR model would repeat component selection rather than answer a video-specific question.

## Normalization and routing

### What normalization means

Extraction engines return imperfect text. Normalization is the deterministic post-processing between extraction and chunking. It can:

- Normalize Unicode forms.
- Normalize line endings and repeated whitespace.
- Repair safe encoding artifacts.
- Join words split by end-of-line hyphenation.
- Remove repeated page headers and footers.

Normalization is dangerous because an aggressive cleaner may delete real content, join separate words, or alter evidence offsets. That is why "cleaner-looking text" is not enough.

| Normalization strategy | Decision | Why |
|---|---|---|
| Minimal | **INCLUDE** | Safe baseline: Unicode, line endings, and whitespace only. |
| Conservative | **INCLUDE** | Adds repairs only when deterministic evidence supports them, such as a repeated header on many pages or a word split at a line boundary. |
| Aggressive | **INCLUDE** | Deliberate challenger used to discover whether extra cleanup helps enough to justify accidental-deletion risk. It is rejected if preservation gates fail. |

The benchmark uses Content Preservation Recall, Corruption Removal Precision/Recall/F1, accidental deletion/merge rate, determinism, and latency. It needs verified clean/corrupted pairs because no external leaderboard knows which characters EduMind is allowed to change.

### What routing means

Routing chooses which PDF extraction path processes a document or page.

Examples:

- A normal digital page should usually use native extraction.
- A scanned page with no text layer needs OCR.
- A PDF with broken encoded text may contain characters but still need OCR.
- A mixed PDF may need native extraction on some pages and OCR on others.

| Routing strategy | Decision | Why |
|---|---|---|
| Always native | **INCLUDE** | Fast control and correct policy for clean digital PDFs. |
| Always OCR | **INCLUDE** | Control for scanned documents and a way to measure unnecessary OCR cost on digital pages. |
| Document-level router | **INCLUDE** | Classifies the complete PDF as digital, scanned, or mixed and applies one main policy. |
| Page-level hybrid router | **INCLUDE** | Evaluates every page and can combine native and OCR output in one document. This should be strongest on mixed PDFs but costs more routing logic. |

Router Selection Accuracy measures whether the router chose the verified best path. Oracle regret measures how much quality was lost compared with an imaginary oracle that always chooses the best extractor for each page:

```text
quality regret = oracle quality - router quality
```

Fallback Success Rate measures whether a failed or unusable primary extraction is recovered by the fallback. Routing is benchmarked because a strong PDF default is likely a policy, not one universally best extractor.

## Vector database benchmark

This benchmark selects server software, not a model. The initial list remains deliberately small while covering four distinct architectures.

| Server | Decision | Why |
|---|---|---|
| Chroma server | **INCLUDE** | Current provisional baseline. |
| Qdrant server | **INCLUDE** | Purpose-built vector server with filtering and hybrid/sparse capabilities. |
| Weaviate | **INCLUDE** | Purpose-built vector server with native BM25 and hybrid retrieval. |
| PostgreSQL + pgvector | **INCLUDE** | Transactional relational alternative using SQL metadata, PostgreSQL text search, and vector indexes. |
| Milvus | **EXCLUDE** | Its primary additional hypothesis is larger-scale/distributed vector operation. EduMind's current experiment does not need that extra system to compare the main architectural categories. |
| OpenSearch | **EXCLUDE** | It is valuable when EduMind requires a general search/analytics platform, but that is not the current question. |
| Embedded databases | **EXCLUDE** | The project decision is server-only retrieval. |

Public database benchmarks select credible servers, but EduMind must run the comparison because recall, filtering, latency, concurrency, memory, and storage change with versions, indexes, dimensions, metadata, and hardware.

## Final included lists

### Embeddings

- MiniLM L6 v2
- EmbeddingGemma 300M
- Jasper Token Compression 600M
- Qwen3 Embedding 0.6B
- Nemotron 3 Embed 1B BF16
- Qwen3 Embedding 4B
- Nemotron 3 Embed 8B BF16

### Chunkers

- Recursive character
- Token 256/32
- Token 384/64
- Token 512/64
- Sentence 8/2
- Semantic
- Section-aware 512/64
- Structure-aware 512/64

### Retrieval and reranking

- Dense
- BM25
- RRF
- RRF + MiniLM reranker
- RRF + BGE v2-m3 reranker
- RRF + Qwen3 0.6B reranker
- RRF + Qwen3 4B reranker

### Generation profiles

- Qwen3 1.7B
- Qwen3.5 4B direct
- Qwen3.5 4B thinking
- Qwen3.5 9B direct
- Qwen3.5 9B thinking
- Gemma 4 12B Q4
- Ministral 3 8B Instruct Q4
- GPT-OSS 20B low reasoning
- GPT-OSS 20B medium reasoning

### Image extraction

- Tesseract 5
- PP-OCRv5 English mobile
- PP-OCRv5 English server
- Docling
- PP-StructureV3
- PaddleOCR-VL-1.6
- GLM-OCR
- MinerU2.5-Pro 1.2B
- olmOCR 2 7B

### PDF extraction

- pypdf
- pdfplumber
- Docling
- PP-StructureV3
- PaddleOCR-VL-1.6
- GLM-OCR
- MinerU2.5-Pro 1.2B
- olmOCR 2 7B
- Page-level native/OCR hybrid

### DOCX extraction

- python-docx
- Mammoth
- Docling
- MinerU

### Audio extraction

- OpenAI Whisper small.en
- faster-whisper tiny.en int8
- faster-whisper base.en int8
- faster-whisper small.en int8
- faster-whisper small.en float16
- faster-whisper turbo int8
- Distil-Whisper large v3.5
- Parakeet TDT 0.6B v3
- Canary-Qwen 2.5B

### Vector database servers

- Chroma
- Qdrant
- Weaviate
- PostgreSQL + pgvector

## Implementation consequences

The runnable YAML files, preparation commands, typed extraction contract, transparent table/formula metrics, and per-experiment documents now match this list. Authoritative extraction runs still require frozen licensed manifests with verified table/formula annotations and any official OmniDocBench evaluators claimed in the report. Metrics absent from the annotations remain omitted rather than filled with placeholder zeroes. Benchmark results never silently change application defaults.
