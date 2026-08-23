# Benchmark candidate decisions

Status: **public shortlist; local feasibility screening must be completed before the runnable benchmark registry is frozen**

Selection version: **benchmark-candidates-v7**

Evidence reviewed: **2026-08-23**

Evidence artifacts:

- `selection_evidence.csv` — SHA-256 `aec760fe24b8cbb981965405ce9bf719f1910ee6b391ecbf00be8e3141a0c672`
- `source_snapshots.json` — SHA-256 `67895d55de86e565e10e1e80da57bfe45c40e31668da5a3dd0b8debcbaa3564d`
- `selection_manifest.json` — verifies the names, byte counts, and hashes of the three companion artifacts; like any checksum manifest, it does not attempt to hash itself

The Markdown stays readable; the evidence artifacts record the screened alternatives, scores, filters, source revisions where available, and dated snapshots used for the selection claims.

### How to read the evidence links

The package uses three kinds of links because they answer different questions:

- A **public benchmark link** shows the benchmark definition, tasks, metric, and current public results. It explains what was measured.
- A **pinned evidence link** opens the exact model card or result file used for a number in this document. It protects the audit from later edits to a mutable `main` page.
- A **model or project link** establishes practical facts such as the model identifier, parameter count, license, supported output, and runtime instructions. It is not ranking evidence unless it also contains the cited benchmark table.

“At revision `abc123...`” means “at the exact Git commit identified by that hash.” Hugging Face model repositories, datasets, and GitHub documentation can change after review; a revision makes the cited state reproducible. A live leaderboard link is still included for convenient browsing, while the pinned result remains the source for the recorded number.

## Model-selection rule

Competing model checkpoints have four selection paths:

1. **Ranked candidate:** define the size tier first, freeze the complete comparison table used for that screen, apply the documented task/license filters, and select the highest-ranked eligible row. This means the table is complete for the recorded comparison claim; it does not mean every model on the internet was evaluated.
2. **Unresolved challenger:** advance a model only when all of the following are recorded: exact model revision, eligible license, release before the evidence cutoff, a public task-specific result from MTEB/RTEB or another named benchmark, the same size tier as the ranked candidate, and a protocol mismatch that prevents a valid numerical comparison.
3. **Benchmark control:** retain the current or established baseline even when it is not a public leader. Controls measure whether a candidate improves on the system EduMind is replacing.
4. **Preliminary screening candidate:** when no task-specific public benchmark compares the current compact models under one protocol, use relevant but non-authoritative public evidence only to create a small local shortlist. Such a candidate is never called a public benchmark winner; EduMind's grounded local benchmark makes the actual selection.

No score from one benchmark is numerically compared with a score from another benchmark. Missing comparability is resolved only by EduMind's frozen local benchmark. The exact source set, filters, rows, revisions, and decisions are recorded in the evidence artifacts; “best” means best inside that frozen screened set, not best across every model on the internet.

For model rows, the evidence ledger uses only **include** or **exclude** as its decision. An included model can be a tier winner, a protocol-mismatched challenger, a preliminary screening candidate, or a required control. `size_tier` now contains only a parameter band or `N/A`; the separate `selection_role` field records the row's role, while `score_status` and `reason` explain its evidence and decision.

Two special cases follow their own simpler rules. HHEM is a model checkpoint, but it is included only as the required automated faithfulness diagnostic; it is never a promotion candidate or authoritative judge. Vector-database products are not model checkpoints: they are included when they are self-hosted network servers, represent a useful architecture for EduMind, and can be evaluated through the common server benchmark. Their local conformance, retrieval quality, and operational results—not a model leaderboard—decide whether they remain candidates.

### Evidence-ledger scope

`selection_evidence.csv` records only externally sourced **model checkpoints** and **vector-database products**. It does not contain rows for internally defined experiment strategies such as chunking, dense/BM25/RRF retrieval, video keyframe selection, or normalization. Those strategies are hypotheses designed by EduMind and are specified in this Markdown and the experiment documentation; they do not have an upstream model revision or public leaderboard decision to record. The Docling control configuration is documented and snapshotted here for reproducibility, but is not a CSV model-selection row.

This package selects model-backed components, complete extraction profiles, and vector-server products. It is not the candidate registry for every modality-level library or engine. The authoritative diagnostic candidate lists and procedures are maintained beside their runners in the [image](extraction/image/doc.md), [PDF](extraction/pdf/doc.md), [DOCX](extraction/docx/doc.md), [routing](extraction/routing/doc.md), and [normalization](extraction/normalization/doc.md) experiment documents. Those diagnostics inform a complete extraction profile but cannot be promoted alone as the application-wide extraction default.

In the CSV, `quality_source_state` and `task_evidence_state` contain an immutable commit or digest when the source provides one. For mutable webpages or operational profiles, the field instead contains a dated snapshot or version note. `retrieved_date` remains the observation date. These source-state fields are audit metadata, not substitutes for the machine-local runnable artifact locks required before authoritative execution.

`selection_role` uses explicit machine-readable values such as `ranked_candidate`, `unresolved_challenger`, `benchmark_control`, and `screened_alternative`. This keeps experimental role separate from model size and allows the ledger to be filtered without interpreting prose.

The license filter is deliberately narrow: an included model must be self-hostable and its published terms must allow the intended EduMind use without requiring a paid hosted service. Non-commercial-only models are excluded because a benchmark winner must remain promotable into the application. License is an eligibility gate, never a quality score, and this engineering review is not legal advice.

The size boundaries are **approximate model-size tiers used only to keep the shortlist compact and diverse**. They are not operational-cost tiers. Actual deployment cost depends on precision/quantization, architecture, embedding dimension, context length, runtime, batch size, RAM/VRAM behavior, and model residency.

### Required benchmark controls

| Component | Control | Purpose |
|---|---|---|
| Chunking | Token 256/32 | Current fixed-window chunking control. |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` at `c9745ed1d9f207416be6d2e6f8de32d1f16199bf` | Current lightweight embedding control. |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` | Established legacy cross-encoder control. |
| Generation | Ollama `qwen3:1.7b` digest `8f68893c685c3ddff2aa3fffce2aa60a30bb2da65ca488b61fff134a4d1730e7`, Q4_K_M | Current application generator control. |
| Document extraction | Configuration-frozen Docling Standard profile | Unified parser with targeted formula-VLM enrichment; full-page VLM remains disabled. |
| ASR | `openai/whisper-small.en` reference profile | Established English ASR control with timestamp output. |
| Vector database | Chroma server | Current server baseline. |

Controls are always run but never promoted merely because they are controls.

### Selection evidence versus runnable artifacts

This package freezes **why a candidate was selected**. An authoritative benchmark run additionally needs a runnable artifact lock that freezes **exactly what was executed**. The two must not be confused.

Before an authoritative run:

- every Hugging Face model must resolve to a full commit and be recorded in `data/benchmarks/models/huggingface.json` or `extraction.json`;
- every Ollama profile must record the full manifest digest, quantization, inference mode, context length, and sampling settings in `data/benchmarks/models/ollama.json`;
- every vector-server image must resolve from its versioned tag to an immutable `repo@sha256:...` digest, with matching client versions, in `data/benchmarks/models/vectordb.json`; and
- Docling must resolve to full commit `f2683c0b5aa14a53b74373b0640260891cdbc1b0`, and the RapidOCR package plus model-file checksums must be recorded.

The preparation commands create these machine-local locks. A missing item blocks an authoritative run; it is never replaced by an unrecorded latest version. `source_snapshots.json` records which freeze items are already known and which remain required.

### Component feasibility screen

Before an interactive model enters the runnable benchmark registry, its exact deployment profile must:

- complete 20 consecutive representative operations without OOM or crash;
- avoid sustained OS paging/swapping;
- avoid unintended model-weight CPU offload caused by insufficient GPU memory;
- leave at least **25% of physical RAM and 25% of physical VRAM free** at steady state; and
- record latency and throughput.

The 25% threshold is an EduMind engineering headroom rule, not an external standard.

Ingestion-time components must complete at least 10 representative files or 50 page-equivalents without OOM/crash, avoid sustained paging, remain below 90% peak RAM/VRAM utilization, and release resources after the job.

### Complete-stack deployment gate

Passing component screening independently is necessary but not sufficient. After component winners are chosen, EduMind must execute the **actual serving sequence** with the selected embedding model, vector database server, reranker, and generator using the intended model-residency/unloading policy.

This is a deployment gate, not another candidate matrix. The stack is accepted only if the real application sequence avoids OOM and sustained paging/offload behavior and preserves the required system headroom. Peak RAM/VRAM and query latency are recorded under the actual policy. If models are intentionally unloaded between stages, their reload latency is part of the measurement.

## Chunking and embeddings

### Embedding candidates

Primary public quality metric: retrieval-specific **nDCG@10 / Retrieval score**, not generic overall MTEB average.

Approximate size tiers:

- **≤350M**
- **>350M–800M**
- **>800M–1.5B**
- **>1.5B–4.5B**

| Size tier | Included candidates | Why they are included | Public benchmark |
|---|---|---|---|
| ≤350M | `Snowflake/snowflake-arctic-embed-m-v2.0` | Highest eligible English Retrieval result in the frozen ≤350M comparison table: **58.4**. Its exact model card confirms Apache-2.0 eligibility. | [MTEB model record](https://leaderboard.mteb.org/models/Snowflake/snowflake-arctic-embed-m-v2.0); [pinned IBM comparison](https://github.com/ibm-granite/granite-embedding-models/tree/250b8522ad2a7ea0c1e26f089d3de212390f614b); [pinned model card](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/blob/95c2741480856aa9666782eb4afe11959938017f/README.md) |
| >350M–800M | `Qwen/Qwen3-Embedding-0.6B`; `Octen/Octen-Embedding-0.6B`; `codefuse-ai/F2LLM-v2-0.6B` | Qwen leads the directly comparable MTEB screen (**61.83 Retrieval**). Octen has a strong RTEB result and F2LLM has official MTEB task results, but neither exposes the same frozen aggregate as Qwen; the local benchmark must resolve them. | [Qwen MTEB record](https://leaderboard.mteb.org/models/Qwen/Qwen3-Embedding-0.6B); [RTEB leaderboard](https://leaderboard.mteb.org/benchmark/RTEB%28beta%29); [F2LLM MTEB record](https://leaderboard.mteb.org/models/codefuse-ai/F2LLM-v2-0.6B) |
| >800M–1.5B | `nvidia/Nemotron-3-Embed-1B-BF16` | NVIDIA's comparable RTEB-16 table reports **72.38 average nDCG@10**, above the reviewed ~1B predecessors in that same table. The evidence ledger records **1.14B total** and approximately **872M active** parameters rather than rounding the same model differently across files. | [MTEB model record](https://leaderboard.mteb.org/models/nvidia/Nemotron-3-Embed-1B-BF16); [pinned RTEB table](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16/blame/c932836c54f75b7df5da0b0f519ea4cfd276a8e4/README.md) |
| >1.5B–4.5B | `Qwen/Qwen3-Embedding-4B`; `Octen/Octen-Embedding-4B` | Qwen reports **68.46 MTEB English-v2 Retrieval**. Octen reports **0.7747 RTEB public mean**. Those values are from different protocols, so both are included and EduMind's benchmark decides. | [Qwen MTEB record](https://leaderboard.mteb.org/models/Qwen/Qwen3-Embedding-4B); [RTEB leaderboard](https://leaderboard.mteb.org/benchmark/RTEB%28beta%29) |

F2LLM advances only as an unresolved challenger. Official task-level MTEB retrieval results exist at revision `54b4e2...`, but no directly comparable frozen English-v2 retrieval aggregate was verified. It is therefore not ranked above or below Qwen from public evidence.

Exact recorded values and model-card evidence:

- [Snowflake MTEB public model record](https://leaderboard.mteb.org/models/Snowflake/snowflake-arctic-embed-m-v2.0); the **58.4** comparison is pinned to IBM repository revision [`250b852...`](https://github.com/ibm-granite/granite-embedding-models/tree/250b8522ad2a7ea0c1e26f089d3de212390f614b), while the runnable identity and Apache-2.0 eligibility are pinned to Snowflake revision [`95c2741...`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/blob/95c2741480856aa9666782eb4afe11959938017f/README.md)
- [Qwen3-Embedding-0.6B MTEB table at revision `d43997...`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/blob/d43997c8a1046d1734f8d519effbb424a832a0a2/README.md)
- [Qwen3-Embedding-4B at revision `af551a...`](https://huggingface.co/Qwen/Qwen3-Embedding-4B/blob/af551aabe3b5e18ade93393f15c5e6d26935ccae/README.md)
- [Octen 0.6B at revision `d15b789...`](https://huggingface.co/Octen/Octen-Embedding-0.6B/blob/d15b7896589e85d23912d5a810a4cf0b8899d302/README.md)
- [Octen 4B at revision `759a644...`](https://huggingface.co/Octen/Octen-Embedding-4B/blob/759a644bb948131d6da7a743b46a2d4bd5c8a82a/README.md)
- [F2LLM-v2-0.6B at revision `54b4e2d...`](https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B/blob/54b4e2dc74e01be7126d4cf5f016af6b21edc563/README.md)
- [Nemotron-3-Embed-1B at revision `c932836...`](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16/blame/c932836c54f75b7df5da0b0f519ea4cfd276a8e4/README.md)
- `selection_evidence.csv` for the exact comparability decisions.

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

**Semantic-chunking interpretation:** non-semantic chunkers keep boundaries fixed across embeddings, so embedding comparisons are relatively clean. Semantic chunking uses the tested embedding to create the boundaries and retrieve the chunks, so each result represents a **joint chunker–embedding pair** and cannot establish an isolated embedding or chunking effect.

## Retrieval and reranking

### Retrieval candidates

| Strategy | What it tests |
|---|---|
| Dense | Pure semantic retrieval using the tested embedding/chunker pair. |
| BM25 | Lexical retrieval for exact terminology, names, and identifiers. |
| RRF | Fusion of dense and BM25 ranks. |
| RRF + reranker | Whether re-scoring the fused candidates improves evidence ordering enough to justify the extra cost. |

### Reranker candidates

The main reranker screen is a single **author-run public comparison**: MTEB English-v2 Retrieval, top-100 reranking, with mean nDCG@10 across six embedder pairings. It is not an independent official leaderboard result, but it is useful because one author evaluated all 23 rows under the same stated protocol. Despite its title, the Ettin article does **not** evaluate only Ettin: the table includes Qwen, Mixedbread, Jina, Alibaba GTE, IBM Granite, BAAI BGE, ZeroEntropy, MiniLM, and Ettin.

| Size tier | Included candidate | Public result | Why it is included |
|---|---|---:|---|
| ≤200M | `cross-encoder/ettin-reranker-150m-v1` | **0.5994** mean nDCG@10 | Highest result in its tier in the complete 23-model common table. |
| >200M–700M | `cross-encoder/ettin-reranker-400m-v1` | **0.6091** mean nDCG@10 | Highest result in its tier in the same table. |
| >700M–1.5B | `cross-encoder/ettin-reranker-1b-v1` | **0.6114** mean nDCG@10 | Highest result in its tier in the same table. |
| >1.5B–4.5B | `Qwen/Qwen3-Reranker-4B` | **0.6367** mean nDCG@10 | Highest result in its tier and overall in the same table. |

The three Ettin models appear because they won their predefined tiers in a table containing many model families, not because the screen considered only Ettin. All ranked reranker candidates now come from one common protocol, which keeps the shortlist directly comparable. The MiniLM control is also run even though it is not a tier winner.

Public evidence:

- [Full 23-model reranker comparison, published 2026-05-19](https://huggingface.co/blog/ettin-reranker); [source pinned at `8dc6a4f...`](https://github.com/huggingface/blog/blob/8dc6a4f4bcdd9fe5ac2a107895b0515377691a17/ettin-reranker.md)
- [MTEB English-v2 benchmark](https://leaderboard.mteb.org/benchmark/MTEB%28eng%2C%20v2%29)
- EduMind runnable checkpoint revisions: [Ettin 150M `3b3282e...`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1/tree/3b3282e9bca7a60211a8b99e2936479703151a4f), [Ettin 400M `5dca362...`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1/tree/5dca36282a5d85f368d2544002513a29159b4c9e), [Ettin 1B `7d20e9b...`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1/tree/7d20e9baad17016fdf5549c08f69a2d7ca3e60c3), [Qwen3 Reranker 4B `22e6836...`](https://huggingface.co/Qwen/Qwen3-Reranker-4B/tree/22e683669bc0f0bd69640a1354a6d0aebcfeede5), and [MiniLM control `233902d...`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/tree/233902d25c440f23af6f7d6e94d2946bac0bee0a). These commits pin what EduMind will execute; the article does not identify the exact checkpoint commits used to produce its public scores.
- `source_snapshots.json` contains the frozen table values used by this selection.

The public nDCG values are therefore attributed to the pinned author-run article, not to the EduMind runnable commits. Results produced locally are attributed to the runnable commits.

If Qwen3-Reranker-4B is benchmarked through a quantized conversion, the exact conversion is a separate deployment profile. A canonical full-precision public result must not be attributed automatically to that conversion.

## Generation

Task-specific grounded-QA and faithfulness evidence is preferred over a general capability score whenever it compares eligible current candidates under one protocol. No reviewed public benchmark does that for these three compact open-weight checkpoints on EduMind's complete target behavior: grounded answering from supplied evidence, faithfulness, correct refusal, citation production, answerability, completeness, and concise response quality.

Artificial Analysis is therefore used only as **preliminary general-capability screening evidence**. Scores marked estimated are explicitly treated as estimates, and mutable rank counts such as `#1/44` or `#1/45` are not used. The recorded rows are a deliberately small set of plausible compact generators, not a systematic survey of the complete catalogue. Each included profile has a direct reason below; EduMind makes no claim that unrecorded models were exhaustively screened.

Closer public benchmarks were reviewed before retaining that decision:

| Benchmark | What it measures that is useful to EduMind | Why it cannot select this shortlist |
|---|---|---|
| [ALCE](https://github.com/princeton-nlp/ALCE) | Answer correctness plus citation recall and precision on ASQA, QAMPARI, and ELI5. | Its published comparison predates and does not contain the three selected checkpoints under one protocol. |
| [FaithJudge](https://github.com/vectara/FaithJudge) | Hallucination rate across RAGTruth and FaithBench summarization, QA, and data-to-text tasks. | It contains useful older compact models, but not MiniCPM5-1B, G9v3-3B, and Qwen3.5-4B together. |
| [ChatRAG-Bench](https://huggingface.co/datasets/nvidia/ChatRAG-Bench) | Conversational QA over supplied documents, including unanswerable questions, tables, arithmetic, and long context. | It provides a benchmark dataset, but no common public result for the three selected checkpoints. |
| [FACTS Grounding](https://www.kaggle.com/benchmarks/google/facts-grounding) | Long-form responses grounded in supplied documents. | Its leaderboard does not provide a comparable compact-model tier containing the three selected checkpoints. |

These benchmarks are closer to EduMind than a general intelligence index. They cannot currently supply a common result for the selected compact checkpoints: choosing only models their authors happened to test would favor benchmark coverage rather than current model quality, while mixing scores from different protocols would create a false ranking. Public hosted latency is also not a valid proxy for the exact local quantization, runtime, context length, and hardware. Artificial Analysis is therefore used only to establish that the three profiles are credible quality points at approximately 1B, 3B, and 5B. **It does not select the final generator. EduMind's identical grounded-RAG benchmark and local latency/resource measurements do that.**

Preliminary screen:

| Role | Preliminary local candidate | Estimated AA Intelligence evidence | Why it is included |
|---|---|---:|---|
| Compact efficiency | `openbmb/MiniCPM5-1B` at `87179e5...`, reasoning | **12** | The reviewed AA table gives its reasoning and non-reasoning profiles the same estimated score. EduMind keeps only reasoning mode to test whether explicit deliberation improves grounded answers, citation selection, answerability, and completeness; its token and latency cost must still pass the local deployment gates. |
| Mid-size reasoning | `ai9stars/G9v3-3B` at `d955344...`, reasoning | **16** | It is the strongest scored row in the reviewed ≤4B table and tests whether a 3B reasoning profile earns its additional runtime cost. |
| Upper compact quality | `Qwen/Qwen3.5-4B` at `851bf6e...`, reasoning | **20** | Its direct AA page reports the strongest preliminary score of the three at 4.7B total parameters, so it is the compact quality ceiling. It is retained instead of the older Qwen3-4B-2507 profiles that AA marks deprecated in favor of Qwen3.5-4B. |

The evidence artifact records the selected representatives and the nearby alternatives that were actually reviewed. It is an auditable record of this curated screen, not an exhaustive snapshot of the live Artificial Analysis catalogue. These public values justify a **preliminary shortlist only**; EduMind's grounded-RAG benchmark selects the final generator.

**Inference-mode policy:** the public value and its reasoning/non-reasoning mode are recorded together. Each included generator is run only in the mode shown in the table. MiniCPM's two public modes are tied, and EduMind deliberately keeps only reasoning mode to evaluate its possible grounded-RAG benefit against its runtime cost. G9v3 and Qwen3.5 are also kept in the reasoning profiles for which their recorded scores were reported. Modes are never averaged, silently changed, or allowed to inherit another mode's public score.

Ollama `qwen3:1.7b` is retained as the application control independently of this public screen. Candidate profiles do not enter the runnable registry until their Ollama digest, quantization, reasoning mode, context length, and sampling settings are pinned.

Public evidence:

- [Artificial Analysis compact open-weight comparison](https://artificialanalysis.ai/models/open-source/tiny); Qwen3.5-4B is taken from its direct model page because AA reports approximately 4.7B total parameters and classifies models above 4B outside Tiny
- [MiniCPM5-1B reasoning](https://artificialanalysis.ai/models/minicpm5-1b); the reviewed table also reports 12 for its non-reasoning profile, but only reasoning mode is included
- [G9v3-3B](https://artificialanalysis.ai/models/g9v3-3b)
- [Qwen3.5-4B](https://artificialanalysis.ai/models/qwen3-5-4b)
- `source_snapshots.json` for the dated screening values.

### Automated faithfulness diagnostic

`vectara/hallucination_evaluation_model` remains an automated diagnostic, not the authoritative evaluator.

Its model card reports evaluation on human-annotated factual-consistency datasets, including RAGTruth-QA. The evidence ledger records **74.28% balanced accuracy** as the single primary scalar; the model card's **60.00% F1** remains a diagnostic value in the explanation. This supports saying **evaluated against human-annotated data**, not statistically calibrated.

- [Pinned HHEM model-card evaluation at revision `d3924de...`](https://huggingface.co/vectara/hallucination_evaluation_model/blob/d3924deeff88f76f9203ae18d11432c400c07f41/README.md)

## Document extraction

This is the production-shaped, end-to-end extraction-profile screen. It asks which complete profile should handle a document and whether routing difficult pages to a visual fallback is worthwhile. It does not rank standalone OCR engines, native PDF libraries, or DOCX parsers; those remain modality-level diagnostic experiments linked in the package-scope section and cannot by themselves become the complete extraction default.

The visual-fallback **numerical screen** contains full-document parsers with at most **1.5B parameters** in the pinned OmniDocBench v1.6 table. The 1.5B boundary defines a compact first-round fallback that tests a different architecture without immediately multiplying operational cost; it is not a claim that larger parsers are worse. PaddleOCR-VL-1.6 has the highest integrated score in that size-filtered table. Eligibility is checked separately: Paddle is the only inspected row for which the license, exact weights, self-hosted execution, and local inference path were all verified. MinerU2.5-Pro and GLM-OCR remain numerical context only. Paddle is therefore included as the single evidence-backed visual-fallback candidate, not declared the winner of a multi-candidate eligible comparison.

The extraction question remains architectural:

> **Can one exact Docling Standard profile with targeted formula-VLM enrichment satisfy EduMind's normal document ingestion requirements, or is a full-document visual fallback worth its additional latency/resource cost on difficult pages?**

### Candidate A — Docling Standard unified profile

**Decision: keep it as the required unified-parser control.** It is a good control configuration, not a claim that these settings are already optimal.

Configuration-frozen profile:

- Docling **v2.117.0**
- release commit **`f2683c0b5aa14a53b74373b0640260891cdbc1b0`**
- pipeline: **standard**, not the full-page VLM pipeline
- output: canonical `DoclingDocument` JSON
- OCR: **enabled**
- OCR engine: **RapidOCR**
- OCR language: **English**
- OCR mode: **`pdf_aware_layout_regions`**
- RapidOCR backend: **ONNX Runtime**
- OCR render scale: **3.0**
- table structure: **enabled**
- TableFormer mode: **accurate**
- table cell matching: **enabled**
- formula enrichment: **enabled** (`do_formula_enrichment=True`)
- code enrichment: **disabled**
- DOCX: native Docling ingestion; no DOCX→PDF conversion
- full-page Docling VLM pipeline: **disabled**

Formula enrichment is enabled because EduMind intends to preserve formulas and structure-aware chunking explicitly protects them. Docling documents formula enrichment as the step that analyzes formula items and extracts their LaTeX representation. It adds processing cost, so that cost is measured rather than hiding it by disabling a required capability.

The remaining choices are deliberate:

- `pdf_aware_layout_regions` keeps the PDF text layer when it is usable and sends image/non-text regions to OCR, which suits mixed digital/scanned documents better than forcing full-page OCR.
- scale `3.0` is Docling's documented default (216 DPI from a 72-DPI page render), so it is a reproducible neutral control rather than a hidden tuning result.
- TableFormer `accurate` with cell matching prioritizes table structure quality; its latency cost is measured.
- the standard pipeline isolates targeted formula-region VLM enrichment inside a conventional document parser from full-document visual parsing.

This profile should remain frozen during the first comparison. If it loses for a specific reason, a later configuration experiment may tune OCR engine, render scale, or table mode; changing them inside the candidate comparison would confound parser choice with configuration tuning.

Pinned public evidence:

- [Docling v2.117.0 release and commit `f2683c0...`](https://github.com/docling-project/docling/releases/tag/v2.117.0)
- [OCR mode, 3.0 OCR scale, RapidOCR backend, TableFormer, and enrichment option definitions at `f2683c0...`](https://github.com/docling-project/docling/blob/f2683c0b5aa14a53b74373b0640260891cdbc1b0/docling/datamodel/pipeline_options.py)
- [`pdf_aware_layout_regions` behavior at `f2683c0...`](https://github.com/docling-project/docling/blob/f2683c0b5aa14a53b74373b0640260891cdbc1b0/docling/models/base_ocr_model.py)
- [TableFormer accurate/cell-matching guidance at `f2683c0...`](https://github.com/docling-project/docling/blob/f2683c0b5aa14a53b74373b0640260891cdbc1b0/docs/usage/model_catalog.md)
- [Formula-to-LaTeX enrichment documentation at `f2683c0...`](https://github.com/docling-project/docling/blob/f2683c0b5aa14a53b74373b0640260891cdbc1b0/docs/usage/enrichments.md)
- [`CodeFormulaVlmModel` wiring in the standard PDF pipeline at `f2683c0...`](https://github.com/docling-project/docling/blob/f2683c0b5aa14a53b74373b0640260891cdbc1b0/docling/pipeline/standard_pdf_pipeline.py)

The selected Docling artifacts are frozen to `docling-project/docling-layout-heron@8f39ad3...`, `docling-project/TableFormerV2@51559fa...`, and `docling-project/CodeFormulaV2@ecedbe1...`. RapidOCR's engine package and model-file revisions are not yet frozen in this selection package. The profile is therefore **configuration-frozen, not fully artifact-frozen**. The runnable registry must pin those versions and checksums before an authoritative run; an unrecorded replacement is a different profile.

### Candidate B — specialized visual fallback

`PaddlePaddle/PaddleOCR-VL-1.6`

**Decision: keep it in the benchmark.** It is not required in production, but it is required to answer whether a specialized 0.9B full-document visual parser materially improves difficult pages containing scans, complex layouts, tables, or formulas. Removing it would leave only the unified-parser control and make the visual-fallback hypothesis untestable.

The reviewed official OmniDocBench v1.6_full table reports **96.34 overall** for PaddleOCR-VL-1.6 (0.9B), ahead of MinerU2.5-Pro (95.75) and GLM-OCR (95.22) in that integrated table.

Newer submitted/self-reported claims that are not integrated into the same official table remain evidence-follow-up items rather than being silently promoted.

The production decision is simple:

- if the frozen Docling profile meets quality requirements across EduMind's representative documents, use Docling only;
- if a clearly defined difficult visual subset is materially better under PaddleOCR-VL and the extra cost is acceptable, use it as a targeted fallback.

Its inclusion does not prejudge that decision. If it provides no meaningful local quality gain, or fails the resource/latency gate, it is not promoted.

Public evidence:

- [OmniDocBench v1.6_full table pinned at `193627a...`](https://github.com/opendatalab/OmniDocBench/blob/193627ae9e97d89188468ed1ee3b7a856ff76044/README.md)
- [PaddleOCR-VL-1.6 documentation pinned at `2661c7c...`](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.en.md); model weights `c5630ab...`
- `source_snapshots.json` for the frozen relevant OmniDocBench rows and metric formula.

## Audio extraction

Primary public quality metric: **`avg` WER** from the Open ASR Leaderboard English short-form file at pinned revision `a0c08d3...`. Lower is better.

This is a short-form selection signal, not evidence that a candidate is best on lectures or other long recordings. The Open ASR project maintains separate short-form and long-form evaluation tracks. EduMind therefore retains all four selected tier representatives plus Whisper for a local long-form screen; no ASR model may be promoted from the short-form table alone. The local screen uses complete educational recordings and reports long-form WER, missing/hallucinated speech, timestamp MAE, real-time factor, and memory. Public long-form results may be added as supporting evidence when an exact selected checkpoint appears under a compatible pinned protocol, but scores from the two tracks are never merged.

At the reviewed revision, the selected rows are:

| Size tier | Candidate | `avg` WER | Task-capability reason |
|---|---|---:|---|
| ≤200M | `nvidia/canary-180m-flash` | **5.6914** | Compact candidate with documented word/segment timestamp support. |
| >200M–800M | `nvidia/parakeet-tdt-0.6b-v2` | **4.8186** | Strongest reviewed eligible tier candidate with documented timestamp support. |
| >800M–1.5B | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | **4.7429** | Best reviewed candidate in the tier with verified timestamp output; a lower-WER row remained capability-unverified. |
| >1.5B–3B | `Qwen/Qwen3-ASR-1.7B-hf` + `Qwen/Qwen3-ForcedAligner-0.6B` | **4.4257** for the ASR checkpoint | Best reviewed candidate in the tier with a verified timestamp path through the official forced aligner. |

The Qwen entry is a **composite execution profile**. The WER belongs to the 2.04B-parameter ASR checkpoint, while timestamp production adds the 0.6B forced aligner, for 2.64B parameters across the two components. Parameter count is not converted into a guessed memory number: the benchmark records both component revisions, total model storage, each cold-load time, observed peak RAM/VRAM, whether the two models are resident together or sequentially, and complete ASR-plus-alignment latency.

Evidence is now split correctly:

- **quality evidence** → Open ASR leaderboard row;
- **task-capability evidence** → the model/aligner documentation that establishes timestamp support.

`openai/whisper-small.en` is also run as the established ASR control. “Timestamp support not documented” is recorded as **unverified**, not as proof that a model cannot produce timestamps.

Public evidence:

- [Open ASR Leaderboard (live)](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)
- [Open ASR Leaderboard methodology and source](https://github.com/huggingface/open_asr_leaderboard)
- [Open ASR results repository containing the separate short- and long-form tracks](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results/tree/a0c08d3ac1ef99ea7148666061839b853cbfa89a)
- [Open ASR result file at revision `a0c08d3...`](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results/blob/a0c08d3ac1ef99ea7148666061839b853cbfa89a/english_short_latest.csv)
- [Parakeet timestamp-support model card at revision `dcb0e1d...`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/blob/dcb0e1db8b2220830fecb8f60df74a88a34cb128/README.md)
- [MOSS timestamp/diarization model card at revision `0844c4a...`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/0844c4adb24300bc7c6cd91e379bc790f939f2d6/README.md)
- [Whisper small.en pinned at `e872752...`](https://huggingface.co/openai/whisper-small.en/tree/e8727524f962ee844a7319d92be39ac1bd25655a)
- [Canary 180M pinned at `b12ab41...`](https://huggingface.co/nvidia/canary-180m-flash/tree/b12ab418510d093e83890178fd0e8b0d0f7918a6)
- [Qwen3-ASR pinned at `bcd2b5b...`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf/tree/bcd2b5b7f32b480ab5790554cfa8347f246a14f3) and [ForcedAligner pinned at `c7cbfc2...`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B/tree/c7cbfc2048c462b0d63a45797104fc9db3ad62b7)
- `source_snapshots.json` for the frozen ASR rows.

The live leaderboard is useful for seeing newer submissions. The revision-pinned CSV is what makes the four recorded WER values reproducible: it identifies the exact result-file commit used during selection, even if the live leaderboard changes later.

If timestamp support ceases to be an EduMind requirement, the shortlist must be regenerated because the task filter changes the winner in some tiers.

## Video extraction

These are alternatives for the same job: selecting frames for downstream visual extraction.

| Strategy | Why it is included |
|---|---|
| Fixed interval | Deterministic coverage and predictable processing cost. |
| Scene change | Reduces redundant frames by sampling visual transitions. |
| Scene change + maximum interval | Adds fallback coverage when content changes gradually without a sharp scene transition. |

The downstream visual parser is held fixed while comparing frame-selection strategies.

## Normalization

Normalization occurs after extraction and before chunking.

| Strategy | Why it is included |
|---|---|
| Minimal | Unicode, line endings, and whitespace only. |
| Conservative | Repairs common extraction artifacts while minimizing content-changing edits. |
| Aggressive | Tests whether stronger cleanup is worth the higher risk of deleting or merging valid content. |

## Vector database servers

This benchmark compares **self-hosted network servers running on the benchmark machine**. Embedded/in-process modes are disabled so deployment mode does not confound the comparison. Vendor benchmarks are not used to rank the products because they rarely hold schema, filters, index settings, hardware, and client path constant. Official sources establish that each product can implement the required server workload; EduMind's conformance and retrieval benchmark provides the comparison.

| Server | Frozen first-round profile | Why it is included | Official capability/deployment source |
|---|---|---|---|
| Chroma | `chromadb/chroma:1.5.9`; Python client `1.5.9` | Current HTTP-server baseline. | [Chroma Docker server](https://docs.trychroma.com/guides/deploy/docker) |
| Qdrant | `qdrant/qdrant:v1.17.0`; Python client `1.18.0` | Purpose-built HNSW server with payload indexes and filtered search. | [Qdrant installation](https://qdrant.tech/documentation/installation/); [filter/index guidance](https://qdrant.tech/documentation/guides/) |
| Weaviate | `cr.weaviate.io/semitechnologies/weaviate:1.38.2`; Python client `4.22.0` | Independent purpose-built HNSW server with structured filtering. | [Weaviate Docker deployment](https://docs.weaviate.io/deploy/installation-guides/docker-installation) |
| PostgreSQL + pgvector | `pgvector/pgvector:0.8.2-pg17-bookworm`; Psycopg `3.3.4` | Relational/transactional design point with SQL metadata and HNSW cosine search. | [pgvector documentation](https://github.com/pgvector/pgvector/tree/v0.8.2) |

These four cover the current baseline, two purpose-built vector servers, and one relational alternative. The server benchmark—not a vendor leaderboard—selects among them.

Milvus and OpenSearch are explicitly **not included in the first round**. Milvus represents a scale/distributed-vector architecture, while OpenSearch represents a broader search-platform architecture. Both are valid self-hosted candidates, but adding them now would enlarge setup and tuning without answering a new first-round question beyond the four selected design points. Milvus is reconsidered when the tested corpus or concurrency requires its scale architecture; OpenSearch is reconsidered when search-platform features become a requirement. This is a scope decision, not a claim that either product is worse. [Milvus standalone deployment](https://milvus.io/docs/install_standalone-docker.md); [OpenSearch vector search](https://docs.opensearch.org/latest/vector-search/).

This is an architectural eligibility screen rather than one of the four model-selection paths. A product is included only when it is self-hostable as a network server, adds a distinct relevant design point, has an implementable benchmark interface, and has a selected non-prerelease server profile. The selected versions are pinned comparison profiles, not claims that each is the newest available release. Passing the benchmark's real conformance gates is required before any performance comparison or promotion.

The versioned image tags above identify the selected server releases, but tags alone are not immutable. `python experiments/benchmarks/prepare.py vectordb` must resolve them to `repo@sha256:...` digests and record the client versions before an authoritative run.
