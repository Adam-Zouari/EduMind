# Benchmark candidate selection

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Benchmark overview](overview.md) · [Benchmark manual](methodology.md) ·
[Selection evidence](../../experiments/benchmarks/selection_evidence.csv)

Status: **public-evidence shortlist; EduMind's local benchmarks inform the engineer's final decisions**

Selection package: **benchmark-candidates**

Evidence reviewed: **2026-09-14**

## How candidates are chosen

Candidate selection follows the same practical sequence for each component:

1. **Define the job.** Retrieval models are screened on retrieval quality, rerankers on reranking quality, ASR models on English transcription plus timestamps, and generators on their suitability for grounded answering.
2. **Inspect relevant public evidence.** Prefer a common task-specific benchmark such as MTEB Retrieval, a common reranker comparison, OmniDocBench, or the Open ASR Leaderboard. General-capability evidence is used only when no current task-specific comparison covers the candidate set.
3. **Apply basic eligibility checks.** A candidate must be downloadable, self-hostable, usable for the intended EduMind deployment, and expose the capability required by the experiment.
4. **Enforce the target-hardware envelope.** Authoritative ASR, embedding, reranking, and generation candidates must run on the laptop's RTX 3050 4 GiB GPU in the stage's frozen supported 16-bit dtype without CPU offload, automatic device splitting, quantization, or silent fallback. Until hardware qualification is complete, every model processes one inference input at a time; runtimes with a batch setting use batch size `1`. NVML-measured peak process VRAM must not exceed 3,584 MiB, preserving at least 512 MiB of the physical 4,096 MiB device for driver/runtime variation. The check uses the longest applicable development input and frozen output limit and is repeated during the real development run. CPU or CUDA may be used for smoke and debugging, but those runs cannot support selection. Models that fail the authoritative gate remain recorded as exclusions rather than receiving a different execution protocol.
5. **Keep different feasible resource scales.** After eligibility screening, the remaining models are organized into approximate parameter-size groups so the local benchmark compares compact, middle, and higher-quality options that can actually be deployed on the target hardware. A size range may remain empty when no reviewed model passes the hardware gate. These groups describe the reviewed shortlist; they were not fixed before the search.
6. **Prefer comparable evidence.** When candidates were tested under the same public protocol, the strongest eligible representatives are kept. When promising models use incompatible protocols, both may be kept and compared locally instead of comparing unlike public scores.
7. **Keep an independent control.** Use the current baseline when it is eligible;
   otherwise use a deliberately lightweight lower bound that is not another mode
   of a candidate checkpoint. This shows whether candidate complexity earns a
   meaningful improvement.
8. **Decide locally.** Public evidence creates the shortlist. An engineer reviews EduMind's frozen-dataset quality, latency, and resource measurements to make the final decision.

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
| `public_metric` | Metric represented by `public_score`, including its unit when needed. | `Retrieval` or `avg WER (%)` |
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
| Document extraction | Docling Standard baseline configuration | Current unified-parser reference. |
| ASR | `openai/whisper-small.en` at `e8727524f962ee844a7319d92be39ac1bd25655a` | Established English ASR reference. |
| Chunking | Token 256/32 | Current fixed-window chunking. |
| Embedding | [`Alibaba-NLP/gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base/tree/e7f32e3c00f91d699e8c43b53106206bcc72bb22) at `e7f32e3c00f91d699e8c43b53106206bcc72bb22` | Provisional 149M lightweight control with an 8,192-token input limit, 768-dimensional CLS-pooled normalized embeddings, and an Apache 2.0 license. Its official card reports 55.33 average nDCG@10 on 15 BEIR retrieval datasets. |
| Reranking | [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base/tree/f7481e6055501a30fb19d090657df9ec1f79ab2c) at `f7481e6055501a30fb19d090657df9ec1f79ab2c` | English 149M long-context control with an 8,192-token input limit and Apache 2.0 license. Its `0.5843` result in the shared public comparison is below all three selected learned rerankers. |
| Vector database | Chroma server | Current server baseline. |
| Generation | [`tiiuae/Falcon-H1-Tiny-R-90M`](https://huggingface.co/tiiuae/Falcon-H1-Tiny-R-90M/tree/7385612bf04c64405a51b29b6229d6d2ab0e72fd), reasoning | Deliberately weak 91M reasoning control executed through the same Hugging Face runtime and resource protocol as the candidates. |

Controls are run alongside the candidates; being a control does not make a component the final choice.

## Document extraction

The benchmark includes three complete parser architectures. The selected
Docling Standard profile comes from the development configuration screen defined
in [methodology.md](methodology.md); configuration values and experiment design
are intentionally documented there rather than repeated in this candidate
selection record.

| Candidate | Configuration | Why it is included | Evidence |
|---|---|---|---|
| Docling Standard finalist | Best measured Standard configuration from the 24-combination screen | Conventional layout/OCR/table pipeline with optional targeted formula enrichment. | [Pinned Docling release](https://github.com/docling-project/docling/releases/tag/v2.117.0); [pipeline options](https://github.com/docling-project/docling/blob/f2683c0b5aa14a53b74373b0640260891cdbc1b0/docling/datamodel/pipeline_options.py) |
| Docling VLM | `VlmPipeline` with [`ibm-granite/granite-docling-258M`](https://huggingface.co/ibm-granite/granite-docling-258M/tree/982fe3b40f2fa73c365bdb1bcacf6c81b7184bfe) | Tests Docling's full-page visual parsing architecture rather than only changing Standard-pipeline options. | [Docling VLM documentation](https://docling-project.github.io/docling/usage/vision_models/); [model catalog](https://docling-project.github.io/docling/usage/model_catalog/) |
| PaddleOCR-VL-1.6 | [`PaddlePaddle/PaddleOCR-VL-1.6`](https://github.com/PaddlePaddle/PaddleOCR/blob/2661c7c0ef5c613e8f93c6e93b2e052399f0f854/docs/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.en.md), weights `c5630abae1d940eafe0697512a0325494b02ab42` | Adds an independent 0.9B document-parser architecture instead of comparing only two configurations from the Docling project; it is also the strongest compact numerical row in the pinned OmniDocBench v1.6 table. | [Pinned OmniDocBench table](https://github.com/opendatalab/OmniDocBench/blob/193627ae9e97d89188468ed1ee3b7a856ff76044/README.md) |

Every architecture is normalized into the same extracted-document contract and evaluated on the same text, reading-order, page-attribution, table, formula, latency, RAM, and VRAM metrics.

## Audio extraction

The public screen uses **`avg` WER (%)** from the pinned Open ASR English short-form results; lower is better. Because EduMind needs cited timestamps, a model also needs a verified timestamp path. The size groups below summarize the reviewed candidates and preserve different resource scales.

| Approximate size | Candidate | Public `avg` WER (%) | Timestamp path and reason |
|---|---|---:|---|
| ≤200M | [`nvidia/canary-180m-flash`](https://huggingface.co/nvidia/canary-180m-flash/tree/b12ab418510d093e83890178fd0e8b0d0f7918a6) | **5.6914** | Compact candidate with documented word and segment timestamps. |
| >200M–800M | [`nvidia/parakeet-tdt-0.6b-v2`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2/blob/dcb0e1db8b2220830fecb8f60df74a88a34cb128/README.md) | **4.8186** | Best reviewed WER in this approximate group among models with documented timestamp output. |
| >800M–1.5B | [`OpenMOSS-Team/MOSS-Transcribe-Diarize`](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/0844c4adb24300bc7c6cd91e379bc790f939f2d6/README.md) | **4.7429** | Segment timestamps and diarization are part of the documented output. |
| Control | [`openai/whisper-small.en`](https://huggingface.co/openai/whisper-small.en/tree/e8727524f962ee844a7319d92be39ac1bd25655a) | Local control | Established reference implementation with timestamp output. |

Shared quality source: [Open ASR methodology](https://github.com/huggingface/open_asr_leaderboard) and the [revision-pinned English short-form result file](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard-results/blob/a0c08d3ac1ef99ea7148666061839b853cbfa89a/english_short_latest.csv).

Short-form public WER only creates the shortlist. Final ASR evaluation follows
the frozen educational-audio procedure in [methodology.md](methodology.md) and
the metric contract in [metrics.md](metrics.md).

## Video extraction

Video extraction introduces no additional model candidate. It freezes the
selected ASR and document parser, then compares the fixed-interval, scene, and
hybrid keyframe-selection configurations defined in
[methodology.md](methodology.md). This prevents model changes from being
mistaken for improvements in frame selection.

## Chunking and embeddings

### Embedding candidates

Primary public quality metric: retrieval-specific **nDCG@10 / Retrieval score**, not generic overall MTEB average.

Approximate size groups observed in the reviewed shortlist:

- **≤350M**
- **>350M–800M**
- **>800M–1.5B**

| Approximate size | Included candidates | Why they are included | Evidence |
|---|---|---|---|
| ≤350M | `Alibaba-NLP/gte-modernbert-base` control; `Snowflake/snowflake-arctic-embed-m-v2.0` candidate | GTE is the lightweight long-context production control and reports **55.33 BEIR-15 average nDCG@10**. Snowflake is the strongest commercially usable model in the separate frozen English MTEB-v2 screening table (**58.4 Retrieval**). These public protocols are not used to rank the two models against each other. | [pinned GTE model card](https://huggingface.co/Alibaba-NLP/gte-modernbert-base/blob/e7f32e3c00f91d699e8c43b53106206bcc72bb22/README.md); [Snowflake MTEB record](https://leaderboard.mteb.org/models/Snowflake/snowflake-arctic-embed-m-v2.0); [pinned comparison](https://github.com/ibm-granite/granite-embedding-models/tree/250b8522ad2a7ea0c1e26f089d3de212390f614b); [pinned Snowflake model card](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/blob/95c2741480856aa9666782eb4afe11959938017f/README.md) |
| >350M–800M | `Qwen/Qwen3-Embedding-0.6B`; `Octen/Octen-Embedding-0.6B`; `codefuse-ai/F2LLM-v2-0.6B` | Qwen leads the directly comparable MTEB screen (**61.83 Retrieval**). Octen has strong RTEB evidence and F2LLM has official MTEB task results, but their available aggregates are not directly comparable to Qwen's frozen result. | [Qwen MTEB](https://leaderboard.mteb.org/models/Qwen/Qwen3-Embedding-0.6B); [Octen/RTEB](https://leaderboard.mteb.org/benchmark/RTEB%28beta%29); [F2LLM MTEB](https://leaderboard.mteb.org/models/codefuse-ai/F2LLM-v2-0.6B) |
| >800M–1.5B | `nvidia/Nemotron-3-Embed-1B-BF16` | NVIDIA's common RTEB-16 table reports **72.38 average nDCG@10**, above the reviewed nearby-size models in that table. | [MTEB record](https://leaderboard.mteb.org/models/nvidia/Nemotron-3-Embed-1B-BF16); [pinned RTEB table](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16/blame/c932836c54f75b7df5da0b0f519ea4cfd276a8e4/README.md) |

The GTE ModernBERT control is evaluated with every chunker. Public evidence sources and exact candidate revisions are recorded in `selection_evidence.csv`.

The benchmark freezes `cl100k_base` as its common chunking and scoring tokenizer.
Tokenizers are controls, not candidates: allowing each embedding model to redefine
chunk boundaries would confound tokenizer, chunker, and embedding effects. Each
embedding model still uses its own native tokenizer to prepare inference inputs
and enforce its input-length contract. A future multilingual benchmark version
may run a separate tokenizer-sensitivity study before freezing a new protocol.

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

| First-stage retriever | What it tests |
|---|---|
| Dense | Pure semantic retrieval using the selected embedding/chunker pair. |
| BM25 | Lexical retrieval for exact terminology, names, and identifiers. |
| RRF | Equal-weight reciprocal-rank fusion of the Dense and BM25 top-20 lists using fusion constant `60`. |

These are frozen, untuned protocol controls. Dense uses exact cosine over
L2-normalized vectors so approximate-index behavior cannot affect this phase.
BM25 uses the selected implementation's explicit `BM25Okapi` defaults:
`k1=1.5`, `b=0.75`, and `epsilon=0.25`. RRF uses equal source weights because
Dense cosine and BM25 scores are not directly comparable and no development-set
evidence justifies favoring either source. It reuses the checksummed Dense and
BM25 indexes and does not build a third search index.

Each first-stage retriever is evaluated with five reranker options: `none` and
each of the four learned rerankers below. This full crossing produces 15 complete
`retriever|reranker` candidates. Reranking is therefore not limited to RRF, and
the Dense, BM25, and RRF no-reranker candidates remain the direct controls for
measuring each reranker's incremental effect.

### Reranker candidates

A reranker receives the query and the top passages returned by the first retrieval stage, then assigns new relevance scores. It can improve ordering without embedding every document again, but it adds query latency and memory use.

The public screen uses a 23-model comparison published by the Ettin authors. “Author-run” means the model authors chose and executed the comparison rather than an independent leaderboard operator. The common protocol makes its rows useful for shortlisting, but author bias remains possible through candidate selection, implementation details, and tuning. EduMind therefore reproduces the comparison on its own data before selecting a reranker.

| Approximate size | Included candidate | Public result | Why it is included |
|---|---|---:|---|
| ≤200M | [`cross-encoder/ettin-reranker-150m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1/tree/3b3282e9bca7a60211a8b99e2936479703151a4f) | **0.5994** mean nDCG@10 | Highest value in this approximate size group in the common table. |
| >200M–700M | [`cross-encoder/ettin-reranker-400m-v1`](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1/tree/5dca36282a5d85f368d2544002513a29159b4c9e) | **0.6091** | Highest value in this approximate size group. |
| >700M–1.5B | [`cross-encoder/ettin-reranker-1b-v1`](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1/tree/7d20e9baad17016fdf5549c08f69a2d7ca3e60c3) | **0.6114** | Highest value in this approximate size group. |
| Control | [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base/tree/f7481e6055501a30fb19d090657df9ec1f79ab2c) | **0.5843** | Weaker than every selected reranker in the same public comparison while providing an 8,192-token input limit compatible with the benchmark's chunking contract. |

Shared evidence: [published comparison](https://huggingface.co/blog/ettin-reranker), [pinned comparison source](https://github.com/huggingface/blog/blob/8dc6a4f4bcdd9fe5ac2a107895b0515377691a17/ettin-reranker.md), and [MTEB English-v2 Retrieval](https://leaderboard.mteb.org/benchmark/MTEB%28eng%2C%20v2%29).

## Vector database servers

The benchmark compares self-hosted network servers with the same vectors, metadata, filters, schema, query order, and client-visible latency. Vendor benchmark numbers are not used to rank them because those numbers do not hold EduMind's workload and environment constant.

| Server | Benchmark profile | Why it is included | Evidence |
|---|---|---|---|
| Chroma | `chromadb/chroma:1.5.9`; client `1.5.9` | Current HTTP-server baseline. | [Docker deployment](https://docs.trychroma.com/guides/deploy/docker) |
| Qdrant | `qdrant/qdrant:v1.17.0`; client `1.18.0` | Purpose-built HNSW server with payload indexes and filtered search. | [Installation](https://qdrant.tech/documentation/installation/); [filtering](https://qdrant.tech/documentation/guides/) |
| Weaviate | `cr.weaviate.io/semitechnologies/weaviate:1.38.2`; client `4.22.0` | Independent purpose-built HNSW server with structured filtering. | [Docker deployment](https://docs.weaviate.io/deploy/installation-guides/docker-installation) |
| PostgreSQL + pgvector | `pgvector/pgvector:0.8.2-pg17-bookworm`; Psycopg `3.3.4` | Relational and transactional design point with SQL metadata and HNSW cosine search. | [pgvector documentation](https://github.com/pgvector/pgvector/tree/v0.8.2) |

## Generation and faithfulness

No public benchmark currently compares the selected compact models under one protocol for all of EduMind's target behavior: grounded correctness, faithfulness, citations, answerability, completeness, refusal, and local latency. [ALCE](https://github.com/princeton-nlp/ALCE), [FaithJudge](https://github.com/vectara/FaithJudge), [ChatRAG-Bench](https://huggingface.co/datasets/nvidia/ChatRAG-Bench), and [FACTS Grounding](https://www.kaggle.com/benchmarks/google/facts-grounding) are relevant, but do not publish a common result for these current checkpoints.

Artificial Analysis is therefore used only to choose plausible compact quality
points. Its current Intelligence Index gives the three candidates one common
screening reference, but it does not evaluate the control and does not select
the final generator. The control instead has a direct same-protocol comparison
with Qwen3-0.6B in the Falcon-H1-Tiny technical report.

| Approximate profile | Candidate and mode | Public screening evidence | Why it is included |
|---|---|---:|---|
| ~0.6B | [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), reasoning | [AA score **5**](https://artificialanalysis.ai/models/qwen3-0.6b-instruct-reasoning) | Small established reasoning candidate and first candidate scale above the control. |
| ~0.8B | [`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17), reasoning | [AA score **6**](https://artificialanalysis.ai/models/qwen3-5-0-8b) | Newer intermediate reasoning candidate that tests whether its modest quality increase is worth its latency and token cost. |
| ~1B | [`openbmb/MiniCPM5-1B`](https://huggingface.co/openbmb/MiniCPM5-1B/tree/87179e5c1f455ef22e6223592d2d61351b525bfc), reasoning | [AA score **9**](https://artificialanalysis.ai/models/minicpm5-1b) | Strongest candidate in the reviewed hardware-feasible range on the common public screen. |
| Control | [`tiiuae/Falcon-H1-Tiny-R-90M`](https://huggingface.co/tiiuae/Falcon-H1-Tiny-R-90M/tree/7385612bf04c64405a51b29b6229d6d2ab0e72fd), reasoning | [Direct reasoning comparison](https://tiiuae-tiny-h1-blogpost.hf.space/) | Independent 91M lower-bound control. The report evaluates it and Qwen3-0.6B on the same AIME24, AIME25, LiveCodeBench v6, and MATH-500 protocol and reports the control below Qwen on all four. |

All four profiles generate with reasoning enabled. The control is not a second
mode of a candidate checkpoint, so candidate gains represent model changes
rather than a relabeled decoding configuration. Public scores are screening
evidence only; the frozen EduMind evaluation determines grounded-RAG quality.

### Automated faithfulness diagnostic

[`vectara/hallucination_evaluation_model`](https://huggingface.co/vectara/hallucination_evaluation_model/blob/d3924deeff88f76f9203ae18d11432c400c07f41/README.md) is included as an automated diagnostic. Its model card reports **74.28% balanced accuracy** and **60.00% F1** on RAGTruth-QA. It does not replace blinded human faithfulness evaluation.
