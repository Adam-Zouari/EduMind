# EduMind benchmark manual

This is the consolidated specification for running, interpreting, and auditing every EduMind experiment. `model_selection.md` contains the public evidence behind candidate inclusion; the experiment-level `doc.md` files are shorter operator references. This manual explains how the pieces fit together and what a result can legitimately prove.

## 1. Purpose and decision boundary

EduMind starts with provisional application defaults. Experiments compare alternatives, but no runner edits `config/base.yaml` or promotes a winner. The decision flow is:

```text
frozen data -> candidates -> per-sample metrics -> confidence intervals
            -> correctness/resource gates -> Pareto set -> human approval
            -> separate production change
```

Public leaderboards are used to create a credible candidate list. They are not accepted as EduMind results because they use different documents, prompts, chunk boundaries, output formats, filters, hardware, and operational limits. EduMind does not rerun an entire public leaderboard; it evaluates the selected candidates on its actual path.

The project separates causal questions:

1. Can a source be extracted correctly, including tables and formulas?
2. Which deterministic normalization and routing policies help?
3. Which chunking and embedding pair retrieves verified evidence?
4. Which lexical/dense/reranking strategy improves that pair?
5. Which database preserves the selected ranking under ANN and filters?
6. Which LLM answers faithfully from frozen evidence?
7. Which complete system works after all components are connected?
8. How much quality is lost when verified text is replaced by extracted text?

Mixing these questions too early makes failures impossible to attribute. For example, a missing table can be an extraction or serialization failure, not an embedding failure.

## 2. Repository layout and execution

Production strategies live under `src/edumind`. Direct experiment code lives under `experiments/benchmarks`. Runtime and benchmarks share production extraction, chunking, embedding, and generation contracts; experiment-only BM25, RRF, rerankers, exact NumPy search, statistics, and database candidate adapters stay under experiments.

Every direct experiment directory has:

- `run.py`: ordinary Python entry point;
- `candidates.yaml`: smoke and standard candidate names;
- `doc.md`: local hypothesis, procedure, metrics, and commands.

No build is required after an editable installation:

```powershell
python -m pip install -r requirements/app.lock
python -m pip install -r requirements/benchmarks.lock
python -m pip install -e . --no-deps
```

## 3. Profiles

### Smoke

Smoke uses tiny real fixtures and normally one repetition. It proves only that the path, dependency, prepared model, and artifact writing work. It cannot support a quality, speed, or winner claim.

### Standard

Standard is the candidate-selection run. It uses the frozen development/validation data, all approved candidates, two warmups, three measured repetitions where applicable, per-sample output, and 10,000 bootstrap resamples.

### Full

Full accepts an explicitly reviewed standard `summary.json`, evaluates finalists on larger or validation workloads, and provides stronger operational evidence. The final RAG full run consumes the locked test once; it is not another tuning run.

## 4. Reproducibility and artifacts

Every run records:

- manifest fingerprint and source checksums;
- candidate settings and random seed 42;
- Git commit, dirty state, and dirty-tree hash;
- dependency-lock hashes;
- model revisions or Ollama/Docker digests;
- CPU, RAM, GPU, operating system, and Python environment;
- per-sample metrics before aggregation;
- failures instead of silent skips.

MLflow uses one parent run per invocation and one child per candidate. The filesystem artifacts are:

```text
plan.json          exact benchmark plan
provenance.json    data, Git, dependency, model, and hardware identity
candidates/*.json candidate status, aggregates, intervals, operations
samples/*.parquet per-sample metrics and metadata
summary.json       gates, comparisons, Pareto candidates, authority flag
```

Only successful completed runs are evidence. A failed candidate remains in the summary and must not be described as evaluated.

## 5. Data contracts and leakage control

A manifest header contains name, version, task, split, source, license, revision, checksum, preprocessing version, and split seed. Each asset sample contains its source path and SHA-256. Standard/full extraction samples additionally require source license, source revision, and document family.

Splits are isolated by source document or recording, not page or question. A paper with several questions stays in one split. Exact and near-duplicate checks reject leakage across splits. The locked test is not used to shortlist models or adjust prompts.

RAG evidence offsets are half-open intervals `[start, end)`. Their overlap is:

```text
max(0, min(chunk_end, evidence_end) - max(chunk_start, evidence_start))
```

Retrieved intervals are merged before evidence recall, preventing negative values and double counting.

RAG selection uses combined manifests rather than QASPER alone. QASPER contributes paper-isolated text questions; verified EduMind/public-document samples contribute table, formula, and mixed-evidence questions serialized in the same Markdown/HTML representation seen by the chunker. `prepare.py rag-selection` requires at least 10 answerable questions with verified spans for each structural evidence type in every split and rejects duplicate/near-duplicate documents. Development selects components, validation selects complete systems, and the locked test is consumed once. Results are aggregated overall and by `evidence_type`; stratum metrics are diagnostics rather than dozens of extra Pareto objectives.

Complete-document samples may add:

```json
"reference_elements": [
  {"kind": "table", "rows": [["Name", "Value"], ["A", "10"]], "page_number": 1},
  {"kind": "formula", "latex": "E=mc^2", "page_number": 1}
]
```

Structural metrics are omitted for samples without those verified annotations. Missing annotations are never represented by a fabricated zero.

## 6. Statistical rules

The unit of analysis is the independent document, page, recording, video, or question—not a repeated timing call. Repetitions estimate determinism and latency; they do not increase the quality sample count.

Standard/full use 10,000 bootstrap resamples with seed 42 and 95% confidence intervals. Candidate comparisons use paired resampling over shared sample IDs. Reports use confidence intervals, correctness gates, and Pareto comparisons; they do not claim hypothesis-test p-values.

Correctness and resource gates are applied first. Passing candidates are compared as a Pareto set: one candidate dominates another only when it is no worse on all relevant objectives and better on at least one. There is no normalized weighted overall score. When quality intervals overlap, the documented tie order is p95 latency, memory, then storage.

## 7. Common extraction metrics

### Text

- Character Error Rate: character Levenshtein distance divided by reference length. Range starts at 0; lower is better.
- Word Error Rate: word-level edit distance divided by reference word count. Range starts at 0; lower is better.
- Content Precision/Recall/F1: multiset overlap of normalized tokens. Range 0–1; higher is better.
- Missing Text Rate: unmatched reference tokens divided by reference tokens. Lower is better.
- Hallucinated Text Rate: unmatched predicted tokens divided by predicted tokens. Lower is better.
- Reading Order Accuracy: fraction of pairwise token orders preserved among tokens occurring once in both strings. Range 0–1; higher is better.
- Block F1: exact normalized non-empty-line matching. It is a transparent block diagnostic, not a substitute for layout detection evaluation.

### Tables

For the transparent built-in evaluator, table candidates are greedily matched by normalized cell-token F1. Detection requires matched content similarity of at least 0.5. Reported dimensions are detection precision/recall/F1, cell-content F1, and row/column adjacency-relation F1. Official OmniDocBench TEDS/TEDS-S must be run with its official evaluator and named separately; EduMind does not relabel its internal metric as TEDS.

### Formulas

Formula candidates are matched by normalized LaTeX edit similarity. Whitespace and outer display delimiters are ignored. The benchmark reports detection precision/recall/F1, LaTeX similarity, and exact match. OmniDocBench CDM is authoritative only when its official reproducible evaluator is run and logged.

### Operations

Extraction reports cold load, p50/p95 latency, items/minute or real-time factor, process RAM, optional VRAM, temporary-disk behavior where instrumented, empty output, duplicate text, failure rate, and determinism. Unmeasured resource values are omitted rather than set to zero.

## 8. Image and complete-page extraction

### Data

The intended corpus has 120 pages split 72/24/24 by document. It covers clean scans, noisy/skewed scans, phone photos, low resolution, multi-column pages, tables, and formulas. OmniDocBench supplies the primary public complete-page annotations; EduMind-specific verified pages add phone-photo and educational conditions.

### Candidates

- Tesseract 5: mature classical text baseline.
- PP-OCRv5 English mobile: efficient conventional neural OCR.
- PP-OCRv5 English server: higher-capacity control in the same family.
- Docling: structured non-generative document parser.
- PP-StructureV3: modular layout/OCR/table/formula pipeline.
- PaddleOCR-VL-1.6: compact end-to-end document VLM.
- GLM-OCR: independent compact structured document VLM.
- MinerU 2.5 Pro 1.2B: end-to-end parser producing ordered Markdown/JSON.
- olmOCR 2 7B: larger quality/compute document parser.

Tesseract and PP-OCRv5 run raw, document, and photo preprocessing. Complete parsers run their official complete-page pipeline once; applying every conventional preprocessing profile to them would change or duplicate their own internal preprocessing.

Text controls can win a text-only objective but cannot become the complete default unless combined with a measured structure-recovery path that passes table/formula gates.

## 9. PDF extraction and routing

The 60-document PDF corpus is split 36/12/12 and stratified across digital, scanned, mixed, broken-encoding, slides, and academic layouts. Candidates are pypdf, pdfplumber, Docling, PP-StructureV3, PaddleOCR-VL-1.6, GLM-OCR, MinerU, olmOCR, and page-level native/OCR hybrid.

Native controls answer whether a text layer is already sufficient. Complete parsers answer whether layout, tables, formulas, and OCR are recovered. Hybrid answers whether native text should be retained on usable pages while only unusable pages are rasterized. Authoritative PDF manifests include page-ordered reference text. Page Coverage measures non-empty attributed pages; same-page Content F1 measures page content; Page Attribution Accuracy asks whether each predicted page's best-matching reference page has the same number. Missing and duplicate page rates expose collapsed or repeated outputs.

Routing separately compares always native, always OCR, a document-level digital/scanned/mixed policy, and a page-level hybrid policy. Router Selection Accuracy is exact agreement with the verified best route. Quality Regret is:

```text
max(0, oracle_quality - routed_quality)
```

Fallback Success Rate indicates whether an unusable primary result was recovered. A routing winner is a policy; it need not use one extractor on every page.

## 10. DOCX extraction

The 45-document corpus is split 27/9/9 and contains paragraphs, hierarchy, lists, captions, embedded images, tables, and Office Math. The candidates are python-docx, Mammoth, Docling, and MinerU.

python-docx is the direct OOXML control. Mammoth tests semantic conversion. Docling and MinerU test complete structured representations. OOXML-derived table cells and Office Math are independently verified references. Rendering DOCX to images would be a separate measured candidate, not a hidden implementation detail.

The benchmark does not claim editing fidelity, style preservation, macro execution, or tracked-change correctness.

## 11. Audio and video

The 90-clip audio corpus is split 54/18/18 and covers clean speech, noise, accents, technical vocabulary, and multiple speakers. Candidates are OpenAI Whisper small.en; faster-whisper tiny/base/small/turbo with the documented precisions; Distil-Whisper large v3.5; Parakeet TDT 0.6B v3; and Canary-Qwen 2.5B. Distil tests efficiency in a Whisper-derived architecture; Parakeet and Canary prevent a Whisper-only conclusion.

Audio primary metrics are WER, CER, timestamp MAE, segment-boundary error, missing/hallucinated speech, and timestamp-alignment coverage. Timestamp errors are emitted only for compatible annotations, but coverage remains a selection objective so missing timestamp capability is not rewarded. Real-Time Factor is processing seconds divided by audio seconds. VAD off/on is tested only for shortlisted models because it is a decoding policy rather than a separate model.

Video freezes one selected ASR and one selected image extractor, then compares fixed-interval keyframes, scene-change keyframes, and scene-change plus maximum-interval fallback. It reports transcript WER, visual-text precision/recall/F1, duplicate visual text, timeline alignment, complete-content recall, and real-time factor. Re-crossing every OCR with every ASR would repeat already answered component questions.

## 12. Normalization

Minimal normalization handles Unicode, line endings, and whitespace. Conservative normalization additionally performs deterministic safe repairs such as dehyphenation. Aggressive normalization is included deliberately to reveal the preservation cost of additional cleanup.

For reference `R`, corrupted observation `O`, and normalized result `N`, let:

```text
B = edit_distance(R, O)
A = edit_distance(R, N)
C = edit_distance(O, N)
corrected = max(0, B - A)
```

Corruption Removal Recall is `corrected/B`; precision is `corrected/C`; F1 is their harmonic mean. Content Preservation Recall prevents a cleaner-looking but destructive result from winning.

## 13. Chunking and embeddings

### Chunkers

- Recursive character: prefers headings, paragraphs, newlines, sentences, then smaller boundaries.
- Token 256/32: precise small chunks and 32-token overlap; provisional default.
- Token 384/64: middle context size.
- Token 512/64: more surrounding explanation.
- Sentence 8/2: keeps complete sentences and overlaps two.
- Semantic: splits where adjacent sentence embeddings show a topic change, with a hard token ceiling.
- Section-aware 512/64: respects headings and section prose but gives tables/formulas no special treatment.
- Structure-aware 512/64: protects Markdown/HTML tables and displayed formulas; oversized Markdown tables split at row boundaries while preserving exact offsets.

### Embeddings

- MiniLM L6 v2: inexpensive 384-dimensional production baseline.
- EmbeddingGemma 300M: modern compact 768-dimensional quality/size candidate.
- Jasper Token Compression 600M: tests internal token compression at ratio 0.3333.
- Qwen3 Embedding 0.6B: compact modern retrieval model.
- Nemotron 3 Embed 1B BF16: retrieval-focused 2,048-dimensional model.
- Qwen3 Embedding 4B: larger family quality comparison.
- Nemotron 3 Embed 8B BF16: open quality ceiling in the selected list.

All eight chunkers are crossed with all seven embeddings: 56 combinations. This full standard matrix is intentional. Chunking and embedding interact through maximum length, tokenization, boundary context, and model training; screening either side with one fixed partner could eliminate a good pair.

Exact normalized NumPy dot-product search removes database ANN error. Primary metrics are nDCG@3/@5, rank-aware Context Precision@3/@5, Context Recall@3/@5, and Context Recall after ranked packing to 2,048 tokens. Diagnostics include Precision/Recall/Hit Rate@1/3/5/10, MAP@3/5/10, MRR, nDCG@10, chunk distribution, indexing time, latency, memory, and vector bytes. At most three explicitly reviewed non-dominated pairs advance.

## 14. Retrieval and reranking

Dense retrieval ranks vector similarity and handles paraphrases. BM25 ranks lexical matches and helps with exact names, acronyms, identifiers, and unusual terms. RRF combines ranks, not incompatible raw scores:

```text
RRF(item) = sum(1 / (60 + rank_in_list))
```

RRF retrieves 20 candidates before reranking. A reranker jointly reads the query and each candidate, enabling richer token interaction than independently encoded embeddings, but adds query-time compute. The candidates are MiniLM cross-encoder, BGE reranker v2-m3, Qwen3 Reranker 0.6B, and Qwen3 Reranker 4B.

The strategies are dense, BM25, RRF, and RRF followed by each reranker. They use the same retrieval primary metrics as the pair experiment plus complete ranking latency and resources. At most three complete retrieval stacks advance.

## 15. Vector database servers

The server comparison is Chroma 1.5.9, Qdrant 1.17.0, Weaviate 1.38.2, and PostgreSQL 17 with pgvector 0.8.2. Every server receives identical precomputed float32 vectors, IDs, text, provenance, and metadata. NumPy is only the exact-neighbor oracle.

Conformance checks health, cosine behavior, dimension rejection, compound/empty filters, duplicate-ID replacement, whole-document replacement without stale chunks, deletion, persistence after restart, ANN index existence/use, and schema incompatibility. A failed conformance candidate cannot support performance claims.

Smoke uses 1,000 vectors/50 queries. Standard uses 100,000 vectors at dimensions 384 and 1,024, 500 queries, and concurrency 1/8/32. Full uses selected real embeddings plus one million clustered vectors and concurrency through 64. Common HNSW tuning tests `m {16,32}`, construction breadth `{100,200}`, and search breadth `{64,128}` where supported. Unsupported settings are recorded, never silently mapped.

ANN Recall@K is exact-neighbor set overlap at K. Filtered recall uses the exact filtered oracle. Eligibility requires Recall@10 and filtered Recall@10 at least 0.99, all correctness flags equal one, zero measured target-concurrency errors, and no flat-search fallback. p50/p95/p99, throughput, ingestion, memory, storage, and restart readiness form the Pareto objectives.

ANN recall is not RAG quality. A separate finalist stage runs the frozen retrieval stack through the two best servers plus Chroma and measures the combined text/table/formula/mixed RAG manifest over the complete client/network/fusion/reranking/context-packing path.

## 16. Generation

The frozen-context screen isolates the LLM. Profiles are Qwen3 1.7B; Qwen3.5 4B/9B direct and thinking; Gemma 4 12B; Ministral 3 8B; and GPT-OSS 20B low/medium reasoning. Each pins its Ollama digest, temperature 0, seed 42, 8,192 context tokens, 256 answer tokens, warmups, and question order.

Primary final judgments are human Faithfulness, Answer Correctness, Completeness, Citation Accuracy, and answerability. Automated diagnostics include Exact Match, Token F1, ROUGE-L, HHEM Faithfulness, citation precision/recall/F1, refusal metrics, unsupported-answer rate, malformed-output rate, and determinism.

HHEM asks whether answer claims are supported by the supplied context. It is diagnostic because mathematical equivalence, table relationships, and citation scope can fool an automated detector. Faithfulness is distinct from correctness and completeness: an answer can faithfully repeat a bad source, or be correct using unsupported outside knowledge.

Operational gates include no crash/OOM, no malformed development answer, combined process and model memory below 28 GB, and complete-answer p95 at most 30 seconds.

## 17. Final RAG and human review

Final validation crosses at most three retrieval stacks, three generation profiles, and top-k 3/5: at most 18 systems. Context packing stops at 2,048 retrieved tokens. Prompts number contexts and require citations.

Three systems are reduced for blinded review. Twenty stratified questions by three anonymous answers produce 60 judgments. Reviewers use 0–2 rubrics:

- Faithfulness: all material claims supported, minor issue, or material unsupported claim.
- Answer Correctness: correct, partly correct, or wrong.
- Completeness: all required elements, some, or none.
- Citation Accuracy: correct support, mixed/incomplete, or absent/wrong.
- Answerability: correct answer/refusal decision as 1 or incorrect as 0.

The import validates all 60 before identities are unblinded. Exactly one approved system is evaluated once on the locked test. Further tuning creates a new benchmark version rather than another attempt on the same locked answers.

For table/formula questions, deterministic structured checks and human review remain primary; generic HHEM/ROUGE cannot establish mathematical or relational correctness by themselves.

## 18. Extraction-to-RAG confirmation

After all components are frozen, run identical questions through:

```text
verified reference representation -> selected RAG
selected extracted representation -> selected RAG
```

Paired deltas report retrieval quality, HHEM/citation diagnostics, human faithfulness/correctness/completeness, and end-to-end latency. Results are stratified into text, table, formula, and mixed evidence. This measures extraction-caused degradation without using downstream answers to repeatedly retune the earlier components.

OHR-Bench is useful in this stage because it was designed for OCR-to-RAG cascading effects. It supplements rather than replaces QASPER and the verified EduMind corpus.

## 19. Preparation and run order

Large models should be prepared separately:

```powershell
python experiments/benchmarks/prepare.py huggingface-models --candidate MODEL_ID
python experiments/benchmarks/prepare.py extraction-models --candidate CANDIDATE_NAME
python experiments/benchmarks/prepare.py ollama-models --candidate MODEL_TAG
```

Completed downloads are locked immediately and Hugging Face resumes partial files. Dataset, model, installation, and server links are maintained in the repository-root `guide.md`.

Before a standard RAG run, prepare QASPER, create and human-verify the three structured manifests, seal/validate their checksums and split isolation, and run `prepare.py rag-selection` for development, validation, and locked-test. Exact commands and the structured question schema are in `guide.md`. This extra data is required because a text-only corpus cannot decide whether section-aware or structure-aware chunking handles tables and formulas better.

Recommended order:

```text
normalization -> image -> audio -> DOCX -> PDF -> routing -> video
chunking+embedding -> retrieval -> vector servers
generation -> final RAG -> blinded review -> locked test
extraction-to-RAG confirmation
```

The exact direct commands are in `guide.md` and each experiment's `doc.md`.

## 20. Invalid conclusions

Do not claim:

- that a smoke winner is better;
- that public leaderboard rank selects EduMind's winner;
- that text CER establishes table or formula correctness;
- that TEDS/CDM were measured unless the official evaluator ran;
- that ANN recall alone establishes RAG quality;
- that automated faithfulness replaces human review;
- that repeated calls increase the independent sample count;
- that an aggregate average is valid when an important evidence stratum fails;
- that results generalize to unrecorded hardware, software, languages, or document types;
- that a benchmark result changed production unless a separate explicit production change was approved.

The benchmark program is complete only when its claims stay inside these boundaries.
