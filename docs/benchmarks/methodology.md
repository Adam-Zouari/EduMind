# How the EduMind experiments work

[Benchmark overview](overview.md) · [Metric definitions](metrics.md) ·
[Run commands](running.md) · [Candidate rationale](model-selection.md)

The experiments form a sequence. First compare extraction components. Separately,
compare chunker/embedding pairs, then retrieval methods and vector servers.
Generation is tested on fixed evidence so retrieval cannot influence it. Finally,
combine the approved retrieval and generation profiles and review their answers.

```text
1. Document extraction: configuration on development, architectures on validation
2. Audio extraction: all candidates on development, finalists on validation
3. Text normalization: all profiles on development, finalists on validation
1 + 2 + 3 --> 4. Video keyframes with parser, ASR, and normalization frozen

5. Chunking x embedding --> 6. Retrieval and reranking
6 --> 7. Real vector-server retrieval

8. Generation on fixed evidence (independent of retrieval)

selected server + selected retrieval + selected generator
--> 9. Final RAG on validation + blinded human review
--> 10. Extraction-to-RAG confirmation on separate non-locked data
--> 11. Exactly one locked-test run
```

Document extraction, audio, normalization, chunking/embedding, the synthetic
vector-server workload, and generation can start independently. Video waits for
a document parser, ASR, and normalization profile. Retrieval waits for
chunking/embedding. Real server retrieval waits for a retrieval stack. Final RAG
waits for one selected server, retrieval stack, and generator.

`smoke` checks that a small real path works and cannot support a selection.
`standard` compares every candidate on development data. `full` compares only
engineer-selected finalists on validation data. Standard/full use seed 42, retain
per-sample results, and report 95% bootstrap confidence intervals. The locked
test is reserved for the one final system.

Every comparison gives its candidates the same samples. MLflow stores the exact
settings, revisions, data checksum, hardware, aggregate metrics, confidence
intervals, and per-sample results. The engineer chooses what continues; the
runner never chooses a winner or changes the application configuration.

Metrics are labeled **primary**, **secondary**, **diagnostic**, or
**operational**. Primary metrics answer the experiment's central question.
Secondary metrics provide important supporting evidence. Diagnostics explain
failure modes but do not select a candidate alone. Operational metrics measure
latency, throughput, and resources.

## 1. Document extraction

Which complete parser best converts educational images, PDFs, and DOCX files into
accurate text and useful structure, including pages, reading order, tables, and
formulas?

OCR is not tested as an isolated product. It is tested inside the document
pipeline because OCR, native PDF text, layout detection, tables, formulas, and
reading order affect one another.

### Phase A: 24 Docling Standard configurations

The first phase compares:

```text
OCR engine:          RapidOCR, Tesseract, EasyOCR                 (3)
OCR mode:            PDF-aware regions, full page                 (2)
TableFormer mode:    fast, accurate                               (2)
Formula enrichment: off, on                                      (2)
                                                                  ───
Total:               3 × 2 × 2 × 2 = 24 configurations
```

Why these settings vary:

| Setting | Why it is tested |
|---|---|
| OCR engine | Recognition quality, bounding boxes, speed, and device use differ. RapidOCR is the production control, Tesseract is a classical CPU comparison, and EasyOCR is a separate neural implementation. |
| PDF-aware versus full-page OCR | PDF-aware mode can preserve correct native text and OCR only necessary regions. Full-page OCR can recover scans or broken encodings but may duplicate correct native text. |
| TableFormer fast versus accurate | Tests the speed/quality trade-off in row and column reconstruction. |
| Formula enrichment off versus on | Tests whether specialized formula recovery justifies its additional load, latency, and memory. |

The following stay fixed so the experiment remains understandable: Docling
2.117.0, English, OCR scale 3.0, table-cell matching enabled, code enrichment
disabled, canonical structured output, and native DOCX ingestion.

### Phase B: parser architectures

After reviewing Phase A, the selected Docling configurations are compared with:

| Parser | Why it is included |
|---|---|
| Selected Docling Standard configuration | Conventional OCR, layout, table, and formula pipeline control. |
| Granite Docling 258M | Compact visual parser integrated with Docling; tests a different architecture. |
| PaddleOCR-VL-1.6 | Independent visual document parser, so the comparison is not limited to Docling implementations. |

The visual-parser comparison uses images and PDFs. DOCX remains native Docling
input instead of being rasterized merely to accommodate visual models.

### Data

Smoke uses two committed images, two PDFs, and two DOCX files. The authoritative
corpus target is:

| Modality | Total | Development | Validation | Locked |
|---|---:|---:|---:|---:|
| Images/pages | 120 | 72 | 24 | 24 |
| PDFs | 60 | 36 | 12 | 12 |
| DOCX | 45 | 27 | 9 | 9 |

The reviewed source pool is OmniDocBench, olmOCR-Bench, PureDocBench, and
EduMind-specific verified samples. The frozen manifests—not the source-pool
names—define the actual cases and record every selected ID, source revision,
license, checksum, and document family. An authoritative run cannot be claimed
until those manifests and references are complete.

The corpus covers:

- clean and degraded images;
- digital, scanned, mixed, and broken-encoding PDFs;
- phone photos, low resolution, skew, and multiple columns;
- headings, paragraphs, lists, captions, and reading order;
- tables and formulas with explicit structural references;
- native DOCX documents.

Every sample includes verified text. Page, table, formula, and layout metrics are
calculated only when the required annotations exist. Every report shows the
number of annotated samples behind each conditional metric; a structure metric
with too few annotations remains descriptive rather than selection evidence.

### Execution

The two phases use different splits:

```text
development:
24 Docling Standard configurations
→ engineer selects configuration finalists

validation:
Docling finalists + Granite Docling + PaddleOCR-VL
→ engineer selects the document parser
```

Within each phase, every candidate processes the same deterministically shuffled
manifest three times after one cold item and warmups. Output is converted to the
common structured-document format before scoring. Parser comparison is performed
before the separately selected cleanup profile is applied, except for fixed
canonical Unicode and line-ending normalization required by every candidate.

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | Content F1, Reading Order Accuracy, Page Content F1; Table Structure F1 and Formula LaTeX Similarity on annotated subsets | Measures complete content, ordering, page assignment, and the two important structured-content types. |
| Secondary | CER, WER, Content Precision/Recall, Page Coverage, Page Attribution Accuracy, Block F1, Table Detection F1, Table Content F1, Formula Detection F1, Formula Exact Match | Explains whether a primary result came from transcription, coverage, detection, or exact reconstruction. |
| Diagnostic | Missing Text Rate, Hallucinated Text Rate, Empty Output Rate, Duplicate Text Rate, Duplicate Page Rate, block/table/formula precision and recall | Locates omission, unsupported-output, duplication, and precision-versus-recall failures. |
| Operational | Cold first-extraction time, p50/p95 item latency, items/minute, peak RAM, peak VRAM, determinism | Shows execution cost and repeatability. |

An engineer reviews quality, latency, and resources, then records the Docling
configuration finalists and parser-architecture finalists. The approved document
parser is combined with the approved normalization profile before video and
downstream extraction are evaluated.

## 2. Audio extraction

Which English speech-to-text profile produces the most accurate educational
transcript with timestamps suitable for navigation and citations?

### Models

| Model/profile | Why it is included |
|---|---|
| Whisper `small.en` | Established English control with timestamp output. |
| Canary 180M | Compact timestamp-capable challenger. |
| Parakeet TDT 0.6B v2 | Mid-size profile with word, segment, and character timestamps. |
| MOSS Transcribe-Diarize | Tests transcription with diarization-oriented segment output. |
| Qwen3 ASR 1.7B + ForcedAligner 0.6B | Strong transcription candidate whose separate aligner provides timestamps. |

For Qwen, transcription runs first. Its ASR model is unloaded before the forced
aligner runs. The benchmark still measures transcription plus alignment as one
complete candidate.

### Data

Smoke uses two committed audio clips. The authoritative corpus contains 90
English clips split 54 development, 18 validation, and 18 locked. It includes:

- verified transcripts;
- complete audio duration;
- segment or word timestamps;
- clean and noisy speech;
- accents and multiple speakers;
- technical and educational vocabulary.

The exact recordings are fixed by the licensed asset plan and manifests, which
record source, revision, selected clip interval, license, checksum, duration, and
speaker/document family. The public Open ASR material is screening context, not a
substitute for this frozen educational corpus.

### Execution

```text
development: all ASR candidates on the same 54 clips
→ engineer selects finalists

validation: ASR finalists on the same 18 clips
→ engineer selects the ASR profile
```

Every candidate uses the same requested device and fixed audio preprocessing.
Each run loads the exact model, transcribes, performs alignment when required,
converts output to timestamped segments, and repeats three times. Candidate text
is scored before the separately selected cleanup profile is applied.

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | WER, Timestamp MAE, Timestamp Alignment Coverage | Measures transcription accuracy and whether useful timestamps are both accurate and sufficiently complete. |
| Secondary | CER, Segment Boundary MAE, Missing Speech Rate, Hallucinated Speech Rate | Reveals technical-term errors, boundary errors, dropped speech, and unsupported speech. |
| Diagnostic | Determinism | Checks whether repeated transcripts and timestamps remain stable. |
| Operational | Real-Time Factor, cold load, p50/p95 clip latency, peak RAM, peak VRAM | Measures complete transcription/alignment cost. |

The engineer approves the ASR profiles that provide the best useful combination
of transcription, timestamp quality, and execution cost. The selected ASR is
frozen for video extraction.

## 3. Text normalization

How much deterministic cleanup should be applied after extraction without
deleting or merging legitimate educational content?

### Profiles

| Profile | What it does | Why it is included |
|---|---|---|
| Minimal | Unicode normalization, line-ending repair, null/soft-hyphen removal, final trim | Safe baseline. |
| Conservative | Minimal plus dehyphenation and restrained whitespace cleanup | Expected production trade-off. |
| Aggressive | Conservative plus stronger page-label and newline cleanup | Tests whether more cleanup helps or becomes destructive. |

### Data and execution

Smoke uses the committed observed/reference text pairs. The authoritative set
contains at least 200 verified cases split by document family into 120
development, 40 validation, and 40 locked cases.

All three profiles receive the same development cases. After the engineer selects
finalists, those profiles receive the same validation cases. Every output is
compared with the verified clean reference and repeated three times.

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | Content Preservation Recall, Corruption Removal F1 | Cleanup must preserve valid content while repairing real corruption. |
| Secondary | Corruption Removal Precision/Recall, WER | Separates unnecessary edits from missed repairs and shows final word error. |
| Diagnostic | CER, Content F1, Missing/Hallucinated Text Rate, Duplicate Text Rate, determinism | Explains character errors, deletion, unsupported additions, repeated text, and instability. |
| Operational | p50/p95 latency | Confirms that deterministic cleanup remains inexpensive. |

The selected profile is applied after the selected document parser or ASR and
before video combination, indexing, and extraction-to-RAG confirmation.

## 4. Video extraction

With the document parser and ASR fixed, which keyframe policy recovers useful
on-screen text without processing too many duplicate frames?

### Candidates

| Policy | Behavior | Why it is included |
|---|---|---|
| Fixed interval | One frame every ten seconds | Predictable-cost baseline. |
| Scene change | A frame when FFmpeg scene score exceeds 0.35 | Avoids repeatedly parsing unchanged slides. |
| Hybrid | Scene changes plus the first frame and a maximum ten-second gap | Recovers gradual changes that do not trigger a strong scene cut. |

### Data

Smoke uses two committed videos. The authoritative set contains 30 educational
videos split 18 development, 6 validation, and 6 locked. Every video has a
verified transcript, duration, visible text, and visual timestamps. The set
includes slides, screen recordings, presenter video, gradual text changes, and
repeated scenes. Exact sources, revisions, licenses, clip intervals, and checksums
are frozen in the manifests.

### Execution

```text
video
├─ FFmpeg extracts mono 16 kHz audio → frozen selected ASR
└─ FFmpeg extracts candidate keyframes → frozen selected document parser
        ↓
apply the selected normalization profile
→ combine timestamped audio and visual segments
→ compare with transcript and visible-text references
```

Only the keyframe policy changes. Reopening parser or ASR selection here would
make it unclear whether a difference came from frame selection, OCR, or speech
recognition. All three policies run on development; engineer-selected finalists
then run on validation.

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | Visual Text F1, Complete Content Recall, Audio/Visual Alignment MAE | Measures visible-text quality, total recovered educational content, and timestamp usefulness. |
| Secondary | Visual Text Precision/Recall, Duplicate Visual Text Rate, Timestamp Coverage | Separates missing text from extra or repeated frames and shows whether alignment coverage is adequate. |
| Diagnostic | Frozen-ASR Transcript WER, recorded once for the shared ASR output | Confirms the audio input to every policy; it is not used to compare keyframe policies because it is constant. |
| Operational | Real-Time Factor, cold load, p50/p95 video latency, peak RAM, peak VRAM | Measures complete extraction cost. |

The selected keyframe policy joins the selected parser and ASR as the provisional
video-extraction profile.

## 5. Chunking and embedding

Which complete chunker/embedding pair retrieves verified educational evidence
best?

Chunking and embedding are evaluated together. A chunker cannot be scored until
its chunks are represented and ranked, and an embedding cannot be judged without
the passages it embeds.

### Chunking strategies

| Strategy | What it tests |
|---|---|
| Recursive character 1000/200 | Cheap textual separators without tokenizer dependence. |
| Token 256/32 | Short, focused passages with modest overlap. |
| Token 384/64 | Middle fixed-token trade-off. |
| Token 512/64 | More local context per chunk. |
| Sentence 8/2 | Linguistic boundaries instead of fixed token windows. |
| Semantic | Topic-change boundaries based on adjacent-sentence embedding similarity. |
| Section-aware 512/64 | Preserves authored Markdown sections before splitting oversized sections. |
| Structure-aware 512/64 | Preserves headings, tables, formulas, and table-row boundaries where possible. |

### Embedding models

| Model | Role in the comparison |
|---|---|
| MiniLM L6 v2 | Lightweight production control. |
| Snowflake Arctic Embed M v2 | Strong small-model retrieval candidate with a long input limit. |
| F2LLM v2 0.6B | Alternative 0.6B retrieval architecture. |
| Octen Embedding 0.6B | Strong 0.6B retrieval challenger. |
| Qwen3 Embedding 0.6B | Strong directly comparable 0.6B candidate using documented last-token pooling. |
| Nemotron 3 Embed 1B | Larger nearby-size retrieval candidate. |
| Octen Embedding 4B | Upper-size challenger. |
| Qwen3 Embedding 4B | Upper-size quality candidate using the same documented Qwen retrieval recipe. |

Full public evidence and exact revisions are in
[model-selection.md](model-selection.md).

### Data

The RAG corpus uses pinned QASPER papers plus EduMind's verified structured
evidence set:

| Split | Papers | Used for |
|---|---:|---|
| Development | 100 | Standard component comparison |
| Validation | 40 | Full finalist comparison |
| Locked test | 40 | One final complete system only |

Each question stores answerability, accepted answers, evidence type, and exact
half-open evidence offsets. The structured supplement contains table, formula,
and mixed-evidence questions because QASPER is primarily text.

### Execution

Standard evaluates all:

```text
8 chunkers × 8 embeddings = 64 complete pairs
```

For every pair:

```text
split documents into chunks with exact source offsets
→ embed every chunk
→ embed each question
→ rank with exact NumPy cosine search
→ retain the top 20
→ compare retrieved spans with verified evidence spans
```

Exact NumPy search removes vector-database approximation from this experiment.
For semantic chunking, the tested embedding also creates the boundaries; that
result intentionally represents the complete semantic-chunker/embedding pair.

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | nDCG@3/@5, rank-aware Context Precision@3/@5, Context Recall@3/@5, Context Recall under 2,048 tokens | Measures early graded ranking, context cleanliness, evidence-span coverage, and coverage at equal context cost. |
| Secondary | nDCG@10, Hit Rate@5/@10, MRR | Shows deeper ranking and whether at least one useful passage appears. |
| Diagnostic | Precision/Recall@1/@3/@5/@10, MAP@3/@5/@10, Hit Rate@1/@3, Context Precision/Recall@1/@10 | Helps explain behavior, but binary chunk relevance depends on candidate-created boundaries. |
| Evidence slices | Primary metrics repeated for text, table, formula, and mixed questions | Reveals candidates that work well only on the majority evidence type. |
| Operational | Chunk count, mean/p95 chunk tokens, indexing time, p50/p95 query latency, vector storage, RAM, VRAM, determinism | Explains the storage and execution cost behind retrieval quality. |

Chunk-level Recall and MAP are comparable within one fixed chunker. Across
different chunkers, the number of relevant chunks changes with the boundaries,
so decisions use span-based Context Recall and graded nDCG instead.

The engineer approves up to three complete chunker/embedding pairs. No separate
embedding winner or chunker winner is required.

## 6. Retrieval and reranking

For the approved chunker/embedding pairs, do exact-term search, rank fusion, or a
learned reranker improve the ordering enough to justify their cost?

### Strategies

| Strategy | What runs | Why it is included |
|---|---|---|
| Dense | Cosine ranking from the selected embedding | Semantic baseline. |
| BM25 | Lexical ranking from word frequency and rarity | Finds names, exact terminology, identifiers, and codes. |
| RRF | Dense top 20 + BM25 top 20 combined by reciprocal ranks | Combines rankings without mixing incompatible raw scores. |
| RRF + reranker | RRF top 20 reordered by a query/passage model | Tests deeper relevance scoring before final context packing. |

RRF is tested with these rerankers:

- MiniLM cross-encoder control;
- Ettin 150M;
- Ettin 400M;
- Ettin 1B;
- Qwen3 Reranker 4B.

Together with Dense, BM25, and RRF, this produces eight retrieval methods for
each selected chunker/embedding pair.

### Data and execution

The experiment uses the same frozen QASPER-plus-structured corpus and evidence
offsets as the previous stage.

```text
use an approved chunker/embedding pair
→ retrieve dense top 20, BM25 top 20, or both
→ apply RRF when required
→ rerank the fused top 20 when required
→ pack results under 2,048 tokens
→ compare the final ranking with verified evidence
```

Dense-only does not secretly build BM25. BM25-only does not embed queries unless
the selected semantic chunker requires its embedding to create boundaries.

### Metrics and why they are used

The metric roles stay the same as in chunking/embedding so changing only the
retrieval strategy remains interpretable:

- **Primary:** nDCG@3/@5, Context Precision@3/@5, Context Recall@3/@5, and
  Context Recall under 2,048 tokens.
- **Secondary:** nDCG@10, Hit Rate@5/@10, and MRR.
- **Diagnostic:** conventional Precision/Recall/Hit Rate at the remaining
  cutoffs, MAP@3/@5/@10, Context Precision/Recall@1/@10, determinism, and the
  primary metrics split by evidence type.
- **Operational:** total retrieval p50/p95 latency, indexing time, retrieved
  tokens, chunk distribution, storage, RAM, and VRAM.

Reranker time is included in total retrieval latency. Binary chunk Recall and MAP
are not used to compare different chunkers because their denominators change with
chunk boundaries.

The engineer approves up to three complete retrieval stacks. A stack contains
the chunker, embedding, retrieval method, and reranker when applicable.

## 7. Vector database servers

Which networked vector server preserves nearest-neighbour and filter correctness
while providing the most useful latency, concurrency, ingestion, memory, and
storage trade-off?

### Servers

| Server | Why it is included |
|---|---|
| Chroma server | Current provisional production baseline. |
| Qdrant server | Purpose-built vector server with payload filtering. |
| Weaviate | Independent vector-server architecture with structured filters. |
| PostgreSQL + pgvector | Transactional relational alternative with JSON metadata and HNSW. |

All servers receive identical precomputed vectors and metadata. They never create
embeddings internally.

### Data and configurations

| Profile | Workload |
|---|---|
| Smoke | 1,000 vectors at dimension 384; 50 queries; concurrency 1 |
| Standard | 100,000 vectors at dimensions 384 and 1,024; 500 queries; concurrency 1/8/32 |
| Full | Selected real embeddings plus 1,000,000 clustered vectors; up to 1,000 queries; concurrency 1/8/32/64 |

Synthetic vectors contain clusters and 5% near-duplicates. Metadata creates
filters matching approximately 50%, 10%, 1%, and in full 0.1% of records.

Standard/full test supported HNSW combinations of:

```text
m:                     16 or 32
construction breadth:  100 or 200
search breadth:         64 or 128
```

This makes sure one server is not compared with an unnecessarily weak default.
Unsupported settings are recorded rather than silently replaced.
Every supported configuration remains visible in MLflow; the runner does not
automatically choose the database winner.

### Execution

The experiment has three steps.

#### A. Conformance

The real server is checked for health, cosine behavior, wrong-dimension
rejection, compound filters, empty filters, duplicate-ID replacement, complete
document replacement, deletion, real ANN-index use, persistence after restart,
and index availability after restart. These are validity checks. A server that
fails one is reported as non-conformant rather than assigned a misleading
performance rank.

#### B. Dense ANN performance

```text
NumPy computes exact top neighbours
→ server returns approximate neighbours
→ compare returned IDs with the exact IDs
→ repeat unfiltered and filtered queries
→ repeat at each configured concurrency
```

#### C. Real retrieval

After the engineer chooses database finalists and one retrieval stack, every
finalist stores the same real chunks and vectors. The complete retrieval strategy
is rerun so database ANN behavior is connected to actual RAG quality. The
engineer then approves one server profile for Final RAG.

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Validity checks | Health, cosine behavior, dimension rejection, compound/empty filters, replacement, deletion, persistence, restart, and ANN-index verification | Determines whether results are trustworthy; these are not quality scores. |
| Primary | ANN Recall@3/@5/@10, Filtered ANN Recall@3/@5/@10, Filter Correctness, Empty-Filter Correctness | Measures preservation of exact neighbours and metadata behavior at application-relevant depths. |
| Secondary | ANN and Filtered ANN Recall@1, complete RAG nDCG@3/@5 and Context Recall/Precision@3/@5 | Shows rank-one behavior and whether ANN results preserve real evidence retrieval. |
| Diagnostic | Unfiltered/filtered p50 latency, first query after restart | Explains typical and cold-query behavior. |
| Operational | Unfiltered/filtered p95/p99, throughput and error rate at each concurrency, build time and vectors/second, incremental upsert/delete throughput, restart readiness, peak server RAM, persistent storage | Measures tail latency, load handling, ingestion, restart, memory, and disk cost. |

The database report remains separate evidence. It shows which server should be
used by the final benchmark, but the benchmark never changes the current Chroma
production default automatically.

## 8. Generation

Which local Hugging Face generator produces the best grounded, cited answer when
every model receives exactly the same verified evidence?

### Models

| Model/profile | Why it is included |
|---|---|
| Qwen3 1.7B, thinking disabled | Small production control. |
| MiniCPM5 1B, reasoning enabled | Compact reasoning candidate. |
| G9v3 3B, reasoning enabled | Middle-size general-capability candidate. |
| Qwen3.5 4B, reasoning enabled | Upper compact quality candidate. |

No trustworthy public benchmark compares all four under EduMind's grounded QA,
citation, refusal, faithfulness, and local-latency protocol. Public evidence made
the shortlist; this experiment makes them directly comparable.

### Data

Standard uses 24 development questions balanced across answerability, answer
type, and evidence type as an initial screen. Full evaluates only
engineer-selected generator finalists on the complete frozen validation question
set; it is not limited to 24 questions.

- Answerable questions receive their verified numbered evidence blocks.
- Unanswerable questions receive text from their document that does not answer
  the question.
- Retrieval is not run in this stage.

Using frozen evidence prevents a good generator from being penalized by a poor
retriever.

### Execution

Every generator uses its exact pinned local snapshot, official chat template,
the same CPU or CUDA device, native checkpoint dtype, temperature 0, seed 42,
8,192 context tokens, and at most 256 generated tokens. No model receives hidden
quantization or CPU/GPU offload.

```text
unload previous generator
→ cold-load candidate
→ run two warmups
→ give every candidate the same question and numbered evidence
→ generate three times
→ separate hidden reasoning from visible answer
→ validate citations and compare with accepted answers
```

### Metrics and why they are used

| Role | Metrics | Why they are needed |
|---|---|---|
| Primary | Citation Precision/Recall/F1 on answerable questions, Answerability Balanced Accuracy, Unsupported Answer Rate, Malformed Output Rate | Measures evidence use, answer/refusal decisions, unsupported answers, and protocol failures without penalizing correct refusals for having no citations. |
| Secondary | Token F1 | Provides partial answer-correctness evidence before human review. |
| Diagnostic | Exact Match, ROUGE-L, Refusal Precision/Recall/F1, HHEM on substantive non-refusal answers, determinism, prompt/answer/reasoning token counts | Explains lexical similarity, refusal errors, automated support estimates, stability, and verbosity. HHEM never replaces human Faithfulness. |
| Operational | Cold load, Time to First Token, generation time, total p50/p95 latency, tokens/second, peak RAM, peak VRAM | Separates startup, responsiveness, decoding speed, total latency, and memory. |

The engineer approves up to three generator profiles after inspecting automatic
quality, citation/refusal behavior, latency, and resources. Human review happens
only after retrieval and generation are combined.

## 9. Final RAG and human review

Which complete retrieval-and-generation system gives the best evidence-backed
answers when every component runs together?

### Systems tested

Standard crosses only approved finalists:

```text
1 approved vector-server profile
× up to 3 retrieval stacks
× up to 3 generators
× top_k {3, 5}
= at most 18 complete systems
```

### Data and execution

Final RAG uses the validation manifest. For every question, the complete path
runs:

```text
chunk document
→ embed chunks and question
→ store/query dense vectors through the approved server
→ apply BM25/RRF/reranking selected for the stack
→ pack top 3 or top 5 under 2,048 context tokens
→ number the evidence blocks
→ generate the answer and citations
```

For `top_k=3`, retrieval quality is reported at 3. For `top_k=5`, it is reported
at 3 and 5. Final RAG does not report @10 from a list that contains only three or
five passages; the dedicated retrieval experiment owns @10 conclusions.

Final RAG reports nDCG, Context Precision, Context Recall, and token-budget recall
at the available cutoff; the primary and secondary generation metrics from
Experiment 8; and retrieval, generation, server-call, and complete end-to-end
p50/p95 latency. RAM and VRAM remain operational measurements.

### Human review

The engineer chooses exactly three successful complete systems. The exporter
selects 20 common questions and creates:

```text
20 questions × 3 anonymous systems = 60 anonymous answer items
```

One reviewer scores all 60 answer items while system identity remains hidden. The
reviewer sees the question, answer, accepted answer, and evidence. Each answer
receives:

| Human metric | Scale | What it measures |
|---|---:|---|
| Faithfulness | 0–2 | Whether every material claim is supported by the supplied evidence. |
| Answer Correctness | 0–2 | Whether the answer is correct for the question. |
| Completeness | 0–2 | Whether all essential parts are covered. |
| Citation Accuracy | 0–2 | Whether citations are attached to the right claims and evidence blocks. |
| Answerability Correctness | 0–1 | Whether the system correctly answered or refused. |

This is single-reviewer evidence, so the report does not claim inter-reviewer
reliability. After ratings are imported and validated, system identities are
revealed and the engineer selects exactly one complete system. A future
multi-reviewer study must define overlap, agreement, and adjudication separately.

## 10. Extraction-to-RAG confirmation

How much does real extraction reduce the quality of the selected RAG system?

### Execution

Two versions of the same documents and questions are compared:

```text
verified reference text → frozen selected RAG
selected extracted text → the same frozen selected RAG
```

Question IDs, document IDs, questions, model profiles, prompt, and retrieval
strategy remain identical. The selected parser, ASR, normalization profile, and
vector server are part of the extracted-text path. Each text version keeps its
own evidence offsets because extraction can change length and layout.

This comparison uses a separate frozen confirmation manifest derived without
locked-test questions. It may describe deployment risk, but it cannot reopen
component selection after the system has been frozen.

### Metrics

The experiment reports the paired extracted-minus-reference difference for:

- **Retrieval:** nDCG, Context Precision, and Context Recall at the system's
  actual top-K, plus Context Recall under 2,048 tokens.
- **Generation:** Token F1, answerable-only Citation Precision/Recall/F1,
  Answerability Balanced Accuracy, Unsupported Answer Rate, and Malformed Output
  Rate.
- **Diagnostics:** refusal metrics, HHEM on substantive answers, and per-question
  error inspection.
- **Operational:** server-call, retrieval, generation, and complete p50/p95
  latency.

This experiment does not select the extractor again. It quantifies the downstream
cost of extraction after component selection.

## 11. Locked test

After Final RAG review and extraction confirmation are complete, the one frozen
system runs exactly once on the locked-test manifest. The selected parser, ASR,
normalization, chunker, embedding, retrieval method, vector server, generator,
prompt, and context settings cannot change between confirmation and this run.

The locked result is the final unbiased estimate. It is not used for more tuning.
If the system changes after the result is inspected, a new benchmark and locked
dataset version are required.
