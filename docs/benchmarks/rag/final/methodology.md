# Final RAG and human review methodology

[Shared methodology](../../methodology.md) · [Metric definitions](metrics.md) · [Run commands](../../running.md)

On this page:

- [Frozen system and execution profile](#frozen-system-and-execution-profile)
- [Data and execution](#data-and-execution)
- [Human review](#human-review)
- [Extraction-to-RAG confirmation](#extraction-to-rag-confirmation)

How well does the already-selected complete system produce evidence-backed
answers when every component runs together?

## Frozen system and execution profile

Final RAG has only the `locked` profile. The engineer records one complete system
assembled from the selected extraction routes, ASR/video policy where applicable,
chunking–embedding pair, retrieval–reranking stack, vector-server profile, and
generator model-mode configuration. Top-K is one frozen value, either `3` or `5`,
chosen from component development and validation evidence before locked data is
examined. The prompt, context packing, and refusal policy are also frozen.

`final-rag-locked.json` identifies this composition and its upstream reviewed
decisions. Final RAG does not cross component finalists, tune settings, or select
a winner. Non-locked integration checks and extraction confirmation occur before
the one locked benchmark invocation. It includes the frozen measured
repetitions and generation seeds; those are not additional selection rounds.

## Data and execution

The locked profile uses the held-out locked-test manifest. For every question,
the complete path runs:

```text
prepare canonical chunks and the selected stack's required indexes
→ query Dense through the approved server, BM25, or both for RRF
→ fuse rankings only for RRF and apply the selected reranker, if any
→ select the ranked top 3 or top 5 chunks
→ number the evidence blocks
→ generate the answer and citations
```

For `top_k=3`, retrieval quality is reported at 3. For `top_k=5`, it is reported
at 3 and 5.

Final RAG reports the same linear-gain nDCG, Evidence-unit Recall, and
Evidence-token Precision at the available cutoff, plus alpha-nDCG
with the frozen greedy normalization as a diagnostic. It also reports the
primary generation-quality metrics, behavioral validity, reliability, and
repeatability diagnostics from the
[Generation benchmark](../generation/metrics.md), plus retrieval, generation,
server-call, and complete end-to-end p50/p95 latency. RAM and VRAM remain
operational measurements.

## Human review

Human review provides a separate qualitative report on the frozen system; it
does not select another system. The exporter selects 20 locked questions using
a selection rule frozen before answers are inspected and creates:

```text
20 questions × 1 anonymous system = 20 anonymous answer items
```

One reviewer scores all 20 answer items while model identity remains hidden. The
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
reliability. Ratings are imported and validated as reporting-only evidence;
they cannot reopen selection after the locked test. A future
multi-reviewer study must define overlap, agreement, and adjudication separately.

## Extraction-to-RAG confirmation

How much does real extraction reduce the quality of the selected RAG system?

### Execution

Two versions of the same documents and questions are compared:

```text
verified reference text → frozen selected RAG
selected extracted text → the same frozen selected RAG
```

Question IDs, document IDs, questions, model profiles, prompt, and retrieval
strategy remain identical. The selected parser, ASR, and vector server are part
of the extracted-text path. Each text version keeps its own evidence offsets
because extraction can change length and layout. Gold evidence-unit IDs,
required claims, verification material, and quality denominators remain the
same in both branches. A unit lost or corrupted by extraction remains a missed
required unit; it is not removed from the extracted branch's reference.
Record unrecoverable units explicitly rather than inventing empty or unrelated
source spans. Review and freeze the variant-specific mappings before scoring
outputs.

This comparison uses a separate frozen confirmation manifest derived without
locked-test questions. It may describe deployment risk, but it cannot reopen
component selection after the system has been frozen.

### Metrics

The experiment reports the paired extracted-minus-reference difference for:

- **Retrieval:** nDCG, Evidence-unit Recall, and Evidence-token Precision at the
  system's actual top-K, plus alpha-nDCG as a diagnostic on the same answerable
  questions.
- **Generation quality:** Faithfulness, Factual Correctness F1, Answer Relevancy,
  and answerable-only Citation Precision/Recall/F1.
- **Validity and diagnostics:** Response Validity Rate, Refusal Validity Rate,
  Malformed Output Rate, reliability rates, repeatability metrics, and
  per-question error inspection.
- **Operational:** server-call, retrieval, generation, and complete p50/p95
  latency.

This experiment does not select the extractor again. It quantifies the downstream
cost of extraction after component selection.
