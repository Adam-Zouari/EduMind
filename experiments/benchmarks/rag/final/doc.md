# Complete RAG, human review, and locked test

## Question and candidates

Which complete local system should be promoted? Provide approved retrieval and generation summaries. The runner accepts at most three of each, crosses them with top-k 3 and 5, and evaluates at most 18 systems. Without summaries, the small `candidates.yaml` baseline is only a wiring check.

## Procedure

Each measured repetition performs query embedding, exact dense/BM25/RRF/reranking as selected, top-k context construction, Ollama prompting, and generation. End-to-end latency is retrieval plus generation; retrieval and generation are also reported separately. Retrieval metrics are computed from the exact chunks used for generation. Standard uses QASPER validation. It never edits production configuration.

```powershell
python experiments/benchmarks/rag/final/run.py --profile standard `
  --retrieval-summary RETRIEVAL_SUMMARY_JSON `
  --generation-summary GENERATION_SUMMARY_JSON
```

Select three successful systems, then export 20 stratified questions x three anonymously shuffled answers. The CSV includes question, accepted reference answer, evidence, answer, blank rubric fields, and opaque item ID; keep the adjacent identity JSON away from reviewers.

```powershell
python experiments/benchmarks/review.py export FINAL_SUMMARY_JSON human-review.csv
python experiments/benchmarks/review.py import human-review.csv
```

Reviewers score Faithfulness, Answer Correctness, Completeness, and Citation Accuracy from 0 to 2, plus answerability correctness from 0 to 1. Import requires exactly 60 unique, valid judgments before unblinding.

- Faithfulness: 2 = every material claim follows from evidence; 1 = a minor unsupported/overstated claim; 0 = a material contradiction or fabrication.
- Answer Correctness: 2 = agrees with an accepted answer; 1 = partly correct but materially imperfect; 0 = wrong.
- Completeness: 2 = covers every required answer element; 1 = covers some; 0 = covers none.
- Citation Accuracy: 2 = all material claims cite the right evidence; 1 = citations are mixed/incomplete; 0 = absent or materially wrong.
- Answerability: 1 = answers when answerable or refuses when unanswerable; otherwise 0.

## Metrics and selection

Primary human metrics are the four rubric means and answerability decision. Automated generation/citation metrics and retrieval nDCG@3/@5, Context Precision/Recall@3/@5, and Context Recall@2,048 tokens remain diagnostics. Operational output includes end-to-end/retrieval/generation p50/p95, TTFT, throughput, tokens, RAM, and optional VRAM. Hard malformed, memory, crash, and 30-second gates precede interval-aware Pareto evidence. Final manual priority is human Faithfulness/correctness, citation behavior, automated quality, p95, then memory.

After approving exactly one reviewed candidate, create a one-candidate final summary and run the locked split once:

```powershell
python experiments/benchmarks/rag/final/run.py --profile full `
  --shortlist ONE_APPROVED_FINAL_SUMMARY_JSON `
  --review-results human-review.results.json `
  --confirm-locked-test
```

The command validates the 60-judgment review and writes `artifacts/benchmarks/rag/final/locked-test-v1.json`; a second v1 run is refused. New tuning requires a new benchmark version. One review round does not establish inter-rater reliability unless multiple reviewers are deliberately added.

## Extraction-to-RAG confirmation

After the final system is frozen, create two RAG manifests with identical document/question IDs and question text: one contains verified text and evidence offsets, the other contains selected-extractor text with independently verified offsets. Then run:

```powershell
python experiments/benchmarks/rag/final/confirm_extraction.py `
  --reference-manifest VERIFIED.json `
  --extracted-manifest EXTRACTED.json `
  --candidate "chunker@@embedding@@retrieval@@generator@@top_k=5"
```

The two child runs use the same frozen system and report paired `verified-reference - selected-extraction` intervals for retrieval, citation/NLI, and end-to-end metrics. To collect the required human deltas, export both systems with `review.py export CONFIRMATION_SUMMARY human-confirmation.csv --finalists 2 --questions 20`, producing 40 blinded judgments. This stage quantifies extraction degradation; it must not retune the selected components.

Artifacts include plan/provenance, per-system candidate JSON/Parquet, paired intervals, human-review CSV plus separate identity/results JSON, and the locked-test marker. Example: the best Token F1 system is not selected when blinded Faithfulness is materially worse.
