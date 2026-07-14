# Complete RAG, human review, and locked test

## Question, candidates, and controls

Which complete local retrieval+generation system should be promoted? Validate at most `3 retrieval stacks x 3 LLM profiles x top_k {3,5}` with no more than 2,048 retrieved-context tokens. Qwen3 1.7B with the baseline promoted retrieval is the control. The final runner executes retrieval, ranked context packing, prompt construction, generation, and citations; it is not the frozen-context screen.

## Dataset, splits, and procedure

Run the 18 or fewer systems on the frozen QASPER validation papers only after component promotion. Apply the 30-second/resource/malformed gates, then select three systems. Export 20 stratified questions x three anonymously ordered answers (60 rows), hide identity, score the rubrics, validate the complete import, and unblind. Select one stack, freeze a new benchmark version/recommendation manifest, and evaluate it exactly once on the locked 40-paper split. Any later tuning requires a new benchmark version.

## Metrics and rationale

Primary human metrics are Faithfulness, Answer Correctness, Completeness, and Citation Accuracy on 0–2 scales plus correct answerability. Citation Precision/Recall/F1 and Answerability Balanced Accuracy range 0–1 and higher. Automated diagnostics are Exact Match, Token F1, ROUGE-L, pinned NLI Faithfulness, answerability/refusal scores, Unsupported Answer Rate, and Malformed Output Rate. Retrieval diagnostics remain nDCG@3/5, Context Precision/Recall@3/5, and Context Recall@2,048 tokens. p50/p95 end-to-end, TTFT, throughput, tokens, and resources are operational.

## Statistics, selection, and artifacts

Persist per-question system results before fixed-seed paired bootstrap intervals; use Holm correction for formal multiple comparisons. Hard gates precede Pareto results. Final selection order is human Faithfulness/correctness, citation behavior, automated quality, p95 latency, then memory. Artifacts include plan/provenance, anonymous CSV plus separate identity map, validated review results, candidate samples/intervals, recommendation run IDs, success marker, and metric-only MLflow hierarchy.

## Extraction-to-RAG confirmation

With the selected RAG frozen, compare verified reference text against selected extracted text for the same raw documents. Report deltas in nDCG@3/5, Context Precision/Recall@3/5, Context Recall@2,048 tokens, human Faithfulness/Correctness/Completeness, Citation F1, and end-to-end latency. This quantifies extraction degradation without retuning a component on final evidence.

## Commands, example, and limitations

```powershell
edumind benchmark --profile standard rag final
edumind benchmark review export SUMMARY_JSON human-review.csv
edumind benchmark review import human-review.csv
edumind benchmark report SUMMARY_JSON
```

Example: a system with the best automated Token F1 but weaker blinded Faithfulness is not selected. Validation, smoke, and component winners are not locked-test claims. One human review round cannot establish inter-rater reliability unless multiple reviewers are deliberately added and reported.
