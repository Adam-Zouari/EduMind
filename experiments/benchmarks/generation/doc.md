# Generation experiment

## Question, candidates, and control

Which local Ollama profile best converts one frozen evidence context into a correct, complete, faithful, cited answer within the target resources? Screen Qwen3 1.7B (installed control); Qwen3.5 4B/9B direct and thinking; Gemma3 4B/12B; Ministral 3 8B; and GPT-OSS 20B low/medium reasoning. Frozen context isolates generation from retrieval.

## Dataset, splits, and procedure

Use 24 stratified QASPER development questions with frozen retrieved evidence, answerability, accepted answers, and provenance. For each pinned Ollama digest use its documented inference settings, seed 42, a maximum 4,096-token packed prompt, 8,192 context, 256 answer tokens, two warmups, measured repetitions, explicit unload, and separate cold load. Candidate order is seeded. Raw prompts/answers are evaluation artifacts only and are not sent to MLflow or normal runtime logs.

## Metrics and rationale

Primary human scores are Faithfulness, Answer Correctness, Completeness, and Citation Accuracy on 0–2 rubrics (higher), plus correct answerability. Citation Precision is supported cited claims/all cited claims; Recall is required supported claims cited/all required supported claims; F1 is harmonic mean (0–1, higher). Answerability Balanced Accuracy is mean recall across answerable/unanswerable classes (0–1, higher).

Diagnostics are Exact Match, Token F1, ROUGE-L, pinned local NLI Faithfulness, Answerability Accuracy, Refusal Precision/Recall/F1, Unsupported Answer Rate, and Malformed Output Rate. Operational measures include answer/reasoning tokens, time to first token, prompt evaluation, generation/end-to-end p50/p95, tokens/second, answers/minute, cold load, CPU/GPU/RAM/VRAM.

## Gates, statistics, and artifacts

Deployment gates are no OOM/crash or malformed development output, peak process memory below 28 GB, p95 complete answer <=30 seconds, and no sustained paging. Per-question bootstrap intervals and Pareto selection follow the gates; up to three deployment-eligible profiles advance. Artifacts contain plan/provenance/digests, per-question results, timings/resources, intervals, Pareto candidates, success marker, and optional metric-only MLflow parent/children.

## Commands, example, and limitations

```powershell
edumind benchmark --profile smoke rag generation
edumind benchmark --profile standard rag generation
edumind benchmark --profile full rag generation
```

Example: higher ROUGE-L cannot rescue an unsupported cited answer or p95 of 45 seconds. Smoke uses deterministic fake generation and is not model evidence. Automated/NLI metrics do not replace blinded humans, and frozen-context results do not establish complete-RAG quality.
