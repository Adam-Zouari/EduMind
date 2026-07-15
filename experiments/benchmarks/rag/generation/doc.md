# Generation benchmark

## Question and candidates

Which pinned local Ollama profile produces the best grounded, cited answer from frozen evidence? Standard screens Qwen3 1.7B, Qwen3.5 4B/9B direct and thinking, Gemma3 4B/12B, Ministral 3 8B, and GPT-OSS 20B low/medium reasoning. Freezing evidence isolates generation.

## Data and procedure

The runner deterministically balances up to 24 development questions across answerable and unanswerable cases. Answerable cases receive verified evidence; unanswerable cases receive their document. Evidence is capped at 3,500 tokens to leave room within the 4,096-token prompt target. Every profile pins its Ollama digest, temperature 0, seed 42, 8,192 context, and 256 answer-token maximum. The model is unloaded for cold measurement, warmed twice, then standard/full generate three measured answers per question. Metrics are averaged within a question; repeated questions remain one bootstrap unit.

## Metrics

Automated generation metrics are Exact Match, Token F1, ROUGE-L, pinned local NLI Faithfulness, answerability correctness/balanced accuracy, refusal precision/recall/F1, unsupported-answer rate, malformed-output rate, and determinism. Citation Precision is supported cited context IDs / cited IDs; Citation Recall is supported context IDs cited / supported context IDs; Citation F1 is their harmonic mean. These are context-citation diagnostics, not claim-level human judgments.

Operations include complete request p50/p95, visible-answer TTFT p50/p95, Ollama prompt-evaluation and model-generation durations, prompt/answer token counts, an explicitly labeled reasoning word-count estimate, tokens/second, answers/minute, cold load, evaluator-process RAM, and Ollama-reported model RAM/VRAM. The resource gate uses evaluator peak RAM plus Ollama's loaded-model allocation and requires the sum below 28 GB; the two sources remain separately visible.

Human Faithfulness, Answer Correctness, Completeness, and Citation Accuracy use 0-2 rubrics only after complete-RAG finalists are exported. They are not invented by this automated screen. Standard/full retain per-question observations and bootstrap/paired intervals; up to three profiles are explicitly approved for final RAG.

## Commands and limits

```powershell
python experiments/benchmarks/rag/generation/run.py --profile smoke --no-mlflow
python experiments/benchmarks/rag/generation/run.py --profile standard
python experiments/benchmarks/rag/generation/run.py --profile full --shortlist SUMMARY_JSON
```

Local MLflow stores the Parquet evaluation artifacts, including answers needed for review; normal application logs do not. Automated or frozen-context results do not establish complete-RAG quality.

Artifacts are plan/provenance, candidate JSON, per-question Parquet, paired intervals, timing/resource metrics, and local MLflow runs. Example: higher ROUGE-L cannot rescue unsupported cited claims, malformed citations, or p95 above 30 seconds.
