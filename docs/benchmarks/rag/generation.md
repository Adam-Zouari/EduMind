# Generation benchmark

[Benchmark overview](../overview.md) · [Benchmark manual](../methodology.md) ·
[Candidate selection](../model-selection.md) · [Preparation guide](../../setup/installation.md)

## Question and candidates

Which pinned direct Hugging Face checkpoint produces the best grounded, cited answer from identical frozen evidence? The candidates are Qwen3 1.7B with thinking disabled (control), MiniCPM5 1B reasoning, G9v3 3B reasoning, and Qwen3.5 4B reasoning.

All use official chat templates, native checkpoint dtype through `torch_dtype="auto"`, temperature 0, seed 42, an 8,192-token model context, a 256-token output limit, and one explicit whole-model device. Quantization, mixed CPU/GPU placement, and silent fallback are forbidden. Standard/full require `--device`; the same value applies to every candidate.

## Procedure

The runner selects up to 24 development questions balanced across evidence and answer types. Answerable inputs receive verified frozen evidence; unanswerable inputs receive their document. This isolates the generator from retrieval. Each model is explicitly unloaded, loaded for a cold observation, warmed twice, measured three times per question, and unloaded with memory cleanup before the next candidate.

## Metrics

Citation F1 is the main metric for this frozen-context screen. Answerability Balanced Accuracy, Exact Match, Token F1, ROUGE-L, pinned HHEM faithfulness, refusal precision/recall/F1, unsupported-answer rate, malformed-output rate, and determinism provide the rest of the automated evidence. Final system selection additionally uses blinded Human Faithfulness, Human Answer Correctness, Human Completeness, and Citation Accuracy.

Operational metrics include cold load, visible-answer TTFT, total p50/p95 latency, prompt and output tokens, prompt evaluation time, generation time, tokens/second, answers/minute, process RAM, and VRAM. HHEM is diagnostic and never replaces human review.

The runner requires the declared metrics but does not combine them into a score or select a model.

```powershell
python experiments/benchmarks/rag/generation/run.py --profile smoke
python experiments/benchmarks/rag/generation/run.py --profile standard --device cuda
python experiments/benchmarks/rag/generation/run.py --profile full --shortlist DECISION_JSON --device cuda
```
