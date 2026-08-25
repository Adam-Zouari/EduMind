# Final RAG benchmark

[Benchmark overview](../overview.md) · [Benchmark manual](../methodology.md) ·
[Retrieval stage](retrieval.md) · [Generation stage](generation.md)

This experiment crosses only explicitly approved retrieval and generator finalists. Every repetition performs chunk/query embedding, selected dense/BM25/RRF/reranking, 2,048-token context packing, direct Hugging Face prompting, and cited generation. Retrieval, generation, and complete latency remain separate.

Standard uses the validation selection manifest and one common generator device. It reports retrieval metrics, automated generation diagnostics, operational metrics, and a human-review export. Twenty stratified questions multiplied by three anonymous systems produce 60 blinded judgments. Reviewers score Faithfulness, Answer Correctness, Completeness, Citation Accuracy, and answerability before system identities are revealed.

Exactly one approved system is run once on the locked test. Tuning after opening locked-test output requires a new benchmark version. The runner never edits production defaults.

```powershell
python experiments/benchmarks/rag/final/run.py --profile standard --retrieval-summary RETRIEVAL_SUMMARY --generation-summary GENERATION_SUMMARY --device cuda
python experiments/benchmarks/rag/final/run.py --profile full --shortlist SUMMARY_JSON --device cuda
python experiments/benchmarks/rag/final/confirm_extraction.py --reference-manifest VERIFIED.json --extracted-manifest EXTRACTED.json
```

Extraction confirmation compares the selected RAG stack on verified reference text and selected-parser text, reporting retrieval, citation, human answer quality, and end-to-end latency deltas.
