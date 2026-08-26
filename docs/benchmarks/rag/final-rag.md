# Final RAG benchmark

[Benchmark overview](../overview.md) · [Benchmark manual](../methodology.md) ·
[Retrieval stage](retrieval.md) · [Generation stage](generation.md)

This experiment crosses only explicitly approved retrieval and generator finalists. Every repetition performs chunk/query embedding, selected dense/BM25/RRF/reranking, 2,048-token context packing, direct Hugging Face prompting, and cited generation. Retrieval, generation, and complete latency remain separate.

Citation F1 is the main automated metric in MLflow. It does not decide the final system; the engineer reviews retrieval, answerability, faithfulness diagnostics, latency/resources, and the blinded human judgments together.

Standard uses the validation selection manifest and one common generator device. It reports retrieval metrics, automated generation diagnostics, operational metrics, and a human-review export. Twenty stratified questions multiplied by three anonymous systems produce 60 blinded judgments. Reviewers score Faithfulness, Answer Correctness, Completeness, Citation Accuracy, and answerability before system identities are revealed.

After the standard run is complete, the engineer records exactly three systems in a decision file. That file controls the blinded export; the export command never chooses the systems from their scores:

```powershell
python experiments/benchmarks/review.py export THREE_SYSTEM_DECISION.json human-review.csv
python experiments/benchmarks/review.py import human-review.csv
```

Import validates every judgment, writes `human-review.results.json`, and logs the
CSV, blinded identity map, per-system human metrics, and results JSON back to the
original final-RAG parent run in MLflow before the engineer chooses the locked-test system.

Exactly one approved system is run once on the locked test. Tuning after opening locked-test output requires a new benchmark version. The runner never edits production defaults.

```powershell
python experiments/benchmarks/rag/final/run.py --profile standard --retrieval-selection RETRIEVAL_DECISION --generation-selection GENERATION_DECISION --device cuda
python experiments/benchmarks/rag/final/run.py --profile full --shortlist FINAL_DECISION --review-results REVIEW_RESULTS --confirm-locked-test --device cuda
python experiments/benchmarks/rag/final/confirm_extraction.py --reference-manifest VERIFIED.json --extracted-manifest EXTRACTED.json
```

Extraction confirmation compares the selected RAG stack on verified reference text and selected-parser text, reporting retrieval, citation, human answer quality, and end-to-end latency deltas.
