# DOCX-to-text benchmark

## Question and candidates

Which adapter best preserves readable DOCX content, hierarchy, tables, formulas, and order? Compare python-docx, Mammoth, Docling, and MinerU through the production `ExtractedDocument` contract.

## Data and procedure

The target corpus is 45 documents split 27/9/9 by source and covers paragraphs, headings, lists, captions, images, tables, and Office Math expressions. Manifests pin licenses, revisions, checksums, references, preprocessing, and seed. Candidate order uses seed 42. Standard/full use a cold invocation, two warmups, then three measured repetitions per document; equality across repetitions is reported without treating repeats as independent documents.

## Metrics

Primary reported metrics are normalized-token Content Precision, Content Recall, and Content F1 plus pairwise Reading Order Accuracy and exact normalized-line Block Precision/Recall/F1. Diagnostics are CER, WER, missing/hallucinated text rates, empty output, repeated-line rate, and determinism. Quality rates range 0-1 and higher is better; error rates are lower-better. Operations are cold invocation, p50/p95 document latency, documents/minute, process RAM, and VRAM when measurable.

The typed contract allows heading/list/caption/table/formula output. Table and formula metrics use independently verified OOXML-derived annotations and are omitted for unannotated documents. Rendering a DOCX to page images is a different measured candidate and cannot be hidden inside a parser configuration.

Standard/full store per-document values and 10,000-resample 95% intervals, then use correctness gates and Pareto selection.

## Commands and limits

```powershell
python experiments/benchmarks/extraction/docx/run.py --profile smoke
python experiments/benchmarks/extraction/docx/run.py --profile standard
python experiments/benchmarks/extraction/docx/run.py --profile full --shortlist SUMMARY_JSON
```

The result does not establish editing fidelity, styling, macros, or tracked-change behavior. Complete-content promotion requires explicit table/formula evidence rather than text-only success.

Artifacts are plan/provenance JSON, candidate JSON, per-document Parquet, summary intervals/comparisons, and local MLflow runs. Example: higher block F1 with lower content recall is a trade-off, not a universal win.
