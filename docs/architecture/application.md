# Application pipeline

[Project overview](../../README.md) ·
[Architecture](overview.md) ·
[Streamlit application](ui.md) · [Extraction](extraction.md) ·
[RAG](rag.md)

## Role

`EduMindPipeline` is the single in-process boundary used by the current
application. It coordinates extraction and RAG without embedding UI state,
benchmark selection, or server lifecycle management into either subsystem.

## Operations

`process_file()`:

1. classifies and extracts the source;
2. normalizes the typed document;
3. creates exact-offset chunks and embeddings;
4. atomically replaces the logical document in Chroma;
5. returns extraction/indexing results, warnings, timings, and progress events.

`query()`:

1. embeds the question;
2. retrieves and token-budget-packs ranked evidence;
3. optionally invokes the pinned local generator;
4. returns retrieval hits, an optional cited answer, and stage timings.

The Streamlit controller owns upload checksums, temporary-file cleanup, duplicate
prevention, and presentation. The pipeline owns document and query processing.

## Readiness

Readiness reports the vector-server connection and local generator availability.
The pipeline does not start Docker or download models. Missing dependencies return
an actionable preparation or Compose command through the application layer.
