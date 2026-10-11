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
4. replaces the logical document's chunks in Chroma;
5. returns extraction/indexing results, warnings, timings, and progress events.

`query()`:

1. embeds the question;
2. retrieves and token-budget-packs ranked evidence;
3. optionally invokes the pinned local generator;
4. returns retrieval hits, an optional cited answer, and stage timings.

The Streamlit controller owns upload checksums, temporary-file cleanup, duplicate
prevention, and presentation. The pipeline owns document and query processing.

## Model-workload isolation

Uploads and answers share one process-wide model-workload lock. A concurrent
request waits until the active operation completes; this is mutual exclusion,
not a separate background job service or a FIFO scheduling guarantee.

Before extraction, the pipeline unloads its cached embedding and generation
models. Extraction models are released in a `finally` block before indexing
starts, including after a failed extraction. Video transcribes audio first,
releases ASR weights, and then parses the selected frames. Ordinary question
answering does not rerun extraction.

This allows an isolated extraction stage to use the available GPU memory without
reserving a fixed allowance for another model. It does not coordinate separate
application processes, external GPU applications, or Windows driver paging.
Run one application process on this laptop and keep unrelated GPU workloads
closed. Provisionally configured CPU execution and explicit request profiles
remain unchanged; benchmark runners do not promote application models/devices.

## Readiness

Readiness reports the vector-server connection and local generator availability.
The pipeline does not start Docker or download models. Missing dependencies return
an actionable preparation or Compose command through the application layer.
