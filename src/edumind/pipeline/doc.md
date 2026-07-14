# Pipeline subsystem

`EduMindPipeline` is the typed application boundary joining extraction, indexing, retrieval, and generation. It contains orchestration only; extraction and RAG strategy logic stays in their owning packages.

`process_file` returns `DocumentProcessResult` with the typed extracted document, optional ingest report, stage timings, and warnings. A logical `source_name` may be supplied for temporary uploads so indexes and citations never expose random temporary filenames. `query` returns `PipelineQueryResult` with ranked hits, an optional generated answer, timings, and warnings.

Both operations emit `ProgressEvent` values. `readiness` distinguishes configured extraction capabilities from local generation availability. `reset_index` delegates the destructive action to the RAG boundary; services serialize it and the UI requires confirmation.

The service client is a transport adapter for deployments that intentionally split extraction and RAG processes. It uses the extraction API terminology and does not recreate local business rules.
