# Pipeline architecture

`EduMindPipeline` composes extraction and RAG while returning typed results, timings, warnings, readiness, and progress events. Details are maintained next to the implementation in [`src/edumind/pipeline/doc.md`](../../../src/edumind/pipeline/doc.md).

Application and service layers call this boundary or the same underlying production protocols. They do not implement alternate extraction, chunking, retrieval, or generation strategies.
