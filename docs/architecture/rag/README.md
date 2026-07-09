# RAG Package Guide

The RAG subsystem lives in `src/edumind/rag` and turns normalized OCR output into a searchable, answerable knowledge base.

## Main modules

```text
src/edumind/rag/
|-- rag_pipeline.py
|-- embedder.py
|-- llm_generator.py
|-- ocr_processor.py
|-- text_chunker.py
`-- vector_store.py
```

## Core workflow

1. receive a normalized document payload
2. split text into chunks
3. embed those chunks
4. store them in ChromaDB
5. query by similarity
6. optionally generate answers through Ollama

## Main entrypoint

```python
from edumind.rag.rag_pipeline import RAGPipeline

rag = RAGPipeline(use_llm=True)

chunks = rag.ingest_document(
    {
        "text": "Example content",
        "source": "example.txt",
        "format_type": "text",
    }
)

answer = rag.generate_answer("What is the example about?")
print(chunks)
print(answer["answer"])
```

## Configuration

The pipeline reads shared defaults from `config/base.yaml`.

Current defaults include:

- embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- chunk size: `1000`
- chunk overlap: `200`
- vector collection: `ocr_documents`
- Ollama model: `qwen3:1.7b`

## Persistence

The local vector store persists in:

`artifacts/rag/vector_store/`

MLflow logging is enabled when the dependency is installed. Set `EDUMIND_MLFLOW_TRACKING_URI` if you want runtime logging to use the shared store under `artifacts/mlflow/`.

## Useful methods

- `ingest_document()`
- `ingest_documents()`
- `ingest_from_json()`
- `query()`
- `generate_answer()`
- `get_stats()`
- `reset()`
