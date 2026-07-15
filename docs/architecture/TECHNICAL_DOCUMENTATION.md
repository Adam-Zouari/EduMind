# Current architecture

```text
Streamlit (apps/) -> EduMindPipeline -> extraction + dense RAG -> Chroma HTTP -> Ollama

datasets -> experiments/benchmarks -> metrics + MLflow -> human approval
                                                        -> later production edit
```

`src/edumind` contains production code only. `apps` owns the complete Streamlit UI and its session controller. `config/base.yaml` is the only production configuration source. There is no FastAPI/service layer, packaged benchmark command, recommendation manifest, embedded vector database, production BM25 index, or duplicate Streamlit application.

The application is intentionally single-process while component selection is still experimental. Chroma server, token 256/32, MiniLM, dense top-5 retrieval, a 2,048-token context budget, and Qwen 3 1.7B are provisional defaults.

Alternative extraction engines reuse production extractor contracts. Alternative chunking, embedding, retrieval, generation, and database implementations live under `experiments/benchmarks`; benchmark results never edit production automatically.
