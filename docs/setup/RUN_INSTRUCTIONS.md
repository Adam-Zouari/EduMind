# Run instructions

Application:

```powershell
docker compose -f infrastructure/chroma.yml up -d
streamlit run apps/streamlit_app.py
```

Common preparation:

```powershell
python experiments/benchmarks/prepare.py smoke-fixtures
python experiments/benchmarks/prepare.py app-models
python experiments/benchmarks/prepare.py qasper
python experiments/benchmarks/prepare.py huggingface-models
python experiments/benchmarks/prepare.py extraction-models
python experiments/benchmarks/prepare.py ollama-models
```

Representative direct benchmarks:

```powershell
python experiments/benchmarks/extraction/image/run.py --profile standard
python experiments/benchmarks/rag/chunking_embedding/run.py --profile standard
python experiments/benchmarks/rag/retrieval/run.py --profile standard --embedding-summary EMBEDDING_SUMMARY_JSON
python experiments/benchmarks/rag/generation/run.py --profile standard
python experiments/benchmarks/rag/final/run.py --profile standard --retrieval-summary RETRIEVAL_SUMMARY_JSON --generation-summary GENERATION_SUMMARY_JSON
```

Vector servers:

```powershell
python experiments/benchmarks/prepare.py vectordb
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml up -d
python experiments/benchmarks/vectordb/run.py --profile smoke
python experiments/benchmarks/vectordb/run.py --profile standard
```

Use `--shortlist STANDARD_SUMMARY_JSON` for finalist-only full component runs. Human review is `python experiments/benchmarks/review.py export|import ...`.
