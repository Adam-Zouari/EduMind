# Run instructions

Application and APIs:

```powershell
edumind ui
edumind extraction-api
edumind rag-api
```

Benchmark examples:

```powershell
edumind benchmark preflight
edumind benchmark prepare qasper
edumind benchmark prepare assets ASSET_PLAN_JSON
edumind benchmark prepare extraction-models
edumind benchmark prepare huggingface-models
edumind benchmark prepare ollama-models
edumind benchmark extraction all
edumind benchmark rag all
edumind benchmark systems vectordb
edumind benchmark all

edumind benchmark --profile standard extraction image
edumind benchmark --profile standard rag chunking-embedding
```

Review/report commands require paths:

```powershell
edumind benchmark review export SUMMARY_JSON REVIEW_CSV
edumind benchmark review import REVIEW_CSV
edumind benchmark report SUMMARY_JSON
```

Smoke validates execution only. Do not promote a strategy from smoke. Standard and full candidates must pass their documented correctness/resource gates; final RAG additionally requires 60 blinded judgments and the locked test.
