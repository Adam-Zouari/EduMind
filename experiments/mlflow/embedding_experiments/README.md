# Embedding Model Experiments

Tests different sentence embedding models for optimal retrieval quality and efficiency.

## Models Tested

1. **all-MiniLM-L6-v2** - Lightweight and fast (384 dim)
2. **all-mpnet-base-v2** - Balanced performance (768 dim)
3. **multi-qa-mpnet-base-dot-v1** - Optimized for QA (768 dim)
4. **bge-small-en-v1.5** - BGE small model (384 dim)
5. **bge-base-en-v1.5** - BGE base model (768 dim)

## Metrics Tracked

- **recall_at_5**: Re trieval quality metric (0-1)
- **mrr**: Mean Reciprocal Rank (0-1)
- **throughput_sent_per_sec**: Encoding throughput
- **avg_query_latency_ms**: Query encoding latency
- **gpu_memory_mb**: GPU memory usage
- **model_load_time_sec**: Model loading time

## How to Run

```bash
# Run all experiments
cd mlflow/embedding_experiments
python run_experiments.py

# Test mode (only first 2 models)
python run_experiments.py --test-mode
```

## Viewing Results

```bash
# Start MLflow UI
mlflow ui

# Open browser to:
# http://localhost:5000
```

Then navigate to the `embedding_experiments` experiment to compare models.
