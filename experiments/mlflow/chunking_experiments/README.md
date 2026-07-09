# Chunking Strategy Experiments

Tests different text chunking strategies to optimize retrieval quality and answer completeness.

## Strategies Tested

1. **Fixed Character (Baseline)** - 1000 chars, 200 overlap
2. **Fixed Character (Large)** - 1500 chars, 300 overlap
3. **Semantic Chunking** - Variable size, breaks at semantic boundaries
4. **Sentence Window** - 10 sentences per chunk, 2 sentence overlap
5. **Hierarchical** - Parent chunks (2000 chars) with child chunks (500 chars)

## Metrics Tracked

- **recall_at_5**: Retrieval quality (0-1)
- **mrr**: Mean Reciprocal Rank
- **answer_quality**: Quality of retrieved context (0-1)
- **total_chunks**: Number of chunks created
- **avg_chunk_size**: Average chunk size in characters
- **std_chunk_size**: Standard deviation of chunk sizes

## Artifacts Logged

- **chunk_distribution.png**: Histogram showing chunk size distribution
- **sample_chunks.json**: First 5 chunks as examples
- **chunk_statistics.json**: Detailed statistics about chunking

## How to Run

```bash
# From mlflow/chunking_experiments directory

# Test mode (first 2 strategies)
python run_experiments.py --test-mode

# Full run (all 5 strategies)
python run_experiments.py
```

## Expected Results

Based on the implementation:
- **Semantic Chunking** should achieve highest recall (~94%) and quality (~4.4/5)
- **Fixed Character (Baseline)** provides good baseline (~92% recall, 4.2/5 quality)
- **Hierarchical** offers good balance but adds complexity

## Viewing Results

```bash
mlflow ui
# Navigate to http://localhost:5000
# Select "chunking_experiments" experiment
```

Compare strategies on:
- Recall@5 vs Average Chunk Size
- Answer Quality across strategies
- Chunk size distribution patterns
