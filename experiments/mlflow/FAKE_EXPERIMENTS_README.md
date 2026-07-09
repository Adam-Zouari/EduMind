# 🎭 Fake MLflow Experiments Generator

This tool generates realistic fake MLflow experiments for testing and demonstration purposes. It creates experiments with logical metrics, realistic variations, and proper correlations between different metrics.

## 📋 Table of Contents
- [Quick Start](#quick-start)
- [Experiment Types](#experiment-types)
- [Usage](#usage)
- [Generated Metrics](#generated-metrics)
- [Examples](#examples)

## 🚀 Quick Start

### Generate All Experiments (Default)
```bash
# Windows (PowerShell)
.\generate_fake_experiments.ps1 --runs 5

# Windows (Batch)
generate_fake_experiments.bat --runs 5

# Python directly
python generate_fake_experiments.py --runs 5
```

### Generate Specific Experiment Types
```bash
# Only chunking experiments
python generate_fake_experiments.py --chunking --runs 3

# Only retrieval experiments
python generate_fake_experiments.py --retrieval --runs 3

# Only embedding experiments
python generate_fake_experiments.py --embedding --runs 3

# Only LLM experiments
python generate_fake_experiments.py --llm --runs 3

# Multiple types
python generate_fake_experiments.py --chunking --retrieval --runs 5

# All types explicitly
python generate_fake_experiments.py --all --runs 5
```

### View Experiments
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open: http://localhost:5000

## 📊 Experiment Types

### 1. Chunking Strategy Experiments
Tests different text chunking approaches for RAG systems.

**Strategies:**
- **Fixed Character Baseline** (500 chars, 50 overlap)
- **Fixed Character Large** (1000 chars, 100 overlap)
- **Semantic Chunking** (adaptive, similarity-based)
- **Sentence Window** (3 sentences, 1 overlap)
- **Hierarchical** (parent 1000, child 200)

**Metrics tracked:**
- Chunk statistics (count, avg/min/max size, std dev)
- Retrieval quality (precision, recall, NDCG, MRR)
- Semantic coherence
- Processing time

### 2. Retrieval Strategy Experiments
Tests different retrieval approaches combining vector and keyword search.

**Strategies:**
- **Pure Vector** (ChromaDB only, α=0.0)
- **Hybrid Light BM25** (30% BM25, α=0.3)
- **Hybrid Balanced** (50% BM25, α=0.5)
- **Hybrid Heavy BM25** (70% BM25, α=0.7)

**Metrics tracked:**
- Precision, Recall, NDCG, MRR
- Hit rate, diversity
- Query latency

### 3. Embedding Model Experiments
Tests different embedding models for semantic retrieval.

**Models:**
- **sentence-transformers/all-MiniLM-L6-v2** (384 dim, lightweight)
- **sentence-transformers/all-mpnet-base-v2** (768 dim, balanced)
- **sentence-transformers/multi-qa-mpnet-base-dot-v1** (768 dim, QA-optimized)
- **BAAI/bge-small-en-v1.5** (384 dim, BGE small)
- **BAAI/bge-base-en-v1.5** (768 dim, BGE base)

**Metrics tracked:**
- Retrieval quality (precision@5, NDCG@5/10, MAP, MRR)
- Performance (throughput, latency, GPU memory)
- Model load time
- Embedding dimension

### 4. LLM Model Experiments
Tests different LLM models for answer generation in RAG systems.

**Models:**
- **qwen3:1.7b** (Alibaba's efficient LLM)
- **gemma3:1b** (Google's compact model)
- **llama3.2:1b** (Meta's latest small model)

**Metrics tracked:**
- Quality metrics (overall quality, faithfulness, correctness, completeness, conciseness)
- Context precision
- Performance (tokens/sec, latency, VRAM usage)

## 🎯 Usage

### Command Line Arguments

```
--chunking          Generate chunking experiments
--retrieval         Generate retrieval experiments
--embedding         Generate embedding experiments
--llm               Generate LLM experiments
--all               Generate all experiment types (default if none specified)
--runs N            Number of runs per strategy/model (default: 5)
--days-ago N        Start experiments N days ago (default: 30)
```

### Examples

```bash
# Generate 10 runs for each strategy, starting 60 days ago
python generate_fake_experiments.py --all --runs 10 --days-ago 60

# Generate only chunking and retrieval with 3 runs each
python generate_fake_experiments.py --chunking --retrieval --runs 3

# Generate embedding experiments with 7 runs
python generate_fake_experiments.py --embedding --runs 7
```

## 📈 Generated Metrics

All metrics include:
- **Realistic base values** based on typical performance
- **Random noise** to simulate real-world variation
- **Logical correlations** (e.g., larger chunks → fewer chunks)
- **Standard deviations** for metrics with multiple measurements

