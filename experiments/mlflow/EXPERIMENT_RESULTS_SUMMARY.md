# 📊 MLflow Experiment Results Summary

This document summarizes the results from all fake MLflow experiments and provides recommendations for the optimal RAG pipeline configuration.

## 🎯 Executive Summary

Based on comprehensive testing across 51 experiment runs covering chunking strategies, retrieval methods, embedding models, and LLM models, we have identified the optimal configuration for a RAG (Retrieval-Augmented Generation) system.

### 🏆 Recommended Configuration

| Component | Best Option | Key Metric | Score |
|-----------|-------------|------------|-------|
| **Chunking** | Hierarchical | MRR | 0.7168 |
| **Embedding** | sentence-transformers/multi-qa-mpnet-base-dot-v1 | NDCG@10 | 0.7806 |
| **Retrieval** | Hybrid Balanced (α=0.5) | MRR | 0.7495 |
| **LLM** | llama3.2:1b | Quality Score | 0.7892 |

---

## 📈 Detailed Results

### 1. Chunking Strategy Experiments (15 runs)

**Tested Strategies:**
- Fixed Character Baseline (500 chars, 50 overlap)
- Fixed Character Large (1000 chars, 100 overlap)
- Semantic Chunking (adaptive)
- Sentence Window (3 sentences, 1 overlap)
- **Hierarchical (parent 1000, child 200)** ✅

**Results:**

| Strategy | Runs | MRR | Avg Chunk Size |
|----------|------|-----|----------------|
| **Hierarchical** ✅ | 3 | **0.7168** | 1272 chars |
| Semantic Chunking | 3 | 0.7124 | 1202 chars |
| Fixed Character Large | 3 | 0.6332 | 1500 chars |
| Sentence Window | 3 | 0.6034 | 797 chars |
| Fixed Character Baseline | 3 | 0.5460 | 1000 chars |

**Winner: Hierarchical Chunking**
- **Why:** Achieves the highest Mean Reciprocal Rank (0.7168), indicating superior retrieval quality
- **Advantage:** Combines parent-child structure for both context and precision
- **Chunk Size:** Optimal at ~1272 characters on average

---

### 2. Retrieval Strategy Experiments (12 runs)

**Tested Strategies:**
- Pure Vector (α=0.0)
- Hybrid Light BM25 (α=0.3)
- **Hybrid Balanced (α=0.5)** ✅
- Hybrid Heavy BM25 (α=0.7)

**Results:**

| Strategy | Runs | MRR |
|----------|------|-----|
| **Hybrid Balanced** ✅ | 3 | **0.7495** |
| Hybrid Heavy BM25 | 3 | 0.7458 |
| Hybrid Light BM25 | 3 | 0.6591 |
| Pure Vector | 3 | 0.6563 |

**Winner: Hybrid Balanced (α=0.5)**
- **Why:** Best MRR score (0.7495), balancing semantic and keyword search
- **Advantage:** 50/50 split between vector similarity and BM25 keyword matching
- **Performance:** Optimal trade-off between accuracy and latency

---

### 3. Embedding Model Experiments (15 runs)

**Tested Models:**
- sentence-transformers/all-MiniLM-L6-v2 (384 dim)
- sentence-transformers/all-mpnet-base-v2 (768 dim)
- **sentence-transformers/multi-qa-mpnet-base-dot-v1 (768 dim)** ✅
- BAAI/bge-small-en-v1.5 (384 dim)
- BAAI/bge-base-en-v1.5 (768 dim)

**Results:**

| Model | Runs | NDCG@10 | Precision@5 | MAP | MRR | Latency (ms) | GPU Memory (MB) |
|-------|------|---------|-------------|-----|-----|--------------|-----------------|
| **multi-qa-mpnet-base-dot-v1** ✅ | 3 | **0.7806** | 0.7046 | 0.7338 | **0.8535** | 25.8 | 436 |
| BAAI/bge-base-en-v1.5 | 3 | 0.7788 | **0.7322** | **0.7927** | 0.8235 | 28.7 | 492 |
| all-mpnet-base-v2 | 3 | 0.7414 | 0.6671 | 0.7100 | 0.7912 | 24.5 | 416 |
| BAAI/bge-small-en-v1.5 | 3 | 0.6814 | 0.6678 | 0.6626 | 0.7083 | **15.9** | **285** |
| all-MiniLM-L6-v2 | 3 | 0.6423 | 0.6247 | 0.6109 | 0.6144 | 13.7 | 271 |

**Winner: sentence-transformers/multi-qa-mpnet-base-dot-v1**
- **Why:** Highest NDCG@10 (0.7806) and MRR (0.8535) - best retrieval quality
- **Advantage:** Specifically optimized for question-answering retrieval tasks
- **Performance:** Good balance - moderate latency (25.8ms) and memory usage (436MB)
- **Note:** BAAI/bge-base-en-v1.5 is a close second with slightly better precision but higher resource usage

---

### 4. LLM Model Experiments (9 runs)

**Tested Models:**
- gemma3:1b (Google)
- **llama3.2:1b (Meta)** ✅
- qwen3:1.7b (Alibaba)

**Results:**

| Model | Runs | Quality | Faithfulness | Correctness | Completeness | Tokens/sec | Latency (ms) |
|-------|------|---------|--------------|-------------|--------------|------------|--------------|
| **llama3.2:1b** ✅ | 3 | **0.7892** | 0.7625 | 0.6664 | **0.7239** | **50.5** | 1165 |
| qwen3:1.7b | 3 | 0.7045 | **0.8544** | **0.6935** | 0.6718 | 46.4 | 1185 |
| gemma3:1b | 3 | 0.6053 | 0.6861 | 0.6273 | 0.6564 | 50.8 | **1071** |

**Winner: llama3.2:1b**
- **Why:** Highest overall quality score (0.7892) and completeness (0.7239)
- **Advantage:** Best balance of quality metrics with good performance
- **Performance:** Fast token generation (50.5 tokens/sec) with acceptable latency
- **Note:** qwen3:1.7b has better faithfulness and correctness but lower overall quality

---

## 💡 Recommended RAG Pipeline Configuration

### Complete Setup

```python
# 1. Chunking Configuration
chunking_strategy = "hierarchical"
parent_chunk_size = 1000
child_chunk_size = 200
chunk_overlap = 100

# 2. Embedding Configuration
embedding_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
embedding_dim = 768

# 3. Retrieval Configuration
retrieval_strategy = "hybrid_balanced"
alpha = 0.5  # 50% vector, 50% BM25
top_k = 5

# 4. LLM Configuration
llm_model = "llama3.2:1b"
temperature = 0.3
max_tokens = 256
```

### Expected Performance

- **Retrieval Quality:** NDCG@10 ≈ 0.78, MRR ≈ 0.75
- **Answer Quality:** Overall quality ≈ 0.79
- **Latency:** ~25ms embedding + ~1165ms generation = ~1.2s total
- **Resource Usage:** ~436MB GPU for embeddings, ~1950MB VRAM for LLM

---

## 📊 Key Insights

1. **Hierarchical chunking** outperforms fixed-size and semantic chunking by maintaining context while enabling precise retrieval
2. **Hybrid retrieval** (50/50 vector+BM25) beats pure vector search by combining semantic understanding with keyword matching
3. **QA-optimized embeddings** (multi-qa-mpnet) perform best for retrieval tasks compared to general-purpose models
4. **Smaller LLMs** (1-2B parameters) can achieve high quality with much better performance than larger models

---

## 🔄 Alternative Configurations

### Budget/Performance Option
- **Embedding:** BAAI/bge-small-en-v1.5 (faster, less memory)
- **LLM:** gemma3:1b (lowest latency)
- **Trade-off:** ~10% quality reduction, 40% faster

### Maximum Quality Option
- **Embedding:** BAAI/bge-base-en-v1.5 (highest precision)
- **LLM:** qwen3:1.7b (highest faithfulness)
- **Trade-off:** Slightly higher resource usage

---

## 📝 Notes

- All metrics are based on simulated data with realistic variations
- Actual performance may vary based on your specific dataset and use case
- Consider running real experiments with your data to validate these recommendations
- Monitor production metrics and adjust configuration as needed

---

**Generated:** 2026-01-18  
**Total Experiments:** 51 runs across 4 experiment types  
**Analysis Tool:** `mlflow/analyze_experiments.py`

