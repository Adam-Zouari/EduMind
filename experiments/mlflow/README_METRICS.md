# MLflow Experiments - Comprehensive Metrics Guide

## 🎯 Quick Reference

This guide provides a complete overview of all metrics used in the MLflow experiments for evaluating the RAG pipeline.

---

## 📊 Metrics by Experiment Type

### 1. Chunking Experiments

**Purpose:** Find optimal text chunking strategy

| Metric | Description | Good Value | Priority |
|--------|-------------|------------|----------|
| **NDCG@5** | Ranking quality (position-aware) | >0.7 | ⭐⭐⭐⭐⭐ |
| **Precision@5** | Relevance of top-5 results | >0.6 | ⭐⭐⭐⭐ |
| **Chunk Coherence** | Semantic unity within chunks | >1.2 | ⭐⭐⭐⭐ |
| **Size Distribution** | Mean, median, std of chunk sizes | Consistent | ⭐⭐⭐ |
| **MRR** | First result quality | >0.7 | ⭐⭐⭐ |

---

### 2. Embedding Experiments

**Purpose:** Select best embedding model for retrieval

| Metric | Description | Good Value | Priority |
|--------|-------------|------------|----------|
| **NDCG@5** | Ranking quality (PRIMARY) | >0.7 | ⭐⭐⭐⭐⭐ |
| **NDCG@10** | Extended ranking quality | >0.65 | ⭐⭐⭐⭐ |
| **MAP** | Mean Average Precision | >0.5 | ⭐⭐⭐⭐ |
| **Precision@5** | Top-5 relevance | >0.6 | ⭐⭐⭐⭐ |
| **MRR** | First result quality | >0.7 | ⭐⭐⭐ |
| **Throughput** | Sentences/sec | >500 | ⭐⭐⭐ |
| **Latency** | Query encoding time (ms) | <10 | ⭐⭐⭐ |

**Note:** Recall@5 removed (was misleading with 60+ relevant docs)

---

### 3. Retrieval Experiments

**Purpose:** Optimize retrieval strategy (vector vs hybrid)

| Metric | Description | Good Value | Priority |
|--------|-------------|------------|----------|
| **NDCG@5** | Ranking quality | >0.7 | ⭐⭐⭐⭐⭐ |
| **Precision@5** | Top-5 relevance | >0.6 | ⭐⭐⭐⭐ |
| **Hit Rate@5** | Success rate (≥1 relevant) | >0.9 | ⭐⭐⭐⭐ |
| **Diversity** | Result variety | 0.3-0.7 | ⭐⭐⭐⭐ |
| **MRR** | First result quality | >0.7 | ⭐⭐⭐ |
| **Latency** | Retrieval time (ms) | <20 | ⭐⭐⭐ |

---

### 4. LLM Experiments

**Purpose:** Select best LLM for answer generation

| Metric | Description | Good Value | Priority |
|--------|-------------|------------|----------|
| **Correctness** | Factual accuracy | >0.7 | ⭐⭐⭐⭐⭐ |
| **Completeness** | Coverage of key points | >0.6 | ⭐⭐⭐⭐⭐ |
| **Conciseness** | Appropriate length | >0.7 | ⭐⭐⭐⭐ |
| **Faithfulness** | Grounded in context | >0.8 | ⭐⭐⭐⭐⭐ |
| **Context Precision** | Context utilization | >0.5 | ⭐⭐⭐⭐ |
| **Latency** | Generation time (ms) | <2000 | ⭐⭐⭐ |
| **Throughput** | Tokens/sec | >20 | ⭐⭐⭐ |

---

## 🔍 Metric Definitions

### Retrieval Quality Metrics

**NDCG@K (Normalized Discounted Cumulative Gain)**
- Industry standard for ranking evaluation
- Considers both relevance AND position
- Higher-ranked relevant docs contribute more
- Range: 0.0 to 1.0 (higher is better)

**Precision@K**
- Precision@K = (relevant docs in top-K) / K
- User-facing metric (what users see)
- Range: 0.0 to 1.0 (higher is better)

**MAP (Mean Average Precision)**
- Average of precision values at each relevant position
- More comprehensive than Precision@K
- Range: 0.0 to 1.0 (higher is better)

**Hit Rate@K**
- Binary: 1 if ≥1 relevant doc in top-K, else 0
- Measures "failure rate"
- Range: 0.0 to 1.0 (higher is better)

**Diversity**
- Diversity = 1 - (average pairwise similarity)
- Prevents redundant results
- Range: 0.0 (identical) to 1.0 (completely different)

**MRR (Mean Reciprocal Rank)**
- MRR = 1 / (rank of first relevant doc)
- Good for "quick answer" scenarios
- Range: 0.0 to 1.0 (higher is better)

### Chunk Quality Metrics

**Chunk Coherence**
- Ratio of intra-chunk to cross-boundary similarity
- Higher = better semantic unity
- Range: >1.0 is good (higher is better)

**Size Distribution**
- Mean, median, std, min, max for chars and tokens
- Ensures chunks fit in context windows
- Identifies outliers

### LLM Quality Metrics

**Correctness**
- Token overlap F1 score with reference
- Production: Use LLM-as-judge (GPT-4)
- Range: 0.0 to 1.0 (higher is better)

**Completeness**
- (key facts in answer) / (key facts in reference)
- Ensures full coverage
- Range: 0.0 to 1.0 (higher is better)

**Conciseness**
- Penalizes overly verbose/short answers
- Ideal: 20-150 words
- Range: 0.0 to 1.0 (higher is better)

**Faithfulness**
- (answer terms in context) / (total answer terms)
- Prevents hallucination
- Range: 0.0 to 1.0 (higher is better)

**Context Precision**
- (chunks used) / (chunks provided)
- Optimizes retrieval count
- Range: 0.0 to 1.0 (higher is better)

---

## 📈 How to Use These Metrics

### 1. Model Selection
- Use **NDCG@5** as primary metric for all retrieval experiments
- Balance with **latency** and **throughput** for production

### 2. Optimization
- Use **Context Precision** to optimize number of chunks retrieved
- Use **Diversity** to avoid redundant results
- Use **Hit Rate** to identify failure cases

### 3. Production Monitoring
- Track **NDCG@5**, **Precision@5**, **MRR** for retrieval
- Track **Correctness**, **Faithfulness** for LLM
- Monitor **Latency** and **Throughput** for performance

---

## 🚀 Running Experiments

```bash
# Chunking
cd mlflow/chunking_experiments && python run_experiments.py

# Embedding
cd mlflow/embedding_experiments && python run_experiments.py

# Retrieval
cd mlflow/retrieval_experiments && python run_experiments.py

# LLM
cd mlflow/llm_experiments && python run_experiments.py
```

---

## 📚 Additional Resources

- `METRICS_IMPROVEMENTS.md` - Detailed explanation of all new metrics
- `METRICS_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `EXPERIMENT_FIXES_SUMMARY.md` - Bug fixes and improvements

---

**All metrics are production-ready and tested!** ✅

