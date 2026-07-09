# Metrics Improvements Summary

This document describes all the new metrics added to the MLflow experiments to provide comprehensive evaluation of the RAG pipeline.

## Overview of Changes

### 1. **Chunking Experiments** - New Metrics Added

#### Chunk Quality Metrics:
- **Chunk Coherence** - Measures semantic coherence within chunks
  - Higher score = chunks contain semantically related content
  - Computed as ratio of intra-chunk similarity to cross-boundary similarity
  
- **Chunk Size Statistics** - Comprehensive size distribution analysis
  - Mean, median, std, min, max for both characters and tokens
  - Helps ensure chunks fit in context windows
  - Identifies outliers and distribution patterns

#### Retrieval Quality Metrics:
- **Precision@5** - Of top-5 results, how many are relevant?
  - More appropriate than Recall when there are many relevant docs
  - User-facing metric (what users actually see)
  
- **NDCG@5** - Normalized Discounted Cumulative Gain
  - **Industry standard** for ranking evaluation
  - Considers both relevance AND position
  - Higher-ranked relevant docs contribute more to score

---

### 2. **Embedding Experiments** - New Metrics Added

#### Added:
- **NDCG@5** - Primary ranking quality metric (industry standard)
- **NDCG@10** - Extended ranking quality for top-10 results
- **MAP** - Mean Average Precision
  - Considers precision at all relevant positions
  - More comprehensive than Precision@K alone

#### Removed:
- **Recall@5** - Removed from primary metrics
  - Was misleading (0.05 = 5%) due to 60+ relevant docs per query
  - Not useful when there are many relevant documents
  - Kept in code but not logged as primary metric

#### Primary Metrics Now:
1. **NDCG@5** - Best overall ranking metric
2. **Precision@5** - User-facing quality
3. **MAP** - Comprehensive precision across all positions
4. **MRR** - First result quality

---

### 3. **Retrieval Experiments** - New Metrics Added

#### Retrieval Quality:
- **Hit Rate@5** - Did we get at least 1 relevant doc in top-5?
  - Binary success metric (0 or 1 per query)
  - Easy to understand for stakeholders
  - Measures "failure rate" of retrieval

- **Diversity** - How different are retrieved documents from each other?
  - Diversity = 1 - (average pairwise similarity)
  - Prevents returning 5 nearly-identical chunks
  - Ensures better coverage of information
  - Computed from document embeddings

#### Also Added:
- **NDCG@5** - Ranking quality
- **Precision@5** - Relevance quality

---

### 4. **LLM Experiments** - New Metrics Added

#### Answer Quality Metrics:

- **Correctness** - Is the answer factually correct?
  - Computed via token overlap F1 score with reference
  - In production: Use LLM-as-judge (GPT-4) or human evaluation
  - Current: Heuristic-based (token overlap)

- **Completeness** - Does answer cover all key points?
  - Completeness = (key facts in answer) / (key facts in reference)
  - Checks if answer addresses all parts of the question
  - Prevents incomplete or partial answers

- **Conciseness** - Is the answer appropriately concise?
  - Penalizes overly verbose or too-short answers
  - Ideal: 20-150 words (configurable)
  - Can compare to reference answer length if available

#### Context Utilization Metrics:

- **Context Precision** - Which context chunks were actually used?
  - Context Precision = (chunks used) / (chunks provided)
  - Helps optimize number of chunks to retrieve
  - Identifies if we're providing too much irrelevant context
  - Returns: precision score + list of used context indices

---

## Metrics by Category

### **Retrieval Quality** (Chunking, Embedding, Retrieval)
| Metric | What it Measures | Good Value | When to Use |
|--------|------------------|------------|-------------|
| **NDCG@K** | Ranking quality (position-aware) | >0.7 | Always - industry standard |
| **Precision@K** | Relevance of top-K | >0.6 | User-facing quality |
| **MAP** | Precision across all positions | >0.5 | Comprehensive evaluation |
| **MRR** | First result quality | >0.7 | Quick answer scenarios |
| **Hit Rate@K** | Success rate (≥1 relevant) | >0.9 | Failure analysis |
| **Diversity** | Result variety | 0.3-0.7 | Avoid redundancy |

### **Chunk Quality** (Chunking)
| Metric | What it Measures | Good Value | When to Use |
|--------|------------------|------------|-------------|
| **Coherence** | Semantic unity within chunks | >1.2 | Chunk strategy comparison |
| **Size Stats** | Distribution of chunk sizes | Consistent | Ensure context window fit |

### **Answer Quality** (LLM)
| Metric | What it Measures | Good Value | When to Use |
|--------|------------------|------------|-------------|
| **Correctness** | Factual accuracy | >0.7 | Primary quality metric |
| **Completeness** | Coverage of key points | >0.6 | Ensure full answers |
| **Conciseness** | Appropriate length | >0.7 | Avoid verbosity |
| **Faithfulness** | Grounded in context | >0.8 | Prevent hallucination |
| **Context Precision** | Context utilization | >0.5 | Optimize retrieval count |

---

## Implementation Notes

### Heuristic vs. LLM-based Evaluation

**Current Implementation (Heuristic):**
- Fast and cheap
- Good for development and iteration
- Based on token overlap, length, etc.

**Production Recommendation (LLM-as-Judge):**
- Use GPT-4 or Claude to evaluate answers
- More accurate and nuanced
- Can detect subtle issues
- Example prompt:
  ```
  Rate the following answer on correctness (1-5):
  Question: {query}
  Reference: {reference}
  Answer: {answer}
  ```

### Computing Diversity

Diversity requires embeddings of retrieved documents:
```python
# Get embeddings for retrieved docs
embeddings = np.array([doc['embedding'] for doc in retrieved_docs])

# Compute diversity
diversity = compute_diversity(embeddings)
# Returns: 0.0 (identical) to 1.0 (completely different)
```

### Context Precision Details

Returns detailed breakdown:
```python
result = evaluate_context_precision(answer, context_chunks)
# {
#   'context_precision': 0.67,  # 2 out of 3 chunks used
#   'contexts_used': 2,
#   'contexts_provided': 3,
#   'used_context_indices': [0, 2]  # Which chunks were used
# }
```

---

## Next Steps

1. **Run experiments** with new metrics to establish baselines
2. **Compare models** using NDCG@5 as primary metric
3. **Optimize chunk count** using context precision
4. **Implement LLM-as-judge** for production correctness evaluation
5. **Set up A/B testing** framework for production deployment

---

## Files Modified

- `mlflow/utils/evaluation.py` - Added all new metric functions
- `mlflow/utils/__init__.py` - Exported new functions
- `mlflow/chunking_experiments/run_experiments.py` - Added coherence, size stats, NDCG
- `mlflow/embedding_experiments/run_experiments.py` - Added NDCG, MAP, removed Recall@5
- `mlflow/retrieval_experiments/run_experiments.py` - Added Hit Rate, Diversity, NDCG
- `mlflow/llm_experiments/run_experiments.py` - Added Correctness, Completeness, Conciseness, Context Precision

