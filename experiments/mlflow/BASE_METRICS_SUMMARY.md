# 📊 Base Metrics Summary (No Noise)

This document shows the **exact base metrics** for all strategies and models without any random noise. These are the theoretical performance values used to generate the experiments.

---

## 🔹 1. Chunking Strategies

| Strategy | MRR | Precision | Recall | NDCG | Coherence | Quality | Faithfulness | Avg Size |
|----------|-----|-----------|--------|------|-----------|---------|--------------|----------|
| **Semantic Chunking** 🏆 | **0.7500** | 0.68 | 0.72 | 0.70 | **0.85** | **0.82** | **0.85** | 1200 |
| **Hierarchical** | **0.7200** | 0.62 | 0.70 | 0.66 | 0.80 | 0.80 | 0.83 | 1250 |
| Sentence Window | 0.6700 | 0.58 | 0.64 | 0.61 | 0.78 | 0.76 | 0.80 | 800 |
| Fixed Large | 0.6200 | 0.52 | 0.58 | 0.55 | 0.72 | 0.75 | 0.78 | 1500 |
| Fixed Baseline | 0.5500 | 0.45 | 0.52 | 0.48 | 0.65 | 0.70 | 0.72 | 1000 |

### 🏆 Winner: **Semantic Chunking**
- **Best MRR:** 0.75
- **Best Coherence:** 0.85 (maintains semantic meaning)
- **Best Quality:** 0.82
- **Best Faithfulness:** 0.85

### 🥈 Runner-up: **Hierarchical**
- **MRR:** 0.72 (close second)
- **Good balance** of context (parent chunks) and precision (child chunks)
- **Coherence:** 0.80

---

## 🔹 2. Retrieval Strategies

| Strategy | MRR | Precision | Recall | NDCG | Hit Rate | Diversity | Latency (ms) |
|----------|-----|-----------|--------|------|----------|-----------|--------------|
| **Hybrid Balanced (α=0.5)** 🏆 | **0.7800** | **0.70** | **0.75** | **0.72** | **0.85** | **0.70** | 65 |
| Hybrid Heavy BM25 (α=0.7) | 0.7400 | 0.65 | 0.72 | 0.68 | 0.82 | **0.72** | 75 |
| Hybrid Light BM25 (α=0.3) | 0.7200 | 0.62 | 0.68 | 0.65 | 0.80 | 0.68 | 55 |
| Pure Vector (α=0.0) | 0.6500 | 0.55 | 0.60 | 0.58 | 0.75 | 0.65 | **45** |

### 🏆 Winner: **Hybrid Balanced (α=0.5)**
- **Best MRR:** 0.78
- **Best across all quality metrics** (Precision, Recall, NDCG, Hit Rate)
- **Optimal balance:** 50% vector similarity + 50% BM25 keyword matching
- **Latency:** 65ms (acceptable trade-off for quality)

---

## 🔹 3. Embedding Models

| Model | NDCG@10 | NDCG@5 | Precision@5 | MAP | MRR | Latency (ms) | GPU Memory (MB) | Throughput (sent/s) |
|-------|---------|--------|-------------|-----|-----|--------------|-----------------|---------------------|
| **BAAI/bge-base-en-v1.5** 🏆 | **0.8200** | **0.7900** | **0.7500** | **0.7700** | **0.8300** | 30 | 480 | 1000 |
| multi-qa-mpnet-base-dot-v1 | 0.7800 | 0.7600 | 0.7200 | 0.7400 | 0.8000 | **26** | 430 | 1150 |
| all-mpnet-base-v2 | 0.7500 | 0.7200 | 0.6800 | 0.7000 | 0.7600 | 25 | 420 | 1200 |
| bge-small-en-v1.5 | 0.7200 | 0.6900 | 0.6500 | 0.6700 | 0.7300 | **18** | **280** | **2200** |
| all-MiniLM-L6-v2 | 0.6500 | 0.6200 | 0.5800 | 0.6000 | 0.6800 | 15 | 250 | 2500 |

### 🏆 Winner: **BAAI/bge-base-en-v1.5**
- **Best NDCG@10:** 0.82
- **Best across all quality metrics**
- **768 dimensions** for rich semantic representation
- **Trade-off:** Higher GPU memory (480MB) and moderate latency (30ms)

### 🥈 Runner-up: **multi-qa-mpnet-base-dot-v1**
- **NDCG@10:** 0.78 (close second)
- **Optimized for Q&A tasks** (better for RAG)
- **Faster:** 26ms latency
- **Good balance** of quality and performance

### 💡 Budget Option: **bge-small-en-v1.5**
- **NDCG@10:** 0.72 (still good)
- **Fastest:** 18ms latency
- **Lowest memory:** 280MB
- **Highest throughput:** 2200 sentences/sec

---

## 🔹 4. LLM Models

| Model | Quality | Faithfulness | Correctness | Completeness | Conciseness | Context Precision | Tokens/sec | Latency (ms) | VRAM (MB) |
|-------|---------|--------------|-------------|--------------|-------------|-------------------|------------|--------------|-----------|
| **llama3.2:1b** 🏆 | **0.7500** | 0.80 | **0.73** | **0.71** | 0.72 | **0.68** | 48 | 1150 | 1950 |
| qwen3:1.7b | 0.7200 | **0.78** | 0.70 | 0.68 | 0.75 | 0.65 | 45 | 1200 | 2100 |
| gemma3:1b | 0.6800 | 0.74 | 0.66 | 0.64 | **0.78** | 0.62 | **52** | **1050** | **1800** |

### 🏆 Winner: **llama3.2:1b (Meta)**
- **Best Overall Quality:** 0.75
- **Best Correctness:** 0.73
- **Best Completeness:** 0.71
- **Best Context Precision:** 0.68
- **Good Faithfulness:** 0.80
- **Balanced performance:** 48 tokens/sec, 1150ms latency

### 🥈 Runner-up: **qwen3:1.7b (Alibaba)**
- **Quality:** 0.72 (close second)
- **Best Faithfulness:** 0.78 (most accurate to source)
- **Slightly larger:** 1.7B parameters vs 1B
- **Trade-off:** Slightly slower and more VRAM

### 💡 Speed Option: **gemma3:1b (Google)**
- **Fastest:** 52 tokens/sec, 1050ms latency
- **Lowest VRAM:** 1800MB
- **Best Conciseness:** 0.78 (shortest answers)
- **Trade-off:** Lower quality (0.68)

---

## 🎯 Recommended Configuration (Based on Base Metrics)

### Option 1: **Maximum Quality** 🏆

```python
chunking = "semantic_chunking"          # MRR: 0.75, Coherence: 0.85
embedding = "BAAI/bge-base-en-v1.5"     # NDCG@10: 0.82
retrieval = "hybrid_balanced"            # MRR: 0.78, α=0.5
llm = "llama3.2:1b"                     # Quality: 0.75
```

**Expected Performance:**
- **Retrieval Quality:** NDCG@10 ≈ 0.82, MRR ≈ 0.78
- **Answer Quality:** 0.75
- **Total Latency:** ~30ms (embedding) + ~1150ms (LLM) = **~1.2 seconds**
- **Resource Usage:** 480MB (embedding) + 1950MB (LLM) = **~2.4GB VRAM**

---

### Option 2: **Balanced Performance** ⚖️

```python
chunking = "hierarchical"                        # MRR: 0.72
embedding = "multi-qa-mpnet-base-dot-v1"        # NDCG@10: 0.78
retrieval = "hybrid_balanced"                    # MRR: 0.78
llm = "llama3.2:1b"                             # Quality: 0.75
```

**Expected Performance:**
- **Retrieval Quality:** NDCG@10 ≈ 0.78, MRR ≈ 0.78
- **Answer Quality:** 0.75
- **Total Latency:** ~26ms (embedding) + ~1150ms (LLM) = **~1.18 seconds**
- **Resource Usage:** 430MB (embedding) + 1950MB (LLM) = **~2.4GB VRAM**

---

### Option 3: **Speed/Budget** ⚡

```python
chunking = "sentence_window"             # MRR: 0.67
embedding = "bge-small-en-v1.5"         # NDCG@10: 0.72
retrieval = "hybrid_light_bm25"          # MRR: 0.72, α=0.3
llm = "gemma3:1b"                       # Quality: 0.68
```

**Expected Performance:**
- **Retrieval Quality:** NDCG@10 ≈ 0.72, MRR ≈ 0.72
- **Answer Quality:** 0.68
- **Total Latency:** ~18ms (embedding) + ~1050ms (LLM) = **~1.07 seconds**
- **Resource Usage:** 280MB (embedding) + 1800MB (LLM) = **~2.1GB VRAM**

---

## 📊 Key Insights

1. **Semantic Chunking** provides the best retrieval quality (MRR: 0.75) and coherence (0.85)
2. **Hybrid Balanced retrieval** (50/50 vector+BM25) consistently outperforms pure vector search
3. **BAAI/bge-base-en-v1.5** has the highest retrieval quality but **multi-qa-mpnet** is optimized for Q&A
4. **llama3.2:1b** offers the best overall answer quality among small LLMs
5. **Quality vs Speed trade-off:** ~10% quality reduction can give you ~15% speed improvement

---

**Note:** These are theoretical base values. Actual performance will vary based on:
- Your specific dataset and domain
- Query complexity
- Hardware specifications
- Batch sizes and optimization settings

