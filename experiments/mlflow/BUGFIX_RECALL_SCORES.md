# Bug Fix: recall_at_5_scores NameError

## Issue

When running the embedding experiments, the following error occurred:

```
NameError: name 'recall_at_5_scores' is not defined
```

**Location:** `mlflow/embedding_experiments/run_experiments.py`, line 265

## Root Cause

When we removed Recall@5 from the primary metrics in the embedding experiments (because it was misleading with 60+ relevant documents), we:

1. ✅ Removed the computation of `recall_at_5_scores` in the evaluation loop
2. ✅ Removed it from the logged metrics
3. ❌ **FORGOT** to remove it from the artifacts dictionary

The code was still trying to include `recall_at_5_scores` in the detailed results artifacts, but the variable was never defined.

## Fix Applied

**File:** `mlflow/embedding_experiments/run_experiments.py`

**Before (lines 261-270):**
```python
artifacts = {
    "detailed_results": {
        "per_query_precision_at_5": precision_at_5_scores,
        "per_query_recall_at_5": recall_at_5_scores,  # ❌ NOT DEFINED
        "per_query_mrr": mrr_scores,
        "per_query_latency_ms": query_latencies
    },
    "sample_embeddings": query_embeddings[:5]
}
```

**After (lines 261-272):**
```python
artifacts = {
    "detailed_results": {
        "per_query_precision_at_5": precision_at_5_scores,
        "per_query_ndcg_at_5": ndcg_at_5_scores,      # ✅ ADDED
        "per_query_ndcg_at_10": ndcg_at_10_scores,    # ✅ ADDED
        "per_query_map": map_scores,                   # ✅ ADDED
        "per_query_mrr": mrr_scores,
        "per_query_latency_ms": query_latencies
    },
    "sample_embeddings": query_embeddings[:5]
}
```

## Changes Made

1. ❌ **Removed:** `per_query_recall_at_5` (not computed)
2. ✅ **Added:** `per_query_ndcg_at_5` (our new primary metric)
3. ✅ **Added:** `per_query_ndcg_at_10` (extended ranking quality)
4. ✅ **Added:** `per_query_map` (comprehensive precision)

## Why This Happened

This is a classic refactoring bug - when removing a feature, we need to check:
- ✅ Where it's computed
- ✅ Where it's logged
- ❌ **Where it's used in artifacts** ← We missed this!

## Verification

After the fix, the code now:
1. ✅ Computes only the metrics we want (NDCG, Precision, MAP, MRR)
2. ✅ Logs only those metrics to MLflow
3. ✅ Includes only those metrics in artifacts
4. ✅ No undefined variables

## Impact

**Before Fix:**
- ❌ Embedding experiments crashed with NameError
- ❌ No results logged to MLflow

**After Fix:**
- ✅ Embedding experiments run successfully
- ✅ All new metrics (NDCG@5, NDCG@10, MAP) are logged
- ✅ Detailed per-query results are saved as artifacts

## Note on Other Experiments

**Retrieval Experiments** still use `recall_at_5_scores` - this is **intentional** and **correct**:
- Retrieval experiments have fewer relevant docs per query
- Recall@5 is meaningful in that context
- No changes needed for retrieval experiments

## Testing

Verified with:
```bash
python mlflow/embedding_experiments/run_experiments.py
```

**Result:** ✅ All experiments run successfully with new metrics!

---

**Status:** ✅ FIXED
**Date:** 2026-01-18
**Files Modified:** `mlflow/embedding_experiments/run_experiments.py`

