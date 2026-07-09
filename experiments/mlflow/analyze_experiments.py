"""
Analyze MLflow experiments and provide recommendations.
"""

import mlflow
from experiments.mlflow.mlflow_config import EVALUATION_DIR, configure_mlflow
import pandas as pd
import numpy as np

# Setup MLflow
configure_mlflow()

def analyze_experiment(experiment_name, key_metrics):
    """Analyze an experiment and return summary statistics."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"❌ Experiment '{experiment_name}' not found")
        return None
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        print(f"❌ No runs found for '{experiment_name}'")
        return None
    
    print(f"\n{'='*80}")
    print(f"📊 {experiment_name}")
    print(f"{'='*80}")
    print(f"Total runs: {len(runs)}")
    
    # Group by strategy/model
    if 'params.strategy_name' in runs.columns:
        group_col = 'params.strategy_name'
    elif 'params.model_name' in runs.columns:
        group_col = 'params.model_name'
    else:
        print("❌ No grouping column found")
        return None
    
    # Calculate average metrics per strategy/model
    results = []
    for name, group in runs.groupby(group_col):
        result = {'name': name, 'runs': len(group)}
        for metric in key_metrics:
            metric_col = f'metrics.{metric}'
            if metric_col in group.columns:
                result[metric] = group[metric_col].mean()
                result[f'{metric}_std'] = group[metric_col].std()
        results.append(result)
    
    df = pd.DataFrame(results)
    return df


def print_recommendations(df, experiment_type, primary_metric, secondary_metrics):
    """Print recommendations based on analysis."""
    if df is None or df.empty:
        return None

    print(f"\n📈 Average Metrics by {experiment_type}:")
    print("-" * 80)

    # Display table - show all available metrics
    available_metrics = [m for m in [primary_metric] + secondary_metrics if m in df.columns]
    if not available_metrics:
        print("⚠️  No metrics found in columns")
        print(f"Available columns: {list(df.columns)}")
        return None

    display_cols = ['name', 'runs'] + available_metrics
    print(df[display_cols].to_string(index=False, float_format='%.4f'))

    # Find best option
    if primary_metric in df.columns:
        best_idx = df[primary_metric].idxmax()
        best = df.loc[best_idx]

        print(f"\n🏆 RECOMMENDED: {best['name']}")
        print("-" * 80)
        print(f"Primary metric ({primary_metric}): {best[primary_metric]:.4f}")
        for metric in secondary_metrics:
            if metric in df.columns:
                print(f"  - {metric}: {best[metric]:.4f}")

        return best['name']

    return None


# Analyze Chunking Experiments
print("\n" + "="*80)
print("🔍 CHUNKING STRATEGY ANALYSIS")
print("="*80)

chunking_df = analyze_experiment(
    "Chunking_Strategy_Experiments",
    ['precision', 'recall', 'ndcg', 'mrr', 'semantic_coherence', 'avg_processing_time_ms',
     'avg_chunk_size_chars', 'total_chunks']
)

best_chunking = None
if chunking_df is not None:
    best_chunking = print_recommendations(
        chunking_df,
        "Chunking Strategy",
        primary_metric='mrr',
        secondary_metrics=['precision', 'recall', 'ndcg', 'semantic_coherence', 'avg_chunk_size_chars']
    )


# Analyze Retrieval Experiments
print("\n" + "="*80)
print("🔍 RETRIEVAL STRATEGY ANALYSIS")
print("="*80)

retrieval_df = analyze_experiment(
    "Retrieval_Strategy_Experiments",
    ['precision', 'recall', 'ndcg', 'mrr', 'hit_rate', 'diversity', 'avg_latency_ms']
)

best_retrieval = None
if retrieval_df is not None:
    best_retrieval = print_recommendations(
        retrieval_df,
        "Retrieval Strategy",
        primary_metric='mrr',
        secondary_metrics=['precision', 'recall', 'ndcg', 'hit_rate', 'avg_latency_ms']
    )


# Analyze Embedding Experiments
print("\n" + "="*80)
print("🔍 EMBEDDING MODEL ANALYSIS")
print("="*80)

embedding_df = analyze_experiment(
    "Embedding_Model_Experiments",
    ['precision_at_5', 'ndcg_at_5', 'ndcg_at_10', 'map', 'mrr',
     'avg_query_latency_ms', 'throughput_sent_per_sec', 'gpu_memory_mb']
)

best_embedding = None
if embedding_df is not None:
    best_embedding = print_recommendations(
        embedding_df,
        "Embedding Model",
        primary_metric='ndcg_at_10',
        secondary_metrics=['precision_at_5', 'map', 'mrr', 'avg_query_latency_ms', 'gpu_memory_mb']
    )


# Analyze LLM Experiments
print("\n" + "="*80)
print("🔍 LLM MODEL ANALYSIS")
print("="*80)

llm_df = analyze_experiment(
    "LLM_Model_Experiments",
    ['quality_score', 'faithfulness', 'correctness', 'completeness',
     'conciseness', 'context_precision', 'tokens_per_sec', 'latency_ms', 'vram_usage_mb']
)

best_llm = None
if llm_df is not None:
    best_llm = print_recommendations(
        llm_df,
        "LLM Model",
        primary_metric='quality_score',
        secondary_metrics=['faithfulness', 'correctness', 'completeness', 'tokens_per_sec', 'latency_ms']
    )


# Final Summary
print("\n" + "="*80)
print("🎯 FINAL RECOMMENDATIONS SUMMARY")
print("="*80)
print("\nBased on the analysis of all experiments, here are the best options:\n")

if best_chunking:
    print(f"✅ Best Chunking Strategy: {best_chunking}")
    print(f"   → Provides optimal balance of retrieval quality and semantic coherence")

if best_retrieval:
    print(f"\n✅ Best Retrieval Strategy: {best_retrieval}")
    print(f"   → Achieves highest retrieval accuracy with acceptable latency")

if best_embedding:
    print(f"\n✅ Best Embedding Model: {best_embedding}")
    print(f"   → Best retrieval quality with reasonable resource usage")

if best_llm:
    print(f"\n✅ Best LLM Model: {best_llm}")
    print(f"   → Highest quality answers with good performance")

print("\n" + "="*80)
print("💡 RECOMMENDED RAG PIPELINE CONFIGURATION")
print("="*80)

if best_chunking and best_retrieval and best_embedding and best_llm:
    print(f"""
1. Chunking: {best_chunking}
2. Embedding: {best_embedding}
3. Retrieval: {best_retrieval}
4. LLM: {best_llm}

This configuration should provide the best overall performance for your RAG system.
""")
else:
    print("\n⚠️  Some experiments are missing. Run all experiments for complete recommendations.")

print("="*80)

