"""
Create visualizations of experiment results.
"""

import mlflow
from experiments.mlflow.mlflow_config import EVALUATION_DIR, configure_mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
configure_mlflow()
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def get_experiment_data(experiment_name, group_col_name):
    """Get aggregated data from an experiment."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        return None
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if runs.empty:
        return None
    
    # Determine grouping column
    if f'params.{group_col_name}' in runs.columns:
        group_col = f'params.{group_col_name}'
    else:
        return None
    
    # Get metric columns
    metric_cols = [col for col in runs.columns if col.startswith('metrics.')]
    
    # Group and aggregate
    grouped = runs.groupby(group_col)[metric_cols].mean()
    grouped.columns = [col.replace('metrics.', '') for col in grouped.columns]
    
    return grouped


# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Chunking Strategies
ax1 = plt.subplot(2, 2, 1)
chunking_data = get_experiment_data("Chunking_Strategy_Experiments", "strategy_name")
if chunking_data is not None and 'mrr' in chunking_data.columns:
    chunking_data['mrr'].sort_values().plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_title('Chunking Strategies - MRR Score', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mean Reciprocal Rank (MRR)', fontsize=11)
    ax1.set_ylabel('')
    ax1.grid(axis='x', alpha=0.3)
    for i, v in enumerate(chunking_data['mrr'].sort_values()):
        ax1.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

# 2. Retrieval Strategies
ax2 = plt.subplot(2, 2, 2)
retrieval_data = get_experiment_data("Retrieval_Strategy_Experiments", "strategy_name")
if retrieval_data is not None and 'mrr' in retrieval_data.columns:
    retrieval_data['mrr'].sort_values().plot(kind='barh', ax=ax2, color='coral')
    ax2.set_title('Retrieval Strategies - MRR Score', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Mean Reciprocal Rank (MRR)', fontsize=11)
    ax2.set_ylabel('')
    ax2.grid(axis='x', alpha=0.3)
    for i, v in enumerate(retrieval_data['mrr'].sort_values()):
        ax2.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

# 3. Embedding Models
ax3 = plt.subplot(2, 2, 3)
embedding_data = get_experiment_data("Embedding_Model_Experiments", "model_name")
if embedding_data is not None and 'ndcg_at_10' in embedding_data.columns:
    # Shorten model names for display
    embedding_data.index = [name.split('/')[-1] if '/' in name else name 
                            for name in embedding_data.index]
    embedding_data['ndcg_at_10'].sort_values().plot(kind='barh', ax=ax3, color='mediumseagreen')
    ax3.set_title('Embedding Models - NDCG@10 Score', fontsize=14, fontweight='bold')
    ax3.set_xlabel('NDCG@10', fontsize=11)
    ax3.set_ylabel('')
    ax3.grid(axis='x', alpha=0.3)
    for i, v in enumerate(embedding_data['ndcg_at_10'].sort_values()):
        ax3.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

# 4. LLM Models
ax4 = plt.subplot(2, 2, 4)
llm_data = get_experiment_data("LLM_Model_Experiments", "model_name")
if llm_data is not None and 'quality_score' in llm_data.columns:
    llm_data['quality_score'].sort_values().plot(kind='barh', ax=ax4, color='mediumpurple')
    ax4.set_title('LLM Models - Quality Score', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Quality Score', fontsize=11)
    ax4.set_ylabel('')
    ax4.grid(axis='x', alpha=0.3)
    for i, v in enumerate(llm_data['quality_score'].sort_values()):
        ax4.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

plt.suptitle('MLflow Experiment Results Summary', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

# Save figure
output_path = 'experiment_results_summary.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Visualization saved to: {output_path}")

# Create detailed comparison for embedding models
fig2, axes = plt.subplots(2, 2, figsize=(14, 10))

embedding_data_full = get_experiment_data("Embedding_Model_Experiments", "model_name")
if embedding_data_full is not None:
    # Shorten names
    embedding_data_full.index = [name.split('/')[-1] if '/' in name else name 
                                  for name in embedding_data_full.index]
    
    # Quality metrics
    if all(col in embedding_data_full.columns for col in ['ndcg_at_10', 'precision_at_5', 'map', 'mrr']):
        quality_metrics = embedding_data_full[['ndcg_at_10', 'precision_at_5', 'map', 'mrr']]
        quality_metrics.plot(kind='bar', ax=axes[0, 0], width=0.8)
        axes[0, 0].set_title('Embedding Models - Quality Metrics', fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(loc='lower right', fontsize=9)
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Performance metrics
    if all(col in embedding_data_full.columns for col in ['avg_query_latency_ms', 'gpu_memory_mb']):
        ax_latency = axes[0, 1]
        ax_memory = ax_latency.twinx()
        
        x = range(len(embedding_data_full))
        ax_latency.bar([i - 0.2 for i in x], embedding_data_full['avg_query_latency_ms'], 
                       width=0.4, label='Latency (ms)', color='coral', alpha=0.7)
        ax_memory.bar([i + 0.2 for i in x], embedding_data_full['gpu_memory_mb'], 
                      width=0.4, label='GPU Memory (MB)', color='steelblue', alpha=0.7)
        
        ax_latency.set_title('Embedding Models - Performance', fontweight='bold')
        ax_latency.set_ylabel('Latency (ms)', color='coral')
        ax_memory.set_ylabel('GPU Memory (MB)', color='steelblue')
        ax_latency.set_xticks(x)
        ax_latency.set_xticklabels(embedding_data_full.index, rotation=45, ha='right')
        ax_latency.legend(loc='upper left', fontsize=9)
        ax_memory.legend(loc='upper right', fontsize=9)
        ax_latency.grid(axis='y', alpha=0.3)

# LLM detailed comparison
llm_data_full = get_experiment_data("LLM_Model_Experiments", "model_name")
if llm_data_full is not None:
    # Quality metrics
    if all(col in llm_data_full.columns for col in ['quality_score', 'faithfulness', 'correctness', 'completeness']):
        quality_metrics = llm_data_full[['quality_score', 'faithfulness', 'correctness', 'completeness']]
        quality_metrics.plot(kind='bar', ax=axes[1, 0], width=0.8)
        axes[1, 0].set_title('LLM Models - Quality Metrics', fontweight='bold')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend(loc='lower right', fontsize=9)
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # Performance metrics
    if all(col in llm_data_full.columns for col in ['tokens_per_sec', 'latency_ms']):
        ax_tokens = axes[1, 1]
        ax_latency_llm = ax_tokens.twinx()
        
        x = range(len(llm_data_full))
        ax_tokens.bar([i - 0.2 for i in x], llm_data_full['tokens_per_sec'], 
                      width=0.4, label='Tokens/sec', color='mediumseagreen', alpha=0.7)
        ax_latency_llm.bar([i + 0.2 for i in x], llm_data_full['latency_ms'], 
                           width=0.4, label='Latency (ms)', color='mediumpurple', alpha=0.7)
        
        ax_tokens.set_title('LLM Models - Performance', fontweight='bold')
        ax_tokens.set_ylabel('Tokens/sec', color='mediumseagreen')
        ax_latency_llm.set_ylabel('Latency (ms)', color='mediumpurple')
        ax_tokens.set_xticks(x)
        ax_tokens.set_xticklabels(llm_data_full.index, rotation=45, ha='right')
        ax_tokens.legend(loc='upper left', fontsize=9)
        ax_latency_llm.legend(loc='upper right', fontsize=9)
        ax_tokens.grid(axis='y', alpha=0.3)

plt.suptitle('Detailed Model Comparisons', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

output_path2 = 'detailed_model_comparison.png'
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"✅ Detailed comparison saved to: {output_path2}")

print("\n✅ All visualizations created successfully!")

