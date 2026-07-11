"""
Generate MLflow experiments with BASE METRICS ONLY (no noise).
This creates 1 run per strategy/model using exact base values.
"""


import mlflow

from experiments.mlflow.mlflow_config import configure_mlflow

configure_mlflow()

import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

# Import configurations from the main script
from experiments.mlflow.generate_fake_experiments import (
    CHUNKING_STRATEGIES,
    EMBEDDING_MODELS,
    LLM_MODELS,
    RETRIEVAL_STRATEGIES,
)


def generate_chunking_base():
    """Generate chunking experiments with base metrics only."""
    mlflow.set_experiment("Chunking_Strategy_Experiments_BASE")
    
    for strategy in CHUNKING_STRATEGIES:
        with mlflow.start_run(run_name=f"{strategy['name']}_base"):
            # Log parameters
            mlflow.log_param("strategy_name", strategy['name'])
            mlflow.log_param("strategy_type", strategy['strategy_type'])
            mlflow.log_param("chunk_size", strategy['chunk_size'])
            mlflow.log_param("chunk_overlap", strategy['chunk_overlap'])
            
            # Log BASE metrics (no noise)
            mlflow.log_metric("precision", strategy['base_precision'])
            mlflow.log_metric("recall", strategy['base_recall'])
            mlflow.log_metric("ndcg", strategy['base_ndcg'])
            mlflow.log_metric("mrr", strategy['base_mrr'])
            mlflow.log_metric("semantic_coherence", strategy['base_coherence'])
            mlflow.log_metric("quality", strategy['base_quality'])
            mlflow.log_metric("faithfulness", strategy['base_faithfulness'])
            mlflow.log_metric("avg_chunk_size_chars", strategy['avg_chunk_size'])
            
            mlflow.set_tag("experiment_type", "chunking")
            mlflow.set_tag("base_metrics", "true")
            
            logger.info(f"  ✓ {strategy['name']}: MRR={strategy['base_mrr']:.4f}")


def generate_retrieval_base():
    """Generate retrieval experiments with base metrics only."""
    mlflow.set_experiment("Retrieval_Strategy_Experiments_BASE")
    
    for strategy in RETRIEVAL_STRATEGIES:
        with mlflow.start_run(run_name=f"{strategy['name']}_base"):
            # Log parameters
            mlflow.log_param("strategy_name", strategy['name'])
            mlflow.log_param("alpha", strategy['alpha'])
            
            # Log BASE metrics (no noise)
            mlflow.log_metric("precision", strategy['base_precision'])
            mlflow.log_metric("recall", strategy['base_recall'])
            mlflow.log_metric("ndcg", strategy['base_ndcg'])
            mlflow.log_metric("mrr", strategy['base_mrr'])
            mlflow.log_metric("hit_rate", strategy['base_hit_rate'])
            mlflow.log_metric("diversity", strategy['base_diversity'])
            mlflow.log_metric("avg_latency_ms", strategy['base_latency_ms'])
            
            mlflow.set_tag("experiment_type", "retrieval")
            mlflow.set_tag("base_metrics", "true")
            
            logger.info(f"  ✓ {strategy['name']}: MRR={strategy['base_mrr']:.4f}")


def generate_embedding_base():
    """Generate embedding experiments with base metrics only."""
    mlflow.set_experiment("Embedding_Model_Experiments_BASE")
    
    for model in EMBEDDING_MODELS:
        with mlflow.start_run(run_name=f"{model['name'].split('/')[-1]}_base"):
            # Log parameters
            mlflow.log_param("model_name", model['name'])
            mlflow.log_param("embedding_dim", model['embedding_dim'])
            
            # Log BASE metrics (no noise)
            mlflow.log_metric("precision_at_5", model['base_precision'])
            mlflow.log_metric("ndcg_at_5", model['base_ndcg_5'])
            mlflow.log_metric("ndcg_at_10", model['base_ndcg_10'])
            mlflow.log_metric("map", model['base_map'])
            mlflow.log_metric("mrr", model['base_mrr'])
            mlflow.log_metric("throughput_sent_per_sec", model['base_throughput'])
            mlflow.log_metric("avg_query_latency_ms", model['base_query_latency_ms'])
            mlflow.log_metric("gpu_memory_mb", model['base_gpu_memory_mb'])
            mlflow.log_metric("model_load_time_sec", model['base_load_time_sec'])
            
            mlflow.set_tag("experiment_type", "embedding")
            mlflow.set_tag("base_metrics", "true")
            
            logger.info(f"  ✓ {model['name']}: NDCG@10={model['base_ndcg_10']:.4f}")


def generate_llm_base():
    """Generate LLM experiments with base metrics only."""
    mlflow.set_experiment("LLM_Model_Experiments_BASE")
    
    for model in LLM_MODELS:
        with mlflow.start_run(run_name=f"{model['name']}_base"):
            # Log parameters
            mlflow.log_param("model_name", model['name'])
            
            # Log BASE metrics (no noise)
            mlflow.log_metric("quality_score", model['base_quality'])
            mlflow.log_metric("faithfulness", model['base_faithfulness'])
            mlflow.log_metric("correctness", model['base_correctness'])
            mlflow.log_metric("completeness", model['base_completeness'])
            mlflow.log_metric("conciseness", model['base_conciseness'])
            mlflow.log_metric("context_precision", model['base_context_precision'])
            mlflow.log_metric("tokens_per_sec", model['base_tokens_per_sec'])
            mlflow.log_metric("latency_ms", model['base_latency_ms'])
            mlflow.log_metric("vram_usage_mb", model['base_vram_mb'])
            
            mlflow.set_tag("experiment_type", "llm")
            mlflow.set_tag("base_metrics", "true")
            
            logger.info(f"  ✓ {model['name']}: Quality={model['base_quality']:.4f}")


if __name__ == "__main__":
    logger.info("="*80)
    logger.info("GENERATING BASE METRICS EXPERIMENTS (NO NOISE)")
    logger.info("="*80)
    
    logger.info("\n🔹 Chunking Strategies:")
    generate_chunking_base()
    
    logger.info("\n🔹 Retrieval Strategies:")
    generate_retrieval_base()
    
    logger.info("\n🔹 Embedding Models:")
    generate_embedding_base()
    
    logger.info("\n🔹 LLM Models:")
    generate_llm_base()
    
    logger.info("\n" + "="*80)
    logger.info("✅ ALL BASE EXPERIMENTS GENERATED")
    logger.info("="*80)
