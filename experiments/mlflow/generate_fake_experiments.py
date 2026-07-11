"""
Generate Fake MLflow Experiments with Logical Metrics

Creates realistic fake experiments for both chunking and retrieval strategies
with logically consistent metrics that follow expected patterns.

Usage:
    python mlflow/generate_fake_experiments.py --chunking --retrieval --runs 10
"""

import argparse
import random
from datetime import datetime, timedelta

import numpy as np

# Add paths
# Configure MLflow database backend
from experiments.mlflow.mlflow_config import configure_mlflow

configure_mlflow()

import logging

import mlflow

from experiments.mlflow.utils import log_dict_as_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CHUNKING STRATEGIES CONFIGURATION
# ============================================================================

CHUNKING_STRATEGIES = [
    {
        "name": "fixed_character_baseline",
        "description": "Fixed 1000 chars, 200 overlap (baseline)",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "strategy_type": "fixed_character",
        # Expected performance characteristics
        "base_precision": 0.45,
        "base_recall": 0.52,
        "base_ndcg": 0.48,
        "base_mrr": 0.55,
        "base_coherence": 0.65,
        "base_quality": 0.70,
        "base_faithfulness": 0.72,
        "avg_chunk_size": 1000,
        "chunk_size_std": 50,
    },
    {
        "name": "fixed_character_large",
        "description": "Fixed 1500 chars, 300 overlap",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "strategy_type": "fixed_character",
        "base_precision": 0.52,
        "base_recall": 0.58,
        "base_ndcg": 0.55,
        "base_mrr": 0.62,
        "base_coherence": 0.72,
        "base_quality": 0.75,
        "base_faithfulness": 0.78,
        "avg_chunk_size": 1500,
        "chunk_size_std": 60,
    },
    {
        "name": "semantic_chunking",
        "description": "Variable size, semantic breaks",
        "chunk_size": 1000,
        "chunk_overlap": 0.1,
        "strategy_type": "semantic",
        "base_precision": 0.68,
        "base_recall": 0.72,
        "base_ndcg": 0.70,
        "base_mrr": 0.75,
        "base_coherence": 0.85,
        "base_quality": 0.82,
        "base_faithfulness": 0.85,
        "avg_chunk_size": 1200,
        "chunk_size_std": 350,
    },
    {
        "name": "sentence_window",
        "description": "10 sentences, 2 sentence overlap",
        "chunk_size": 10,
        "chunk_overlap": 2,
        "strategy_type": "sentence_window",
        "base_precision": 0.58,
        "base_recall": 0.64,
        "base_ndcg": 0.61,
        "base_mrr": 0.67,
        "base_coherence": 0.78,
        "base_quality": 0.76,
        "base_faithfulness": 0.80,
        "avg_chunk_size": 800,
        "chunk_size_std": 200,
    },
    {
        "name": "hierarchical",
        "description": "Parent (2000) + Child (500) chunks",
        "chunk_size": 2000,
        "child_size": 500,
        "chunk_overlap": 0,
        "strategy_type": "hierarchical",
        "base_precision": 0.62,
        "base_recall": 0.70,
        "base_ndcg": 0.66,
        "base_mrr": 0.72,
        "base_coherence": 0.80,
        "base_quality": 0.80,
        "base_faithfulness": 0.83,
        "avg_chunk_size": 1250,
        "chunk_size_std": 600,
    },
]


# ============================================================================
# RETRIEVAL STRATEGIES CONFIGURATION
# ============================================================================

RETRIEVAL_STRATEGIES = [
    {
        "name": "pure_vector",
        "description": "Pure vector search using ChromaDB",
        "alpha": 0.0,
        "base_precision": 0.55,
        "base_recall": 0.60,
        "base_ndcg": 0.58,
        "base_mrr": 0.65,
        "base_hit_rate": 0.75,
        "base_diversity": 0.65,
        "base_latency_ms": 45,
    },
    {
        "name": "hybrid_light_bm25",
        "description": "Hybrid with 30% BM25 weight",
        "alpha": 0.3,
        "base_precision": 0.62,
        "base_recall": 0.68,
        "base_ndcg": 0.65,
        "base_mrr": 0.72,
        "base_hit_rate": 0.80,
        "base_diversity": 0.68,
        "base_latency_ms": 55,
    },
    {
        "name": "hybrid_balanced",
        "description": "Hybrid with 50% BM25 weight",
        "alpha": 0.5,
        "base_precision": 0.70,
        "base_recall": 0.75,
        "base_ndcg": 0.72,
        "base_mrr": 0.78,
        "base_hit_rate": 0.85,
        "base_diversity": 0.70,
        "base_latency_ms": 65,
    },
    {
        "name": "hybrid_heavy_bm25",
        "description": "Hybrid with 70% BM25 weight",
        "alpha": 0.7,
        "base_precision": 0.65,
        "base_recall": 0.72,
        "base_ndcg": 0.68,
        "base_mrr": 0.74,
        "base_hit_rate": 0.82,
        "base_diversity": 0.72,
        "base_latency_ms": 75,
    },
]


# ============================================================================
# EMBEDDING MODELS CONFIGURATION
# ============================================================================

EMBEDDING_MODELS = [
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "description": "Lightweight and fast",
        "base_precision": 0.58,
        "base_ndcg_5": 0.62,
        "base_ndcg_10": 0.65,
        "base_map": 0.60,
        "base_mrr": 0.68,
        "base_throughput": 2500,  # sentences/sec
        "base_query_latency_ms": 15,
        "base_gpu_memory_mb": 250,
        "base_load_time_sec": 2.5,
    },
    {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "embedding_dim": 768,
        "description": "Balanced performance",
        "base_precision": 0.68,
        "base_ndcg_5": 0.72,
        "base_ndcg_10": 0.75,
        "base_map": 0.70,
        "base_mrr": 0.76,
        "base_throughput": 1200,
        "base_query_latency_ms": 25,
        "base_gpu_memory_mb": 420,
        "base_load_time_sec": 3.2,
    },
    {
        "name": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "embedding_dim": 768,
        "description": "Optimized for QA retrieval",
        "base_precision": 0.72,
        "base_ndcg_5": 0.76,
        "base_ndcg_10": 0.78,
        "base_map": 0.74,
        "base_mrr": 0.80,
        "base_throughput": 1150,
        "base_query_latency_ms": 26,
        "base_gpu_memory_mb": 430,
        "base_load_time_sec": 3.3,
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "description": "BGE small model",
        "base_precision": 0.65,
        "base_ndcg_5": 0.69,
        "base_ndcg_10": 0.72,
        "base_map": 0.67,
        "base_mrr": 0.73,
        "base_throughput": 2200,
        "base_query_latency_ms": 18,
        "base_gpu_memory_mb": 280,
        "base_load_time_sec": 2.8,
    },
    {
        "name": "BAAI/bge-base-en-v1.5",
        "embedding_dim": 768,
        "description": "BGE base model",
        "base_precision": 0.75,
        "base_ndcg_5": 0.79,
        "base_ndcg_10": 0.82,
        "base_map": 0.77,
        "base_mrr": 0.83,
        "base_throughput": 1000,
        "base_query_latency_ms": 30,
        "base_gpu_memory_mb": 480,
        "base_load_time_sec": 3.5,
    },
]


# ============================================================================
# LLM MODELS CONFIGURATION
# ============================================================================

LLM_MODELS = [
    {
        "name": "qwen3:1.7b",
        "description": "Qwen 3 1.7B - Alibaba's efficient LLM",
        "base_tokens_per_sec": 45,
        "base_latency_ms": 1200,
        "base_quality": 0.72,
        "base_faithfulness": 0.78,
        "base_correctness": 0.70,
        "base_completeness": 0.68,
        "base_conciseness": 0.75,
        "base_context_precision": 0.65,
        "base_vram_mb": 2100,
    },
    {
        "name": "gemma3:1b",
        "description": "Gemma 3 1B - Google's compact model",
        "base_tokens_per_sec": 52,
        "base_latency_ms": 1050,
        "base_quality": 0.68,
        "base_faithfulness": 0.74,
        "base_correctness": 0.66,
        "base_completeness": 0.64,
        "base_conciseness": 0.78,
        "base_context_precision": 0.62,
        "base_vram_mb": 1800,
    },
    {
        "name": "llama3.2:1b",
        "description": "Llama 3.2 1B - Meta's latest small model",
        "base_tokens_per_sec": 48,
        "base_latency_ms": 1150,
        "base_quality": 0.75,
        "base_faithfulness": 0.80,
        "base_correctness": 0.73,
        "base_completeness": 0.71,
        "base_conciseness": 0.72,
        "base_context_precision": 0.68,
        "base_vram_mb": 1950,
    },
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def add_noise(value: float, noise_level: float = 0.05) -> float:
    """Add random noise to a metric value."""
    noise = np.random.normal(0, noise_level * value)
    return max(0.0, min(1.0, value + noise))


def generate_chunk_sizes(avg_size: int, std: int, num_chunks: int) -> list:
    """Generate realistic chunk sizes."""
    sizes = np.random.normal(avg_size, std, num_chunks)
    sizes = np.clip(sizes, avg_size * 0.3, avg_size * 2.5)  # Reasonable bounds
    return sizes.astype(int).tolist()


def generate_chunking_metrics(strategy: dict, num_queries: int = 10) -> dict:
    """
    Generate realistic metrics for a chunking strategy.

    Args:
        strategy: Strategy configuration with base metrics
        num_queries: Number of queries evaluated

    Returns:
        Dictionary of metrics with realistic values and variations
    """
    # Add noise to base metrics
    precision = add_noise(strategy['base_precision'], 0.08)
    recall = add_noise(strategy['base_recall'], 0.08)
    ndcg = add_noise(strategy['base_ndcg'], 0.08)
    mrr = add_noise(strategy['base_mrr'], 0.08)
    coherence = add_noise(strategy['base_coherence'], 0.05)
    quality = add_noise(strategy['base_quality'], 0.06)
    faithfulness = add_noise(strategy['base_faithfulness'], 0.06)

    # Generate chunk statistics
    num_chunks = random.randint(18000, 22000)
    chunk_sizes = generate_chunk_sizes(
        strategy['avg_chunk_size'],
        strategy['chunk_size_std'],
        num_chunks
    )

    # Token estimation (roughly 4 chars per token)
    chunk_tokens = [size // 4 for size in chunk_sizes]

    metrics = {
        # Retrieval quality metrics
        "precision_at_5": precision,
        "precision_at_5_std": abs(np.random.normal(0, 0.05)),
        "ndcg_at_5": ndcg,
        "ndcg_at_5_std": abs(np.random.normal(0, 0.05)),
        "recall_at_5": recall,
        "recall_at_5_std": abs(np.random.normal(0, 0.05)),
        "mrr": mrr,
        "mrr_std": abs(np.random.normal(0, 0.04)),

        # Chunk statistics
        "total_chunks": num_chunks,
        "avg_chunk_size_chars": float(np.mean(chunk_sizes)),
        "median_chunk_size_chars": float(np.median(chunk_sizes)),
        "std_chunk_size_chars": float(np.std(chunk_sizes)),
        "min_chunk_size_chars": int(np.min(chunk_sizes)),
        "max_chunk_size_chars": int(np.max(chunk_sizes)),
        "avg_chunk_size_tokens": float(np.mean(chunk_tokens)),
        "median_chunk_size_tokens": float(np.median(chunk_tokens)),
        "std_chunk_size_tokens": float(np.std(chunk_tokens)),
        "min_chunk_size_tokens": int(np.min(chunk_tokens)),
        "max_chunk_size_tokens": int(np.max(chunk_tokens)),

        # Chunk quality metrics
        "chunk_coherence": coherence,

        # Answer quality metrics
        "answer_quality": quality,
        "answer_quality_std": abs(np.random.normal(0, 0.05)),
        "faithfulness": faithfulness,
        "faithfulness_std": abs(np.random.normal(0, 0.05)),

        "num_queries_evaluated": num_queries
    }

    return metrics


def generate_retrieval_metrics(strategy: dict, num_queries: int = 2000) -> dict:
    """
    Generate realistic metrics for a retrieval strategy.

    Args:
        strategy: Strategy configuration with base metrics
        num_queries: Number of queries evaluated

    Returns:
        Dictionary of metrics with realistic values and variations
    """
    # Add noise to base metrics
    precision = add_noise(strategy['base_precision'], 0.08)
    recall = add_noise(strategy['base_recall'], 0.08)
    ndcg = add_noise(strategy['base_ndcg'], 0.08)
    mrr = add_noise(strategy['base_mrr'], 0.08)
    hit_rate = add_noise(strategy['base_hit_rate'], 0.06)
    diversity = add_noise(strategy['base_diversity'], 0.05)

    # Latency with some variation
    latency = strategy['base_latency_ms'] + np.random.normal(0, 5)
    latency = max(10, latency)  # Minimum 10ms

    metrics = {
        # Retrieval quality metrics
        "precision_at_5": precision,
        "precision_at_5_std": abs(np.random.normal(0, 0.05)),
        "ndcg_at_5": ndcg,
        "ndcg_at_5_std": abs(np.random.normal(0, 0.05)),
        "hit_rate_at_5": hit_rate,
        "hit_rate_at_5_std": abs(np.random.normal(0, 0.04)),
        "diversity": diversity,
        "diversity_std": abs(np.random.normal(0, 0.04)),
        "recall_at_5": recall,
        "recall_at_5_std": abs(np.random.normal(0, 0.05)),
        "mrr": mrr,
        "mrr_std": abs(np.random.normal(0, 0.04)),

        # Performance metrics
        "latency_ms": latency,
        "latency_std_ms": abs(np.random.normal(0, 3)),

        # Dataset info
        "num_queries": num_queries
    }

    return metrics


def generate_embedding_metrics(model: dict, num_queries: int = 2000, num_chunks: int = 20000) -> dict:
    """
    Generate realistic metrics for an embedding model.

    Args:
        model: Model configuration with base metrics
        num_queries: Number of queries evaluated
        num_chunks: Number of chunks in the dataset

    Returns:
        Dictionary of metrics with realistic values and variations
    """
    # Add noise to base metrics
    precision = add_noise(model['base_precision'], 0.08)
    ndcg_5 = add_noise(model['base_ndcg_5'], 0.08)
    ndcg_10 = add_noise(model['base_ndcg_10'], 0.08)
    map_score = add_noise(model['base_map'], 0.08)
    mrr = add_noise(model['base_mrr'], 0.08)

    # Performance metrics with variation
    throughput = model['base_throughput'] + np.random.normal(0, model['base_throughput'] * 0.1)
    query_latency = model['base_query_latency_ms'] + np.random.normal(0, 2)
    gpu_memory = model['base_gpu_memory_mb'] + np.random.normal(0, 20)
    load_time = model['base_load_time_sec'] + np.random.normal(0, 0.3)

    # Ensure positive values
    throughput = max(100, throughput)
    query_latency = max(5, query_latency)
    gpu_memory = max(100, gpu_memory)
    load_time = max(1, load_time)

    metrics = {
        # Performance metrics
        "throughput_sent_per_sec": throughput,
        "avg_query_latency_ms": query_latency,
        "gpu_memory_mb": gpu_memory,
        "model_load_time_sec": load_time,
        "embedding_dim": model["embedding_dim"],

        # Retrieval quality metrics
        "precision_at_5": precision,
        "precision_at_5_std": abs(np.random.normal(0, 0.05)),
        "ndcg_at_5": ndcg_5,
        "ndcg_at_5_std": abs(np.random.normal(0, 0.05)),
        "ndcg_at_10": ndcg_10,
        "ndcg_at_10_std": abs(np.random.normal(0, 0.05)),
        "map": map_score,
        "map_std": abs(np.random.normal(0, 0.05)),
        "mrr": mrr,
        "mrr_std": abs(np.random.normal(0, 0.04)),

        # Dataset info
        "num_queries": num_queries,
        "num_chunks": num_chunks
    }

    return metrics


def generate_llm_metrics(model: dict, num_queries: int = 10) -> dict:
    """
    Generate realistic metrics for an LLM model.

    Args:
        model: Model configuration with base metrics
        num_queries: Number of queries evaluated

    Returns:
        Dictionary of metrics with realistic values and variations
    """
    # Add noise to base metrics
    tokens_per_sec = model['base_tokens_per_sec'] + np.random.normal(0, 3)
    latency_ms = model['base_latency_ms'] + np.random.normal(0, 100)
    quality = add_noise(model['base_quality'], 0.08)
    faithfulness = add_noise(model['base_faithfulness'], 0.08)
    correctness = add_noise(model['base_correctness'], 0.08)
    completeness = add_noise(model['base_completeness'], 0.08)
    conciseness = add_noise(model['base_conciseness'], 0.08)
    context_precision = add_noise(model['base_context_precision'], 0.08)
    vram_mb = model['base_vram_mb'] + np.random.normal(0, 100)

    # Ensure positive values
    tokens_per_sec = max(10, tokens_per_sec)
    latency_ms = max(500, latency_ms)
    vram_mb = max(1000, vram_mb)

    metrics = {
        # Performance metrics
        "tokens_per_sec": tokens_per_sec,
        "tokens_per_sec_std": abs(np.random.normal(0, 2)),
        "latency_sec": latency_ms / 1000,
        "latency_ms": latency_ms,
        "latency_std_ms": abs(np.random.normal(0, 50)),
        "vram_usage_mb": vram_mb,

        # Quality metrics
        "quality_score": quality,
        "quality_score_std": abs(np.random.normal(0, 0.05)),
        "faithfulness": faithfulness,
        "faithfulness_std": abs(np.random.normal(0, 0.05)),
        "correctness": correctness,
        "correctness_std": abs(np.random.normal(0, 0.05)),
        "completeness": completeness,
        "completeness_std": abs(np.random.normal(0, 0.05)),
        "conciseness": conciseness,
        "conciseness_std": abs(np.random.normal(0, 0.05)),
        "context_precision": context_precision,
        "context_precision_std": abs(np.random.normal(0, 0.05)),

        # Dataset info
        "num_queries": num_queries
    }

    return metrics


# ============================================================================
# EXPERIMENT GENERATION FUNCTIONS
# ============================================================================

def generate_chunking_experiments(num_runs: int = 5, start_date: datetime = None):
    """
    Generate fake chunking experiments.

    Args:
        num_runs: Number of runs per strategy
        start_date: Starting date for experiments (defaults to 30 days ago)
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)

    experiment_name = "Chunking_Strategy_Experiments"
    logger.info(f"Generating {num_runs} runs for each chunking strategy...")

    # Set experiment (don't use context manager to avoid auto-starting runs)
    mlflow.set_experiment(experiment_name)

    run_count = 0

    for strategy in CHUNKING_STRATEGIES:
        for run_idx in range(num_runs):
            # Generate unique run timestamp
            run_time = start_date + timedelta(
                days=random.randint(0, 25),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )

            # Generate metrics
            metrics = generate_chunking_metrics(strategy)

            # Start run with timestamp
            with mlflow.start_run(run_name=f"{strategy['name']}_run_{run_idx + 1}"):
                # Log parameters
                mlflow.log_param("strategy_name", strategy['name'])
                mlflow.log_param("strategy_type", strategy['strategy_type'])
                mlflow.log_param("description", strategy['description'])
                mlflow.log_param("chunk_size", strategy['chunk_size'])
                mlflow.log_param("chunk_overlap", strategy['chunk_overlap'])
                if 'child_size' in strategy:
                    mlflow.log_param("child_size", strategy['child_size'])

                # Log all metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Log sample artifacts
                sample_chunks = {
                    "strategy": strategy['name'],
                    "sample_chunks": [
                        {
                            "text": f"Sample chunk {i+1} for {strategy['name']}...",
                            "size": int(metrics['avg_chunk_size_chars'] + random.randint(-100, 100)),
                            "index": i
                        }
                        for i in range(5)
                    ]
                }
                log_dict_as_json(sample_chunks, "sample_chunks.json")

                # Add tags
                mlflow.set_tag("experiment_type", "chunking")
                mlflow.set_tag("generated", "true")
                mlflow.set_tag("run_date", run_time.strftime("%Y-%m-%d"))

                run_count += 1
                logger.info(f"  ✓ Created run {run_count}: {strategy['name']} (run {run_idx + 1})")

    logger.info(f"✅ Generated {run_count} chunking experiment runs")


def generate_retrieval_experiments(num_runs: int = 5, start_date: datetime = None):
    """
    Generate fake retrieval experiments.

    Args:
        num_runs: Number of runs per strategy
        start_date: Starting date for experiments (defaults to 30 days ago)
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)

    experiment_name = "Retrieval_Strategy_Experiments"
    logger.info(f"Generating {num_runs} runs for each retrieval strategy...")

    # Set experiment (don't use context manager to avoid auto-starting runs)
    mlflow.set_experiment(experiment_name)

    run_count = 0

    for strategy in RETRIEVAL_STRATEGIES:
            for run_idx in range(num_runs):
                # Generate unique run timestamp
                run_time = start_date + timedelta(
                    days=random.randint(0, 25),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )

                # Generate metrics
                metrics = generate_retrieval_metrics(strategy)

                # Start run with timestamp
                with mlflow.start_run(run_name=f"{strategy['name']}_run_{run_idx + 1}"):
                    # Log parameters
                    mlflow.log_param("strategy_name", strategy['name'])
                    mlflow.log_param("description", strategy['description'])
                    mlflow.log_param("alpha", strategy['alpha'])
                    mlflow.log_param("top_k", 5)

                    # Log all metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)

                    # Log sample artifacts
                    query_results = {
                        "strategy": strategy['name'],
                        "sample_results": [
                            {
                                "query": f"Sample query {i+1}",
                                "precision_at_5": add_noise(strategy['base_precision'], 0.1),
                                "ndcg_at_5": add_noise(strategy['base_ndcg'], 0.1),
                                "latency_ms": strategy['base_latency_ms'] + random.uniform(-10, 10)
                            }
                            for i in range(5)
                        ]
                    }
                    log_dict_as_json(query_results, "query_results.json")

                    # Add tags
                    mlflow.set_tag("experiment_type", "retrieval")
                    mlflow.set_tag("generated", "true")
                    mlflow.set_tag("run_date", run_time.strftime("%Y-%m-%d"))

                    run_count += 1
                    logger.info(f"  ✓ Created run {run_count}: {strategy['name']} (run {run_idx + 1})")

    logger.info(f"✅ Generated {run_count} retrieval experiment runs")


def generate_embedding_experiments(num_runs: int = 5, start_date: datetime = None):
    """
    Generate fake embedding experiments.

    Args:
        num_runs: Number of runs per model
        start_date: Starting date for experiments (defaults to 30 days ago)
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)

    experiment_name = "Embedding_Model_Experiments"
    logger.info(f"Generating {num_runs} runs for each embedding model...")

    # Set experiment (don't use context manager to avoid auto-starting runs)
    mlflow.set_experiment(experiment_name)

    run_count = 0

    for model in EMBEDDING_MODELS:
            for run_idx in range(num_runs):
                # Generate unique run timestamp
                run_time = start_date + timedelta(
                    days=random.randint(0, 25),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )

                # Generate metrics
                metrics = generate_embedding_metrics(model)

                # Start run with timestamp
                with mlflow.start_run(run_name=f"{model['name'].split('/')[-1]}_run_{run_idx + 1}"):
                    # Log parameters
                    mlflow.log_param("model_name", model['name'])
                    mlflow.log_param("description", model['description'])
                    mlflow.log_param("embedding_dim", model['embedding_dim'])

                    # Log all metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)

                    # Log sample artifacts
                    detailed_results = {
                        "model": model['name'],
                        "sample_query_results": [
                            {
                                "query_id": i,
                                "precision_at_5": add_noise(model['base_precision'], 0.1),
                                "ndcg_at_5": add_noise(model['base_ndcg_5'], 0.1),
                                "latency_ms": model['base_query_latency_ms'] + random.uniform(-5, 5)
                            }
                            for i in range(5)
                        ]
                    }
                    log_dict_as_json(detailed_results, "detailed_results.json")

                    # Add tags
                    mlflow.set_tag("experiment_type", "embedding")
                    mlflow.set_tag("generated", "true")
                    mlflow.set_tag("run_date", run_time.strftime("%Y-%m-%d"))
                    mlflow.set_tag("model_family", model['name'].split('/')[0])

                    run_count += 1
                    logger.info(f"  ✓ Created run {run_count}: {model['name']} (run {run_idx + 1})")

    logger.info(f"✅ Generated {run_count} embedding experiment runs")


def generate_llm_experiments(num_runs: int = 5, start_date: datetime = None):
    """
    Generate fake LLM experiments.

    Args:
        num_runs: Number of runs per model
        start_date: Starting date for experiments (defaults to 30 days ago)
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=30)

    experiment_name = "LLM_Model_Experiments"
    logger.info(f"Generating {num_runs} runs for each LLM model...")

    # Set experiment (don't use context manager to avoid auto-starting runs)
    mlflow.set_experiment(experiment_name)

    run_count = 0

    for model in LLM_MODELS:
            for run_idx in range(num_runs):
                # Generate unique run timestamp
                run_time = start_date + timedelta(
                    days=random.randint(0, 25),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )

                # Generate metrics
                metrics = generate_llm_metrics(model)

                # Start run with timestamp
                with mlflow.start_run(run_name=f"{model['name']}_run_{run_idx + 1}"):
                    # Log parameters
                    mlflow.log_param("model_name", model['name'])
                    mlflow.log_param("description", model['description'])
                    mlflow.log_param("temperature", 0.3)
                    mlflow.log_param("max_tokens", 256)

                    # Log all metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)

                    # Log sample artifacts
                    sample_answers = {
                        "model": model['name'],
                        "sample_answers": [
                            {
                                "query": f"Sample query {i+1}",
                                "answer": f"Sample answer {i+1} generated by {model['name']}...",
                                "quality": add_noise(model['base_quality'], 0.1),
                                "faithfulness": add_noise(model['base_faithfulness'], 0.1),
                                "latency_ms": model['base_latency_ms'] + random.uniform(-100, 100)
                            }
                            for i in range(5)
                        ]
                    }
                    log_dict_as_json(sample_answers, "sample_answers.json")

                    # Add tags
                    mlflow.set_tag("experiment_type", "llm")
                    mlflow.set_tag("generated", "true")
                    mlflow.set_tag("run_date", run_time.strftime("%Y-%m-%d"))
                    mlflow.set_tag("model_family", model['name'].split(':')[0])

                    run_count += 1
                    logger.info(f"  ✓ Created run {run_count}: {model['name']} (run {run_idx + 1})")

    logger.info(f"✅ Generated {run_count} LLM experiment runs")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to generate fake experiments."""
    parser = argparse.ArgumentParser(
        description="Generate fake MLflow experiments with logical metrics"
    )
    parser.add_argument(
        "--chunking",
        action="store_true",
        help="Generate chunking experiments"
    )
    parser.add_argument(
        "--retrieval",
        action="store_true",
        help="Generate retrieval experiments"
    )
    parser.add_argument(
        "--embedding",
        action="store_true",
        help="Generate embedding experiments"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Generate LLM experiments"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all experiment types"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per strategy/model (default: 5)"
    )
    parser.add_argument(
        "--days-ago",
        type=int,
        default=30,
        help="Start experiments N days ago (default: 30)"
    )

    args = parser.parse_args()

    # If --all is specified or no specific experiment type is selected, generate all
    if args.all or (not args.chunking and not args.retrieval and not args.embedding and not args.llm):
        args.chunking = True
        args.retrieval = True
        args.embedding = True
        args.llm = True

    start_date = datetime.now() - timedelta(days=args.days_ago)

    logger.info("=" * 80)
    logger.info("GENERATING FAKE MLFLOW EXPERIMENTS")
    logger.info("=" * 80)
    logger.info(f"Runs per strategy/model: {args.runs}")
    logger.info(f"Start date: {start_date.strftime('%Y-%m-%d')}")
    logger.info("")

    # Generate experiments
    if args.chunking:
        logger.info("\n" + "=" * 80)
        logger.info("CHUNKING EXPERIMENTS")
        logger.info("=" * 80)
        generate_chunking_experiments(num_runs=args.runs, start_date=start_date)

    if args.retrieval:
        logger.info("\n" + "=" * 80)
        logger.info("RETRIEVAL EXPERIMENTS")
        logger.info("=" * 80)
        generate_retrieval_experiments(num_runs=args.runs, start_date=start_date)

    if args.embedding:
        logger.info("\n" + "=" * 80)
        logger.info("EMBEDDING EXPERIMENTS")
        logger.info("=" * 80)
        generate_embedding_experiments(num_runs=args.runs, start_date=start_date)

    if args.llm:
        logger.info("\n" + "=" * 80)
        logger.info("LLM EXPERIMENTS")
        logger.info("=" * 80)
        generate_llm_experiments(num_runs=args.runs, start_date=start_date)

    logger.info("\n" + "=" * 80)
    logger.info("✅ ALL EXPERIMENTS GENERATED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("\nTo view the experiments, run:")
    logger.info("  mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db")
    logger.info("\nThen open: http://localhost:5000")


if __name__ == "__main__":
    main()


