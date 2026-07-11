"""
Embedding Model Experiments

Tests different embedding models to find the optimal balance between:
- Retrieval quality (Recall@K, MRR)
- Inference speed (throughput, latency)
- Resource efficiency (GPU memory)

Models tested:
1. all-MiniLM-L6-v2 (lightweight, fast)
2. all-mpnet-base-v2 (balanced)
3. e5-large-v2 (high quality)
4. bge-large-en-v1.5 (state-of-the-art)
5. multilingual-e5-large (multilingual support)
"""


# Add parent directory to path

# Configure MLflow database backend
from experiments.mlflow.mlflow_config import EVALUATION_DIR, configure_mlflow

configure_mlflow()

import json
import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from experiments.mlflow.utils import (
    MLflowExperiment,
    compute_map,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    get_gpu_memory_usage,
    is_cuda_available,
    measure_latency,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Embedding models to test
EMBEDDING_MODELS = [
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": 384,
        "description": "Lightweight and fast"
    },
    {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "embedding_dim": 768,
        "description": "Balanced performance"
    },
    {
        "name": "sentence-transformers/multi-qa-mpnet-base-dot-v1",
        "embedding_dim": 768,
        "description": "Optimized for QA retrieval"
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "embedding_dim": 384,
        "description": "BGE small model"
    },
    {
        "name": "BAAI/bge-base-en-v1.5",
        "embedding_dim": 768,
        "description": "BGE base model"
    }
]


def load_evaluation_data():
    """Load queries and ground truth data."""
    # Load queries
    queries_path = EVALUATION_DIR / "eval_queries.json"
    with open(queries_path) as f:
        queries = json.load(f)
    
    # Load ground truth chunks
    gt_path = EVALUATION_DIR / "ground_truth.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)
    
    return queries, ground_truth


def evaluate_embedding_model(model_config: dict, queries: list[dict], ground_truth: dict):
    """
    Evaluate a single embedding model.
    
    Args:
        model_config: Model configuration
        queries: List of evaluation queries
        ground_truth: Ground truth chunks
        
    Returns:
        Dictionary of metrics
    """
    model_name = model_config["name"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*60}")
    
    # Determine device
    device = "cuda" if is_cuda_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info("Loading model...")
    load_start = time.time()
    model = SentenceTransformer(model_name, device=device)
    load_time = time.time() - load_start
    logger.info(f"Model loaded in {load_time:.2f}s")
    
    # Prepare data
    chunk_ids = list(ground_truth.keys())
    chunk_texts = [ground_truth[cid]["text"] for cid in chunk_ids]
    query_texts = [q["query"] for q in queries]
    
    # === Benchmark: Encoding Throughput ===
    logger.info("\n--- Encoding Document Chunks ---")
    
    with measure_latency() as timer_chunks:
        chunk_embeddings = model.encode(
            chunk_texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
    
    chunk_encoding_time = timer_chunks['latency_ms'] / 1000  # Convert to seconds
    chunk_throughput = len(chunk_texts) / chunk_encoding_time
    logger.info(f"Encoded {len(chunk_texts)} chunks in {chunk_encoding_time:.2f}s")
    logger.info(f"Throughput: {chunk_throughput:.2f} sentences/sec")
    
    # === Benchmark: Query Latency ===
    logger.info("\n--- Encoding Queries ---")
    
    query_latencies = []
    query_embeddings = []
    
    for query_text in query_texts:
        with measure_latency() as timer:
            query_emb = model.encode(query_text, convert_to_numpy=True)
        query_latencies.append(timer['latency_ms'])
        query_embeddings.append(query_emb)
    
    query_embeddings = np.array(query_embeddings)
    avg_query_latency = np.mean(query_latencies)
    logger.info(f"Average query encoding latency: {avg_query_latency:.2f}ms")
    
    # === Evaluate Retrieval Quality ===
    logger.info("\n--- Evaluating Retrieval Quality ---")

    precision_at_5_scores = []
    ndcg_at_5_scores = []
    ndcg_at_10_scores = []
    map_scores = []
    mrr_scores = []

    for idx, query_data in enumerate(queries):
        query_emb = query_embeddings[idx]

        # Compute similarity scores (cosine similarity via dot product for normalized embeddings)
        # Normalize embeddings first
        query_emb_norm = query_emb / np.linalg.norm(query_emb)
        chunk_embeddings_norm = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)

        similarities = np.dot(chunk_embeddings_norm, query_emb_norm)

        # Get top-k results
        top_k = 10
        top_indices = np.argsort(similarities)[::-1][:top_k]
        retrieved_ids = [chunk_ids[i] for i in top_indices]

        # Get ground truth relevant chunks
        # NOTE: In synthetic data, ALL chunks with matching (domain, topic, variant) are relevant
        # The original "relevant_chunks" field only contains a random subset
        expected_domain = query_data.get("domain")
        expected_variant = query_data.get("expected_variant")

        # Get topic from one of the relevant chunks
        if query_data["relevant_chunks"]:
            sample_chunk_id = query_data["relevant_chunks"][0]
            expected_topic = ground_truth[sample_chunk_id]["topic"]

            # Find all truly relevant chunks (all with matching domain/topic/variant)
            all_relevant_ids = [
                cid for cid in chunk_ids
                if ground_truth[cid]["domain"] == expected_domain
                and ground_truth[cid]["topic"] == expected_topic
                and ground_truth[cid].get("variant") == expected_variant
            ]
        else:
            # Fallback to original logic if no relevant chunks
            all_relevant_ids = query_data["relevant_chunks"]

        # Compute metrics using utility functions
        precision_5 = compute_precision_at_k(retrieved_ids, all_relevant_ids, k=5)
        ndcg_5 = compute_ndcg_at_k(retrieved_ids, all_relevant_ids, k=5)
        ndcg_10 = compute_ndcg_at_k(retrieved_ids, all_relevant_ids, k=10)
        map_score = compute_map(retrieved_ids, all_relevant_ids)
        mrr = compute_mrr(retrieved_ids, all_relevant_ids)

        precision_at_5_scores.append(precision_5)
        ndcg_at_5_scores.append(ndcg_5)
        ndcg_at_10_scores.append(ndcg_10)
        map_scores.append(map_score)
        mrr_scores.append(mrr)
    
    # === GPU Memory Usage ===
    gpu_metrics = get_gpu_memory_usage()
    gpu_memory_mb = gpu_metrics.get('allocated_mb', 0)
    
    # === Compile Results ===
    metrics = {
        # Performance metrics
        "throughput_sent_per_sec": chunk_throughput,
        "avg_query_latency_ms": avg_query_latency,
        "gpu_memory_mb": gpu_memory_mb,
        "model_load_time_sec": load_time,
        "embedding_dim": model_config["embedding_dim"],

        # Retrieval quality metrics (PRIMARY)
        "precision_at_5": np.mean(precision_at_5_scores),
        "precision_at_5_std": np.std(precision_at_5_scores),
        "ndcg_at_5": np.mean(ndcg_at_5_scores),
        "ndcg_at_5_std": np.std(ndcg_at_5_scores),
        "ndcg_at_10": np.mean(ndcg_at_10_scores),
        "ndcg_at_10_std": np.std(ndcg_at_10_scores),
        "map": np.mean(map_scores),
        "map_std": np.std(map_scores),
        "mrr": np.mean(mrr_scores),
        "mrr_std": np.std(mrr_scores),

        # Dataset info
        "num_queries": len(queries),
        "num_chunks": len(chunk_texts)
    }

    logger.info("\n--- Results Summary ---")
    logger.info(f"NDCG@5: {metrics['ndcg_at_5']:.4f} ± {metrics['ndcg_at_5_std']:.4f} (PRIMARY METRIC)")
    logger.info(f"Precision@5: {metrics['precision_at_5']:.4f} ± {metrics['precision_at_5_std']:.4f}")
    logger.info(f"MAP: {metrics['map']:.4f} ± {metrics['map_std']:.4f}")
    logger.info(f"MRR: {metrics['mrr']:.4f} ± {metrics['mrr_std']:.4f}")
    logger.info(f"Throughput: {metrics['throughput_sent_per_sec']:.2f} sent/sec")
    logger.info(f"Query Latency: {metrics['avg_query_latency_ms']:.2f}ms")
    logger.info(f"GPU Memory: {metrics['gpu_memory_mb']:.2f}MB")

    # Prepare artifacts
    artifacts = {
        "detailed_results": {
            "per_query_precision_at_5": precision_at_5_scores,
            "per_query_ndcg_at_5": ndcg_at_5_scores,
            "per_query_ndcg_at_10": ndcg_at_10_scores,
            "per_query_map": map_scores,
            "per_query_mrr": mrr_scores,
            "per_query_latency_ms": query_latencies
        },
        "sample_embeddings": query_embeddings[:5]  # First 5 query embeddings
    }
    
    return metrics, artifacts


def run_all_experiments(test_mode: bool = False):
    """
    Run experiments for all embedding models.
    
    Args:
        test_mode: If True, only test first 2 models
    """
    logger.info("="*80)
    logger.info("EMBEDDING MODEL EXPERIMENTS")
    logger.info("="*80)
    
    # Load evaluation data
    logger.info("\nLoading evaluation data...")
    queries, ground_truth = load_evaluation_data()
    logger.info(f"Loaded {len(queries)} queries and {len(ground_truth)} ground truth chunks")
    
    # Models to test
    models_to_test = EMBEDDING_MODELS[:2] if test_mode else EMBEDDING_MODELS
    
    # Run experiments
    all_results = []
    
    for model_config in models_to_test:
        experiment_name = "embedding_experiments"
        run_name = f"embedding_{model_config['name'].split('/')[-1]}"
        
        with MLflowExperiment(experiment_name, run_name) as exp:
            # Log parameters
            params = {
                "model_name": model_config["name"],
                "embedding_dim": model_config["embedding_dim"],
                "description": model_config["description"],
                "device": "cuda" if is_cuda_available() else "cpu",
                "batch_size": 32
            }
            exp.log_params(params)
            
            try:
                # Run evaluation
                metrics, artifacts = evaluate_embedding_model(model_config, queries, ground_truth)
                
                # Log metrics
                exp.log_metrics(metrics)
                
                # Log artifacts
                exp.log_artifact("evaluation_results.json", artifacts["detailed_results"])
                exp.log_artifact("sample_embeddings.npy", artifacts["sample_embeddings"])
                
                all_results.append({
                    "model": model_config["name"],
                    "metrics": metrics
                })
                
                logger.info(f"\n✓ Completed: {model_config['name']}")
                
            except Exception as e:
                logger.error(f"\n✗ Error with {model_config['name']}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)

    print("\n{:<40} {:<15} {:<15} {:<15} {:<15}".format(
        "Model", "Precision@5", "MRR", "Latency (ms)", "Throughput"
    ))
    print("-" * 100)

    for result in all_results:
        model_name = result["model"].split("/")[-1]
        metrics = result["metrics"]
        print("{:<40} {:<15.4f} {:<15.4f} {:<15.2f} {:<15.2f}".format(
            model_name,
            metrics["precision_at_5"],
            metrics["mrr"],
            metrics["avg_query_latency_ms"],
            metrics["throughput_sent_per_sec"]
        ))
    
    logger.info("\n✓ All experiments completed!")
    logger.info("\nView results: mlflow ui")
    logger.info("Then navigate to: http://localhost:5000")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run embedding model experiments")
    parser.add_argument("--test-mode", action="store_true", help="Test mode: only run first 2 models")
    args = parser.parse_args()
    
    run_all_experiments(test_mode=args.test_mode)
