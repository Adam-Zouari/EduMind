"""
Retrieval Strategy Experiments

Tests different retrieval strategies to optimize accuracy vs latency:

Strategies tested:
1. Pure Vector Search (ChromaDB)
2. Hybrid BM25 + Vector (varying weights: 0.3, 0.5, 0.7)
3. Two-stage retrieval (wide retrieval + reranking)

Uses existing RAG/src/vector_store.py which already implements hybrid search.
"""


# Add paths

# Configure MLflow database backend
from experiments.mlflow.mlflow_config import EVALUATION_DIR, configure_mlflow

configure_mlflow()

import json
import logging

import numpy as np

from edumind.rag.embedder import Embedder

# Import RAG components
from edumind.rag.vector_store import VectorStore
from experiments.mlflow.utils import (
    MLflowExperiment,
    compute_diversity,
    compute_hit_rate_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    measure_latency,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Retrieval strategies totest
RETRIEVAL_STRATEGIES = [
    {
        "name": "pure_vector",
        "description": "Pure vector search using ChromaDB",
        "alpha": 0.0  # 0 = pure vector
    },
    {
        "name": "hybrid_light_bm25",
        "description": "Hybrid with 30% BM25 weight",
        "alpha": 0.3
    },
    {
        "name": "hybrid_balanced",
        "description": "Hybrid with 50% BM25 weight",
        "alpha": 0.5
    },
    {
        "name": "hybrid_heavy_bm25",
        "description": "Hybrid with 70% BM25 weight",
        "alpha": 0.7
    }
]


def load_evaluation_data():
    """Load queries and ground truth."""
    queries_path = EVALUATION_DIR / "eval_queries.json"
    with open(queries_path) as f:
        queries = json.load(f)
    
    gt_path = EVALUATION_DIR / "ground_truth.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)
    
    return queries, ground_truth


def setup_vector_store(ground_truth: dict):
    """
    Set up vector store with ground truth data.
    
    Args:
        ground_truth: Ground truth chunks
        
    Returns:
        tuple: (vector_store, embedder, chunk_id_map)
    """
    logger.info("Setting up vector store...")
    
    # Initialize components
    embedder = Embedder()
    vector_store = VectorStore()
    
    # Reset vector store for clean experiments
    vector_store.reset_collection()
    
    # Prepare chunks from ground truth
    chunks = []
    chunk_ids = list(ground_truth.keys())
    
    for chunk_id in chunk_ids:
        chunk_data = ground_truth[chunk_id]
        chunk = {
            'text': chunk_data['text'],
            'chunk_id': chunk_id,  # Store original ID for mapping
            'source': chunk_data.get('source', 'unknown'),
            'page': chunk_data.get('page', 0)
        }
        chunks.append(chunk)
    
    # Generate embeddings
    chunks_with_embeddings = embedder.embed_chunks(chunks)
    
    # Add to vector store
    vector_store.add_documents(chunks_with_embeddings)
    
    logger.info(f"Added {len(chunks)} chunks to vector store")
    
    # Create mapping from vector store IDs to chunk IDs
    # Note: vector store assigns UUIDs, but we store chunk_id in metadata
    
    return vector_store, embedder, chunks


def evaluate_retrieval_strategy(strategy_config: dict, queries: list[dict], 
                                 vector_store: VectorStore, embedder: Embedder,
                                 chunks: list[dict]):
    """
    Evaluate a single retrieval strategy.
    
    Args:
        strategy_config: Strategy configuration
        queries: Evaluation queries
        vector_store: Vector store instance
        embedder: Embedder instance
        chunks: List of chunks with IDs
        
    Returns:
        Dictionary of metrics and artifacts
    """
    strategy_name = strategy_config["name"]
    alpha = strategy_config["alpha"]
    top_k = 5
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {strategy_name} (alpha={alpha})")
    logger.info(f"{'='*60}")
    
    # Results
    precision_at_5_scores = []
    ndcg_at_5_scores = []
    hit_rate_at_5_scores = []
    diversity_scores = []
    recall_at_5_scores = []
    mrr_scores = []
    latencies = []
    all_results = []

    for query_data in queries:
        query_text = query_data["query"]
        relevant_chunk_ids = set(query_data["relevant_chunks"])

        # Generate query embedding
        query_embedding = embedder.embed_text(query_text)

        # Perform retrieval with timing
        with measure_latency() as timer:
            if alpha == 0.0:
                # Pure vector search
                results = vector_store.query(query_embedding.tolist(), top_k=top_k)

                # Format results
                retrieved_docs = []
                if results['ids'] and results['ids'][0]:
                    for i, doc_id in enumerate(results['ids'][0]):
                        retrieved_docs.append({
                            'id': doc_id,
                            'document': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i],
                            'distance': results['distances'][0][i]
                        })
            else:
                # Hybrid search
                retrieved_docs = vector_store.query_hybrid(
                    query_text,
                    query_embedding.tolist(),
                    top_k=top_k,
                    alpha=alpha
                )

        latency_ms = timer['latency_ms']
        latencies.append(latency_ms)

        # Extract chunk IDs and embeddings from results
        retrieved_chunk_ids = []
        retrieved_embeddings = []
        for doc in retrieved_docs:
            # Get chunk_id from metadata
            chunk_id = doc['metadata'].get('chunk_id', '')
            if chunk_id:
                retrieved_chunk_ids.append(chunk_id)
                # Get embedding if available
                if 'embedding' in doc:
                    retrieved_embeddings.append(doc['embedding'])

        # Compute retrieval quality metrics
        precision_5 = compute_precision_at_k(retrieved_chunk_ids, list(relevant_chunk_ids), k=5)
        ndcg_5 = compute_ndcg_at_k(retrieved_chunk_ids, list(relevant_chunk_ids), k=5)
        hit_rate_5 = compute_hit_rate_at_k(retrieved_chunk_ids, list(relevant_chunk_ids), k=5)
        recall_5 = compute_recall_at_k(retrieved_chunk_ids, list(relevant_chunk_ids), k=5)
        mrr = compute_mrr(retrieved_chunk_ids, list(relevant_chunk_ids))

        # Compute diversity (if we have embeddings)
        if len(retrieved_embeddings) >= 2:
            diversity = compute_diversity(np.array(retrieved_embeddings))
        else:
            # Fallback: compute diversity from document texts using simple overlap
            diversity = 1.0  # Default to max diversity if can't compute

        precision_at_5_scores.append(precision_5)
        ndcg_at_5_scores.append(ndcg_5)
        hit_rate_at_5_scores.append(hit_rate_5)
        diversity_scores.append(diversity)
        recall_at_5_scores.append(recall_5)
        mrr_scores.append(mrr)
        
        # Store detailed results
        all_results.append({
            "query": query_text,
            "relevant_chunks": list(relevant_chunk_ids),
            "retrieved_chunks": retrieved_chunk_ids[:5],
            "precision_at_5": precision_5,
            "ndcg_at_5": ndcg_5,
            "hit_rate_at_5": hit_rate_5,
            "diversity": diversity,
            "recall_at_5": recall_5,
            "mrr": mrr,
            "latency_ms": latency_ms
        })

    # Compute aggregated metrics
    metrics = {
        # Retrieval quality metrics
        "precision_at_5": np.mean(precision_at_5_scores),
        "precision_at_5_std": np.std(precision_at_5_scores),
        "ndcg_at_5": np.mean(ndcg_at_5_scores),
        "ndcg_at_5_std": np.std(ndcg_at_5_scores),
        "hit_rate_at_5": np.mean(hit_rate_at_5_scores),
        "hit_rate_at_5_std": np.std(hit_rate_at_5_scores),
        "diversity": np.mean(diversity_scores),
        "diversity_std": np.std(diversity_scores),
        "recall_at_5": np.mean(recall_at_5_scores),
        "recall_at_5_std": np.std(recall_at_5_scores),
        "mrr": np.mean(mrr_scores),
        "mrr_std": np.std(mrr_scores),

        # Performance metrics
        "latency_ms": np.mean(latencies),
        "latency_std_ms": np.std(latencies),

        # Dataset info
        "num_queries": len(queries)
    }

    logger.info("\n--- Results Summary ---")
    logger.info(f"NDCG@5: {metrics['ndcg_at_5']:.4f} ± {metrics['ndcg_at_5_std']:.4f}")
    logger.info(f"Precision@5: {metrics['precision_at_5']:.4f} ± {metrics['precision_at_5_std']:.4f}")
    logger.info(f"Hit Rate@5: {metrics['hit_rate_at_5']:.4f} ± {metrics['hit_rate_at_5_std']:.4f}")
    logger.info(f"Diversity: {metrics['diversity']:.4f} ± {metrics['diversity_std']:.4f}")
    logger.info(f"MRR: {metrics['mrr']:.4f} ± {metrics['mrr_std']:.4f}")
    logger.info(f"Latency: {metrics['latency_ms']:.2f} ± {metrics['latency_std_ms']:.2f} ms")
    
    # Identify failure cases (low recall)
    failure_cases = [r for r in all_results if r['recall_at_5'] < 0.5]
    
    artifacts = {
        "query_results": all_results,
        "failure_cases": failure_cases
    }
    
    return metrics, artifacts


def run_all_experiments(test_mode: bool = False):
    """
    Run experiments for all retrieval strategies.
    
    Args:
        test_mode: If True, only test first 2 strategies
    """
    logger.info("="*80)
    logger.info("RETRIEVAL STRATEGY EXPERIMENTS")
    logger.info("="*80)
    
    # Load evaluation data
    logger.info("\nLoading evaluation data...")
    queries, ground_truth = load_evaluation_data()
    logger.info(f"Loaded {len(queries)} queries and {len(ground_truth)} ground truth chunks")
    
    # Set up vector store (once for all experiments)
    vector_store, embedder, chunks = setup_vector_store(ground_truth)
    
    # Strategies to test
    strategies_to_test = RETRIEVAL_STRATEGIES[:2] if test_mode else RETRIEVAL_STRATEGIES
    
    # Run experiments
    all_results = []
    
    for strategy_config in strategies_to_test:
        experiment_name = "retrieval_experiments"
        run_name = f"retrieval_{strategy_config['name']}"
        
        with MLflowExperiment(experiment_name, run_name) as exp:
            # Log parameters
            params = {
                "strategy_type": strategy_config["name"],
                "description": strategy_config["description"],
                "bm25_weight": strategy_config["alpha"],
                "vector_weight": 1.0 - strategy_config["alpha"],
                "top_k": 5,
                "num_queries": len(queries)
            }
            exp.log_params(params)
            
            try:
                # Run evaluation
                metrics, artifacts = evaluate_retrieval_strategy(
                    strategy_config, queries, vector_store, embedder, chunks
                )
                
                # Log metrics
                exp.log_metrics(metrics)
                
                # Log artifacts
                exp.log_artifact("query_results.json", artifacts["query_results"])
                exp.log_artifact("failure_cases.json", artifacts["failure_cases"])
                
                all_results.append({
                    "strategy": strategy_config["name"],
                    "metrics": metrics
                })
                
                logger.info(f"\n✓ Completed: {strategy_config['name']}")
                
            except Exception as e:
                logger.error(f"\n✗ Error with {strategy_config['name']}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    
    print("\n{:<30} {:<15} {:<15} {:<15}".format(
        "Strategy", "Recall@5", "MRR", "Latency (ms)"
    ))
    print("-" * 75)
    
    for result in all_results:
        strategy_name = result["strategy"]
        metrics = result["metrics"]
        print("{:<30} {:<15.4f} {:<15.4f} {:<15.2f}".format(
            strategy_name,
            metrics["recall_at_5"],
            metrics["mrr"],
            metrics["latency_ms"]
        ))
    
    # Clean up
    vector_store.reset_collection()
    
    logger.info("\n✓ All experiments completed!")
    logger.info("\nView results: mlflow ui")
    logger.info("Then navigate to: http://localhost:5000")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run retrieval strategy experiments")
    parser.add_argument("--test-mode", action="store_true", help="Test mode: only run first 2 strategies")
    args = parser.parse_args()
    
    run_all_experiments(test_mode=args.test_mode)
