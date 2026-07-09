"""
Chunking Strategy Experiments

Tests different text chunking strategies to optimize retrieval quality.

Strategies tested:
1. Fixed Character (Baseline) - 1000 chars, 200 overlap
2. Fixed Character (Large) - 1500 chars, 300 overlap
3. Semantic Chunking - Variable size, semantic breaks
4. Sentence Window - 10 sentences, 2 sentence overlap
5. Hierarchical (Parent+Child) - 2000 parent + 500 child chunks

Each strategy is evaluated on:
- Retrieval quality (Recall@5)
- Answer quality (faithfulness, completeness)
- Chunk statistics (avg size, total count)
"""

import sys
import os
from pathlib import Path
import re

# Add paths

# Configure MLflow database backend
from experiments.mlflow.mlflow_config import EVALUATION_DIR, configure_mlflow
configure_mlflow()

from experiments.mlflow.utils import (
    MLflowExperiment,
    compute_recall_at_k,
    compute_mrr,
    compute_precision_at_k,
    compute_ndcg_at_k,
    compute_diversity,
    evaluate_answer_quality,
    evaluate_faithfulness,
    compute_chunk_size_statistics,
    compute_chunk_coherence,
    log_dict_as_json,
    log_figure,
    create_comparison_plot
)

# Import RAG components
from edumind.rag.vector_store import VectorStore
from edumind.rag.embedder import Embedder

import json
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Chunking strategies configuration
CHUNKING_STRATEGIES = [
    {
        "name": "fixed_character_baseline",
        "description": "Fixed 1000 chars, 200 overlap (baseline)",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "strategy_type": "fixed_character"
    },
    {
        "name": "fixed_character_large",
        "description": "Fixed 1500 chars, 300 overlap",
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "strategy_type": "fixed_character"
    },
    {
        "name": "semantic_chunking",
        "description": "Variable size, semantic breaks",
        "chunk_size": 1000,  # Target size
        "chunk_overlap": 0.1,  # 10% overlap
        "strategy_type": "semantic"
    },
    {
        "name": "sentence_window",
        "description": "10 sentences, 2 sentence overlap",
        "chunk_size": 10,  # 10 sentences
        "chunk_overlap": 2,   # 2 sentences
        "strategy_type": "sentence_window"
    },
    {
        "name": "hierarchical",
        "description": "Parent (2000) + Child (500) chunks",
        "chunk_size": 2000,  # Parent size
        "child_size": 500,    # Child size
        "chunk_overlap": 0,
        "strategy_type": "hierarchical"
    }
]


class ChunkingStrategy:
    """Base class for chunking strategies."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.name = config["name"]
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """Chunk text according to strategy."""
        raise NotImplementedError


class FixedCharacterChunker(ChunkingStrategy):
    """Fixed character-based chunking."""
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        chunk_size = self.config["chunk_size"]
        overlap = self.config["chunk_overlap"]
        
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunk = {
                    'text': chunk_text,
                    'chunk_index': chunk_idx,
                    'start_char': start,
                    'end_char': end
                }
                if metadata:
                    chunk.update(metadata)
                chunks.append(chunk)
                chunk_idx += 1
            
            start = end - overlap
        
        return chunks


class SentenceWindowChunker(ChunkingStrategy):
    """Sentence-based windowing."""
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        num_sentences = self.config["chunk_size"]
        overlap = self.config["chunk_overlap"]
        
        sentences = self._split_sentences(text)
        chunks = []
        
        idx = 0
        chunk_idx = 0
        
        while idx < len(sentences):
            end_idx = min(idx + num_sentences, len(sentences))
            chunk_sentences = sentences[idx:end_idx]
            chunk_text = " ".join(chunk_sentences)
            
            chunk = {
                'text': chunk_text,
                'chunk_index': chunk_idx,
                'num_sentences': len(chunk_sentences)
            }
            if metadata:
                chunk.update(metadata)
            chunks.append(chunk)
            
            idx += (num_sentences - overlap)
            chunk_idx += 1
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """Semantic similarity-based chunking."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')
    
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        target_size = self.config["chunk_size"]
        min_size = int(target_size * 0.5)
        
        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return [{'text': text, 'chunk_index': 0, **(metadata or {})}]
        
        # Encode sentences
        embeddings = self.model.encode(sentences)
        
        # Find semantic breaks
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
            similarities.append(sim)
        
        # Low similarity = semantic break
        threshold = np.percentile(similarities, 10)
        break_points = [i for i, sim in enumerate(similarities) if sim < threshold]
        
        # Create chunks respecting size constraints
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_idx = 0
        
        for i, sentence in enumerate(sentences):
            current_chunk.append(sentence)
            current_size += len(sentence)
            
            is_break = i in break_points
            is_big_enough = current_size >= min_size
            is_too_big = current_size >= target_size * 1.5
            is_last = i == len(sentences) - 1
            
            if (is_break and is_big_enough) or is_too_big or is_last:
                chunk_text = " ".join(current_chunk)
                chunk = {
                    'text': chunk_text,
                    'chunk_index': chunk_idx,
                    'num_sentences': len(current_chunk)
                }
                if metadata:
                    chunk.update(metadata)
                chunks.append(chunk)
                
                chunk_idx += 1
                current_chunk = []
                current_size = 0
        
        return chunks


class HierarchicalChunker(ChunkingStrategy):
    """Hierarchical parent + child chunking."""
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        parent_size = self.config["chunk_size"]
        child_size = self.config["child_size"]
        
        chunks = []
        chunk_idx = 0
        
        # Create parent chunks
        parent_start = 0
        parent_idx = 0
        
        while parent_start < len(text):
            parent_end = parent_start + parent_size
            parent_text = text[parent_start:parent_end]
            
            # Create child chunks within parent
            child_start = 0
            while child_start < len(parent_text):
                child_end = child_start + child_size
                child_text = parent_text[child_start:child_end]
                
                if child_text.strip():
                    chunk = {
                        'text': child_text,
                        'chunk_index': chunk_idx,
                        'parent_index': parent_idx,
                        'is_child': True
                    }
                    if metadata:
                        chunk.update(metadata)
                    chunks.append(chunk)
                    chunk_idx += 1
                
                child_start = child_end
            
            parent_start = parent_end
            parent_idx += 1
        
        return chunks


def get_chunker(strategy_config: Dict) -> ChunkingStrategy:
    """Factory function to create chunker based on strategy."""
    strategy_type = strategy_config["strategy_type"]
    
    if strategy_type == "fixed_character":
        return FixedCharacterChunker(strategy_config)
    elif strategy_type == "sentence_window":
        return SentenceWindowChunker(strategy_config)
    elif strategy_type == "semantic":
        return SemanticChunker(strategy_config)
    elif strategy_type == "hierarchical":
        return HierarchicalChunker(strategy_config)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def load_evaluation_data():
    """Load queries and ground truth."""
    queries_path = EVALUATION_DIR / "eval_queries.json"
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    
    gt_path = EVALUATION_DIR / "ground_truth.json"
    with open(gt_path, 'r') as f:
        ground_truth = json.load(f)
    
    return queries, ground_truth


def create_chunk_distribution_plot(chunk_sizes: List[int], strategy_name: str) -> plt.Figure:
    """Create histogram of chunk sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(chunk_sizes, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(chunk_sizes), color='red', linestyle='--', 
               label=f'Mean: {np.mean(chunk_sizes):.0f} chars')
    ax.axvline(np.median(chunk_sizes), color='green', linestyle='--',
               label=f'Median: {np.median(chunk_sizes):.0f} chars')
    
    ax.set_xlabel('Chunk Size (characters)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Chunk Size Distribution - {strategy_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def evaluate_chunking_strategy(
    strategy_config: Dict,
    queries: List[Dict],
    ground_truth: Dict
) -> Tuple[Dict, Dict]:
    """
    Evaluate a chunking strategy.
    
    Args:
        strategy_config: Strategy configuration
        queries: Evaluation queries
        ground_truth: Ground truth chunks
        
    Returns:
        Tuple of (metrics, artifacts)
    """
    strategy_name = strategy_config["name"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {strategy_name}")
    logger.info(f"{'='*60}")
    
    # Initialize components
    embedder = Embedder()
    vector_store = VectorStore()
    vector_store.reset_collection()
    
    # Get chunker
    chunker = get_chunker(strategy_config)
    
    # Chunk all ground truth documents
    logger.info("Chunking documents...")
    all_chunks = []
    chunk_sizes = []
    chunk_id_mapping = {}  # Maps new_chunk_id -> original_chunk_id

    for chunk_id, chunk_data in ground_truth.items():
        text = chunk_data['text']
        metadata = {
            'original_chunk_id': chunk_id,
            'source': chunk_data.get('source', 'unknown'),
            'page': chunk_data.get('page', 0),
            'domain': chunk_data.get('domain', 'unknown'),
            'topic': chunk_data.get('topic', 'unknown')
        }

        chunks = chunker.chunk_text(text, metadata)

        # Map each new chunk back to its original chunk
        for chunk in chunks:
            new_chunk_id = chunk.get('chunk_id', chunk.get('chunk_index', len(all_chunks)))
            chunk_id_mapping[str(new_chunk_id)] = chunk_id
            # Store original chunk ID directly in chunk (not nested under 'metadata')
            chunk['original_chunk_id'] = chunk_id

        all_chunks.extend(chunks)
        chunk_sizes.extend([len(c['text']) for c in chunks])

    logger.info(f"Created {len(all_chunks)} chunks from {len(ground_truth)} original chunks")
    logger.info(f"Average chunk size: {np.mean(chunk_sizes):.2f} chars")
    logger.info(f"Chunk size std: {np.std(chunk_sizes):.2f} chars")
    logger.info(f"Chunk ID mapping created: {len(chunk_id_mapping)} mappings")

    # Add embeddings
    chunks_with_embeddings = embedder.embed_chunks(all_chunks)

    # Add to vector store
    vector_store.add_documents(chunks_with_embeddings)
    
    # Evaluate retrieval quality
    logger.info("\n--- Evaluating Retrieval Quality ---")
    recall_scores = []
    precision_scores = []
    ndcg_scores = []
    mrr_scores = []
    answer_quality_scores = []
    faithfulness_scores = []

    for idx, query_data in enumerate(queries[:10]):  # Limit to 10 queries for speed
        query_text = query_data["query"]
        relevant_chunk_ids = set(query_data["relevant_chunks"])

        # Retrieve
        query_embedding = embedder.embed_text(query_text)
        results = vector_store.query(query_embedding.tolist(), top_k=5)

        # Extract retrieved original chunk IDs
        # CRITICAL FIX: Map new chunk IDs back to original chunk IDs
        retrieved_original_ids = []
        retrieved_texts = []
        if results['ids'] and results['ids'][0]:
            for i, new_chunk_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i]
                # Get original chunk ID from metadata (stored during chunking)
                original_id = metadata.get('original_chunk_id', '')
                if original_id:
                    retrieved_original_ids.append(original_id)
                else:
                    # Fallback: try to get from mapping
                    original_id = chunk_id_mapping.get(str(new_chunk_id), '')
                    if original_id:
                        retrieved_original_ids.append(original_id)

                retrieved_texts.append(results['documents'][0][i])

        # Debug logging for first query
        if idx == 0:
            logger.info(f"DEBUG Query 1: {query_text[:100]}")
            logger.info(f"DEBUG Expected: {list(relevant_chunk_ids)[:3]}")
            logger.info(f"DEBUG Retrieved: {retrieved_original_ids}")
            matches = set(retrieved_original_ids) & relevant_chunk_ids
            logger.info(f"DEBUG Matches: {list(matches)}")

        # Compute retrieval metrics using ORIGINAL chunk IDs
        recall = compute_recall_at_k(retrieved_original_ids, list(relevant_chunk_ids), k=5)
        precision = compute_precision_at_k(retrieved_original_ids, list(relevant_chunk_ids), k=5)
        ndcg = compute_ndcg_at_k(retrieved_original_ids, list(relevant_chunk_ids), k=5)
        mrr = compute_mrr(retrieved_original_ids, list(relevant_chunk_ids))

        recall_scores.append(recall)
        precision_scores.append(precision)
        ndcg_scores.append(ndcg)
        mrr_scores.append(mrr)

        # Evaluate answer quality (simulated: based on retrieved context)
        context = "\n\n".join(retrieved_texts[:3])

        # Simple quality check: does context contain key terms from query?
        quality = evaluate_answer_quality(context, context=context)
        faithfulness = evaluate_faithfulness(context, context)

        answer_quality_scores.append(quality['basic_quality_score'])
        faithfulness_scores.append(faithfulness)
    
    # Compute chunk size statistics using utility function
    chunk_texts = [chunk['text'] for chunk in all_chunks]
    size_stats = compute_chunk_size_statistics(chunk_texts)

    # Compute chunk coherence (simplified - using embeddings of chunks)
    chunk_embeddings_array = np.array([chunk['embedding'] for chunk in chunks_with_embeddings])
    # For coherence, we'd ideally compare sentence embeddings within vs across chunks
    # For now, we'll use a simplified metric based on chunk embedding similarity
    if len(chunk_embeddings_array) > 1:
        chunk_coherence = 1.0 - compute_diversity(chunk_embeddings_array)  # Inverse of diversity
    else:
        chunk_coherence = 1.0

    # Compute metrics
    metrics = {
        # Retrieval quality metrics
        "precision_at_5": np.mean(precision_scores),
        "precision_at_5_std": np.std(precision_scores),
        "ndcg_at_5": np.mean(ndcg_scores),
        "ndcg_at_5_std": np.std(ndcg_scores),
        "recall_at_5": np.mean(recall_scores),
        "recall_at_5_std": np.std(recall_scores),
        "mrr": np.mean(mrr_scores),
        "mrr_std": np.std(mrr_scores),

        # Chunk statistics
        "total_chunks": size_stats['num_chunks'],
        "avg_chunk_size_chars": size_stats['mean_chars'],
        "median_chunk_size_chars": size_stats['median_chars'],
        "std_chunk_size_chars": size_stats['std_chars'],
        "min_chunk_size_chars": size_stats['min_chars'],
        "max_chunk_size_chars": size_stats['max_chars'],
        "avg_chunk_size_tokens": size_stats['mean_tokens'],
        "median_chunk_size_tokens": size_stats['median_tokens'],
        "std_chunk_size_tokens": size_stats['std_tokens'],
        "min_chunk_size_tokens": size_stats['min_tokens'],
        "max_chunk_size_tokens": size_stats['max_tokens'],

        # Chunk quality metrics
        "chunk_coherence": chunk_coherence,

        # Answer quality metrics
        "answer_quality": np.mean(answer_quality_scores),
        "answer_quality_std": np.std(answer_quality_scores),
        "faithfulness": np.mean(faithfulness_scores),
        "faithfulness_std": np.std(faithfulness_scores),

        "num_queries_evaluated": len(recall_scores)
    }

    logger.info(f"\n--- Results Summary ---")
    logger.info(f"Precision@5: {metrics['precision_at_5']:.4f} ± {metrics['precision_at_5_std']:.4f}")
    logger.info(f"NDCG@5: {metrics['ndcg_at_5']:.4f} ± {metrics['ndcg_at_5_std']:.4f}")
    logger.info(f"Recall@5: {metrics['recall_at_5']:.4f} ± {metrics['recall_at_5_std']:.4f}")
    logger.info(f"MRR: {metrics['mrr']:.4f}")
    logger.info(f"Chunk Coherence: {metrics['chunk_coherence']:.4f}")
    logger.info(f"Answer Quality: {metrics['answer_quality']:.3f} ± {metrics['answer_quality_std']:.3f}")
    logger.info(f"Total Chunks: {metrics['total_chunks']}")
    logger.info(f"Avg Chunk Size: {metrics['avg_chunk_size_chars']:.0f} chars ({metrics['avg_chunk_size_tokens']:.0f} tokens)")
    
    # Create artifacts
    distribution_plot = create_chunk_distribution_plot(chunk_sizes, strategy_name)
    
    # Sample chunks
    sample_chunks = [
        {
            'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
            'size': len(chunk['text']),
            'index': chunk['chunk_index']
        }
        for chunk in all_chunks[:5]
    ]
    
    artifacts = {
        "chunk_distribution": distribution_plot,
        "sample_chunks": sample_chunks,
        "chunk_statistics": {
            "sizes": chunk_sizes,
            "total": len(all_chunks),
            "mean": float(np.mean(chunk_sizes)),
            "std": float(np.std(chunk_sizes))
        }
    }
    
    # Cleanup
    vector_store.reset_collection()
    
    return metrics, artifacts


def run_all_experiments(test_mode: bool = False):
    """
    Run experiments for all chunking strategies.
    
    Args:
        test_mode: If True, only test first 2 strategies
    """
    logger.info("="*80)
    logger.info("CHUNKING STRATEGY EXPERIMENTS")
    logger.info("="*80)
    
    # Load evaluation data
    logger.info("\nLoading evaluation data...")
    queries, ground_truth = load_evaluation_data()
    logger.info(f"Loaded {len(queries)} queries and {len(ground_truth)} ground truth chunks")
    
    # Strategies to test
    strategies_to_test = CHUNKING_STRATEGIES[:2] if test_mode else CHUNKING_STRATEGIES
    
    # Run experiments
    all_results = []
    
    for strategy_config in strategies_to_test:
        experiment_name = "chunking_experiments"
        run_name = f"chunking_{strategy_config['name']}"
        
        with MLflowExperiment(experiment_name, run_name) as exp:
            # Log parameters
            params = {
                "chunking_strategy": strategy_config["name"],
                "description": strategy_config["description"],
                "chunk_size": strategy_config["chunk_size"],
                "chunk_overlap": strategy_config.get("chunk_overlap", 0),
                "strategy_type": strategy_config["strategy_type"]
            }
            
            if "child_size" in strategy_config:
                params["child_size"] = strategy_config["child_size"]
            
            exp.log_params(params)
            
            try:
                # Run evaluation
                metrics, artifacts = evaluate_chunking_strategy(
                    strategy_config, queries, ground_truth
                )
                
                # Log metrics
                exp.log_metrics(metrics)
                
                # Log artifacts
                exp.log_artifact("chunk_distribution.png", artifacts["chunk_distribution"])
                exp.log_artifact("sample_chunks.json", artifacts["sample_chunks"])
                exp.log_artifact("chunk_statistics.json", artifacts["chunk_statistics"])
                
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
    
    print("\n{:<30} {:<12} {:<12} {:<15} {:<12}".format(
        "Strategy", "Recall@5", "Quality", "Avg Chunk Size", "Total Chunks"
    ))
    print("-" * 85)
    
    for result in all_results:
        strategy_name = result["strategy"]
        metrics = result["metrics"]
        print("{:<30} {:<12.4f} {:<12.3f} {:<15.0f} {:<12}".format(
            strategy_name,
            metrics["recall_at_5"],
            metrics["answer_quality"],
            metrics["avg_chunk_size"],
            metrics["total_chunks"]
        ))
    
    logger.info("\n✓ All experiments completed!")
    logger.info(f"\nView results: mlflow ui")
    logger.info(f"Then navigate to: http://localhost:5000")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run chunking strategy experiments")
    parser.add_argument("--test-mode", action="store_true", 
                       help="Test mode: only run first 2 strategies")
    args = parser.parse_args()
    
    run_all_experiments(test_mode=args.test_mode)
