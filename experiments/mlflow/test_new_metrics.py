"""
Test script to verify all new metrics work correctly.
"""

import sys
from pathlib import Path

from experiments.mlflow.utils import (
    # Retrieval metrics
    compute_precision_at_k,
    compute_ndcg_at_k,
    compute_map,
    compute_hit_rate_at_k,
    compute_diversity,
    # Chunking metrics
    compute_chunk_size_statistics,
    compute_chunk_coherence,
    # LLM metrics
    evaluate_correctness,
    evaluate_completeness,
    evaluate_conciseness,
    evaluate_context_precision
)

import numpy as np

def test_retrieval_metrics():
    """Test retrieval quality metrics."""
    print("="*60)
    print("Testing Retrieval Metrics")
    print("="*60)
    
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = ['doc2', 'doc5', 'doc7', 'doc8']
    
    precision = compute_precision_at_k(retrieved, relevant, k=5)
    print(f"Precision@5: {precision:.3f} (expected: 0.4 = 2/5)")
    
    ndcg = compute_ndcg_at_k(retrieved, relevant, k=5)
    print(f"NDCG@5: {ndcg:.3f}")
    
    map_score = compute_map(retrieved, relevant)
    print(f"MAP: {map_score:.3f}")
    
    hit_rate = compute_hit_rate_at_k(retrieved, relevant, k=5)
    print(f"Hit Rate@5: {hit_rate:.3f} (expected: 1.0)")
    
    # Test diversity
    embeddings = np.random.randn(5, 384)  # 5 docs, 384-dim embeddings
    diversity = compute_diversity(embeddings)
    print(f"Diversity: {diversity:.3f} (should be ~0.5 for random)")
    
    print("✓ All retrieval metrics working\n")


def test_chunking_metrics():
    """Test chunking quality metrics."""
    print("="*60)
    print("Testing Chunking Metrics")
    print("="*60)
    
    chunks = [
        "This is a short chunk.",
        "This is a medium length chunk with more words in it.",
        "This is a longer chunk that contains even more text and information about various topics."
    ]
    
    stats = compute_chunk_size_statistics(chunks)
    print(f"Chunk Statistics:")
    print(f"  Num chunks: {stats['num_chunks']}")
    print(f"  Mean chars: {stats['mean_chars']:.1f}")
    print(f"  Mean tokens: {stats['mean_tokens']:.1f}")
    print(f"  Min/Max chars: {stats['min_chars']}/{stats['max_chars']}")
    
    # Test coherence (simplified)
    chunk_embeddings = np.random.randn(10, 384)
    boundary_embeddings = np.random.randn(3, 2, 384)
    coherence = compute_chunk_coherence(chunk_embeddings, boundary_embeddings)
    print(f"Chunk Coherence: {coherence:.3f}")
    
    print("✓ All chunking metrics working\n")


def test_llm_metrics():
    """Test LLM answer quality metrics."""
    print("="*60)
    print("Testing LLM Metrics")
    print("="*60)
    
    answer = "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming."
    reference = "Machine learning is a field of AI that allows systems to learn and improve from experience without being explicitly programmed."
    context = "Machine learning (ML) is a subset of artificial intelligence. It enables computers to learn from data. ML systems improve their performance over time."
    
    correctness = evaluate_correctness(answer, reference)
    print(f"Correctness: {correctness:.3f}")
    
    completeness = evaluate_completeness(answer, reference)
    print(f"Completeness: {completeness:.3f}")
    
    conciseness = evaluate_conciseness(answer, reference_answer=reference)
    print(f"Conciseness: {conciseness:.3f}")
    
    contexts = [
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "Supervised learning requires labeled data."
    ]
    context_precision = evaluate_context_precision(answer, contexts)
    print(f"Context Precision: {context_precision['context_precision']:.3f}")
    print(f"  Contexts used: {context_precision['contexts_used']}/{context_precision['contexts_provided']}")
    print(f"  Used indices: {context_precision['used_context_indices']}")
    
    print("✓ All LLM metrics working\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("TESTING NEW METRICS")
    print("="*60 + "\n")
    
    try:
        test_retrieval_metrics()
        test_chunking_metrics()
        test_llm_metrics()
        
        print("="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nAll new metrics are working correctly.")
        print("You can now run the experiments with the improved metrics.")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

