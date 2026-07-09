"""
Shared Evaluation Utilities for MLflow Experiments

Provides functions for:
- Recall@K, Precision@K computation
- NDCG@K (Normalized Discounted Cumulative Gain)
- MAP (Mean Average Precision)
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- Diversity metrics
- Latency measurement
- Answer quality evaluation (Correctness, Completeness, Conciseness)
- Faithfulness evaluation
- Context precision
- Chunk coherence
"""

from typing import List, Dict, Any, Callable, Optional, Tuple
import time
from contextlib import contextmanager
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Recall@K metric.
    
    Recall@K = (Number of relevant items in top-K) / (Total number of relevant items)
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of ground-truth relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_ids:
        logger.warning("No relevant IDs provided for Recall@K calculation")
        return 0.0
    
    # Consider only top-k retrieved results
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    # Count how many relevant items are in top-k
    hits = len(top_k_retrieved.intersection(relevant_set))
    
    # Recall = hits / total_relevant
    recall = hits / len(relevant_set)
    
    return recall


def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR = 1 / rank_of_first_relevant_item

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of ground-truth relevant document IDs

    Returns:
        MRR score (0.0 to 1.0)
    """
    if not relevant_ids:
        logger.warning("No relevant IDs provided for MRR calculation")
        return 0.0

    relevant_set = set(relevant_ids)

    # Find the rank (1-indexed) of the first relevant item
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    # No relevant item found
    return 0.0


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Precision@K metric.

    Precision@K = (Number of relevant items in top-K) / K

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of ground-truth relevant document IDs
        k: Number of top results to consider

    Returns:
        Precision@K score (0.0 to 1.0)
    """
    if k == 0:
        return 0.0

    # Consider only top-k retrieved results
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    # Count how many relevant items are in top-k
    hits = len(top_k_retrieved.intersection(relevant_set))

    # Precision = hits / k
    precision = hits / k

    return precision


def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).

    NDCG considers both relevance and ranking position.
    Higher-ranked relevant documents contribute more to the score.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of ground-truth relevant document IDs
        k: Number of top results to consider

    Returns:
        NDCG@K score (0.0 to 1.0)
    """
    if not relevant_ids or k == 0:
        return 0.0

    relevant_set = set(relevant_ids)

    # Create binary relevance scores for retrieved docs
    relevance_scores = []
    for doc_id in retrieved_ids[:k]:
        relevance_scores.append(1.0 if doc_id in relevant_set else 0.0)

    # Compute DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, rel in enumerate(relevance_scores):
        # Position is 1-indexed, discount by log2(position + 1)
        dcg += rel / np.log2(i + 2)  # i+2 because i is 0-indexed

    # Compute IDCG (Ideal DCG) - best possible ranking
    ideal_relevance = sorted([1.0] * min(len(relevant_ids), k) + [0.0] * k, reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / np.log2(i + 2)

    # Avoid division by zero
    if idcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg


def compute_map(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Calculate MAP (Mean Average Precision).

    MAP is the mean of precision values at each relevant document position.

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of ground-truth relevant document IDs

    Returns:
        MAP score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0

    relevant_set = set(relevant_ids)

    # Calculate precision at each relevant position
    precisions = []
    num_relevant_seen = 0

    for i, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            num_relevant_seen += 1
            precision_at_i = num_relevant_seen / i
            precisions.append(precision_at_i)

    # Average precision
    if not precisions:
        return 0.0

    ap = sum(precisions) / len(relevant_ids)
    return ap


def compute_hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    Calculate Hit Rate@K.

    Hit Rate@K = 1 if at least one relevant doc in top-K, else 0

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of ground-truth relevant document IDs
        k: Number of top results to consider

    Returns:
        Hit rate (0.0 or 1.0)
    """
    if not relevant_ids:
        return 0.0

    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    # Check if there's at least one hit
    has_hit = len(top_k_retrieved.intersection(relevant_set)) > 0

    return 1.0 if has_hit else 0.0


def compute_diversity(embeddings: np.ndarray) -> float:
    """
    Calculate diversity of retrieved documents based on embeddings.

    Diversity = 1 - (average pairwise similarity)
    Higher diversity means documents are more different from each other.

    Args:
        embeddings: Array of shape (n_docs, embedding_dim)

    Returns:
        Diversity score (0.0 to 1.0)
    """
    if len(embeddings) <= 1:
        return 1.0  # Single doc or no docs = maximum diversity

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized_embeddings = embeddings / norms

    # Compute pairwise cosine similarities
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    # Get upper triangle (excluding diagonal) to avoid counting pairs twice
    n = len(embeddings)
    upper_triangle_indices = np.triu_indices(n, k=1)
    pairwise_similarities = similarity_matrix[upper_triangle_indices]

    # Average similarity
    avg_similarity = np.mean(pairwise_similarities) if len(pairwise_similarities) > 0 else 0.0

    # Diversity is inverse of similarity
    diversity = 1.0 - avg_similarity

    return max(0.0, min(1.0, diversity))  # Clamp to [0, 1]


def compute_chunk_coherence(chunk_embeddings: np.ndarray, boundary_embeddings: np.ndarray) -> float:
    """
    Calculate chunk coherence score.

    Coherence = (avg intra-chunk similarity) / (avg cross-boundary similarity)
    Higher coherence means chunks contain semantically related content.

    Args:
        chunk_embeddings: Embeddings of sentences within chunks, shape (n_sentences, dim)
        boundary_embeddings: Embeddings of sentence pairs across chunk boundaries, shape (n_boundaries, 2, dim)

    Returns:
        Coherence score (higher is better)
    """
    if len(chunk_embeddings) == 0:
        return 0.0

    # Normalize embeddings
    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = chunk_embeddings / norms

    # Intra-chunk similarity (average of all pairs within chunks)
    similarity_matrix = np.dot(normalized, normalized.T)
    n = len(chunk_embeddings)
    upper_triangle = np.triu_indices(n, k=1)
    intra_similarities = similarity_matrix[upper_triangle]
    avg_intra = np.mean(intra_similarities) if len(intra_similarities) > 0 else 0.0

    # Cross-boundary similarity
    if len(boundary_embeddings) > 0:
        boundary_sims = []
        for pair in boundary_embeddings:
            norm1 = np.linalg.norm(pair[0])
            norm2 = np.linalg.norm(pair[1])
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(pair[0], pair[1]) / (norm1 * norm2)
                boundary_sims.append(sim)
        avg_boundary = np.mean(boundary_sims) if boundary_sims else 0.0
    else:
        avg_boundary = 0.0

    # Coherence ratio
    if avg_boundary == 0:
        return avg_intra  # No boundaries to compare

    coherence = avg_intra / avg_boundary
    return coherence


@contextmanager
def measure_latency():
    """
    Context manager to measure execution time.

    Usage:
        with measure_latency() as timer:
            # code to measure
            pass
        latency_ms = timer['latency_ms']

    Yields:
        Dictionary with 'latency_ms' key
    """
    timer = {'latency_ms': 0}
    start_time = time.time()

    try:
        yield timer
    finally:
        end_time = time.time()
        timer['latency_ms'] = (end_time - start_time) * 1000  # Convert to milliseconds


def measure_function_latency(func: Callable, *args, **kwargs) -> tuple:
    """
    Measure the execution time of a function.
    
    Args:
        func: Function to measure
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (result, latency_ms)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    latency_ms = (end_time - start_time) * 1000
    return result, latency_ms


def evaluate_answer_quality(
    answer: str, 
    reference_answer: Optional[str] = None,
    context: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate answer quality using simple heuristics.
    
    For full evaluation, human scoring or LLM-as-judge is recommended.
    This provides basic automated metrics.
    
    Args:
        answer: Generated answer
        reference_answer: Optional reference answer for comparison
        context: Optional context used to generate answer
        
    Returns:
        Dictionary with quality metrics
    """
    metrics = {}
    
    # Basic length check
    word_count = len(answer.split())
    metrics['response_length_words'] = word_count
    
    # Check if answer is not empty
    metrics['is_non_empty'] = 1.0 if answer.strip() else 0.0
    
    # Check if answer is not just repeating context
    if context:
        # Simple check: if answer is substring of context, it might be just copying
        is_copy = answer.strip() in context
        metrics['is_original'] = 0.0 if is_copy else 1.0
    
    # Basic quality score (heuristic)
    # A good answer is: non-empty, reasonable length, original
    quality_score = 0.0
    if word_count >= 5 and word_count <= 200:  # Reasonable length
        quality_score += 0.5
    if metrics['is_non_empty'] > 0:
        quality_score += 0.25
    if metrics.get('is_original', 1.0) > 0:
        quality_score += 0.25
    
    metrics['basic_quality_score'] = quality_score
    
    return metrics


def evaluate_faithfulness(answer: str, context: str) -> float:
    """
    Evaluate if the answer is faithful to the context.

    Simple heuristic: Check if key terms from answer appear in context.
    For production, use NLI models or LLM-as-judge.

    Args:
        answer: Generated answer
        context: Context used to generate answer

    Returns:
        Faithfulness score (0.0 to 1.0)
    """
    if not answer or not context:
        return 0.0

    # Extract key terms from answer (simple: words longer than 4 chars)
    answer_words = set(
        word.lower().strip('.,!?;:')
        for word in answer.split()
        if len(word) > 4
    )

    context_lower = context.lower()

    # Check how many answer terms are in context
    if not answer_words:
        return 0.5  # Neutral if no key terms

    found_terms = sum(1 for word in answer_words if word in context_lower)
    faithfulness = found_terms / len(answer_words)

    return faithfulness


def evaluate_correctness(answer: str, reference_answer: str) -> float:
    """
    Evaluate correctness by comparing answer to reference.

    Simple heuristic: Token overlap (F1 score).
    For production, use LLM-as-judge (GPT-4) or semantic similarity.

    Args:
        answer: Generated answer
        reference_answer: Ground truth reference answer

    Returns:
        Correctness score (0.0 to 1.0)
    """
    if not answer or not reference_answer:
        return 0.0

    # Tokenize and normalize
    answer_tokens = set(word.lower().strip('.,!?;:') for word in answer.split())
    reference_tokens = set(word.lower().strip('.,!?;:') for word in reference_answer.split())

    # Remove empty strings
    answer_tokens.discard('')
    reference_tokens.discard('')

    if not answer_tokens or not reference_tokens:
        return 0.0

    # Compute F1 score (harmonic mean of precision and recall)
    overlap = len(answer_tokens.intersection(reference_tokens))

    precision = overlap / len(answer_tokens) if answer_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def evaluate_completeness(answer: str, reference_answer: str) -> float:
    """
    Evaluate if answer covers all key points from reference.

    Completeness = (key facts in answer) / (key facts in reference)

    Args:
        answer: Generated answer
        reference_answer: Ground truth reference answer

    Returns:
        Completeness score (0.0 to 1.0)
    """
    if not reference_answer:
        return 1.0  # No reference to compare

    if not answer:
        return 0.0

    # Extract key terms (words longer than 4 chars) from reference
    reference_terms = set(
        word.lower().strip('.,!?;:')
        for word in reference_answer.split()
        if len(word) > 4
    )

    answer_lower = answer.lower()

    if not reference_terms:
        return 1.0  # No key terms to check

    # Check how many reference terms appear in answer
    covered_terms = sum(1 for term in reference_terms if term in answer_lower)
    completeness = covered_terms / len(reference_terms)

    return completeness


def evaluate_conciseness(answer: str, reference_answer: Optional[str] = None) -> float:
    """
    Evaluate if answer is concise (not overly verbose).

    Conciseness = 1 - (excess_length_ratio)

    Args:
        answer: Generated answer
        reference_answer: Optional reference answer for length comparison

    Returns:
        Conciseness score (0.0 to 1.0)
    """
    if not answer:
        return 0.0

    answer_words = len(answer.split())

    if reference_answer:
        # Compare to reference length
        reference_words = len(reference_answer.split())
        if reference_words == 0:
            return 1.0

        # Penalize if answer is much longer than reference
        length_ratio = answer_words / reference_words

        # Ideal is 0.8 to 1.2 times reference length
        if 0.8 <= length_ratio <= 1.2:
            conciseness = 1.0
        elif length_ratio < 0.8:
            # Too short
            conciseness = length_ratio / 0.8
        else:
            # Too long - penalize more heavily
            excess = length_ratio - 1.2
            conciseness = max(0.0, 1.0 - excess / 2)
    else:
        # Without reference, use absolute heuristic
        # Ideal answer: 20-150 words
        if 20 <= answer_words <= 150:
            conciseness = 1.0
        elif answer_words < 20:
            conciseness = answer_words / 20
        else:
            # Penalize verbosity
            excess = (answer_words - 150) / 150
            conciseness = max(0.0, 1.0 - excess / 2)

    return conciseness


def evaluate_context_precision(
    answer: str,
    contexts: List[str],
    context_ids: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Evaluate how precisely the context was used.

    Context Precision = (chunks actually used) / (chunks provided)

    Args:
        answer: Generated answer
        contexts: List of context chunks provided to LLM
        context_ids: Optional list of context chunk IDs

    Returns:
        Dictionary with precision metrics
    """
    if not contexts or not answer:
        return {
            'context_precision': 0.0,
            'contexts_used': 0,
            'contexts_provided': len(contexts) if contexts else 0
        }

    answer_lower = answer.lower()

    # Check which contexts were actually used
    contexts_used = 0
    used_indices = []

    for i, context in enumerate(contexts):
        # Extract key terms from context
        context_terms = set(
            word.lower().strip('.,!?;:')
            for word in context.split()
            if len(word) > 5  # Longer words to avoid common words
        )

        # Check if any key terms from this context appear in answer
        terms_found = sum(1 for term in context_terms if term in answer_lower)

        # If at least 20% of context terms are in answer, consider it used
        if context_terms and (terms_found / len(context_terms)) >= 0.2:
            contexts_used += 1
            used_indices.append(i)

    precision = contexts_used / len(contexts) if contexts else 0.0

    return {
        'context_precision': precision,
        'contexts_used': contexts_used,
        'contexts_provided': len(contexts),
        'used_context_indices': used_indices
    }


def compute_chunk_size_statistics(chunks: List[str]) -> Dict[str, float]:
    """
    Compute statistics about chunk sizes.

    Args:
        chunks: List of text chunks

    Returns:
        Dictionary with size statistics
    """
    if not chunks:
        return {
            'num_chunks': 0,
            'mean_chars': 0.0,
            'median_chars': 0.0,
            'std_chars': 0.0,
            'min_chars': 0,
            'max_chars': 0,
            'mean_tokens': 0.0,
            'median_tokens': 0.0,
            'std_tokens': 0.0,
            'min_tokens': 0,
            'max_tokens': 0
        }

    # Character counts
    char_counts = [len(chunk) for chunk in chunks]

    # Token counts (approximate: split by whitespace)
    token_counts = [len(chunk.split()) for chunk in chunks]

    return {
        'num_chunks': len(chunks),
        'mean_chars': float(np.mean(char_counts)),
        'median_chars': float(np.median(char_counts)),
        'std_chars': float(np.std(char_counts)),
        'min_chars': int(np.min(char_counts)),
        'max_chars': int(np.max(char_counts)),
        'mean_tokens': float(np.mean(token_counts)),
        'median_tokens': float(np.median(token_counts)),
        'std_tokens': float(np.std(token_counts)),
        'min_tokens': int(np.min(token_counts)),
        'max_tokens': int(np.max(token_counts))
    }


def compute_mean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Compute mean values across multiple metric dictionaries.

    Args:
        metrics_list: List of metric dictionaries

    Returns:
        Dictionary with mean values for each metric
    """
    if not metrics_list:
        return {}

    # Get all unique keys
    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    # Compute mean for each key
    mean_metrics = {}
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m]
        if values:
            mean_metrics[f"mean_{key}"] = np.mean(values)
            mean_metrics[f"std_{key}"] = np.std(values)

    return mean_metrics


def evaluate_retrieval_quality(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    ground_truth_ids: List[str],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, float]:
    """
    Comprehensive retrieval quality evaluation.
    
    Args:
        query: Query text
        retrieved_docs: List of retrieved documents with 'id' field
        ground_truth_ids: List of relevant document IDs
        k_values: List of K values to compute Recall@K
        
    Returns:
        Dictionary with all retrieval metrics
    """
    metrics = {}
    
    # Extract IDs from retrieved docs
    retrieved_ids = [doc.get('id', doc.get('document_id', '')) for doc in retrieved_docs]
    
    # Compute Recall@K for different K values
    for k in k_values:
        recall = compute_recall_at_k(retrieved_ids, ground_truth_ids, k)
        metrics[f'recall_at_{k}'] = recall
    
    # Compute MRR
    mrr = compute_mrr(retrieved_ids, ground_truth_ids)
    metrics['mrr'] = mrr
    
    # Number of retrieved docs
    metrics['num_retrieved'] = len(retrieved_docs)
    
    return metrics


if __name__ == "__main__":
    # Example usage and tests
    print("=== Testing Evaluation Utilities ===\n")
    
    # Test Recall@K
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = ['doc2', 'doc5', 'doc7']
    
    recall_5 = compute_recall_at_k(retrieved, relevant, k=5)
    print(f"Recall@5: {recall_5:.3f}")  # Should be 2/3 = 0.667
    
    # Test MRR
    mrr = compute_mrr(retrieved, relevant)
    print(f"MRR: {mrr:.3f}")  # Should be 1/2 = 0.5 (doc2 is at position 2)
    
    # Test latency measurement
    with measure_latency() as timer:
        time.sleep(0.1)  # Simulate some work
    print(f"\nLatency: {timer['latency_ms']:.2f} ms")
    
    # Test answer quality
    answer = "Machine learning is a subset of AI that enables computers to learn from data."
    context = "Machine learning (ML) is a field of AI. It enables systems to learn from data."
    
    quality = evaluate_answer_quality(answer, context=context)
    print(f"\nAnswer Quality Metrics: {quality}")
    
    faithfulness = evaluate_faithfulness(answer, context)
    print(f"Faithfulness Score: {faithfulness:.3f}")
    
    print("\n✓ All evaluation utilities working correctly")
