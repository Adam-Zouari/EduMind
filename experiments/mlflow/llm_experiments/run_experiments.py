"""
LLM Model Experiments

Tests different small LLMs for answer generation quality and efficiency.

Models tested:
1. Qwen 3 1.7B
2. Gemma 3 1B  
3. Llama 3.2 1B

Evaluation rubric (1-5 scale):
- Correctness: Is the answer factually correct?
- Completeness: Does it address all parts of the question?
- Faithfulness: Is it grounded in the provided context?
"""


# Add parent directory to path

# Configure MLflow database backend
from experiments.mlflow.mlflow_config import EVALUATION_DIR, configure_mlflow

configure_mlflow()

import json
import logging
import time

import numpy as np
import requests

from experiments.mlflow.utils import (
    MLflowExperiment,
    evaluate_answer_quality,
    evaluate_completeness,
    evaluate_conciseness,
    evaluate_context_precision,
    evaluate_correctness,
    evaluate_faithfulness,
    get_gpu_memory_usage,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# LLM models to test
LLM_MODELS = [
    {
        "name": "qwen3:1.7b",
        "description": "Qwen 3 1.7B - Alibaba's efficient LLM"
    },
    {
        "name": "gemma3:1b",
        "description": "Gemma 3 1B - Google's compact model"
    },
    {
        "name": "llama3.2:1b",
        "description": "Llama 3.2 1B - Meta's latest small model"
    }
]

# Ollama API configuration
OLLAMA_BASE_URL = "http://localhost:11434"


def check_ollama_available():
    """Check if Ollama server is running."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def check_model_available(model_name: str) -> bool:
    """Check if a specific model is available in Ollama."""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            available_names = [m["name"] for m in models]
            return model_name in available_names
    except:
        pass
    return False


def generate_answer(model_name: str, query: str, context: str, temperature: float = 0.7, max_tokens: int = 512):
    """
    Generate answer using Ollama.
    
    Args:
        model_name: Model identifier
        query: User question
        context: Retrieved context
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        tuple: (answer, latency_ms, tokens_per_sec)
    """
    system_prompt = """You are a helpful AI assistant. Answer the question based on the provided context.
Be concise, accurate, and faithful to the context. If the context doesn't contain enough information, say so."""
    
    prompt = f"""Context:
{context}

Question: {query}

Answer:"""
    
    payload = {
        "model": model_name,
        "prompt": prompt,
        "system": system_prompt,
        "temperature": temperature,
        "num_predict": max_tokens,
        "stream": False
    }
    
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            answer = result.get("response", "")
            
            # Calculate tokens per second
            # Note: Ollama doesn't always return eval_count, so we estimate
            total_duration_ns = result.get("total_duration", latency_ms * 1_000_000)
            total_duration_s = total_duration_ns / 1_000_000_000
            
            # Estimate token count (rough: ~4 chars per token)
            estimated_tokens = len(answer) / 4
            tokens_per_sec = estimated_tokens / total_duration_s if total_duration_s > 0 else 0
            
            return answer, latency_ms, tokens_per_sec
        else:
            logger.error(f"Ollama API error: {response.status_code}")
            return "", latency_ms, 0
    
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        latency_ms = (time.time() - start_time) * 1000
        return "", latency_ms, 0


def load_evaluation_data():
    """Load queries and ground truth."""
    queries_path = EVALUATION_DIR / "eval_queries.json"
    with open(queries_path) as f:
        queries = json.load(f)
    
    gt_path = EVALUATION_DIR / "ground_truth.json"
    with open(gt_path) as f:
        ground_truth = json.load(f)
    
    return queries, ground_truth


def evaluate_llm_model(model_config: dict, queries: list[dict], ground_truth: dict, num_queries: int = 5):
    """
    Evaluate a single LLM model.
    
    Args:
        model_config: Model configuration
        queries: List of evaluation queries
        ground_truth: Ground truth chunks
        num_queries: Number of queries to evaluate (use fewer for faster testing)
        
    Returns:
        Dictionary of metrics and artifacts
    """
    model_name = model_config["name"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {model_name}")
    logger.info(f"{'='*60}")
    
    # Check model availability
    if not check_model_available(model_name):
        logger.warning(f"Model {model_name} not found in Ollama!")
        logger.warning(f"Pull it with: ollama pull {model_name}")
        return None, None
    
    # Limit queries for faster evaluation
    eval_queries = queries[:num_queries]
    logger.info(f"Evaluating on {len(eval_queries)} queries")
    
    # Results
    all_answers = []
    latencies = []
    throughputs = []
    quality_scores = []
    faithfulness_scores = []
    correctness_scores = []
    completeness_scores = []
    conciseness_scores = []
    context_precision_scores = []

    # Get initial GPU memory
    initial_gpu = get_gpu_memory_usage()

    for idx, query_data in enumerate(eval_queries):
        query = query_data["query"]
        relevant_chunk_ids = query_data["relevant_chunks"]

        # Get context from ground truth (simulate retrieval)
        context_texts = [ground_truth[cid]["text"] for cid in relevant_chunk_ids[:3]]
        context = "\n\n".join(context_texts)

        logger.info(f"\n[Query {idx+1}/{len(eval_queries)}] {query[:50]}...")

        # Generate answer
        answer, latency_ms, tokens_per_sec = generate_answer(
            model_name, query, context, temperature=0.3, max_tokens=256
        )

        if not answer:
            logger.warning("Empty answer generated")
            continue

        logger.info(f"Answer: {answer[:100]}...")
        logger.info(f"Latency: {latency_ms:.2f}ms | Tokens/sec: {tokens_per_sec:.2f}")

        # Evaluate quality
        quality = evaluate_answer_quality(answer, context=context)
        faithfulness = evaluate_faithfulness(answer, context)

        # NEW: Evaluate correctness (using context as reference since we don't have ground truth answers)
        # In production, you'd use LLM-as-judge or human evaluation
        reference_answer = query_data.get("reference_answer", context[:200])  # Fallback to context snippet
        correctness = evaluate_correctness(answer, reference_answer)

        # NEW: Evaluate completeness (does answer cover key points from context?)
        completeness = evaluate_completeness(answer, context[:300])  # Use first part of context

        # NEW: Evaluate conciseness
        conciseness = evaluate_conciseness(answer, reference_answer=None)  # Use absolute heuristic

        # NEW: Evaluate context precision (which contexts were actually used?)
        context_precision_result = evaluate_context_precision(answer, context_texts)
        context_precision = context_precision_result['context_precision']

        # Store results
        all_answers.append({
            "query": query,
            "answer": answer,
            "latency_ms": latency_ms,
            "tokens_per_sec": tokens_per_sec,
            "quality": quality,
            "faithfulness": faithfulness,
            "correctness": correctness,
            "completeness": completeness,
            "conciseness": conciseness,
            "context_precision": context_precision,
            "contexts_used": context_precision_result['contexts_used'],
            "contexts_provided": context_precision_result['contexts_provided']
        })

        latencies.append(latency_ms)
        throughputs.append(tokens_per_sec)
        quality_scores.append(quality["basic_quality_score"])
        faithfulness_scores.append(faithfulness)
        correctness_scores.append(correctness)
        completeness_scores.append(completeness)
        conciseness_scores.append(conciseness)
        context_precision_scores.append(context_precision)
    
    if not all_answers:
        logger.error("No valid answers generated!")
        return None, None

    # Get final GPU memory
    final_gpu = get_gpu_memory_usage()
    vram_usage_mb = final_gpu.get('allocated_mb', 0)

    # Compute aggregated metrics
    metrics = {
        # Performance metrics
        "tokens_per_sec": np.mean(throughputs),
        "tokens_per_sec_std": np.std(throughputs),
        "latency_sec": np.mean(latencies) / 1000,  # Convert to seconds
        "latency_ms": np.mean(latencies),
        "latency_std_ms": np.std(latencies),
        "vram_usage_mb": vram_usage_mb,

        # Answer quality metrics (PRIMARY)
        "correctness": np.mean(correctness_scores),
        "correctness_std": np.std(correctness_scores),
        "completeness": np.mean(completeness_scores),
        "completeness_std": np.std(completeness_scores),
        "conciseness": np.mean(conciseness_scores),
        "conciseness_std": np.std(conciseness_scores),
        "faithfulness_score": np.mean(faithfulness_scores),
        "faithfulness_std": np.std(faithfulness_scores),

        # Context utilization metrics
        "context_precision": np.mean(context_precision_scores),
        "context_precision_std": np.std(context_precision_scores),

        # Legacy metrics
        "answer_quality": np.mean(quality_scores),
        "answer_quality_std": np.std(quality_scores),

        # Dataset info
        "num_queries_evaluated": len(all_answers),
        "avg_response_length_words": np.mean([len(a["answer"].split()) for a in all_answers])
    }

    logger.info("\n--- Results Summary ---")
    logger.info(f"Correctness: {metrics['correctness']:.3f} ± {metrics['correctness_std']:.3f}")
    logger.info(f"Completeness: {metrics['completeness']:.3f} ± {metrics['completeness_std']:.3f}")
    logger.info(f"Conciseness: {metrics['conciseness']:.3f} ± {metrics['conciseness_std']:.3f}")
    logger.info(f"Faithfulness: {metrics['faithfulness_score']:.3f} ± {metrics['faithfulness_std']:.3f}")
    logger.info(f"Context Precision: {metrics['context_precision']:.3f} ± {metrics['context_precision_std']:.3f}")
    logger.info(f"Latency: {metrics['latency_ms']:.2f} ± {metrics['latency_std_ms']:.2f} ms")
    logger.info(f"Throughput: {metrics['tokens_per_sec']:.2f} ± {metrics['tokens_per_sec_std']:.2f} tokens/sec")
    logger.info(f"VRAM: {metrics['vram_usage_mb']:.2f} MB")
    
    # Prepare artifacts
    artifacts = {
        "answers": all_answers,
        "prompt_template": """System: You are a helpful AI assistant. Answer the question based on the provided context.
Be concise, accurate, and faithful to the context.

Context: {context}
Question: {query}
Answer:"""
    }
    
    return metrics, artifacts


def run_all_experiments(test_mode: bool = False, num_queries: int = 5):
    """
    Run experiments for all LLM models.
    
    Args:
        test_mode: If True, only test first model
        num_queries: Number of queries to evaluate per model
    """
    logger.info("="*80)
    logger.info("LLM MODEL EXPERIMENTS")
    logger.info("="*80)
    
    # Check Ollama availability
    if not check_ollama_available():
        logger.error("❌ Ollama is not running!")
        logger.error("Start it with: ollama serve")
        return
    
    logger.info("✓ Ollama server is running")
    
    # Load evaluation data
    logger.info("\nLoading evaluation data...")
    queries, ground_truth = load_evaluation_data()
    logger.info(f"Loaded {len(queries)} queries and {len(ground_truth)} ground truth chunks")
    
    # Models to test
    models_to_test = LLM_MODELS[:1] if test_mode else LLM_MODELS
    
    # Run experiments
    all_results = []
    
    for model_config in models_to_test:
        experiment_name = "llm_experiments"
        run_name = f"llm_{model_config['name'].replace(':', '_').replace('.', '_')}"
        
        with MLflowExperiment(experiment_name, run_name) as exp:
            # Log parameters
            params = {
                "model_name": model_config["name"],
                "description": model_config["description"],
                "temperature": 0.3,
                "max_tokens": 256,
                "system_prompt_id": "instructional_rag",
                "num_queries": num_queries
            }
            exp.log_params(params)
            
            try:
                # Run evaluation
                metrics, artifacts = evaluate_llm_model(model_config, queries, ground_truth, num_queries)
                
                if metrics is None:
                    logger.error(f"Skipping {model_config['name']} - evaluation failed")
                    continue
                
                # Log metrics
                exp.log_metrics(metrics)
                
                # Log artifacts
                exp.log_artifact("answers.json", artifacts["answers"])
                exp.log_artifact("prompt_template.txt", artifacts["prompt_template"])
                
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
    if all_results:
        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*80)
        
        print("\n{:<20} {:<15} {:<15} {:<15} {:<15}".format(
            "Model", "Quality", "Faithfulness", "Latency (ms)", "Tokens/sec"
        ))
        print("-" * 80)
        
        for result in all_results:
            model_name = result["model"]
            metrics = result["metrics"]
            print("{:<20} {:<15.3f} {:<15.3f} {:<15.2f} {:<15.2f}".format(
                model_name,
                metrics["answer_quality"],
                metrics["faithfulness_score"],
                metrics["latency_ms"],
                metrics["tokens_per_sec"]
            ))
        
        logger.info("\n✓ All experiments completed!")
        logger.info("\nView results: mlflow ui")
        logger.info("Then navigate to: http://localhost:5000")
    else:
        logger.warning("\nNo experiments completed successfully")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM model experiments")
    parser.add_argument("--test-mode", action="store_true", help="Test mode: only run first model")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of queries to evaluate")
    args = parser.parse_args()
    
    run_all_experiments(test_mode=args.test_mode, num_queries=args.num_queries)
