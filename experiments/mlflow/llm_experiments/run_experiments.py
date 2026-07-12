"""LLM experiments aligned with the current Ollama client and evaluation fixtures."""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np

from edumind.rag.errors import OllamaConnectionError, OllamaRequestError
from edumind.rag.llm_generator import OllamaGenerator
from edumind.rag.types import LLMSettings
from experiments.mlflow.mlflow_config import configure_mlflow
from experiments.mlflow.utils import (
    MLflowExperiment,
    evaluate_answer_quality,
    evaluate_completeness,
    evaluate_conciseness,
    evaluate_context_precision,
    evaluate_correctness,
    evaluate_faithfulness,
    get_gpu_memory_usage,
    load_evaluation_dataset,
    resolve_query_relevant_ids,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMExperiment:
    """One maintained Ollama model comparison target."""

    model_name: str
    description: str


LLM_MODELS = (
    LLMExperiment("qwen3:1.7b", "Qwen 3 1.7B"),
    LLMExperiment("gemma3:1b", "Gemma 3 1B"),
    LLMExperiment("llama3.2:1b", "Llama 3.2 1B"),
)


def evaluate_llm_model(
    experiment: LLMExperiment,
    *,
    test_mode: bool,
    num_queries: int,
) -> tuple[dict[str, float], dict[str, object]] | None:
    """Evaluate one Ollama model on a small fixture-backed question set."""
    queries, documents = load_evaluation_dataset()
    documents_by_id = {document.id: document for document in documents}
    evaluation_queries = queries[: min(num_queries, 2 if test_mode else num_queries)]

    generator = OllamaGenerator(
        settings=LLMSettings(
            model_name=experiment.model_name,
            base_url="http://localhost:11434",
            temperature=0.3,
            max_tokens=256,
            request_timeout=120,
        )
    )
    if not generator.health_check():
        logger.warning("Skipping %s because Ollama is unavailable.", experiment.model_name)
        return None

    available_models = set(generator.list_models())
    if experiment.model_name not in available_models:
        logger.warning("Skipping %s because the model is not installed.", experiment.model_name)
        return None

    all_answers: list[dict[str, object]] = []
    latencies: list[float] = []
    throughput_scores: list[float] = []
    correctness_scores: list[float] = []
    completeness_scores: list[float] = []
    conciseness_scores: list[float] = []
    faithfulness_scores: list[float] = []
    context_precision_scores: list[float] = []
    answer_quality_scores: list[float] = []

    for query in evaluation_queries:
        relevant_ids = resolve_query_relevant_ids(query, documents_by_id)
        contexts = [
            documents_by_id[chunk_id].text
            for chunk_id in relevant_ids[:3]
            if chunk_id in documents_by_id
        ]
        context = "\n\n".join(contexts)
        if not context:
            continue

        start_time = time.perf_counter()
        try:
            answer = generator.generate(query.query, context)
        except (OllamaConnectionError, OllamaRequestError) as exc:
            logger.warning("Skipping %s after Ollama failure: %s", experiment.model_name, exc)
            return None
        latency_ms = (time.perf_counter() - start_time) * 1000
        if not answer:
            continue

        reference_answer = context[:400]
        answer_quality = evaluate_answer_quality(answer, context=context)
        faithfulness = evaluate_faithfulness(answer, context)
        correctness = evaluate_correctness(answer, reference_answer)
        completeness = evaluate_completeness(answer, reference_answer)
        conciseness = evaluate_conciseness(answer)
        context_precision_result = evaluate_context_precision(answer, contexts)
        estimated_tokens = max(1, len(answer.split()))
        throughput = estimated_tokens / (latency_ms / 1000) if latency_ms > 0 else 0.0

        latencies.append(latency_ms)
        throughput_scores.append(throughput)
        correctness_scores.append(correctness)
        completeness_scores.append(completeness)
        conciseness_scores.append(conciseness)
        faithfulness_scores.append(faithfulness)
        context_precision_scores.append(float(context_precision_result["context_precision"]))
        answer_quality_scores.append(answer_quality["basic_quality_score"])
        all_answers.append(
            {
                "query": query.query,
                "relevant_chunk_ids": relevant_ids,
                "answer": answer,
                "latency_ms": latency_ms,
                "tokens_per_sec": throughput,
                "correctness": correctness,
                "completeness": completeness,
                "conciseness": conciseness,
                "faithfulness": faithfulness,
                "context_precision": context_precision_result["context_precision"],
            }
        )

    if not all_answers:
        return None

    gpu_metrics = get_gpu_memory_usage()
    metrics = {
        "tokens_per_sec": float(np.mean(throughput_scores)),
        "tokens_per_sec_std": float(np.std(throughput_scores)),
        "latency_sec": float(np.mean(latencies) / 1000),
        "latency_ms": float(np.mean(latencies)),
        "latency_std_ms": float(np.std(latencies)),
        "vram_usage_mb": float(gpu_metrics.get("allocated_mb", 0.0)),
        "correctness": float(np.mean(correctness_scores)),
        "correctness_std": float(np.std(correctness_scores)),
        "completeness": float(np.mean(completeness_scores)),
        "completeness_std": float(np.std(completeness_scores)),
        "conciseness": float(np.mean(conciseness_scores)),
        "conciseness_std": float(np.std(conciseness_scores)),
        "faithfulness_score": float(np.mean(faithfulness_scores)),
        "faithfulness_std": float(np.std(faithfulness_scores)),
        "context_precision": float(np.mean(context_precision_scores)),
        "context_precision_std": float(np.std(context_precision_scores)),
        "answer_quality": float(np.mean(answer_quality_scores)),
        "answer_quality_std": float(np.std(answer_quality_scores)),
        "num_queries_evaluated": float(len(all_answers)),
        "avg_response_length_words": float(
            np.mean([len(item["answer"].split()) for item in all_answers])
        ),
    }
    artifacts = {
        "answers.json": all_answers,
        "prompt_template.txt": (
            "System: answer the question using only the provided context.\n\n"
            "Context: {context}\nQuestion: {query}\nAnswer:"
        ),
    }
    return metrics, artifacts


def run_all_experiments(test_mode: bool = False, num_queries: int = 5) -> int:
    """Run all maintained Ollama experiments."""
    configure_mlflow(verbose=False)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    models = LLM_MODELS[:1] if test_mode else LLM_MODELS
    for experiment in models:
        run_name = experiment.model_name.replace(":", "_").replace(".", "_")
        with MLflowExperiment("llm_experiments", f"llm_{run_name}") as run:
            run.log_params(
                {
                    "model_name": experiment.model_name,
                    "description": experiment.description,
                    "temperature": 0.3,
                    "max_tokens": 256,
                    "num_queries": num_queries,
                    "test_mode": test_mode,
                }
            )
            result = evaluate_llm_model(experiment, test_mode=test_mode, num_queries=num_queries)
            if result is None:
                logger.info("Skipped %s", experiment.model_name)
                continue

            metrics, artifacts = result
            run.log_metrics(metrics)
            for filename, content in artifacts.items():
                run.log_artifact(filename, content)
            logger.info(
                "Completed %s: correctness=%.3f faithfulness=%.3f latency=%.2f ms",
                experiment.model_name,
                metrics["correctness"],
                metrics["faithfulness_score"],
                metrics["latency_ms"],
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the LLM-experiment CLI parser."""
    parser = argparse.ArgumentParser(description="Run maintained Ollama experiments.")
    parser.add_argument("--test-mode", action="store_true", help="Run only the first model.")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of queries to evaluate.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for maintained Ollama experiments."""
    args = build_parser().parse_args(argv)
    return run_all_experiments(test_mode=args.test_mode, num_queries=args.num_queries)


if __name__ == "__main__":
    raise SystemExit(main())
