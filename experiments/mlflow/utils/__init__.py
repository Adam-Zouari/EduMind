"""Lazy exports for MLflow experiment utilities."""

from __future__ import annotations

from importlib import import_module

_EVALUATION_EXPORTS = {
    "compute_recall_at_k",
    "compute_precision_at_k",
    "compute_ndcg_at_k",
    "compute_map",
    "compute_hit_rate_at_k",
    "compute_mrr",
    "compute_diversity",
    "compute_chunk_coherence",
    "compute_chunk_size_statistics",
    "evaluate_answer_quality",
    "evaluate_faithfulness",
    "evaluate_correctness",
    "evaluate_completeness",
    "evaluate_conciseness",
    "evaluate_context_precision",
    "measure_latency",
    "measure_function_latency",
    "compute_mean_metrics",
    "evaluate_retrieval_quality",
}

_FIXTURE_EXPORTS = {
    "EvaluationDocument",
    "EvaluationQuery",
    "build_chunk_record",
    "build_reference_chunk_records",
    "index_documents_by_id",
    "load_evaluation_dataset",
    "load_evaluation_documents",
    "load_evaluation_queries",
    "resolve_query_relevant_ids",
}

_GPU_EXPORTS = {
    "is_cuda_available",
    "get_gpu_memory_usage",
    "get_gpu_utilization",
    "get_gpu_info",
    "monitor_gpu_during_execution",
    "measure_throughput",
    "reset_peak_memory_stats",
    "get_peak_memory_stats",
}

_LOGGER_EXPORTS = {
    "set_experiment",
    "start_run",
    "end_run",
    "log_params",
    "log_metrics",
    "log_dict_as_json",
    "log_text_as_artifact",
    "log_numpy_array",
    "log_figure",
    "log_experiment_results",
    "create_comparison_plot",
    "MLflowExperiment",
}

__all__ = sorted(_EVALUATION_EXPORTS | _FIXTURE_EXPORTS | _GPU_EXPORTS | _LOGGER_EXPORTS)


def __getattr__(name: str):
    if name in _EVALUATION_EXPORTS:
        return getattr(import_module("experiments.mlflow.utils.evaluation"), name)
    if name in _FIXTURE_EXPORTS:
        return getattr(import_module("experiments.mlflow.utils.fixtures"), name)
    if name in _GPU_EXPORTS:
        return getattr(import_module("experiments.mlflow.utils.gpu_utils"), name)
    if name in _LOGGER_EXPORTS:
        return getattr(import_module("experiments.mlflow.utils.metrics_logger"), name)
    raise AttributeError(name)
