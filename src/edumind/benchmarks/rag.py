"""Chunking/embedding, retrieval, generation, and complete-RAG benchmarks."""

from __future__ import annotations

import hashlib
import math
import random
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from edumind.common.config import load_settings
from edumind.common.paths import PROJECT_ROOT
from edumind.rag.contracts import GenerationProfile, embedding_spec
from edumind.rag.embedder import Embedder
from edumind.rag.llm_generator import OllamaGenerator
from edumind.rag.retrieval import (
    BM25Ranker,
    CrossEncoderReranker,
    build_reranker,
    reciprocal_rank_fusion,
)
from edumind.rag.text_chunker import build_chunking_strategy
from edumind.rag.tokenizers import HuggingFaceOffsetTokenizer, RegexOffsetTokenizer
from edumind.rag.types import RetrievalHit

from .contracts import BenchmarkPlan, BenchmarkResult, SampleResult
from .datasets import DatasetManifest, load_manifest
from .harness import BenchmarkHarness
from .metrics import (
    citation_scores,
    context_precision_at_k,
    context_recall,
    exact_match,
    hit_rate_at_k,
    interval_overlap,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    rouge_l,
    token_f1,
)
from .prepare import load_model_lock
from .registry import candidates_for

CHUNKERS = ("recursive-character", "token-256-32", "token-384-64", "sentence-8-2", "semantic")
EMBEDDINGS = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
    "Qwen/Qwen3-Embedding-0.6B",
)
RETRIEVALS = ("dense", "bm25", "rrf", "rrf-minilm-reranker", "rrf-qwen3-reranker")
GENERATION_PROFILES = candidates_for("rag", "generation")
FINAL_RETRIEVALS = ("dense", "rrf", "rrf-minilm-reranker")
FINAL_GENERATORS = GENERATION_PROFILES[:3]


class HashEmbedder:
    """Deterministic smoke-only fake; never eligible for quality recommendations."""

    def __init__(self, salt: str, dimension: int = 192) -> None:
        self.salt = salt
        self.dimension = dimension

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray([self._vector(text) for text in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"\w+", text.casefold())
        for token in tokens:
            digest = hashlib.sha256(f"{self.salt}:{token}".encode()).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            vector[index] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


class ProductionEmbedderAdapter:
    def __init__(self, name: str, revision: str) -> None:
        self.runtime = Embedder(
            spec=embedding_spec(
                name,
                document_device="cpu",
                query_device="cpu",
                revision=revision,
            )
        )

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.runtime.embed_texts(texts)

    def encode_query(self, text: str) -> np.ndarray:
        return self.runtime.embed_query(text)


@dataclass(frozen=True)
class CorpusChunk:
    id: str
    document_id: str
    text: str
    start: int
    end: int
    tokens: int


def run_chunking_embedding(
    profile: str,
    *,
    artifact_root: Path | None = None,
    manifest_path: Path | None = None,
) -> BenchmarkResult:
    settings = load_settings()
    manifest = load_manifest(manifest_path or _rag_manifest(profile))
    candidates = tuple(f"{chunker}|{embedding}" for chunker in CHUNKERS for embedding in EMBEDDINGS)
    plan = BenchmarkPlan(
        "rag",
        "chunking-embedding",
        profile,
        manifest.name,
        candidates,
        bootstrap_resamples=_resamples(profile, settings.benchmark.bootstrap_resamples),
    )
    harness = BenchmarkHarness(
        artifact_root or settings.benchmark.artifact_directory,
        tracking_uri=settings.benchmark.tracking_uri,
    )

    def evaluate(candidate: str):
        chunker_name, embedding_name = candidate.split("|", 1)
        return _evaluate_retrieval_candidate(
            manifest, chunker_name, embedding_name, "dense", profile
        )

    return harness.run(
        plan,
        evaluate,
        dataset_checksum=manifest.checksum,
        directions={
            "ndcg_at_3": "max",
            "ndcg_at_5": "max",
            "context_recall_at_3": "max",
            "context_recall_at_5": "max",
            "context_precision_at_3": "max",
            "context_precision_at_5": "max",
            "context_recall_at_2048_tokens": "max",
            "operational.p95_latency_seconds": "min",
            "operational.storage_bytes": "min",
        },
        model_revisions={name: _model_revision(name, profile) for name in EMBEDDINGS},
    )


def run_retrieval(
    profile: str,
    *,
    artifact_root: Path | None = None,
    manifest_path: Path | None = None,
) -> BenchmarkResult:
    settings = load_settings()
    manifest = load_manifest(manifest_path or _rag_manifest(profile))
    plan = BenchmarkPlan(
        "rag",
        "retrieval",
        profile,
        manifest.name,
        RETRIEVALS,
        bootstrap_resamples=_resamples(profile, settings.benchmark.bootstrap_resamples),
    )
    harness = BenchmarkHarness(
        artifact_root or settings.benchmark.artifact_directory,
        tracking_uri=settings.benchmark.tracking_uri,
    )

    def evaluate(candidate: str):
        return _evaluate_retrieval_candidate(
            manifest, "token-384-64", "sentence-transformers/all-MiniLM-L6-v2", candidate, profile
        )

    return harness.run(
        plan,
        evaluate,
        dataset_checksum=manifest.checksum,
        directions={
            "ndcg_at_3": "max",
            "ndcg_at_5": "max",
            "context_recall_at_3": "max",
            "context_recall_at_5": "max",
            "context_precision_at_3": "max",
            "context_precision_at_5": "max",
            "context_recall_at_2048_tokens": "max",
            "operational.p95_latency_seconds": "min",
            "operational.storage_bytes": "min",
        },
    )


def run_generation(
    profile: str,
    *,
    artifact_root: Path | None = None,
    manifest_path: Path | None = None,
    _stage: str = "generation",
    _candidates: tuple[str, ...] | None = None,
) -> BenchmarkResult:
    settings = load_settings()
    manifest = load_manifest(manifest_path or _rag_manifest(profile))
    questions = _stratified_questions(manifest.samples, 24, seed=42)
    candidates = _candidates or GENERATION_PROFILES
    plan = BenchmarkPlan(
        "rag",
        _stage,
        profile,
        manifest.name,
        candidates,
        bootstrap_resamples=_resamples(profile, settings.benchmark.bootstrap_resamples),
        warmups=2,
    )
    harness = BenchmarkHarness(
        artifact_root or settings.benchmark.artifact_directory,
        tracking_uri=settings.benchmark.tracking_uri,
    )
    documents = {
        str(item["id"]): str(item["text"])
        for item in manifest.samples
        if item.get("kind") == "document"
    }
    class_counts = {
        value: sum(bool(question.get("answerable")) is value for question in questions)
        for value in (False, True)
    }
    retrieval_cache: dict[tuple[str, int], dict[str, list[RetrievalHit]]] = {}

    def evaluate(candidate: str):
        generation_candidate = candidate
        retrieval_name: str | None = None
        final_top_k = 1
        if _stage == "final":
            retrieval_name, generation_candidate, top_k_value = candidate.split("|", 2)
            final_top_k = int(top_k_value.removeprefix("top_k="))
            cache_key = (retrieval_name, final_top_k)
            if cache_key not in retrieval_cache:
                retrieval_cache[cache_key] = _build_final_contexts(
                    manifest, retrieval_name, final_top_k, profile
                )
        generator = None if profile == "smoke" else _ollama_generator(generation_candidate)
        samples: list[SampleResult] = []
        latencies: list[float] = []
        first_token_latencies: list[float] = []
        prompt_evaluation_times: list[float] = []
        generation_times: list[float] = []
        answer_token_counts: list[float] = []
        reasoning_token_counts: list[float] = []
        token_rates: list[float] = []
        cold_load_seconds = 0.0

        def context_for(question: Mapping[str, object]) -> tuple[list[RetrievalHit], str]:
            if retrieval_name is None:
                context = documents[str(question["document_id"])]
                return [
                    RetrievalHit(
                        "frozen-1",
                        context,
                        {"source": question["document_id"]},
                        1.0,
                        1,
                        "frozen",
                        len(context.split()),
                    )
                ], context
            hits = retrieval_cache[(retrieval_name, final_top_k)][str(question["id"])]
            return hits, "\n\n".join(hit.document for hit in hits)

        if generator is not None and questions:
            warmup_hits, _ = context_for(questions[0])
            generator.unload()
            cold = generator.generate_measured_with_results(
                str(questions[0]["question"]), warmup_hits
            )
            cold_load_seconds = cold.load_seconds
            for _ in range(plan.warmups):
                generator.generate_measured_with_results(str(questions[0]["question"]), warmup_hits)
        for question in questions:
            started = time.perf_counter()
            answerable = bool(question.get("answerable"))
            context_hits, context = context_for(question)
            if generator is None:
                answer = (
                    f"{question['answer']} [1]"
                    if answerable
                    else "I don't have enough evidence to answer."
                )
                latency = time.perf_counter() - started
            else:
                measurement = generator.generate_measured_with_results(
                    str(question["question"]), context_hits
                )
                answer = measurement.answer
                latency = measurement.total_seconds
                first_token_latencies.append(measurement.time_to_first_token_seconds)
                prompt_evaluation_times.append(measurement.prompt_evaluation_seconds)
                generation_times.append(measurement.generation_seconds)
                answer_token_counts.append(float(measurement.answer_tokens))
                reasoning_token_counts.append(float(measurement.reasoning_tokens_estimate))
                token_rates.append(measurement.tokens_per_second)
            latencies.append(latency)
            predicted_answerable = "don't have enough evidence" not in answer.casefold()
            class_weight = len(questions) / (2 * class_counts[answerable])
            clean_answer = re.sub(r"\[\d+\]", "", answer).strip()
            supported_contexts = (
                _supported_context_ids(question, context_hits) if answerable else set()
            )
            citation = citation_scores(answer, supported_contexts)
            samples.append(
                SampleResult(
                    str(question["id"]),
                    {
                        "exact_match": exact_match(clean_answer, str(question.get("answer", "")))
                        if answerable
                        else float(not predicted_answerable),
                        "token_f1": token_f1(clean_answer, str(question.get("answer", "")))
                        if answerable
                        else float(not predicted_answerable),
                        "rouge_l": rouge_l(clean_answer, str(question.get("answer", "")))
                        if answerable
                        else float(not predicted_answerable),
                        "citation_precision": citation["citation_precision"],
                        "citation_recall": citation["citation_recall"],
                        "citation_f1": citation["citation_f1"],
                        "answerability_balanced_accuracy": float(predicted_answerable == answerable)
                        * class_weight,
                        "unsupported_answer_rate": float(not answerable and predicted_answerable),
                        "malformed_output_rate": float(not answer.strip()),
                    },
                    latency,
                    {
                        "answerable": answerable,
                        "profile": generation_candidate,
                        "retrieval": retrieval_name or "frozen",
                        "top_k": final_top_k if retrieval_name else None,
                        "question": question["question"],
                        "reference_answer": question.get("answer", ""),
                        "generated_answer": answer,
                        "frozen_context": context,
                    },
                )
            )
        if generator is not None:
            generator.unload()
        return samples, {
            "p50_latency_seconds": float(np.median(latencies)),
            "p95_latency_seconds": float(np.quantile(latencies, 0.95)),
            "p50_time_to_first_token_seconds": _median_or_zero(first_token_latencies),
            "p95_time_to_first_token_seconds": _quantile_or_zero(first_token_latencies, 0.95),
            "mean_prompt_evaluation_seconds": _mean_or_zero(prompt_evaluation_times),
            "mean_generation_seconds": _mean_or_zero(generation_times),
            "mean_answer_tokens": _mean_or_zero(answer_token_counts),
            "mean_reasoning_tokens_estimate": _mean_or_zero(reasoning_token_counts),
            "tokens_per_second": _mean_or_zero(token_rates),
            "answers_per_minute": 60.0 / float(np.mean(latencies)) if latencies else 0.0,
            "cold_load_seconds": cold_load_seconds,
            "peak_process_memory_gb": 0.0,
        }

    return harness.run(
        plan,
        evaluate,
        dataset_checksum=manifest.checksum,
        directions={
            "citation_f1": "max",
            "answerability_balanced_accuracy": "max",
            "operational.p95_latency_seconds": "min",
        },
        hard_gates={
            "malformed_output_rate": ("min", 0.0),
            "operational.p95_latency_seconds": ("min", 30.0),
            "operational.peak_process_memory_gb": ("min", 28.0),
        },
    )


def run_final(
    profile: str,
    *,
    artifact_root: Path | None = None,
    manifest_path: Path | None = None,
) -> BenchmarkResult:
    """Run retrieval, context packing, and generation as one complete system."""
    candidates = tuple(
        f"{retrieval}|{generator}|top_k={top_k}"
        for retrieval in FINAL_RETRIEVALS
        for generator in FINAL_GENERATORS
        for top_k in (3, 5)
    )
    return run_generation(
        profile,
        artifact_root=artifact_root,
        manifest_path=manifest_path,
        _stage="final",
        _candidates=candidates,
    )


def _evaluate_retrieval_candidate(
    manifest: DatasetManifest,
    chunker_name: str,
    embedding_name: str,
    retrieval_name: str,
    profile: str,
) -> tuple[list[SampleResult], Mapping[str, float]]:
    embedder = (
        HashEmbedder(embedding_name)
        if profile == "smoke"
        else ProductionEmbedderAdapter(embedding_name, _model_revision(embedding_name, profile))
    )
    tokenizer = (
        RegexOffsetTokenizer()
        if profile == "smoke"
        else HuggingFaceOffsetTokenizer(
            embedding_spec(embedding_name).tokenizer,
            revision=_model_revision(embedding_name, profile),
        )
    )
    strategy = build_chunking_strategy(
        chunker_name,
        tokenizer=tokenizer,
        embed_sentences=embedder.encode_documents,
    )
    indexing_started = time.perf_counter()
    chunks: list[CorpusChunk] = []
    for document in (item for item in manifest.samples if item.get("kind") == "document"):
        text = str(document["text"])
        for index, (start, end, token_count) in enumerate(strategy.split(text)):
            chunks.append(
                CorpusChunk(
                    f"{document['id']}:{index}",
                    str(document["id"]),
                    text[start:end],
                    start,
                    end,
                    token_count,
                )
            )
    vectors = embedder.encode_documents([chunk.text for chunk in chunks])
    bm25 = None if profile == "smoke" else BM25Ranker([chunk.text for chunk in chunks])
    reranker = _benchmark_reranker(retrieval_name, profile)
    indexing_seconds = time.perf_counter() - indexing_started
    samples: list[SampleResult] = []
    latencies: list[float] = []
    retrieval_questions = [
        item
        for item in manifest.samples
        if item.get("kind") == "question" and item.get("answerable")
    ]
    random.Random(42).shuffle(retrieval_questions)
    for question in retrieval_questions:
        started = time.perf_counter()
        query = str(question["question"])
        query_vector = embedder.encode_query(query)
        dense_scores = vectors @ query_vector
        dense_order = list(np.argsort(-dense_scores)[:20])
        if bm25 is None:
            lexical_scores = np.asarray([_lexical_score(query, chunk.text) for chunk in chunks])
            lexical_order = list(np.argsort(-lexical_scores)[:20])
        else:
            lexical_ranking = bm25.rank(query, 20)
            lexical_order = [index for index, _ in lexical_ranking]
            lexical_scores = np.zeros(len(chunks), dtype=np.float64)
            for index, score in lexical_ranking:
                lexical_scores[index] = score
        if retrieval_name == "dense":
            order = dense_order
        elif retrieval_name == "bm25":
            order = lexical_order
        else:
            dense_hits = [
                _temporary_hit(chunks[index], rank, float(dense_scores[index]), "dense")
                for rank, index in enumerate(dense_order, 1)
            ]
            lexical_hits = [
                _temporary_hit(chunks[index], rank, float(lexical_scores[index]), "bm25")
                for rank, index in enumerate(lexical_order, 1)
            ]
            fused = reciprocal_rank_fusion([dense_hits, lexical_hits], limit=20)
            order = [
                next(index for index, chunk in enumerate(chunks) if chunk.id == hit.id)
                for hit in fused
            ]
            if reranker is not None:
                reranked = reranker.rerank(query, fused, 20)
                order = [
                    next(index for index, chunk in enumerate(chunks) if chunk.id == hit.id)
                    for hit in reranked
                ]
            elif retrieval_name.endswith("reranker"):
                order = sorted(
                    order, key=lambda index: _lexical_score(query, chunks[index].text), reverse=True
                )
        selected = [chunks[index] for index in order[:10]]
        latency = time.perf_counter() - started
        latencies.append(latency)
        raw_evidence = question.get("evidence", [])
        if not isinstance(raw_evidence, (list, tuple)):
            raise ValueError(f"Question {question['id']} has malformed evidence")
        evidence = [item for item in raw_evidence if isinstance(item, Mapping)]
        all_grades = [_chunk_grade(chunk, evidence) for chunk in chunks]
        relevant_total = sum(grade > 0 for grade in all_grades)
        grades = [_chunk_grade(chunk, evidence) for chunk in selected]
        metrics: dict[str, float] = {"mrr": reciprocal_rank(grades)}
        for k in (1, 3, 5, 10):
            current = selected[:k]
            intervals = [
                (chunk.start, chunk.end)
                for chunk in current
                if chunk.document_id == question["document_id"]
            ]
            gold = [(_evidence_int(item, "start"), _evidence_int(item, "end")) for item in evidence]
            metrics.update(
                {
                    f"precision_at_{k}": precision_at_k(grades, k),
                    f"recall_at_{k}": recall_at_k(grades, relevant_total, k),
                    f"hit_rate_at_{k}": hit_rate_at_k(grades, k),
                    f"context_precision_at_{k}": context_precision_at_k(grades, k),
                    f"context_recall_at_{k}": context_recall(gold, intervals),
                }
            )
            if k in {3, 5, 10}:
                metrics[f"map_at_{k}"] = _average_precision(grades, relevant_total, k)
            if k in {3, 5, 10}:
                metrics[f"ndcg_at_{k}"] = ndcg_at_k(grades, k, all_grades)
        budget_intervals: list[tuple[int, int]] = []
        token_total = 0
        for chunk in selected:
            remaining = 2048 - token_total
            if remaining <= 0:
                break
            if chunk.tokens <= remaining:
                if chunk.document_id == question["document_id"]:
                    budget_intervals.append((chunk.start, chunk.end))
                token_total += chunk.tokens
                continue
            truncated = tokenizer.truncate(chunk.text, remaining)
            if truncated and chunk.document_id == question["document_id"]:
                budget_intervals.append((chunk.start, chunk.start + len(truncated)))
            token_total += tokenizer.count(truncated)
            break
        metrics["context_recall_at_2048_tokens"] = context_recall(
            [(_evidence_int(item, "start"), _evidence_int(item, "end")) for item in evidence],
            budget_intervals,
        )
        samples.append(
            SampleResult(str(question["id"]), metrics, latency, {"retrieved_tokens": token_total})
        )
    return samples, {
        "indexing_seconds": indexing_seconds,
        "chunks": float(len(chunks)),
        "p50_latency_seconds": float(np.median(latencies)),
        "p95_latency_seconds": float(np.quantile(latencies, 0.95)),
        "storage_bytes": float(vectors.nbytes),
    }


def _build_final_contexts(
    manifest: DatasetManifest,
    retrieval_name: str,
    top_k: int,
    profile: str,
) -> dict[str, list[RetrievalHit]]:
    """Build final-system contexts through the benchmarked production contracts."""
    embedding_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedder = (
        HashEmbedder(embedding_name)
        if profile == "smoke"
        else ProductionEmbedderAdapter(embedding_name, _model_revision(embedding_name, profile))
    )
    tokenizer = (
        RegexOffsetTokenizer()
        if profile == "smoke"
        else HuggingFaceOffsetTokenizer(
            embedding_spec(embedding_name).tokenizer,
            revision=_model_revision(embedding_name, profile),
        )
    )
    strategy = build_chunking_strategy("token-384-64", tokenizer=tokenizer)
    chunks: list[CorpusChunk] = []
    for document in (item for item in manifest.samples if item.get("kind") == "document"):
        text = str(document["text"])
        for index, (start, end, token_count) in enumerate(strategy.split(text)):
            chunks.append(
                CorpusChunk(
                    f"{document['id']}:{index}",
                    str(document["id"]),
                    text[start:end],
                    start,
                    end,
                    token_count,
                )
            )
    vectors = embedder.encode_documents([chunk.text for chunk in chunks])
    bm25 = None if profile == "smoke" else BM25Ranker([chunk.text for chunk in chunks])
    reranker = _benchmark_reranker(retrieval_name, profile)
    contexts: dict[str, list[RetrievalHit]] = {}
    for question in (item for item in manifest.samples if item.get("kind") == "question"):
        query = str(question["question"])
        dense_scores = vectors @ embedder.encode_query(query)
        dense_order = list(np.argsort(-dense_scores)[:20])
        if bm25 is None:
            lexical_scores = np.asarray([_lexical_score(query, chunk.text) for chunk in chunks])
            lexical_order = list(np.argsort(-lexical_scores)[:20])
        else:
            lexical_ranking = bm25.rank(query, 20)
            lexical_order = [index for index, _ in lexical_ranking]
            lexical_scores = np.zeros(len(chunks), dtype=np.float64)
            for index, score in lexical_ranking:
                lexical_scores[index] = score
        dense_hits = [
            _temporary_hit(chunks[index], rank, float(dense_scores[index]), "dense")
            for rank, index in enumerate(dense_order, 1)
        ]
        lexical_hits = [
            _temporary_hit(chunks[index], rank, float(lexical_scores[index]), "bm25")
            for rank, index in enumerate(lexical_order, 1)
        ]
        if retrieval_name == "dense":
            ranked = dense_hits
        elif retrieval_name == "bm25":
            ranked = lexical_hits
        else:
            ranked = reciprocal_rank_fusion([dense_hits, lexical_hits], limit=20)
        if reranker is not None:
            ranked = reranker.rerank(query, ranked, 20)
        elif retrieval_name.endswith("reranker"):
            ranked = sorted(
                ranked,
                key=lambda hit: _lexical_score(query, hit.document),
                reverse=True,
            )
        packed: list[RetrievalHit] = []
        token_total = 0
        for hit in ranked:
            if len(packed) >= top_k:
                break
            remaining = 2048 - token_total
            if remaining <= 0:
                break
            if hit.token_count > remaining:
                truncated = tokenizer.truncate(hit.document, remaining)
                if truncated:
                    metadata = dict(hit.metadata)
                    metadata_start = metadata.get("start")
                    if isinstance(metadata_start, int):
                        metadata["end"] = metadata_start + len(truncated)
                    packed.append(
                        RetrievalHit(
                            hit.id,
                            truncated,
                            metadata,
                            hit.score,
                            len(packed) + 1,
                            hit.retrieval_method,
                            tokenizer.count(truncated),
                        )
                    )
                break
            token_total += hit.token_count
            packed.append(replace_hit_rank(hit, len(packed) + 1))
        contexts[str(question["id"])] = packed
    return contexts


def replace_hit_rank(hit: RetrievalHit, rank: int) -> RetrievalHit:
    return RetrievalHit(
        hit.id,
        hit.document,
        hit.metadata,
        hit.score,
        rank,
        hit.retrieval_method,
        hit.token_count,
    )


def _benchmark_reranker(retrieval_name: str, profile: str) -> CrossEncoderReranker | None:
    if profile == "smoke" or not retrieval_name.endswith("reranker"):
        return None
    model = (
        "Qwen/Qwen3-Reranker-0.6B"
        if retrieval_name == "rrf-qwen3-reranker"
        else "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    return build_reranker(
        retrieval_name,
        revision=_model_revision(model, profile),
        device="cpu",
    )


def _temporary_hit(chunk: CorpusChunk, rank: int, score: float, method: str) -> RetrievalHit:
    return RetrievalHit(
        chunk.id,
        chunk.text,
        {
            "document_id": chunk.document_id,
            "source": chunk.document_id,
            "start": chunk.start,
            "end": chunk.end,
        },
        score,
        rank,
        method,
        chunk.tokens,
    )


def _supported_context_ids(
    question: Mapping[str, object], hits: Sequence[RetrievalHit]
) -> set[int]:
    raw_evidence = question.get("evidence", [])
    if not isinstance(raw_evidence, (list, tuple)):
        return set()
    supported: set[int] = set()
    for rank, hit in enumerate(hits, 1):
        document_id = str(hit.metadata.get("document_id", hit.metadata.get("source", "")))
        start = hit.metadata.get("start")
        end = hit.metadata.get("end")
        if start is None or end is None:
            if document_id == str(question.get("document_id", "")):
                supported.add(rank)
            continue
        for evidence in raw_evidence:
            if not isinstance(evidence, Mapping):
                continue
            if document_id == str(evidence.get("document_id", "")) and interval_overlap(
                (_numeric_int(start, "hit start"), _numeric_int(end, "hit end")),
                (_evidence_int(evidence, "start"), _evidence_int(evidence, "end")),
            ):
                supported.add(rank)
                break
    return supported


def _chunk_grade(chunk: CorpusChunk, evidence: Sequence[Mapping[str, object]]) -> float:
    overlap = 0
    for item in evidence:
        if chunk.document_id != str(item["document_id"]):
            continue
        overlap += max(
            0,
            min(chunk.end, _evidence_int(item, "end"))
            - max(chunk.start, _evidence_int(item, "start")),
        )
    return min(1.0, overlap / max(1, chunk.end - chunk.start))


def _evidence_int(item: Mapping[str, object], key: str) -> int:
    return _numeric_int(item.get(key), f"evidence {key}")


def _numeric_int(value: object, label: str) -> int:
    if not isinstance(value, (str, int, float)):
        raise ValueError(f"{label} must be numeric")
    return int(value)


def _average_precision(grades: Sequence[float], total: int, k: int) -> float:
    from .metrics import average_precision_at_k

    return average_precision_at_k(grades, total, k)


def _lexical_score(query: str, document: str) -> float:
    query_tokens = set(re.findall(r"\w+", query.casefold()))
    document_tokens = set(re.findall(r"\w+", document.casefold()))
    return len(query_tokens & document_tokens) / math.sqrt(
        max(1, len(query_tokens) * len(document_tokens))
    )


def _ollama_generator(name: str) -> OllamaGenerator:
    settings = load_settings().generation
    model_map = {
        "qwen3:1.7b": "qwen3:1.7b",
        "qwen3.5:4b-direct": "qwen3.5:4b-q4_K_M",
        "qwen3.5:4b-thinking": "qwen3.5:4b-q4_K_M",
        "qwen3.5:9b-direct": "qwen3.5:9b-q4_K_M",
        "qwen3.5:9b-thinking": "qwen3.5:9b-q4_K_M",
        "gemma3:4b": "gemma3:4b",
        "gemma3:12b": "gemma3:12b",
        "ministral-3:8b-instruct-2512-q4_K_M": "ministral-3:8b-instruct-2512-q4_K_M",
        "gpt-oss:20b-low": "gpt-oss:20b",
        "gpt-oss:20b-medium": "gpt-oss:20b",
    }
    try:
        model = model_map[name]
    except KeyError as exc:
        raise ValueError(f"Unknown generation profile: {name}") from exc
    thinking = (
        "medium"
        if "medium" in name
        else "low"
        if "low" in name
        else "on"
        if "thinking" in name
        else "off"
    )
    digests = load_model_lock(PROJECT_ROOT / "data/benchmarks/models/ollama.json")
    try:
        digest = digests[model]
    except KeyError as exc:
        raise RuntimeError(f"Ollama model lock has no digest for {model}") from exc
    return OllamaGenerator(
        GenerationProfile(model, digest, thinking, 0.0, 42, 8192, 256),
        base_url=settings.base_url,
        timeout_seconds=settings.timeout_seconds,
    )


def _rag_manifest(profile: str) -> Path:
    if profile == "smoke":
        return PROJECT_ROOT / "data/benchmarks/rag/smoke.json"
    return (
        PROJECT_ROOT
        / f"data/benchmarks/rag/qasper-{'dev' if profile == 'standard' else 'validation'}.json"
    )


def _resamples(profile: str, configured: int) -> int:
    return min(500, configured) if profile == "smoke" else configured


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _median_or_zero(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _quantile_or_zero(values: Sequence[float], quantile: float) -> float:
    return float(np.quantile(values, quantile)) if values else 0.0


def _stratified_questions(
    samples: Sequence[Mapping[str, object]], count: int, *, seed: int
) -> list[Mapping[str, object]]:
    answerable = [
        sample
        for sample in samples
        if sample.get("kind") == "question" and sample.get("answerable")
    ]
    unanswerable = [
        sample
        for sample in samples
        if sample.get("kind") == "question" and not sample.get("answerable")
    ]
    random_state = random.Random(seed)
    random_state.shuffle(answerable)
    random_state.shuffle(unanswerable)
    unanswerable_count = min(len(unanswerable), max(1, count // 3))
    selected = answerable[: count - unanswerable_count] + unanswerable[:unanswerable_count]
    if len(selected) < count:
        selected_ids = {str(item.get("id")) for item in selected}
        remaining = [
            item for item in (*answerable, *unanswerable) if str(item.get("id")) not in selected_ids
        ]
        selected.extend(remaining[: count - len(selected)])
    random_state.shuffle(selected)
    return selected


def _model_revision(name: str, profile: str) -> str:
    if profile == "smoke":
        return embedding_spec(name).revision if name in EMBEDDINGS else "smoke"
    revisions = load_model_lock(PROJECT_ROOT / "data/benchmarks/models/huggingface.json")
    try:
        return revisions[name]
    except KeyError as exc:
        raise RuntimeError(f"Hugging Face model lock has no revision for {name}") from exc
