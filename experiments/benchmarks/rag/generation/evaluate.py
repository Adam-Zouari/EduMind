"""Measured direct Hugging Face generation over frozen or retrieved contexts."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from edumind.rag.tokenizers import TiktokenOffsetTokenizer
from edumind.rag.types import RetrievalHit

from experiments.benchmarks.common.contracts import DatasetManifest, SampleResult
from experiments.benchmarks.common.metrics import (
    balanced_accuracy,
    citation_scores,
    exact_match,
    rouge_l,
    token_f1,
)
from experiments.benchmarks.rag.evaluation import (
    ExactIndex,
    rank,
    reranker_for,
    retrieval_metrics,
)
from experiments.benchmarks.rag.generation.models import (
    GENERATOR_PROFILES,
    generator_for,
)


class LocalFaithfulness:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model = None

    def score(self, context: str, answer: str) -> float:
        if not answer.strip() or _is_refusal(answer):
            return 1.0
        if self.model is None:
            from transformers import AutoModelForSequenceClassification

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
        score = self.model.predict([(context, _clean_answer(answer))])
        return float(score[0])


def evaluate_candidate(
    candidate: str,
    manifest: DatasetManifest,
    model_lock: Mapping[str, Mapping[str, object]],
    *,
    final_index: ExactIndex | None = None,
    retrieval_method: str = "frozen",
    top_k: int = 5,
    repetitions: int = 1,
    device: str = "cpu",
):
    questions = _questions(manifest, 24)
    documents = {
        str(row["id"]): str(row["text"])
        for row in manifest.samples
        if row.get("kind") == "document"
    }
    generator = generator_for(candidate, model_lock, device)
    tokenizer = TiktokenOffsetTokenizer()
    faithfulness = LocalFaithfulness(
        str(model_lock["vectara/hallucination_evaluation_model"]["model_path"])
    )
    retrieval_reranker = (
        reranker_for(retrieval_method, model_lock) if final_index is not None else None
    )
    context_cache: dict[str, tuple[list[RetrievalHit], str, float]] = {}
    context_questions = questions[:1] if final_index is not None else questions
    for question in context_questions:
        context_cache[str(question["id"])] = (
            _retrieved_hits(question, final_index, retrieval_method, top_k, retrieval_reranker)
            if final_index is not None
            else _frozen_hits(question, documents, tokenizer)
        )

    first_hits, _, _ = context_cache[str(questions[0]["id"])]
    generator.unload()
    cold = generator.generate_measured_with_results(str(questions[0]["question"]), first_hits)
    for _ in range(2):
        generator.generate_measured_with_results(str(questions[0]["question"]), first_hits)

    samples: list[SampleResult] = []
    measurements = []
    total_latencies: list[float] = []
    retrieval_latencies: list[float] = []
    predictions: list[bool] = []
    labels: list[bool] = []
    for question in questions:
        repeated = []
        repeated_inputs = []
        for _ in range(repetitions):
            hits, context, retrieval_seconds = (
                _retrieved_hits(
                    question, final_index, retrieval_method, top_k, retrieval_reranker
                )
                if final_index is not None
                else context_cache[str(question["id"])]
            )
            repeated_inputs.append((hits, context, retrieval_seconds))
            repeated.append(
                generator.generate_measured_with_results(str(question["question"]), hits)
            )
            retrieval_latencies.append(retrieval_seconds)
        hits, context, _ = repeated_inputs[0]
        measurements.extend(repeated)
        total_latencies.extend(
            measurement.total_seconds + inputs[2]
            for measurement, inputs in zip(repeated, repeated_inputs, strict=True)
        )
        answer = repeated[0].answer
        answerable = bool(question.get("answerable"))
        repeat_predictions = [
            not _is_refusal(measurement.answer)
            for measurement in repeated
        ]
        predicted_answerable = Counter(repeat_predictions).most_common(1)[0][0]
        predictions.append(predicted_answerable)
        labels.append(answerable)
        raw_references = question.get("accepted_answers", [])
        references = (
            [str(value) for value in raw_references if str(value).strip()]
            if isinstance(raw_references, Sequence) and not isinstance(raw_references, str)
            else []
        )
        if not references:
            references = [str(question.get("answer", ""))]
        reference = references[0]
        repeated_metrics = []
        for measurement, repeat_prediction in zip(repeated, repeat_predictions, strict=True):
            repeat_answer = measurement.answer
            clean = _clean_answer(repeat_answer)
            repeated_metrics.append(
                {
                    "exact_match": max(exact_match(clean, value) for value in references) if answerable else float(not repeat_prediction),
                    "token_f1": max(token_f1(clean, value) for value in references) if answerable else float(not repeat_prediction),
                    "rouge_l": max(rouge_l(clean, value) for value in references) if answerable else float(not repeat_prediction),
                    **citation_scores(repeat_answer, _supported_contexts(question, hits)),
                    "answerability_correct": float(repeat_prediction == answerable),
                    "hhem_faithfulness": faithfulness.score(context, repeat_answer),
                    "unsupported_answer_rate": float(not answerable and repeat_prediction),
                    "malformed_output_rate": float(_malformed(repeat_answer, len(hits))),
                }
            )
        metric_names = repeated_metrics[0]
        averaged_metrics = {
            name: float(np.mean([row[name] for row in repeated_metrics])) for name in metric_names
        }
        averaged_metrics["determinism"] = float(
            all(measurement.answer == answer for measurement in repeated)
        )
        evidence_type = str(question.get("evidence_type", "text"))
        if final_index is not None:
            by_id = {chunk.identifier: chunk for chunk in final_index.chunks}
            selected = [by_id[hit.id] for hit in hits if hit.id in by_id]
            retrieval_quality, retrieved_tokens = retrieval_metrics(
                question, selected, final_index.chunks, final_index.tokenizer
            )
            averaged_metrics.update(retrieval_quality)
        else:
            retrieved_tokens = sum(hit.token_count for hit in hits)
        averaged_metrics.update(
            {
                f"stratum.{evidence_type}.{name}": value
                for name, value in averaged_metrics.items()
                if name
                in {
                    "citation_f1",
                    "hhem_faithfulness",
                    "token_f1",
                    "answerability_correct",
                    "ndcg_at_3",
                    "ndcg_at_5",
                    "context_recall_at_3",
                    "context_recall_at_5",
                }
            }
        )
        sample_latency = float(
            np.median(
                [
                    measurement.total_seconds + inputs[2]
                    for measurement, inputs in zip(repeated, repeated_inputs, strict=True)
                ]
            )
        )
        samples.append(
            SampleResult(
                str(question["id"]),
                averaged_metrics,
                sample_latency,
                {
                    "answerable": answerable,
                    "question": question["question"],
                    "reference_answer": " | ".join(references),
                    "generated_answer": answer,
                    "frozen_context": context,
                    "retrieval": retrieval_method,
                    "top_k": top_k,
                    "measured_repetitions": repetitions,
                    "retrieved_tokens": retrieved_tokens,
                    "evidence_type": evidence_type,
                },
            )
        )
    runtime_memory = generator.runtime_memory()
    generator.unload()
    refusal = _refusal_scores(labels, predictions)
    return samples, {
        "p50_latency_seconds": float(np.median(total_latencies)),
        "p95_latency_seconds": float(np.quantile(total_latencies, 0.95)),
        "p50_retrieval_seconds": float(np.median(retrieval_latencies)),
        "p95_retrieval_seconds": float(np.quantile(retrieval_latencies, 0.95)),
        "p50_generation_seconds": float(np.median([row.total_seconds for row in measurements])),
        "p95_generation_seconds": float(np.quantile([row.total_seconds for row in measurements], 0.95)),
        "p50_time_to_first_token_seconds": float(np.median([row.time_to_first_token_seconds for row in measurements])),
        "p95_time_to_first_token_seconds": float(np.quantile([row.time_to_first_token_seconds for row in measurements], 0.95)),
        "mean_prompt_evaluation_seconds": float(np.mean([row.prompt_evaluation_seconds for row in measurements])),
        "mean_model_generation_seconds": float(np.mean([row.generation_seconds for row in measurements])),
        "mean_prompt_tokens": float(np.mean([row.prompt_tokens for row in measurements])),
        "tokens_per_second": float(np.mean([row.tokens_per_second for row in measurements])),
        "answers_per_minute": 60.0 / max(float(np.mean(total_latencies)), 1e-9),
        "mean_answer_tokens": float(np.mean([row.answer_tokens for row in measurements])),
        "mean_reasoning_tokens": float(
            np.mean([row.reasoning_tokens for row in measurements])
        ),
        "mean_generated_tokens": float(
            np.mean([row.generated_tokens for row in measurements])
        ),
        "cold_load_seconds": cold.load_seconds,
        **runtime_memory,
    }, {
        "answerability_balanced_accuracy": balanced_accuracy(labels, predictions),
        **refusal,
        "human_review_required": 1.0,
    }


def _frozen_hits(question, documents, tokenizer) -> tuple[list[RetrievalHit], str, float]:
    document = documents[str(question["document_id"])]
    evidence = [row for row in question.get("evidence", []) if isinstance(row, Mapping)]
    texts = [str(row.get("text") or document[int(row["start"]):int(row["end"])]).strip() for row in evidence]
    if not texts:
        texts = [tokenizer.truncate(document, 3500)]
    packed_texts = []
    used = 0
    for text in texts:
        available = 3500 - used
        if available <= 0:
            break
        packed = tokenizer.truncate(text, available)
        if packed.strip():
            packed_texts.append(packed)
            used += tokenizer.count(packed)
    hits = [
        RetrievalHit(
            f"frozen-{index}",
            text,
            {"source": question["document_id"], "document_id": question["document_id"], "page": "N/A"},
            1.0,
            index,
            "frozen-oracle-context",
            tokenizer.count(text),
        )
        for index, text in enumerate(packed_texts, 1)
    ]
    return hits, "\n\n".join(hit.document for hit in hits), 0.0


def _retrieved_hits(question, index, method, top_k, reranker) -> tuple[list[RetrievalHit], str, float]:
    import time

    started = time.perf_counter()
    positions = rank(index, str(question["question"]), method, reranker)
    hits = []
    used_tokens = 0
    for position in positions:
        if len(hits) >= top_k or used_tokens >= 2048:
            break
        chunk = index.chunks[position]
        remaining = 2048 - used_tokens
        text = chunk.text
        end = chunk.end
        tokens = chunk.tokens
        if tokens > remaining:
            spans = index.tokenizer.spans(text)
            if not spans or remaining <= 0:
                break
            end_offset = spans[min(remaining, len(spans)) - 1][1]
            text = text[:end_offset]
            end = chunk.start + end_offset
            tokens = min(remaining, len(spans))
        hits.append(
            RetrievalHit(
                chunk.identifier,
                text,
                {
                    "source": chunk.document_id,
                    "document_id": chunk.document_id,
                    "start": chunk.start,
                    "end": end,
                },
                1.0,
                len(hits) + 1,
                method,
                tokens,
            )
        )
        used_tokens += tokens
    return hits, "\n\n".join(hit.document for hit in hits), time.perf_counter() - started


def _supported_contexts(question, hits) -> set[int]:
    result = set()
    for rank_number, hit in enumerate(hits, 1):
        for evidence in question.get("evidence", []):
            if hit.metadata.get("document_id") != evidence.get("document_id"):
                continue
            if "start" not in hit.metadata or max(0, min(int(hit.metadata["end"]), int(evidence["end"])) - max(int(hit.metadata["start"]), int(evidence["start"]))) > 0:
                result.add(rank_number)
    return result


def _questions(manifest, count):
    import random

    rows = [
        row
        for row in manifest.samples
        if row.get("kind") == "question"
        and (not row.get("answerable") or bool(row.get("evidence")))
    ]
    groups = {}
    for row in rows:
        answer_type = str(
            row.get("answer_type")
            or ("answerable" if row.get("answerable") else "unanswerable")
        )
        key = f"{row.get('evidence_type', 'text')}|{answer_type}"
        groups.setdefault(key, []).append(row)
    for key, group in groups.items():
        random.Random(42 + sum(ord(character) for character in key)).shuffle(group)
    selected = []
    while len(selected) < min(count, len(rows)) and any(groups.values()):
        for key in sorted(groups):
            if groups[key] and len(selected) < count:
                selected.append(groups[key].pop())
    return selected


def _clean_answer(answer: str) -> str:
    return re.sub(r"\[\d+\]", "", answer).strip()


def _is_refusal(answer: str) -> bool:
    normalized = answer.casefold().replace("\u2019", "'")
    return bool(
        re.search(
            r"(?:don't|do not) have enough evidence|insufficient evidence|"
            r"cannot answer (?:from|based on) (?:the )?(?:given )?evidence",
            normalized,
        )
    )


def _malformed(answer: str, context_count: int) -> bool:
    if not answer.strip():
        return True
    if _is_refusal(answer):
        return False
    citations = [int(value) for value in re.findall(r"\[(\d+)\]", answer)]
    return not citations or any(value < 1 or value > context_count for value in citations)


def _refusal_scores(answerable, predicted_answerable):
    truth = [not value for value in answerable]
    predicted = [not value for value in predicted_answerable]
    tp = sum(left and right for left, right in zip(truth, predicted))
    fp = sum(not left and right for left, right in zip(truth, predicted))
    fn = sum(left and not right for left, right in zip(truth, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "refusal_precision": precision,
        "refusal_recall": recall,
        "refusal_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }
