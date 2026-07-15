"""Experiment-only BM25, rank fusion, and reranking implementations."""

from __future__ import annotations

import re
from collections.abc import Sequence


class BM25:
    def __init__(self, documents: Sequence[str]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install requirements/benchmarks.lock for BM25") from exc
        self.model = BM25Okapi([_tokens(text) for text in documents])

    def rank(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores = self.model.get_scores(_tokens(query))
        return sorted(enumerate(map(float, scores)), key=lambda row: (-row[1], row[0]))[:limit]


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], limit: int, rrf_k: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] = scores.get(identifier, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:limit]


class Reranker:
    def __init__(self, model: str, revision: str) -> None:
        self.model_name = model
        self.revision = revision
        self.model = None

    def rank(self, query: str, documents: Sequence[str]) -> list[int]:
        if self.model is None:
            if self.model_name.startswith("Qwen/"):
                self.model = _QwenReranker(self.model_name, self.revision)
            else:
                from sentence_transformers import CrossEncoder

                self.model = CrossEncoder(
                    self.model_name,
                    revision=self.revision,
                    device="cpu",
                    local_files_only=True,
                )
        scores = (
            self.model.predict(query, documents)
            if isinstance(self.model, _QwenReranker)
            else self.model.predict([(query, document) for document in documents])
        )
        return sorted(range(len(documents)), key=lambda index: (-float(scores[index]), index))


class _QwenReranker:
    """Generative yes/no relevance scoring required by Qwen3-Reranker."""

    def __init__(self, model_name: str, revision: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=True,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).to("cpu")
        self.model.eval()
        self.yes_id = self.tokenizer.encode("yes", add_special_tokens=False)[0]
        self.no_id = self.tokenizer.encode("no", add_special_tokens=False)[0]

    def predict(self, query: str, documents: Sequence[str]) -> list[float]:
        instruction = (
            "Given an educational question, determine whether the document contains "
            "evidence that helps answer it."
        )
        prompts = [
            "<|im_start|>system\nJudge whether the Document meets the requirements "
            "based on the Query and the Instruct. Output only yes or no."
            "<|im_end|>\n<|im_start|>user\n"
            f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
            for document in documents
        ]
        scores: list[float] = []
        for start in range(0, len(prompts), 4):
            encoded = self.tokenizer(
                prompts[start : start + 4],
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt",
            )
            with self.torch.no_grad():
                logits = self.model(**encoded).logits[:, -1, [self.no_id, self.yes_id]]
                probabilities = self.torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(float(value) for value in probabilities.cpu().tolist())
        return scores


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())
