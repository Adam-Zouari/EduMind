"""Experiment-only BM25, rank fusion, and reranking implementations."""

from __future__ import annotations

import pickle
import re
from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import numpy as np


class BM25:
    def __init__(
        self,
        documents: Sequence[str],
        *,
        k1: float,
        b: float,
        epsilon: float,
    ) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install requirements/benchmarks.lock for BM25") from exc
        self.model = BM25Okapi(
            [_tokens(text) for text in documents],
            k1=k1,
            b=b,
            epsilon=epsilon,
        )

    @property
    def storage_bytes(self) -> int:
        return len(self.serialized)

    @cached_property
    def serialized(self) -> bytes:
        return pickle.dumps(self.model, protocol=pickle.HIGHEST_PROTOCOL)

    def rank(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores = self.model.get_scores(_tokens(query))
        return sorted(enumerate(map(float, scores)), key=lambda row: (-row[1], row[0]))[
            :limit
        ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    limit: int,
    rrf_k: int,
    *,
    weights: Sequence[float],
    tie_keys: dict[int, str] | None = None,
) -> list[int]:
    return [
        identifier
        for identifier, _ in reciprocal_rank_fusion_with_scores(
            rankings, limit, rrf_k, weights=weights, tie_keys=tie_keys
        )
    ]


def reciprocal_rank_fusion_with_scores(
    rankings: Sequence[Sequence[int]],
    limit: int,
    rrf_k: int,
    *,
    weights: Sequence[float],
    tie_keys: dict[int, str] | None = None,
) -> list[tuple[int, float]]:
    """Fuse ranks with the frozen RRF score and deterministic documented ties."""

    weights = tuple(weights)
    if len(weights) != len(rankings) or any(weight <= 0 for weight in weights):
        raise ValueError("RRF requires one positive weight per source ranking")
    if limit <= 0 or rrf_k < 0:
        raise ValueError("RRF limit must be positive and rrf_k non-negative")
    scores: dict[int, float] = {}
    source_ranks: list[dict[int, int]] = []
    for ranking, weight in zip(rankings, weights, strict=True):
        if len(ranking) != len(set(ranking)):
            raise ValueError("RRF source rankings must contain unique identifiers")
        positions = {identifier: rank for rank, identifier in enumerate(ranking, 1)}
        source_ranks.append(positions)
        for rank, identifier in enumerate(ranking, 1):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (rrf_k + rank)
    missing_rank = limit + 1

    def key(identifier: int) -> tuple[float, int, int, str]:
        ranks = [source.get(identifier, missing_rank) for source in source_ranks]
        return (
            -scores[identifier],
            min(ranks),
            sum(ranks),
            (tie_keys or {}).get(identifier, str(identifier)),
        )

    return [
        (identifier, scores[identifier])
        for identifier in sorted(scores, key=key)[:limit]
    ]


class Reranker:
    def __init__(
        self,
        model: str,
        revision: str,
        model_path: str,
        *,
        maximum_length: int,
        device: str = "cpu",
        dtype: str = "float32",
        batch_size: int = 1,
    ) -> None:
        self.model_name = model
        self.revision = revision
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.maximum_length = maximum_length
        if batch_size <= 0 or maximum_length <= 0:
            raise ValueError("Reranker batch size and maximum length must be positive")
        self.model = None

    @property
    def snapshot_bytes(self) -> int:
        root = Path(self.model_path)
        return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())

    def prepare(self) -> None:
        if self.model is None:
            import torch
            from sentence_transformers import CrossEncoder

            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[self.dtype]
            self.model = CrossEncoder(
                self.model_path,
                device=self.device,
                local_files_only=True,
                trust_remote_code=False,
                max_length=self.maximum_length,
                model_kwargs={"torch_dtype": dtype},
            )
            parameter = next(self.model.model.parameters(), None)
            if (
                parameter is not None
                and self.device == "cuda"
                and parameter.device.type != "cuda"
            ):
                raise RuntimeError("Reranker silently fell back from CUDA")

    def input_token_counts(self, query: str, documents: Sequence[str]) -> list[int]:
        self.prepare()
        tokenizer = self.model.tokenizer
        return [
            len(
                tokenizer(
                    query,
                    document,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
            for document in documents
        ]

    def rank_with_scores(
        self,
        query: str,
        documents: Sequence[str],
        *,
        validate_inputs: bool = True,
    ) -> list[tuple[int, float]]:
        self.prepare()
        import torch

        if validate_inputs:
            counts = self.input_token_counts(query, documents)
            if any(count > self.maximum_length for count in counts):
                raise ValueError(
                    f"{self.model_name} input exceeds {self.maximum_length} tokens; "
                    "truncation is forbidden"
                )
        scores = np.asarray(
            self.model.predict(
                [(query, document) for document in documents],
                batch_size=self.batch_size,
                show_progress_bar=False,
                activation_fn=torch.nn.Identity(),
                apply_softmax=False,
                convert_to_numpy=True,
            ),
            dtype=float,
        ).reshape(-1)
        if scores.shape != (len(documents),) or not np.isfinite(scores).all():
            raise RuntimeError("Reranker returned malformed or non-finite scores")
        return sorted(
            [(index, float(scores[index])) for index in range(len(documents))],
            key=lambda row: (-row[1], row[0]),
        )

    def rank(self, query: str, documents: Sequence[str]) -> list[int]:
        return [index for index, _ in self.rank_with_scores(query, documents)]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())
