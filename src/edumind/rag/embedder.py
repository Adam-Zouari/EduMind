"""Model-contract-aware lazy embedding runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from .contracts import EmbeddingSpec
from .errors import RAGConfigurationError
from .types import ChunkRecord


class Embedder:
    def __init__(
        self,
        spec: EmbeddingSpec,
        batch_size: int = 1,
        *,
        dtype: str | None = None,
        enforce_device: bool = False,
    ) -> None:
        self.spec = spec
        self.batch_size = batch_size
        self.dtype = dtype
        self.enforce_device = enforce_device
        self._models: dict[str, object] = {}

    @property
    def model_loaded(self) -> bool:
        return bool(self._models)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode(
            self._prepared_texts([text], "query"),
            device=self.spec.query_device,
            role="query",
        )[0]

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(
            self._prepared_texts(texts, "document"),
            device=self.spec.document_device,
            role="document",
        )

    def prepare(self) -> None:
        """Load every requested runtime before measured work begins."""

        for device in {self.spec.document_device, self.spec.query_device}:
            self._model(device)

    def input_token_counts(self, texts: Sequence[str], *, role: str) -> list[int]:
        """Count complete model inputs without allowing tokenizer truncation."""

        if role not in {"query", "document"}:
            raise ValueError("Embedding input role must be query or document")
        device = (
            self.spec.query_device if role == "query" else self.spec.document_device
        )
        model = self._model(device)
        prepared = self._prepared_texts(texts, role)
        prompt_name, prompt = self._resolved_prompt(model, role)
        counts: list[int] = []
        for text in prepared:
            try:
                features = model.preprocess(
                    [text],
                    prompt=prompt or None,
                    task=role if self.spec.interface == "query-document" else None,
                    processing_kwargs={"text": {"padding": False, "truncation": False}},
                )
            except Exception as exc:
                raise RAGConfigurationError(
                    f"Could not validate complete {role} input for "
                    f"{self.spec.model_name} with prompt {prompt_name!r}"
                ) from exc
            input_ids = features.get("input_ids")
            if input_ids is None:
                raise RAGConfigurationError(
                    f"{self.spec.model_name} preprocessing returned no input IDs"
                )
            size = (
                int(input_ids.numel())
                if hasattr(input_ids, "numel")
                else int(np.asarray(input_ids).size)
            )
            counts.append(size)
        return counts

    def input_configuration(self, role: str) -> dict[str, object]:
        """Return the exact role-specific preprocessing applied by the runtime."""

        if role not in {"query", "document"}:
            raise ValueError("Embedding input role must be query or document")
        device = (
            self.spec.query_device if role == "query" else self.spec.document_device
        )
        prompt_name, prompt = self._resolved_prompt(self._model(device), role)
        prefix = (
            self.spec.query_prefix if role == "query" else self.spec.document_prefix
        )
        return {
            "interface": self.spec.interface,
            "prefix": prefix,
            "prompt_name": prompt_name,
            "prompt": prompt,
            "special_token_policy": "sentence-transformers-model-preprocessing",
        }

    def embed_chunks(self, chunks: Sequence[ChunkRecord]) -> list[ChunkRecord]:
        if not chunks:
            return []
        vectors = self.embed_texts([chunk.text for chunk in chunks])
        return [
            replace(chunk, embedding=vector.astype(float).tolist())
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

    def _encode(self, texts: Sequence[str], *, device: str, role: str) -> np.ndarray:
        model = self._model(device)
        options: dict[str, object] = {
            "batch_size": self.batch_size,
            "normalize_embeddings": self.spec.normalize,
            "convert_to_numpy": True,
            "show_progress_bar": False,
            **dict(self.spec.encode_options),
        }
        prompt_name, _ = self._resolved_prompt(model, role)
        if prompt_name:
            options["prompt_name"] = prompt_name
        if self.spec.interface == "query-document":
            encoder = model.encode_query if role == "query" else model.encode_document
        else:
            encoder = model.encode
        vectors = np.asarray(
            encoder(list(texts), **options),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape[1] != self.spec.dimension:
            actual = vectors.shape[1] if vectors.ndim == 2 else "invalid"
            raise RAGConfigurationError(
                f"Embedding contract dimension mismatch for {self.spec.model_name}: expected "
                f"{self.spec.dimension}, received {actual}"
            )
        return vectors

    def _model(self, device: str):
        if device not in self._models:
            try:
                from sentence_transformers import SentenceTransformer
            except ModuleNotFoundError as exc:
                raise RAGConfigurationError(
                    "sentence-transformers is required; install requirements/app.lock"
                ) from exc
            source = self.spec.local_path or self.spec.model_name
            kwargs: dict[str, object] = {
                "device": device,
                "local_files_only": True,
            }
            if self.dtype is not None:
                try:
                    import torch

                    kwargs["model_kwargs"] = {
                        "torch_dtype": {
                            "float32": torch.float32,
                            "float16": torch.float16,
                            "bfloat16": torch.bfloat16,
                        }[self.dtype]
                    }
                except (ImportError, KeyError) as exc:
                    raise RAGConfigurationError(
                        f"Unsupported embedding dtype: {self.dtype}"
                    ) from exc
            if self.spec.local_path is None:
                kwargs["revision"] = self.spec.revision
            if self.spec.trust_remote_code:
                kwargs["trust_remote_code"] = True
            model = SentenceTransformer(source, **kwargs)
            native_maximum = getattr(model, "max_seq_length", None)
            if (
                isinstance(native_maximum, int)
                and native_maximum > 0
                and self.spec.maximum_length > native_maximum
            ):
                raise RAGConfigurationError(
                    f"Embedding contract for {self.spec.model_name} declares "
                    f"{self.spec.maximum_length} tokens, but the prepared model exposes "
                    f"{native_maximum}"
                )
            model.max_seq_length = self.spec.maximum_length
            actual_device = str(model.device)
            if self.enforce_device and not (
                actual_device == device
                or (device == "cuda" and actual_device.startswith("cuda:"))
            ):
                raise RAGConfigurationError(
                    f"Embedding device fallback is forbidden: requested {device}, "
                    f"loaded {actual_device}"
                )
            self._models[device] = model
        return self._models[device]

    def _prepared_texts(self, texts: Sequence[str], role: str) -> list[str]:
        if self.spec.interface == "query-document":
            return list(texts)
        prefix = (
            self.spec.query_prefix if role == "query" else self.spec.document_prefix
        )
        return [prefix + text for text in texts]

    def _resolved_prompt(self, model, role: str) -> tuple[str | None, str]:
        prompt_name = (
            self.spec.query_prompt_name
            if role == "query"
            else self.spec.document_prompt_name
        )
        prompts = getattr(model, "prompts", {})
        if prompt_name is None and self.spec.interface == "query-document":
            candidates = (
                ("query",) if role == "query" else ("document", "passage", "corpus")
            )
            prompt_name = next((name for name in candidates if name in prompts), None)
        if prompt_name is None:
            prompt_name = getattr(model, "default_prompt_name", None)
        if prompt_name is not None and prompt_name not in prompts:
            raise RAGConfigurationError(
                f"Embedding prompt {prompt_name!r} is absent from {self.spec.model_name}"
            )
        return prompt_name, str(prompts.get(prompt_name, "")) if prompt_name else ""
