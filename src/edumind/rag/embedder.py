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
        batch_size: int = 32,
    ) -> None:
        self.spec = spec
        self.model_name = spec.model_name
        self.embedding_dim = spec.dimension
        self.batch_size = batch_size
        self._models: dict[str, object] = {}

    @property
    def model_loaded(self) -> bool:
        return bool(self._models)

    def embed_query(self, text: str) -> np.ndarray:
        prepared = text if self.spec.interface == "query-document" else self.spec.query_prefix + text
        return self._encode([prepared], device=self.spec.query_device, role="query")[0]

    def embed_texts(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        del show_progress
        prepared = (
            list(texts)
            if self.spec.interface == "query-document"
            else [self.spec.document_prefix + text for text in texts]
        )
        return self._encode(prepared, device=self.spec.document_device, role="document")

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
        prompt_name = (
            self.spec.query_prompt_name if role == "query" else self.spec.document_prompt_name
        )
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
            if self.spec.local_path is None:
                kwargs["revision"] = self.spec.revision
            if self.spec.trust_remote_code:
                kwargs["trust_remote_code"] = True
            model = SentenceTransformer(source, **kwargs)
            model.max_seq_length = self.spec.maximum_length
            self._models[device] = model
        return self._models[device]
