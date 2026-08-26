"""Experiment-only embedding runtimes that need model-specific inference."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from edumind.rag.contracts import EmbeddingSpec
from edumind.rag.embedder import Embedder
from edumind.rag.errors import RAGConfigurationError


QWEN_EMBEDDING_MODELS = {
    "Qwen/Qwen3-Embedding-0.6B",
    "Qwen/Qwen3-Embedding-4B",
}


def build_embedder(spec: EmbeddingSpec) -> Embedder:
    """Use Qwen's documented recipe while leaving all other models unchanged."""
    if spec.model_name in QWEN_EMBEDDING_MODELS:
        return QwenEmbedder(spec)
    return Embedder(spec)


class QwenEmbedder(Embedder):
    """Qwen3 embeddings with an appended end token and last-token pooling."""

    def _encode(self, texts: Sequence[str], *, device: str, role: str) -> np.ndarray:
        del role  # Query/document prefixes are applied by the inherited public methods.
        if not texts:
            return np.empty((0, self.spec.dimension), dtype=np.float32)

        tokenizer, model, torch = self._qwen_model(device)
        end_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            encoded = tokenizer(
                list(texts[start : start + self.batch_size]),
                padding=False,
                truncation=True,
                max_length=self.spec.maximum_length - 2,
            )
            for input_ids, attention_mask in zip(
                encoded["input_ids"], encoded["attention_mask"], strict=True
            ):
                input_ids.append(end_token_id)
                attention_mask.append(1)
            encoded = tokenizer.pad(encoded, padding=True, return_tensors="pt")
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
                vectors = hidden[:, -1]
                if self.spec.normalize:
                    vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
            batches.append(vectors.float().cpu().numpy())

        result = np.concatenate(batches).astype(np.float32, copy=False)
        if result.shape[1] != self.spec.dimension:
            raise RAGConfigurationError(
                f"Embedding contract dimension mismatch for {self.spec.model_name}: expected "
                f"{self.spec.dimension}, received {result.shape[1]}"
            )
        return result

    def _qwen_model(self, device: str):
        if device not in self._models:
            try:
                import torch
                from transformers import AutoModel, AutoTokenizer
            except ModuleNotFoundError as exc:
                raise RAGConfigurationError(
                    "transformers and torch are required; install requirements/benchmarks.lock"
                ) from exc

            source = self.spec.local_path or self.spec.model_name
            load_options: dict[str, object] = {"local_files_only": True}
            if self.spec.local_path is None:
                load_options["revision"] = self.spec.revision
            tokenizer = AutoTokenizer.from_pretrained(
                source,
                padding_side="left",
                **load_options,
            )
            model = AutoModel.from_pretrained(source, **load_options).to(device)
            model.eval()
            self._models[device] = (tokenizer, model, torch)
        return self._models[device]
