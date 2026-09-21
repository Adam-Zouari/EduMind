"""Generator profiles and loaders that remain experimental until promotion."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from edumind.rag.contracts import GenerationProfile
from edumind.rag.llm_generator import HuggingFaceGenerator
from experiments.benchmarks.rag.generation.protocol import (
    GENERATOR_LOADERS,
    GenerationProtocol,
)


class MultimodalBenchmarkGenerator(HuggingFaceGenerator):
    """Use the registry-selected multimodal loader inside the experiment."""

    def _load_components(self, path: Path) -> tuple[Any, Any, Any, Any | None]:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=self.profile.dtype,
        )
        return torch, model, processor.tokenizer, processor


def generator_for(
    candidate: str,
    model_lock: Mapping[str, Mapping[str, object]],
    device: str,
    dtype: str,
    protocol: GenerationProtocol,
) -> HuggingFaceGenerator:
    model = protocol.model_id(candidate)
    entry = model_lock[model]
    profile = GenerationProfile(
        model_name=model,
        revision=str(entry["revision"]),
        model_path=str(entry["model_path"]),
        device=device,
        dtype=dtype,
        reasoning=True,
        temperature=protocol.temperature,
        do_sample=protocol.do_sample,
        seed=protocol.meta.seed,
        context_tokens=protocol.context_tokens,
        maximum_answer_tokens=protocol.maximum_answer_tokens,
        streamer_timeout_seconds=protocol.streamer_timeout_seconds,
    )
    return (
        MultimodalBenchmarkGenerator(profile)
        if GENERATOR_LOADERS[candidate] == "multimodal-lm"
        else HuggingFaceGenerator(profile)
    )
