"""Generator profiles and loaders that remain experimental until promotion."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from edumind.rag.contracts import GenerationProfile
from edumind.rag.llm_generator import HuggingFaceGenerator

GENERATOR_PROFILES = {
    "qwen3-1.7b-control": ("Qwen/Qwen3-1.7B", False),
    "minicpm5-1b-reasoning": ("openbmb/MiniCPM5-1B", True),
    "g9v3-3b-reasoning": ("ai9stars/G9v3-3B", True),
    "qwen3.5-4b-reasoning": ("Qwen/Qwen3.5-4B", True),
}


class BenchmarkGenerator(HuggingFaceGenerator):
    """Use candidate-specific architecture loaders only inside the experiment."""

    def _load_components(self, path: Path) -> tuple[Any, Any, Any, Any | None]:
        if self.model_name != "Qwen/Qwen3.5-4B":
            return super()._load_components(path)
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
) -> HuggingFaceGenerator:
    model, reasoning = GENERATOR_PROFILES[candidate]
    entry = model_lock[model]
    profile = GenerationProfile(
        model_name=model,
        revision=str(entry["revision"]),
        model_path=str(entry["model_path"]),
        device=device,
        dtype="auto",
        reasoning=reasoning,
        temperature=0.0,
        seed=42,
        context_tokens=8192,
        maximum_answer_tokens=256,
    )
    return BenchmarkGenerator(profile) if model == "Qwen/Qwen3.5-4B" else HuggingFaceGenerator(profile)

