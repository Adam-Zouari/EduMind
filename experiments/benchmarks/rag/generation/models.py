"""Generator profiles and loaders that remain experimental until promotion."""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from edumind.rag.contracts import GenerationProfile
from edumind.rag.llm_generator import HuggingFaceGenerator
from experiments.benchmarks.rag.generation.protocol import GenerationProtocol

_CANDIDATES_PATH = Path(__file__).with_name("candidates.yaml")


@lru_cache(maxsize=1)
def generator_profiles() -> dict[str, tuple[str, str]]:
    payload = yaml.safe_load(_CANDIDATES_PATH.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") if isinstance(payload, Mapping) else None
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("Generation candidates.yaml requires a candidate registry")
    result: dict[str, tuple[str, str]] = {}
    for alias, value in candidates.items():
        if not isinstance(alias, str) or not isinstance(value, Mapping):
            raise ValueError("Generation candidate registry is malformed")
        unknown = set(value) - {"model_id", "loader", "profiles"}
        if unknown:
            raise ValueError(
                f"Generation candidate {alias!r} has unknown fields: "
                + ", ".join(sorted(unknown))
            )
        model_id, loader, profiles = (
            value.get("model_id"),
            value.get("loader"),
            value.get("profiles"),
        )
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"Generation candidate {alias!r} lacks model_id")
        if loader not in {"causal-lm", "multimodal-lm"}:
            raise ValueError(f"Generation candidate {alias!r} has invalid loader")
        if not isinstance(profiles, list) or not profiles:
            raise ValueError(f"Generation candidate {alias!r} lacks profiles")
        if not all(
            isinstance(profile_name, str)
            and profile_name in {"smoke", "development", "validation"}
            for profile_name in profiles
        ):
            raise ValueError(f"Generation candidate {alias!r} has invalid profiles")
        result[alias] = (model_id, loader)
    return result


GENERATOR_PROFILES = generator_profiles()


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
    model, loader = GENERATOR_PROFILES[candidate]
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
        if loader == "multimodal-lm"
        else HuggingFaceGenerator(profile)
    )
