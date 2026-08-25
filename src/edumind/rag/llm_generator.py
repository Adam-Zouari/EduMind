"""Pinned local Hugging Face generation with citation-grounded prompts."""

from __future__ import annotations

import gc
import re
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import GenerationProfile
from .errors import GenerationError, ModelLoadError
from .types import RetrievalHit

DEFAULT_SYSTEM_PROMPT = """You are a careful study assistant.
Answer only from the numbered evidence.
Every factual claim must cite one or more evidence blocks using [1], [2], and so on.
If the evidence cannot answer the question, say exactly: I don't have enough evidence to answer.
Do not invent sources, page numbers, facts, or citations. Keep the answer concise."""


@dataclass(frozen=True)
class GenerationMeasurement:
    answer: str
    total_seconds: float
    time_to_first_token_seconds: float
    load_seconds: float
    prompt_evaluation_seconds: float
    generation_seconds: float
    prompt_tokens: int
    answer_tokens: int
    reasoning_tokens: int
    generated_tokens: int
    tokens_per_second: float


class HuggingFaceGenerator:
    """Load one exact local checkpoint without quantization or device offload."""

    def __init__(self, profile: GenerationProfile) -> None:
        self.profile = profile
        self.model_name = profile.model_name
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._processor: Any | None = None
        self._torch: Any | None = None

    def health_check(self) -> bool:
        path = Path(self.profile.model_path)
        return path.is_dir() and (path / "config.json").is_file()

    def generate(
        self,
        query: str,
        context: str,
        *,
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        measurement = self._generate_measured(query, context, system_prompt)
        return iter((measurement.answer,)) if stream else measurement.answer

    def generate_with_results(
        self,
        query: str,
        results: Sequence[RetrievalHit],
        system_prompt: str | None = None,
        stream: bool = False,
    ) -> str:
        output = self.generate(
            query,
            self.build_context(results),
            system_prompt=system_prompt,
            stream=stream,
        )
        return "".join(output) if not isinstance(output, str) else output

    def generate_measured_with_results(
        self,
        query: str,
        results: Sequence[RetrievalHit],
        system_prompt: str | None = None,
    ) -> GenerationMeasurement:
        return self._generate_measured(
            query, self.build_context(results), system_prompt
        )

    def _generate_measured(
        self, query: str, context: str, system_prompt: str | None
    ) -> GenerationMeasurement:
        load_seconds = self._ensure_loaded()
        assert self._model is not None and self._tokenizer is not None and self._torch is not None
        messages = [
            {"role": "system", "content": system_prompt or DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Evidence:\n{context}\n\nQuestion: {query}\nAnswer with citations:",
            },
        ]
        prompt = self._chat_prompt(messages)
        encoded = (
            self._processor(
                text=prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.profile.context_tokens,
            )
            if self._processor is not None
            else self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.profile.context_tokens,
            )
        )
        encoded = {name: value.to(self.profile.device) for name, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        self._torch.manual_seed(self.profile.seed)
        if self.profile.device.startswith("cuda"):
            self._torch.cuda.manual_seed_all(self.profile.seed)
            self._torch.cuda.reset_peak_memory_stats()

        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=600,
        )
        errors: list[BaseException] = []
        generation_options: dict[str, object] = {
            **encoded,
            "streamer": streamer,
            "max_new_tokens": self.profile.maximum_answer_tokens,
            "do_sample": self.profile.temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if self.profile.temperature > 0:
            generation_options["temperature"] = self.profile.temperature

        def run() -> None:
            try:
                with self._torch.inference_mode():
                    self._model.generate(**generation_options)
            except BaseException as exc:
                errors.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        started = time.perf_counter()
        worker = threading.Thread(target=run, name="edumind-hf-generation", daemon=True)
        worker.start()
        first_token_at: float | None = None
        pieces: list[str] = []
        for text in streamer:
            if text and first_token_at is None:
                first_token_at = time.perf_counter()
            pieces.append(text)
        worker.join()
        finished = time.perf_counter()
        if errors:
            raise GenerationError(
                f"Generation failed for {self.profile.model_name}: {errors[0]}"
            ) from errors[0]

        raw_answer = "".join(pieces).strip()
        answer, reasoning_text = _visible_answer(raw_answer)
        generated_tokens = len(
            self._tokenizer.encode(raw_answer, add_special_tokens=False)
        )
        answer_tokens = len(self._tokenizer.encode(answer, add_special_tokens=False))
        reasoning_tokens = len(
            self._tokenizer.encode(reasoning_text, add_special_tokens=False)
        )
        first = first_token_at or finished
        prompt_seconds = first - started
        generation_seconds = max(finished - first, 0.0)
        return GenerationMeasurement(
            answer=answer,
            total_seconds=finished - started,
            time_to_first_token_seconds=prompt_seconds,
            load_seconds=load_seconds,
            prompt_evaluation_seconds=prompt_seconds,
            generation_seconds=generation_seconds,
            prompt_tokens=prompt_tokens,
            answer_tokens=answer_tokens,
            reasoning_tokens=reasoning_tokens,
            generated_tokens=generated_tokens,
            tokens_per_second=generated_tokens / max(generation_seconds, 1e-9),
        )

    def _chat_prompt(self, messages: list[dict[str, str]]) -> str:
        assert self._tokenizer is not None
        template_owner = self._processor or self._tokenizer
        options: dict[str, object] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        options["enable_thinking"] = self.profile.reasoning
        try:
            return str(template_owner.apply_chat_template(messages, **options))
        except TypeError as exc:
            raise ModelLoadError(
                f"The pinned chat template for {self.model_name} does not support its "
                "declared reasoning profile"
            ) from exc

    def _ensure_loaded(self) -> float:
        if self._model is not None:
            return 0.0
        path = Path(self.profile.model_path)
        if not path.is_dir():
            raise ModelLoadError(
                f"Pinned model is missing at {path}; run `python "
                "experiments/benchmarks/prepare.py app-models` or `all-models`."
            )
        started = time.perf_counter()
        try:
            self._torch, self._model, self._tokenizer, self._processor = (
                self._load_components(path)
            )
            self._model = self._model.to(self.profile.device)
            self._model.eval()
        except Exception as exc:
            self.unload()
            raise ModelLoadError(
                f"Cannot load pinned model {self.model_name} on {self.profile.device}: {exc}"
            ) from exc
        return time.perf_counter() - started

    def _load_components(self, path: Path) -> tuple[Any, Any, Any, Any | None]:
        """Load the production causal-LM profile from an exact local snapshot."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=self.profile.dtype,
        )
        return torch, model, tokenizer, None

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        self._processor = None
        torch = self._torch
        self._torch = None
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def runtime_memory(self) -> dict[str, float]:
        if self._torch is None or not self.profile.device.startswith("cuda"):
            return {}
        return {
            "model_peak_vram_mb": float(self._torch.cuda.max_memory_allocated())
            / (1024**2),
            "model_reserved_vram_mb": float(self._torch.cuda.max_memory_reserved())
            / (1024**2),
        }

    @staticmethod
    def build_context(results: Sequence[RetrievalHit]) -> str:
        return "\n\n".join(
            f"[{index}] source={hit.source}; page={hit.page}\n{hit.document}"
            for index, hit in enumerate(results, start=1)
        )


def _visible_answer(text: str) -> tuple[str, str]:
    reasoning = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    visible = re.sub(
        r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    return visible or text.strip(), "\n".join(reasoning).strip()
