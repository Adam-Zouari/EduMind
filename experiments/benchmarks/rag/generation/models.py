"""Local Hugging Face backend, separate from the application's legacy generator.

Backend interface: prepare(question), load(), placement(), infer(prepared, seed),
close(). prepare never loads weights and must not truncate. infer retains token
events including EOS. Alternative backends can implement this small interface.
"""

import json
import time
from pathlib import Path

from edumind.common.artifacts import sha256_file
from edumind.common.gpu import release_runtime_memory
from edumind.common.model_placement import inspect_model_placement

from .metrics import REFUSAL, parse_response
from .protocol import GENERATOR_LOADERS

SYSTEM_PROMPT = (
    "Answer only from the supplied evidence. Return exactly one JSON object with "
    'exactly these fields: "status", "answer", "citations". '
    'For a supported answer use status "answered", a substantive answer string, '
    "and an ordered array of evidence-block ID strings in citations. "
    "Do not place reasoning in the answer. If evidence is insufficient, return "
    + json.dumps(
        {"status": "insufficient_evidence", "answer": REFUSAL, "citations": []}
    )
    + ". Do not add fields, markdown fences, sources or facts not supplied."
)


class LocalBackend:
    def __init__(self, candidate, model_lock, device, dtype, protocol):
        self.candidate, self.protocol, self.device, self.dtype = (
            candidate,
            protocol,
            device,
            dtype,
        )
        self.entry = model_lock[protocol.model_id(candidate)]
        self.path = Path(str(self.entry["model_path"]))
        if not self.path.is_dir() or not self.entry.get("revision"):
            raise ValueError(
                "Missing exact local model snapshot/revision; preparation is separate"
            )
        self.model = self.torch = self.tokenizer = self.processor = None
        self.resolved_decoder = protocol.decoder(candidate)
        self.decoder_source = None

    def _tokenizer(self):
        if self.tokenizer is not None:
            return
        from transformers import AutoProcessor, AutoTokenizer

        if GENERATOR_LOADERS[self.candidate] == "multimodal-lm":
            self.processor = AutoProcessor.from_pretrained(
                self.path, local_files_only=True
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.path, local_files_only=True
            )

    def prepare(self, question):
        self._tokenizer()
        context = "\n\n".join(
            f"[{block['id']}]\n{block['text']}" for block in question["blocks"]
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Evidence:\n{context}\n\nQuestion: {question['question']}",
            },
        ]
        owner = self.processor or self.tokenizer
        options = {"tokenize": False, "add_generation_prompt": True}
        if not self.candidate.startswith("falcon-"):
            if "enable_thinking" not in str(
                getattr(owner, "chat_template", None) or self.tokenizer.chat_template
            ):
                raise ValueError(
                    "Pinned native chat template does not expose the required hard thinking switch"
                )
            options["enable_thinking"] = (
                self.protocol.mode(self.candidate) == "reasoning"
            )
        prompt = owner.apply_chat_template(messages, **options)
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        return {
            "prompt": prompt,
            "prompt_tokens": len(ids),
            "context_tokens": len(
                self.tokenizer.encode(context, add_special_tokens=False)
            ),
            "context_text": context,
            "messages": messages,
            "max_new_tokens": self.protocol.context_tokens - len(ids),
        }

    def load(self):
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            GenerationConfig,
        )

        self.torch = torch
        # Compatibility preloading is outside measurement. Reload native runtime
        # components here so the fresh cold-load timer includes their readiness.
        self.tokenizer = self.processor = None
        self._tokenizer()
        if self.resolved_decoder.get("source") == "pinned-generation-config":
            source = self.path / "generation_config.json"
            config = GenerationConfig.from_pretrained(self.path, local_files_only=True)
            if not source.is_file():
                raise ValueError(
                    "Falcon requires its pinned official generation configuration"
                )
            self.resolved_decoder = {
                key: getattr(config, key)
                for key in (
                    "do_sample",
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "repetition_penalty",
                )
                if getattr(config, key, None) is not None
            }
            self.decoder_source = {"path": str(source), "sha256": sha256_file(source)}
        loader = (
            AutoModelForImageTextToText
            if GENERATOR_LOADERS[self.candidate] == "multimodal-lm"
            else AutoModelForCausalLM
        )
        self.model = loader.from_pretrained(
            self.path, local_files_only=True, torch_dtype=getattr(torch, self.dtype)
        )
        # Explicit whole-device placement; never device_map=auto or quantization.
        self.model = self.model.to(self.device).eval()
        self.inherited_generation_config = self.model.generation_config.to_dict()
        self.synchronize()

    def synchronize(self):
        if self.torch is not None and self.device == "cuda":
            self.torch.cuda.synchronize()

    def placement(self):
        return inspect_model_placement(self.model, expected_device=self.device)

    def infer(self, prepared, seed):
        from transformers import (
            LogitsProcessor,
            LogitsProcessorList,
            StoppingCriteria,
            StoppingCriteriaList,
        )

        request_started = time.perf_counter()
        torch = self.torch
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        encoded = self.tokenizer(
            prepared["prompt"],
            add_special_tokens=False,
            return_tensors="pt",
            truncation=False,
        )
        if (
            int(encoded["input_ids"].shape[-1]) != prepared["prompt_tokens"]
            or prepared["max_new_tokens"] <= 0
        ):
            raise ValueError("Native input changed or has no output capacity")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        events, token_ids, first_put = [], [], True
        backend = self

        class EventStreamer:
            def put(self, value):
                nonlocal first_put
                if first_put:
                    first_put = False  # generate's first callback is the prompt.
                    return
                backend.synchronize()
                ids = value.detach().cpu().reshape(-1).tolist()
                now = time.perf_counter() - request_started
                token_ids.extend(ids)
                events.extend([now] * len(ids))

            def end(self):
                pass

        class Deadline(StoppingCriteria):
            reached = False

            def __call__(self, input_ids, scores, **kwargs):
                self.reached = (
                    time.perf_counter() - started
                    >= backend.protocol.streamer_timeout_seconds
                )
                return self.reached

        class PresencePenalty(LogitsProcessor):
            def __call__(self, input_ids, scores):
                seen = input_ids[0, prepared["prompt_tokens"] :].unique()
                scores[0, seen] -= presence
                return scores

        options = dict(self.resolved_decoder)
        presence = options.pop("presence_penalty", 0)
        stop = Deadline()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        eos.discard(None)
        if not eos:
            raise ValueError(
                "Pinned backend has no observable EOS termination contract"
            )
        started = time.perf_counter()
        error = None
        try:
            with torch.inference_mode():
                started = time.perf_counter()
                self.model.generate(
                    **encoded,
                    **options,
                    max_new_tokens=prepared["max_new_tokens"],
                    streamer=EventStreamer(),
                    stopping_criteria=StoppingCriteriaList([stop]),
                    logits_processor=LogitsProcessorList(
                        [PresencePenalty()] if presence else []
                    ),
                )
            self.synchronize()
        except Exception as exc:  # noqa: BLE001 - retain backend failures and emitted token events
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        # Record the first real terminal cause; EOS at the last permitted token wins
        # over context exhaustion, and late parsing does not manufacture a timeout.
        call_offset = started - request_started
        eos_at = next(
            (
                stamp - call_offset
                for token, stamp in zip(token_ids, events, strict=True)
                if token in eos
            ),
            None,
        )
        if error:
            outcome, finish = "runtime_error", "backend_exception"
        elif eos_at is not None and eos_at <= self.protocol.streamer_timeout_seconds:
            outcome, finish = "completed", "eos"
        elif stop.reached or elapsed >= self.protocol.streamer_timeout_seconds:
            outcome, finish = "timeout", "timeout"
        elif len(token_ids) >= prepared["max_new_tokens"]:
            outcome, finish = "context_limit", "context_boundary"
        else:
            outcome, finish = "runtime_error", "unresolved_termination"
            error = (
                "Backend returned without observable EOS, context boundary, or timeout"
            )
        raw = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        visible, reasoning, reasoning_count = self._separate(token_ids, prepared)
        response, _ = parse_response(visible)
        return {
            "outcome": outcome,
            "finish_reason": finish,
            "error": error,
            "raw_text": raw,
            "visible_text": visible,
            "reasoning_text": reasoning,
            "generated_token_ids": token_ids,
            "token_event_seconds": events,
            "timing": {
                "generation_seconds": elapsed,
                "ttft_seconds": events[0] - call_offset if events else None,
                "generation_call_offset_seconds": call_offset,
                "first_token_seconds": events[0] if events else None,
                "last_token_seconds": events[-1] if events else None,
            },
            "workload": {
                "prompt_tokens": prepared["prompt_tokens"],
                "context_tokens": prepared["context_tokens"],
                "reasoning_tokens": reasoning_count,
                "total_output_tokens": len(token_ids),
                "visible_answer_tokens": len(
                    self.tokenizer.encode(response["answer"], add_special_tokens=False)
                )
                if response
                else None,
            },
        }

    def _separate(self, ids, prepared):
        close = self.tokenizer.encode("</think>", add_special_tokens=False)
        opening = self.tokenizer.encode("<think>", add_special_tokens=False)
        for index in range(len(ids) - len(close) + 1):
            if ids[index : index + len(close)] == close:
                start = len(opening) if ids[: len(opening)] == opening else 0
                reasoning = self.tokenizer.decode(
                    ids[start:index], skip_special_tokens=True
                )
                return (
                    self.tokenizer.decode(
                        ids[index + len(close) :], skip_special_tokens=True
                    ).strip(),
                    reasoning,
                    index - start,
                )
        visible = self.tokenizer.decode(ids, skip_special_tokens=True).strip()
        # Direct mode's verified hard switch establishes known absence, not a guess
        # from a fast response. Unexposed reasoning remains unavailable otherwise.
        return (
            visible,
            None,
            0 if self.protocol.mode(self.candidate) == "direct" else None,
        )

    def close(self):
        self.model = self.torch = self.tokenizer = self.processor = None
        release_runtime_memory()


def generator_for(candidate, model_lock, device, dtype, protocol):
    """Retained factory name; returns the benchmark backend, not production code."""
    return LocalBackend(candidate, model_lock, device, dtype, protocol)
