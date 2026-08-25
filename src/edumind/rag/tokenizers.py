"""Token counting, exact offsets, and truncation."""

from __future__ import annotations

import os
import bisect
from typing import Protocol


class OffsetTokenizer(Protocol):
    name: str

    def spans(self, text: str) -> list[tuple[int, int]]: ...

    def count(self, text: str) -> int: ...

    def truncate(self, text: str, maximum_tokens: int) -> str: ...


class TiktokenOffsetTokenizer:
    """Tiktoken with UTF-8 byte-to-character boundary reconstruction."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        from edumind.common.paths import PROJECT_ROOT

        cache_directory = PROJECT_ROOT / "data/benchmarks/downloads/tiktoken"
        if not (cache_directory / f"{encoding_name}.ready").is_file():
            raise RuntimeError(
                f"Tiktoken encoding {encoding_name} is not prepared locally; run a model "
                "preparation target before this benchmark."
            )
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache_directory))
        try:
            import tiktoken
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "tiktoken is required for runtime token chunking; install requirements/app.lock"
            ) from exc
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.name = f"tiktoken:{encoding_name}"

    def spans(self, text: str) -> list[tuple[int, int]]:
        token_ids = self.encoding.encode(text, disallowed_special=())
        character_bytes = [0]
        for char in text:
            character_bytes.append(character_bytes[-1] + len(char.encode("utf-8")))
        spans: list[tuple[int, int]] = []
        byte_offset = 0
        for token_id in token_ids:
            token_bytes = self.encoding.decode_single_token_bytes(token_id)
            byte_end = byte_offset + len(token_bytes)
            start = max(0, bisect.bisect_right(character_bytes, byte_offset) - 1)
            end = min(len(text), bisect.bisect_left(character_bytes, byte_end))
            if end < start:
                end = start
            spans.append((start, end))
            byte_offset = byte_end
        return spans

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def truncate(self, text: str, maximum_tokens: int) -> str:
        if maximum_tokens <= 0:
            return ""
        ids = self.encoding.encode(text, disallowed_special=())
        return self.encoding.decode(ids[:maximum_tokens])


class HuggingFaceOffsetTokenizer:
    def __init__(
        self, model_name: str, revision: str = "main", local_path: str | None = None
    ) -> None:
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is required for exact model-tokenizer chunking"
            ) from exc
        options = {"use_fast": True, "local_files_only": True}
        if local_path is None:
            options["revision"] = revision
        self.tokenizer = AutoTokenizer.from_pretrained(local_path or model_name, **options)
        if not self.tokenizer.is_fast:
            raise RuntimeError(f"Tokenizer {model_name} does not expose exact offset mappings")
        self.name = f"huggingface:{model_name}@{revision}"

    def spans(self, text: str) -> list[tuple[int, int]]:
        payload = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return [(int(start), int(end)) for start, end in payload["offset_mapping"]]

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def truncate(self, text: str, maximum_tokens: int) -> str:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)[:maximum_tokens]
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))


class LazyHuggingFaceOffsetTokenizer:
    """Defer optional tokenizer loading until the first indexing/query operation."""

    def __init__(
        self, model_name: str, revision: str, local_path: str | None = None
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.local_path = local_path
        self.name = f"huggingface:{model_name}@{revision}"
        self._runtime: HuggingFaceOffsetTokenizer | None = None

    def _get(self) -> HuggingFaceOffsetTokenizer:
        if self._runtime is None:
            try:
                self._runtime = HuggingFaceOffsetTokenizer(
                    self.model_name, self.revision, self.local_path
                )
            except OSError as exc:
                raise RuntimeError(
                    f"Tokenizer {self.model_name}@{self.revision} is not prepared locally. "
                    "Run the model preparation command."
                ) from exc
        return self._runtime

    def spans(self, text: str) -> list[tuple[int, int]]:
        return self._get().spans(text)

    def count(self, text: str) -> int:
        return self._get().count(text)

    def truncate(self, text: str, maximum_tokens: int) -> str:
        return self._get().truncate(text, maximum_tokens)
