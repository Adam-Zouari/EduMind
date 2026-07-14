"""Token counting, exact offsets, and truncation."""

from __future__ import annotations

import bisect
import re
from typing import Protocol


class OffsetTokenizer(Protocol):
    name: str

    def spans(self, text: str) -> list[tuple[int, int]]: ...

    def count(self, text: str) -> int: ...

    def truncate(self, text: str, maximum_tokens: int) -> str: ...


class RegexOffsetTokenizer:
    name = "regex-smoke-v1"
    _pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def spans(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in self._pattern.finditer(text)]

    def count(self, text: str) -> int:
        return len(self.spans(text))

    def truncate(self, text: str, maximum_tokens: int) -> str:
        spans = self.spans(text)
        if len(spans) <= maximum_tokens:
            return text
        return text[: spans[maximum_tokens - 1][1]] if maximum_tokens > 0 else ""


class TiktokenOffsetTokenizer:
    """Tiktoken with UTF-8 byte-to-character boundary reconstruction."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        try:
            import tiktoken
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "tiktoken is required for runtime token chunking; install .[rag]"
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
    def __init__(self, model_name: str, revision: str = "main") -> None:
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is required for model-tokenizer benchmark chunking"
            ) from exc
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            use_fast=True,
            local_files_only=True,
        )
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

    def __init__(self, model_name: str, revision: str) -> None:
        self.model_name = model_name
        self.revision = revision
        self.name = f"huggingface:{model_name}@{revision}"
        self._runtime: HuggingFaceOffsetTokenizer | None = None

    def _get(self) -> HuggingFaceOffsetTokenizer:
        if self._runtime is None:
            try:
                self._runtime = HuggingFaceOffsetTokenizer(self.model_name, self.revision)
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
