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
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_directory)
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
