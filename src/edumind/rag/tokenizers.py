"""Token counting, exact offsets, and truncation."""

from __future__ import annotations

import base64
import bisect
import hashlib
from functools import lru_cache
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
        if encoding_name != "cl100k_base":
            raise RuntimeError(f"Unsupported pinned tiktoken encoding: {encoding_name}")
        # The cache key and content digest are pinned by tiktoken's cl100k_base
        # constructor. Read once and construct directly from verified bytes: even
        # deletion after this read cannot trigger the network-capable loader.
        url = (
            "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
        )
        cache_path = cache_directory / hashlib.sha1(url.encode()).hexdigest()
        try:
            content = cache_path.read_bytes()
        except OSError as exc:
            raise RuntimeError(_PREPARATION_GUIDANCE) from exc
        if (
            not (cache_directory / f"{encoding_name}.ready").is_file()
            or hashlib.sha256(content).hexdigest() != _CL100K_SHA256
        ):
            raise RuntimeError(_PREPARATION_GUIDANCE)
        self.encoding = _offline_encoding(content)
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
            end = max(end, start)
            spans.append((start, end))
            byte_offset = byte_end
        return spans

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text, disallowed_special=()))

    def truncate(self, text: str, maximum_tokens: int) -> str:
        if maximum_tokens <= 0:
            return ""
        ids = self.encoding.encode(text, disallowed_special=())
        # A token boundary may split a UTF-8 character. Drop that incomplete
        # character instead of inventing a replacement character in the context.
        return self.encoding.decode(ids[:maximum_tokens], errors="ignore")


_CL100K_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
_PREPARATION_GUIDANCE = (
    "Pinned tiktoken cl100k_base assets are missing or corrupt; repair with "
    "python -m experiments.benchmarks.prepare models before execution. "
    "Runtime tokenizer loading is offline and does not download or repair assets."
)


@lru_cache(maxsize=1)
def _offline_encoding(content: bytes):
    try:
        import tiktoken
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tiktoken is required for runtime token chunking; install requirements/app.lock"
        ) from exc
    ranks = {
        base64.b64decode(token, validate=True): int(rank)
        for token, rank in (line.split() for line in content.splitlines() if line)
    }
    # Match the pinned cl100k_base regex and special-token vocabulary without
    # calling get_encoding/load_tiktoken_bpe (both may fetch missing assets).
    return tiktoken.Encoding(
        name="cl100k_base",
        pat_str=r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s",
        mergeable_ranks=ranks,
        special_tokens={
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
    )
