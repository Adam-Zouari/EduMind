"""Token counting, exact offsets, and truncation."""

from __future__ import annotations

import base64
import bisect
import hashlib
import os
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from edumind.common.artifacts import atomic_write_text
from edumind.common.models import ModelPreparationError
from edumind.common.paths import PROJECT_ROOT

_CL100K_URL = (
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
)
_CL100K_KEY = hashlib.sha1(_CL100K_URL.encode(), usedforsecurity=False).hexdigest()
_CL100K_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
_READY_FILE = "cl100k_base.ready"


def tokenizer_cache_directory() -> Path:
    return PROJECT_ROOT / "data/models/tiktoken"


def _read_tiktoken(cache_directory: Path) -> bytes | None:
    try:
        content = (cache_directory / _CL100K_KEY).read_bytes()
    except FileNotFoundError:
        if not (cache_directory / _READY_FILE).exists():
            return None
        raise ModelPreparationError(
            f"Prepared tokenizer file is missing from {cache_directory}"
        ) from None
    if hashlib.sha256(content).hexdigest() != _CL100K_SHA256:
        raise ModelPreparationError(
            f"Pinned tokenizer checksum mismatch: {cache_directory}"
        )
    return content


def tiktoken_ready() -> bool:
    cache_directory = tokenizer_cache_directory()
    return (
        _read_tiktoken(cache_directory) is not None
        and (cache_directory / _READY_FILE).is_file()
    )


def prepare_tiktoken(cache_directory: Path | None = None) -> None:
    """Prepare the shared tokenizer explicitly; runtime never fetches it."""
    from tiktoken.load import read_file_cached

    cache_directory = cache_directory or tokenizer_cache_directory()
    if _read_tiktoken(cache_directory) is None:
        previous = os.environ.get("TIKTOKEN_CACHE_DIR")
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_directory)
        try:
            read_file_cached(_CL100K_URL, expected_hash=_CL100K_SHA256)
            if _read_tiktoken(cache_directory) is None:
                raise ModelPreparationError(
                    f"Tokenizer download missing from {cache_directory}"
                )
        finally:
            if previous is None:
                os.environ.pop("TIKTOKEN_CACHE_DIR", None)
            else:
                os.environ["TIKTOKEN_CACHE_DIR"] = previous
    if not (cache_directory / _READY_FILE).is_file():
        atomic_write_text(cache_directory / _READY_FILE, "cl100k_base\n")


class OffsetTokenizer(Protocol):
    name: str

    def spans(self, text: str) -> list[tuple[int, int]]: ...

    def count(self, text: str) -> int: ...

    def truncate(self, text: str, maximum_tokens: int) -> str: ...


class TiktokenOffsetTokenizer:
    """Tiktoken with UTF-8 byte-to-character boundary reconstruction."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        if encoding_name != "cl100k_base":
            raise RuntimeError(f"Unsupported pinned tiktoken encoding: {encoding_name}")
        cache_directory = tokenizer_cache_directory()
        try:
            content = _read_tiktoken(cache_directory)
        except (OSError, ModelPreparationError) as exc:
            raise ModelPreparationError(f"{exc}. {_PREPARATION_GUIDANCE}") from exc
        if content is None or not (cache_directory / _READY_FILE).is_file():
            raise ModelPreparationError(_PREPARATION_GUIDANCE)
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


_PREPARATION_GUIDANCE = (
    "Pinned tiktoken cl100k_base assets are missing or corrupt. Use "
    "`python -m edumind.prepare_models` for the app or rerun the benchmark's "
    "model preparation command. Runtime tokenizer loading is offline."
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
