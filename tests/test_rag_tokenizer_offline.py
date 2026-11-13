"""Runtime token counting must never repair or fetch encoding assets."""

import hashlib

import pytest

from edumind.common.paths import PROJECT_ROOT
from edumind.rag.tokenizers import TiktokenOffsetTokenizer

URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
KEY = hashlib.sha1(URL.encode()).hexdigest()


@pytest.mark.parametrize("state", ["stale_marker", "corrupt", "missing_marker"])
def test_missing_or_corrupt_encoding_fails_without_any_loader_call(
    tmp_path, monkeypatch, state
):
    import tiktoken
    import tiktoken.load

    from edumind.common import paths

    cache = tmp_path / "data/benchmarks/downloads/tiktoken"
    cache.mkdir(parents=True)
    if state != "missing_marker":
        (cache / "cl100k_base.ready").write_text("cl100k_base\n")
    if state != "stale_marker":
        (cache / KEY).write_bytes(b"corrupt encoding")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        tiktoken,
        "get_encoding",
        lambda *a: pytest.fail("network-capable loader called"),
    )
    monkeypatch.setattr(
        tiktoken.load, "read_file", lambda *a: pytest.fail("downloader called")
    )
    before = {path.name: path.read_bytes() for path in cache.iterdir()}
    with pytest.raises(RuntimeError, match="prepare models"):
        TiktokenOffsetTokenizer()
    assert {path.name: path.read_bytes() for path in cache.iterdir()} == before


def test_verified_encoding_is_offline_even_with_invalid_external_cache(monkeypatch):
    import tiktoken
    import tiktoken.load

    assert (PROJECT_ROOT / "data/benchmarks/downloads/tiktoken" / KEY).is_file()
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "unprepared-user-cache")
    monkeypatch.setattr(
        tiktoken, "get_encoding", lambda *a: pytest.fail("global loader called")
    )
    monkeypatch.setattr(
        tiktoken.load, "read_file", lambda *a: pytest.fail("downloader called")
    )
    tokenizer = TiktokenOffsetTokenizer()
    assert tokenizer.encoding.encode("hello world") == [15339, 1917]
    text = "évidence 中文 <|endoftext|>"
    assert len(tokenizer.spans(text)) == tokenizer.count(text)
    assert tokenizer.truncate(text, 4) in text
