from __future__ import annotations

import json

import pytest

from edumind.common.artifacts import atomic_write_json, local_file_lock, stable_hash


def test_atomic_json_and_stable_hash(tmp_path) -> None:
    path = tmp_path / "nested" / "value.json"
    atomic_write_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text()) == {"a": 1, "b": 2}
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_local_lock_times_out_when_held(tmp_path) -> None:
    path = tmp_path / "held.lock"
    with local_file_lock(path):
        with pytest.raises(TimeoutError):
            with local_file_lock(path, timeout_seconds=0.01):
                pass
    assert not path.exists()
