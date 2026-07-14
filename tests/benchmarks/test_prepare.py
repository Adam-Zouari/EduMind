from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from edumind.benchmarks import prepare
from edumind.benchmarks.prepare import (
    prepare_huggingface_models,
    prepare_ollama_models,
    prepare_public_assets,
    prepare_qasper,
)


def _paper(prefix: str, index: int) -> dict[str, object]:
    paper_id = f"{prefix}-{index:03d}"
    evidence = f"{paper_id} evidence text"
    return {
        "id": paper_id,
        "title": "Paper",
        "abstract": "Abstract",
        "full_text": [{"section_name": "Body", "paragraphs": [evidence]}],
        "qas": [
            {
                "question_id": f"q-{paper_id}",
                "question": "What is the evidence?",
                "answers": [
                    {
                        "answer": {
                            "unanswerable": False,
                            "extractive_spans": [evidence],
                            "free_form_answer": evidence,
                            "yes_no": False,
                            "evidence": [evidence],
                            "highlighted_evidence": [evidence],
                        }
                    }
                ],
            }
        ],
    }


def test_prepare_qasper_builds_isolated_valid_offset_manifests(monkeypatch, tmp_path) -> None:
    dataset = {
        "train": [_paper("train", index) for index in range(100)],
        "validation": [_paper("validation", index) for index in range(40)],
        "test": [_paper("test", index) for index in range(40)],
    }
    fake_module = SimpleNamespace(load_dataset=lambda *args, **kwargs: dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)
    outputs = prepare_qasper(tmp_path)
    assert [path.name for path in outputs] == [
        "qasper-dev.json",
        "qasper-validation.json",
        "qasper-locked-test.json",
    ]
    payload = json.loads(outputs[0].read_text(encoding="utf-8"))
    question = next(sample for sample in payload["samples"] if sample["kind"] == "question")
    evidence = question["evidence"][0]
    document = next(sample for sample in payload["samples"] if sample["kind"] == "document")
    assert document["text"][evidence["start"] : evidence["end"]].endswith("evidence text")


class FakeDownload:
    def __init__(self, content: bytes):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, chunk_size):
        yield self.content


def test_prepare_public_assets_requires_license_https_and_checksum(monkeypatch, tmp_path) -> None:
    content = b"licensed fixture"
    plan = tmp_path / "assets.json"
    plan.write_text(
        json.dumps(
            {
                "assets": [
                    {
                        "url": "https://example.test/fixture.bin",
                        "filename": "fixture.bin",
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "license": "CC-BY-4.0",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "edumind.benchmarks.prepare.requests.get",
        lambda *args, **kwargs: FakeDownload(content),
    )
    outputs = prepare_public_assets(plan, tmp_path / "raw")
    assert outputs[0].read_bytes() == content

    plan.write_text(json.dumps({"assets": [{"url": "http://unsafe"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="require HTTPS"):
        prepare_public_assets(plan, tmp_path / "bad")


def test_model_preparation_resolves_revisions_and_digests(monkeypatch, tmp_path) -> None:
    snapshots = []

    class API:
        def model_info(self, model):
            return SimpleNamespace(sha=f"revision-{model}")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=lambda: API(),
            snapshot_download=lambda **kwargs: snapshots.append(kwargs),
        ),
    )
    hf_lock = prepare_huggingface_models(tmp_path / "huggingface.json")
    assert len(snapshots) == len(prepare.HUGGINGFACE_MODELS)
    assert json.loads(hf_lock.read_text(encoding="utf-8"))["models"]

    pulled = []
    monkeypatch.setattr(
        prepare.subprocess,
        "run",
        lambda command, check: pulled.append(command),
    )

    class TagsResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "models": [
                    {"name": model, "digest": f"digest-{index}"}
                    for index, model in enumerate(prepare.OLLAMA_MODELS)
                ]
            }

    monkeypatch.setattr(prepare.requests, "get", lambda *args, **kwargs: TagsResponse())
    ollama_lock = prepare_ollama_models(tmp_path / "ollama.json")
    assert len(pulled) == len(prepare.OLLAMA_MODELS)
    assert json.loads(ollama_lock.read_text(encoding="utf-8"))["models"]
