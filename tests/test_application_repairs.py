"""Behavioral regressions for application handoff A01–A08; no model inference."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from queue import Empty
from threading import Event
from types import SimpleNamespace

import pytest

from edumind.common.config import load_settings
from edumind.common.models import ModelPreparationError, directory_manifest_hash
from edumind.extraction import (
    ExtractedDocument,
    ExtractedSegment,
    ExtractionProfile,
    ExtractionRequest,
    ExtractionWarning,
    SegmentKind,
    SourceKind,
)
from edumind.extraction.extractors.audio import WhisperExtractor, transcribe_whisper
from edumind.extraction.extractors.base import build_document
from edumind.extraction.extractors.video import VideoExtractor
from edumind.extraction.pipeline import ExtractionPipeline
from edumind.extraction.registry import ExtractorRegistration, ExtractorRegistry
from edumind.rag.contracts import GenerationProfile
from edumind.rag.errors import GenerationError
from edumind.rag.llm_generator import HuggingFaceGenerator, _visible_answer
from edumind.rag.pipeline import RAGPipeline
from edumind.rag.text_chunker import TextChunker
from edumind.rag.types import RetrievalHit
from edumind.rag.vector_store import VectorStore
from edumind.ui.controller import AppController
from edumind.ui.state import DocumentStatus


def _model_lock(tmp_path, name="openai/whisper-small.en"):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    weights = snapshot / "weights.bin"
    weights.write_bytes(b"unit-test weights, not a model")
    entry = {
        "revision": "pinned-unit-test-revision",
        "model_path": str(snapshot),
        "model_cache_manifest_sha256": directory_manifest_hash(snapshot),
    }
    lock = tmp_path / "selected.json"

    def publish():
        lock.write_text(json.dumps({"schema_version": 3, "models": {name: entry}}))

    publish()
    return lock, entry, weights, publish


def test_verified_bundle_changes_cache_and_tampering_fails_before_cache(tmp_path):
    lock, entry, weights, publish = _model_lock(tmp_path)
    source = tmp_path / "audio.wav"
    source.write_bytes(b"unit-test source")
    calls = []

    class Extractor:
        def extract(self, request, kind):
            calls.append(request)
            return build_document(request, kind, request.profile, ["spoken text"])

    registry = ExtractorRegistry()
    registry.register(
        ExtractorRegistration(
            "whisper-small-en-control",
            frozenset({SourceKind.AUDIO}),
            Extractor,
        )
    )
    settings = load_settings(
        overrides={
            "models": {"lock_path": str(lock)},
            "extraction": {"cache_directory": str(tmp_path / "cache")},
        }
    )
    pipeline = ExtractionPipeline(settings, registry)
    first = pipeline.extract(source)
    assert pipeline.extract(source).cache_hit
    assert len(calls) == 1
    identity = first.metadata["model_bundles"]["model_bundle_identity"]
    weights.write_bytes(b"changed test-only weights")
    with pytest.raises(ModelPreparationError, match="checksum mismatch"):
        pipeline.extract(source)
    assert len(calls) == 1
    entry["model_cache_manifest_sha256"] = directory_manifest_hash(weights.parent)
    publish()
    changed = pipeline.extract(source)
    assert not changed.cache_hit
    assert len(calls) == 2
    assert identity != changed.metadata["model_bundles"]["model_bundle_identity"]


def test_video_verifies_both_snapshots_even_when_paths_are_supplied(tmp_path):
    lock, entry, weights, _publish = _model_lock(tmp_path)
    payload = json.loads(lock.read_text())
    payload["models"]["docling-standard"] = dict(entry)
    lock.write_text(json.dumps(payload))
    pipeline = ExtractionPipeline(
        load_settings(overrides={"models": {"lock_path": str(lock)}})
    )
    profile = ExtractionProfile(
        "video",
        "video-hybrid",
        "ffmpeg-system",
        options={
            "audio_model_path": entry["model_path"],
            "image_model_path": entry["model_path"],
        },
    )
    _, options = pipeline._prepare_profile(profile)
    assert options["audio_model_bundle_identity"]
    assert options["image_model_bundle_identity"]
    weights.write_bytes(b"tampered")
    with pytest.raises(ModelPreparationError, match="checksum mismatch"):
        pipeline._prepare_profile(profile)


def test_production_model_paths_cannot_bypass_lock(tmp_path):
    pipeline = ExtractionPipeline(
        load_settings(
            overrides={
                "models": {"lock_path": str(tmp_path / "missing.json")},
            }
        )
    )
    with pytest.raises(RuntimeError, match="Missing extraction model lock"):
        pipeline._prepare_profile(
            ExtractionProfile(
                "audio",
                "whisper-small-en-control",
                "explicit-revision",
                options={"model_path": str(tmp_path)},
            )
        )


def test_request_cannot_override_verified_model_path(tmp_path):
    lock, _entry, _weights, _publish = _model_lock(tmp_path)
    source = tmp_path / "audio.wav"
    source.write_bytes(b"source")
    pipeline = ExtractionPipeline(
        load_settings(overrides={"models": {"lock_path": str(lock)}})
    )
    with pytest.raises(RuntimeError, match="model_path disagrees"):
        pipeline.extract(source, options={"model_path": str(tmp_path)})


@pytest.mark.parametrize("timed", [False, True])
def test_provenance_survives_ingest_store_pack_context_and_view(
    tmp_path, monkeypatch, timed
):
    profile = ExtractionProfile("fixture", "fixture", "fixture")
    document = ExtractedDocument(
        "notes",
        "notes",
        SourceKind.AUDIO if timed else SourceKind.PDF,
        "fixture",
        None,
        "first\nsecond",
        (
            ExtractedSegment(
                "first",
                0,
                5,
                element_id="one",
                page_number=None if timed else 1,
                timestamp_start=1 if timed else None,
                timestamp_end=2 if timed else None,
                kind=SegmentKind.AUDIO if timed else SegmentKind.HEADING,
                structured_content={"level": 1},
            ),
            ExtractedSegment(
                "second",
                6,
                12,
                element_id="two",
                page_number=None if timed else 2,
                timestamp_start=3 if timed else None,
                timestamp_end=4 if timed else None,
            ),
        ),
        profile,
    )
    chunk = TextChunker(SimpleNamespace(split=lambda _: [(0, 12, 2)])).chunk_document(
        RAGPipeline._normalize_document(document)
    )[0]
    assert len(chunk.metadata["source_locations"]) == 2
    metadata = VectorStore._deserialize(VectorStore._serialize(chunk))
    assert metadata["source_locations"][0]["structured_content"] == {"level": 1}
    pipeline = object.__new__(RAGPipeline)
    pipeline.tokenizer = SimpleNamespace(
        count=lambda value: len(value.split()),
        truncate=lambda value, count: value.split("\n")[0] if count == 1 else value,
    )
    hit = RetrievalHit("chunk", document.text, metadata, 1, token_count=2)
    overhead = pipeline.tokenizer.count(f"[1] source={hit.source}; {hit.location}\n")
    packed = pipeline._pack_hits([hit], 1, overhead + 1)[0]
    assert packed.document == "first"
    assert packed.metadata["end"] == 5
    assert [item["element_id"] for item in packed.metadata["source_locations"]] == [
        "one"
    ]
    assert (
        document.text[packed.metadata["start"] : packed.metadata["end"]]
        == packed.document
    )
    context = pipeline.generate_context([packed])
    assert ("time=1–2s" if timed else "page=1") in context
    assert "second" not in context
    if not timed:
        assert hit.page == "1,2"
        assert packed.page == "1"

    from edumind.ui import streamlit_app

    labels = []
    monkeypatch.setattr(streamlit_app.st, "text_area", lambda *a, **kw: "question")
    monkeypatch.setattr(streamlit_app.st, "select_slider", lambda *a, **kw: 1)
    monkeypatch.setattr(streamlit_app.st, "button", lambda *a, **kw: True)
    monkeypatch.setattr(streamlit_app.st, "spinner", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(
        streamlit_app.st,
        "expander",
        lambda label: labels.append(label) or nullcontext(),
    )
    for name in ("caption", "subheader", "write"):
        monkeypatch.setattr(streamlit_app.st, name, lambda *a, **kw: None)
    controller = SimpleNamespace(
        pipeline=SimpleNamespace(rag=SimpleNamespace(top_k=1)),
        query=lambda *a, **kw: SimpleNamespace(answer=None, hits=[packed], timings={}),
    )
    streamlit_app._render_query(controller)
    assert packed.location in labels[0]


def test_upload_retry_and_logical_document_deduplication():
    calls = []
    indexed = {}

    def process(path, **kwargs):
        calls.append((kwargs["source_name"], Path(path).read_bytes()))
        if len(calls) == 1:
            raise RuntimeError("transient failure")
        indexed[kwargs["source_name"]] = Path(path).read_bytes()
        return SimpleNamespace(
            extraction=SimpleNamespace(source_kind=SourceKind.PDF, text="indexed"),
            ingest=SimpleNamespace(chunks_created=1),
            timings={},
            warnings=(),
        )

    controller = AppController(
        SimpleNamespace(
            extraction=SimpleNamespace(
                settings=SimpleNamespace(
                    extraction=SimpleNamespace(maximum_upload_bytes=1024)
                )
            ),
            process_file=process,
        )
    )
    records = {}
    failed, processed = controller.process_upload("a.pdf", b"shared", records)
    assert processed and failed.status is DocumentStatus.FAILED
    ready, processed = controller.process_upload("a.pdf", b"shared", records)
    assert processed and ready.status is DocumentStatus.READY
    assert controller.process_upload("a.pdf", b"shared", records)[1] is False
    assert controller.process_upload("b.pdf", b"old", records)[1] is True
    assert controller.process_upload("b.pdf", b"shared", records)[1] is True
    assert calls[-1] == ("b.pdf", b"shared")
    assert indexed == {"a.pdf": b"shared", "b.pdf": b"shared"}
    assert len(records) == 2
    assert {row["checksum"] for row in records.values()} == {ready.checksum}


@pytest.mark.parametrize("payload", [None, [], "speech", {}, {"text": None}])
def test_whisper_rejects_malformed_native_envelope(payload):
    with pytest.raises(ValueError, match="object with string text"):
        transcribe_whisper(lambda *a, **kw: payload, Path("fixture.wav"))


@pytest.mark.parametrize(
    "chunks,code",
    [
        ([{"text": "alpha", "timestamp": [0, 1]}], "transcript_chunk_mismatch"),
        ([{"text": "wrong words", "timestamp": [0, 1]}], "transcript_chunk_mismatch"),
        ([None, {"text": "alpha beta", "timestamp": [0, 1]}], "malformed_chunk"),
        ("bad", "malformed_chunks"),
        (
            [{"text": "alpha beta", "timestamp": [float("nan"), 1]}],
            "malformed_timestamp",
        ),
        ([{"text": "alpha beta", "timestamp": [2, 1]}], "malformed_timestamp"),
    ],
)
def test_whisper_preserves_complete_text_and_diagnoses_partial_chunks(
    tmp_path, monkeypatch, chunks, code
):
    runtime = lambda *a, **kw: {"text": "alpha beta", "chunks": chunks}
    transcript = transcribe_whisper(runtime, Path("fixture.wav"))
    assert transcript.text == "alpha beta"
    assert " ".join(str(s["text"]) for s in transcript.segments) == "alpha beta"
    assert code in {warning.code for warning in transcript.warnings}
    monkeypatch.setattr(
        "edumind.extraction.extractors.audio.load_whisper_runtime",
        lambda *_: (runtime, "float32"),
    )
    request = ExtractionRequest(
        Path("fixture.wav"),
        "fixture",
        profile=ExtractionProfile("audio", "whisper", "fixture"),
        options={"model_path": str(tmp_path)},
    )
    document = WhisperExtractor("fixture").extract(request, SourceKind.AUDIO)
    assert document.text == "alpha beta"
    assert code in {warning.code for warning in document.warnings}
    assert document.segments[0].kind is SegmentKind.AUDIO
    if code == "transcript_chunk_mismatch":
        assert document.metadata["native_segments"][0]["text"] == chunks[0]["text"]


def test_whisper_accepts_explicit_silence():
    result = transcribe_whisper(
        lambda *a, **kw: {"text": "", "chunks": []}, Path("fixture.wav")
    )
    assert result.text == "" and result.segments == () and result.warnings == ()


def test_video_keeps_native_structures_warnings_and_exact_offsets(
    tmp_path, monkeypatch
):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"test source")
    profile = ExtractionProfile("video", "video-fixed", "fixture")
    request = ExtractionRequest.from_path(source, profile=profile)

    def audio_extract(request, kind):
        return replace(
            build_document(
                request, kind, request.profile, ["speech"], timestamps=[(0, 1)]
            ),
            metadata={"audio_evidence": "native"},
        )

    def image_extract(request, kind):
        document = build_document(request, kind, request.profile, ["heading", "table"])
        return replace(
            document,
            segments=(
                replace(
                    document.segments[0],
                    kind=SegmentKind.HEADING,
                    element_id="heading",
                    structured_content={"level": 1},
                ),
                replace(
                    document.segments[1],
                    kind=SegmentKind.TABLE,
                    parent_id="heading",
                    structured_content={"rows": [["cell"]]},
                ),
            ),
            metadata={"frame_evidence": "native"},
            warnings=(
                ExtractionWarning("partial", "one block failed", segment_index=1),
            ),
        )

    video = VideoExtractor(
        "fixed",
        audio_factory=lambda *_: SimpleNamespace(extract=audio_extract),
        image_factory=lambda *_: SimpleNamespace(extract=image_extract),
        fixed_interval_seconds=5,
        scene_threshold=0.35,
        maximum_hybrid_gap_seconds=10,
    )
    monkeypatch.setattr(
        video, "_run_ffmpeg", lambda args: Path(args[-1]).write_bytes(b"test audio")
    )

    def frames(_source, directory, _policy):
        frame = directory / "frame.png"
        frame.write_bytes(b"test frame")
        return [(frame, 5)]

    monkeypatch.setattr(video, "_extract_frames", frames)
    document = video.extract(request, SourceKind.VIDEO)
    assert document.text == "speech\nheading\ntable"
    assert [s.kind for s in document.segments] == [
        SegmentKind.AUDIO,
        SegmentKind.HEADING,
        SegmentKind.TABLE,
    ]
    assert [s.metadata["modality"] for s in document.segments] == [
        "audio",
        "visual",
        "visual",
    ]
    assert document.segments[2].structured_content == {"rows": [["cell"]]}
    assert document.segments[2].parent_id == document.segments[1].element_id
    assert document.segments[1].timestamp_start == 5
    assert document.warnings[0].segment_index == 2
    assert "frame-0 at 5s" in document.warnings[0].message
    assert document.segments[1].metadata["native_document_metadata"] == {
        "frame_evidence": "native"
    }
    for segment in document.segments:
        assert document.text[segment.start : segment.end] == segment.text


@pytest.mark.parametrize(
    "text,prompt,answer,reasoning",
    [
        ("<think>private</think>", "", "", "private"),
        ("<think>interrupted", "", "", "interrupted"),
        (
            "private</think>Answer [1]",
            "<|im_start|>assistant\n<think>\n",
            "Answer [1]",
            "private",
        ),
        ("interrupted", "<|im_start|>assistant\n<think>\n", "", "interrupted"),
        ("Answer [1]", "<think>\n\n</think>\n", "Answer [1]", ""),
        ("<think>private</think>Answer [1]", "", "Answer [1]", "private"),
        (
            "<think>outer<think>inner</think>still private",
            "",
            "",
            "outerinnerstill private",
        ),
        ("private</think>", "", "", "private"),
    ],
)
def test_reasoning_is_never_used_as_visible_fallback(text, prompt, answer, reasoning):
    assert _visible_answer(text, prompt=prompt) == (answer, reasoning)


@pytest.mark.parametrize(
    "prompt,completion,answer",
    [
        ("<|im_start|>assistant\n<think>\n", "private only", ""),
        ("<|im_start|>assistant\n<think>\n", "private</think>Answer [1]", "Answer [1]"),
        ("<think>\n\n</think>\n", "Answer [1]", "Answer [1]"),
        ("direct prompt", "<think>private</think>", ""),
    ],
)
def test_generation_measurement_uses_rendered_template_boundaries(
    monkeypatch, prompt, completion, answer
):
    import transformers

    class Tokenizer:
        eos_token_id = 0

        def apply_chat_template(self, *a, **kw):
            return prompt

        def __call__(self, *a, **kw):
            return {
                "input_ids": SimpleNamespace(to=lambda _: SimpleNamespace(shape=[1, 1]))
            }

        def encode(self, text, **kwargs):
            return list(range(len(text.split())))

    class Streamer:
        def __init__(self, *a, **kw):
            pass

        def __iter__(self):
            return iter([completion])

    monkeypatch.setattr(transformers, "TextIteratorStreamer", Streamer)
    generator = HuggingFaceGenerator(
        GenerationProfile(
            "fixture",
            "fixture",
            "fixture",
            "cpu",
            "auto",
            True,
            0,
            False,
            42,
            100,
            10,
            1,
        )
    )
    generator._model = SimpleNamespace(generate=lambda **kw: None)
    generator._tokenizer = Tokenizer()
    generator._torch = SimpleNamespace(
        manual_seed=lambda _: None, inference_mode=nullcontext
    )
    result = generator.generate_measured_with_results("question", [])
    assert result.answer == answer
    assert result.answer_tokens == len(answer.split())
    if not answer:
        assert result.reasoning_tokens > 0


@pytest.mark.parametrize("failure", ["timeout", "consumer", "backend", "slow_cleanup"])
def test_stream_failure_cancels_and_joins_before_unload(monkeypatch, failure):
    import transformers

    exited = Event()
    started = Event()
    cancellation_seen, release_cleanup, unload_started, unloaded = (
        Event() for _ in range(4)
    )
    backend_error = RuntimeError("original backend failure")

    class Streamer:
        def __init__(self, *a, **kw):
            pass

        def __iter__(self):
            assert started.wait(5)
            if failure == "timeout":
                raise Empty()
            raise RuntimeError("consumer failed")

        def on_finalized_text(self, *a, **kw):
            pass

    class Model:
        def generate(self, **options):
            started.set()
            try:
                if failure == "backend":
                    raise backend_error
                while not options["stopping_criteria"][0]():
                    exited.wait(0.001)
                if failure == "slow_cleanup":
                    cancellation_seen.set()
                    assert release_cleanup.wait(5)
            finally:
                exited.set()

    tokenizer = SimpleNamespace(
        eos_token_id=0,
        apply_chat_template=lambda *a, **kw: "prompt",
    )

    class Tokenizer:
        eos_token_id = 0
        apply_chat_template = tokenizer.apply_chat_template

        def __call__(self, *a, **kw):
            return {
                "input_ids": SimpleNamespace(
                    shape=[1, 1], to=lambda _: SimpleNamespace(shape=[1, 1])
                )
            }

    generator = HuggingFaceGenerator(
        GenerationProfile(
            "fixture",
            "fixture",
            "fixture",
            "cpu",
            "auto",
            False,
            0,
            False,
            42,
            100,
            10,
            0.01,
        )
    )
    generator._model = Model()
    generator._tokenizer = Tokenizer()
    generator._torch = SimpleNamespace(
        manual_seed=lambda _: None, inference_mode=nullcontext
    )
    monkeypatch.setattr(transformers, "TextIteratorStreamer", Streamer)
    if failure == "slow_cleanup":

        def unload():
            unload_started.set()
            generator.unload()
            unloaded.set()

        with ThreadPoolExecutor(max_workers=2) as threads:
            request = threads.submit(generator.generate, "question", "evidence")
            try:
                assert cancellation_seen.wait(5)
                cleanup = threads.submit(unload)
                assert unload_started.wait(5)
                assert not unloaded.wait(0.05)
                assert generator._model is not None
            finally:
                release_cleanup.set()
            with pytest.raises(GenerationError):
                request.result(timeout=5)
            cleanup.result(timeout=5)
        assert exited.is_set() and unloaded.is_set()
        return
    with pytest.raises(GenerationError) as error:
        generator.generate("question", "evidence")
    assert exited.is_set()
    assert (
        error.value.__cause__ is backend_error
        if failure == "backend"
        else isinstance(
            error.value.__cause__, Empty if failure == "timeout" else RuntimeError
        )
    )
    generator.unload()
    assert generator._model is None
