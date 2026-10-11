"""GPU workload ownership across upload, indexing, and answering stages."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from edumind.application import EduMindPipeline
from edumind.common.gpu import exclusive_model_workload
from edumind.extraction import ExtractionProfile, ExtractionRequest, SourceKind
from edumind.extraction.errors import ExtractionBackendError
from edumind.extraction.extractors.audio import WhisperExtractor
from edumind.extraction.extractors.document import _require_cuda_ocr
from edumind.extraction.extractors.video import VideoExtractor


@pytest.mark.parametrize("failure", [False, True])
def test_extraction_releases_models_before_indexing_and_on_failure(failure):
    events = []

    def extract(*args, **kwargs):
        events.append("extract")
        if failure:
            raise RuntimeError("extraction failed")
        return SimpleNamespace(warnings=())

    pipeline = EduMindPipeline(
        extraction=SimpleNamespace(
            extract=extract, unload=lambda: events.append("release-extraction")
        ),
        rag=SimpleNamespace(
            unload=lambda: events.append("release-rag"),
            ingest_document=lambda _document: events.append("index"),
        ),
    )
    if failure:
        with pytest.raises(RuntimeError, match="extraction failed"):
            pipeline.process_file("source")
    else:
        pipeline.process_file("source")
    assert events == ["release-rag", "extract", "release-extraction"] + (
        [] if failure else ["index"]
    )


def test_upload_and_query_use_the_same_process_wide_lock():
    extracting, waiting, release, querying = (Event() for _ in range(4))

    def extract(*args, **kwargs):
        extracting.set()
        assert release.wait(5)
        return SimpleNamespace(warnings=())

    def query(*args, **kwargs):
        querying.set()
        return []

    pipeline = EduMindPipeline(
        extraction=SimpleNamespace(extract=extract, unload=lambda: None),
        rag=SimpleNamespace(unload=lambda: None, query=query),
    )

    def answer():
        waiting.set()
        return pipeline.query("question", generate_answer=False)

    with ThreadPoolExecutor(max_workers=2) as threads:
        upload = threads.submit(pipeline.process_file, "source", ingest=False)
        try:
            assert extracting.wait(5)
            answer_future = threads.submit(answer)
            assert waiting.wait(5)
            assert not querying.wait(0.05)
        finally:
            release.set()
        upload.result(timeout=5)
        answer_future.result(timeout=5)
    assert querying.is_set()
    # The shared lock remains usable after both operations finish.
    with exclusive_model_workload():
        pass


def test_video_releases_asr_before_selecting_or_parsing_frames(tmp_path, monkeypatch):
    events = []
    source = tmp_path / "video.mp4"
    source.write_bytes(b"fixture")
    audio = SimpleNamespace(
        extract=lambda *_args: SimpleNamespace(text="speech", segments=()),
        unload=lambda: events.append("release-asr"),
    )
    video = VideoExtractor(
        "fixed",
        audio_factory=lambda *_args: audio,
        fixed_interval_seconds=5,
        scene_threshold=0.35,
        maximum_hybrid_gap_seconds=10,
    )

    def decode(arguments):
        from pathlib import Path

        Path(arguments[-1]).write_bytes(b"audio")

    def frames(*_args):
        assert events == ["release-asr"]
        raise RuntimeError("stop before visual inference")

    monkeypatch.setattr(video, "_run_ffmpeg", decode)
    monkeypatch.setattr(video, "_extract_frames", frames)
    request = ExtractionRequest.from_path(
        source, profile=ExtractionProfile("video", "video-fixed", "test")
    )
    with pytest.raises(ExtractionBackendError, match="Video extraction failed"):
        video.extract(request, SourceKind.VIDEO)
    assert events == ["release-asr"]


def test_audio_unload_releases_its_runtime(monkeypatch):
    released = []
    monkeypatch.setattr(
        "edumind.extraction.extractors.audio.release_runtime_memory",
        lambda: released.append(True),
    )
    extractor = WhisperExtractor("test")
    extractor._runtime = object()
    extractor.unload()
    assert extractor._runtime is None
    assert released == [True]


def test_rapidocr_cuda_session_is_verified_and_cached(monkeypatch):
    import sys

    created = []

    def verified(original):
        created.append(original._model_path)
        native = object()
        return SimpleNamespace(
            _sess=native,
            _edumind_verified_native_session=native,
            _edumind_cuda_graph={"verification": "optimized-ort-graph-v1"},
            _enable_fallback=False,
        )

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        SimpleNamespace(preload_dlls=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="pinned"))
    monkeypatch.setattr(
        "edumind.extraction.extractors.document.cuda_shape_session", verified
    )
    reader = SimpleNamespace(
        **{
            name: SimpleNamespace(
                session=SimpleNamespace(session=SimpleNamespace(_model_path=name))
            )
            for name in ("text_det", "text_cls", "text_rec")
        }
    )
    converter = SimpleNamespace(
        initialize_pipeline=lambda _format: None,
        initialized_pipelines={
            "image": SimpleNamespace(ocr_model=SimpleNamespace(reader=reader)),
            "native-docx": SimpleNamespace(),
        },
    )
    _require_cuda_ocr(converter, "image")
    _require_cuda_ocr(converter, "image")
    assert created == ["text_det", "text_cls", "text_rec"]

    def reject(*args, **kwargs):
        created.append("rejected")
        raise RuntimeError("CPU graph node")

    monkeypatch.setattr(
        "edumind.extraction.extractors.document.cuda_shape_session", reject
    )
    reader.text_det.session.session = SimpleNamespace(_model_path="unverified")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="CPU graph node"):
            _require_cuda_ocr(converter, "image")
    assert created[-2:] == ["rejected", "rejected"]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_paddle_sets_framework_device_before_building_pipeline(
    monkeypatch, tmp_path, device
):
    import sys

    from experiments.benchmarks.extraction.document import adapters

    events = []
    paddle_env = SimpleNamespace(is_cuda_available=lambda: True)
    monkeypatch.setitem(sys.modules, "paddlex.utils", SimpleNamespace(env=paddle_env))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="pinned"))
    monkeypatch.setitem(
        sys.modules,
        "paddle",
        SimpleNamespace(
            __version__="3.3.1",
            is_compiled_with_cuda=lambda: True,
            set_device=lambda value: events.append(value),
        ),
    )

    def build(**options):
        events.append(options["device"])
        assert paddle_env.is_cuda_available() is (device == "cuda")
        return object()

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCRVL=build))
    monkeypatch.setattr(adapters, "version", lambda _package: "3.7.0")
    request = ExtractionRequest(
        tmp_path / "image.png",
        "checksum",
        profile=ExtractionProfile("paddle", "paddleocr-vl-1.6", "test", device=device),
        options={
            "model_path": str(tmp_path),
            "paddle_cache_path": str(tmp_path),
            "pipeline_version": "v1.6",
            "recognition_backend": "native",
        },
    )
    adapters.ExperimentalDocumentExtractor("paddleocr-vl-1.6", "test")._paddle_runtime(
        request
    )
    assert events == (["gpu:0", "gpu"] if device == "cuda" else ["cpu", "cpu"])
    assert paddle_env.is_cuda_available() is True  # The workaround is scoped.
