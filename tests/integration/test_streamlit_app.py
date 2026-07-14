from __future__ import annotations

from apps import streamlit_app
from edumind.app.state import DocumentRecord, DocumentStatus
from edumind.pipeline import PipelineQueryResult
from edumind.rag.types import AnswerResult, RetrievalHit


class SessionState(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Upload:
    name = "notes.pdf"

    @staticmethod
    def getbuffer():
        return b"pdf"


class FakeStreamlit:
    def __init__(self):
        self.session_state = SessionState()
        self.buttons: dict[str, bool] = {}
        self.uploads = []
        self.question = ""
        self.messages: list[tuple[str, str]] = []
        self.rerun_called = False

    def cache_resource(self, function):
        return function

    def file_uploader(self, *args, **kwargs):
        return self.uploads

    def button(self, label, **kwargs):
        return self.buttons.get(label, False)

    def text_area(self, *args, **kwargs):
        return self.question

    def select_slider(self, *args, **kwargs):
        return kwargs.get("value", 5)

    def checkbox(self, *args, **kwargs):
        return self.buttons.get("confirm", False)

    def tabs(self, labels):
        return [self for _ in labels]

    def spinner(self, *args, **kwargs):
        return self

    def expander(self, *args, **kwargs):
        return self

    def empty(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def progress(self, value):
        self.messages.append(("progress", str(value)))
        return self

    def rerun(self):
        self.rerun_called = True

    def __getattr__(self, name):
        if name in {
            "success",
            "warning",
            "error",
            "info",
            "caption",
            "markdown",
            "subheader",
            "write",
            "set_page_config",
            "title",
        }:
            return lambda message=None, **kwargs: self.messages.append((name, str(message)))
        raise AttributeError(name)


class FakeController:
    def __init__(self):
        self.generation_ready = True
        self.processed = 0
        self.reset_called = False

    def readiness(self):
        return {"generation_ready": self.generation_ready}

    def process_upload(self, filename, content, records, progress=None):
        self.processed += 1
        record = DocumentRecord(
            "checksum",
            filename,
            DocumentStatus.READY,
            characters=10,
            chunks=1,
            timings={"total_seconds": 0.1},
            warnings=("check",),
        )
        records["checksum"] = record.to_dict()
        return record, True

    def query(self, question, top_k=5):
        hit = RetrievalHit("1", "evidence", {"source": "notes.pdf", "page": 1}, 0.9, 1)
        answer = AnswerResult("answer [1]", [hit], "[1] evidence")
        return PipelineQueryResult(
            question,
            (hit,),
            answer,
            {"retrieval_seconds": 0.1, "generation_seconds": 0.2, "total_seconds": 0.3},
        )

    def reset(self, records):
        records.clear()
        self.reset_called = True


def test_streamlit_helpers_cover_upload_query_documents_and_readiness(monkeypatch) -> None:
    ui = FakeStreamlit()
    controller = FakeController()
    monkeypatch.setattr(streamlit_app, "st", ui)
    streamlit_app._render_readiness(controller)
    controller.generation_ready = False
    streamlit_app._render_readiness(controller)

    ui.uploads = [Upload()]
    ui.buttons["Extract and index"] = True
    streamlit_app._render_upload(controller)
    assert controller.processed == 1 and streamlit_app._records()

    ui.question = "What?"
    ui.buttons["Answer"] = True
    streamlit_app._render_query(controller)
    assert any(kind == "markdown" and "answer" in message for kind, message in ui.messages)

    ui.buttons["confirm"] = True
    ui.buttons["Reset local index"] = True
    streamlit_app._render_documents(controller)
    assert controller.reset_called and ui.rerun_called


def test_streamlit_main_is_thin_and_readiness_errors_are_safe(monkeypatch) -> None:
    ui = FakeStreamlit()
    controller = FakeController()
    monkeypatch.setattr(streamlit_app, "st", ui)
    monkeypatch.setattr(streamlit_app, "_controller", lambda: controller)
    streamlit_app.main()
    assert any(kind == "title" for kind, _ in ui.messages)

    controller.readiness = lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    streamlit_app._render_readiness(controller)
    assert any(kind == "error" and "offline" in message for kind, message in ui.messages)
