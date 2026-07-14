"""Local-only FastAPI boundary for bounded multimodal extraction."""

from __future__ import annotations

import logging
import tempfile
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from edumind.common.config import load_settings
from edumind.extraction.errors import ExtractionError
from edumind.extraction.pipeline import ExtractionPipeline

logger = logging.getLogger(__name__)
app = FastAPI(title="EduMind Extraction Service", version="0.2.0")


@lru_cache(maxsize=1)
def get_extraction_pipeline() -> ExtractionPipeline:
    return ExtractionPipeline()


def _error(code: str, message: str, status: int, request_id: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": True,
            "code": code,
            "message": message,
            "request_id": request_id or str(uuid.uuid4()),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    logger.info("Invalid extraction request: %s", exc)
    return _error("invalid_request", "The request payload is invalid.", 422)


@app.get("/health/live")
def liveness() -> dict[str, object]:
    return {"status": "alive", "checks": {}}


@app.get("/health/ready")
def readiness() -> JSONResponse:
    try:
        formats = get_extraction_pipeline().supported_sources()
    except Exception as exc:
        logger.exception("Extraction readiness failed: %s", exc)
        return JSONResponse(
            status_code=503, content={"status": "not_ready", "checks": {"registry": False}}
        )
    return JSONResponse(
        content={"status": "ready", "checks": {"registry": True, "sources": formats}}
    )


@app.get("/health")
def health() -> JSONResponse:
    return readiness()


@app.get("/formats")
def supported_formats() -> dict[str, object]:
    return {"sources": get_extraction_pipeline().supported_sources()}


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)) -> JSONResponse:
    settings = load_settings()
    filename = Path(file.filename or "upload.bin").name
    suffix = Path(filename).suffix[:16]
    temporary_path: Path | None = None
    size = 0
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="edumind-upload-"
        ) as handle:
            temporary_path = Path(handle.name)
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.extraction.maximum_upload_bytes:
                    return _error(
                        "upload_too_large",
                        "Upload exceeds the "
                        f"{settings.extraction.maximum_upload_bytes} byte limit.",
                        413,
                    )
                handle.write(chunk)
        document = get_extraction_pipeline().extract(temporary_path)
        payload = document.to_dict()
        payload["source_name"] = filename
        payload["source_path"] = filename
        return JSONResponse(content={"success": True, "document": payload})
    except ExtractionError as exc:
        logger.warning("Extraction failed (%s): %s", exc.code, exc.detail or exc)
        return _error(exc.code, exc.public_message, 422 if exc.recoverable else 500)
    except Exception as exc:
        logger.exception("Unexpected extraction failure: %s", exc)
        return _error("internal_error", "Extraction failed. Check service logs for details.", 500)
    finally:
        await file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
