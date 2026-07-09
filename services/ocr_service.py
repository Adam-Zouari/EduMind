"""FastAPI service for OCR extraction."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from edumind.common.schemas import ServiceHealth
from edumind.ocr.core.pipeline import DataIngestionPipeline

app = FastAPI(title="EduMind OCR Service", version="0.1.0")
ocr_pipeline = DataIngestionPipeline()


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "OCR Extraction Service", "status": "running"}


@app.get("/health", response_model=ServiceHealth)
def health() -> ServiceHealth:
    return ServiceHealth(status="healthy")


@app.post("/extract")
async def extract_text(file: UploadFile = File(...)) -> JSONResponse:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        result = ocr_pipeline.process_file(tmp_path)
        os.unlink(tmp_path)
        return JSONResponse(
            content={
                "success": result.success,
                "text": result.text,
                "metadata": result.metadata,
                "format_type": result.format_type,
                "extraction_time": result.extraction_time,
            }
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/formats")
def supported_formats() -> dict[str, list[str]]:
    return {"formats": ["pdf", "docx", "png", "jpg", "jpeg", "html", "mp3", "wav", "mp4", "avi"]}
