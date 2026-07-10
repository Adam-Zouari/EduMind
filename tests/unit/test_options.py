from __future__ import annotations

import pytest

from edumind.ocr.core.options import BatchProcessingOptions, ProcessFileOptions


def test_process_file_options_build_format_specific_kwargs() -> None:
    options = ProcessFileOptions(
        pdf_ocr_mode="force",
        include_layout=True,
        languages=["eng", "fra"],
        extra_kwargs={"custom": "value"},
    )

    assert options.build_extract_kwargs("pdf") == {
        "custom": "value",
        "languages": ["eng", "fra"],
        "pdf_ocr_mode": "force",
        "include_layout": True,
    }
    assert options.build_extract_kwargs("image") == {
        "custom": "value",
        "languages": ["eng", "fra"],
        "return_ocr_data": True,
    }


def test_batch_processing_options_force_sequential_when_parallel_is_disabled() -> None:
    options = BatchProcessingOptions(parallel=False, batch_strategy="threads")

    assert options.effective_strategy == "sequential"


def test_process_file_options_reject_invalid_pdf_ocr_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported pdf_ocr_mode"):
        ProcessFileOptions(pdf_ocr_mode="invalid")  # type: ignore[arg-type]
