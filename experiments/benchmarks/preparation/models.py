"""Prepare immutable model snapshots and validate the generated model lock."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from edumind.common.artifacts import sha256_file
from edumind.common.docling_pins import FILE_COMPONENTS as APP_FILE_COMPONENTS
from edumind.common.docling_pins import HF_COMPONENTS as APP_HF_COMPONENTS
from edumind.common.model_assets import (
    DOCLING_STANDARD,
    prepare_hub_entry,
)
from edumind.common.model_assets import (
    _merge_model_lock as _write_model_lock,
)
from edumind.common.model_assets import (
    _prepare_docling_standard as _prepare_docling_bundle,
)
from edumind.common.models import directory_manifest_hash as _directory_manifest_hash
from edumind.common.models import model_lock_schema, validate_model_entry
from edumind.common.models import verify_system_component as _verify_system_component
from edumind.extraction.extractors.document import DOCLING_VERSION
from edumind.rag.tokenizers import prepare_tiktoken
from experiments.benchmarks.common.selection import SelectionEntry, selection_entries
from experiments.benchmarks.preparation.docling_pins import (
    FILE_COMPONENTS as EXTRA_FILE_COMPONENTS,
)
from experiments.benchmarks.preparation.docling_pins import (
    HF_COMPONENTS as EXTRA_HF_COMPONENTS,
)

MODEL_COMPONENTS = frozenset(
    {"embedding", "reranker", "generator", "asr", "document_extraction"}
)
RAG_COMPONENTS = frozenset({"embedding", "reranker", "generator"})
EMBEDDING_COMPONENTS = frozenset({"embedding"})
EXTRACTION_COMPONENTS = frozenset({"asr", "document_extraction"})
HF_COMPONENTS = {**APP_HF_COMPONENTS, **EXTRA_HF_COMPONENTS}
FILE_COMPONENTS = {**APP_FILE_COMPONENTS, **EXTRA_FILE_COMPONENTS}
DOCLING_BENCHMARK_COMPONENTS = (
    "layout",
    "tableformer",
    "code_formula",
    "rapidocr",
    "easyocr",
    "tesseract-cli",
)


SELECTION_PACKAGE = "benchmark-candidates"


def prepare_selected_models(
    output_path: Path,
    cache_directory: Path,
    selected: Sequence[str],
    *,
    docling_components: Sequence[str] = (),
) -> Path:
    """Download only approved immutable snapshots into the project model directory."""
    entries = {entry.candidate: entry for entry in selection_entries()}
    unknown = sorted(set(selected) - set(entries))
    if unknown:
        raise ValueError(
            f"Candidates are not included in model selection: {', '.join(unknown)}"
        )
    cache_directory = cache_directory.expanduser().resolve()
    # Keep HF_HOME so an existing Hub login remains available.
    os.environ["HF_HUB_CACHE"] = str(cache_directory.parent / "huggingface/hub")
    cache_directory.mkdir(parents=True, exist_ok=True)
    for candidate in selected:
        entry = entries[candidate]
        lock_entry = prepare_hub_entry(
            cache_directory, entry.component, entry.revision, snapshot_specs(entry)
        )
        if candidate == "PaddlePaddle/PaddleOCR-VL-1.6":
            paddle_cache = _prepare_paddle_components(
                cache_directory, Path(str(lock_entry["model_path"]))
            )
            lock_entry["paddle_cache_path"] = str(paddle_cache)
            lock_entry["paddle_cache_manifest_sha256"] = _directory_manifest_hash(
                paddle_cache
            )
        _merge_model_lock(output_path, {candidate: lock_entry})
    if docling_components:
        _prepare_docling_standard(output_path, cache_directory, docling_components)
    prepare_tiktoken()
    return output_path


def selected_model_names(components: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        entry.candidate
        for entry in selection_entries()
        if entry.component in components
    )


def preparation_plan(
    selected: Sequence[str], docling_components: Sequence[str] = ()
) -> list[dict[str, object]]:
    entries = {entry.candidate: entry for entry in selection_entries()}
    plan = [
        {
            "candidate": candidate,
            "component": entries[candidate].component,
            "snapshots": [
                {"repository": repository, "revision": revision, "role": role}
                for repository, revision, role in snapshot_specs(entries[candidate])
            ],
        }
        for candidate in selected
    ]
    if docling_components:
        plan.append(
            {
                "candidate": DOCLING_STANDARD,
                "component": "document_extraction",
                "runtime_version": DOCLING_VERSION,
                "subcomponents": list(docling_components),
                "component_pins": {
                    name: {
                        "repository": HF_COMPONENTS[name][0],
                        "revision": HF_COMPONENTS[name][1],
                    }
                    if name in HF_COMPONENTS
                    else FILE_COMPONENTS[name]
                    for name in docling_components
                    if name != "tesseract-cli"
                },
            }
        )
    return plan


def load_selected_model_lock(
    path: Path, *, candidates: Sequence[str] | None = None
) -> dict[str, dict[str, object]]:
    """Load and verify the requested entries from the generated runtime lock."""
    if not path.is_file():
        raise RuntimeError(
            f"Missing model lock {path}; run the stage-specific benchmark preparation command"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    model_lock_schema(payload)
    raw_models = payload.get("models", {})
    if not isinstance(raw_models, Mapping) or not raw_models:
        raise RuntimeError(f"Model lock is empty or malformed: {path}")
    approved = {entry.candidate: entry for entry in selection_entries()}
    requested = set(candidates or ())
    unknown = sorted(
        candidate
        for candidate in requested
        if candidate != DOCLING_STANDARD and candidate not in approved
    )
    if unknown:
        raise RuntimeError(
            "Requested models are outside the approved selection: " + ", ".join(unknown)
        )
    result: dict[str, dict[str, object]] = {}
    for raw_candidate, raw_entry in raw_models.items():
        candidate = str(raw_candidate)
        if requested and candidate not in requested:
            continue
        if candidate == "Qwen/Qwen3-1.7B" and candidate not in approved:
            continue  # Read mixed locks written before app preparation was split.
        if candidate != DOCLING_STANDARD and candidate not in approved:
            raise RuntimeError(
                f"Model lock contains a candidate outside the approved selection: {candidate}"
            )
        if not isinstance(raw_entry, Mapping) or not raw_entry.get("revision"):
            raise RuntimeError(f"Malformed selected-model lock entry: {candidate}")
        entry = dict(raw_entry)
        if candidate == DOCLING_STANDARD:
            if str(entry["revision"]) != DOCLING_VERSION:
                raise RuntimeError(
                    f"Docling lock revision must be {DOCLING_VERSION}, received "
                    f"{entry['revision']}"
                )
            if (
                not isinstance(entry.get("component_models"), Mapping)
                or not entry["component_models"]
            ):
                raise RuntimeError(
                    "Docling lock lacks immutable component pins; prepare a separate lock/cache before authoritative execution"
                )
            if "tesseract-cli" in entry.get("prepared_components", []):
                system_components = entry.get("system_components")
                if not isinstance(system_components, Mapping):
                    raise RuntimeError(
                        "Docling lock lacks the prepared Tesseract system identity"
                    )
                _verify_system_component(
                    system_components.get("tesseract-cli"), "Tesseract CLI"
                )
        else:
            if str(entry.get("selection_revision", "")) != approved[candidate].revision:
                raise RuntimeError(
                    "Model lock selection revision is inconsistent with selection "
                    f"evidence: {candidate}"
                )
            expected = snapshot_specs(approved[candidate])
            actual = [(str(entry.get("model", "")), str(entry["revision"]), "primary")]
            submodels = entry.get("submodels", [])
            if (
                submodels
                and isinstance(submodels, Sequence)
                and not isinstance(submodels, (str, bytes))
            ):
                actual = [
                    (
                        str(item.get("repository", "")),
                        str(item.get("revision", "")),
                        str(item.get("role", "")),
                    )
                    for item in submodels
                    if isinstance(item, Mapping)
                ]
            if actual != list(expected):
                raise RuntimeError(
                    f"Model lock revision is inconsistent with selection evidence: {candidate}"
                )
        validate_model_entry(entry, candidate)
        result[candidate] = entry
    missing = sorted(requested - set(result))
    if missing:
        raise RuntimeError(
            "Model lock lacks requested candidates: " + ", ".join(missing)
        )
    return result


def model_revisions(lock: Mapping[str, Mapping[str, object]]) -> dict[str, str]:
    return {
        name: str(entry.get("selection_revision", entry["revision"]))
        for name, entry in lock.items()
    }


def snapshot_specs(entry: SelectionEntry) -> tuple[tuple[str, str, str], ...]:
    revision = entry.revision
    if entry.candidate == "PaddlePaddle/PaddleOCR-VL-1.6":
        revision = next(
            item.removeprefix("model@")
            for item in entry.revision.split("; ")
            if item.startswith("model@")
        )
    if ";" in revision:
        raise ValueError(
            f"Unsupported composite revision for {entry.candidate}: {revision}"
        )
    return ((entry.candidate, revision, "primary"),)


def _merge_model_lock(path: Path, updates: Mapping[str, Mapping[str, object]]) -> None:
    _write_model_lock(path, updates, selection_package=SELECTION_PACKAGE)


def _prepare_docling_standard(
    output_path: Path, cache_directory: Path, components: Sequence[str]
) -> None:
    _prepare_docling_bundle(
        output_path,
        cache_directory,
        components,
        hf_pins=HF_COMPONENTS,
        file_pins=FILE_COMPONENTS,
        file_downloaders={"easyocr": _prepare_easyocr_english},
        system_downloaders={"tesseract-cli": _prepare_tesseract},
        selection_package=SELECTION_PACKAGE,
    )


def _prepare_easyocr_english(docling_directory: Path) -> None:
    """Prepare only the EasyOCR models selected by ``lang=['en']``, idempotently."""

    from docling.models.stages.ocr.easyocr_model import EasyOcrModel
    from easyocr.config import detection_models, recognition_models

    target = docling_directory / EasyOcrModel._model_repo_folder
    detector = str(detection_models["craft"]["filename"])
    recognizer = str(recognition_models["gen2"]["english_g2"]["filename"])
    if all((target / name).is_file() for name in (detector, recognizer)):
        return
    EasyOcrModel.download_models(
        detection_models=["craft"],
        recognition_models=["english_g2"],
        local_dir=target,
        force=False,
        progress=True,
    )


def _prepare_paddle_components(cache_directory: Path, model_path: Path) -> Path:
    """Resolve PaddleOCR-VL layout dependencies during preparation, never at runtime."""
    paddle_cache = cache_directory / "paddleocr-vl-1.6-components"
    paddle_cache.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddle_cache)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    try:
        # PaddleX imports ModelScope, which imports PyTorch. On Windows, load
        # PyTorch's DLLs first to avoid shared-library name collisions.
        import torch
        from paddleocr import PaddleOCRVL
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PaddleOCR-VL dependencies are required; install requirements/benchmarks.lock"
        ) from exc
    _ = torch.__version__
    PaddleOCRVL(
        pipeline_version="v1.6",
        vl_rec_backend="native",
        vl_rec_model_dir=str(model_path),
        device="cpu",
    )
    if not any(paddle_cache.rglob("*")):
        raise RuntimeError("PaddleOCR-VL component preparation produced no artifacts")
    return paddle_cache


def _prepare_tesseract() -> dict[str, object]:
    executable = shutil.which("tesseract")
    if not executable:
        raise RuntimeError(
            "The document benchmark includes Tesseract but tesseract is not on PATH"
        )
    completed = subprocess.run(
        [executable, "--version"], check=True, capture_output=True, text=True
    )
    languages = subprocess.run(
        [executable, "--list-langs"], check=True, capture_output=True, text=True
    )
    available = [
        value.strip() for value in languages.stdout.splitlines()[1:] if value.strip()
    ]
    if "eng" not in available:
        raise RuntimeError("The document benchmark requires Tesseract English data")
    match = re.search(r'in\s+"([^"]+)"', languages.stdout.splitlines()[0])
    english_data = (
        Path(match.group(1)) / "eng.traineddata"
        if match
        else Path(executable).resolve().parent / "tessdata" / "eng.traineddata"
    )
    if not english_data.is_file():
        raise RuntimeError("Cannot locate the selected Tesseract eng.traineddata file")
    path = Path(executable).resolve()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "version": completed.stdout.splitlines()[0].strip(),
        "languages": available,
        "english_data_path": str(english_data.resolve()),
        "english_data_sha256": sha256_file(english_data),
    }
