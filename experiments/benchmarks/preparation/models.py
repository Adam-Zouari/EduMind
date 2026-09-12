"""Prepare immutable model snapshots and validate the generated model lock."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, atomic_write_text, sha256_file, stable_hash
from edumind.extraction.extractors.document import DOCLING_VERSION
from edumind.rag.contracts import PRODUCTION_EMBEDDING_MODEL

from experiments.benchmarks.common.selection import SelectionEntry, selection_entries

MODEL_COMPONENTS = frozenset(
    {"embedding", "reranker", "generator", "evaluator", "asr", "document_extraction"}
)
RAG_COMPONENTS = frozenset({"embedding", "reranker", "generator", "evaluator"})
EMBEDDING_COMPONENTS = frozenset({"embedding"})
EXTRACTION_COMPONENTS = frozenset({"asr", "document_extraction"})
EMBEDDING_SNAPSHOT_IGNORE_PATTERNS = (
    "onnx/**",
    "openvino/**",
    "*.gguf",
    "*.onnx",
    "*.tflite",
    "tf_model.h5",
    "flax_model.msgpack",
)
APP_CANDIDATES = (
    PRODUCTION_EMBEDDING_MODEL,
    "Qwen/Qwen3-1.7B",
    "openai/whisper-small.en",
)
DOCLING_STANDARD = "docling-standard"
DOCLING_APP_COMPONENTS = ("layout", "tableformer", "rapidocr")
DOCLING_BENCHMARK_COMPONENTS = (
    "layout",
    "tableformer",
    "code_formula",
    "rapidocr",
    "easyocr",
    "tesseract-cli",
)


def prepare_app_models(root: Path, *, dry_run: bool = False) -> list[Path]:
    """Prepare only the provisional production controls."""
    output = root / "data/benchmarks/models/selected.json"
    prepare_selected_models(
        output,
        root / "data/benchmarks/downloads/models",
        APP_CANDIDATES,
        docling_components=DOCLING_APP_COMPONENTS,
        dry_run=dry_run,
    )
    return [output]


def prepare_selected_models(
    output_path: Path,
    cache_directory: Path,
    selected: Sequence[str],
    *,
    docling_components: Sequence[str] = (),
    dry_run: bool = False,
) -> Path:
    """Download only approved immutable snapshots into the project model directory."""
    entries = {entry.candidate: entry for entry in selection_entries()}
    unknown = sorted(set(selected) - set(entries))
    if unknown:
        raise ValueError(f"Candidates are not included in model selection: {', '.join(unknown)}")
    cache_directory = cache_directory.expanduser().resolve()
    if dry_run:
        print(json.dumps(preparation_plan(selected, docling_components), indent=2))
        return output_path
    huggingface_home = cache_directory.parent / "huggingface"
    # Keep the caller's normal HF_HOME so an existing `hf auth login` remains
    # available. Only transient Hub cache files belong in the project cache.
    os.environ["HF_HUB_CACHE"] = str(huggingface_home / "hub")
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "huggingface-hub is required; install requirements/benchmarks.lock"
        ) from exc
    cache_directory.mkdir(parents=True, exist_ok=True)
    for candidate in selected:
        entry = entries[candidate]
        snapshots = snapshot_specs(entry)
        ignore_patterns = (
            EMBEDDING_SNAPSHOT_IGNORE_PATTERNS
            if entry.component == "embedding"
            else ()
        )
        downloaded: list[dict[str, str]] = []
        for repository, revision, role in snapshots:
            local_directory = cache_directory / repository.replace("/", "--")
            snapshot_download(
                repo_id=repository,
                revision=revision,
                local_dir=local_directory,
                ignore_patterns=ignore_patterns or None,
            )
            downloaded.append(
                {
                    "role": role,
                    "repository": repository,
                    "revision": revision,
                    "model_path": str(local_directory),
                    "cache_manifest_sha256": _directory_manifest_hash(local_directory),
                }
            )
        primary = downloaded[0]
        lock_entry: dict[str, object] = {
            "component": entry.component,
            "provider": "huggingface",
            "model": primary["repository"],
            "revision": primary["revision"],
            "selection_revision": entry.revision,
            "model_path": primary["model_path"],
            "model_cache_manifest_sha256": primary["cache_manifest_sha256"],
        }
        if ignore_patterns:
            lock_entry["snapshot_ignore_patterns"] = list(ignore_patterns)
        if len(downloaded) > 1:
            lock_entry["submodels"] = downloaded
        if candidate == "PaddlePaddle/PaddleOCR-VL-1.6":
            paddle_cache = _prepare_paddle_components(
                cache_directory, Path(primary["model_path"])
            )
            lock_entry["paddle_cache_path"] = str(paddle_cache)
            lock_entry["paddle_cache_manifest_sha256"] = _directory_manifest_hash(
                paddle_cache
            )
        _merge_model_lock(output_path, {candidate: lock_entry})
    if docling_components:
        _prepare_docling_standard(output_path, cache_directory, docling_components)
    _prepare_tiktoken(cache_directory.parent / "tiktoken")
    return output_path


def selected_model_names(components: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        entry.candidate for entry in selection_entries() if entry.component in components
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
            }
        )
    return plan


def load_selected_model_lock(
    path: Path, *, candidates: Sequence[str] | None = None
) -> dict[str, dict[str, object]]:
    """Load and verify the requested entries from the generated runtime lock."""
    if not path.is_file():
        raise RuntimeError(
            f"Missing model lock {path}; run `python experiments/benchmarks/prepare.py "
            "app-models` for the controls or the stage-specific model preparation command"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema_version = int(payload.get("schema_version", 0))
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
            if schema_version >= 3 and "tesseract-cli" in entry.get(
                "prepared_components", []
            ):
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
            if submodels and isinstance(submodels, Sequence) and not isinstance(
                submodels, (str, bytes)
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
        model_path = entry.get("model_path")
        if model_path and not Path(str(model_path)).exists():
            raise RuntimeError(f"Prepared model path no longer exists: {model_path}")
        if schema_version >= 3 and model_path:
            _verify_directory_manifest(
                Path(str(model_path)),
                entry.get("model_cache_manifest_sha256"),
                f"{candidate} primary model",
            )
        submodels = entry.get("submodels", [])
        if isinstance(submodels, Sequence) and not isinstance(submodels, (str, bytes)):
            for submodel in submodels:
                if isinstance(submodel, Mapping) and not Path(
                    str(submodel.get("model_path", ""))
                ).exists():
                    raise RuntimeError(
                        f"Prepared submodel path no longer exists: {submodel.get('model_path')}"
                    )
                if schema_version >= 3 and isinstance(submodel, Mapping):
                    _verify_directory_manifest(
                        Path(str(submodel.get("model_path", ""))),
                        submodel.get("cache_manifest_sha256"),
                        f"{candidate} {submodel.get('role', 'submodel')}",
                    )
        if schema_version >= 3 and entry.get("paddle_cache_path"):
            _verify_directory_manifest(
                Path(str(entry["paddle_cache_path"])),
                entry.get("paddle_cache_manifest_sha256"),
                f"{candidate} Paddle components",
            )
        result[candidate] = entry
    missing = sorted(requested - set(result))
    if missing:
        raise RuntimeError("Model lock lacks requested candidates: " + ", ".join(missing))
    return result


def model_revisions(lock: Mapping[str, Mapping[str, object]]) -> dict[str, str]:
    return {
        name: str(entry.get("selection_revision", entry["revision"]))
        for name, entry in lock.items()
    }


def snapshot_specs(entry: SelectionEntry) -> tuple[tuple[str, str, str], ...]:
    if entry.candidate == "Qwen/Qwen3-ASR-1.7B-hf":
        asr_revision, aligner_revision = (
            item.split("@", 1)[1] for item in entry.revision.split("; ")
        )
        return (
            (entry.candidate, asr_revision, "asr"),
            ("Qwen/Qwen3-ForcedAligner-0.6B-hf", aligner_revision, "forced-aligner"),
        )
    revision = entry.revision
    if entry.candidate == "PaddlePaddle/PaddleOCR-VL-1.6":
        revision = next(
            item.removeprefix("model@")
            for item in entry.revision.split("; ")
            if item.startswith("model@")
        )
    if ";" in revision:
        raise ValueError(f"Unsupported composite revision for {entry.candidate}: {revision}")
    return ((entry.candidate, revision, "primary"),)


def _merge_model_lock(path: Path, updates: Mapping[str, object]) -> None:
    existing: dict[str, object] = {}
    allowed = {entry.candidate for entry in selection_entries()} | {DOCLING_STANDARD}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        models = payload.get("models") if isinstance(payload, Mapping) else None
        if not isinstance(models, Mapping):
            raise ValueError(f"Existing model lock is malformed: {path}")
        existing.update(
            {str(name): value for name, value in models.items() if str(name) in allowed}
        )
    existing.update(updates)
    atomic_write_json(
        path,
        {
            "schema_version": 3,
            "selection_package": "benchmark-candidates",
            "models": existing,
        },
    )


def _prepare_docling_standard(
    output_path: Path,
    cache_directory: Path,
    components: Sequence[str],
) -> None:
    installed = version("docling")
    if installed != DOCLING_VERSION:
        raise RuntimeError(f"Docling {DOCLING_VERSION} is required, but {installed} is installed")
    docling_directory = cache_directory / "docling-standard"
    downloadable = [
        component
        for component in components
        if component not in {"easyocr", "rapidocr", "tesseract-cli"}
    ]
    cli_name = "docling-tools.exe" if os.name == "nt" else "docling-tools"
    cli_path = Path(sys.executable).resolve().parent / cli_name
    if not cli_path.is_file():
        raise RuntimeError(f"The pinned Docling CLI is missing beside Python: {cli_path}")
    command = [str(cli_path), "models", "download", *downloadable]
    command.extend(["--output-dir", str(docling_directory)])
    subprocess.run(command, check=True)
    if "easyocr" in components:
        _prepare_easyocr_english(docling_directory)
    if "rapidocr" in components:
        # Docling's generic CLI fetches four backend/language combinations. The
        # benchmark freezes exactly ONNX Runtime + English, so prepare only that set.
        from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

        RapidOcrModel.download_models(
            backend="onnxruntime",
            lang="english",
            local_dir=docling_directory / RapidOcrModel._model_repo_folder,
            force=False,
            progress=True,
        )
    if not docling_directory.is_dir() or not any(docling_directory.rglob("*")):
        raise RuntimeError("Docling model preparation produced an empty artifact directory")
    system_components = _prepare_system_components(components)
    _merge_model_lock(
        output_path,
        {
            DOCLING_STANDARD: {
                "component": "document_extraction",
                "provider": "docling",
                "model": "standard-pipeline-artifacts",
                "revision": installed,
                "model_path": str(docling_directory),
                "prepared_components": list(components),
                "system_components": system_components,
                "preparation": {
                    "command": command,
                    "docling_cli_sha256": sha256_file(cli_path),
                    "easyocr": {
                        "languages": ["en"],
                        "detection_model": "craft",
                        "recognition_model": "english_g2",
                        "downloader": (
                            "docling.models.stages.ocr.easyocr_model."
                            "EasyOcrModel.download_models"
                        ),
                    },
                    "rapidocr": {
                        "backend": "onnxruntime",
                        "language": "english",
                        "downloader": (
                            "docling.models.stages.ocr.rapid_ocr_model."
                            "RapidOcrModel.download_models"
                        ),
                    },
                },
                "model_cache_manifest_sha256": _directory_manifest_hash(
                    docling_directory
                ),
            }
        },
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


def _prepare_tiktoken(cache_directory: Path) -> None:
    """Resolve the benchmark token counter during explicit preparation, never a run."""
    try:
        import tiktoken
    except ModuleNotFoundError as exc:
        raise RuntimeError("tiktoken is required; install the application lock") from exc
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_directory)
    tiktoken.get_encoding("cl100k_base").encode("EduMind preparation check")
    atomic_write_text(cache_directory / "cl100k_base.ready", "cl100k_base\n")


def _directory_manifest_hash(directory: Path) -> str:
    """Hash every prepared file name and digest, independent of absolute location."""

    if not directory.is_dir():
        raise RuntimeError(f"Prepared model directory is missing: {directory}")
    files = [path for path in directory.rglob("*") if path.is_file()]
    if not files:
        raise RuntimeError(f"Prepared model directory is empty: {directory}")
    return stable_hash(
        [
            {
                "path": path.relative_to(directory).as_posix(),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(files, key=lambda value: value.relative_to(directory).as_posix())
        ]
    )


def _verify_directory_manifest(directory: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or not expected:
        raise RuntimeError(f"Model lock lacks a cache-manifest checksum for {label}")
    actual = _directory_manifest_hash(directory)
    if actual != expected:
        raise RuntimeError(
            f"Prepared cache manifest checksum mismatch for {label}: "
            f"expected {expected}, computed {actual}"
        )


def _prepare_system_components(components: Sequence[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    if "tesseract-cli" not in components:
        return result
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
    available = [value.strip() for value in languages.stdout.splitlines()[1:] if value.strip()]
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
    result["tesseract-cli"] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "version": completed.stdout.splitlines()[0].strip(),
        "languages": available,
        "english_data_path": str(english_data.resolve()),
        "english_data_sha256": sha256_file(english_data),
    }
    return result


def _verify_system_component(value: object, label: str) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Model lock lacks the {label} identity")
    path = Path(str(value.get("path", "")))
    checksum = str(value.get("sha256", ""))
    if not path.is_file() or not checksum or sha256_file(path) != checksum:
        raise RuntimeError(f"Prepared {label} executable is missing or changed")
    if not str(value.get("version", "")).strip():
        raise RuntimeError(f"Model lock lacks the {label} version")
    english_data = Path(str(value.get("english_data_path", "")))
    english_checksum = str(value.get("english_data_sha256", ""))
    if (
        not english_data.is_file()
        or not english_checksum
        or sha256_file(english_data) != english_checksum
    ):
        raise RuntimeError(f"Prepared {label} English model data is missing or changed")
