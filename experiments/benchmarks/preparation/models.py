"""Prepare immutable model snapshots and validate the generated model lock."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from importlib.metadata import version
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, atomic_write_text

from experiments.benchmarks.common.selection import SelectionEntry, selection_entries

MODEL_COMPONENTS = frozenset(
    {"embedding", "reranker", "generator", "evaluator", "asr", "document_extraction"}
)
RAG_COMPONENTS = frozenset({"embedding", "reranker", "generator", "evaluator"})
EXTRACTION_COMPONENTS = frozenset({"asr", "document_extraction"})
APP_CANDIDATES = (
    "sentence-transformers/all-MiniLM-L6-v2",
    "Qwen/Qwen3-1.7B",
    "openai/whisper-small.en",
)
DOCLING_STANDARD = "docling-standard"
DOCLING_VERSION = "2.117.0"


def prepare_app_models(root: Path, *, dry_run: bool = False) -> list[Path]:
    """Prepare only the provisional production controls."""
    output = root / "data/benchmarks/models/selected.json"
    prepare_selected_models(
        output,
        root / "data/benchmarks/downloads/models",
        APP_CANDIDATES,
        include_docling=True,
        dry_run=dry_run,
    )
    return [output]


def prepare_selected_models(
    output_path: Path,
    cache_directory: Path,
    selected: Sequence[str],
    *,
    include_docling: bool = False,
    dry_run: bool = False,
) -> Path:
    """Download only approved immutable snapshots into the project model directory."""
    entries = {entry.candidate: entry for entry in selection_entries()}
    unknown = sorted(set(selected) - set(entries))
    if unknown:
        raise ValueError(f"Candidates are not included in model selection: {', '.join(unknown)}")
    cache_directory = cache_directory.expanduser().resolve()
    if dry_run:
        print(json.dumps(preparation_plan(selected, include_docling), indent=2))
        return output_path
    huggingface_home = cache_directory.parent / "huggingface"
    os.environ["HF_HOME"] = str(huggingface_home)
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
        downloaded: list[dict[str, str]] = []
        for repository, revision, role in snapshots:
            local_directory = cache_directory / repository.replace("/", "--")
            snapshot_download(repo_id=repository, revision=revision, local_dir=local_directory)
            downloaded.append(
                {
                    "role": role,
                    "repository": repository,
                    "revision": revision,
                    "model_path": str(local_directory),
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
        }
        if len(downloaded) > 1:
            lock_entry["submodels"] = downloaded
        if candidate == "PaddlePaddle/PaddleOCR-VL-1.6":
            lock_entry["paddle_cache_path"] = str(
                _prepare_paddle_components(cache_directory, Path(primary["model_path"]))
            )
        _merge_model_lock(output_path, {candidate: lock_entry})
    if include_docling:
        _prepare_docling_standard(output_path, cache_directory)
    _prepare_tiktoken(cache_directory.parent / "tiktoken")
    return output_path


def selected_model_names(components: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        entry.candidate for entry in selection_entries() if entry.component in components
    )


def preparation_plan(
    selected: Sequence[str], include_docling: bool
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
    if include_docling:
        plan.append(
            {
                "candidate": DOCLING_STANDARD,
                "component": "document_extraction",
                "runtime_version": DOCLING_VERSION,
                "subcomponents": [
                    "layout",
                    "tableformer",
                    "code_formula",
                    "rapidocr",
                    "easyocr",
                    "tesseract-cli (system)",
                ],
            }
        )
    return plan


def load_selected_model_lock(path: Path) -> dict[str, dict[str, object]]:
    """Load the generated runtime lock and verify every recorded local snapshot."""
    if not path.is_file():
        raise RuntimeError(f"Missing model lock {path}; run the matching preparation command")
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_models = payload.get("models", {})
    if not isinstance(raw_models, Mapping) or not raw_models:
        raise RuntimeError(f"Model lock is empty or malformed: {path}")
    approved = {entry.candidate: entry for entry in selection_entries()}
    result: dict[str, dict[str, object]] = {}
    for raw_candidate, raw_entry in raw_models.items():
        candidate = str(raw_candidate)
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
        else:
            if str(entry.get("selection_revision", "")) != approved[candidate].revision:
                raise RuntimeError(
                    "Model lock selection revision is inconsistent with selection "
                    f"evidence: {candidate}"
                )
            expected = snapshot_specs(approved[candidate])
            actual = [(str(entry.get("model", "")), str(entry["revision"]), "primary")]
            submodels = entry.get("submodels", [])
            if isinstance(submodels, Sequence) and not isinstance(submodels, (str, bytes)):
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
        submodels = entry.get("submodels", [])
        if isinstance(submodels, Sequence) and not isinstance(submodels, (str, bytes)):
            for submodel in submodels:
                if isinstance(submodel, Mapping) and not Path(
                    str(submodel.get("model_path", ""))
                ).exists():
                    raise RuntimeError(
                        f"Prepared submodel path no longer exists: {submodel.get('model_path')}"
                    )
        result[candidate] = entry
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
            ("Qwen/Qwen3-ForcedAligner-0.6B", aligner_revision, "forced-aligner"),
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
            "schema_version": 2,
            "selection_package": "benchmark-candidates",
            "models": existing,
        },
    )


def _prepare_docling_standard(output_path: Path, cache_directory: Path) -> None:
    installed = version("docling")
    if installed != DOCLING_VERSION:
        raise RuntimeError(f"Docling {DOCLING_VERSION} is required, but {installed} is installed")
    docling_directory = cache_directory / "docling-standard"
    subprocess.run(
        [
            "docling-tools",
            "models",
            "download",
            "layout",
            "tableformer",
            "code_formula",
            "rapidocr",
            "easyocr",
            "--rapidocr-backend-lang",
            "onnxruntime:english",
            "--easyocr-lang",
            "en",
            "--output-dir",
            str(docling_directory),
        ],
        check=True,
    )
    if not docling_directory.is_dir() or not any(docling_directory.rglob("*")):
        raise RuntimeError("Docling model preparation produced an empty artifact directory")
    _merge_model_lock(
        output_path,
        {
            DOCLING_STANDARD: {
                "component": "document_extraction",
                "provider": "docling",
                "model": "standard-pipeline-artifacts",
                "revision": installed,
                "model_path": str(docling_directory),
            }
        },
    )


def _prepare_paddle_components(cache_directory: Path, model_path: Path) -> Path:
    """Resolve PaddleOCR-VL layout dependencies during preparation, never at runtime."""
    paddle_cache = cache_directory / "paddleocr-vl-1.6-components"
    paddle_cache.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(paddle_cache)
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    try:
        from paddleocr import PaddleOCRVL
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PaddleOCR-VL dependencies are required; install requirements/benchmarks.lock"
        ) from exc
    PaddleOCRVL(
        pipeline_version="v1.6",
        engine="transformers",
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

