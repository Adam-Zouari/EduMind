"""Prepare immutable model snapshots and validate the generated model lock."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from fnmatch import fnmatch
from importlib.metadata import version
from pathlib import Path

from edumind.common.artifacts import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
    stable_hash,
)
from edumind.common.models import (
    directory_manifest_hash as _directory_manifest_hash,
)
from edumind.common.models import (
    model_lock_schema,
    validate_model_entry,
)
from edumind.common.models import (
    verify_directory_manifest as _verify_directory_manifest,
)
from edumind.common.models import (
    verify_system_component as _verify_system_component,
)
from edumind.extraction.extractors.document import DOCLING_VERSION
from edumind.rag.contracts import PRODUCTION_EMBEDDING_MODEL
from experiments.benchmarks.common.selection import SelectionEntry, selection_entries
from experiments.benchmarks.preparation.docling_pins import (
    FILE_COMPONENTS,
    HF_COMPONENTS,
)

MODEL_COMPONENTS = frozenset(
    {"embedding", "reranker", "generator", "asr", "document_extraction"}
)
RAG_COMPONENTS = frozenset({"embedding", "reranker", "generator"})
EMBEDDING_COMPONENTS = frozenset({"embedding"})
EXTRACTION_COMPONENTS = frozenset({"asr", "document_extraction"})
INFERENCE_SNAPSHOT_IGNORE_PATTERNS = (
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
# Application controls are not benchmark promotions. This pin is the historical
# Qwen3-1.7B row's candidate_revision in selection_evidence.csv (decision=exclude).
APP_GENERATOR = SelectionEntry(
    "generator",
    "Qwen/Qwen3-1.7B",
    "application-control",
    "b9352fbb8ce704292730cf54b3b1dceb2a808738",
    "https://huggingface.co/Qwen/Qwen3-1.7B",
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


def prepare_app_models(
    root: Path, *, dry_run: bool = False, output_path: Path | None = None
) -> list[Path]:
    """Prepare only the provisional production controls."""
    output = (
        output_path
        if output_path is not None
        else root / "data/benchmarks/models/selected.json"
    )
    prepare_selected_models(
        output,
        root / "data/benchmarks/downloads/models",
        APP_CANDIDATES,
        docling_components=DOCLING_APP_COMPONENTS,
        dry_run=dry_run,
        application=True,
    )
    return [output]


def prepare_selected_models(
    output_path: Path,
    cache_directory: Path,
    selected: Sequence[str],
    *,
    docling_components: Sequence[str] = (),
    dry_run: bool = False,
    application: bool = False,
) -> Path:
    """Download only approved immutable snapshots into the project model directory."""
    entries = {entry.candidate: entry for entry in selection_entries()}
    if application:
        entries[APP_GENERATOR.candidate] = APP_GENERATOR
    unknown = sorted(set(selected) - set(entries))
    if unknown:
        raise ValueError(
            f"Candidates are not included in model selection: {', '.join(unknown)}"
        )
    cache_directory = cache_directory.expanduser().resolve()
    if dry_run:
        print(json.dumps(preparation_plan(selected, docling_components), indent=2))
        return output_path
    huggingface_home = cache_directory.parent / "huggingface"
    # Keep the caller's normal HF_HOME so an existing `hf auth login` remains
    # available. Only transient Hub cache files belong in the project cache.
    os.environ["HF_HUB_CACHE"] = str(huggingface_home / "hub")
    cache_directory.mkdir(parents=True, exist_ok=True)
    for candidate in selected:
        entry = entries[candidate]
        snapshots = snapshot_specs(entry)
        ignore_patterns = snapshot_ignore_patterns(entry.component)
        downloaded: list[dict[str, str]] = []
        for repository, revision, role in snapshots:
            local_directory = _prepare_snapshot(
                cache_directory,
                repository,
                revision,
                ignore_patterns,
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
        entry.candidate
        for entry in selection_entries()
        if entry.component in components
    )


def snapshot_ignore_patterns(component: str) -> tuple[str, ...]:
    return (
        INFERENCE_SNAPSHOT_IGNORE_PATTERNS
        if component in {"embedding", "reranker"}
        else ()
    )


def preparation_plan(
    selected: Sequence[str], docling_components: Sequence[str] = ()
) -> list[dict[str, object]]:
    entries = {entry.candidate: entry for entry in selection_entries()}
    entries[APP_GENERATOR.candidate] = APP_GENERATOR
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
            f"Missing model lock {path}; run `python -m experiments.benchmarks.prepare "
            "app-models` for the controls or the stage-specific model preparation command"
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
        if candidate == APP_GENERATOR.candidate and not requested:
            continue  # Application-only entries coexist without entering a roster.
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


def _prepare_snapshot(
    cache_directory: Path,
    repository: str,
    revision: str,
    ignore_patterns: Sequence[str] = (),
) -> Path:
    """Publish only a complete inventory authenticated against the pinned Hub tree.

    Hub's local fallback is not proof of revision. Verify both its returned
    snapshot identity and each upstream git-blob/LFS digest before copying bytes
    into the project-owned revision directory. Old downloads are never removed.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise RuntimeError(
            f"An immutable commit is required for {repository}: {revision}"
        )
    target = cache_directory / repository.replace("/", "--") / revision
    receipt = target.with_name(revision + ".complete.json")
    identity = {
        "repository": repository,
        "revision": revision,
        "ignore_patterns": list(ignore_patterns),
    }
    if receipt.is_file():
        completed = json.loads(receipt.read_text(encoding="utf-8"))
        if completed.get("identity") != identity:
            raise RuntimeError(
                f"Prepared snapshot policy differs; use a separate cache: {target}"
            )
        _verify_directory_manifest(target, completed.get("manifest_sha256"), repository)
        _verify_hub_inventory(target, completed["files"])
        return target
    if target.exists():
        raise RuntimeError(
            f"Uncompleted snapshot cannot be relabelled: {target}; retain it and use a separate cache"
        )
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(repository, revision=revision, files_metadata=True)
    if info.sha != revision:
        raise RuntimeError(f"Hub resolved a different revision for {repository}")
    inventory = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if any(fnmatch(name, pattern) for pattern in ignore_patterns):
            continue
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or ".cache" in relative.parts
            or "\\" in name
        ):
            raise RuntimeError(f"Unsafe Hub snapshot path: {name}")
        lfs = sibling.lfs
        checksum = (
            (
                lfs.get("sha256")
                if isinstance(lfs, Mapping)
                else getattr(lfs, "sha256", None)
            )
            if lfs
            else sibling.blob_id
        )
        algorithm = "sha256" if lfs else "git-sha1"
        if (
            type(sibling.size) is not int
            or sibling.size < 0
            or not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}" if lfs else r"[0-9a-f]{40}", checksum)
        ):
            raise RuntimeError(
                f"Hub lacks immutable file metadata for {repository}/{name}"
            )
        inventory.append(
            {
                "path": name,
                "size": sibling.size,
                "algorithm": algorithm,
                "digest": checksum,
            }
        )
    if not inventory:
        raise RuntimeError(f"Hub snapshot inventory is empty: {repository}")
    hub_cache = cache_directory.parent / "huggingface/hub"
    downloaded = Path(
        snapshot_download(
            repo_id=repository,
            revision=revision,
            cache_dir=hub_cache,
            ignore_patterns=list(ignore_patterns) or None,
        )
    )
    expected = (
        hub_cache
        / ("models--" + repository.replace("/", "--"))
        / "snapshots"
        / revision
    )
    if downloaded.resolve() != expected.resolve():
        raise RuntimeError(
            f"Hub returned an unverifiable snapshot revision for {repository}: {downloaded}"
        )
    _verify_hub_inventory(downloaded, inventory)
    partial = target.with_name(revision + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    for item in inventory:
        destination = partial / item["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded / item["path"], destination)
    _verify_hub_inventory(partial, inventory)
    partial.replace(target)
    atomic_write_json(
        receipt,
        {
            "schema_version": 1,
            "identity": identity,
            "files": inventory,
            "manifest_sha256": _directory_manifest_hash(target),
            "snapshot_size_bytes": _snapshot_size_bytes(target),
        },
    )
    return target


def _verify_hub_inventory(
    directory: Path, inventory: Sequence[Mapping[str, object]]
) -> None:
    expected = {item["path"] for item in inventory}
    actual = {
        p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()
    }
    if not expected or actual != expected:
        raise RuntimeError(
            f"Prepared snapshot has missing, mixed or stale excluded files: {directory}"
        )
    for item in inventory:
        path = directory / str(item["path"])
        if path.stat().st_size != item["size"]:
            raise RuntimeError(f"Prepared snapshot file size differs: {path}")
        if item["algorithm"] == "sha256":
            digest = sha256_file(path)
        elif item["algorithm"] == "git-sha1":
            hasher = hashlib.sha1(usedforsecurity=False)
            hasher.update(f"blob {item['size']}\0".encode())
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
        else:
            raise RuntimeError("Unsupported snapshot digest algorithm")
        if digest != item["digest"]:
            raise RuntimeError(f"Prepared snapshot file content differs: {path}")


def _merge_model_lock(path: Path, updates: Mapping[str, Mapping[str, object]]) -> None:
    existing: dict[str, object] = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_lock_schema(payload)
        models = payload.get("models") if isinstance(payload, Mapping) else None
        if not isinstance(models, Mapping):
            raise ValueError(f"Existing model lock is malformed: {path}")
        existing.update(models)
    # Never upgrade untouched legacy entries into schema 3 without integrity.
    # Do not hash unrelated/missing snapshots: only their structure is in scope.
    for candidate, entry in existing.items():
        if candidate not in updates:
            validate_model_entry(entry, str(candidate), verify=False)
    for candidate, entry in updates.items():
        validate_model_entry(entry, candidate)
    existing.update(
        {
            candidate: {
                **entry,
                "snapshot_size_bytes": _snapshot_size_bytes(
                    Path(str(entry["model_path"]))
                ),
            }
            for candidate, entry in updates.items()
        }
    )
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
        raise RuntimeError(
            f"Docling {DOCLING_VERSION} is required, but {installed} is installed"
        )
    unknown = set(components) - set(DOCLING_BENCHMARK_COMPONENTS)
    if unknown:
        raise ValueError(f"Unknown Docling components: {sorted(unknown)}")
    previous = {}
    if output_path.is_file():
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        model_lock_schema(payload)
        previous = payload.get("models", {}).get(DOCLING_STANDARD, {})
    if previous:
        validate_model_entry(previous, DOCLING_STANDARD)
        if not previous.get("component_models"):
            raise RuntimeError(
                "Legacy Docling bundle lacks immutable component pins; retain it and prepare a separate lock/cache"
            )
    desired = sorted(set(components) | set(previous.get("prepared_components", [])))
    if previous and set(desired) == set(previous["prepared_components"]):
        return
    bundle_id = stable_hash(
        {
            "package": installed,
            "components": desired,
            "hf": HF_COMPONENTS,
            "files": FILE_COMPONENTS,
        }
    )
    docling_directory = cache_directory / "docling-bundles" / bundle_id
    partial = docling_directory.with_name(bundle_id + ".partial")
    if docling_directory.exists():
        raise RuntimeError(
            f"Unpublished Docling bundle already exists; retain it and use a separate cache: {docling_directory}"
        )
    partial.mkdir(parents=True, exist_ok=True)
    component_models = {}
    expected_bundle_files = set()
    systems = dict(previous.get("system_components", {}))
    for component in desired:
        if component == "tesseract-cli":
            if component not in systems:
                systems.update(_prepare_system_components((component,)))
            _verify_system_component(systems[component], "Tesseract CLI")
            continue
        old_component = previous.get("component_models", {}).get(component)
        if old_component:
            folder = old_component.get("directory", old_component.get("folder"))
            if not isinstance(folder, str) or Path(folder).name != folder:
                raise RuntimeError(f"Invalid Docling component directory: {component}")
            prepared = Path(previous["model_path"]) / folder
            _verify_directory_manifest(
                prepared, old_component.get("manifest_sha256"), component
            )
            shutil.copytree(prepared, partial / folder, dirs_exist_ok=True)
            expected_bundle_files.update(
                f"{folder}/{p.relative_to(prepared).as_posix()}"
                for p in prepared.rglob("*")
                if p.is_file()
            )
            component_models[component] = dict(old_component)
            continue
        if component in HF_COMPONENTS:
            repository, revision = HF_COMPONENTS[component]
            prepared = _prepare_snapshot(cache_directory, repository, revision)
            target = partial / repository.replace("/", "--")
            shutil.copytree(prepared, target, dirs_exist_ok=True)
            expected_bundle_files.update(
                f"{target.name}/{p.relative_to(prepared).as_posix()}"
                for p in prepared.rglob("*")
                if p.is_file()
            )
            component_models[component] = {
                "repository": repository,
                "revision": revision,
                "directory": target.name,
                "manifest_sha256": _directory_manifest_hash(target),
            }
        else:
            pin = FILE_COMPONENTS[component]
            target = partial / pin["folder"]
            if component == "easyocr":
                _prepare_easyocr_english(partial)
            else:
                from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

                RapidOcrModel.download_models(
                    backend="onnxruntime",
                    lang="english",
                    local_dir=target,
                    force=False,
                    progress=True,
                )
            expected = pin["files"]
            actual = {
                p.relative_to(target).as_posix(): sha256_file(p)
                for p in target.rglob("*")
                if p.is_file()
            }
            if actual != expected:
                raise RuntimeError(
                    f"Docling {component} differs from the pinned file inventory"
                )
            component_models[component] = {
                **pin,
                "manifest_sha256": _directory_manifest_hash(target),
            }
            expected_bundle_files.update(f"{target.name}/{name}" for name in expected)
    # Preserve verified systems/preparation metadata; construct a new bundle,
    # never mutate a directory referenced by the previous completed lock.
    actual_bundle_files = {
        p.relative_to(partial).as_posix() for p in partial.rglob("*") if p.is_file()
    }
    if actual_bundle_files != expected_bundle_files:
        raise RuntimeError("Docling bundle contains stale/unaccounted component files")
    partial.replace(docling_directory)
    _merge_model_lock(
        output_path,
        {
            DOCLING_STANDARD: {
                "component": "document_extraction",
                "provider": "docling",
                "model": "standard-pipeline-artifacts",
                "revision": installed,
                "model_path": str(docling_directory),
                "runtime_version": installed,
                "prepared_components": desired,
                "component_models": component_models,
                "system_components": systems,
                "preparation": {
                    **previous.get("preparation", {}),
                    "component_policy": "immutable-components-v1",
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
        raise RuntimeError(
            "tiktoken is required; install the application lock"
        ) from exc
    cache_directory.mkdir(parents=True, exist_ok=True)
    os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_directory)
    tiktoken.get_encoding("cl100k_base").encode("EduMind preparation check")
    atomic_write_text(cache_directory / "cl100k_base.ready", "cl100k_base\n")


def _snapshot_size_bytes(directory: Path) -> int:
    """Count logical snapshot-file bytes, excluding download and Python caches."""
    if not directory.is_dir():
        raise RuntimeError(f"Prepared model directory is missing: {directory}")
    sizes = [
        path.stat().st_size
        for path in directory.rglob("*")
        if not {".cache", "__pycache__"}.intersection(path.relative_to(directory).parts)
        and path.is_file()
    ]
    if not sizes:
        raise RuntimeError(f"Prepared model snapshot is empty: {directory}")
    return sum(sizes)


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
    result["tesseract-cli"] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "version": completed.stdout.splitlines()[0].strip(),
        "languages": available,
        "english_data_path": str(english_data.resolve()),
        "english_data_sha256": sha256_file(english_data),
    }
    return result
