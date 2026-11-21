"""Shared preparation of pinned local snapshots, bundles, and model locks."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from fnmatch import fnmatch
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.docling_pins import FILE_COMPONENTS, HF_COMPONENTS
from edumind.common.models import (
    ModelPreparationError,
    model_lock_schema,
    validate_model_entry,
)
from edumind.common.models import (
    directory_manifest_hash as _directory_manifest_hash,
)
from edumind.common.models import (
    verify_directory_manifest as _verify_directory_manifest,
)
from edumind.common.models import (
    verify_system_component as _verify_system_component,
)
from edumind.extraction.extractors.document import DOCLING_VERSION

INFERENCE_SNAPSHOT_IGNORE_PATTERNS = (
    "onnx/**",
    "openvino/**",
    "*.gguf",
    "*.onnx",
    "*.tflite",
    "tf_model.h5",
    "flax_model.msgpack",
)
DOCLING_STANDARD = "docling-standard"
DOCLING_APP_COMPONENTS = ("layout", "tableformer", "rapidocr")


def require_docling_version() -> str:
    for name in ("docling", "docling-slim"):
        try:
            installed = version(name)
        except PackageNotFoundError as exc:
            raise RuntimeError(f"{name}=={DOCLING_VERSION} is not installed") from exc
        if installed != DOCLING_VERSION:
            raise RuntimeError(
                f"{name}=={DOCLING_VERSION} is required, but {installed} is installed"
            )
    return DOCLING_VERSION


def snapshot_ignore_patterns(component: str) -> tuple[str, ...]:
    return (
        INFERENCE_SNAPSHOT_IGNORE_PATTERNS
        if component in {"embedding", "reranker"}
        else ()
    )


def prepare_hub_entry(
    cache_directory: Path,
    component: str,
    selection_revision: str,
    snapshots: Sequence[tuple[str, str, str]],
) -> dict[str, object]:
    ignore_patterns = snapshot_ignore_patterns(component)
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
        "component": component,
        "provider": "huggingface",
        "model": primary["repository"],
        "revision": primary["revision"],
        "selection_revision": selection_revision,
        "model_path": primary["model_path"],
        "model_cache_manifest_sha256": primary["cache_manifest_sha256"],
    }
    if ignore_patterns:
        lock_entry["snapshot_ignore_patterns"] = list(ignore_patterns)
    if len(downloaded) > 1:
        lock_entry["submodels"] = downloaded
    return lock_entry


def _prepare_snapshot(
    cache_directory: Path,
    repository: str,
    revision: str,
    ignore_patterns: Sequence[str] = (),
    *,
    reuse_only: bool = False,
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
    if reuse_only:
        raise ModelPreparationError(
            f"Cannot recover the bundle without a verified snapshot receipt: {receipt}"
        )
    import httpx
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import (
        FileMetadataError,
        HfHubHTTPError,
        LocalEntryNotFoundError,
    )

    # Offline mode and native Xet transport failures use the built-in types.
    transport_errors = (httpx.RequestError, ConnectionError, TimeoutError)
    retry = "Check network and Hub access, then retry preparation."
    try:
        info = HfApi().model_info(repository, revision=revision, files_metadata=True)
    except transport_errors as exc:
        raise ModelPreparationError(
            f"Could not fetch Hub inventory for {repository}@{revision}. {retry}"
        ) from exc
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
    if target.exists():
        # Publication may have finished just before the receipt write was interrupted.
        # Authenticate the existing files against the pinned Hub inventory in place.
        _verify_hub_inventory(target, inventory)
    else:
        hub_cache = cache_directory.parent / "huggingface/hub"
        try:
            downloaded = Path(
                snapshot_download(
                    repo_id=repository,
                    revision=revision,
                    cache_dir=hub_cache,
                    ignore_patterns=list(ignore_patterns) or None,
                )
            )
        except (*transport_errors, LocalEntryNotFoundError) as exc:
            # Hub can wrap a rejected request or invalid metadata on a cache miss.
            cause = exc.__cause__
            if isinstance(exc, LocalEntryNotFoundError) and isinstance(
                cause, (HfHubHTTPError, FileMetadataError)
            ):
                raise cause from None
            raise ModelPreparationError(
                f"Could not download Hub weights for {repository}@{revision}. {retry}"
            ) from exc
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


def _merge_model_lock(
    path: Path, updates: Mapping[str, Mapping[str, object]], *, selection_package: str
) -> None:
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
            "selection_package": selection_package,
            "models": existing,
        },
    )


def _prepare_docling_standard(
    output_path: Path,
    cache_directory: Path,
    components: Sequence[str],
    *,
    hf_pins=HF_COMPONENTS,
    file_pins=FILE_COMPONENTS,
    file_downloaders=None,
    system_downloaders=None,
    selection_package: str = "benchmark-candidates",
    recover_existing: bool = False,
) -> None:
    installed = require_docling_version()
    file_downloaders = file_downloaders or {}
    system_downloaders = system_downloaders or {}
    unknown = set(components) - (
        set(hf_pins) | set(file_pins) | set(system_downloaders)
    )
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
            "hf": hf_pins,
            "files": file_pins,
        }
    )
    docling_directory = cache_directory / "docling-bundles" / bundle_id
    recovering = docling_directory.exists()
    if recovering and (not recover_existing or previous or system_downloaders):
        raise RuntimeError(
            f"Unpublished Docling bundle already exists; retain it and use a separate cache: {docling_directory}"
        )
    working_directory = (
        docling_directory
        if recovering
        else docling_directory.with_name(bundle_id + ".partial")
    )
    if not recovering:
        working_directory.mkdir(parents=True, exist_ok=True)
    component_models = {}
    expected_bundle_files = set()
    systems = dict(previous.get("system_components", {}))
    for component in desired:
        if component in system_downloaders:
            if component not in systems:
                systems[component] = system_downloaders[component]()
            _verify_system_component(systems[component], component)
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
            shutil.copytree(prepared, working_directory / folder, dirs_exist_ok=True)
            expected_bundle_files.update(
                f"{folder}/{p.relative_to(prepared).as_posix()}"
                for p in prepared.rglob("*")
                if p.is_file()
            )
            component_models[component] = dict(old_component)
            continue
        if component in hf_pins:
            repository, revision = hf_pins[component]
            target = working_directory / repository.replace("/", "--")
            if recovering:
                # The bundle name is not proof of its contents. Check against
                # the independently verified pinned snapshot, never the bundle itself.
                prepared = _prepare_snapshot(
                    cache_directory, repository, revision, reuse_only=True
                )
                component_digest = _directory_manifest_hash(prepared)
                _verify_directory_manifest(target, component_digest, component)
            else:
                prepared = _prepare_snapshot(cache_directory, repository, revision)
                shutil.copytree(prepared, target, dirs_exist_ok=True)
                component_digest = _directory_manifest_hash(target)
            expected_bundle_files.update(
                f"{target.name}/{p.relative_to(prepared).as_posix()}"
                for p in prepared.rglob("*")
                if p.is_file()
            )
            component_models[component] = {
                "repository": repository,
                "revision": revision,
                "directory": target.name,
                "manifest_sha256": component_digest,
            }
        else:
            pin = file_pins[component]
            target = working_directory / pin["folder"]
            expected = pin["files"]
            if not recovering:
                if component in file_downloaders:
                    file_downloaders[component](working_directory)
                else:
                    from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel

                    # force=False skips any existing file, even a truncated one.
                    staging = (
                        docling_directory.parent.resolve() / working_directory.name
                    )
                    for name, digest in expected.items():
                        path = target / name
                        if path.resolve() != staging / pin["folder"] / name:
                            raise RuntimeError(
                                f"Unexpected linked RapidOCR staging path: {path}"
                            )
                        if path.is_file() and sha256_file(path) != digest:
                            path.unlink()
                    RapidOcrModel.download_models(
                        backend="onnxruntime",
                        lang="english",
                        local_dir=target,
                        force=False,
                        progress=True,
                    )
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
        p.relative_to(working_directory).as_posix()
        for p in working_directory.rglob("*")
        if p.is_file()
    }
    if actual_bundle_files != expected_bundle_files:
        raise RuntimeError("Docling bundle contains stale/unaccounted component files")
    if not recovering:
        working_directory.replace(docling_directory)
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
        selection_package=selection_package,
    )


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
