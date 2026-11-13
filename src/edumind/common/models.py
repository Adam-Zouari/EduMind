"""Resolve immutable production model snapshots from the generated local lock."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from edumind.common.artifacts import sha256_file, stable_hash


class ModelPreparationError(RuntimeError):
    """A required pinned model entry or local snapshot is unavailable."""


@dataclass(frozen=True)
class ModelSnapshot:
    name: str
    revision: str
    path: Path
    values: Mapping[str, object]

    @property
    def integrity_sha256(self) -> str:
        """Identity of all verified model bytes, including auxiliary components."""
        return stable_hash(dict(self.values))


def model_lock_schema(payload: object) -> int:
    if not isinstance(payload, Mapping):
        raise ModelPreparationError("Malformed model lock: expected an object")
    schema = payload.get("schema_version", 1)
    if type(schema) is not int or schema not in {1, 2, 3}:
        raise ModelPreparationError(f"Unsupported model-lock schema: {schema!r}")
    return schema


def directory_manifest_hash(directory: Path) -> str:
    """Legacy-compatible inventory hash; includes every retained file."""
    if not directory.is_dir():
        raise ModelPreparationError(f"Prepared model directory is missing: {directory}")
    files = sorted(
        directory.rglob("*"), key=lambda p: p.relative_to(directory).as_posix()
    )
    inventory = [
        {
            "path": p.relative_to(directory).as_posix(),
            "size": p.stat().st_size,
            "sha256": sha256_file(p),
        }
        for p in files
        if p.is_file()
    ]
    if not inventory:
        raise ModelPreparationError(f"Prepared model directory is empty: {directory}")
    return stable_hash(inventory)


def verify_directory_manifest(directory: Path, expected: object, label: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ModelPreparationError(
            f"Model lock lacks a cache-manifest checksum for {label}"
        )
    actual = directory_manifest_hash(directory)
    if actual != expected:
        raise ModelPreparationError(
            f"Prepared cache manifest checksum mismatch for {label}: expected {expected}, computed {actual}"
        )


def validate_model_entry(entry: object, label: str, *, verify: bool = True) -> None:
    """Validate integrity without importing benchmark selection into production.

    Structural-only validation permits an atomic lock merge without accessing
    unrelated snapshots. Loading a requested model always verifies its bytes.
    """
    if not isinstance(entry, Mapping):
        raise ModelPreparationError(f"Malformed model entry: {label}")
    for field in ("revision", "model_path", "model_cache_manifest_sha256"):
        if not isinstance(entry.get(field), str) or not entry[field].strip():
            raise ModelPreparationError(
                f"Model {label} lacks {field}; reprepare legacy entries into a separate lock before merging"
            )
    checksum = entry["model_cache_manifest_sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ModelPreparationError(
            f"Model {label} has an invalid cache-manifest checksum"
        )
    if verify:
        verify_directory_manifest(
            Path(entry["model_path"]).expanduser(), checksum, label
        )
    submodels = entry.get("submodels", [])
    if not isinstance(submodels, list):
        raise ModelPreparationError(f"Malformed submodels for {label}")
    for index, item in enumerate(submodels):
        if not isinstance(item, Mapping):
            raise ModelPreparationError(f"Malformed submodel for {label}")
        validate_model_entry(
            {**item, "model_cache_manifest_sha256": item.get("cache_manifest_sha256")},
            f"{label} submodel {index}",
            verify=verify,
        )
    if "paddle_cache_path" in entry or "paddle_cache_manifest_sha256" in entry:
        validate_model_entry(
            {
                "revision": entry["revision"],
                "model_path": entry.get("paddle_cache_path"),
                "model_cache_manifest_sha256": entry.get(
                    "paddle_cache_manifest_sha256"
                ),
            },
            f"{label} Paddle components",
            verify=verify,
        )
    systems = entry.get("system_components", {})
    if not isinstance(systems, Mapping):
        raise ModelPreparationError(f"Malformed system components for {label}")
    prepared = entry.get("prepared_components", [])
    if (
        not isinstance(prepared, list)
        or any(not isinstance(name, str) or not name for name in prepared)
        or len(set(prepared)) != len(prepared)
    ):
        raise ModelPreparationError(f"Malformed prepared components for {label}")
    components = entry.get("component_models")
    if components is not None:
        if not isinstance(components, Mapping) or set(components) != set(
            prepared
        ) - set(systems):
            raise ModelPreparationError(
                f"Incomplete component-model inventory for {label}"
            )
        for name, component in components.items():
            if not isinstance(component, Mapping):
                raise ModelPreparationError(f"Malformed component identity for {name}")
            folder = component.get("directory", component.get("folder"))
            digest = component.get("manifest_sha256")
            if (
                not isinstance(folder, str)
                or not folder
                or Path(folder).name != folder
                or folder in {".", ".."}
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ModelPreparationError(f"Malformed component identity for {name}")
            if "repository" in component and (
                not isinstance(component.get("revision"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", component["revision"])
            ):
                raise ModelPreparationError(
                    f"Component {name} lacks an immutable revision"
                )
            if verify:
                verify_directory_manifest(
                    Path(entry["model_path"]) / folder, digest, str(name)
                )
    if "tesseract-cli" in prepared and "tesseract-cli" not in systems:
        raise ModelPreparationError(
            "Docling lock lacks the prepared Tesseract system identity"
        )
    for name, system in systems.items():
        verify_system_component(system, str(name), verify=verify)


def verify_system_component(value: object, label: str, *, verify: bool = True) -> None:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("version"), str)
        or not value["version"].strip()
    ):
        raise ModelPreparationError(f"Model lock lacks the {label} identity/version")
    for path_field, checksum_field in (
        ("path", "sha256"),
        ("english_data_path", "english_data_sha256"),
    ):
        path, checksum = value.get(path_field), value.get(checksum_field)
        if (
            not isinstance(path, str)
            or not path.strip()
            or not isinstance(checksum, str)
            or not re.fullmatch(r"[0-9a-f]{64}", checksum)
        ):
            raise ModelPreparationError(
                f"Model lock lacks the {label} {path_field} identity"
            )
        if verify and (not Path(path).is_file() or sha256_file(Path(path)) != checksum):
            raise ModelPreparationError(
                f"Prepared {label} {path_field} is missing or changed"
            )


def load_model_lock(path: Path) -> dict[str, dict[str, object]]:
    if not path.is_file():
        raise ModelPreparationError(
            f"Missing model lock {path}; run `python "
            "-m experiments.benchmarks.prepare app-models`."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelPreparationError(f"Cannot read model lock {path}: {exc}") from exc
    model_lock_schema(payload)
    models = payload.get("models")
    if not isinstance(models, Mapping):
        raise ModelPreparationError(f"Malformed model lock: {path}")
    return {
        str(name): dict(entry)
        for name, entry in models.items()
        if isinstance(entry, Mapping)
    }


def require_model(lock: Mapping[str, Mapping[str, object]], name: str) -> ModelSnapshot:
    entry = lock.get(name)
    if not isinstance(entry, Mapping):
        raise ModelPreparationError(
            f"Model {name} is absent from selected.json; run `python "
            "-m experiments.benchmarks.prepare app-models`."
        )
    validate_model_entry(entry, name)
    revision = entry.get("revision")
    raw_path = entry.get("model_path")
    if (
        not isinstance(revision, str)
        or not revision.strip()
        or not isinstance(raw_path, str)
        or not raw_path.strip()
        or not (path := Path(raw_path).expanduser()).is_dir()
    ):
        raise ModelPreparationError(
            f"Prepared model {name} is incomplete; run `python "
            "-m experiments.benchmarks.prepare app-models`."
        )
    return ModelSnapshot(name, revision.strip(), path.resolve(), dict(entry))
