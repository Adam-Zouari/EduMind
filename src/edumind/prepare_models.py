"""Explicit, standalone preparation of the provisional application models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from edumind.common.config import ConfigurationError, load_settings
from edumind.common.docling_pins import FILE_COMPONENTS, HF_COMPONENTS
from edumind.common.model_assets import (
    DOCLING_APP_COMPONENTS,
    DOCLING_STANDARD,
    _merge_model_lock,
    _prepare_docling_standard,
    prepare_hub_entry,
    require_docling_version,
    snapshot_ignore_patterns,
)
from edumind.common.models import (
    ModelPreparationError,
    load_model_lock,
    validate_model_entry,
)
from edumind.common.paths import PROJECT_ROOT
from edumind.extraction.extractors.audio import WhisperExtractor
from edumind.extraction.extractors.document import DOCLING_VERSION
from edumind.rag.contracts import PRODUCTION_EMBEDDING_MODEL
from edumind.rag.tokenizers import prepare_tiktoken, tiktoken_ready

APP_GENERATOR = "Qwen/Qwen3-1.7B"
APP_MODELS = {
    PRODUCTION_EMBEDDING_MODEL: (
        "embedding",
        "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
    ),
    APP_GENERATOR: ("generator", "b9352fbb8ce704292730cf54b3b1dceb2a808738"),
    WhisperExtractor.model: ("asr", "e8727524f962ee844a7319d92be39ac1bd25655a"),
}
SELECTION_PACKAGE = "application-defaults"


def _plan() -> dict[str, object]:
    return {
        "models": {
            name: {
                "component": component,
                "revision": revision,
                "ignore_patterns": snapshot_ignore_patterns(component),
            }
            for name, (component, revision) in APP_MODELS.items()
        },
        DOCLING_STANDARD: {
            "runtime_version": DOCLING_VERSION,
            "components": {
                **{
                    name: {"repository": repository, "revision": revision}
                    for name, (repository, revision) in HF_COMPONENTS.items()
                },
                **FILE_COMPONENTS,
            },
        },
        "tokenizer": "tiktoken:cl100k_base",
    }


def _missing_models(lock_path: Path, cache: Path) -> set[str]:
    entries = load_model_lock(lock_path) if lock_path.exists() else {}
    required = {*APP_MODELS, DOCLING_STANDARD}
    if set(entries) - required:
        raise ModelPreparationError(f"Not an app-only model lock: {lock_path}")
    for name, entry in entries.items():
        validate_model_entry(entry, name, verify=False)
        if not Path(entry["model_path"]).resolve().is_relative_to(cache):
            raise ModelPreparationError(
                f"Prepared {name} must be stored inside {cache}"
            )
        if name in APP_MODELS:
            component, revision = APP_MODELS[name]
            if (
                entry.get("model") != name
                or entry["revision"] != revision
                or entry.get("snapshot_ignore_patterns", [])
                != list(snapshot_ignore_patterns(component))
                or entry.get("submodels")
            ):
                raise ModelPreparationError(
                    f"Prepared {name} does not match the app pin"
                )
        else:
            components = entry.get("component_models", {})
            if (
                entry["revision"] != DOCLING_VERSION
                or set(components) != set(DOCLING_APP_COMPONENTS)
                or entry.get("system_components")
                or any(
                    (components[key].get("repository"), components[key].get("revision"))
                    != pin
                    for key, pin in HF_COMPONENTS.items()
                )
                or any(
                    components[key].get(field) != value
                    for key, pin in FILE_COMPONENTS.items()
                    for field, value in pin.items()
                )
            ):
                raise ModelPreparationError(
                    "Prepared Docling bundle does not match the app pins"
                )
        validate_model_entry(entry, name)
    return required - entries.keys()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="List the pinned assets")
    mode.add_argument(
        "--check",
        action="store_true",
        help="Verify offline (exit 2 if preparation is needed)",
    )
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(_plan(), indent=2))
        return 0
    try:
        settings = load_settings()
        for option, configured, supported in (
            ("embedding", settings.embedding.model_name, PRODUCTION_EMBEDDING_MODEL),
            ("generation", settings.generation.model_name, APP_GENERATOR),
        ):
            if configured != supported:
                raise ConfigurationError(
                    f"{option}.model_name={configured!r} is not supported by app "
                    f"model preparation; expected {supported!r}."
                )
        require_docling_version()
        lock_path = settings.models.lock_path
        cache = (PROJECT_ROOT / "data/models/snapshots").resolve()
        missing = _missing_models(lock_path, cache)
        tokenizer_ready = tiktoken_ready()
        if args.check:
            return 2 if missing or not tokenizer_ready else 0
        for name, (component, revision) in APP_MODELS.items():
            if name in missing:
                entry = prepare_hub_entry(
                    cache, component, revision, [(name, revision, "primary")]
                )
                _merge_model_lock(
                    lock_path, {name: entry}, selection_package=SELECTION_PACKAGE
                )
        if DOCLING_STANDARD in missing:
            _prepare_docling_standard(
                lock_path,
                cache,
                DOCLING_APP_COMPONENTS,
                selection_package=SELECTION_PACKAGE,
                recover_existing=True,
            )
        if not tokenizer_ready:
            prepare_tiktoken()
    except (ConfigurationError, RuntimeError, ValueError, OSError, ImportError) as exc:
        print(f"App model preparation failed: {exc}", file=sys.stderr)
        return 1
    print(f"App models ready: {lock_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
