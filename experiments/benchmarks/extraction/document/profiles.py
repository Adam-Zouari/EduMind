"""Canonical document benchmark profile parsing and model-lock identity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


_CANDIDATE_PATH = Path(__file__).with_name("candidates.yaml")


def _model_identities() -> dict[str, str]:
    payload = yaml.safe_load(_CANDIDATE_PATH.read_text(encoding="utf-8"))
    candidates = payload.get("candidates") if isinstance(payload, Mapping) else None
    if not isinstance(candidates, Mapping) or not candidates:
        raise ValueError("Document candidates.yaml requires a candidate registry")
    result = {}
    for alias, value in candidates.items():
        if not isinstance(alias, str) or not isinstance(value, Mapping):
            raise ValueError("Document candidate registry is malformed")
        model_id = value.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError(f"Document candidate {alias!r} requires model_id")
        result[alias] = model_id
    return result


DOCUMENT_LOCK_CANDIDATES = _model_identities()
FACTOR_OPTIONS = {
    "ocr": "ocr_engine",
    "mode": "ocr_mode",
    "table": "table_mode",
    "formula": "formula_enrichment",
}


@dataclass(frozen=True)
class DocumentProfile:
    requested_engine: str
    runtime_engine: str
    lock_candidate: str
    factors: Mapping[str, str]
    options: Mapping[str, object]


def parse_document_profile(candidate: str) -> DocumentProfile:
    parts = candidate.split("|")
    requested_engine = parts[0]
    runtime_engine = (
        "docling-standard"
        if requested_engine == "docling-standard-native"
        else requested_engine
    )
    if runtime_engine not in DOCUMENT_LOCK_CANDIDATES:
        raise ValueError(f"Unknown document benchmark engine: {requested_engine}")
    factors: dict[str, str] = {}
    options: dict[str, object] = {}
    for factor in parts[1:]:
        key, separator, value = factor.partition("=")
        if not separator or key not in FACTOR_OPTIONS:
            raise ValueError(f"Unknown document-profile factor: {factor}")
        factors[key] = value
        options[FACTOR_OPTIONS[key]] = value == "on" if key == "formula" else value
    return DocumentProfile(
        requested_engine,
        runtime_engine,
        DOCUMENT_LOCK_CANDIDATES[runtime_engine],
        factors,
        options,
    )


def lock_paths(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): value
        for key, value in entry.items()
        if str(key).endswith("_path") or str(key).endswith("_dir")
    }
