from __future__ import annotations

import importlib.util
from importlib.resources import files

from edumind.rag.contracts import load_recommendation_manifest


def test_new_public_packages_import_without_optional_models() -> None:

    assert files("edumind").joinpath("defaults.yaml").is_file()
    assert files("edumind").joinpath("recommendations/default.json").is_file()
    recommendation = load_recommendation_manifest()
    assert recommendation.authoritative is False
    assert recommendation.rag["chunking"] == "token-256-32"


def test_removed_ocr_package_has_no_compatibility_shim() -> None:
    assert importlib.util.find_spec("edumind.ocr") is None
