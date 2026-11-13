"""Behavioral preparation regressions; all transfers/services are mocked."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from edumind.common.artifacts import atomic_write_json, sha256_file, stable_hash
from edumind.common.models import (
    ModelPreparationError,
    directory_manifest_hash,
    load_model_lock,
    require_model,
)
from experiments.benchmarks import prepare
from experiments.benchmarks.common.selection import selection_entries
from experiments.benchmarks.preparation import evaluators, models, vectordb


def entry_for(directory, candidate="openai/whisper-small.en"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "weights.bin").write_bytes(b"unit-test-only bytes")
    revision = next(
        row.revision for row in selection_entries() if row.candidate == candidate
    )
    return {
        "model": candidate,
        "revision": revision,
        "selection_revision": revision,
        "model_path": str(directory),
        "model_cache_manifest_sha256": directory_manifest_hash(directory),
    }


@pytest.mark.parametrize(
    "target,extra",
    [
        ("app-models", []),
        ("embedding-models", []),
        ("rag-models", []),
        ("extraction-models", []),
        ("all-models", []),
        ("qasper", []),
        (
            "rag-selection",
            [
                "--qasper-manifest",
                "absent.json",
                "--structured-manifest",
                "absent-structured.json",
                "--output",
                "combined.json",
            ],
        ),
        ("assets", ["--plan", "absent.json"]),
        ("vectordb", []),
        ("smoke-fixtures", []),
        ("evaluators", []),
    ],
)
def test_every_cli_dry_run_stops_before_mutating_dispatch(
    monkeypatch, capsys, target, extra
):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry-run crossed a mutating preparation boundary")

    for name in (
        "prepare_app_models",
        "prepare_selected_models",
        "prepare_qasper",
        "prepare_rag_selection_manifest",
        "prepare_public_assets",
        "prepare_vectordb",
        "prepare_evaluators",
        "prepare_smoke_fixtures",
    ):
        monkeypatch.setattr(prepare, name, forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(sys, "argv", ["prepare", target, "--dry-run", *extra])
    assert prepare.main() == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


@pytest.mark.parametrize("schema", [True, False, "3", None, 0, 4, 999])
def test_unsupported_or_mistyped_lock_schema_fails(tmp_path, schema):
    path = tmp_path / "lock.json"
    atomic_write_json(path, {"schema_version": schema, "models": {"x": {}}})
    with pytest.raises(ModelPreparationError, match="schema"):
        load_model_lock(path)
    with pytest.raises(ModelPreparationError, match="schema"):
        models.load_selected_model_lock(path)


@pytest.mark.parametrize("field", ["model_path", "model_cache_manifest_sha256"])
@pytest.mark.parametrize("value", [None, "", 0, False, [], {}])
def test_schema_three_requires_typed_path_and_checksum(tmp_path, field, value):
    entry = entry_for(tmp_path / "snapshot")
    entry[field] = value
    path = tmp_path / "lock.json"
    atomic_write_json(path, {"schema_version": 3, "models": {entry["model"]: entry}})
    with pytest.raises(ModelPreparationError):
        models.load_selected_model_lock(path, candidates=[entry["model"]])


def test_missing_fields_and_file_instead_of_directory_are_rejected(tmp_path):
    entry = entry_for(tmp_path / "snapshot")
    for field in ("model_path", "model_cache_manifest_sha256"):
        missing = {name: value for name, value in entry.items() if name != field}
        with pytest.raises(ModelPreparationError):
            require_model({"x": missing}, "x")
    entry["model_path"] = str(tmp_path / "snapshot/weights.bin")
    with pytest.raises(ModelPreparationError, match="directory"):
        require_model({"x": entry}, "x")


def test_partial_legacy_migration_is_rejected_without_publication(tmp_path):
    entry = entry_for(tmp_path / "new")
    path = tmp_path / "lock.json"
    atomic_write_json(
        path,
        {
            "schema_version": 2,
            "models": {"legacy": {"revision": "old", "model_path": "unavailable"}},
        },
    )
    before = path.read_bytes()
    with pytest.raises(ModelPreparationError, match="reprepare legacy"):
        models._merge_model_lock(path, {entry["model"]: entry})
    assert path.read_bytes() == before


def test_merge_preserves_app_and_unrequested_missing_snapshot(tmp_path):
    entry = entry_for(tmp_path / "new")
    app = {
        **entry,
        "model": models.APP_GENERATOR.candidate,
        "revision": models.APP_GENERATOR.revision,
        "selection_revision": models.APP_GENERATOR.revision,
        "model_path": "unavailable-but-unrequested",
    }
    path = tmp_path / "lock.json"
    atomic_write_json(
        path, {"schema_version": 3, "models": {models.APP_GENERATOR.candidate: app}}
    )
    models._merge_model_lock(path, {entry["model"]: entry})
    payload = json.loads(path.read_text())
    assert payload["models"][models.APP_GENERATOR.candidate] == app
    assert payload["models"][entry["model"]]["snapshot_size_bytes"] == len(
        b"unit-test-only bytes"
    )
    assert set(models.load_selected_model_lock(path, candidates=[entry["model"]])) == {
        entry["model"]
    }
    assert models.APP_GENERATOR.candidate not in models.selected_model_names(
        models.MODEL_COMPONENTS
    )


def test_app_preparation_uses_pinned_excluded_generator_without_changing_roster(
    tmp_path, monkeypatch
):
    prepared = []

    def snapshot(cache, repository, revision, *args):
        prepared.append((repository, revision))
        directory = cache / repository.replace("/", "--") / revision
        directory.mkdir(parents=True)
        (directory / "weights.bin").write_bytes(b"test only")
        return directory

    monkeypatch.setattr(models, "_prepare_snapshot", snapshot)
    monkeypatch.setattr(models, "_prepare_docling_standard", lambda *args: None)
    monkeypatch.setattr(models, "_prepare_tiktoken", lambda *args: None)
    before = models.selected_model_names(models.MODEL_COMPONENTS)
    outputs = models.prepare_app_models(tmp_path)
    assert (models.APP_GENERATOR.candidate, models.APP_GENERATOR.revision) in prepared
    assert models.APP_GENERATOR.candidate in load_model_lock(outputs[0])
    assert models.selected_model_names(models.MODEL_COMPONENTS) == before


def hub_fixture(
    tmp_path, monkeypatch, *, returned_revision=None, extra=False, corrupt=False
):
    import huggingface_hub

    revision = models.APP_GENERATOR.revision
    repository = "unit-test/model"
    cache = tmp_path / "models"
    directory = (
        tmp_path
        / "huggingface/hub/models--unit-test--model/snapshots"
        / (returned_revision or revision)
    )
    directory.mkdir(parents=True)
    content = b"unit-test snapshot"
    (directory / "weights.bin").write_bytes(b"corrupt" if corrupt else content)
    if extra:
        (directory / "stale.onnx").write_bytes(b"excluded")
    git_blob = hashlib.sha1(
        f"blob {len(content)}\0".encode() + content, usedforsecurity=False
    ).hexdigest()
    info = SimpleNamespace(
        sha=revision,
        siblings=[
            SimpleNamespace(
                rfilename="weights.bin", size=len(content), lfs=None, blob_id=git_blob
            )
        ],
    )
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(model_info=lambda *args, **kwargs: info),
    )
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda **kwargs: str(directory)
    )
    return cache, repository, revision, directory


@pytest.mark.parametrize("mode", ["wrong_revision", "stale_export", "corrupt"])
def test_snapshot_fallback_mixed_files_and_corruption_cannot_publish(
    tmp_path, monkeypatch, mode
):
    cache, repo, revision, _ = hub_fixture(
        tmp_path,
        monkeypatch,
        returned_revision="0" * 40 if mode == "wrong_revision" else None,
        extra=mode == "stale_export",
        corrupt=mode == "corrupt",
    )
    with pytest.raises(RuntimeError):
        models._prepare_snapshot(cache, repo, revision, ("*.onnx",))
    assert not list(cache.rglob("*.complete.json"))
    assert not (cache / repo.replace("/", "--") / revision).exists()


def test_completed_snapshot_can_resume_offline_and_rechecks_bytes(
    tmp_path, monkeypatch
):
    import huggingface_hub

    cache, repo, revision, _ = hub_fixture(tmp_path, monkeypatch)
    result = models._prepare_snapshot(cache, repo, revision)
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: pytest.fail("completed resume must not contact Hub"),
    )
    assert models._prepare_snapshot(cache, repo, revision) == result
    receipt = json.loads(result.with_name(revision + ".complete.json").read_text())
    assert receipt["snapshot_size_bytes"] == len(b"unit-test snapshot")
    (result / "weights.bin").write_bytes(b"changed")
    with pytest.raises(ModelPreparationError, match="checksum mismatch"):
        models._prepare_snapshot(cache, repo, revision)


def test_interrupted_snapshot_never_publishes_or_erases_old_downloads(
    tmp_path, monkeypatch
):
    cache, repo, revision, old = hub_fixture(tmp_path, monkeypatch)
    original = models.shutil.copy2
    monkeypatch.setattr(
        models.shutil,
        "copy2",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("interrupted copy")),
    )
    with pytest.raises(OSError, match="interrupted"):
        models._prepare_snapshot(cache, repo, revision)
    assert (old / "weights.bin").is_file()
    assert not list(cache.rglob("*.complete.json"))
    monkeypatch.setattr(models.shutil, "copy2", original)
    assert models._prepare_snapshot(cache, repo, revision).is_dir()


def test_unavailable_new_revision_does_not_relabel_old_bytes(tmp_path, monkeypatch):
    import huggingface_hub

    cache, repo, revision, old = hub_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        huggingface_hub,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda *args, **kwargs: (_ for _ in ()).throw(OSError("offline"))
        ),
    )
    with pytest.raises(OSError, match="offline"):
        models._prepare_snapshot(cache, repo, revision)
    assert (old / "weights.bin").read_bytes() == b"unit-test snapshot"
    assert not list(cache.rglob("*.complete.json"))


def evaluator_fixture(root):
    source = root / "data/benchmarks/evaluators/OmniDocBench"
    source.mkdir(parents=True)
    (source / "table_metric.py").write_text("# unit test scorer source\n")
    (source / "requirements.txt").write_text("# unit test dependency inventory\n")
    (source / ".edumind-revision").write_text(evaluators.OMNIDOCBENCH_REVISION)
    files = evaluators._source_inventory(source)
    atomic_write_json(
        source / evaluators.OMNIDOCBENCH_SOURCE_MANIFEST,
        {
            "schema_version": 1,
            "source_revision": evaluators.OMNIDOCBENCH_REVISION,
            "files": files,
            "source_manifest_sha256": stable_hash(files),
        },
    )
    atomic_write_json(
        source / evaluators.OMNIDOCBENCH_IMAGE_LOCK,
        {
            "source_revision": evaluators.OMNIDOCBENCH_REVISION,
            "tag": evaluators.OMNIDOCBENCH_IMAGE,
            "digest": "unit-test/image@sha256:" + "a" * 64,
        },
    )
    return source


@pytest.mark.parametrize("mode", ["changed", "deleted", "added", "dependency"])
def test_evaluator_corruption_rejected_before_docker_scoring(
    tmp_path, monkeypatch, mode
):
    from experiments.benchmarks.extraction.document import official_metrics

    source = evaluator_fixture(tmp_path)
    assert evaluators.evaluator_identity(source)["source_manifest_sha256"]
    monkeypatch.setattr(official_metrics, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("corrupt scorer reached Docker"),
    )
    if mode == "deleted":
        (source / "table_metric.py").unlink()
    else:
        (
            source
            / (
                "added.py"
                if mode == "added"
                else "requirements.txt"
                if mode == "dependency"
                else "table_metric.py"
            )
        ).write_text("# corruption\n")
    with pytest.raises(RuntimeError, match="source integrity"):
        official_metrics.score_official_metrics(
            [("<table/>", "<table/>")], [], timeout_seconds=1
        )
    with pytest.raises(RuntimeError, match="source integrity"):
        evaluators.prepare_evaluators(tmp_path)


def test_validation_identity_binds_actual_evaluator_content(tmp_path, monkeypatch):
    from experiments.benchmarks.common import data_validation

    source = evaluator_fixture(tmp_path)
    manifest = tmp_path / "manifest.json"
    atomic_write_json(manifest, {})
    monkeypatch.setattr(data_validation, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        data_validation,
        "load_manifest",
        lambda *args, **kwargs: SimpleNamespace(split="smoke", samples=[]),
    )
    monkeypatch.setattr(data_validation, "sha256_file", lambda p: "mocked-code-digest")
    before, resolved = data_validation.input_identity(
        "document", "smoke", manifest, {}, Path(__file__)
    )
    assert resolved["requirements"]["official_evaluator"][
        "source_manifest_sha256"
    ] == evaluators.verify_evaluator_source(source)
    (source / "table_metric.py").write_text("# changed after reference certification\n")
    with pytest.raises(RuntimeError, match="source integrity"):
        data_validation.input_identity(
            "document", "smoke", manifest, {}, Path(__file__)
        )
    assert before


def test_application_requires_full_primary_and_auxiliary_integrity(tmp_path):
    entry = entry_for(tmp_path / "primary")
    auxiliary = entry_for(tmp_path / "auxiliary")
    entry.update(
        paddle_cache_path=auxiliary["model_path"],
        paddle_cache_manifest_sha256=auxiliary["model_cache_manifest_sha256"],
    )
    snapshot = require_model({"x": entry}, "x")
    before = snapshot.integrity_sha256
    (tmp_path / "auxiliary/weights.bin").write_bytes(b"altered")
    with pytest.raises(ModelPreparationError, match="checksum mismatch"):
        require_model({"x": entry}, "x")
    entry["paddle_cache_manifest_sha256"] = directory_manifest_hash(
        tmp_path / "auxiliary"
    )
    assert require_model({"x": entry}, "x").integrity_sha256 != before


@pytest.mark.parametrize("destination", ["lock.json", "custom/nested/lock.json"])
def test_vectordb_custom_output_never_changes_compose_ownership(
    tmp_path, monkeypatch, destination
):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vectordb, "PROJECT_ROOT", root)
    monkeypatch.setattr(vectordb, "version", lambda *args: "unit-test-version")
    monkeypatch.setattr(
        vectordb.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="unit-test/image@sha256:" + "b" * 64
        ),
    )
    output = Path(destination)
    vectordb.prepare_vectordb(output)
    assert output.is_file()
    environment = root / "experiments/benchmarks/vectordb/.env"
    assert environment.is_file()
    assert environment.read_text().count("_IMAGE=") == len(vectordb.VECTOR_IMAGES)


@pytest.mark.parametrize("mode", ["directory", "parent_file", "env_collision"])
def test_vectordb_invalid_destinations_fail_before_services_or_writes(
    tmp_path, monkeypatch, mode
):
    monkeypatch.setattr(vectordb, "PROJECT_ROOT", tmp_path)
    if mode == "directory":
        output = tmp_path
    elif mode == "parent_file":
        parent = tmp_path / "file"
        parent.write_bytes(b"keep")
        output = parent / "lock.json"
    else:
        output = tmp_path / "experiments/benchmarks/vectordb/.env"
    monkeypatch.setattr(
        vectordb.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("invalid destination reached Docker"),
    )
    with pytest.raises(ValueError):
        vectordb.prepare_vectordb(output)


def test_docling_fresh_preparation_requests_immutable_component_commits(
    tmp_path, monkeypatch
):
    requested = []

    def snapshot(cache, repository, revision):
        requested.append((repository, revision))
        directory = cache / repository.replace("/", "--") / revision
        directory.mkdir(parents=True)
        (directory / "test-only.bin").write_bytes(b"mocked component bytes")
        return directory

    monkeypatch.setattr(models, "version", lambda name: models.DOCLING_VERSION)
    monkeypatch.setattr(models, "_prepare_snapshot", snapshot)
    path = tmp_path / "lock.json"
    models._prepare_docling_standard(
        path, tmp_path / "cache", tuple(models.HF_COMPONENTS)
    )
    assert set(requested) == set(models.HF_COMPONENTS.values())
    entry = load_model_lock(path)[models.DOCLING_STANDARD]
    assert entry["runtime_version"] == models.DOCLING_VERSION
    assert set(entry["component_models"]) == set(models.HF_COMPONENTS)
    assert models.load_selected_model_lock(path, candidates=[models.DOCLING_STANDARD])


def test_docling_app_subset_preserves_verified_components_and_systems(
    tmp_path, monkeypatch
):
    entry = entry_for(tmp_path / "bundle")
    entry["revision"] = models.DOCLING_VERSION
    entry["prepared_components"] = ["layout", "tableformer", "code_formula", "rapidocr"]
    entry["component_models"] = {}
    for component in entry["prepared_components"]:
        folder = tmp_path / "bundle" / component
        folder.mkdir()
        (folder / "test-only.bin").write_bytes(component.encode())
        entry["component_models"][component] = {
            "directory": component,
            "manifest_sha256": directory_manifest_hash(folder),
            "retained_provenance": "unit-test",
        }
    entry["model_cache_manifest_sha256"] = directory_manifest_hash(tmp_path / "bundle")
    executable = tmp_path / "tesseract.exe"
    language = tmp_path / "eng.traineddata"
    executable.write_bytes(b"test-only executable")
    language.write_bytes(b"test-only English data")
    entry["prepared_components"].append("tesseract-cli")
    entry["system_components"] = {
        "tesseract-cli": {
            "path": str(executable),
            "sha256": sha256_file(executable),
            "version": "test-only",
            "english_data_path": str(language),
            "english_data_sha256": sha256_file(language),
        }
    }
    path = tmp_path / "lock.json"
    atomic_write_json(
        path, {"schema_version": 3, "models": {models.DOCLING_STANDARD: entry}}
    )
    before = path.read_bytes()
    monkeypatch.setattr(models, "version", lambda name: models.DOCLING_VERSION)
    monkeypatch.setattr(
        models,
        "_prepare_snapshot",
        lambda *args: pytest.fail("subset must reuse verified components"),
    )
    models._prepare_docling_standard(
        path, tmp_path / "cache", models.DOCLING_APP_COMPONENTS
    )
    assert path.read_bytes() == before


def test_docling_augmentation_copies_old_components_without_relabelling(
    tmp_path, monkeypatch
):
    path = tmp_path / "lock.json"
    folder = tmp_path / "old/layout"
    folder.mkdir(parents=True)
    (folder / "weights.bin").write_bytes(b"unit-test layout bytes")
    old = {
        "revision": models.DOCLING_VERSION,
        "model_path": str(tmp_path / "old"),
        "model_cache_manifest_sha256": directory_manifest_hash(tmp_path / "old"),
        "prepared_components": ["layout"],
        "component_models": {
            "layout": {
                "directory": "layout",
                "manifest_sha256": directory_manifest_hash(folder),
                "retained_provenance": "keep",
            }
        },
    }
    atomic_write_json(
        path, {"schema_version": 3, "models": {models.DOCLING_STANDARD: old}}
    )
    requested = []

    def snapshot(cache, repository, revision):
        requested.append(repository)
        directory = cache / "test-only-tableformer"
        directory.mkdir(parents=True)
        (directory / "weights.bin").write_bytes(b"unit-test table bytes")
        return directory

    monkeypatch.setattr(models, "version", lambda name: models.DOCLING_VERSION)
    monkeypatch.setattr(models, "_prepare_snapshot", snapshot)
    models._prepare_docling_standard(path, tmp_path / "cache", ["tableformer"])
    entry = load_model_lock(path)[models.DOCLING_STANDARD]
    assert requested == [models.HF_COMPONENTS["tableformer"][0]]
    assert entry["component_models"]["layout"] == old["component_models"]["layout"]
    assert set(entry["prepared_components"]) == {"layout", "tableformer"}
    assert Path(entry["model_path"]) != Path(old["model_path"])
    assert (folder / "weights.bin").read_bytes() == b"unit-test layout bytes"


@pytest.mark.parametrize("field", ["path", "english_data_path"])
def test_prepared_tesseract_identity_rejects_substitution(tmp_path, monkeypatch, field):
    executable = tmp_path / "prepared/tesseract.exe"
    language = tmp_path / "prepared/tessdata/eng.traineddata"
    language.parent.mkdir(parents=True)
    executable.write_bytes(b"unit-test executable")
    language.write_bytes(b"unit-test language")
    identity = {
        "path": str(executable),
        "sha256": sha256_file(executable),
        "version": "test-only",
        "english_data_path": str(language),
        "english_data_sha256": sha256_file(language),
    }
    monkeypatch.setenv("PATH", str(tmp_path / "other-installation"))
    monkeypatch.setenv("TESSDATA_PREFIX", str(tmp_path / "other-language"))
    models._verify_system_component(identity, "Tesseract CLI")
    Path(identity[field]).write_bytes(b"substituted")
    with pytest.raises(ModelPreparationError, match="missing or changed"):
        models._verify_system_component(identity, "Tesseract CLI")


def test_evaluator_fresh_preparation_records_verified_checkout_and_content(
    tmp_path, monkeypatch
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["git", "clone"]:
            destination = Path(command[-1])
            destination.mkdir()
            (destination / ".git").mkdir()
            (destination / "table_metric.py").write_text("# mocked pinned checkout\n")
        stdout = evaluators.OMNIDOCBENCH_REVISION if "rev-parse" in command else ""
        if "--format" in command:
            stdout = "unit-test/image@sha256:" + "c" * 64
        return SimpleNamespace(stdout=stdout)

    monkeypatch.setattr(evaluators.subprocess, "run", run)
    outputs = evaluators.prepare_evaluators(tmp_path)
    identity = evaluators.evaluator_identity(outputs[0])
    assert identity["source_manifest_sha256"] == evaluators.verify_evaluator_source(
        outputs[0]
    )
    assert evaluators.OMNIDOCBENCH_REVISION in next(
        command for command in calls if "checkout" in command
    )
    assert not (outputs[0] / ".git").exists()


def test_evaluator_marker_alone_cannot_be_sealed_or_prepared(tmp_path, monkeypatch):
    source = tmp_path / "data/benchmarks/evaluators/OmniDocBench"
    source.mkdir(parents=True)
    (source / ".edumind-revision").write_text(evaluators.OMNIDOCBENCH_REVISION)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail(
            "unverified legacy source reached network/Docker"
        ),
    )
    with pytest.raises(RuntimeError, match="source integrity"):
        evaluators.prepare_evaluators(tmp_path)
    assert (source / ".edumind-revision").is_file()


@pytest.mark.parametrize("schema", [1, 2, 3])
def test_supported_lock_schemas_keep_integrity_and_optional_size(tmp_path, schema):
    entry = entry_for(tmp_path / "snapshot")
    path = tmp_path / "lock.json"
    atomic_write_json(
        path, {"schema_version": schema, "models": {entry["model"]: entry}}
    )
    assert require_model(load_model_lock(path), entry["model"])
    assert models.load_selected_model_lock(path, candidates=[entry["model"]])
    assert "snapshot_size_bytes" not in entry


def test_stage_loading_does_not_read_unrequested_snapshot(tmp_path):
    entry = entry_for(tmp_path / "snapshot")
    path = tmp_path / "lock.json"
    atomic_write_json(
        path,
        {
            "schema_version": 3,
            "models": {
                entry["model"]: entry,
                "Alibaba-NLP/gte-modernbert-base": {
                    "revision": "irrelevant",
                    "model_path": "unavailable",
                },
            },
        },
    )
    assert set(models.load_selected_model_lock(path, candidates=[entry["model"]])) == {
        entry["model"]
    }


def test_missing_submodel_integrity_cannot_pass_as_a_valid_primary(tmp_path):
    entry = entry_for(tmp_path / "primary")
    auxiliary = entry_for(tmp_path / "submodel")
    entry["submodels"] = [
        {"revision": auxiliary["revision"], "model_path": auxiliary["model_path"]}
    ]
    with pytest.raises(ModelPreparationError, match="cache_manifest"):
        require_model({"x": entry}, "x")
