
from edumind.common.paths import CONFIG_DIR, PROJECT_ROOT, artifact_path, resolve_config_path


def test_project_root_contains_readme() -> None:
    assert (PROJECT_ROOT / "README.md").exists()


def test_resolve_default_config_path() -> None:
    assert resolve_config_path() == CONFIG_DIR / "base.yaml"


def test_artifact_path_builds_inside_artifacts() -> None:
    path = artifact_path("unit-tests", "result.txt", create=False)
    assert "artifacts" in str(path)
    assert path.name == "result.txt"
