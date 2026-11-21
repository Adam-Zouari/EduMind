"""Exercise app-only preparation and Windows launch without real downloads."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from edumind.common.paths import PROJECT_ROOT


def test_app_preparation_without_benchmark_tree(tmp_path):
    # Use the real CLI and lock/bundle construction from an isolated source
    # tree. Only the download providers and their fixture bytes are mocked.
    shutil.copytree(
        PROJECT_ROOT / "src/edumind",
        tmp_path / "src/edumind",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(
        PROJECT_ROOT / "config",
        tmp_path / "config",
        ignore=shutil.ignore_patterns("local.yaml"),
    )
    shutil.copy2(PROJECT_ROOT / "pyproject.toml", tmp_path)
    env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("EDUMIND_")
    }
    env["PYTHONPATH"] = str(tmp_path / "src")
    dry = subprocess.run(
        [sys.executable, "-B", "-m", "edumind.prepare_models", "--dry-run"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(dry.stdout)
    assert set(plan["models"]) == {
        "Alibaba-NLP/gte-modernbert-base",
        "Qwen/Qwen3-1.7B",
        "openai/whisper-small.en",
    }
    assert set(plan["docling-standard"]["components"]) == {
        "layout",
        "tableformer",
        "rapidocr",
    }
    assert plan["tokenizer"] == "tiktoken:cl100k_base"
    assert not (tmp_path / "data").exists()
    offline = """
import base64
import hashlib
import os
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import huggingface_hub
from edumind import prepare_models as app
from edumind.common import model_assets
from edumind.common.models import load_model_lock, require_model
from edumind.rag import tokenizers
from tiktoken import load

assert app.PROJECT_ROOT == Path.cwd()
assert not Path('benchmarks').exists()
assert not Path('data/benchmarks').exists()
sys.modules['benchmarks'] = None

def run(*args):
    sys.argv = ['prepare_models', *args]
    return app.main()

weights = b'fake weights'
sibling = SimpleNamespace(
    rfilename='fixture.weights', size=len(weights),
    lfs={'sha256': hashlib.sha256(weights).hexdigest()},
)

def download(*, repo_id, revision, cache_dir, **kwargs):
    folder = cache_dir / ('models--' + repo_id.replace('/', '--')) / 'snapshots' / revision
    folder.mkdir(parents=True)
    (folder / sibling.rfilename).write_bytes(weights)
    return folder

ocr_files = {'fixture.onnx': b'fake OCR weights', 'font.ttf': b'fake OCR font'}
ocr_downloads = []

def rapidocr(*, local_dir, force, **kwargs):
    local_dir.mkdir(parents=True, exist_ok=True)
    for name, content in ocr_files.items():
        path = local_dir / name
        if path.exists() and not force:
            continue
        path.write_bytes(content[:4] if not ocr_downloads else content)
        ocr_downloads.append(name)

tokenizer_bytes = b''.join(
    base64.b64encode(bytes([i])) + b' ' + str(i).encode() + b'\\n' for i in range(256)
)
lock_path = Path('data/models/app.json')
pin = app.FILE_COMPONENTS['rapidocr']
ocr_module = 'docling.models.stages.ocr.rapid_ocr_model'
with (
    patch.object(huggingface_hub.HfApi, 'model_info',
                 side_effect=lambda repo, *, revision, **kw: SimpleNamespace(sha=revision, siblings=[sibling])),
    patch.object(huggingface_hub, 'snapshot_download', download),
    patch.object(model_assets, 'version', return_value=app.DOCLING_VERSION),
    patch.dict(sys.modules, {ocr_module: SimpleNamespace(RapidOcrModel=SimpleNamespace(download_models=rapidocr))}),
    patch.dict(pin['files'], {name: hashlib.sha256(content).hexdigest() for name, content in ocr_files.items()}, clear=True),
    patch.object(tokenizers, '_CL100K_SHA256', hashlib.sha256(tokenizer_bytes).hexdigest()),
    patch.object(load, 'read_file', return_value=tokenizer_bytes),
):
    assert run('--check') == 2
    # One verified model in a partial lock is not enough to launch.
    name, (component, revision) = next(iter(app.APP_MODELS.items()))
    cache = (Path.cwd() / 'data/models/snapshots').resolve()
    entry = model_assets.prepare_hub_entry(cache, component, revision, [(name, revision, 'primary')])
    model_assets._merge_model_lock(lock_path, {name: entry}, selection_package=app.SELECTION_PACKAGE)
    assert run('--check') == 2
    assert run() == 1
    partial, = (cache / 'docling-bundles').iterdir()
    assert partial.name.endswith('.partial')
    assert (partial / pin['folder'] / 'fixture.onnx').read_bytes() == b'fake'
    assert app.DOCLING_STANDARD not in load_model_lock(lock_path)
    assert run() == 0
    assert ocr_downloads == ['fixture.onnx', 'font.ttf', 'fixture.onnx']

    before = lock_path.read_bytes()
    with patch.object(model_assets, '_prepare_snapshot', side_effect=AssertionError('redownloaded')):
        with patch.object(load, 'read_file', side_effect=AssertionError('tokenizer fetched again')):
            assert run('--check') == run() == 0
            assert tokenizers.TiktokenOffsetTokenizer().count('hello') == 5
    assert lock_path.read_bytes() == before
    for dist in ('docling', 'docling-slim'):
        for missing in (False, True):
            def get_version(name):
                if name != dist:
                    return app.DOCLING_VERSION
                if missing:
                    raise PackageNotFoundError(name)
                return '0.0.0'

            with patch.object(model_assets, 'version', side_effect=get_version):
                assert run('--check') == run() == 1
    assert lock_path.read_bytes() == before

    lock = load_model_lock(lock_path)
    assert set(lock) == {*app.APP_MODELS, app.DOCLING_STANDARD}
    for key in lock:
        assert require_model(lock, key).path.is_relative_to(cache)

    with (
        patch.object(huggingface_hub.HfApi, 'model_info', side_effect=AssertionError('Hub call on recovery')),
        patch.object(huggingface_hub, 'snapshot_download', side_effect=AssertionError('redownload on recovery')),
        patch.object(sys.modules[ocr_module].RapidOcrModel, 'download_models', side_effect=AssertionError('OCR download on recovery')),
        patch.object(load, 'read_file', side_effect=AssertionError('tokenizer download on recovery')),
    ):
        lock_path.unlink()
        assert run('--check') == 2
        assert run() == run('--check') == 0
        assert lock_path.read_bytes() == before

        lock_path.unlink()
        bundle = Path(lock[app.DOCLING_STANDARD]['model_path'])
        layout = lock[app.DOCLING_STANDARD]['component_models']['layout']['directory']
        layout_file = bundle / layout / sibling.rfilename
        layout_file.write_bytes(b'corrupt docling')
        assert run() == 1
        assert layout_file.read_bytes() == b'corrupt docling'
        assert app.DOCLING_STANDARD not in load_model_lock(lock_path)
        layout_file.write_bytes(weights)
        assert run() == 0

        ocr_file = bundle / pin['folder'] / 'fixture.onnx'
        ocr_file.write_bytes(b'fake')
        assert run('--check') == run() == 1
        assert lock_path.read_bytes() == before
        lock_path.unlink()
        assert run() == 1
        assert ocr_file.read_bytes() == b'fake'
        ocr_file.write_bytes(ocr_files['fixture.onnx'])
        assert run() == 0

    config = Path('config/base.yaml').read_text()
    for candidate in (app.APP_GENERATOR, app.PRODUCTION_EMBEDDING_MODEL):
        invalid = Path('config/invalid.yaml')
        invalid.write_text(config.replace(candidate, 'unsupported/model'))
        with patch.dict(os.environ, {'EDUMIND_CONFIG': str(invalid)}):
            with patch.object(app, '_missing_models', side_effect=AssertionError('checked the wrong roster')):
                assert run('--check') == run() == 1

    damaged = Path(entry['model_path']) / 'fixture.weights'
    damaged.write_bytes(b'corrupt')
    assert run('--check') == run() == 1
    assert damaged.read_bytes() == b'corrupt'
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", offline],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell launcher")
@pytest.mark.parametrize(
    ("answer", "prepare_exit", "heartbeat_exit", "launches"),
    [
        ("yes", 0, 0, True),
        ("no", 0, 0, False),
        ("yes", 7, 0, False),
        ("yes", 0, 1, False),
    ],
)
def test_windows_launcher_starts_app_only_after_setup_succeeds(
    tmp_path, answer, prepare_exit, heartbeat_exit, launches
):
    release = tmp_path / "source release"
    release.mkdir()
    shutil.copy2(PROJECT_ROOT / "start.ps1", release)
    (release / "requirements").mkdir()
    (release / "requirements/app.lock").write_text("# Test requirements\n")

    # Real Python/PowerShell process boundaries, with cheap command stubs
    # instead of installing packages, downloading weights, or starting Docker.
    modules = tmp_path / "stubs"
    (modules / "edumind/rag").mkdir(parents=True)
    (modules / "edumind/__init__.py").touch()
    stub = """
import os
import sys
from pathlib import Path

name = Path(__file__).stem
args = sys.argv[1:]
with Path(os.environ['TEST_STEPS']).open('a', encoding='ascii') as handle:
    handle.write(' '.join([name, *args]) + '\\n')
if name == 'prepare_models':
    ready = Path(os.environ['TEST_STEPS']).with_suffix('.ready')
    if args == ['--check']:
        sys.exit(0 if ready.exists() else 2)
    status = int(os.environ['TEST_PREPARE_EXIT'])
    if status == 0:
        ready.touch()
    sys.exit(status)
"""
    for filename in ("pip.py", "streamlit.py", "edumind/prepare_models.py"):
        (modules / filename).write_text(stub, encoding="ascii")
    (modules / "edumind/rag/vector_store.py").write_text(
        """
import os
import sys
from pathlib import Path

def wait_for_chroma():
    with Path(os.environ['TEST_STEPS']).open('a', encoding='ascii') as handle:
        handle.write('chroma heartbeat\\n')
    sys.exit(int(os.environ['TEST_HEARTBEAT_EXIT']))
""",
        encoding="ascii",
    )
    harness = tmp_path / "harness.ps1"
    harness.write_text(
        """
function py {
    if ($args -contains 'venv') {
        $target = Join-Path $env:TEST_RELEASE '.venv'
        $scripts = New-Item -Path (Join-Path $target 'Scripts') -ItemType Directory
        Copy-Item -LiteralPath $env:TEST_PYTHON -Destination (Join-Path $scripts.FullName 'python.exe')
        if ($env:TEST_VENV_CONFIG) {
            Copy-Item -LiteralPath $env:TEST_VENV_CONFIG -Destination $target
        } else {
            Get-ChildItem -Path (Split-Path $env:TEST_PYTHON) -Filter 'python*.dll' |
                Copy-Item -Destination $scripts.FullName
        }
    }
    $global:LASTEXITCODE = 0
}
function ffmpeg {}
function docker {
    Add-Content -LiteralPath $env:TEST_STEPS -Value "docker $args"
    $global:LASTEXITCODE = 0
}
function Read-Host {
    Add-Content -LiteralPath $env:TEST_STEPS -Value 'prompt'
    $env:TEST_ANSWER
}
& (Join-Path $env:TEST_RELEASE 'start.ps1')
""",
        encoding="ascii",
    )
    python = Path(sys.executable)
    venv_config = python.parent.parent / "pyvenv.cfg"
    steps = tmp_path / "steps.txt"
    env = {
        **os.environ,
        "PYTHONPATH": str(modules),
        "TEST_RELEASE": str(release),
        "TEST_PYTHON": str(python),
        "TEST_VENV_CONFIG": str(venv_config) if venv_config.exists() else "",
        "TEST_STEPS": str(steps),
        "TEST_ANSWER": answer,
        "TEST_PREPARE_EXIT": str(prepare_exit),
        "TEST_HEARTBEAT_EXIT": str(heartbeat_exit),
    }

    def launch():
        return subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(harness),
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=45,
        )

    result = launch()
    observed = steps.read_text().splitlines()
    assert (
        "pip install -r requirements/app.lock" in observed
    ) and "prompt" in observed, result.stderr
    assert ("prepare_models" in observed) == (answer == "yes")
    starts_chroma = answer == "yes" and prepare_exit == 0
    if starts_chroma:
        assert observed.index(
            "docker compose -f infrastructure/chroma.yml up -d"
        ) < observed.index("chroma heartbeat")
    else:
        assert "docker compose -f infrastructure/chroma.yml up -d" not in observed
        assert "chroma heartbeat" not in observed
    assert ("streamlit run src/edumind/ui/streamlit_app.py" in observed) == launches
    assert (result.returncode == 0) == (prepare_exit == heartbeat_exit == 0), (
        result.stderr
    )
    if launches:
        result = launch()
        assert result.returncode == 0, result.stderr
        assert steps.read_text().splitlines()[len(observed) :] == [
            "docker info --format {{.ServerVersion}}",
            "docker compose version",
            "prepare_models --check",
            "docker compose -f infrastructure/chroma.yml up -d",
            "chroma heartbeat",
            "streamlit run src/edumind/ui/streamlit_app.py",
        ]
