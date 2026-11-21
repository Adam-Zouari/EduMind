# Install the EduMind environment

[Project overview](../../README.md) · [Documentation map](../README.md)

For the app on Windows, install the system requirements below, start Docker
Desktop, and run `.\start.ps1` from the extracted release folder in PowerShell.
The launcher sets up the Python environment, asks before downloading models,
and starts the app. Use the rest of this guide for manual setup or benchmarks.

Run the commands from the repository root in PowerShell. The app and benchmarks
use Python 3.12, but the app does not require the benchmark dependencies or
evaluation datasets.

## 1. System requirements

Install the tools needed for your chosen workflow:

| Program | Required for | Download |
|---|---|---|
| Python 3.12.x (64-bit) | application and benchmarks; include the Windows `py` launcher for first-time setup | [python.org](https://www.python.org/downloads/windows/) |
| Git | benchmark provenance and cloning the repository; not needed to run the app from an extracted release | [git-scm.com](https://git-scm.com/download/win) |
| Docker Desktop with Compose | Chroma, vector-server benchmarks, and official OmniDocBench table/formula scoring | [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) |
| FFmpeg | audio/video decoding and video keyframes | [FFmpeg download page](https://ffmpeg.org/download.html) |
| Tesseract 5 with English data | one Docling Standard configuration | [Tesseract installation](https://tesseract-ocr.github.io/tessdoc/Installation.html) and [Windows builds](https://tesseract-ocr.github.io/tessdoc/Downloads.html) |
| NVIDIA driver | Required for authoritative GPU-backed extraction and RAG comparisons; optional for the CPU application and CPU smoke/debug runs | [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) |

Start Docker Desktop before checking the engine. EduMind does not start Docker
or a model-serving service automatically.

```powershell
py -3.12 --version
docker version
docker compose version
ffmpeg -version
```

For benchmarks, check `git --version` as well. Only the Tesseract benchmark
configuration needs `tesseract --version` and
`tesseract --list-langs` (check that `eng` is listed). For CUDA profiles, check
`nvidia-smi` too. The provisional CPU app and CPU smoke runs do not need CUDA.
Authoritative GPU-backed benchmarks must use their declared hardware and
placement; see the [benchmark methodology](../benchmarks/methodology.md).

## 2. Create an isolated Python environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install **one** dependency lock in this environment:

| If you want to… | Command |
|---|---|
| Run the application only | `python -m pip install -r requirements/app.lock` |
| Run benchmarks, with or without the application | `python -m pip install -r requirements/benchmarks.lock` (includes `app.lock`) |

Then install the local project and check the environment:

```powershell
python -m pip install -e . --no-deps
python -m pip check
```

## 3. Configure CUDA for GPU benchmarks

On a Windows benchmark host using an NVIDIA driver compatible with CUDA 13,
install the matching PyTorch build after the benchmark dependency lock:

```powershell
python -m pip install --upgrade --force-reinstall --no-deps `
  torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130 `
  --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip check
```

Keep those versions aligned with `requirements/app.lock`. A positive CUDA
availability check verifies the installation, not candidate eligibility.
Smoke and preflight verify actual inference and device placement; see the
[measurement contract](../benchmarks/metrics.md#peak-device-vram).

The Windows Python 3.12 benchmark lock installs PaddlePaddle GPU 3.3.1 from
[Paddle's official CUDA 11.8 distribution](https://www.paddlepaddle.org.cn/packages/stable/cu118/paddlepaddle-gpu/).
It uses cuDNN 8 DLLs, separate from PyTorch's cuDNN 9 DLLs. Do not substitute
the CUDA 12.9 Paddle wheel in this shared Windows environment: its cuDNN 9
libraries conflict with the pinned PyTorch CUDA build.
The application lock installs ONNX Runtime GPU 1.30.0 and RapidOCR 3.9.2. These
GPU distributions also support explicit CPU smoke. Never co-install their CPU
distributions: they share Python package directories. The Chroma HTTP client is
`chromadb-client`; the database server remains in Docker.

When upgrading an existing environment that used the CPU packages, remove them
first, then reinstall the locks and the CUDA PyTorch build above:

```powershell
python -m pip uninstall -y onnxruntime paddlepaddle chromadb
python -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.30.0 chromadb-client==1.5.9
python -m pip install -r requirements/benchmarks.lock
python -c "import torch, onnxruntime as ort, paddle; print(ort.get_available_providers()); print(paddle.__version__, paddle.is_compiled_with_cuda())"
python -m pip check
```

If a CUDA 12 Paddle wheel was previously installed, uninstall it and its
`nvidia-*-cu12` packages before installing the benchmark lock; CUDA 11 and
CUDA 12 NVIDIA packages share Python directories. This project's ONNX Runtime
GPU 1.30.0 uses the pinned CUDA 13 PyTorch libraries. Do not remove libraries
needed by unrelated packages in another environment.

## 4. Choose the next step

- **Run the app manually:** [prepare the default models](model-preparation.md#preparation-commands),
  then [start Chroma and launch the assistant](running.md). No benchmark
  datasets, judge, or benchmark database services are needed.
- **Run a benchmark:** [prepare its candidates](../benchmarks/model-preparation.md#preparation-commands)
  or [database services](../benchmarks/running.md#9-vector-databases) as applicable.
  For authoritative runs, [prepare the data](../benchmarks/dataset-preparation.md)
  and [validate the manifest](../benchmarks/data-validation.md), then follow
  the [benchmark runbook](../benchmarks/running.md). Smoke uses committed
  fixtures; it does not require the full public datasets.
