# EduMind complete installation, download, and experiment guide

This is the end-to-end setup runbook for the provisional application and every
implemented benchmark. It lists the required software, Python packages, model
weights, Docker images, public datasets, local annotations, file locations,
verification commands, and experiment order. Run every command from the repository
root unless a section says otherwise.

There are two meanings of "complete":

- **Smoke-complete** means the committed tiny fixtures, real lightweight engines,
  Chroma, Ollama, and direct smoke runners work. This can be achieved entirely by
  following the automatic preparation steps below.
- **Research-complete** means standard/full extraction and RAG comparisons are valid.
  This additionally requires licensed public data, EduMind-specific samples, verified
  reference text/timestamps/table/formula annotations, frozen manifests, shortlisted
  summaries, and the final human review. No download command can replace those human
  verification steps.

A smoke run proves only that a real execution path works. Only standard/full runs on
frozen data can support model or strategy comparisons.

The complete procedure and formulas are in `experiments/benchmarks/benchmark_manual.md`; public evidence for every candidate decision is in `experiments/benchmarks/model_selection.md`.

## 1. Completion map: what is automatic and what is not

Preparation commands download model weights, QASPER, and pinned Docker images and then write machine-local lock files. They do not choose winners, edit `config/base.yaml`, fabricate extraction references, or promote a benchmark result into the application.

| Requirement | How it is obtained | Human work required? |
| --- | --- | --- |
| Python packages | pinned lock files in `requirements/` | no |
| Application models | `prepare.py app-models` | no |
| RAG embedding/reranker models | `prepare.py huggingface-models` | accept any gated licenses first |
| Ollama generation models | `prepare.py ollama-models` | no after Ollama is installed |
| Extraction model weights | `prepare.py extraction-models` | some candidates also need their official runtime |
| Four vector servers | `prepare.py vectordb` | Docker Desktop must be running |
| QASPER | `prepare.py qasper` | no |
| Smoke extraction/RAG data | committed to Git | no |
| OmniDocBench/olmOCR-Bench/OHR-Bench/PureDocBench/ASR data | official dataset downloads | license review and conversion to EduMind manifests |
| Structured RAG evidence | create three local manifests | yes: answers and offsets must be verified |
| Standard/full extraction manifests | create local manifests | yes: reference text, structure, timestamps, provenance |
| Human final-RAG review | export, judge, import | yes: 60 blinded judgments |

The repository already contains tiny smoke fixtures in:

```text
data/benchmarks/fixtures/extraction/
data/benchmarks/extraction/smoke.json
data/benchmarks/rag/smoke.json
```

Authoritative image/PDF/DOCX/audio/video data is not included. You must provide licensed source files plus verified annotations as described in section 7. QASPER is downloaded and transformed by the provided command.

Machine-specific downloads and locks are intentionally ignored by Git:

```text
data/benchmarks/raw/                 downloaded extraction assets
data/benchmarks/downloads/models/    explicitly downloaded extraction weights
data/benchmarks/models/*.json        resolved model/image revisions and digests
artifacts/benchmarks/                run outputs before MLflow logs them
mlflow.db, mlruns/, mlartifacts/     local experiment history
```

## 2. Required system software and official downloads

Install these programs before the Python dependencies:

| Software | Official source | Required for | Important installation choice |
| --- | --- | --- | --- |
| Python 3.11, 64-bit | [Python Windows downloads](https://www.python.org/downloads/windows/) and [Windows installation guide](https://docs.python.org/3.11/using/windows.html) | everything | Python 3.10+ is allowed by the package, but 3.11 is the safest common version for the pinned ML packages; enable `pip` and the launcher |
| Git for Windows | [official Git download](https://git-scm.com/downloads/win) | cloning and provenance | install Git Bash; normal defaults are sufficient |
| Docker Desktop | [official Windows installation](https://docs.docker.com/desktop/setup/install/windows-install/) | Chroma and vector-server benchmark | use the WSL 2 backend and Linux containers; Docker Compose v2 is bundled |
| Ollama | [official Windows download](https://ollama.com/download/windows) | generation and final RAG | keep the local service reachable at `http://127.0.0.1:11434` |
| Tesseract OCR 5 | [official installation documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html) and [download guidance](https://tesseract-ocr.github.io/tessdoc/Downloads.html) | image OCR | the Tesseract project does not publish a current Windows installer; its official docs link the maintained UB Mannheim builds; install English `eng.traineddata` and add the install directory to `PATH` |
| FFmpeg | [official download page](https://ffmpeg.org/download.html) | audio/video probing and keyframes | the project links Windows builds; add the directory containing `ffmpeg.exe` and `ffprobe.exe` to `PATH` |
| NVIDIA driver | [official driver download](https://www.nvidia.com/Download/index.aspx) | CUDA candidates and VRAM measurement | optional for smoke/CPU candidates; required for profiles configured with `device=cuda` |

Do not separately install PostgreSQL, Chroma, Qdrant, or Weaviate on Windows. Their
exact benchmark versions are Docker images prepared in section 8.

Check the executables:

```bash
python --version
python -c "import sys; print(sys.executable); print(sys.version)"
git --version
docker version
docker compose version
ollama --version
tesseract --version
tesseract --list-langs
ffmpeg -version
ffprobe -version
```

`tesseract --list-langs` must include `eng`. `docker version` must show both Client
and Server; seeing only Client means Docker Desktop is not running.

### Recommended free space

The exact total changes as model repositories are updated, but a full setup is large.
Reserve at least:

- 60-100 GB for Hugging Face, extraction, Paddle, and Docling model caches;
- roughly 40-50 GB for the six Ollama tags;
- 10-30 GB for public extraction/ASR datasets, depending on chosen subsets;
- 20 GB or more for Docker images, vector indexes, MLflow artifacts, and repetitions.

For a genuinely complete setup, start with **at least 150 GB free**, preferably on an
SSD. Full million-vector runs and copied dataset variants may need substantially more.

On Windows, start Docker Desktop and Ollama before commands that contact them. The committed audio/video smoke fixtures are ready to use; only regenerating them uses Windows `System.Speech`.

## 3. Create the isolated Python environment

Do not install these locks into a global Python environment. In Git Bash:

```bash
py -3.11 -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/app.lock
python -m pip install -r requirements/benchmarks.lock
python -m pip install -e . --no-deps
```

In PowerShell, activation is:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

`requirements/benchmarks.lock` includes `app.lock`, so installing it again is safe. The separate commands make the production-only dependency set visible. The editable install is performed once; later source edits do not require rebuilding.

Confirm that `pip` is attached to the virtual environment before downloading anything:

```bash
python -c "import sys; print(sys.executable)"
python -m pip --version
python -m pip check
```

Both paths must contain this repository's `.venv`. If they point to a global Python,
stop and reactivate the environment. The dependency-conflict warning produced by a
global `pip` installation is not evidence about this clean environment.

Copy `.env.example` to `.env` only when you need overrides. The normal defaults already point to `config/base.yaml`, Chroma on port 8001, Ollama on port 11434, and local MLflow SQLite.

### Hugging Face account, licenses, and cache location

Create a free [Hugging Face account](https://huggingface.co/join), create a read token
on the [token settings page](https://huggingface.co/settings/tokens), then authenticate
inside the activated environment. The official references are the
[authentication guide](https://huggingface.co/docs/huggingface_hub/package_reference/authentication),
[gated-model guide](https://huggingface.co/docs/hub/models-gated), and
[download/cache guide](https://huggingface.co/docs/huggingface_hub/guides/download).

```bash
hf auth login
hf auth whoami
```

Visit [EmbeddingGemma 300M](https://huggingface.co/google/embeddinggemma-300m) in a
browser while signed in and accept its terms before running the all-model download.
If another model page later becomes gated, accept its displayed terms instead of
trying to bypass the gate.

Hugging Face downloads are resumable and content-addressed. To place all Hugging Face
and `datasets` caches on a larger drive, set `HF_HOME` **before the first download**:

Git Bash:

```bash
export HF_HOME=/d/edumind-cache/huggingface
```

PowerShell:

```powershell
$env:HF_HOME = "D:\edumind-cache\huggingface"
```

Use the same value in every later terminal. Changing it makes already downloaded files
appear missing because the tools look in a different cache. Do not set
`HF_HUB_OFFLINE=1` during preparation; the benchmark itself enables offline behavior
where required after locks exist.

## 4. Minimum preparation for the application

The provisional application uses:

- `sentence-transformers/all-MiniLM-L6-v2` at revision `c9745ed1d9f207416be6d2e6f8de32d1f16199bf`;
- `Systran/faster-whisper-base.en` in int8 mode for audio/video;
- `qwen3:1.7b` through Ollama;
- Chroma server `chromadb/chroma:1.5.9`.

Download and lock the three models:

```bash
python experiments/benchmarks/prepare.py app-models
```

This writes/updates:

```text
data/benchmarks/models/huggingface.json
data/benchmarks/models/extraction.json
data/benchmarks/models/ollama.json
```

Start the provisional Chroma server and app:

```bash
docker compose -f infrastructure/chroma.yml up -d
streamlit run apps/streamlit_app.py
```

Stop it with:

```bash
docker compose -f infrastructure/chroma.yml down
```

The production defaults are token chunks of 256 tokens with 32-token overlap, MiniLM embeddings, dense top-5 retrieval under a 2,048-token context budget, and `qwen3:1.7b`. They are provisional and benchmarks never modify them automatically.

## 5. Models required by the experiments

### Hugging Face RAG models

Run:

```bash
python experiments/benchmarks/prepare.py huggingface-models
```

It resolves each repository to an immutable commit, downloads it through the Hugging
Face cache, and writes `data/benchmarks/models/huggingface.json`.

| Role | Exact repository used by code | Official model page |
| --- | --- | --- |
| embedding baseline | `sentence-transformers/all-MiniLM-L6-v2` | [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| embedding | `google/embeddinggemma-300m` | [model card and license gate](https://huggingface.co/google/embeddinggemma-300m) |
| embedding/compression | `infgrad/Jasper-Token-Compression-600M` | [model card](https://huggingface.co/infgrad/Jasper-Token-Compression-600M) |
| embedding | `Qwen/Qwen3-Embedding-0.6B` | [model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| embedding | `nvidia/Nemotron-3-Embed-1B-BF16` | [model card](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16) |
| embedding | `Qwen/Qwen3-Embedding-4B` | [model card](https://huggingface.co/Qwen/Qwen3-Embedding-4B) |
| embedding quality ceiling | `nvidia/Nemotron-3-Embed-8B-BF16` | [model card](https://huggingface.co/nvidia/Nemotron-3-Embed-8B-BF16) |
| reranker baseline | `cross-encoder/ms-marco-MiniLM-L6-v2` | [canonical model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) |
| reranker | `BAAI/bge-reranker-v2-m3` | [model card](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| reranker | `Qwen/Qwen3-Reranker-0.6B` | [model card](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |
| reranker quality ceiling | `Qwen/Qwen3-Reranker-4B` | [model card](https://huggingface.co/Qwen/Qwen3-Reranker-4B) |
| automated faithfulness diagnostic | `vectara/hallucination_evaluation_model` | [HHEM model card](https://huggingface.co/vectara/hallucination_evaluation_model) |

The first seven are embedding candidates, the next four are rerankers, and HHEM is a
diagnostic rather than a human judge. The weights stay in the Hugging Face cache; the
JSON lock records exactly which revisions were used. Downloading all of them can take
a long time. A safer resumable sequence is one candidate at a time:

```bash
python experiments/benchmarks/prepare.py huggingface-models --candidate sentence-transformers/all-MiniLM-L6-v2
python experiments/benchmarks/prepare.py huggingface-models --candidate google/embeddinggemma-300m
python experiments/benchmarks/prepare.py huggingface-models --candidate infgrad/Jasper-Token-Compression-600M
python experiments/benchmarks/prepare.py huggingface-models --candidate Qwen/Qwen3-Embedding-0.6B
python experiments/benchmarks/prepare.py huggingface-models --candidate nvidia/Nemotron-3-Embed-1B-BF16
python experiments/benchmarks/prepare.py huggingface-models --candidate Qwen/Qwen3-Embedding-4B
python experiments/benchmarks/prepare.py huggingface-models --candidate nvidia/Nemotron-3-Embed-8B-BF16
python experiments/benchmarks/prepare.py huggingface-models --candidate cross-encoder/ms-marco-MiniLM-L6-v2
python experiments/benchmarks/prepare.py huggingface-models --candidate BAAI/bge-reranker-v2-m3
python experiments/benchmarks/prepare.py huggingface-models --candidate Qwen/Qwen3-Reranker-0.6B
python experiments/benchmarks/prepare.py huggingface-models --candidate Qwen/Qwen3-Reranker-4B
python experiments/benchmarks/prepare.py huggingface-models --candidate vectara/hallucination_evaluation_model
```

The option is repeatable. Completed candidates are written to the lock immediately,
and rerunning uses Hugging Face's resumable cache. Interrupting a snapshot does not
delete already completed blobs; run the identical command again to resume.

### Ollama generation models

Run this once Ollama is running:

```bash
python experiments/benchmarks/prepare.py ollama-models
```

It pulls the following exact base tags sequentially and records their installed digests
in `data/benchmarks/models/ollama.json`:

| Exact tag | Official Ollama page | Benchmark profiles produced from it |
| --- | --- | --- |
| `qwen3:1.7b` | [Qwen3 1.7B](https://ollama.com/library/qwen3:1.7b) | baseline |
| `qwen3.5:4b-q4_K_M` | [Qwen3.5 4B Q4_K_M](https://ollama.com/library/qwen3.5:4b-q4_K_M) | direct and thinking |
| `qwen3.5:9b-q4_K_M` | [Qwen3.5 9B Q4_K_M](https://ollama.com/library/qwen3.5:9b-q4_K_M) | direct and thinking |
| `gemma4:12b-it-q4_K_M` | [Gemma 4 12B Q4_K_M](https://ollama.com/library/gemma4:12b-it-q4_K_M) | documented inference settings |
| `ministral-3:8b-instruct-2512-q4_K_M` | [Ministral 3 8B Q4_K_M](https://ollama.com/library/ministral-3:8b-instruct-2512-q4_K_M) | direct |
| `gpt-oss:20b` | [GPT-OSS 20B](https://ollama.com/library/gpt-oss:20b) | low and medium reasoning |

The benchmark creates nine profiles from those six tags: Qwen 3.5 4B/9B are each tested with direct and thinking modes, and GPT-OSS 20B is tested with low and medium reasoning. Do not pull separate model names for those profiles. `--candidate MODEL_TAG` also works here.

For maximum recoverability, pull and lock one tag per command:

```bash
python experiments/benchmarks/prepare.py ollama-models --candidate qwen3:1.7b
python experiments/benchmarks/prepare.py ollama-models --candidate qwen3.5:4b-q4_K_M
python experiments/benchmarks/prepare.py ollama-models --candidate qwen3.5:9b-q4_K_M
python experiments/benchmarks/prepare.py ollama-models --candidate gemma4:12b-it-q4_K_M
python experiments/benchmarks/prepare.py ollama-models --candidate ministral-3:8b-instruct-2512-q4_K_M
python experiments/benchmarks/prepare.py ollama-models --candidate gpt-oss:20b
```

Verify installed tags with:

```bash
ollama list
```

Do not rename tags or create aliases: the runner compares exact tag names and recorded
digests. Upgrade Ollama if a tag reports that it needs a newer engine.

### Extraction/OCR/ASR models

The Python lock already installs OpenAI Whisper, faster-whisper, PaddleOCR/PaddleX,
Docling, and Mammoth. The preparation command below downloads their weights plus the
Hugging Face snapshots used by the remaining candidates:

```bash
python experiments/benchmarks/prepare.py extraction-models
```

The exact candidates and authoritative pages are:

| Area | Prepared candidate or asset | Official source |
| --- | --- | --- |
| classic OCR | Tesseract 5 + English data | [Tesseract](https://github.com/tesseract-ocr/tesseract) |
| OCR | PP-OCRv5 English mobile detector/recognizer | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| OCR | PP-OCRv5 server detector/recognizer | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) |
| structured parsing | PP-StructureV3 | [PP-StructureV3 docs](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PP-StructureV3.html) |
| document VLM | PaddleOCR-VL-1.6 | [model card](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6), [PaddleOCR-VL docs](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html) |
| structured parsing | Docling local artifacts | [Docling](https://github.com/docling-project/docling) |
| document VLM | `zai-org/GLM-OCR` | [model](https://huggingface.co/zai-org/GLM-OCR), [runtime](https://github.com/zai-org/GLM-OCR) |
| document VLM | `opendatalab/MinerU2.5-Pro-2605-1.2B` | [model](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B), [runtime](https://github.com/opendatalab/MinerU) |
| document VLM | `allenai/olmOCR-2-7B-1025` | [model](https://huggingface.co/allenai/olmOCR-2-7B-1025), [runtime](https://github.com/allenai/olmocr) |
| ASR baseline | OpenAI Whisper `small.en` | [Whisper](https://github.com/openai/whisper) |
| ASR | faster-whisper tiny/base/small English | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| ASR ceiling | faster-whisper large-v3-turbo conversion | [model](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) |
| ASR | Distil-Whisper large v3.5 | [model card](https://huggingface.co/distil-whisper/distil-large-v3.5) |
| ASR | NVIDIA Parakeet TDT 0.6B v3 | [model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| ASR | NVIDIA Canary-Qwen 2.5B | [model card](https://huggingface.co/nvidia/canary-qwen-2.5b) |

The native text extractors `pypdf`, `pdfplumber`, `python-docx`, and `mammoth` have no
separate weights; they are installed by the lock files.

Large extraction preparation should normally be run one candidate at a time:

```bash
python experiments/benchmarks/prepare.py extraction-models --candidate openai-whisper-small-en
python experiments/benchmarks/prepare.py extraction-models --candidate faster-whisper-tiny-int8
python experiments/benchmarks/prepare.py extraction-models --candidate faster-whisper-base-int8
python experiments/benchmarks/prepare.py extraction-models --candidate faster-whisper-small-int8 --candidate faster-whisper-small-float16
python experiments/benchmarks/prepare.py extraction-models --candidate faster-whisper-turbo-int8
python experiments/benchmarks/prepare.py extraction-models --candidate distil-whisper-large-v3.5
python experiments/benchmarks/prepare.py extraction-models --candidate parakeet-tdt-0.6b-v3
python experiments/benchmarks/prepare.py extraction-models --candidate canary-qwen-2.5b
python experiments/benchmarks/prepare.py extraction-models --candidate paddleocr-v5-mobile
python experiments/benchmarks/prepare.py extraction-models --candidate paddleocr-v5-server
python experiments/benchmarks/prepare.py extraction-models --candidate docling
python experiments/benchmarks/prepare.py extraction-models --candidate pp-structure-v3
python experiments/benchmarks/prepare.py extraction-models --candidate paddleocr-vl-1.6
python experiments/benchmarks/prepare.py extraction-models --candidate glm-ocr
python experiments/benchmarks/prepare.py extraction-models --candidate mineru-2.5-pro
python experiments/benchmarks/prepare.py extraction-models --candidate olmocr-2-7b
```

OpenAI Whisper, faster-whisper, Hugging Face document-VLM, and ASR assets are placed under `data/benchmarks/downloads/models/`; PaddleOCR-VL and PP-StructureV3 use the prepared `~/.paddlex` cache and are locked by their PaddleOCR/PaddleX package versions. Tesseract is a separately installed system engine. pypdf, pdfplumber, Docling, python-docx, and Mammoth come from the lock files.

### Candidate-specific runtimes not contained in the main lock

The following candidates need an official runtime or service in addition to prepared
weights. The preparation lock intentionally does not pretend that a downloaded
snapshot is an executable parser.

Create CLI environments outside `.venv` so their fast-moving dependencies do not
replace the pinned benchmark stack:

```bash
py -3.11 -m venv .candidate-envs/glmocr
py -3.11 -m venv .candidate-envs/mineru
py -3.11 -m venv .candidate-envs/olmocr
```

Install each runtime using its current official instructions. At the time this guide
was verified, their documented package forms were:

```bash
.candidate-envs/glmocr/Scripts/python -m pip install "glmocr[selfhosted]"
.candidate-envs/mineru/Scripts/python -m pip install --upgrade pip
.candidate-envs/mineru/Scripts/python -m pip install -U "mineru[all]"
.candidate-envs/olmocr/Scripts/python -m pip install "olmocr[gpu]" --extra-index-url https://download.pytorch.org/whl/cu128
```

Before the image/PDF/DOCX run, expose all three CLI directories to the process:

Git Bash:

```bash
export PATH="$PWD/.candidate-envs/glmocr/Scripts:$PWD/.candidate-envs/mineru/Scripts:$PWD/.candidate-envs/olmocr/Scripts:$PATH"
glmocr --help
mineru --help
olmocr --help
```

Check the linked upstream README immediately before installation. In particular,
olmOCR documents a clean environment, Linux-oriented GPU dependencies, at least 12 GB
VRAM for local inference, and about 30 GB free disk. If a candidate cannot execute on
the benchmark machine, retain its visible failed result; do not silently substitute a
hosted model or another checkpoint.

Runtime details required by EduMind:

- **GLM-OCR:** serve the exact local directory recorded for `glm-ocr` in
  `data/benchmarks/models/extraction.json`. Create a YAML config with hosted MaaS
  disabled (`pipeline.maas.enabled: false`) and `pipeline.ocr_api.api_host` pointing to
  that self-hosted service. Put its path in each relevant sample's
  `options.glm_config_path`. The runner rejects hosted/API-key mode. Follow the
  [official self-hosted deployment instructions](https://github.com/zai-org/GLM-OCR)
  for vLLM/SGLang and keep the served model name tied to the prepared snapshot.
- **MinerU:** the preparation command creates `mineru-2.5-pro.json` and records it as
  `mineru_config_path`. The runner passes it, forces `MINERU_MODEL_SOURCE=local` and
  offline Hugging Face mode, and selects `vlm-transformers`; no unrecorded default
  checkpoint may replace it.
- **olmOCR:** the runner calls the `olmocr` executable with the prepared local
  `allenai/olmOCR-2-7B-1025` directory. Its official local GPU stack is the authority
  for CUDA/PyTorch/vLLM requirements.
- **Canary-Qwen:** the runner imports NVIDIA NeMo inside the Python process. Install
  `nemo_toolkit[asr]` according to the
  [official model card](https://huggingface.co/nvidia/canary-qwen-2.5b). If it conflicts
  with `.venv`, create a second complete benchmark environment, install both lock files
  and editable EduMind there, install NeMo, and run the audio benchmark from that
  environment. A separate CLI-only environment is not enough for Canary because it is
  imported rather than launched as a subprocess.

  ```bash
  python -m pip install -U "nemo_toolkit[asr]"
  python -c "from nemo.collections.speechlm2.models import SALM; print('Canary runtime: OK')"
  ```

The benchmark never silently downloads a missing model. A candidate that is not prepared is recorded as failed.

If a preparation command fails, do not manually create a lock file. Fix the missing tool/network/dependency and rerun it; an incomplete candidate must remain a visible failure rather than being silently skipped.

## 6. RAG data: QASPER plus structured educational evidence

Prepare the pinned `allenai/qasper` revision:

```bash
python experiments/benchmarks/prepare.py qasper
```

The command writes three text-evidence source manifests:

```text
data/benchmarks/rag/qasper-dev.json          100 papers for standard component selection
data/benchmarks/rag/qasper-validation.json    40 papers for full/finalist validation
data/benchmarks/rag/qasper-locked-test.json   40 papers for the one-time final test
```

The split unit is a paper. The preparation code verifies paper isolation, normalized evidence offsets, checksums, source revision, preprocessing version, and seed 42. If any evidence offset cannot be validated, preparation fails instead of dropping it.

QASPER alone is not sufficient to select section-aware versus structure-aware chunking because it does not provide the Markdown table/formula evidence needed by that decision. Create three independently verified manifests:

```text
data/benchmarks/rag/structured-dev.json
data/benchmarks/rag/structured-validation.json
data/benchmarks/rag/structured-locked-test.json
```

Use the same manifest header as QASPER. Each structured document is serialized exactly as the RAG system receives it, including headings, Markdown/HTML tables, and display formulas. A minimal document/question pair is:

```json
{
  "id": "structured-doc-001",
  "kind": "document",
  "text": "## Results\n\n| Model | Accuracy |\n|---|---:|\n| A | 91% |\n\n$$E=mc^2$$",
  "source": "verified-source-001.pdf"
},
{
  "id": "structured-q-001",
  "kind": "question",
  "document_id": "structured-doc-001",
  "question": "What accuracy is reported for model A?",
  "answer": "91%",
  "accepted_answers": ["91%"],
  "answerable": true,
  "answer_type": "extractive",
  "evidence_type": "table",
  "evidence": [
    {
      "document_id": "structured-doc-001",
      "start": 12,
      "end": 55,
      "text": "| Model | Accuracy |\n|---|---:|\n| A | 91% |"
    }
  ]
}
```

These offsets exactly select the displayed table string; calculate every real offset from the exact stored text in the same way. Every question must use `evidence_type` equal to `text`, `table`, `formula`, or `mixed`. Each structured split needs at least 10 answerable table questions, 10 formula questions, and 10 mixed-evidence questions, all with verified non-empty spans. This is a minimum validity gate, not proof that the corpus is representative. IDs must be globally unique across QASPER and the structured data. Keep source licenses, revisions, and split families in the manifest provenance. Seal the checksum after editing, then validate all three splits for exact/near-duplicate leakage:

```bash
python -m experiments.benchmarks.common.datasets seal data/benchmarks/rag/structured-dev.json data/benchmarks/rag/structured-validation.json data/benchmarks/rag/structured-locked-test.json
python -m experiments.benchmarks.common.datasets validate data/benchmarks/rag/structured-dev.json data/benchmarks/rag/structured-validation.json data/benchmarks/rag/structured-locked-test.json
```

Sealing proves only that the file is internally consistent; it does not verify that the reference answer or evidence annotation is factually correct. Those annotations still require human verification.

Combine QASPER and structured evidence one split at a time:

```bash
python experiments/benchmarks/prepare.py rag-selection --qasper-manifest data/benchmarks/rag/qasper-dev.json --structured-manifest data/benchmarks/rag/structured-dev.json --output data/benchmarks/rag/rag-selection-dev.json
python experiments/benchmarks/prepare.py rag-selection --qasper-manifest data/benchmarks/rag/qasper-validation.json --structured-manifest data/benchmarks/rag/structured-validation.json --output data/benchmarks/rag/rag-selection-validation.json
python experiments/benchmarks/prepare.py rag-selection --qasper-manifest data/benchmarks/rag/qasper-locked-test.json --structured-manifest data/benchmarks/rag/structured-locked-test.json --output data/benchmarks/rag/rag-selection-locked-test.json
python -m experiments.benchmarks.common.datasets validate data/benchmarks/rag/rag-selection-dev.json data/benchmarks/rag/rag-selection-validation.json data/benchmarks/rag/rag-selection-locked-test.json
```

The RAG standard/full runners use the `rag-selection-*` files by default. Do not inspect or tune against locked-test answers.

## 7. Data required by extraction/OCR experiments

### Required files and locations

Put downloaded source files under `data/benchmarks/raw/`. Create frozen manifests beside the committed smoke manifest:

```text
data/benchmarks/extraction/image-validation.json
data/benchmarks/extraction/image-locked-test.json
data/benchmarks/extraction/pdf-validation.json
data/benchmarks/extraction/pdf-locked-test.json
data/benchmarks/extraction/docx-validation.json
data/benchmarks/extraction/docx-locked-test.json
data/benchmarks/extraction/audio-validation.json
data/benchmarks/extraction/audio-locked-test.json
data/benchmarks/extraction/video-validation.json
data/benchmarks/extraction/video-locked-test.json
data/benchmarks/extraction/normalization-validation.json
data/benchmarks/extraction/normalization-locked-test.json
data/benchmarks/extraction/routing-validation.json
data/benchmarks/extraction/routing-locked-test.json
```

Each asset sample needs at least:

- unique `id` and correct `kind`;
- repository-relative `source_path` pointing to the real file;
- exact `asset_sha256`;
- verified normalized `reference` text;
- `source_license` and `source_revision`;
- `document_family` for family-level split isolation;
- annotations needed by the modality. Every authoritative PDF sample has `reference_pages` plus a same-length `reference_page_texts` array in page order; audio/video add word or segment timestamps, clip duration, and visible video text as applicable.
- `reference_elements` for pages/documents containing tables or formulas. A table element contains `{"kind": "table", "rows": [[...]], "page_number": 1}`; a formula contains `{"kind": "formula", "latex": "...", "page_number": 1}`. Bounding boxes may be added when the source benchmark supplies them.

The manifest header follows `data/benchmarks/extraction/smoke.json` and includes `name`, `version`, `task`, `split`, `source`, `license`, `revision`, `checksum`, `preprocessing_version`, and `split_seed`.

The runners enforce these minimum sample counts in both validation and locked-test manifests:

| Experiment | Minimum per manifest | Intended complete corpus |
| --- | ---: | ---: |
| Image/OCR | 24 pages | 120 pages: 72 development, 24 validation, 24 locked test |
| PDF | 12 documents | 60 documents: 36/12/12 |
| PDF routing | 12 PDF documents | uses the verified PDF routing corpus |
| DOCX | 9 documents | 45 documents: 27/9/9 |
| Audio | 18 clips | 90 clips: 54/18/18 |
| Video | 6 clips | 30 clips: 18/6/6 |
| Normalization | 40 cases | at least 200 deterministic corruption/preservation cases overall |

The image corpus should cover clean scans, noisy/skewed scans, phone photos, low resolution, and multi-column pages. PDF should cover digital, scanned, mixed, broken-encoding, slide, and academic layouts. Audio should cover clean/noisy speech, accents, technical vocabulary, and multiple speakers. Video needs verified transcripts, timestamps, and on-screen text. These categories are part of experimental validity, even when the minimum count check passes.

Tables and formulas are part of complete image/PDF/DOCX extraction. They are not separate upload types or separate product APIs. Web and dedicated form extraction remain out of scope.

### Authoritative public data sources

Use the official repositories/dataset pages and preserve their exact revision and license in the local manifest:

- [OmniDocBench v1.6 dataset](https://huggingface.co/datasets/opendatalab/OmniDocBench) and [official evaluator](https://github.com/opendatalab/OmniDocBench): **required primary source** for complete-page text, layout, reading order, tables, and formulas. Put the snapshot under `data/benchmarks/raw/omnidocbench/`. Do not mix releases in one manifest.
- [olmOCR-Bench dataset](https://huggingface.co/datasets/allenai/olmOCR-bench) and [official toolkit/evaluator](https://github.com/allenai/olmocr): **required challenge source** for old scans, multi-column pages, tiny text, tables, math, headers, and footers. Put it under `data/benchmarks/raw/olmocr-bench/`.
- [OHR-Bench dataset](https://huggingface.co/datasets/opendatalab/OHR-Bench) and [official code](https://github.com/opendatalab/OHR-Bench): **required extraction-to-RAG confirmation source**. Put it under `data/benchmarks/raw/ohr-bench/`; preserve its own splits and do not merge its downstream results into QASPER component selection.
- [PureDocBench dataset](https://huggingface.co/datasets/zhihengli-casia/puredocbench): **optional robustness track**, not required for the first authoritative result. Put it under `data/benchmarks/raw/puredocbench/` and report clean/digitally degraded/real-degraded conditions separately.
- [Open ASR Leaderboard dataset collection](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard) and [leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard): **required public ASR source**. It is about 26 GB as a complete snapshot. Put it under `data/benchmarks/raw/asr/open-asr-leaderboard/`, then select English clips from its documented constituent datasets and add EduMind technical-vocabulary/noise clips.
- [QASPER](https://huggingface.co/datasets/allenai/qasper): downloaded automatically by section 6 and combined with verified structured evidence for chunking, retrieval, generation, and final RAG.

OmniDocBench and olmOCR-Bench provide credible public parsing evidence, but EduMind still runs its own benchmark because it requires English-specific strata, exact typed output/provenance, the selected table/formula serialization, local latency/resources, and downstream retrieval/answer quality. OHR-Bench directly helps the downstream confirmation but does not replace QASPER or the component experiments.

### Exact public-dataset download commands

Create the directories, resolve the current official repository commits, and download
those exact commits. The variables and downloads below must be run in the same Git Bash
terminal. Copy the printed SHAs into the matching manifest `revision` fields:

```bash
mkdir -p data/benchmarks/raw
OMNIDOC_REV=$(python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('opendatalab/OmniDocBench').sha)")
OLMOCR_BENCH_REV=$(python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('allenai/olmOCR-bench').sha)")
OHR_REV=$(python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('opendatalab/OHR-Bench').sha)")
OPEN_ASR_REV=$(python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('hf-audio/open-asr-leaderboard').sha)")
printf 'OmniDocBench=%s\nolmOCR-Bench=%s\nOHR-Bench=%s\nOpen-ASR=%s\n' "$OMNIDOC_REV" "$OLMOCR_BENCH_REV" "$OHR_REV" "$OPEN_ASR_REV"
hf download opendatalab/OmniDocBench --repo-type dataset --revision "$OMNIDOC_REV" --local-dir data/benchmarks/raw/omnidocbench
hf download allenai/olmOCR-bench --repo-type dataset --revision "$OLMOCR_BENCH_REV" --local-dir data/benchmarks/raw/olmocr-bench
hf download opendatalab/OHR-Bench --repo-type dataset --revision "$OHR_REV" --local-dir data/benchmarks/raw/ohr-bench
hf download hf-audio/open-asr-leaderboard --repo-type dataset --revision "$OPEN_ASR_REV" --local-dir data/benchmarks/raw/asr/open-asr-leaderboard
```

Download the optional robustness dataset only if you intend to report that track:

```bash
PUREDOC_REV=$(python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('zhihengli-casia/puredocbench').sha)")
echo "PureDocBench=$PUREDOC_REV"
hf download zhihengli-casia/puredocbench --repo-type dataset --revision "$PUREDOC_REV" --local-dir data/benchmarks/raw/puredocbench
```

Re-running the same command is safe. Do not use a moving `main` snapshot for a frozen
benchmark without recording its resolved SHA.

The official evaluator repositories are code, not duplicates of the dataset files.
Clone them under a tool-only directory if you will report their named metrics:

```bash
mkdir -p data/benchmarks/tools
git clone https://github.com/opendatalab/OmniDocBench.git data/benchmarks/tools/OmniDocBench
git clone https://github.com/allenai/olmocr.git data/benchmarks/tools/olmocr
git clone https://github.com/opendatalab/OHR-Bench.git data/benchmarks/tools/OHR-Bench
```

Record each repository commit with `git -C PATH rev-parse HEAD`. Install evaluator
dependencies in separate environments when they conflict with EduMind. A metric may be
called `TEDS`, `CDM`, or an official olmOCR-Bench score only when that official
evaluator actually ran and its revision was logged.

### Data that must be created or curated locally

The public downloads above are source pools; the EduMind runners do **not** accept
their native layouts directly. Complete these steps before claiming standard/full
readiness:

1. Select English pages/documents by `document_family`, never by individual question,
   so pages from one source document cannot leak across splits.
2. Build the 72/24/24 image split from OmniDocBench plus verified educational phone
   photos and low-resolution/noisy pages. Keep table/formula annotations when present.
3. Build the 36/12/12 PDF split across digital, scanned, mixed, broken-encoding, slide,
   and academic documents. Use OHR/olmOCR/OmniDoc sources only where their licenses and
   family boundaries allow it.
4. Create 27/9/9 DOCX documents. Public PDF parsing sets do not supply the required
   OOXML references. Preserve original paragraphs, headings, list levels, captions,
   table cells/relations, embedded images, and Office Math expressions as ground truth.
5. Select 54/18/18 English audio clips from the Open ASR source pool plus locally
   licensed educational/technical/noisy/accented clips. Normalize all to a documented
   reference convention and include word/segment timestamps where timing is scored.
6. Create 18/6/6 short educational videos with permission to evaluate them. Store the
   verified speech transcript/timestamps and every relevant on-screen text interval.
   A video URL alone is not a reproducible licensed asset.
7. Create at least 200 normalization cases with `input`, expected preserved content,
   and explicitly identified corruption to remove.
8. Derive routing samples from verified PDFs and label the correct native/OCR/hybrid
   choice only after comparing page references; do not use a router's own prediction
   as ground truth.
9. Create the three structured RAG manifests from section 6 with independently checked
   table, formula, and mixed evidence spans.
10. Run `seal` only after annotation review, then run cross-split `validate`.

This is the unavoidable manual part. As currently implemented, the repository cannot
truthfully offer a single command that converts all these heterogeneous public datasets
into authoritative EduMind manifests, because automatic conversion would invent or
silently discard the exact reference fields the metrics depend on.

### Licensed asset download plan

For reproducible downloads, create a reviewed JSON plan such as:

```json
{
  "assets": [
    {
      "url": "https://authoritative.example/dataset/page-001.png",
      "filename": "page-001.png",
      "sha256": "64-lowercase-hex-characters",
      "license": "license-name"
    }
  ]
}
```

Then run:

```bash
python experiments/benchmarks/prepare.py assets --plan extraction-assets.json --output data/benchmarks/raw
```

The downloader accepts HTTPS only, stores each entry by its filename under the output directory, streams each file, validates its SHA-256, and removes partial downloads. It deliberately does not generate reference text or annotations: those must be independently verified before a standard/full manifest is frozen.

## 8. Vector database servers

The production app needs only Chroma. The vector benchmark needs four real servers:

| Candidate | Pinned server image | Loopback ports |
| --- | --- | --- |
| [Chroma server](https://docs.trychroma.com/guides/deploy/docker) | `chromadb/chroma:1.5.9` | 8001 |
| [Qdrant server](https://qdrant.tech/documentation/installation/) | `qdrant/qdrant:v1.17.0` | 6333 REST, 6334 gRPC |
| [Weaviate](https://docs.weaviate.io/deploy/installation-guides/docker-installation) | `cr.weaviate.io/semitechnologies/weaviate:1.38.2` | 8080 HTTP, 50051 gRPC |
| [PostgreSQL/pgvector](https://github.com/pgvector/pgvector) | `pgvector/pgvector:0.8.2-pg17-bookworm` | 5433 PostgreSQL |

Prepare images and immutable digests:

```bash
python experiments/benchmarks/prepare.py vectordb
```

This also pulls the pinned Alpine inspector used only to measure Docker-volume size and writes:

```text
data/benchmarks/models/vectordb.json
experiments/benchmarks/vectordb/.env
```

`vectordb.json` records immutable image digests and matching Python client versions.
Do not replace `.env` with `latest` tags: doing so makes the server being measured
different from the recorded plan.

Stop the production Chroma Compose stack before starting the benchmark stack because both use port 8001:

```bash
docker compose -f infrastructure/chroma.yml down
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml up -d
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml ps
```

All four rows must be running before the benchmark starts. Basic endpoint checks are:

```bash
curl --fail http://127.0.0.1:8001/api/v2/heartbeat
curl --fail http://127.0.0.1:6333/
curl --fail http://127.0.0.1:8080/v1/.well-known/ready
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml exec -T pgvector pg_isready -U edumind -d edumind
```

If a server is still starting, inspect its real error instead of repeatedly launching
new stacks:

```bash
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml logs --tail=200
```

Run the real-server smoke benchmark:

```bash
python experiments/benchmarks/vectordb/run.py --profile smoke
```

When finished:

```bash
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml down
```

Use `down -v` only when you intentionally want to delete all benchmark database volumes and start a clean run. Never use it on the production Compose file unless you intend to delete the production Chroma index.

The vector benchmark checks health, vector dimension, cosine behavior, compound/empty filters, replacement, deletion, persistence after restart, and real ANN index reporting before performance evidence is accepted. NumPy supplies only the exact-neighbor oracle; it is not a production database candidate.

## 9. MLflow and artifact locations

MLflow logging is enabled by default. Start its local browser in a separate terminal:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://127.0.0.1:5000`. Each invocation creates one parent run and one child per candidate. Important files include `plan.json`, `provenance.json`, `summary.json`, candidate JSON, and per-sample Parquet. Failed candidates remain visible. Use `--no-mlflow` only for debugging a runner.

Generated summaries are normally under:

```text
artifacts/benchmarks/<family>/<experiment>/<run-id>/summary.json
```

Pass those exact summaries to downstream experiments; do not copy candidate names by hand when a runner expects evidence from an upstream selection.

## 10. Recommended experiment order

First verify the repository invariants:

```bash
python -m pytest tests/test_benchmark_metrics.py tests/test_benchmark_datasets.py
```

Then use this order.

### Extraction chain

```bash
python experiments/benchmarks/extraction/normalization/run.py --profile smoke
python experiments/benchmarks/extraction/image/run.py --profile standard
python experiments/benchmarks/extraction/audio/run.py --profile standard
python experiments/benchmarks/extraction/docx/run.py --profile standard
python experiments/benchmarks/extraction/pdf/run.py --profile standard --image-summary IMAGE_SUMMARY_JSON
python experiments/benchmarks/extraction/routing/run.py --profile standard --image-summary IMAGE_SUMMARY_JSON
python experiments/benchmarks/extraction/video/run.py --profile standard --image-summary IMAGE_SUMMARY_JSON --audio-summary AUDIO_SUMMARY_JSON
```

Image must be selected before hybrid PDF/routing, and image plus audio must be selected before video. Full runs additionally require `--shortlist STANDARD_SUMMARY_JSON` and use the corresponding locked-test manifest.

### RAG chain

```bash
python experiments/benchmarks/rag/chunking_embedding/run.py --profile standard
python experiments/benchmarks/rag/retrieval/run.py --profile standard --embedding-summary EMBEDDING_SUMMARY_JSON
python experiments/benchmarks/rag/generation/run.py --profile standard
python experiments/benchmarks/rag/final/run.py --profile standard --retrieval-summary RETRIEVAL_SUMMARY_JSON --generation-summary GENERATION_SUMMARY_JSON
```

Chunking/embedding uses exact NumPy search so database approximation cannot affect strategy selection. Generation uses frozen contexts. Final RAG crosses only shortlisted components.

### Vector database chain

After the real-server smoke succeeds:

```bash
python experiments/benchmarks/vectordb/run.py --profile standard
```

Full runs require an explicitly reduced server shortlist and exactly one approved embedding result:

```bash
python experiments/benchmarks/vectordb/run.py --profile full --shortlist VECTOR_SUMMARY_JSON --embedding-summary EMBEDDING_SUMMARY_JSON
```

Complete retrieval on the database finalists is a separate stage:

```bash
python experiments/benchmarks/vectordb/retrieval_run.py --database-summary VECTOR_SUMMARY_JSON --embedding-summary EMBEDDING_SUMMARY_JSON --retrieval-summary RETRIEVAL_SUMMARY_JSON
```

### Human review and locked test

Export three anonymous systems across 20 questions:

```bash
python experiments/benchmarks/review.py export FINAL_SUMMARY_JSON human-review.csv
```

Fill the rubric columns without viewing the adjacent identity file, then import:

```bash
python experiments/benchmarks/review.py import human-review.csv
```

The locked test accepts exactly one approved candidate, a complete 60-judgment result, and the explicit `--confirm-locked-test` flag. Read `experiments/benchmarks/rag/final/doc.md` before running it; the runner writes a marker and refuses a second version-1 locked-test run.

## 11. Complete setup verification

Run this section after the downloads. It separates a missing installation from a model
or dataset problem before an expensive benchmark starts.

### Python and executable checks

```bash
python -c "import sys; print(sys.executable); assert '.venv' in sys.executable"
python -m pip check
python -c "import chromadb, datasets, docling, faster_whisper, fitz, mlflow, numpy, paddleocr, pandas, pyarrow, sentence_transformers, torch, transformers, weaviate; print('main Python imports: OK')"
python -c "from qdrant_client import QdrantClient; from psycopg import connect; print('vector clients: OK')"
tesseract --list-langs
ffmpeg -version
ffprobe -version
ollama list
docker version
docker compose version
```

Expected results:

- Python points inside `.venv` and `pip check` reports no broken requirements.
- `tesseract --list-langs` contains `eng`.
- `ollama list` contains all six exact tags from section 5.
- Docker reports a running Server, not only a Client.

### Lock-file checks

All four machine-local locks must exist after full preparation:

```bash
python -c "import json, pathlib; paths=['data/benchmarks/models/huggingface.json','data/benchmarks/models/extraction.json','data/benchmarks/models/ollama.json','data/benchmarks/models/vectordb.json']; [(json.loads(pathlib.Path(p).read_text()), print('OK', p)) for p in paths]"
```

Inspect them once. Every candidate name in `candidates.yaml` that requires a model must
have a matching entry, and extraction entries containing `model_path` or an artifacts
directory must point to a path that still exists. Never hand-edit a digest to make this
check pass; rerun the corresponding preparation command.

### Dataset checks

The committed smoke manifests should validate immediately:

```bash
python -m experiments.benchmarks.common.datasets validate data/benchmarks/rag/smoke.json
python -m experiments.benchmarks.common.datasets validate data/benchmarks/extraction/smoke.json
```

After QASPER and manual manifests are ready, validate all related splits together so
cross-split leakage is checked:

```bash
python -m experiments.benchmarks.common.datasets validate data/benchmarks/rag/qasper-dev.json data/benchmarks/rag/qasper-validation.json data/benchmarks/rag/qasper-locked-test.json
python -m experiments.benchmarks.common.datasets validate data/benchmarks/rag/structured-dev.json data/benchmarks/rag/structured-validation.json data/benchmarks/rag/structured-locked-test.json
python -m experiments.benchmarks.common.datasets validate data/benchmarks/rag/rag-selection-dev.json data/benchmarks/rag/rag-selection-validation.json data/benchmarks/rag/rag-selection-locked-test.json
```

Validate each extraction family as a validation/locked-test pair in the same way. A
file merely existing is not sufficient; checksums, evidence offsets, asset hashes, and
family isolation must pass.

### Small validity checks

These are the intentionally small checks retained by the project; they do not lint,
build a wheel, calculate coverage, or install every optional stack:

```bash
python -m pytest tests/test_benchmark_metrics.py tests/test_benchmark_datasets.py
```

### Direct smoke runs

Run every family directly. `--no-mlflow` is useful here because the purpose is path
verification, not experiment history:

```bash
python experiments/benchmarks/extraction/normalization/run.py --profile smoke --no-mlflow
python experiments/benchmarks/extraction/image/run.py --profile smoke --no-mlflow
python experiments/benchmarks/extraction/pdf/run.py --profile smoke --no-mlflow
python experiments/benchmarks/extraction/docx/run.py --profile smoke --no-mlflow
python experiments/benchmarks/extraction/audio/run.py --profile smoke --no-mlflow
python experiments/benchmarks/extraction/routing/run.py --profile smoke --no-mlflow
python experiments/benchmarks/extraction/video/run.py --profile smoke --no-mlflow

python experiments/benchmarks/rag/chunking_embedding/run.py --profile smoke --no-mlflow
python experiments/benchmarks/rag/retrieval/run.py --profile smoke --no-mlflow
python experiments/benchmarks/rag/generation/run.py --profile smoke --no-mlflow
python experiments/benchmarks/rag/final/run.py --profile smoke --no-mlflow
```

Then start the four-server Compose stack from section 8 and run:

```bash
python experiments/benchmarks/vectordb/run.py --profile smoke --no-mlflow
```

All smoke candidates must finish successfully. Smoke does not execute every large
candidate and therefore does not prove those candidate-specific runtimes work. Before
standard runs, test `glmocr --help`, `mineru --help`, `olmocr --help`, and a one-file
official example for each runtime in the environment where it will be measured.

### Application check

Stop the benchmark vector stack, start the production Chroma stack, and launch the app:

```bash
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml down
docker compose -f infrastructure/chroma.yml up -d
streamlit run apps/streamlit_app.py
```

Upload one committed fixture from `data/benchmarks/fixtures/extraction/`, ingest it,
ask a question, confirm that the answer includes a source citation, and confirm that a
second Streamlit rerun does not duplicate the document. This is the final application
path check.

## 12. Troubleshooting downloads and setup

### A Hugging Face download looks frozen

Large Xet-backed snapshots can spend a long time reconstructing files after the byte
counter is nearly complete. First wait while disk activity continues. If it must be
interrupted, already completed cache blobs are not wasted; run the identical
`prepare.py ... --candidate ...` command again and `snapshot_download` resumes.

If Ctrl+C does not return the terminal, open a new PowerShell window and identify the
processes without closing Docker or Ollama:

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|hf|xet' } | Select-Object ProcessId, ParentProcessId, Name, CommandLine
```

Kill only the stuck Python process tree from a new Git Bash or Command Prompt:

```bash
taskkill //PID PROCESS_ID //T //F
```

Or from PowerShell:

```powershell
Stop-Process -Id PROCESS_ID -Force
```

Do not delete the Hugging Face cache or `*.incomplete` blobs first. Retry the exact
candidate. Delete cache data only after a reproducible checksum/corruption error.

### Hugging Face returns 401 or 403

Run `hf auth whoami`, sign in to the model page, accept its terms, and retry. Tokens
belong in Hugging Face's credential store or an environment variable, never in Git,
`.env`, a manifest, MLflow parameters, or screenshots.

### `pip` reports unrelated dependency conflicts

Run `python -c "import sys; print(sys.executable)"`. If it is outside `.venv`, you used
global `pip`; reactivate `.venv` and reinstall the two locks there. Do not try to make
EduMind compatible with unrelated packages already installed globally.

### A command cannot find a model that was downloaded

Check that the same `HF_HOME` is active, then inspect the relevant JSON lock. Hugging
Face RAG weights are in the shared cache; extraction weights are intentionally copied
under `data/benchmarks/downloads/models/`; Paddle uses `%USERPROFILE%/.paddlex`;
Ollama uses its own model store. Moving one of these directories invalidates paths in
the extraction lock and requires preparation again.

### Docker reports an occupied port

Production Chroma and benchmark Chroma both use `127.0.0.1:8001`. Stop one Compose
stack before starting the other. On PowerShell, identify an unexpected listener with:

```powershell
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in 8001,6333,6334,8080,50051,5433,11434 } | Select-Object LocalAddress,LocalPort,OwningProcess
```

### CUDA/OOM failures

A CUDA profile must fail visibly when its required hardware/runtime is unavailable;
do not silently switch it to CPU and compare the result as if it were the same
candidate. Reduce unrelated GPU usage, unload Ollama models, verify `nvidia-smi`, and
rerun. If the candidate still cannot execute, retain the failed candidate artifact and
do not promote it.

### A standard/full run says a manifest or summary is missing

That is an ordering error, not a package error. Create the frozen manifest from
sections 6-7, run the upstream standard stage, review its Pareto set, and pass the
actual generated `summary.json`. Full profiles always require an approved shortlist;
the final locked-test stage additionally requires imported 60-judgment review results
and `--confirm-locked-test`.

## 13. Final research-readiness checklist

Do not call the installation complete until every applicable box is true:

- [ ] System commands report Python 3.11, Docker/Compose, Ollama, Tesseract `eng`,
  FFmpeg, and FFprobe.
- [ ] `.venv` contains both lock files and the editable project; `pip check` passes.
- [ ] Hugging Face authentication and gated-model acceptance are complete.
- [ ] All 12 RAG model entries exist in `huggingface.json`.
- [ ] All six exact Ollama tags exist and their digests are in `ollama.json`.
- [ ] All extraction candidates are present in `extraction.json`; GLM-OCR, MinerU,
  olmOCR, and Canary runtime checks pass where they will be benchmarked.
- [ ] All five Docker images, including the inspector, are digest-pinned in
  `vectordb.json`; all four real servers pass smoke conformance.
- [ ] QASPER dev/validation/locked-test manifests were produced from the pinned commit.
- [ ] Structured RAG dev/validation/locked-test manifests contain verified table,
  formula, and mixed evidence and pass leakage checks.
- [ ] Required public extraction/ASR source snapshots and evaluator revisions are
  recorded with their licenses.
- [ ] Image, PDF, DOCX, audio, video, normalization, and routing manifests meet the
  required sample counts, annotations, hashes, provenance, and family isolation.
- [ ] Metric/dataset checks and every direct smoke family succeed.
- [ ] Standard stages produce real per-sample artifacts and Pareto summaries.
- [ ] Full stages use explicitly reviewed shortlists rather than all candidates.
- [ ] Sixty blinded human judgments are imported before the one-time locked test.

## 14. What a successful setup does and does not prove

A valid smoke run means dependencies, prepared files, real engines, and data contracts
connect correctly. It does not prove one model/server is better. A standard/full
comparison is valid only when all candidates use the same frozen manifest, prepared
model locks, environment, and successful correctness gates. Reports are tied to their
recorded hardware and software environment.

No benchmark result changes production automatically. After reviewing the Pareto set,
confidence intervals, human judgments, operational gates, and limitations, update
production code/config in a separate explicit change. Until then, Chroma, MiniLM,
token 256/32, dense top-5 retrieval, and Qwen 3 1.7B remain provisional defaults.
