# EduMind installation and benchmark preparation

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Run only the application](running.md) ·
[Benchmark overview](../benchmarks/overview.md)

This guide prepares the provisional application and every approved benchmark candidate. The authoritative shortlist is described in [model selection](../benchmarks/model-selection.md); exact executable revisions are read from [selection evidence](../../experiments/benchmarks/selection_evidence.csv). Excluded rows are historical evidence and are never downloaded.

## 1. System requirements

Install these programs before creating the Python environment:

| Program | Required for | Download |
|---|---|---|
| Python 3.11 (64-bit) | application and benchmarks | [python.org](https://www.python.org/downloads/windows/) |
| Git | editable installation and provenance | [git-scm.com](https://git-scm.com/download/win) |
| Docker Desktop with Compose | Chroma and vector-server benchmarks | [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) |
| FFmpeg | audio/video decoding and video keyframes | [FFmpeg download page](https://ffmpeg.org/download.html) |
| Tesseract 5 with English data | one Docling Standard configuration | [Tesseract installation](https://tesseract-ocr.github.io/tessdoc/Installation.html) and [Windows builds](https://tesseract-ocr.github.io/tessdoc/Downloads.html) |
| NVIDIA driver | CUDA benchmark profiles | [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) |

The application does not require a separate model-serving service. Generation uses exact local Hugging Face snapshots directly. Docker is never started by an import, preparation command, or application startup.

Check the system tools:

```powershell
py -3.11 --version
git --version
docker version
docker compose version
ffmpeg -version
tesseract --version
tesseract --list-langs
nvidia-smi
```

`eng` must appear in the Tesseract language list. CUDA is optional for the provisional CPU application, but a standard generation comparison must explicitly use one common `--device cpu` or `--device cuda` for every candidate.

## 2. Create an isolated Python environment

Do not install these locks into a global Python environment.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/app.lock
python -m pip install -r requirements/benchmarks.lock
python -m pip install -e . --no-deps
python -m pip check
```

Source edits are immediately visible because the repository is installed in editable mode. Rebuilding a wheel is not part of the development workflow.

## 3. Model storage and immutable lock

All prepared weights are stored inside:

```text
data/benchmarks/downloads/models/
```

Hugging Face transfer metadata/cache and the pinned tiktoken encoding are also
kept under `data/benchmarks/downloads/huggingface/` and
`data/benchmarks/downloads/tiktoken/`; preparation does not rely on the user's
shared model cache.

Successful preparation writes one generated file:

```text
data/benchmarks/models/selected.json
```

That file records the exact repository, revision, local path, composite submodels, and Docling artifacts used by runs. A benchmark refuses missing paths, excluded candidates, and revisions that disagree with `selection_evidence.csv`. Do not edit `selected.json` manually.

Inspect the complete download plan without network access:

```powershell
python experiments/benchmarks/prepare.py --list
python experiments/benchmarks/prepare.py all-models --dry-run
```

Prepare only the provisional application controls:

```powershell
python experiments/benchmarks/prepare.py app-models
```

This installs MiniLM embeddings, Hugging Face Qwen3 1.7B, Whisper `small.en`, and Docling Standard artifacts. Prepare experiment candidates separately or together:

```powershell
python experiments/benchmarks/prepare.py rag-models
python experiments/benchmarks/prepare.py extraction-models
python experiments/benchmarks/prepare.py all-models
```

Downloads are resumable through Hugging Face and are placed in deterministic project directories. Preparation never substitutes a newer repository head when the pinned revision is unavailable.

### RAG models

The RAG download contains these exact approved identities:

- Embeddings: [MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [Snowflake Arctic Embed M v2](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0), [F2LLM v2 0.6B](https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B), [Octen 0.6B](https://huggingface.co/Octen/Octen-Embedding-0.6B), [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [Nemotron Embed 1B](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16), [Octen 4B](https://huggingface.co/Octen/Octen-Embedding-4B), and [Qwen3 Embedding 4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B).
- Rerankers: [MiniLM control](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), [Ettin 150M](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1), [Ettin 400M](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1), [Ettin 1B](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1), and [Qwen3 Reranker 4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B).
- Generators: [Qwen3 1.7B control](https://huggingface.co/Qwen/Qwen3-1.7B), [MiniCPM5 1B](https://huggingface.co/openbmb/MiniCPM5-1B), [G9v3 3B](https://huggingface.co/ai9stars/G9v3-3B), and [Qwen3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B).
- Diagnostic evaluator: [HHEM](https://huggingface.co/vectara/hallucination_evaluation_model).

Generators run with native checkpoint precision, no quantization, no automatic CPU/GPU split, temperature 0, and seed 42. The benchmark records the whole-model device.

### Extraction models

The extraction download contains:

- [Docling Standard](https://github.com/docling-project/docling/releases/tag/v2.117.0) layout, TableFormer, CodeFormula, RapidOCR, and EasyOCR components. Tesseract remains a system executable.
- [Granite Docling 258M](https://huggingface.co/ibm-granite/granite-docling-258M).
- [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) plus its Paddle layout components in a project-controlled cache.
- [Whisper small.en](https://huggingface.co/openai/whisper-small.en), [Canary 180M](https://huggingface.co/nvidia/canary-180m-flash), [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [MOSS Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize), and [Qwen3 ASR 1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf).
- The Qwen ASR profile also downloads the pinned [Qwen3 ForcedAligner 0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B). ASR and alignment execute sequentially, while their combined time and peak resource use are measured.

NeMo and MOSS have heavier runtime dependency trees. Install them only inside the project environment from `requirements/benchmarks.lock`; do not combine this lock with unrelated ML environments.

## 4. Datasets

Tiny committed fixtures support smoke execution. Smoke validates wiring only and cannot establish quality or speed claims.

### RAG: QASPER

The preparation script downloads the pinned [QASPER dataset](https://huggingface.co/datasets/allenai/qasper) and creates paper-isolated development, validation, and locked-test manifests:

```powershell
python experiments/benchmarks/prepare.py qasper
```

Files are written under `data/benchmarks/rag/`. Structured table/formula/mixed questions require a separately verified manifest with exact normalized-text evidence offsets. Combine it with a QASPER split only after annotation:

```powershell
python experiments/benchmarks/prepare.py rag-selection `
  --qasper-manifest data/benchmarks/rag/qasper-validation.json `
  --structured-manifest data/benchmarks/rag/structured-validation.json `
  --output data/benchmarks/rag/rag-selection-validation.json
```

### Document extraction

Use licensed source documents and retain original checksums and source revisions. Relevant public sources are:

- [OmniDocBench](https://github.com/opendatalab/OmniDocBench): full document parsing, reading order, tables, and formulas.
- [olmOCR-Bench](https://huggingface.co/datasets/allenai/olmOCR-bench): difficult scanned and born-digital pages.
- [OHR-Bench](https://huggingface.co/datasets/opendatalab/OHR-Bench): retrieval-oriented document understanding confirmation.
- [PureDocBench](https://github.com/opendatalab/PureDocBench): structured document parsing cases.

The repository does not silently redistribute these corpora. Create a reviewed JSON asset plan containing, for each selected file, an HTTPS URL, destination filename, SHA-256 checksum, and license; then run:

```powershell
python experiments/benchmarks/prepare.py assets `
  --plan data/benchmarks/extraction/assets-plan.json `
  --output data/benchmarks/raw/document
```

Create `document-validation.json` and `document-locked-test.json` under `data/benchmarks/extraction/`. Each sample needs an ID, `kind` (`image`, `pdf`, or `docx`), repository-relative `source_path`, verified `reference`, `asset_sha256`, source license/revision, and `document_family`. Add page text and canonical structure annotations when their dependent metrics are claimed.

### Audio and video

Use [Open ASR Leaderboard datasets/methodology](https://github.com/huggingface/open_asr_leaderboard) as public screening context, then prepare EduMind-specific educational recordings with verified transcripts and timestamps. Store manifests as:

```text
data/benchmarks/extraction/audio-validation.json
data/benchmarks/extraction/audio-locked-test.json
data/benchmarks/extraction/video-validation.json
data/benchmarks/extraction/video-locked-test.json
```

Every asset row must contain a checksum, license, source revision, document family, and reference transcript. Timestamp metrics require reference segments. Video visual-text metrics additionally require verified visible text and timestamps.

### Normalization

Normalization needs at least 200 deterministic corruption/preservation cases split by document family. Each row contains `observed`, `reference`, provenance, and split identity. No model download is required.

## 5. Vector database servers

Prepare and digest-lock the four server images:

```powershell
python experiments/benchmarks/prepare.py vectordb
```

The compared servers are [Chroma](https://docs.trychroma.com/guides/deploy/docker), [Qdrant](https://qdrant.tech/documentation/installation/), [Weaviate](https://docs.weaviate.io/deploy/installation-guides/docker-installation), and [PostgreSQL with pgvector](https://github.com/pgvector/pgvector). The command writes `data/benchmarks/models/vectordb.json` and a digest-based Compose environment.

Start all benchmark servers explicitly:

```powershell
docker compose -f experiments/benchmarks/vectordb/compose.yml up -d
docker compose -f experiments/benchmarks/vectordb/compose.yml ps
```

Start only provisional Chroma for the application:

```powershell
docker compose -f infrastructure/chroma.yml up -d
```

Servers bind to loopback ports. Keep benchmark server data separate from application data.

## 6. Run the application

Copy `.env.example` to `.env` only when overrides are necessary. Then:

```powershell
docker compose -f infrastructure/chroma.yml up -d
streamlit run src/edumind/ui/streamlit_app.py
```

The provisional path is Docling Standard (RapidOCR, PDF-aware regions for PDFs, full-page OCR for images, TableFormer fast, formulas off), Whisper `small.en`, token 256/32, MiniLM, dense Chroma retrieval, and direct Hugging Face Qwen3 1.7B on CPU.

## 7. Run benchmarks

Start local MLflow in a separate terminal:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Representative commands:

```powershell
python experiments/benchmarks/extraction/document/run.py --profile standard --phase configuration
python experiments/benchmarks/extraction/audio/run.py --profile standard --device cuda
python experiments/benchmarks/extraction/video/run.py --profile standard --document-summary DOCUMENT_SUMMARY --audio-summary AUDIO_SUMMARY
python experiments/benchmarks/extraction/normalization/run.py --profile standard

python experiments/benchmarks/rag/chunking_embedding/run.py --profile standard
python experiments/benchmarks/rag/retrieval/run.py --profile standard --embedding-summary EMBEDDING_SUMMARY
python experiments/benchmarks/rag/generation/run.py --profile standard --device cuda
python experiments/benchmarks/rag/final/run.py --profile standard --retrieval-summary RETRIEVAL_SUMMARY --generation-summary GENERATION_SUMMARY --device cuda

python experiments/benchmarks/vectordb/run.py --profile smoke
python experiments/benchmarks/vectordb/run.py --profile standard
```

Use `--no-mlflow` only for debugging. Use `--shortlist summary.json` for full profiles. A full run never silently expands to all candidates.

## 8. Verification checklist

Before claiming an authoritative result:

- `prepare.py --list` names only included models, the HHEM diagnostic, and documented Docling subcomponents.
- `selected.json` exists and every recorded directory exists.
- standard/full manifests pass checksum, provenance, evidence-offset, and split-leakage validation.
- all candidate runs retain per-sample results and exact revisions.
- generation candidates share the same explicit device.
- vector servers report healthy and use actual ANN indexes.
- failed candidates remain visible; smoke output is never treated as comparative evidence.
- final human review is imported before selecting the locked-test system.

If a model is missing, rerun the matching preparation target. If a pinned revision no longer resolves, stop and review the selection package; do not replace it with the repository's current head.
