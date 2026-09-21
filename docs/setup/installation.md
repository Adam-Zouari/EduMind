# EduMind installation and benchmark preparation

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Run only the application](running.md) ·
[Benchmark overview](../benchmarks/overview.md)

This guide prepares the provisional application and every approved benchmark candidate. The authoritative shortlist is described in [model selection](../benchmarks/model-selection.md); exact executable revisions are read from [selection evidence](../../experiments/benchmarks/selection_evidence.csv). Excluded rows are historical evidence and are never downloaded.

## 1. System requirements

Install these programs before creating the Python environment:

| Program | Required for | Download |
|---|---|---|
| Python 3.12.x (64-bit) | application and benchmarks; this is the project's supported Python minor release | [python.org](https://www.python.org/downloads/windows/) |
| Git | editable installation and provenance | [git-scm.com](https://git-scm.com/download/win) |
| Docker Desktop with Compose | Chroma, vector-server benchmarks, and official OmniDocBench table/formula scoring | [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) |
| FFmpeg | audio/video decoding and video keyframes | [FFmpeg download page](https://ffmpeg.org/download.html) |
| Tesseract 5 with English data | one Docling Standard configuration | [Tesseract installation](https://tesseract-ocr.github.io/tessdoc/Installation.html) and [Windows builds](https://tesseract-ocr.github.io/tessdoc/Downloads.html) |
| NVIDIA driver | Required for authoritative ASR and RAG model comparisons; optional for the CPU application and smoke/debug runs | [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) |

The application does not require a separate model-serving service. Generation uses exact local Hugging Face snapshots directly. Docker is never started by an import, preparation command, or application startup.

Check the system tools:

```powershell
py -3.12 --version
git --version
docker version
docker compose version
ffmpeg -version
tesseract --version
tesseract --list-langs
# CUDA profiles only:
nvidia-smi
```

`eng` must appear in the Tesseract language list. CUDA is optional for the
provisional CPU application and for smoke/debug checks. Authoritative development,
validation, and locked ASR, embedding, learned-reranking, and generation runs use the
RTX 3050, batch size `1`, a stage-frozen supported 16-bit dtype, and one whole
model on the GPU. They permit no fallback, offload, automatic device splitting,
or quantization, and peak process VRAM must not exceed `3,584 MiB`.

## 2. Create an isolated Python environment

Do not install these locks into a global Python environment.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/app.lock
python -m pip install -r requirements/benchmarks.lock
python -m pip install -e . --no-deps
python -m pip check
```

On a Windows benchmark host using an NVIDIA driver compatible with CUDA 13,
replace the CPU PyTorch wheels with the matching official build after installing
the locks:

```powershell
python -m pip install --upgrade --force-reinstall --no-deps `
  torch==2.11.0+cu130 torchvision==0.26.0+cu130 torchaudio==2.11.0+cu130 `
  --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip check
```

The version numbers must stay aligned with `requirements/app.lock`. A CUDA
benchmark is valid only when the requested model remains on `cuda` and the run
records a non-zero measured VRAM peak; there is no CPU fallback.

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

This downloads the pinned GTE ModernBERT base embedding snapshot, Hugging Face
Qwen3 1.7B, Whisper `small.en`,
and only the Docling components used by the provisional application: layout,
TableFormer, and RapidOCR. Prepare the additional experiment candidates and
Docling components separately or together:

```powershell
python experiments/benchmarks/prepare.py embedding-models
python experiments/benchmarks/prepare.py rag-models
python experiments/benchmarks/prepare.py extraction-models
python experiments/benchmarks/prepare.py all-models
```

`embedding-models` prepares only the six candidates used by the
chunking--embedding matrix. `rag-models` is the aggregate RAG target and also
prepares the selected rerankers, generators, and evaluator.
The application's provisional CPU Qwen3-1.7B snapshot is separate from the
generation benchmark: it is not the benchmark control and is not part of the
three-candidate generator shortlist.
Embedding preparation excludes alternate ONNX, OpenVINO, GGUF, TensorFlow,
Flax, and TFLite exports because these benchmarks load the native Transformers
checkpoint. The exact exclusions and resulting directory checksum are recorded
in the model lock.

Downloads are resumable through Hugging Face and are placed in deterministic project directories. Preparation never substitutes a newer repository head when the pinned revision is unavailable.

Benchmark `protocol.yaml` files do not download models and do not replace the
model lock. Protocols define executable behavior; preparation resolves the
model identities from candidate registries and writes immutable revisions,
local snapshot paths, and checksums to
`data/benchmarks/models/selected.json`.

### RAG models

The RAG download contains these exact approved identities:

- Embeddings: [GTE ModernBERT base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base/tree/e7f32e3c00f91d699e8c43b53106206bcc72bb22), [Snowflake Arctic Embed M v2](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0), [F2LLM v2 0.6B](https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B), [Octen 0.6B](https://huggingface.co/Octen/Octen-Embedding-0.6B), [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), and [Nemotron Embed 1B](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16).
- Rerankers: [GTE ModernBERT control](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base), [Ettin 150M](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1), [Ettin 400M](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1), and [Ettin 1B](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1).
- Generators: [Falcon-H1-Tiny-R-90M reasoning control](https://huggingface.co/tiiuae/Falcon-H1-Tiny-R-90M/tree/7385612bf04c64405a51b29b6229d6d2ab0e72fd), [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), [Qwen3.5 0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17), and [MiniCPM5 1B](https://huggingface.co/openbmb/MiniCPM5-1B/tree/87179e5c1f455ef22e6223592d2d61351b525bfc). All four use their reasoning profile.
- Diagnostic evaluator: [HHEM](https://huggingface.co/vectara/hallucination_evaluation_model).

Authoritative generator comparisons run on CUDA with `float16`, batch size `1`,
no quantization, no automatic CPU/GPU split, temperature 0, and seed 42. The
benchmark records the whole-model device and processes repetitions sequentially.

### Migrating an existing MiniLM index

GTE produces 768-dimensional vectors, so an existing 384-dimensional MiniLM
collection cannot be reused. EduMind keeps the `edumind` collection name and
checks its embedding/chunking fingerprint at startup; it rejects the old index
instead of deleting or silently mixing vectors.

After preparing the new application models, stop the application and explicitly
remove the dedicated Chroma volume, then recreate it:

```powershell
docker compose -f infrastructure/chroma.yml down -v
docker compose -f infrastructure/chroma.yml up -d
```

This deletes the existing local Chroma data. Re-upload the source documents to
rebuild the index with GTE. No application or benchmark command performs this
destructive migration automatically.

### Extraction models

The extraction download contains:

- [Docling Standard](https://github.com/docling-project/docling/releases/tag/v2.117.0) layout, TableFormer, CodeFormula, RapidOCR, and EasyOCR components. Tesseract remains a system executable.
- [Granite Docling 258M](https://huggingface.co/ibm-granite/granite-docling-258M).
- [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) plus its Paddle layout components in a project-controlled cache.
- [Whisper small.en](https://huggingface.co/openai/whisper-small.en), [Canary 180M](https://huggingface.co/nvidia/canary-180m-flash), [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), and [MOSS Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize).

NeMo and MOSS have heavier runtime dependency trees. Install them only inside
the Python 3.12 project environment from `requirements/benchmarks.lock`; do not
combine this lock with unrelated ML environments.

### Official document-metric code

TEDS, TEDS-S, and CDM come from the
[pinned OmniDocBench evaluator](https://github.com/opendatalab/OmniDocBench/tree/193627ae9e97d89188468ed1ee3b7a856ff76044)
rather than being reimplemented by EduMind. The main EduMind environment remains Python
3.12. These three specialized metrics run in OmniDocBench's verified Python
3.10 Docker environment, which also contains its required TeX, ImageMagick, and
Ghostscript versions. They do not need to be installed on Windows.

Start Docker Desktop, then prepare the evaluator explicitly:

```powershell
python experiments/benchmarks/prepare.py evaluators
```

The command checks out source revision
`193627ae9e97d89188468ed1ee3b7a856ff76044`, pulls
`ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204`, and records the
resolved immutable image digest under
`data/benchmarks/evaluators/OmniDocBench/`. An authoritative run uses that
digest, disables container networking, and fails if the source, image, or lock
does not match instead of logging an approximation.

This evaluator never runs Docling, Granite Docling, or PaddleOCR-VL. EduMind
first produces each parser's canonical prediction and calculates the common
text/page/layout/reliability/operational metrics in Python 3.12. It then sends
only the table HTML and formula LaTeX reference/prediction pairs to the isolated
official scorer. The scoring time is evaluation overhead and is not included in
parser latency.

## 4. Datasets

Tiny committed fixtures support smoke execution. Smoke validates wiring only and cannot establish quality or speed claims.

### RAG: QASPER

The benchmark dataset guide owns the pinned QASPER revision, paper allocation,
structured-evidence requirements, and all three combined-manifest commands.
Prepare the initial paper-isolated QASPER manifests with:

```powershell
python experiments/benchmarks/prepare.py qasper
```

Continue with the [RAG dataset preparation instructions](../benchmarks/datasets.md#3-install-the-rag-datasets).

### Extraction datasets

Follow the [benchmark dataset guide](../benchmarks/datasets.md) for
the reviewed document, audio, video, and RAG source pools; exact download commands;
pinned revisions; licenses; checksums; directory layout; sample allocation; and
manifest requirements. That guide is the single source of truth for benchmark
data. The repository does not silently download or redistribute the public
corpora.

To regenerate the committed multimodal smoke files, including real synthesized
speech WAVs and deterministic silence/noise controls:

```powershell
python experiments/benchmarks/prepare.py smoke-fixtures
```

Regenerate only one fixture group when working on a single experiment:

```powershell
python experiments/benchmarks/prepare.py smoke-fixtures --modality document
python experiments/benchmarks/prepare.py smoke-fixtures --modality audio
python experiments/benchmarks/prepare.py smoke-fixtures --modality video
```

Audio and video regeneration requires an FFmpeg build containing the optional
`flite` filter. This is needed only to recreate the committed synthetic speech;
running the existing smoke benchmarks does not require `flite`.

## 5. Vector database servers

Prepare and digest-lock the four server images:

```powershell
python experiments/benchmarks/prepare.py vectordb
```

The compared servers are [Chroma](https://docs.trychroma.com/guides/deploy/docker), [Qdrant](https://qdrant.tech/documentation/installation/), [Weaviate](https://docs.weaviate.io/deploy/installation-guides/docker-installation), and [PostgreSQL with pgvector](https://github.com/pgvector/pgvector). The command writes `data/benchmarks/models/vectordb.json` and a digest-based Compose environment.

Server start/stop commands belong to the
[benchmark runbook](../benchmarks/running.md#6-run-vector-server-experiments) and
[application run guide](running.md#2-start-chroma). Servers bind to loopback
ports, and benchmark data remains separate from application data.

## 6. Next steps

Software, models, and source datasets are now prepared. Development/validation extraction
benchmarks remain unavailable until the downloaded samples have been reviewed,
annotated, checksummed, and frozen into the manifests required by the
[dataset guide](../benchmarks/datasets.md) and
[pending-data checklist](../benchmarks/pending-data-review.md).

- use [Running the application](running.md) for daily start, stop, readiness, and
  troubleshooting commands;
- use the [benchmark runbook](../benchmarks/running.md) for MLflow, experiment
  order, decision files, and every benchmark command.

The command references are kept in those pages so changing a CLI does not
require updating several copies.

## 7. Verification checklist

Before using a run as comparative evidence:

- `prepare.py --list` names only included models, the HHEM diagnostic, and documented Docling subcomponents.
- `selected.json` exists and every recorded directory exists.
- Development, validation, and locked manifests pass checksum, provenance,
  evidence-offset, and split-leakage validation.
- all candidate runs retain per-sample results and exact revisions.
- generation candidates share the same explicit device.
- vector servers report healthy and use actual ANN indexes.
- the parent MLflow run reports `benchmark_complete=1`; failed candidates and partial artifacts remain visible, and smoke output is never treated as comparative evidence.
- final human review is imported before selecting the locked-test system.

If a model is missing, rerun the matching preparation target. If a pinned revision no longer resolves, stop and review the selection package; do not replace it with the repository's current head.
