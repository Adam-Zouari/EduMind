# EduMind installation and benchmark preparation

[Project overview](../../README.md) · [Documentation map](../README.md) ·
[Run only the application](running.md) ·
[Benchmark program](../README.md#experiments)

This guide prepares the provisional application and every approved benchmark
candidate. The authoritative shortlist is described in
[model selection](../benchmarks/model-selection.md); exact executable revisions
are read from [selection evidence](../../experiments/benchmarks/selection_evidence.csv).
Benchmark model preparation downloads only included rows. Pinned application
controls are separate: `app-models` also prepares the documented Qwen3-1.7B
control from its historical revision without changing its excluded benchmark
decision or adding it to the benchmark roster.

Run the commands below from the repository root. Benchmark commands use Python
module names (`python -m experiments.benchmarks...`).

For application setup, complete the system and environment steps, prepare
`app-models`, and follow [Running the application](running.md). For benchmarks,
continue with the relevant model, evaluator, dataset, and server preparation
sections, then use the [benchmark runbook](../benchmarks/running.md).

## 1. System requirements

Install these programs before creating the Python environment:

| Program | Required for | Download |
|---|---|---|
| Python 3.12.x (64-bit) | application and benchmarks; this is the project's supported Python minor release | [python.org](https://www.python.org/downloads/windows/) |
| Git | editable installation and provenance | [git-scm.com](https://git-scm.com/download/win) |
| Docker Desktop with Compose | Chroma, vector-server benchmarks, and official OmniDocBench table/formula scoring | [Docker Desktop](https://docs.docker.com/desktop/setup/install/windows-install/) |
| FFmpeg | audio/video decoding and video keyframes | [FFmpeg download page](https://ffmpeg.org/download.html) |
| Tesseract 5 with English data | one Docling Standard configuration | [Tesseract installation](https://tesseract-ocr.github.io/tessdoc/Installation.html) and [Windows builds](https://tesseract-ocr.github.io/tessdoc/Downloads.html) |
| NVIDIA driver | Required for authoritative GPU-backed extraction and RAG comparisons; optional for the CPU application and CPU smoke/debug runs | [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) |

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
or quantization. Extraction (document, ASR, video visual processing, and frozen
video ASR) uses the isolated GPU without an artificial VRAM cap in any profile,
including preflight. Actual OOM, offloading, and unavailable required telemetry
remain failures. Embedding, learned-reranking, and generation retain their
`3,584 MiB` limit. Measurement follows the
[Peak Device VRAM contract](../benchmarks/metrics.md#peak-device-vram); execution
requirements and background-workload checks are in the
[methodology](../benchmarks/methodology.md) and [runbook](../benchmarks/running.md).

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
CPU VRAM is `null/inapplicable`; missing required CUDA telemetry is
`null/unavailable`. See [Peak Device VRAM](../benchmarks/metrics.md#peak-device-vram)
for these distinct cases.
Missing required CUDA telemetry still blocks hardware qualification.

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
python -m pip install -r requirements/app.lock -r requirements/benchmarks.lock
python -c "import torch, onnxruntime as ort, paddle; print(ort.get_available_providers()); print(paddle.__version__, paddle.is_compiled_with_cuda())"
python -m pip check
```

If a CUDA 12 Paddle wheel was previously installed, uninstall it and its
`nvidia-*-cu12` packages before installing the benchmark lock; CUDA 11 and
CUDA 12 NVIDIA packages share Python directories. This project's ONNX Runtime
GPU 1.30.0 uses the pinned CUDA 13 PyTorch libraries. Do not remove libraries
needed by unrelated packages in another environment.

A listed CUDA provider or CUDA-enabled Paddle build is an installation check,
not proof of candidate eligibility. Real smoke and preflight also verify model
placement and inference. RapidOCR verifies the actual device assignments in
ONNX Runtime's optimized graph during model initialization. CPU operations are
allowed only when their dataflow proves shape bookkeeping, not learned model
computation. For example, the prepared recognition graph uses CPU `Slice` and
`Concat` nodes to construct a GPU `Reshape` argument from tensor dimensions.
Operator names alone do not establish this exception. Unknown assignments,
CPU feature/weight computation, and automatic CPU retry after a CUDA error are
rejected. The placement artifact records the graph checksum and permitted CPU
shape nodes; no extra inference or warmup is required for verification.
Tesseract and native DOCX parsing remain explicit CPU-only routes.

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

The default generated runtime lock is:

```text
data/benchmarks/models/selected.json
```

That file records repositories, revisions, local paths, checksums, composite
submodels, and Docling component/system identities. Benchmark loading checks
only the models requested by that stage against selection evidence and actual
local bytes. Application loading verifies integrity without importing benchmark
eligibility. Do not hand-edit paths, revisions, or checksums to bypass checks.

### Verified snapshots and retained locks

Fresh Hugging Face preparation publishes a completed directory at
`data/benchmarks/downloads/models/<repository-with-double-dashes>/<commit>/`.
Before publication it verifies the resolved immutable commit, returned snapshot
location, exact file inventory, file sizes, Git blob hashes or LFS SHA256 values,
and the selected export exclusions. A sibling `<commit>.complete.json` records
that inventory, policy, checksum, and measured size. Reusing a completed snapshot
offline requires rechecking its receipt and bytes; an unavailable new revision
cannot relabel an old download. Mixed revisions, stale excluded exports, and
unverified partial directories fail closed. Existing downloads are not erased
as an implicit repair; interrupted work remains available for explicit recovery.
If publication was interrupted after the final directory appeared but before
its completion receipt was written, that directory is still unverified. Retain
it and use a separate preparation cache; do not relabel it as completed.

Lock schemas 1, 2, and 3 are recognized, but requested entries must have nonempty
paths to actual directories and valid integrity checksums. Primary snapshots,
submodels, auxiliary caches, and prepared Tesseract files are verified when
requested. Optional size metadata does not substitute for integrity. A lock
merge checks the structure of untouched entries without reading unrelated
snapshots and refuses to publish a schema-3 lock containing checksum-less legacy
entries. Successful entries can precede a later preparation failure; a partially
prepared environment is not complete setup.

The 2026-11-13 repair check verified all **17 existing local lock entries** against
their recorded bytes, including auxiliary/system files, without changing the
original lock or model cache. Stage-scoped loading of the existing GTE snapshot
also passed. That GTE snapshot does **not** need repreparation or relocation just
because fresh downloads now use revision-specific directories. This was an
integrity check, not model inference, CPU/CUDA qualification, or comparative
evidence; Docling component-pin migration and evaluator readiness are separate.

To recheck just that existing GTE entry without downloads, inference, or writes:

```powershell
python -c "from pathlib import Path; from experiments.benchmarks.preparation.models import load_selected_model_lock; lock = load_selected_model_lock(Path('data/benchmarks/models/selected.json'), candidates=['Alibaba-NLP/gte-modernbert-base']); print(lock['Alibaba-NLP/gte-modernbert-base']['model_path'])"
```

### Snapshot size metadata

Preparation also measures `snapshot_size_bytes` for each entry it writes. This
integer is the sum of logical file lengths under that entry's `model_path`,
excluding `.cache` download metadata and `__pycache__` trees. It includes local
weights, configuration, tokenizer files, and other retained snapshot files; it
is not just parameter count multiplied by weight precision. Display tools may
convert bytes to GB (1,000,000,000 bytes) or GiB (1,073,741,824 bytes), but the
stored value stays in bytes.

For `docling-standard`, the path is a prepared component bundle, so its size
includes files inside that bundle. Separately located Paddle auxiliary caches,
composite submodel directories, Tesseract executables/language data, Python
packages, and model-serving memory are not added to a primary snapshot's size.
This is local snapshot inventory, not total pipeline deployment size or VRAM.

A retained GTE entry, for example, includes these fields alongside its existing
revision and cache checksum:

```json
{
  "model": "Alibaba-NLP/gte-modernbert-base",
  "revision": "e7f32e3c00f91d699e8c43b53106206bcc72bb22",
  "model_path": "data/benchmarks/downloads/models/Alibaba-NLP--gte-modernbert-base",
  "snapshot_size_bytes": 599710360
}
```

This is an excerpt, not a replacement lock file. Sizes describe the files
actually retained locally and can differ across preparation exclusion policies.
Old locks without this optional metadata remain readable. Repreparing an entry
automatically measures its current size; no separate size file or download
command is needed. Adding size metadata does not change model-file checksums,
but it changes the lock file's own checksum. Existing run artifacts remain
unchanged and retain their original provenance. Snapshot size is not required
as a per-run benchmark metric.

### Preparation commands

Inspect the complete download plan without network access:

```powershell
python -m experiments.benchmarks.prepare --list
python -m experiments.benchmarks.prepare app-models --dry-run
python -m experiments.benchmarks.prepare all-models --dry-run
```

Every preparation target accepts `--dry-run`. It validates required option
combinations and prints a plan before calling any preparation helper: no network,
Docker calls, model preparation, fixture regeneration, or filesystem writes.
Planning does not open or validate the referenced dataset plan/manifests and
does not establish environment or data readiness.

Prepare only the provisional application controls:

```powershell
python -m experiments.benchmarks.prepare app-models
```

This downloads the pinned GTE ModernBERT base embedding snapshot, Hugging Face
Qwen3 1.7B, Whisper `small.en`,
and only the Docling components used by the provisional application: layout,
TableFormer, and RapidOCR. Prepare the additional experiment candidates and
Docling components separately or together:

```powershell
python -m experiments.benchmarks.prepare embedding-models
python -m experiments.benchmarks.prepare rag-models
python -m experiments.benchmarks.prepare extraction-models
python -m experiments.benchmarks.prepare all-models
```

`embedding-models` prepares only the six candidates used by the
chunking–embedding matrix. `rag-models` is the aggregate target for the approved
benchmark embeddings, rerankers, and generators. Judge setup follows the later
calibrated evaluation contract: a hosted judge needs credentials rather than a
local snapshot, while a self-hosted judge needs an explicit pinned preparation
step. No unselected judge is downloaded by these commands.
The application's provisional CPU Qwen3-1.7B snapshot is pinned independently of
benchmark eligibility; later benchmark preparation preserves its lock entry.
It is separate from the
generation benchmark: it is neither the reasoning control nor one of its three
challengers. The declared generator roster has four model identities; direct
and reasoning modes produce separate applicable configurations.
Embedding and reranker preparation exclude alternate ONNX, OpenVINO, GGUF, TensorFlow,
Flax, and TFLite exports because these benchmarks load the native Transformers
checkpoint. The exact exclusions and resulting directory checksum are recorded
in the model lock.

Transfers can resume through the project-controlled Hugging Face cache;
completed snapshots still require the inventory checks above. Preparation never
substitutes a newer repository head when the pinned revision is unavailable.

Benchmark `protocol.yaml` files do not download models and do not replace the
model lock. Protocols define executable behavior and candidate rosters where
needed. Preparation uses included entries in `selection_evidence.csv` and writes
immutable revisions, local snapshot paths, and checksums to
`data/benchmarks/models/selected.json`.
Each successfully written entry also receives its measured `snapshot_size_bytes`.

### Legacy-lock and component-pin migration

Keep existing locks and downloads as evidence. If an untouched legacy entry
lacks integrity fields, prepare a separate lock rather than silently upgrading
it. A legacy `docling-standard` entry with only package version `2.117.0` and a
directory checksum is not an immutable component-model pin; benchmark loading
requires the component inventory described below. Do not copy current checksums
onto old entries or replace historical artifacts to make validation pass.

Only if migration is needed, and after coordinating the required downloads,
prepare a separate generated lock:

```powershell
python -m experiments.benchmarks.prepare app-models --output data/benchmarks/models/selected-next.json
python -m experiments.benchmarks.prepare all-models --output data/benchmarks/models/selected-next.json
```

Review and verify the new lock before explicitly configuring its consumers;
application `models.lock_path` and supported benchmark `--model-lock` overrides
are deliberate choices. These commands do not promote benchmark winners or
require deleting/repreparing an already verified GTE snapshot for current use.
An interrupted unpublished target or conflicting snapshot policy may need a
separate `cache_directory` through the preparation Python API; retain the old
directory instead of deleting it as an automatic retry.

### RAG models

The RAG download contains these exact approved identities:

- Embeddings: [GTE ModernBERT base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base/tree/e7f32e3c00f91d699e8c43b53106206bcc72bb22), [Snowflake Arctic Embed M v2](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0), [F2LLM v2 0.6B](https://huggingface.co/codefuse-ai/F2LLM-v2-0.6B), [Octen 0.6B](https://huggingface.co/Octen/Octen-Embedding-0.6B), [Qwen3 Embedding 0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), and [Nemotron Embed 1B](https://huggingface.co/nvidia/Nemotron-3-Embed-1B-BF16).
- Rerankers: [GTE ModernBERT control](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base), [Ettin 150M](https://huggingface.co/cross-encoder/ettin-reranker-150m-v1), [Ettin 400M](https://huggingface.co/cross-encoder/ettin-reranker-400m-v1), and [Ettin 1B](https://huggingface.co/cross-encoder/ettin-reranker-1b-v1).
- Generators: [Falcon-H1-Tiny-R-90M reasoning control](https://huggingface.co/tiiuae/Falcon-H1-Tiny-R-90M/tree/7385612bf04c64405a51b29b6229d6d2ab0e72fd), [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/tree/c1899de289a04d12100db370d81485cdf75e47ca), [Qwen3.5 0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17), and [MiniCPM5 1B](https://huggingface.co/openbmb/MiniCPM5-1B/tree/87179e5c1f455ef22e6223592d2d61351b525bfc). Falcon is reasoning-only; the other three run their official direct and reasoning modes.

Authoritative generator comparisons run on CUDA with `float16`, batch size `1`,
no quantization, and no automatic CPU/GPU split. Each model-mode configuration
uses its frozen official decoder and aligned measured seeds `42`, `43`, and `44`.
The benchmark records the whole-model device and processes repetitions
sequentially.

Semantic generation metrics will use one protocol-pinned LLM judge. No judge
identity is currently approved. Its identity, version, rubrics, and
human-calibration artifact must exist
before an authoritative generation run. Hosted credentials, if required by the
selected judge, are supplied through the environment and are never written to a
protocol or MLflow artifact.
Judge preparation includes the response rubric's separate full-context and
cited-passage support labels, correctness/relevancy labels, and the separate
semantic-pair rubric. Credentials alone do not complete this setup: the
[judge setup instructions](../benchmarks/running.md#generation-execution-and-judge-setup) distinguish
the implemented replay, automatic command-transport, and saved-output recovery
paths from the separate judge-selection and calibration requirements.

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

Docling's `runtime_version` is separate from `component_models`. The evidence-backed
[component pins](../../experiments/benchmarks/preparation/docling_pins.py) record
immutable Hub commits for layout, TableFormer, and CodeFormula, plus measured
RapidOCR/EasyOCR SHA256 inventories. Their evidence comes from the retained local
download metadata/files, not invented SHAs or a quality assessment. Fresh
preparation explicitly requests those commits and verifies OCR file checksums
instead of relying on moving model heads. New bundles have separate paths;
augmentation preserves verified components and system provenance, and preparing
an application subset does not discard benchmark components.

Tesseract is recorded separately by exact executable path/hash and English
traineddata path/hash. These identities are verified rather than inferred from
`PATH` or `TESSDATA_PREFIX`; the selected runtime must use the verified executable
and language directory. A package version, language-list output, or a successful
download alone is not a real-runtime smoke check.

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

Inspect the plan first; then, when source/image downloads are coordinated and
Docker's Linux engine is available, prepare the evaluator explicitly:

```powershell
python -m experiments.benchmarks.prepare evaluators --dry-run
docker version
python -m experiments.benchmarks.prepare evaluators
```

The command checks out source revision
`193627ae9e97d89188468ed1ee3b7a856ff76044`, pulls
`ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204`, and records the
resolved immutable image digest under
`data/benchmarks/evaluators/OmniDocBench/`. Before source publication it verifies
a clean checkout of that commit and writes `.edumind-source.json`, covering
retained evaluator code and dependency files. The revision marker alone is not
proof of source integrity. Scoring rehashes actual mounted source content and
uses the immutable image digest with container networking disabled. Changed,
deleted, added, or unverifiable source files fail before scoring rather than
becoming candidate quality zeroes.

A legacy marker-only evaluator, changed source tree, or retained `.partial`
checkout is not automatically trusted, deleted, or overwritten. Preserve/archive
that directory explicitly before preparing a clean owned destination; do not
hand-create a source manifest from an unverified checkout. Once prepared, inspect
the verified source/image identity without starting Docker:

```powershell
python -c "import json; from experiments.benchmarks.extraction.document.official_metrics import official_evaluator_identity; print(json.dumps(official_evaluator_identity(), indent=2))"
```

The identity includes source revision/content hash, immutable image digest, and
image-lock hash. Document validation seals bind this identity; changing it
requires a new matching full-validation report before execution. Retain it with
scoring provenance and saved predictions. This read-only check does not establish
that the Docker daemon/image can execute or that nonempty official scoring passes.

This evaluator never runs Docling, Granite Docling, or PaddleOCR-VL. EduMind
first produces each parser's canonical prediction and calculates the common
text/page/layout/reliability/operational metrics in Python 3.12. It then sends
only the table HTML and formula LaTeX reference/prediction pairs to the isolated
official scorer. The scoring time is evaluation overhead and is not included in
parser latency.

Preparing or verifying evaluator identity does not establish scorer conformance.
The wrapper's symmetric normalization and count-based ExpRate behavior have
fixture coverage and live nonempty TEDS/TEDS-S/CDM acceptance on the pinned
Docker image. Header perturbations and formula identity, mismatch, missing and
invalid predictions were verified; this is scorer conformance, not a model or
dataset quality result. See the
[document implementation boundary](../benchmarks/extraction/document/methodology.md#evaluation-setup).
Strict ExpRate requires raw matching counts, not equality of the rounded CDM
display score to one.

## 4. Datasets

Tiny committed fixtures support smoke execution. Smoke validates wiring only and cannot establish quality or speed claims.

### RAG: QASPER

The benchmark dataset guide owns the pinned QASPER revision, paper allocation,
structured-evidence requirements, and all three combined-manifest commands.
Prepare the initial paper-isolated QASPER manifests with:

```powershell
python -m experiments.benchmarks.prepare qasper
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
python -m experiments.benchmarks.prepare smoke-fixtures
```

Regenerate only one fixture group when working on a single experiment:

```powershell
python -m experiments.benchmarks.prepare smoke-fixtures --modality document
python -m experiments.benchmarks.prepare smoke-fixtures --modality audio
python -m experiments.benchmarks.prepare smoke-fixtures --modality video
```

Audio and video regeneration requires an FFmpeg build containing the optional
`flite` filter. This is needed only to recreate the committed synthetic speech;
running the existing smoke benchmarks does not require `flite`.

Prepared assets still need reference and split review before benchmark execution.
The [shared data-validation guide](../benchmarks/data-validation.md) defines the
standalone checks and sealed-report verification. All eight benchmarks implement
this interface: validate prepared inputs before starting their profiles. Vector
Database verifies the actual vector/query/corpus inventory; Final RAG validates
its inputs separately from component winner decisions and system qualification.

## 5. Vector databases

Prepare and digest-lock the four server images:

```powershell
python -m experiments.benchmarks.prepare vectordb
```

The compared servers are [Chroma](https://docs.trychroma.com/guides/deploy/docker), [Qdrant](https://qdrant.tech/documentation/installation/), [Weaviate](https://docs.weaviate.io/deploy/installation-guides/docker-installation), and [PostgreSQL with pgvector](https://github.com/pgvector/pgvector). The command writes `data/benchmarks/models/vectordb.json` and a digest-based Compose environment.

With `--output PATH`, the image lock can use a relative or absolute custom file;
the Compose environment remains at
`experiments/benchmarks/vectordb/.env` under the project root. Directory/file
conflicts, a lock/environment collision, or escaped Compose ownership fail
before Docker calls or publication. Do not derive the environment location from
the custom lock path. Image identities must be full immutable SHA256 digests;
preparation pulls images but does not start services.

Server start/stop commands belong to the
[benchmark runbook](../benchmarks/running.md#9-vector-databases) and
[application run guide](running.md#2-start-chroma). Servers bind to loopback
ports, and benchmark data remains separate from application data.

## 6. Next steps

After the relevant preparation steps, software, model snapshots, and raw source
pools are available. Authoritative development, validation, and locked runs
still require reviewed, annotated, checksummed manifests and all applicable
qualification/evaluator dependencies described in the
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

- `python -m experiments.benchmarks.prepare --list` names only included application and benchmark models plus documented Docling subcomponents.
- The requested entries in `selected.json` pass structural and actual-byte
  integrity checks; directory existence or optional size metadata alone is not enough.
- Required Docling components have independent pins, and any required official
  evaluator has verified source-content/image identity and real scorer conformance.
- Development, validation, and locked manifests pass checksum, provenance,
  evidence-offset, and split-leakage validation.
- all candidate runs retain per-sample results and exact revisions.
- generation candidates share the same explicit device.
- vector servers report healthy, preserve the required data, and verify their
  configured indexes/planning rules. Recorded native exact-search paths are
  allowed; every query need not traverse an ANN index.
- the parent MLflow run reports `benchmark_complete=1`; failed candidates and partial artifacts remain visible, and smoke output is never treated as comparative evidence.
- component winners and the complete system are frozen before candidates are evaluated on their applicable locked data; offline annotation review remains separate from candidate selection.

If a model is missing, rerun the matching preparation target. If a pinned revision no longer resolves, stop and review the selection package; do not replace it with the repository's current head.
