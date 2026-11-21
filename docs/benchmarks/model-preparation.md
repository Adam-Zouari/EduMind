# Prepare benchmark models

[Install the benchmark environment](../setup/installation.md) · [Model selection](model-selection.md) ·
[Benchmark runbook](running.md)

Run these commands from the repository root in the activated benchmark
environment. They prepare pinned candidates for experiments; they do not run
inference, qualify a model on the GPU, or select a winner. To run the study
assistant instead, follow [app model preparation](../setup/model-preparation.md).

## Preparation commands

Choose the smallest target that covers the comparison:

| Prepare | Command |
|---|---|
| Only the six benchmark embeddings | `python -m benchmarks.prepare embedding-models` |
| All approved benchmark embeddings, rerankers and generators | `python -m benchmarks.prepare rag-models` |
| Extraction candidates and benchmark Docling components | `python -m benchmarks.prepare extraction-models` |
| All benchmark model groups | `python -m benchmarks.prepare all-models` |

Preview a target with `--dry-run`. To prepare just one included candidate,
pass its exact ID with `--candidate`; repeat the option to select more than one
candidate in the same target. For example:

```powershell
python -m benchmarks.prepare embedding-models --candidate Alibaba-NLP/gte-modernbert-base --dry-run
```

Remove `--dry-run` to download and verify the files. Use
`python -m benchmarks.prepare --list` to inspect the benchmark
catalog. Neither listing nor a dry run verifies local files.

No unselected judge is downloaded. A hosted judge needs credentials rather
than a local snapshot; a self-hosted judge needs an explicitly pinned model.
Embedding and reranker preparation exclude alternate ONNX, OpenVINO, GGUF, TensorFlow,
Flax, and TFLite exports because these benchmarks load the native Transformers
checkpoint. The exact exclusions and resulting directory checksum are recorded
in the model lock.

Transfers can resume through the project-controlled Hugging Face cache;
completed snapshots still require the [inventory checks below](#verified-snapshots-and-retained-locks). Preparation never
substitutes a newer repository head when the pinned revision is unavailable.

Benchmark `protocol.yaml` files do not download models and do not replace the
model lock. Protocols define executable behavior and candidate rosters where
needed. Preparation uses included entries in `selection_evidence.csv` and writes
immutable revisions, local snapshot paths, and checksums to
`data/benchmarks/models/selected.json`.
Each successfully written entry also receives its measured `snapshot_size_bytes`.

## Model storage and immutable lock

All prepared weights are stored inside:

```text
data/benchmarks/downloads/models/
```

Hugging Face transfer metadata and cache files are kept under
`data/benchmarks/downloads/huggingface/`. Benchmark model preparation also
verifies or prepares the shared pinned tokenizer under `data/models/tiktoken/`;
no app model download is required. To reuse an existing tokenizer, copy the
contents of the old `data/benchmarks/downloads/tiktoken/` into the empty new
location before running model preparation. The pinned checksum is still checked.

The default generated runtime lock is:

```text
data/benchmarks/models/selected.json
```

That file records repositories, revisions, local paths, checksums, composite
submodels, and Docling component/system identities. Benchmark loading checks
only the models requested by that stage against selection evidence and actual
local bytes. An old mixed lock remains readable for benchmark models; app
preparation writes a separate lock. Do not hand-edit paths, revisions, or
checksums to bypass checks.

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
python -c "from pathlib import Path; from benchmarks.preparation.models import load_selected_model_lock; lock = load_selected_model_lock(Path('data/benchmarks/models/selected.json'), candidates=['Alibaba-NLP/gte-modernbert-base']); print(lock['Alibaba-NLP/gte-modernbert-base']['model_path'])"
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

## Legacy-lock and component-pin migration

Keep existing locks and downloads as evidence. If an untouched legacy entry
lacks integrity fields, prepare a separate lock rather than silently upgrading
it. A legacy `docling-standard` entry with only package version `2.117.0` and a
directory checksum is not an immutable component-model pin; benchmark loading
requires the component inventory described below. Do not copy current checksums
onto old entries or replace historical artifacts to make validation pass.

Only if migration is needed, and after coordinating the required downloads,
prepare a separate generated benchmark lock:

```powershell
python -m benchmarks.prepare all-models --output data/benchmarks/models/selected-next.json
```

Review and verify the new lock before passing it to a benchmark that supports
the `--model-lock` override. This command does not promote benchmark winners or
require deleting/repreparing an already verified GTE snapshot for current use.
An interrupted unpublished target or conflicting snapshot policy may need a
separate `cache_directory` through the preparation Python API; retain the old
directory instead of deleting it as an automatic retry.

## RAG models

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
[judge setup instructions](running.md#generation-execution-and-judge-setup) distinguish
the implemented replay, automatic command-transport, and saved-output recovery
paths from the separate judge-selection and calibration requirements.

## Extraction models

The extraction download contains:

- [Docling Standard](https://github.com/docling-project/docling/releases/tag/v2.117.0) layout, TableFormer, CodeFormula, RapidOCR, and EasyOCR components. Tesseract remains a system executable.
- [Granite Docling 258M](https://huggingface.co/ibm-granite/granite-docling-258M).
- [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) plus its Paddle layout components in a project-controlled cache.
- [Whisper small.en](https://huggingface.co/openai/whisper-small.en), [Canary 180M](https://huggingface.co/nvidia/canary-180m-flash), [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), and [MOSS Transcribe-Diarize](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize).

Docling's `runtime_version` is separate from `component_models`. The evidence-backed
[shared component pins](../../src/edumind/common/docling_pins.py) and
[benchmark-only component pins](../../benchmarks/preparation/docling_pins.py) record
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

If a model is missing, rerun the matching preparation target. If a pinned
revision no longer resolves, review the [selection evidence](../../benchmarks/selection_evidence.csv);
do not substitute the repository's current head.
