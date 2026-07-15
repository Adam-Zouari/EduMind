# EduMind installation and experiment guide

This is the complete preparation checklist for the provisional application and every benchmark. Run commands from the repository root. A smoke run proves that a real path executes; only standard/full runs on frozen data can support comparisons.

## 1. What is automatic and what is not

Preparation commands download model weights, QASPER, and pinned Docker images and then write machine-local lock files. They do not choose winners, edit `config/base.yaml`, fabricate extraction references, or promote a benchmark result into the application.

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

## 2. Required system software

Install these programs before the Python dependencies:

- Python 3.10 or newer. Use one fresh virtual environment for this repository.
- Git.
- Docker Desktop with Docker Compose v2. It is required for Chroma production and all four vector-server benchmarks.
- Ollama. Keep its local service available at `http://127.0.0.1:11434` while preparing or evaluating generation models.
- Tesseract OCR 5 with the English language data and `tesseract` on `PATH`.
- FFmpeg with `ffmpeg` on `PATH` for audio/video handling.
- An NVIDIA driver is optional. CPU execution remains valid, but GPU candidates and resource measurements need a working CUDA-compatible installation.

Check the executables:

```bash
python --version
git --version
docker version
docker compose version
ollama --version
tesseract --version
ffmpeg -version
```

On Windows, start Docker Desktop and Ollama before commands that contact them. The committed audio/video smoke fixtures are ready to use; only regenerating them uses Windows `System.Speech`.

## 3. Create the isolated Python environment

Do not install these locks into a global Python environment. In Git Bash:

```bash
python -m venv venv
source venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/app.lock
python -m pip install -r requirements/benchmarks.lock
python -m pip install -e . --no-deps
```

In PowerShell, activation is:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

`requirements/benchmarks.lock` includes `app.lock`, so installing it again is safe. The separate commands make the production-only dependency set visible. The editable install is performed once; later source edits do not require rebuilding.

Copy `.env.example` to `.env` only when you need overrides. The normal defaults already point to `config/base.yaml`, Chroma on port 8001, Ollama on port 11434, and local MLflow SQLite.

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

It resolves immutable revisions, downloads them through the Hugging Face cache, and writes `data/benchmarks/models/huggingface.json` for:

- `sentence-transformers/all-MiniLM-L6-v2`
- `BAAI/bge-base-en-v1.5`
- `nomic-ai/nomic-embed-text-v1.5`
- `Qwen/Qwen3-Embedding-0.6B`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `Qwen/Qwen3-Reranker-0.6B`
- `cross-encoder/nli-deberta-v3-base`

The first four are embedding candidates, the next two are rerankers, and the last model supplies the pinned local NLI diagnostic. The weights remain in the normal Hugging Face cache; the JSON lock records exactly which revisions were used.

### Ollama generation models

Run this once Ollama is running:

```bash
python experiments/benchmarks/prepare.py ollama-models
```

It pulls the following base model tags sequentially and records their installed digests in `data/benchmarks/models/ollama.json`:

- `qwen3:1.7b`
- `qwen3.5:4b-q4_K_M`
- `qwen3.5:9b-q4_K_M`
- `gemma3:4b`
- `gemma3:12b`
- `ministral-3:8b-instruct-2512-q4_K_M`
- `gpt-oss:20b`

The benchmark creates ten profiles from those seven tags: Qwen 3.5 4B/9B are each tested with direct and thinking modes, and GPT-OSS 20B is tested with low and medium reasoning. Do not pull separate model names for those profiles.

Verify installed tags with:

```bash
ollama list
```

### Extraction/OCR/ASR models

Run:

```bash
python experiments/benchmarks/prepare.py extraction-models
```

It prepares and locks:

- OpenAI Whisper `small.en`;
- faster-whisper `tiny.en`, `base.en`, `small.en` and large-v3-turbo weights used by the int8/float16 profiles;
- PaddleOCR PP-OCRv5 English mobile detector/recognizer;
- PaddleOCR PP-OCRv5 server detector/recognizer;
- docTR `fast_base` detector with PARSeq recognition.

OpenAI Whisper, faster-whisper, and docTR assets are placed under `data/benchmarks/downloads/models/` where configured. PaddleOCR uses its Windows user cache under `~/.paddlex/official_models`. Tesseract has no downloaded model lock because it is a separately installed system engine. PDF/DOCX candidates such as pypdf, pdfplumber, Docling, Mammoth, and Unstructured are Python packages installed from `requirements/benchmarks.lock`.

If a preparation command fails, do not manually create a lock file. Fix the missing tool/network/dependency and rerun it; an incomplete candidate must remain a visible failure rather than being silently skipped.

## 6. QASPER data for RAG experiments

Prepare the pinned `allenai/qasper` revision:

```bash
python experiments/benchmarks/prepare.py qasper
```

The command writes:

```text
data/benchmarks/rag/qasper-dev.json          100 papers for standard component selection
data/benchmarks/rag/qasper-validation.json    40 papers for full/finalist validation
data/benchmarks/rag/qasper-locked-test.json   40 papers for the one-time final test
```

The split unit is a paper. The preparation code verifies paper isolation, normalized evidence offsets, checksums, source revision, preprocessing version, and seed 42. Do not inspect or tune against the locked-test answers. If any evidence offset cannot be validated, preparation fails instead of dropping the evidence.

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
- annotations needed by the modality, such as page text/order and page labels, word/segment timestamps, clip duration, or visible video text.

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

Tables and formulas may appear only as flattened ordinary text. There is no web, structured-table, formula, or dedicated form benchmark in this project.

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
| Chroma | `chromadb/chroma:1.5.9` | 8001 |
| Qdrant | `qdrant/qdrant:v1.17.0` | 6333 REST, 6334 gRPC |
| Weaviate | `cr.weaviate.io/semitechnologies/weaviate:1.38.2` | 8080 HTTP, 50051 gRPC |
| PostgreSQL/pgvector | `pgvector/pgvector:0.8.2-pg17-bookworm` | 5433 PostgreSQL |

Prepare images and immutable digests:

```bash
python experiments/benchmarks/prepare.py vectordb
```

This also pulls the pinned Alpine inspector used only to measure Docker-volume size and writes:

```text
data/benchmarks/models/vectordb.json
experiments/benchmarks/vectordb/.env
```

Stop the production Chroma Compose stack before starting the benchmark stack because both use port 8001:

```bash
docker compose -f infrastructure/chroma.yml down
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml up -d
docker compose --env-file experiments/benchmarks/vectordb/.env -f experiments/benchmarks/vectordb/compose.yml ps
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

## 11. What a successful setup does and does not prove

A valid smoke run means dependencies, prepared files, real engines, and data contracts connect correctly. It does not prove one model/server is better. A standard/full comparison is valid only when all candidates use the same frozen manifest, prepared model locks, environment, and successful correctness gates. Reports are tied to their recorded hardware and software environment.

No benchmark result changes production automatically. After reviewing the Pareto set, confidence intervals, human judgments, operational gates, and limitations, update production code/config in a separate explicit change. Until then, Chroma, MiniLM, token 256/32, dense top-5 retrieval, and Qwen 3 1.7B remain provisional defaults.
