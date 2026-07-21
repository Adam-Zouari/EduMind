# EduMind installation and experiment guide

This is the complete preparation checklist for the provisional application and every benchmark. Run commands from the repository root. A smoke run proves that a real path executes; only standard/full runs on frozen data can support comparisons.

The complete procedure and formulas are in `experiments/benchmarks/benchmark_manual.md`; public evidence for every candidate decision is in `experiments/benchmarks/model_selection.md`.

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
- `google/embeddinggemma-300m` (accept the Gemma license on Hugging Face first)
- `infgrad/Jasper-Token-Compression-600M`
- `Qwen/Qwen3-Embedding-0.6B`
- `nvidia/Nemotron-3-Embed-1B-BF16`
- `Qwen/Qwen3-Embedding-4B`
- `nvidia/Nemotron-3-Embed-8B-BF16`
- `cross-encoder/ms-marco-MiniLM-L-6-v2`
- `BAAI/bge-reranker-v2-m3`
- `Qwen/Qwen3-Reranker-0.6B`
- `Qwen/Qwen3-Reranker-4B`
- `vectara/hallucination_evaluation_model`

The first seven are embedding candidates, the next four are rerankers, and HHEM supplies the automated faithfulness diagnostic. The weights remain in the normal Hugging Face cache; the JSON lock records exactly which revisions were used. To download one large model at a time, use for example:

```bash
python experiments/benchmarks/prepare.py huggingface-models --candidate Qwen/Qwen3-Embedding-4B
```

The option is repeatable. Completed candidates are written to the lock immediately, and rerunning uses Hugging Face's resumable cache.

### Ollama generation models

Run this once Ollama is running:

```bash
python experiments/benchmarks/prepare.py ollama-models
```

It pulls the following base model tags sequentially and records their installed digests in `data/benchmarks/models/ollama.json`:

- `qwen3:1.7b`
- `qwen3.5:4b-q4_K_M`
- `qwen3.5:9b-q4_K_M`
- `gemma4:12b-it-q4_K_M`
- `ministral-3:8b-instruct-2512-q4_K_M`
- `gpt-oss:20b`

The benchmark creates nine profiles from those six tags: Qwen 3.5 4B/9B are each tested with direct and thinking modes, and GPT-OSS 20B is tested with low and medium reasoning. Do not pull separate model names for those profiles. `--candidate MODEL_TAG` also works here.

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
- Docling's local layout, table, and formula artifacts;
- PP-StructureV3;
- PaddleOCR-VL-1.6;
- `zai-org/GLM-OCR`;
- `opendatalab/MinerU2.5-Pro-2605-1.2B`;
- `allenai/olmOCR-2-7B-1025`;
- Distil-Whisper large v3.5, Parakeet TDT 0.6B v3, and Canary-Qwen 2.5B.

Large extraction preparation should normally be run one candidate at a time:

```bash
python experiments/benchmarks/prepare.py extraction-models --candidate docling
python experiments/benchmarks/prepare.py extraction-models --candidate pp-structure-v3
python experiments/benchmarks/prepare.py extraction-models --candidate paddleocr-vl-1.6
python experiments/benchmarks/prepare.py extraction-models --candidate olmocr-2-7b
```

OpenAI Whisper, faster-whisper, Hugging Face document-VLM, and ASR assets are placed under `data/benchmarks/downloads/models/`; PaddleOCR-VL and PP-StructureV3 use the prepared `~/.paddlex` cache and are locked by their PaddleOCR/PaddleX package versions. Tesseract is a separately installed system engine. pypdf, pdfplumber, Docling, python-docx, and Mammoth come from the lock files.

The following candidates need an official runtime or service in addition to their prepared weights because installing every rapidly changing VLM stack into the application environment would create dependency conflicts:

- GLM-OCR: install `glmocr[selfhosted]` from the [official GLM-OCR repository](https://github.com/zai-org/GLM-OCR), serve the exact local model directory recorded in `data/benchmarks/models/extraction.json`, create a YAML config whose `pipeline.maas.enabled` is `false` and whose `pipeline.ocr_api` points to that service, and put the config path in each relevant manifest sample as `options.glm_config_path`. Hosted/API-key mode is rejected.
- MinerU: install `mineru[all]` using the [official MinerU instructions](https://github.com/opendatalab/MinerU). Preparation writes a candidate-specific `mineru_config_path` pointing to the pinned 2605 weights. The runner passes that config, forces `MINERU_MODEL_SOURCE=local`/offline Hugging Face mode, and calls the `vlm-transformers` backend so an unrecorded default model cannot replace the candidate.
- olmOCR: install the toolkit from the [official olmOCR repository](https://github.com/allenai/olmocr); the runner calls `olmocr` with the prepared local model path.
- Canary-Qwen: install the NeMo revision required by the [official model card](https://huggingface.co/nvidia/canary-qwen-2.5b). Keep that dependency isolated if it conflicts with the main environment, and run the audio candidate from that environment against the same manifest.

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

- [OmniDocBench v1.6/v1.7](https://github.com/opendatalab/OmniDocBench): primary complete-page corpus and official text, layout, reading-order, table TEDS, and formula CDM evaluators. Put downloaded pages/PDFs under `data/benchmarks/raw/omnidocbench/`. Do not mix versions in one manifest.
- [olmOCR-Bench](https://github.com/allenai/olmocr): difficult PDF linearization cases for old scans, multi-column pages, tiny text, tables, math, headers, and footers. Put assets under `data/benchmarks/raw/olmocr-bench/`.
- [OHR-Bench](https://github.com/opendatalab/OHR-Bench): 350 PDFs and questions designed to measure extraction-caused RAG degradation. Put it under `data/benchmarks/raw/ohr-bench/`; use its official split and keep its downstream results separate from QASPER selection.
- [PureDocBench dataset](https://huggingface.co/datasets/zhihengli-casia/puredocbench): optional robustness track with clean, digitally degraded, and real-degraded pages. Put it under `data/benchmarks/raw/puredocbench/` and do not combine its degradation conditions without stratified reporting.
- [Open ASR Leaderboard dataset collection](https://huggingface.co/datasets/hf-audio/open-asr-leaderboard): source for reproducible English ASR test sets such as LibriSpeech, Earnings22, SPGISpeech, VoxPopuli, GigaSpeech, and AMI. Select licensed subsets and put them under `data/benchmarks/raw/asr/`; add EduMind technical-vocabulary/noise clips rather than treating a public average as sufficient.
- [QASPER](https://huggingface.co/datasets/allenai/qasper): downloaded automatically by section 6 and combined with verified structured evidence for chunking, retrieval, generation, and final RAG.

OmniDocBench and olmOCR-Bench provide credible public parsing evidence, but EduMind still runs its own benchmark because it requires English-specific strata, exact typed output/provenance, the selected table/formula serialization, local latency/resources, and downstream retrieval/answer quality. OHR-Bench directly helps the downstream confirmation but does not replace QASPER or the component experiments.

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
