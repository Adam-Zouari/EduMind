# EduMind

EduMind is a local study assistant for educational material. It extracts content
from PDFs, DOCX files, images, audio, and video; indexes that content; and answers
questions with citations to the source. The application keeps pages, timestamps,
and other provenance with extracted evidence so an answer can be checked against
the original material. Extraction, retrieval, and generation run locally.

The repository also contains the experiments used to choose those components.
Instead of treating public model rankings as the answer, EduMind benchmarks
document and speech extraction, video, chunking and embeddings, retrieval and
reranking, vector databases, and generation on project-specific data before any
winner is promoted to the application.

> **Under active development.** The application and benchmark code are available,
> but authoritative dataset review, comparative runs, and final component
> selection are still pending. Current defaults are provisional, and smoke tests
> do not establish model quality. This notice will be removed once validated
> results are ready.

## Project journey and evidence

Read the project in this order:

1. [Architecture overview](docs/architecture/overview.md) explains the local application and its separation from experiments.
2. [Dataset guide](docs/benchmarks/datasets.md) describes benchmark sources, manifests, and evidence requirements.
3. [Model selection](docs/benchmarks/model-selection.md) records why candidates entered or left the shortlist; [selection evidence](experiments/benchmarks/selection_evidence.csv) pins their identities and revisions.
4. [Methodology](docs/benchmarks/methodology.md) explains the experiment stages; [metrics](docs/benchmarks/metrics.md) defines what each stage measures.
5. [Benchmark runbook](docs/benchmarks/running.md) gives preparation and execution commands.
6. [Pending data review](docs/benchmarks/pending-data-review.md) tracks decisions that require the real, reviewed datasets before results can be trusted.

```text
candidate roster + reviewed datasets -> CPU/CUDA smoke -> GPU preflight
    -> qualified-candidate development -> finalist validation
    -> human review -> one locked test
    -> explicit application configuration change
```

Benchmark protocols, model snapshots, input manifests, per-sample artifacts,
and MLflow runs preserve what was actually evaluated. Experiments never silently
change the application's configuration. No final comparative result or selected
component is claimed yet.

## Reproduction

The [installation guide](docs/setup/installation.md) covers Python 3.12, system
tools, model preparation, and the pinned environments. After that one-time setup,
start the provisional application from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
docker compose -f infrastructure/chroma.yml up -d
streamlit run src/edumind/ui/streamlit_app.py
```

The [application guide](docs/setup/running.md) covers readiness, shutdown, and
troubleshooting. If you have an older MiniLM index, follow the documented
[reset and reindex procedure](docs/setup/installation.md#migrating-an-existing-minilm-index);
the application does not delete it automatically.

For experiments, the [benchmark runbook](docs/benchmarks/running.md) gives the
commands in stage order. Smoke fixtures test execution without authoritative
datasets, while preflight establishes target-GPU feasibility without ranking
quality. Development, validation, and locked comparisons require the reviewed
manifests described in the [dataset guide](docs/benchmarks/datasets.md).

## Repository layout

```text
.
|-- src/edumind/              # Application, extraction, RAG, and Streamlit UI
|-- experiments/benchmarks/   # Benchmark protocols, runners, metrics, preparation
|-- data/benchmarks/          # Smoke fixtures and local prepared assets
|-- config/base.yaml          # Provisional application defaults
|-- infrastructure/           # Local Chroma Docker Compose service
|-- requirements/             # Pinned application and benchmark environments
|-- tests/                    # Unit and benchmark-contract tests
|-- docs/                     # Architecture, setup, and benchmark guides
`-- artifacts/                # Ignored local run output
```

## Local application

```text
PDF / DOCX / image / audio / video
    -> extraction with pages, timestamps, and source provenance
    -> chunking and embeddings -> Chroma HTTP
    -> evidence retrieval -> local generation -> cited answer
```

The Streamlit interface accepts uploads, indexes extracted evidence, and shows
numbered citations with answers. Current choices are defined in
[config/base.yaml](config/base.yaml), not inferred from benchmark results. The
application does not download models, start Docker, or promote benchmark winners
on its own. See the [architecture overview](docs/architecture/overview.md) for
the implementation boundaries.

## Tests

After installing the pinned environment, run the repository test suite with:

```powershell
python -m pytest -q
```

These tests do not replace real-model smoke inference or dataset-backed
benchmark comparisons. EduMind is currently English-first and local; it does
not expose a public API or require hosted inference.

## License

The source code is licensed under the [MIT License](LICENSE). Model weights and
benchmark datasets retain their own upstream terms; consult the
[dataset guide](docs/benchmarks/datasets.md) and
[model-selection record](docs/benchmarks/model-selection.md) before redistribution.
