# Prepare the app models

[Install the environment](installation.md) · [Run the application](running.md)

Use this guide to prepare the local study assistant after installing its
dependencies. Run the commands from the repository root with the virtual
environment active. You do not need benchmark datasets or candidate downloads.

## Preparation commands

```powershell
python -m experiments.benchmarks.prepare app-models
```

The command currently lives under `experiments.benchmarks`, but the
`app-models` target prepares only the fixed app configuration. It does not
run experiments or download evaluation data. To preview the list without
downloading anything, add `--dry-run`.

| Used for | What the app prepares |
|---|---|
| Documents and video frames | Docling Standard with layout, TableFormer, and RapidOCR |
| Speech in audio and video | Whisper `small.en` |
| Embedding documents and queries | GTE ModernBERT base |
| Generating answers | Qwen3 1.7B |

Preparation also caches the `cl100k_base` tokenizer used for chunking.
These are the app's current defaults, not benchmark winners.

## Where the files go

The default command stores model files under `data/benchmarks/downloads/models/`
and the chunking tokenizer under `data/benchmarks/downloads/tiktoken/`. On
success it prints the path to `data/benchmarks/models/selected.json`: this is
the model lock that records the prepared files, revisions, and checksums for the
app to verify. A lock left behind by an interrupted command does not prove
preparation finished.

After the command exits successfully, [start Chroma and launch the assistant](running.md).
Check the extraction and generator readiness messages in the UI; the vector
index also needs Chroma running. If a model is reported missing, rerun
`app-models` and check its output rather than editing the lock. The app does
not download missing weights at startup.

If your Chroma index was built with MiniLM, follow the [index migration instructions](running.md#migrating-an-existing-minilm-index) before using it with GTE. A fresh installation needs no index reset.
