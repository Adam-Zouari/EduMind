# Prepare the app models

[Install the environment](installation.md) · [Run the application](running.md)

Use this guide to prepare the local study assistant after installing its
dependencies. Run the commands from the repository root with the virtual
environment active. You do not need benchmark datasets or candidate downloads.

## Preparation commands

```powershell
python -m edumind.prepare_models
```

Use this command to download and verify the app's models without starting the
application or Chroma. It needs no benchmark files. Add `--dry-run` to preview
the pinned assets without writing files or accessing the network. To start the
app and be prompted to prepare missing models, use [the launcher](running.md).

| Used for | What the app prepares |
|---|---|
| Documents and video frames | Docling Standard with layout, TableFormer, and RapidOCR |
| Speech in audio and video | Whisper `small.en` |
| Embedding documents and queries | GTE ModernBERT base |
| Generating answers | Qwen3 1.7B |

Preparation also caches the `cl100k_base` tokenizer used for chunking.
These are the app's current defaults, not benchmark winners.

## Where the files go

Model snapshots and the Docling bundle live under `data/models/snapshots/`. The
pinned tokenizer lives under `data/models/tiktoken/`. On success, the command
prints the path to the generated app lock (by default `data/models/app.json`),
which records model revisions, local directories, and checksums.
`EDUMIND_MODEL_LOCK` can override the lock file path. A lock created before an
interruption may be incomplete. Use `python -m edumind.prepare_models --check`
to verify readiness offline: exit code 0 means ready, 2 means preparation is
needed, and 1 means a verification or configuration error.
An incompatible or missing Docling installation is a dependency error, not
a reason to download the models again.
The preparation command supports the app's pinned embedding and generator
only. If a custom config names a different model, the check reports it before
downloading anything.

If you lose the lock but the prepared files are intact, rerun the preparation
command. It can rebuild the lock offline using the saved snapshot receipts and
the verified Docling bundle. A corrupt bundle is reported, not overwritten.
If a download was interrupted after the snapshot files were published but
before their receipt was saved, rerun preparation with internet access.
It checks the local files against the pinned Hugging Face file inventory
without downloading the weights again. Files that do not match are left untouched.

After the command exits successfully, [start Chroma and launch the assistant](running.md).
Check the extraction and generator readiness messages in the UI; the vector
index also needs Chroma running. If a model is reported missing, rerun
`python -m edumind.prepare_models` and check its output rather than editing the
lock. App startup does not download missing weights.

## Reusing downloads from an older installation

If your `.env` or app configuration still points at the old mixed lock, remove
that override or change it to `data/models/app.json` before preparing the app.

If you already have a revision directory and its matching
`<commit>.complete.json` receipt, copy both into the corresponding
`data/models/snapshots/<repository-with-double-dashes>/` directory. Do this
before preparing the app, and do not overwrite an existing destination. This
works for the app's Hugging Face models and Docling's Hugging Face component
snapshots, provided the pinned revision and file policy match. An older
unversioned directory without a receipt is not reusable through this path.
Do not copy a mixed model lock or a completed benchmark Docling bundle. The
app command verifies each receipt against the copied files, downloads
anything else it needs, and writes its own lock.

To reuse an older chunking tokenizer, copy the existing
`data/benchmarks/downloads/tiktoken/` contents into an empty
`data/models/tiktoken/` directory. The command verifies the pinned SHA-256
before accepting the bytes. Keep the source downloads and lock untouched.

The current Chroma index identity includes the embedding snapshot's path.
Moving from the old storage layout to the new one can therefore require
[rebuilding the index](running.md#migrating-an-existing-index), even if the
GTE revision did not change. Preparation never resets Chroma.
