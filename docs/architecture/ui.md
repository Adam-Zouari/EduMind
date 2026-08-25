# Streamlit application

[Project overview](../../README.md) · [Run instructions](../setup/running.md) ·
[Architecture](overview.md) ·
[Pipeline](application.md)

## Role

`edumind.ui` is the user-facing layer. It presents upload, readiness, query, evidence,
timing, document-status, and reset controls while delegating all extraction and
RAG behavior to the production pipeline.

```text
streamlit_app.py -> AppController -> EduMindPipeline
```

The UI is installed with the rest of `edumind`; there is no separate app tree or
API transport layer in the current local application.

## Files

| File | Responsibility |
|---|---|
| `streamlit_app.py` | Page layout, widgets, progress display, answers, and cited evidence |
| `controller.py` | Upload size enforcement, temporary-file cleanup, duplicate prevention, safe pipeline calls, readiness, and reset |
| `state.py` | Typed session records and document status normalization |

The controller is kept independent of Streamlit widgets so application behavior
can be checked without duplicating business logic in the view.

## Runtime behavior

- The UI constructs one cached `EduMindPipeline` for the process.
- Uploads are limited to the configured 100 MiB default and written to a temporary
  file that is removed after processing.
- Content checksums prevent duplicate ingestion during Streamlit reruns.
- Re-uploading a changed file with the same logical name replaces its old chunks.
- Readiness distinguishes the Chroma connection from the local generator snapshot.
- Starting the UI never starts Docker or downloads models.
- User-visible errors are safe; full exceptions remain in local logs.

## Run

```powershell
docker compose -f infrastructure/chroma.yml up -d
streamlit run src/edumind/ui/streamlit_app.py
```

Follow the [run instructions](../setup/running.md) if the model lock,
model directories, external tools, or Chroma server have not been prepared.
