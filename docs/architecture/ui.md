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
- Session records identify logical filename and content checksum. Only a
  successful same-document, same-version ingestion is skipped; pressing
  **Extract and index** again retries a failed upload. This is not cross-process
  ingestion coordination.
- Re-uploading a changed file with the same logical name replaces its old chunks.
- Identical bytes under another logical name are processed, including when
  replacing that name's older content. Successful replacement retires its stale
  session records.
- Evidence expanders show every overlapping page and original segment time range
  for the displayed text, including after context truncation.
- Readiness reports extraction/model requirements, Chroma/index state, and the
  local generator snapshot separately.
- Starting the UI never starts Docker or downloads models.
- User-visible errors are safe; full exceptions remain in local logs.

Operational commands and troubleshooting belong in the
[run instructions](../setup/running.md); this page describes UI responsibilities
only.
