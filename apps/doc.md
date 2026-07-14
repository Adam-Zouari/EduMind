# Applications

`apps/streamlit_app.py` is a thin rendering entry point. Testable upload, query, reset, readiness, and state-normalization behavior lives in `edumind.app`.

Uploads are identified by SHA-256, so a Streamlit rerun does not ingest the same bytes twice. Temporary files are deleted in `finally` blocks, while the original safe filename is preserved for indexing and citations. The UI reports progress, document status, warnings, cited sources, and timings. Reset requires explicit confirmation, and user-facing failures use safe messages with model-install guidance rather than raw tracebacks.

Run locally with `edumind ui`. The app is intentionally local and single-user; it must not be exposed as a multi-tenant service without adding authentication, durable job coordination, and tenant-aware storage.
