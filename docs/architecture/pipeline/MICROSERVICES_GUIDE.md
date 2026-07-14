# Local service guide

Start the local adapters with:

```powershell
edumind extraction-api
edumind rag-api
```

Both bind to `127.0.0.1`. Use `/health/live` for process liveness and `/health/ready` for dependency readiness. Uploads are streamed and bounded; reset operations are serialized. See [`services/doc.md`](../../../services/doc.md) for the maintained contract and safety boundary.
