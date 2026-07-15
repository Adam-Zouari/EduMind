# Streamlit application

`apps/streamlit_app.py` is the complete current application. `apps/controller.py` owns upload cleanup, the configured 100 MiB limit, duplicate prevention, pipeline calls, safe errors, readiness, and reset. `apps/state.py` defines session records. No Streamlit module exists under `src` and there is no API transport layer.

The controller constructs the production `EduMindPipeline`, which connects to the provisional Chroma HTTP server configured in `config/base.yaml`. Starting the UI never starts Docker or downloads models. When Chroma is unavailable the application shows the exact Compose command.

```powershell
docker compose -f infrastructure/chroma.yml up -d
streamlit run apps/streamlit_app.py
```
