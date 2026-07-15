# Production pipeline

`EduMindPipeline` is the one in-process application boundary. It classifies and extracts a source, normalizes it, chunks and embeds it, replaces that logical document in Chroma server, retrieves dense evidence, packs it under the token budget, and optionally calls Ollama.

It returns typed extraction, ingestion, query, answer, timing, warning, and progress values. Streamlit calls it directly. There is no API/service client and experiment runners do not create an alternative product pipeline.
