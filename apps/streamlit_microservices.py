"""Legacy microservices Streamlit entrypoint retained as a thin notice page."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Render a deprecation notice for the old microservices UI."""
    st.set_page_config(page_title="Legacy Microservices UI", layout="centered")
    st.title("Legacy Microservices UI")
    st.warning("This entrypoint is deprecated.")
    st.write("The main supported UI is now the primary local app:")
    st.code("streamlit run apps/streamlit_app.py", language="bash")
    st.write("If you still want API mode, run the services directly:")
    st.code("edumind-ocr-api", language="bash")
    st.code("edumind-rag-api", language="bash")


if __name__ == "__main__":
    main()
