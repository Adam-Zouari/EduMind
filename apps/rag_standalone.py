"""Legacy Streamlit entrypoint retained only as a deprecation notice."""

from __future__ import annotations

import streamlit as st


def main() -> None:
    """Render a small deprecation page for the old standalone RAG UI."""
    st.set_page_config(page_title="Legacy RAG UI", layout="centered")
    st.title("Legacy RAG UI")
    st.warning("This entrypoint is deprecated.")
    st.write("Use the primary app instead:")
    st.code("streamlit run apps/streamlit_app.py", language="bash")
    st.write("Or start it through the package CLI:")
    st.code("edumind-ui", language="bash")


if __name__ == "__main__":
    main()
