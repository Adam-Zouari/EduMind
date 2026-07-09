"""
Streamlit Interface for OCR-RAG Pipeline
Complete interface for document processing and question answering

USAGE:
  streamlit run apps/streamlit_app.py
"""

import streamlit as st
from edumind.pipeline.orchestrator import OCRRAGOrchestrator
from datetime import datetime
from pathlib import Path
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="OCR-RAG Pipeline",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
    st.session_state.initialized = False
    st.session_state.total_chunks = 0
    st.session_state.processed_files = []
    st.session_state.chat_history = []

# Title and description
st.title("🚀 OCR-RAG Pipeline")
st.markdown("**Extract content from any file format and ask questions using AI**")

# Sidebar for configuration and stats
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Initialize pipeline button
    if not st.session_state.initialized:
        if st.button("🚀 Initialize Pipeline", type="primary"):
            with st.spinner("Initializing OCR-RAG Pipeline..."):
                try:
                    st.session_state.orchestrator = OCRRAGOrchestrator(use_llm=True)
                    st.session_state.initialized = True
                    st.success("✅ Pipeline initialized!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("Make sure Ollama is running: `ollama serve`")
    else:
        st.success("✅ Pipeline Ready")
        
        # Display stats
        st.header("📊 Statistics")
        if st.session_state.orchestrator:
            stats = st.session_state.orchestrator.get_stats()
            st.metric("Total Documents", stats['rag']['total_documents'])
            st.metric("Total Chunks", st.session_state.total_chunks)
            st.metric("Files Processed", len(st.session_state.processed_files))
            st.info(f"**Embedding Model:** {stats['rag']['embedding_model']}")
            st.info(f"**LLM:** qwen3:1.7b")
        
        # Supported formats
        st.header("📁 Supported Formats")
        st.markdown("""
        - 📄 PDF
        - 📝 DOCX
        - 🖼️ Images (PNG, JPG)
        - 🌐 HTML/Web
        - 🎵 Audio (MP3, WAV)
        - 🎬 Video (MP4, AVI)
        """)
        
        # Reset button
        if st.button("🗑️ Reset Database", type="secondary"):
            if st.session_state.orchestrator:
                st.session_state.orchestrator.reset_rag()
                st.session_state.total_chunks = 0
                st.session_state.processed_files = []
                st.session_state.chat_history = []
                st.success("Database reset!")
                st.rerun()

# Main content area
if st.session_state.initialized:
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "💬 Ask Questions", "📚 Chat History", "📋 Processed Files"])
    
    # Tab 1: Upload & Process Files
    with tab1:
        st.header("Upload and Process Documents")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to process",
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg', 'html', 'mp3', 'wav', 'mp4', 'avi'],
            accept_multiple_files=True,
            help="Upload documents to extract content and add to knowledge base"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            clean_text = st.checkbox("Clean extracted text", value=True)
        
        with col2:
            auto_ingest = st.checkbox("Auto-ingest to RAG", value=True)
        
        if uploaded_files and st.button("🔄 Process Files", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            results = []
            
            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        tmp_path = tmp_file.name
                    
                    # Process file
                    result = st.session_state.orchestrator.process_file(
                        file_path=tmp_path,
                        ingest_to_rag=auto_ingest,
                        clean_text=clean_text
                    )
                    
                    result['filename'] = uploaded_file.name
                    result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    results.append(result)
                    
                    # Update stats
                    if result.get('rag_ingested'):
                        st.session_state.total_chunks += result.get('rag_chunks', 0)
                        st.session_state.processed_files.append(result)
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                # Update progress
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            status_text.text("Processing complete!")
            
            # Display results
            st.subheader("📊 Processing Results")

            for result in results:
                with st.expander(f"{'✅' if result['ocr_success'] else '❌'} {result['filename']}"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Format", result.get('format_type', 'Unknown'))

                    with col2:
                        st.metric("Extraction Time", f"{result.get('extraction_time', 0):.2f}s")

                    with col3:
                        if result.get('rag_ingested'):
                            st.metric("RAG Chunks", result.get('rag_chunks', 0))
                        else:
                            st.metric("RAG Status", "Not ingested")

                    if result['ocr_success']:
                        st.text_area("Extracted Text (Preview)", result['text'][:500] + "...", height=150, key=f"preview_{result['filename']}")
                    else:
                        st.error(f"Error: {result.get('ocr_error', 'Unknown error')}")

    # Tab 2: Ask Questions
    with tab2:
        st.header("Ask Questions")

        if st.session_state.total_chunks == 0:
            st.warning("⚠️ No documents in the knowledge base yet. Upload and process files first!")
        else:
            st.info(f"📚 Knowledge base contains {st.session_state.total_chunks} chunks from {len(st.session_state.processed_files)} files")

            # Query input
            query = st.text_input(
                "Your question:",
                placeholder="What would you like to know?",
                key="query_input"
            )

            col1, col2 = st.columns([1, 3])

            with col1:
                top_k = st.slider("Number of sources", 1, 10, 5)

            if st.button("🔍 Ask", type="primary"):
                if query.strip():
                    with st.spinner("Searching knowledge base and generating answer..."):
                        try:
                            # Generate answer
                            result = st.session_state.orchestrator.query(
                                query_text=query,
                                top_k=top_k,
                                generate_answer=True
                            )

                            # Display answer
                            st.subheader("💡 Answer")
                            st.markdown(result['answer'])

                            # Display sources
                            st.subheader("📚 Sources")
                            for i, source in enumerate(result['sources'], 1):
                                with st.expander(f"Source {i}: {source['source']} (Page {source['page']}) - Similarity: {source['similarity']}"):
                                    # Get the actual document text from context
                                    context_lines = result['context'].split('[Document ')
                                    if i < len(context_lines):
                                        doc_text = context_lines[i].split('\n', 1)[1] if '\n' in context_lines[i] else ""
                                        st.text(doc_text[:500] + "..." if len(doc_text) > 500 else doc_text)

                            # Add to chat history
                            st.session_state.chat_history.append({
                                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                'query': query,
                                'answer': result['answer'],
                                'sources': result['sources']
                            })

                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.warning("Please enter a question!")

    # Tab 3: Chat History
    with tab3:
        st.header("Chat History")

        if not st.session_state.chat_history:
            st.info("No questions asked yet.")
        else:
            # Display in reverse order (newest first)
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"🕐 {chat['timestamp']} - {chat['query'][:50]}..."):
                    st.markdown(f"**Question:** {chat['query']}")
                    st.markdown(f"**Answer:** {chat['answer']}")
                    st.markdown("**Sources:**")
                    for j, source in enumerate(chat['sources'], 1):
                        st.text(f"  {j}. {source['source']} (Page {source['page']}) - {source['similarity']}")

            # Clear history button
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()

    # Tab 4: Processed Files
    with tab4:
        st.header("Processed Files")

        if not st.session_state.processed_files:
            st.info("No files processed yet.")
        else:
            st.write(f"**Total files processed:** {len(st.session_state.processed_files)}")

            # Display as table
            for i, file_info in enumerate(reversed(st.session_state.processed_files)):
                with st.expander(f"📄 {file_info['filename']} - {file_info['timestamp']}"):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Format", file_info.get('format_type', 'Unknown'))

                    with col2:
                        st.metric("Chunks", file_info.get('rag_chunks', 0))

                    with col3:
                        st.metric("Extraction Time", f"{file_info.get('extraction_time', 0):.2f}s")

                    with col4:
                        text_length = len(file_info.get('text', ''))
                        st.metric("Text Length", f"{text_length:,} chars")

                    # Show metadata
                    if file_info.get('metadata'):
                        st.json(file_info['metadata'])

else:
    # Show instructions when not initialized
    st.info("👈 Click **Initialize Pipeline** in the sidebar to get started!")

    st.markdown("""
    ### 🚀 How to use:

    1. **Initialize Pipeline** - Click the button in the sidebar
    2. **Upload Files** - Go to "Upload & Process" tab and upload your documents
    3. **Ask Questions** - Go to "Ask Questions" tab and query your knowledge base

    ### 📋 Requirements:

    - Ollama must be running (`ollama serve`)
    - Qwen model must be available (`ollama pull qwen3:1.7b`)

    ### ✨ Features:

    - ✅ **Multi-format OCR** - PDF, DOCX, Images, Audio, Video, HTML
    - ✅ **Automatic text extraction** - Clean and structured text
    - ✅ **Semantic search** - Find relevant information using embeddings
    - ✅ **AI-powered answers** - Get answers using Qwen LLM
    - ✅ **Source attribution** - See where answers come from
    - ✅ **Chat history** - Track your questions and answers
    - ✅ **Batch processing** - Upload multiple files at once

    ### 🎯 Supported File Formats:

    | Format | Extensions | Description |
    |--------|-----------|-------------|
    | PDF | .pdf | Portable Document Format |
    | Word | .docx | Microsoft Word documents |
    | Images | .png, .jpg, .jpeg | OCR from images |
    | Web | .html | HTML web pages |
    | Audio | .mp3, .wav | Speech-to-text transcription |
    | Video | .mp4, .avi | Extract audio and transcribe |
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, OCR Pipeline, ChromaDB, and Ollama")
