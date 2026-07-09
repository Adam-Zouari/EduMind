"""
OCR-RAG Pipeline Streamlit App (Microservices Version)
Uses API-based orchestrator - no dependency conflicts!
"""

import streamlit as st
from pathlib import Path
import tempfile
from edumind.pipeline.orchestrator_api import APIOrchestrator

st.set_page_config(
    page_title="OCR-RAG Pipeline",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'orchestrator' not in st.session_state:
    st.session_state.orchestrator = None
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
st.sidebar.title("🔍 OCR-RAG Pipeline")
st.sidebar.markdown("**Microservices Architecture**")
st.sidebar.markdown("---")

# Service URLs
ocr_url = st.sidebar.text_input("OCR Service URL", "http://localhost:8000")
rag_url = st.sidebar.text_input("RAG Service URL", "http://localhost:8001")

if st.sidebar.button("🚀 Connect to Services", type="primary"):
    try:
        with st.spinner("Connecting to services..."):
            st.session_state.orchestrator = APIOrchestrator(ocr_url, rag_url)
        st.sidebar.success("✅ Connected!")
    except Exception as e:
        st.sidebar.error(f"❌ Connection failed: {e}")
        st.sidebar.info("""
        **Start the services:**
        
        Terminal 1:
        `edumind-ocr-api`

        Terminal 2:
        `edumind-rag-api`
        """)

# Main content
st.title("🔍 OCR-RAG Pipeline")
st.markdown("Upload documents, extract text, and ask questions!")

if st.session_state.orchestrator is None:
    st.warning("⚠️ Please connect to services using the sidebar")
    st.stop()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "💬 Ask Questions", "📜 Chat History", "📋 Processed Files"])

# Tab 1: Upload & Process
with tab1:
    st.header("Upload Files")
    
    uploaded_files = st.file_uploader(
        "Choose files to process",
        type=['pdf', 'docx', 'png', 'jpg', 'jpeg', 'html', 'mp3', 'wav', 'mp4', 'avi'],
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        ingest_to_rag = st.checkbox("Ingest to RAG", value=True)
    
    if uploaded_files and st.button("🚀 Process Files", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            try:
                # Process file
                result = st.session_state.orchestrator.process_file(tmp_path, ingest_to_rag)
                
                # Store result
                st.session_state.processed_files.append({
                    'filename': uploaded_file.name,
                    'result': result
                })
                
                # Show result
                with st.expander(f"✅ {uploaded_file.name}"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Format", result['format_type'])
                    col2.metric("Characters", len(result['text']))
                    col3.metric("RAG Chunks", result['rag_chunks'])
                    
                    st.text_area("Extracted Text (preview)", result['text'][:500] + "...", height=150)
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        status_text.text("✅ All files processed!")
        st.success(f"Processed {len(uploaded_files)} files")

# Tab 2: Ask Questions
with tab2:
    st.header("Ask Questions")
    
    query = st.text_input("Enter your question:", placeholder="What is this document about?")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        top_k = st.slider("Number of sources", 1, 10, 5)
    
    if query and st.button("🔍 Search", type="primary"):
        with st.spinner("Searching..."):
            try:
                result = st.session_state.orchestrator.query(query, top_k=top_k, generate_answer=True)
                
                # Store in chat history
                st.session_state.chat_history.append({
                    'question': query,
                    'answer': result.get('answer', 'No answer generated'),
                    'sources': result.get('sources', [])
                })
                
                # Display answer
                st.markdown("### 💡 Answer")
                st.info(result.get('answer', 'No answer generated'))
                
                # Display sources
                st.markdown("### 📚 Sources")
                for i, source in enumerate(result.get('sources', []), 1):
                    with st.expander(f"Source {i}: {source.get('source', 'Unknown')} (Page {source.get('page', 'N/A')})"):
                        # Similarity is already formatted as a string (e.g., "95.5%")
                        similarity = source.get('similarity', 'N/A')
                        st.write(f"**Similarity:** {similarity}")
                        st.write(source.get('text', ''))
                
            except Exception as e:
                st.error(f"❌ Query failed: {e}")

# Tab 3: Chat History
with tab3:
    st.header("Chat History")
    
    if st.session_state.chat_history:
        for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
            with st.expander(f"Q{len(st.session_state.chat_history) - i + 1}: {chat['question']}"):
                st.markdown(f"**Answer:** {chat['answer']}")
                st.markdown(f"**Sources:** {len(chat['sources'])} documents")
    else:
        st.info("No questions asked yet")

# Tab 4: Processed Files
with tab4:
    st.header("Processed Files")
    
    if st.session_state.processed_files:
        for file_info in st.session_state.processed_files:
            with st.expander(f"📄 {file_info['filename']}"):
                result = file_info['result']
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Format", result['format_type'])
                col2.metric("Characters", len(result['text']))
                col3.metric("RAG Chunks", result['rag_chunks'])
                # Handle extraction_time which might be a string or number
                extraction_time = result.get('extraction_time', 0)
                if isinstance(extraction_time, (int, float)):
                    col4.metric("Time", f"{extraction_time:.2f}s")
                else:
                    col4.metric("Time", str(extraction_time))
    else:
        st.info("No files processed yet")
