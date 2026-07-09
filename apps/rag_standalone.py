"""
Streamlit Interface for OCR-to-RAG Pipeline with Ollama
Simple interface to add context and ask questions using RAG + Qwen

USAGE:
  Linux/Mac: streamlit run apps/rag_standalone.py

DO NOT RUN: python app.py (won't work!)
"""

import streamlit as st
from edumind.rag.rag_pipeline import RAGPipeline
import json
from datetime import datetime
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="RAG Pipeline with Qwen",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.initialized = False
    st.session_state.total_chunks = 0
    st.session_state.chat_history = []

# Title and description
st.title("🤖 RAG Pipeline with Qwen")
st.markdown("Add context to the knowledge base and ask questions using RAG + Ollama")

# Sidebar for configuration and stats
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Initialize pipeline button
    if not st.session_state.initialized:
        if st.button("🚀 Initialize Pipeline", type="primary"):
            with st.spinner("Initializing RAG Pipeline with Qwen..."):
                try:
                    st.session_state.pipeline = RAGPipeline(use_llm=True)
                    st.session_state.initialized = True
                    st.success("✅ Pipeline initialized!")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("Make sure Ollama is running: `ollama serve`")
    else:
        st.success("✅ Pipeline Ready")
        
        # Display stats
        st.header("📊 Statistics")
        if st.session_state.pipeline:
            stats = st.session_state.pipeline.get_stats()
            st.metric("Total Documents", stats['total_documents'])
            st.metric("Total Chunks", st.session_state.total_chunks)
            st.info(f"**Model:** {stats['embedding_model']}")
            st.info(f"**LLM:** qwen3:1.7b")
        
        # Reset button
        if st.button("🗑️ Reset Database", type="secondary"):
            if st.session_state.pipeline:
                st.session_state.pipeline.reset()
                st.session_state.total_chunks = 0
                st.session_state.chat_history = []
                st.success("Database reset!")
                st.rerun()

# Main content area
if st.session_state.initialized:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Add Context", "💬 Ask Questions", "📚 Chat History"])
    
    # Tab 1: Add Context
    with tab1:
        st.header("Add Context to Knowledge Base")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Text Input", "JSON Input", "Upload JSON File"],
            horizontal=True
        )
        
        if input_method == "Text Input":
            st.subheader("Enter Text Directly")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                text_input = st.text_area(
                    "Enter your text:",
                    height=200,
                    placeholder="Paste your text here..."
                )
            
            with col2:
                st.write("**Metadata (Optional)**")
                source = st.text_input("Source", placeholder="e.g., document.pdf")
                page = st.text_input("Page", placeholder="e.g., 1")
                category = st.text_input("Category", placeholder="e.g., AI/ML")
            
            if st.button("➕ Add to Knowledge Base", type="primary"):
                if text_input.strip():
                    # Create document
                    document = {"text": text_input}
                    if source:
                        document["source"] = source
                    if page:
                        document["page"] = page
                    if category:
                        document["category"] = category
                    
                    with st.spinner("Processing and storing..."):
                        try:
                            chunks = st.session_state.pipeline.ingest_document(document)
                            st.session_state.total_chunks += chunks
                            st.success(f"✅ Added! Created {chunks} chunks")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                else:
                    st.warning("Please enter some text!")
        
        elif input_method == "JSON Input":
            st.subheader("Enter JSON Directly")
            
            json_example = """[
  {
    "text": "Your text here...",
    "source": "document.pdf",
    "page": 1
  }
]"""
            
            json_input = st.text_area(
                "Enter JSON:",
                height=200,
                placeholder=json_example
            )
            
            if st.button("➕ Add JSON to Knowledge Base", type="primary"):
                if json_input.strip():
                    try:
                        documents = json.loads(json_input)
                        if not isinstance(documents, list):
                            documents = [documents]
                        
                        with st.spinner("Processing and storing..."):
                            chunks = st.session_state.pipeline.ingest_documents(documents)
                            st.session_state.total_chunks += chunks
                            st.success(f"✅ Added {len(documents)} documents! Created {chunks} chunks")
                    except json.JSONDecodeError as e:
                        st.error(f"❌ Invalid JSON: {e}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                else:
                    st.warning("Please enter JSON data!")
        
        else:  # Upload JSON File
            st.subheader("Upload JSON File")
            
            uploaded_file = st.file_uploader(
                "Choose a JSON file",
                type=['json'],
                help="Upload a JSON file with OCR output"
            )
            
            if uploaded_file is not None:
                if st.button("➕ Process and Add File", type="primary"):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp_file:
                            tmp_file.write(uploaded_file.getbuffer())
                            temp_path = tmp_file.name
                        
                        with st.spinner("Processing file..."):
                            chunks = st.session_state.pipeline.ingest_from_json(temp_path)
                            st.session_state.total_chunks += chunks
                            st.success(f"✅ File processed! Created {chunks} chunks")
                        
                        os.remove(temp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    # Tab 2: Ask Questions
    with tab2:
        st.header("Ask Questions")
        
        if st.session_state.total_chunks == 0:
            st.warning("⚠️ No context in the knowledge base yet. Add some context first!")
        else:
            # Query input
            query = st.text_input(
                "Your question:",
                placeholder="What would you like to know?",
                key="query_input"
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                top_k = st.slider("Number of sources", 1, 10, 3)
            
            with col2:
                stream = st.checkbox("Stream response", value=True)
            
            if st.button("🔍 Ask", type="primary"):
                if query.strip():
                    with st.spinner("Searching knowledge base..."):
                        try:
                            # Generate answer
                            result = st.session_state.pipeline.generate_answer(
                                query=query,
                                top_k=top_k,
                                stream=False  # We'll handle display ourselves
                            )
                            
                            # Display answer
                            st.subheader("💡 Answer")
                            st.markdown(result['answer'])
                            
                            # Display sources
                            st.subheader("📚 Sources")
                            for i, source in enumerate(result['sources'], 1):
                                with st.expander(f"Source {i}: {source['source']} (Page {source['page']}) - Similarity: {source['similarity']}"):
                                    # Get the actual document text from results
                                    results = st.session_state.pipeline.query(query, top_k)
                                    if i <= len(results):
                                        st.text(results[i-1]['document'])
                            
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

else:
    # Show instructions when not initialized
    st.info("👈 Click **Initialize Pipeline** in the sidebar to get started!")
    
    st.markdown("""
    ### How to use:
    
    1. **Initialize Pipeline** - Click the button in the sidebar
    2. **Add Context** - Go to the "Add Context" tab and add your documents
    3. **Ask Questions** - Go to "Ask Questions" tab and query your knowledge base
    
    ### Requirements:
    
    - Ollama must be running (`ollama serve`)
    - Qwen model must be available (`ollama pull qwen3:1.7b`)
    
    ### Features:
    
    - ✅ Add context via text, JSON, or file upload
    - ✅ Semantic search using embeddings
    - ✅ AI-powered answers using Qwen
    - ✅ Source attribution
    - ✅ Chat history
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, ChromaDB, and Ollama")

