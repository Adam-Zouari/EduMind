from experiments.mlflow.mlflow_config import EVALUATION_DIR
"""
Test a single chunking strategy with debug output.
"""

import sys
from pathlib import Path

import json
import re
from edumind.rag.embedder import Embedder
from edumind.rag.vector_store import VectorStore

# Load data
print("Loading data...")
with open(EVALUATION_DIR / "ground_truth.json", 'r') as f:
    ground_truth = json.load(f)

with open(EVALUATION_DIR / "eval_queries.json", 'r') as f:
    queries = json.load(f)

print(f"Loaded {len(ground_truth)} chunks and {len(queries)} queries")

# Simple fixed character chunker
class FixedCharacterChunker:
    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text, metadata=None):
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunk = {
                    'text': chunk_text,
                    'chunk_index': chunk_idx,
                    'start_char': start,
                    'end_char': end
                }
                if metadata:
                    chunk.update(metadata)
                chunks.append(chunk)
                chunk_idx += 1
            
            start = end - self.overlap
        
        return chunks

# Initialize
print("\nInitializing components...")
embedder = Embedder()
vector_store = VectorStore()
vector_store.reset_collection()
chunker = FixedCharacterChunker(chunk_size=500, overlap=100)

# Chunk all ground truth documents
print("\nChunking documents...")
all_chunks = []
chunk_id_mapping = {}

for chunk_id, chunk_data in ground_truth.items():
    text = chunk_data['text']
    metadata = {
        'original_chunk_id': chunk_id,
        'source': chunk_data.get('source', 'unknown'),
        'page': chunk_data.get('page', 0),
        'domain': chunk_data.get('domain', 'unknown'),
        'topic': chunk_data.get('topic', 'unknown')
    }
    
    chunks = chunker.chunk_text(text, metadata)
    
    # Map each new chunk back to its original chunk
    for chunk in chunks:
        new_chunk_id = chunk.get('chunk_id', chunk.get('chunk_index', len(all_chunks)))
        chunk_id_mapping[str(new_chunk_id)] = chunk_id
        chunk['original_chunk_id'] = chunk_id
    
    all_chunks.extend(chunks)

print(f"Created {len(all_chunks)} chunks from {len(ground_truth)} original chunks")

# Add embeddings
print("\nGenerating embeddings...")
chunks_with_embeddings = embedder.embed_chunks(all_chunks)

# Add to vector store
print("Adding to vector store...")
vector_store.add_documents(chunks_with_embeddings)

# Test with first query
print("\n" + "="*80)
print("TESTING RETRIEVAL")
print("="*80)

test_query = queries[0]
query_text = test_query["query"]
relevant_chunk_ids = set(test_query["relevant_chunks"])

print(f"\nQuery: {query_text}")
print(f"Expected relevant chunks: {list(relevant_chunk_ids)[:3]}")

# Retrieve
query_embedding = embedder.embed_text(query_text)
results = vector_store.query(query_embedding.tolist(), top_k=5)

# Extract retrieved original chunk IDs
retrieved_original_ids = []
if results['ids'] and results['ids'][0]:
    for i, new_chunk_id in enumerate(results['ids'][0]):
        metadata = results['metadatas'][0][i]
        original_id = metadata.get('original_chunk_id', '')
        if original_id:
            retrieved_original_ids.append(original_id)
        
        print(f"\nResult {i+1}:")
        print(f"  Original Chunk ID: {original_id}")
        print(f"  Is Relevant: {original_id in relevant_chunk_ids} {'✅' if original_id in relevant_chunk_ids else '❌'}")
        print(f"  Text: {results['documents'][0][i][:100]}...")

# Compute metrics
matches = set(retrieved_original_ids) & relevant_chunk_ids
precision = len(matches) / len(retrieved_original_ids) if retrieved_original_ids else 0

print(f"\n" + "="*80)
print(f"Retrieved: {retrieved_original_ids}")
print(f"Expected: {list(relevant_chunk_ids)[:5]}")
print(f"Matches: {list(matches)}")
print(f"Precision@5: {precision:.3f}")
print("="*80)

if precision > 0.5:
    print("\n✅ SUCCESS! The fix is working!")
else:
    print("\n❌ Still low precision. Possible issues:")
    print("   - Embedding quality")
    print("   - Semantic mismatch")
    print("   - Re-chunking splits relevant content")

