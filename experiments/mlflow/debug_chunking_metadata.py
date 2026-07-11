from experiments.mlflow.mlflow_config import EVALUATION_DIR

"""
Debug script to check if metadata is being stored and retrieved correctly.
"""


import json

from edumind.rag.embedder import Embedder
from edumind.rag.vector_store import VectorStore

# Load data
print("Loading data...")
with open(EVALUATION_DIR / "ground_truth.json") as f:
    ground_truth = json.load(f)

with open(EVALUATION_DIR / "eval_queries.json") as f:
    queries = json.load(f)

print(f"Loaded {len(ground_truth)} chunks and {len(queries)} queries")

# Initialize
print("\nInitializing components...")
embedder = Embedder()
vector_store = VectorStore()
vector_store.reset_collection()

# Add first 100 chunks with original_chunk_id
print("\nAdding chunks to vector store...")
chunks_to_add = []
for i, (chunk_id, chunk_data) in enumerate(list(ground_truth.items())[:100]):
    chunk = {
        'text': chunk_data['text'],
        'original_chunk_id': chunk_id,  # Store original ID
        'source': chunk_data.get('source', 'unknown'),
        'page': chunk_data.get('page', 0)
    }
    chunks_to_add.append(chunk)

# Generate embeddings
chunks_with_embeddings = embedder.embed_chunks(chunks_to_add)

# Add to vector store
vector_store.add_documents(chunks_with_embeddings)
print(f"Added {len(chunks_with_embeddings)} chunks")

# Test retrieval
print("\n" + "="*80)
print("TESTING RETRIEVAL")
print("="*80)

test_query = queries[0]
query_text = test_query["query"]
relevant_chunks = test_query["relevant_chunks"]

print(f"\nQuery: {query_text}")
print(f"Expected relevant chunks: {relevant_chunks[:3]}")

# Generate query embedding
query_embedding = embedder.embed_text(query_text)

# Retrieve
results = vector_store.query(query_embedding.tolist(), top_k=5)

print(f"\nRetrieved {len(results['ids'][0])} results")
print("\nDETAILED RESULTS:")
print("-" * 80)

retrieved_original_ids = []
for i in range(len(results['ids'][0])):
    vector_id = results['ids'][0][i]
    metadata = results['metadatas'][0][i]
    document = results['documents'][0][i][:150]
    
    print(f"\nResult {i+1}:")
    print(f"  Vector ID: {vector_id}")
    print(f"  Metadata keys: {list(metadata.keys())}")
    print(f"  Metadata: {metadata}")
    
    # Try to get original_chunk_id
    original_id = metadata.get('original_chunk_id', 'NOT_FOUND')
    print(f"  Original Chunk ID: {original_id}")
    
    if original_id != 'NOT_FOUND':
        retrieved_original_ids.append(original_id)
    
    # Check if relevant
    is_relevant = original_id in relevant_chunks
    print(f"  Is Relevant: {is_relevant} {'✅' if is_relevant else '❌'}")
    print(f"  Document: {document}...")

# Compute metrics
print("\n" + "="*80)
print("METRICS")
print("="*80)

relevant_set = set(relevant_chunks)
retrieved_set = set(retrieved_original_ids)
matches = relevant_set.intersection(retrieved_set)

print(f"\nRelevant chunks: {len(relevant_set)}")
print(f"Retrieved chunks: {len(retrieved_original_ids)}")
print(f"Matches: {list(matches)}")
print(f"Precision@5: {len(matches) / len(retrieved_original_ids) if retrieved_original_ids else 0:.3f}")

# Diagnosis
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if len(matches) > 0:
    print(f"✅ SUCCESS: Found {len(matches)} matches!")
    print("   The metadata is being stored and retrieved correctly.")
else:
    print("❌ PROBLEM: No matches found!")
    
    # Check if original_chunk_id is in metadata
    has_original_id = all('original_chunk_id' in m for m in results['metadatas'][0])
    print(f"\n1. original_chunk_id in metadata: {has_original_id}")
    
    if not has_original_id:
        print("   ❌ ISSUE: original_chunk_id not being stored in metadata!")
    else:
        print("   ✅ OK: original_chunk_id is in metadata")
        
        # Check if any relevant chunks were added
        added_chunk_ids = [c['original_chunk_id'] for c in chunks_to_add]
        relevant_in_added = [c for c in relevant_chunks if c in added_chunk_ids]
        print(f"\n2. Relevant chunks in vector store: {len(relevant_in_added)}/{len(relevant_chunks)}")
        
        if len(relevant_in_added) == 0:
            print("   ❌ ISSUE: None of the relevant chunks were added to vector store!")
            print("   This is expected if we only added 100 chunks and relevant chunks are not in first 100.")
        else:
            print("   ✅ OK: Some relevant chunks are in vector store")
            print(f"   Relevant chunks in store: {relevant_in_added}")
            
            # Check if they were retrieved
            print("\n3. Were they retrieved?")
            print(f"   Retrieved IDs: {retrieved_original_ids}")
            print(f"   Expected IDs: {relevant_in_added}")
            
            if not any(r in relevant_in_added for r in retrieved_original_ids):
                print("   ❌ ISSUE: Relevant chunks in store but not retrieved!")
                print("   This suggests embedding quality or semantic mismatch issues.")
            else:
                print("   ✅ OK: Some relevant chunks were retrieved!")

print("\n" + "="*80)

