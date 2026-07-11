from experiments.mlflow.mlflow_config import EVALUATION_DIR

"""
Diagnostic script to test why retrieval metrics are so low.
This will help us understand if the chunk IDs are being stored and retrieved correctly.
"""


import json

from edumind.rag.embedder import Embedder
from edumind.rag.vector_store import VectorStore


def main():
    print("="*80)
    print("RETRIEVAL DIAGNOSIS TEST")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    with open(EVALUATION_DIR / "ground_truth.json") as f:
        ground_truth = json.load(f)
    
    with open(EVALUATION_DIR / "eval_queries.json") as f:
        queries = json.load(f)
    
    print(f"   Loaded {len(ground_truth)} chunks")
    print(f"   Loaded {len(queries)} queries")
    
    # Initialize components
    print("\n2. Initializing embedder and vector store...")
    embedder = Embedder()
    vector_store = VectorStore()
    vector_store.reset_collection()
    
    # Add first 100 chunks to vector store
    print("\n3. Adding chunks to vector store...")
    chunks_to_add = []
    chunk_ids_added = []
    
    for i, (chunk_id, chunk_data) in enumerate(list(ground_truth.items())[:100]):
        chunk = {
            'text': chunk_data['text'],
            'chunk_id': chunk_id,  # Store original ID
            'source': chunk_data.get('source', 'unknown'),
            'page': chunk_data.get('page', 0),
            'domain': chunk_data.get('domain', 'unknown'),
            'topic': chunk_data.get('topic', 'unknown')
        }
        chunks_to_add.append(chunk)
        chunk_ids_added.append(chunk_id)
    
    # Generate embeddings
    chunks_with_embeddings = embedder.embed_chunks(chunks_to_add)
    
    # Add to vector store
    vector_store.add_documents(chunks_with_embeddings)
    print(f"   Added {len(chunks_with_embeddings)} chunks")
    
    # Test retrieval with first query
    print("\n4. Testing retrieval...")
    test_query = queries[0]
    query_text = test_query["query"]
    relevant_chunks = test_query["relevant_chunks"]
    
    print(f"   Query: {query_text}")
    print(f"   Expected relevant chunks: {relevant_chunks[:3]}...")
    
    # Generate query embedding
    query_embedding = embedder.embed_text(query_text)
    
    # Retrieve
    results = vector_store.query(query_embedding.tolist(), top_k=5)
    
    print("\n5. Retrieval Results:")
    print(f"   Retrieved {len(results['ids'][0])} results")
    
    # Check what we got
    retrieved_chunk_ids = []
    for i in range(len(results['ids'][0])):
        vector_id = results['ids'][0][i]
        metadata = results['metadatas'][0][i]
        document = results['documents'][0][i][:100] + "..."
        
        print(f"\n   Result {i+1}:")
        print(f"      Vector ID: {vector_id}")
        print(f"      Metadata: {metadata}")
        print(f"      Document: {document}")
        
        # Try to get chunk_id from metadata
        chunk_id = metadata.get('chunk_id', 'NOT_FOUND')
        retrieved_chunk_ids.append(chunk_id)
        print(f"      Chunk ID: {chunk_id}")
        
        # Check if it's relevant
        is_relevant = chunk_id in relevant_chunks
        print(f"      Is Relevant: {is_relevant}")
    
    # Compute metrics
    print("\n6. Metrics:")
    relevant_set = set(relevant_chunks)
    retrieved_set = set(retrieved_chunk_ids)
    
    matches = relevant_set.intersection(retrieved_set)
    precision = len(matches) / len(retrieved_chunk_ids) if retrieved_chunk_ids else 0
    
    print(f"   Relevant chunks: {len(relevant_set)}")
    print(f"   Retrieved chunks: {len(retrieved_chunk_ids)}")
    print(f"   Matches: {list(matches)}")
    print(f"   Precision@5: {precision:.3f}")
    
    # Diagnosis
    print("\n7. Diagnosis:")
    if precision > 0.5:
        print("   ✅ GOOD: Retrieval is working correctly!")
    elif precision > 0:
        print("   ⚠️  PARTIAL: Some matches found, but precision is low")
        print("      Possible causes:")
        print("      - Not all relevant chunks were added to vector store")
        print("      - Embedding quality issues")
    else:
        print("   ❌ BAD: No matches found!")
        print("      Possible causes:")
        print("      - Chunk IDs not being stored in metadata")
        print("      - Chunk IDs not being retrieved from metadata")
        print("      - Relevant chunks not in vector store")
        
        # Check if relevant chunks were added
        relevant_in_store = [c for c in relevant_chunks if c in chunk_ids_added]
        print(f"      Relevant chunks in vector store: {len(relevant_in_store)}/{len(relevant_chunks)}")
        if len(relevant_in_store) == 0:
            print("      ❌ PROBLEM: None of the relevant chunks were added to vector store!")
        
        # Check if chunk_id is in metadata
        has_chunk_id = all('chunk_id' in m for m in results['metadatas'][0])
        print(f"      Chunk ID in metadata: {has_chunk_id}")
        if not has_chunk_id:
            print("      ❌ PROBLEM: chunk_id not being stored in metadata!")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

