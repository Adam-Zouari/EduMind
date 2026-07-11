from experiments.mlflow.mlflow_config import EVALUATION_DIR

"""
Simple diagnostic to check if the data is structured correctly.
No dependencies on torch or sentence-transformers.
"""

import json
from collections import defaultdict


def main():
    print("="*80)
    print("DATA STRUCTURE DIAGNOSIS")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    with open(EVALUATION_DIR / "ground_truth.json") as f:
        ground_truth = json.load(f)
    
    with open(EVALUATION_DIR / "eval_queries.json") as f:
        queries = json.load(f)
    
    print(f"   Loaded {len(ground_truth)} chunks")
    print(f"   Loaded {len(queries)} queries")
    
    # Check ground truth structure
    print("\n2. Ground Truth Structure:")
    sample_chunk_id = list(ground_truth.keys())[0]
    sample_chunk = ground_truth[sample_chunk_id]
    print(f"   Sample Chunk ID: {sample_chunk_id}")
    print(f"   Sample Chunk Keys: {list(sample_chunk.keys())}")
    print(f"   Sample Chunk Domain: {sample_chunk.get('domain', 'N/A')}")
    print(f"   Sample Chunk Topic: {sample_chunk.get('topic', 'N/A')}")
    print(f"   Sample Chunk Variant: {sample_chunk.get('variant', 'N/A')}")
    print(f"   Sample Chunk Text (first 100 chars): {sample_chunk['text'][:100]}...")
    
    # Check query structure
    print("\n3. Query Structure:")
    sample_query = queries[0]
    print(f"   Sample Query: {sample_query['query']}")
    print(f"   Sample Query Keys: {list(sample_query.keys())}")
    print(f"   Relevant Chunks: {sample_query['relevant_chunks'][:3]}...")
    print(f"   Number of Relevant Chunks: {len(sample_query['relevant_chunks'])}")
    print(f"   Domain: {sample_query.get('domain', 'N/A')}")
    print(f"   Expected Variant: {sample_query.get('expected_variant', 'N/A')}")
    
    # Check if relevant chunks exist in ground truth
    print("\n4. Checking Relevant Chunks Existence:")
    missing_chunks = []
    for chunk_id in sample_query['relevant_chunks']:
        if chunk_id not in ground_truth:
            missing_chunks.append(chunk_id)
    
    if missing_chunks:
        print(f"   ❌ PROBLEM: {len(missing_chunks)} relevant chunks NOT in ground truth!")
        print(f"   Missing: {missing_chunks[:5]}...")
    else:
        print("   ✅ GOOD: All relevant chunks exist in ground truth")
    
    # Check domain/topic/variant matching
    print("\n5. Checking Domain/Topic/Variant Matching:")
    query_domain = sample_query.get('domain')
    query_variant = sample_query.get('expected_variant')
    
    # Get topic from first relevant chunk
    if sample_query['relevant_chunks']:
        first_relevant_id = sample_query['relevant_chunks'][0]
        if first_relevant_id in ground_truth:
            first_relevant_chunk = ground_truth[first_relevant_id]
            query_topic = first_relevant_chunk.get('topic')
            
            print(f"   Query Domain: {query_domain}")
            print(f"   Query Topic: {query_topic}")
            print(f"   Query Variant: {query_variant}")
            
            # Check how many chunks match this (domain, topic, variant)
            matching_chunks = []
            for chunk_id, chunk_data in ground_truth.items():
                if (chunk_data.get('domain') == query_domain and
                    chunk_data.get('topic') == query_topic and
                    chunk_data.get('variant') == query_variant):
                    matching_chunks.append(chunk_id)
            
            print(f"   Chunks with matching (domain, topic, variant): {len(matching_chunks)}")
            print(f"   Sample matching chunk IDs: {matching_chunks[:5]}...")
            
            # Check overlap with relevant chunks
            relevant_set = set(sample_query['relevant_chunks'])
            matching_set = set(matching_chunks)
            overlap = relevant_set.intersection(matching_set)
            
            print(f"   Overlap with relevant chunks: {len(overlap)}/{len(relevant_set)}")
            if len(overlap) < len(relevant_set):
                print("   ⚠️  WARNING: Not all relevant chunks match (domain, topic, variant)")
    
    # Statistics across all queries
    print("\n6. Overall Statistics:")
    total_relevant = sum(len(q['relevant_chunks']) for q in queries)
    avg_relevant = total_relevant / len(queries)
    print(f"   Total relevant chunks across all queries: {total_relevant}")
    print(f"   Average relevant chunks per query: {avg_relevant:.2f}")
    
    # Check how many queries have relevant chunks in ground truth
    queries_with_all_relevant = 0
    queries_with_some_relevant = 0
    queries_with_no_relevant = 0
    
    for query in queries:
        relevant_in_gt = sum(1 for c in query['relevant_chunks'] if c in ground_truth)
        if relevant_in_gt == len(query['relevant_chunks']):
            queries_with_all_relevant += 1
        elif relevant_in_gt > 0:
            queries_with_some_relevant += 1
        else:
            queries_with_no_relevant += 1
    
    print(f"   Queries with ALL relevant chunks in GT: {queries_with_all_relevant}/{len(queries)}")
    print(f"   Queries with SOME relevant chunks in GT: {queries_with_some_relevant}/{len(queries)}")
    print(f"   Queries with NO relevant chunks in GT: {queries_with_no_relevant}/{len(queries)}")
    
    # Domain/Topic distribution
    print("\n7. Domain/Topic Distribution:")
    domain_counts = defaultdict(int)
    topic_counts = defaultdict(int)
    variant_counts = defaultdict(int)
    
    for chunk_data in ground_truth.values():
        domain_counts[chunk_data.get('domain', 'unknown')] += 1
        topic_counts[chunk_data.get('topic', 'unknown')] += 1
        variant_counts[chunk_data.get('variant', 'unknown')] += 1
    
    print(f"   Domains: {dict(domain_counts)}")
    print(f"   Variants: {dict(variant_counts)}")
    print(f"   Number of unique topics: {len(topic_counts)}")
    
    # Final diagnosis
    print("\n8. DIAGNOSIS:")
    if queries_with_no_relevant > 0:
        print(f"   ❌ CRITICAL: {queries_with_no_relevant} queries have NO relevant chunks in ground truth!")
        print("      This will cause 0% precision for those queries.")
    elif queries_with_some_relevant > 0:
        print(f"   ⚠️  WARNING: {queries_with_some_relevant} queries have SOME missing relevant chunks")
        print("      This will reduce precision.")
    else:
        print("   ✅ GOOD: All queries have all relevant chunks in ground truth")
        print("      The low metrics are likely due to:")
        print("      - Embedding quality issues")
        print("      - Chunk ID mapping issues (re-chunking problem)")
        print("      - Vector store not returning correct metadata")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

