import os
import sys

# Ensure 'src' is importable
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.retriever.retriever import retrieve

def verify_real_integration():
    print("=== Starting Real Integration Verification ===")
    
    # 1. Intuitively relevant query test
    query1 = "Superman film directed by Bryan Singer"
    print(f"\nQuery: '{query1}' (K=1)")
    res1 = retrieve(query1, top_k=1)
    
    print(f"Results returned: {len(res1)}")
    if len(res1) > 0:
        print(f"Rank 1 Score: {res1[0]['score']}")
        print(f"Text snippet: {res1[0]['text'][:200]}...")
        assert "document_id" in res1[0]
        assert "chunk_id" in res1[0]
        assert "text" in res1[0]
    
    # 2. Test K=2
    query2 = "What are some popular American science fiction films?"
    print(f"\nQuery: '{query2}' (K=2)")
    res2 = retrieve(query2, top_k=2)
    print(f"Results returned: {len(res2)}")
    assert len(res2) == 2
    assert res2[0]['score'] >= res2[1]['score']  # Ordered by relevance
    
    # 3. Test K=5
    query3 = "Information about wildlife and nature parks"
    print(f"\nQuery: '{query3}' (K=5)")
    res3 = retrieve(query3, top_k=5)
    print(f"Results returned: {len(res3)}")
    assert len(res3) == 5
    
    print("\n=== Verification Completed Successfully ===")

if __name__ == "__main__":
    verify_real_integration()
