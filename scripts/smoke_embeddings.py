import time
import json
import numpy as np
import sys
import os
from pathlib import Path

# Ensure 'src' is importable if run as a script directly
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.retriever.embeddings import get_default_embedder

def cosine_similarity(v1, v2):
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def run_smoke_test():
    print("--- Starting Embedding Smoke Test ---")
    embedder = get_default_embedder()
    
    # 1. Measure model load time
    start_load = time.time()
    embedder._load()
    load_time = time.time() - start_load
    
    dim = embedder.dimensionality
    print(f"Model: {embedder.model_name}")
    print(f"Device actually used: {embedder._model.device}")
    print(f"Embedding dimensionality: {dim}")
    print(f"Model load time: {load_time:.2f} seconds")
    
    # 2. Single-text inference latency
    text_single = "This is a simple test sentence."
    start_single = time.time()
    vec = embedder.embed(text_single)
    single_time = time.time() - start_single
    
    print(f"Single encoding returned vector of length {len(vec)}")
    print(f"Single-text inference latency: {single_time:.4f} seconds")
    
    assert isinstance(vec[0], float), "Vector elements must be numeric."
    
    # 3. Batch encoding
    batch = ["First sentence", "Second sentence", "Third sentence"]
    start_batch = time.time()
    vecs = embedder.embed_batch(batch)
    batch_time = time.time() - start_batch
    
    print(f"Batch encoding returned {len(vecs)} vectors.")
    print(f"Batch inference latency (size 3): {batch_time:.4f} seconds")
    
    assert len(vecs) == len(batch), "Batch size mismatch."
    assert len(vecs[0]) == dim, "Dimensionality mismatch."
    
    # 4. Semantic sanity check
    text_related_1 = "The cat sat on the mat."
    text_related_2 = "A feline is resting on the rug."
    text_unrelated = "Quantum physics explores the behavior of subatomic particles."
    
    v_r1 = embedder.embed(text_related_1)
    v_r2 = embedder.embed(text_related_2)
    v_u = embedder.embed(text_unrelated)
    
    sim_related = cosine_similarity(v_r1, v_r2)
    sim_unrelated = cosine_similarity(v_r1, v_u)
    
    print(f"Semantic Check: Related similarity = {sim_related:.4f}")
    print(f"Semantic Check: Unrelated similarity = {sim_unrelated:.4f}")
    
    if sim_related > sim_unrelated:
        print("Semantic sanity check: PASS")
    else:
        print("Semantic sanity check: FAIL")
        
    # 5. Corpus integration check
    corpus_path = Path(parent_dir) / "datasets" / "corpus" / "chunks.jsonl"
    if not corpus_path.exists():
        print(f"Corpus not found at {corpus_path}")
        return
        
    chunks = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for _ in range(5):
            line = f.readline()
            if not line:
                break
            chunks.append(json.loads(line)["text"])
            
    print(f"\nEncoding {len(chunks)} real corpus chunks...")
    vecs_chunks = embedder.embed_batch(chunks)
    print("Corpus chunk encoding: PASS")
    print("All smoke tests completed successfully.")

if __name__ == "__main__":
    run_smoke_test()
