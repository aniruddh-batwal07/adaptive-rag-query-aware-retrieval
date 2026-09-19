import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Ensure 'src' is importable
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.retriever.embeddings import get_default_embedder
from src.retriever.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger("build_index")

def build_index(corpus_path: str = "datasets/corpus/chunks.jsonl", 
                persist_directory: str = "data/index", 
                collection_name: str = "adaptive_rag_corpus",
                batch_size: int = 128):
    """
    Builds the vector index from the chunked corpus.
    """
    if not os.path.exists(corpus_path):
        logger.error(f"Corpus file not found at {corpus_path}")
        return

    logger.info(f"Starting index build from {corpus_path}")
    logger.info(f"Persistence location: {persist_directory}")
    logger.info(f"Collection name: {collection_name}")

    embedder = get_default_embedder()
    embedding_dimension = embedder.dimensionality
    logger.info(f"Embedding dimension: {embedding_dimension}")

    vector_store = VectorStore(persist_directory=persist_directory, collection_name=collection_name)
    
    start_time = time.time()
    
    # Batch processing variables
    chunk_ids = []
    texts = []
    metadatas = []
    
    total_processed = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
                
            record = json.loads(line)
            chunk_id = record["chunk_id"]
            text = record["text"]
            document_id = record["document_id"]
            source_metadata = record.get("source_metadata", {})
            
            # Preserve essential metadata
            metadata = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "source_metadata": source_metadata
            }
            
            chunk_ids.append(chunk_id)
            texts.append(text)
            metadatas.append(metadata)
            
            if len(chunk_ids) >= batch_size:
                # Generate embeddings and insert
                embeddings = embedder.embed_batch(texts)
                vector_store.add_records(chunk_ids, texts, embeddings, metadatas)
                
                total_processed += len(chunk_ids)
                logger.info(f"Processed {total_processed} chunks...")
                
                # Clear batches
                chunk_ids = []
                texts = []
                metadatas = []
                
        # Process remaining
        if chunk_ids:
            embeddings = embedder.embed_batch(texts)
            vector_store.add_records(chunk_ids, texts, embeddings, metadatas)
            total_processed += len(chunk_ids)
            logger.info(f"Processed {total_processed} chunks...")

    build_time = time.time() - start_time
    final_count = vector_store.count()
    
    logger.info("=== Index Build Complete ===")
    logger.info(f"Corpus Path: {corpus_path}")
    logger.info(f"Number of chunks processed: {total_processed}")
    logger.info(f"Embedding dimension: {embedding_dimension}")
    logger.info(f"Index/Collection name: {collection_name}")
    logger.info(f"Persistence location: {persist_directory}")
    logger.info(f"Index build time: {build_time:.2f} seconds")
    logger.info(f"Final indexed record count: {final_count}")

if __name__ == "__main__":
    # We resolve paths relative to the project root
    project_root = Path(parent_dir)
    corpus_file = str(project_root / "datasets" / "corpus" / "chunks.jsonl")
    index_dir = str(project_root / "data" / "index")
    
    build_index(corpus_path=corpus_file, persist_directory=index_dir)
