import time
import json
from typing import List, Dict, Any
from pathlib import Path
import os
import sys

# Ensure 'src' is importable
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.retriever.embeddings import get_default_embedder, Embedder
from src.retriever.vector_store import VectorStore
from src.utils.logger import get_logger

logger = get_logger("retriever")

class Retriever:
    def __init__(self, 
                 vector_store: VectorStore = None, 
                 embedder: Embedder = None,
                 persist_directory: str = None, 
                 collection_name: str = "adaptive_rag_corpus"):
        """
        Initialize the Retriever.
        If vector_store or embedder are not provided, they will be initialized using defaults.
        """
        if persist_directory is None:
            # Default to data/index relative to project root
            project_root = Path(__file__).parent.parent.parent
            persist_directory = str(project_root / "data" / "index")
            
        self.vector_store = vector_store if vector_store is not None else VectorStore(
            persist_directory=persist_directory, 
            collection_name=collection_name
        )
        self.embedder = embedder if embedder is not None else get_default_embedder()

    def retrieve(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Retrieve the top_k most relevant chunks for the given query.
        
        Args:
            query: The natural language search query.
            top_k: The number of results to return.
            
        Returns:
            A list of ranked documents (dictionaries), each containing:
            - document_id
            - chunk_id
            - text
            - score (cosine similarity)
            - rank
            - source_metadata (if available)
        """
        start_time = time.time()
        
        # 1. Validate inputs
        if not isinstance(query, str):
            raise TypeError("Query must be a string.")
        if not query.strip():
            raise ValueError("Query cannot be empty or whitespace only.")
            
        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
            
        # 2. Check vector store availability
        try:
            total_records = self.vector_store.count()
        except Exception as e:
            raise RuntimeError(f"Failed to access vector store: {e}")
            
        if total_records == 0:
            raise RuntimeError("The vector index is empty.")
            
        # Adjust top_k if it exceeds available records
        if top_k > total_records:
            top_k = total_records
            
        # 3. Generate query embedding
        try:
            query_embedding = self.embedder.embed(query)
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}")
            
        # 4. Query vector store
        try:
            raw_results = self.vector_store.query([query_embedding], top_k=top_k)
        except Exception as e:
            logger.error(f"Vector store query failed: {e}")
            raise RuntimeError(f"Retrieval from vector store failed: {e}")
            
        # 5. Process results
        ranked_documents = []
        if not raw_results or not raw_results.get("ids") or not raw_results["ids"][0]:
            return ranked_documents
            
        ids = raw_results["ids"][0]
        distances = raw_results["distances"][0]
        texts = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        
        for i in range(len(ids)):
            # Convert cosine distance back to similarity (similarity = 1 - distance)
            distance = distances[i]
            similarity = 1.0 - distance
            
            raw_meta = metadatas[i]
            document_id = raw_meta.get("document_id", "")
            
            # source_metadata might be stored as a JSON string in chroma
            source_metadata = raw_meta.get("source_metadata", {})
            if isinstance(source_metadata, str):
                try:
                    source_metadata = json.loads(source_metadata)
                except json.JSONDecodeError:
                    pass
            
            doc = {
                "document_id": document_id,
                "chunk_id": ids[i],
                "text": texts[i],
                "score": float(similarity),
                "rank": i + 1,
                "source_metadata": source_metadata
            }
            ranked_documents.append(doc)
            
        latency = (time.time() - start_time) * 1000  # in ms
        logger.info(f"Retrieved {len(ranked_documents)} chunks for query in {latency:.2f}ms")
        
        return ranked_documents

# Default singleton for simple module-level interface
_default_retriever = None

def get_default_retriever():
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = Retriever()
    return _default_retriever

def retrieve(query: str, top_k: int) -> List[Dict[str, Any]]:
    return get_default_retriever().retrieve(query, top_k)
