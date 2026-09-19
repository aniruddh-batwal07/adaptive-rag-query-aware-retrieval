import os
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
import json

class VectorStore:
    def __init__(self, persist_directory: str, collection_name: str = "adaptive_rag_corpus"):
        """
        Initializes the VectorStore with ChromaDB.
        
        Args:
            persist_directory: Path to the directory where the Chroma index should be persisted.
            collection_name: Name of the ChromaDB collection.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Ensure the persistence directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the persistent client
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # Get or create the collection
        # We don't provide an embedding function here because we'll pass embeddings explicitly
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"} # Use cosine similarity
        )

    def add_records(self, chunk_ids: List[str], texts: List[str], embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        """
        Adds records and their embeddings to the vector store.
        
        Args:
            chunk_ids: List of unique chunk identifiers.
            texts: List of text chunks.
            embeddings: List of embedding vectors for the texts.
            metadatas: List of metadata dictionaries (must include document_id).
        """
        if not chunk_ids:
            return
            
        # Serialize nested metadata if necessary, Chroma only supports str, int, float, bool
        processed_metadatas = []
        for meta in metadatas:
            processed_meta = {}
            for k, v in meta.items():
                if isinstance(v, (str, int, float, bool)):
                    processed_meta[k] = v
                else:
                    processed_meta[k] = json.dumps(v)
            processed_metadatas.append(processed_meta)
            
        self.collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=processed_metadatas
        )

    def get_record(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a specific record by its chunk_id.
        Useful for verification.
        """
        result = self.collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas", "embeddings"]
        )
        
        if not result or not result["ids"]:
            return None
            
        return {
            "chunk_id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0],
            "embedding": result["embeddings"][0] if result.get("embeddings") is not None and len(result["embeddings"]) > 0 else None
        }

    def count(self) -> int:
        """
        Returns the number of items in the collection.
        """
        return self.collection.count()
