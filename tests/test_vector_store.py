import pytest
import tempfile
import os
import json
from src.retriever.vector_store import VectorStore

@pytest.fixture
def temp_persist_dir():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        yield temp_dir

def test_insert_and_retrieve_chunks(temp_persist_dir):
    """A. Insert 5 known chunks & D. Record count"""
    store = VectorStore(persist_directory=temp_persist_dir, collection_name="test_collection")
    
    chunk_ids = [f"chunk_{i}" for i in range(5)]
    texts = [f"This is test chunk number {i}" for i in range(5)]
    # Use dummy embeddings of dimension 3 for testing
    embeddings = [[0.1, 0.2, float(i)] for i in range(5)]
    metadatas = [{"document_id": f"doc_{i}", "source": "test"} for i in range(5)]
    
    store.add_records(chunk_ids, texts, embeddings, metadatas)
    
    assert store.count() == 5
    
    # Retrieve one to verify it exists
    record = store.get_record("chunk_2")
    assert record is not None
    assert record["chunk_id"] == "chunk_2"
    assert record["text"] == "This is test chunk number 2"

def test_metadata_preservation(temp_persist_dir):
    """B. Metadata preservation"""
    store = VectorStore(persist_directory=temp_persist_dir, collection_name="test_collection")
    
    source_metadata = {"dataset": "popqa", "split": "train", "nested": {"key": "value"}}
    metadatas = [{"document_id": "doc_1", "chunk_id": "chunk_1", "source_metadata": source_metadata}]
    
    store.add_records(["chunk_1"], ["text"], [[0.1, 0.2, 0.3]], metadatas)
    
    record = store.get_record("chunk_1")
    assert record is not None
    assert record["metadata"]["document_id"] == "doc_1"
    
    # source_metadata was dict, so it was json dumped
    recovered_source = json.loads(record["metadata"]["source_metadata"])
    assert recovered_source["dataset"] == "popqa"
    assert recovered_source["nested"]["key"] == "value"

def test_persistence_and_reload(temp_persist_dir):
    """C. Persistence/reload"""
    # 1. Build and close
    store1 = VectorStore(persist_directory=temp_persist_dir, collection_name="test_collection")
    store1.add_records(
        ["chunk_1"], 
        ["persistent text"], 
        [[0.5, 0.5, 0.5]], 
        [{"document_id": "doc_1"}]
    )
    assert store1.count() == 1
    
    # Force delete reference to close the SQLite connection Chroma uses
    del store1 
    
    # 2. Reopen and verify
    store2 = VectorStore(persist_directory=temp_persist_dir, collection_name="test_collection")
    assert store2.count() == 1
    
    record = store2.get_record("chunk_1")
    assert record is not None
    assert record["text"] == "persistent text"

def test_deterministic_behavior(temp_persist_dir):
    """E. Deterministic indexing behavior"""
    # Same inputs should result in same internal state and overwrite safely.
    store = VectorStore(persist_directory=temp_persist_dir, collection_name="test_collection")
    
    chunk_ids = ["chunk_1", "chunk_2"]
    texts = ["text1", "text2"]
    embeddings = [[0.1, 0.1], [0.2, 0.2]]
    metadatas = [{"document_id": "d1"}, {"document_id": "d2"}]
    
    # Insert first time
    store.add_records(chunk_ids, texts, embeddings, metadatas)
    assert store.count() == 2
    
    # Insert same items again (Chroma upserts by default)
    store.add_records(chunk_ids, texts, embeddings, metadatas)
    assert store.count() == 2

def test_error_handling(temp_persist_dir):
    """F. Error handling"""
    store = VectorStore(persist_directory=temp_persist_dir, collection_name="test_collection")
    
    # Empty input should not crash
    store.add_records([], [], [], [])
    assert store.count() == 0
    
    # Invalid retrieval should return None
    assert store.get_record("nonexistent_chunk") is None
