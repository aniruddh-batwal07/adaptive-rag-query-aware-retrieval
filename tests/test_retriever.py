import pytest
from unittest.mock import MagicMock
from src.retriever.retriever import Retriever

class MockVectorStore:
    def __init__(self, size=10):
        self._size = size
        self.query_calls = []

    def count(self):
        return self._size
        
    def query(self, query_embeddings, top_k):
        self.query_calls.append((query_embeddings, top_k))
        # Return mock results
        actual_k = min(top_k, self._size)
        if actual_k == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            
        ids = [f"chunk_{i}" for i in range(actual_k)]
        documents = [f"Text for chunk {i}" for i in range(actual_k)]
        metadatas = [{"document_id": f"doc_{i}", "source_metadata": {"foo": "bar"}} for i in range(actual_k)]
        # Distances: lower is better. Distances from 0.1 to 0.1*k
        distances = [0.1 * (i + 1) for i in range(actual_k)]
        
        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances]
        }

class MockEmbedder:
    def __init__(self):
        self.embed_calls = []

    def embed(self, text):
        self.embed_calls.append(text)
        return [0.5, 0.5, 0.5]

@pytest.fixture
def mock_retriever():
    return Retriever(vector_store=MockVectorStore(), embedder=MockEmbedder())

def test_basic_top_k_retrieval(mock_retriever):
    """1. Basic top-K retrieval"""
    results = mock_retriever.retrieve("test query", 2)
    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk_0"
    assert results[1]["chunk_id"] == "chunk_1"

def test_ranking_and_result_schema(mock_retriever):
    """2. Ranking and 3. Result schema"""
    results = mock_retriever.retrieve("test query", 2)
    
    # Check schema
    for i, res in enumerate(results):
        assert "document_id" in res
        assert "chunk_id" in res
        assert "text" in res
        assert "score" in res
        assert "rank" in res
        assert res["rank"] == i + 1
        
    # Check scores (distance was 0.1 and 0.2, so similarities are 0.9 and 0.8)
    assert abs(results[0]["score"] - 0.9) < 1e-5
    assert abs(results[1]["score"] - 0.8) < 1e-5
    assert results[0]["score"] > results[1]["score"]

def test_different_k_values(mock_retriever):
    """4. Different K values"""
    res1 = mock_retriever.retrieve("test query", 1)
    assert len(res1) == 1
    
    res3 = mock_retriever.retrieve("test query", 3)
    assert len(res3) == 3

def test_k_greater_than_corpus_size():
    """5. K greater than corpus size"""
    # Corpus size is 2
    small_store = MockVectorStore(size=2)
    retriever = Retriever(vector_store=small_store, embedder=MockEmbedder())
    
    res = retriever.retrieve("test query", 10)
    assert len(res) == 2
    
def test_invalid_query(mock_retriever):
    """6. Invalid query"""
    with pytest.raises(ValueError, match="empty or whitespace"):
        mock_retriever.retrieve("", 5)
        
    with pytest.raises(ValueError, match="empty or whitespace"):
        mock_retriever.retrieve("   ", 5)
        
    with pytest.raises(TypeError, match="must be a string"):
        mock_retriever.retrieve(None, 5)

def test_invalid_top_k(mock_retriever):
    """7. Invalid top_k"""
    with pytest.raises(ValueError, match="positive integer"):
        mock_retriever.retrieve("query", 0)
        
    with pytest.raises(ValueError, match="positive integer"):
        mock_retriever.retrieve("query", -5)
        
    with pytest.raises(TypeError, match="must be an integer"):
        mock_retriever.retrieve("query", 5.5)

def test_dependency_reuse(mock_retriever):
    """8. Dependency reuse"""
    res = mock_retriever.retrieve("hello", 3)
    
    # Verify Embedder was called
    assert len(mock_retriever.embedder.embed_calls) == 1
    assert mock_retriever.embedder.embed_calls[0] == "hello"
    
    # Verify VectorStore was called
    assert len(mock_retriever.vector_store.query_calls) == 1
    query_embeddings, top_k = mock_retriever.vector_store.query_calls[0]
    assert query_embeddings == [[0.5, 0.5, 0.5]]
    assert top_k == 3

def test_deterministic_ranking(mock_retriever):
    """9. Deterministic ranking"""
    res1 = mock_retriever.retrieve("test", 3)
    res2 = mock_retriever.retrieve("test", 3)
    
    assert res1 == res2
