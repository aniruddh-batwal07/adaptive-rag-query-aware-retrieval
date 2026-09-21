import pytest
from unittest.mock import MagicMock
from src.pipeline.adaptive_rag import Pipeline, PipelineResult
from src.utils.config import Config
from src.generator.response_generator import GeneratedAnswer

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.retrieval = MagicMock()
    config.retrieval.baseline_k = 5
    return config

def test_fixed_rag_pipeline_execution(mock_config):
    mock_retriever = MagicMock()
    # Mock returning exactly 5 chunks with some metadata
    mock_chunks = [
        {"text": f"Chunk {i} text.", "document_id": f"doc_{i}", "score": 0.9 - (0.1 * i)} 
        for i in range(5)
    ]
    mock_retriever.retrieve.return_value = mock_chunks
    
    mock_generator = MagicMock()
    mock_generator.generate.return_value = GeneratedAnswer(
        text="Mock generated answer.",
        generation_latency_ms=100.0
    )
    
    pipeline = Pipeline(config=mock_config, retriever=mock_retriever, generator=mock_generator, baseline="fixed_rag")
    
    query = "What is X?"
    result = pipeline.run(query)
    
    # 1. Fixed-RAG execution succeeds & 11. Answer is returned as PipelineResult.
    assert isinstance(result, PipelineResult)
    assert result.answer == "Mock generated answer."
    
    # 3. Retriever is called exactly once.
    # 4. The configured baseline K is passed to Retriever.
    # 5. Default K=5 is respected.
    mock_retriever.retrieve.assert_called_once_with("What is X?", top_k=5)
    
    # 8. Retrieved chunks are assembled in rank order.
    # 9. Generator receives the retrieved context.
    expected_context = "\n\n".join([c["text"] for c in mock_chunks])
    mock_generator.generate.assert_called_once_with("What is X?", expected_context)
    
    metadata = result.execution_metadata
    
    # 12. ExecutionMetadata contains retrieval_k.
    assert metadata.retrieval_k == 5
    # 13. Retrieval latency is measured.
    assert metadata.retrieval_latency_ms > 0
    # 14. Generation latency is measured.
    assert metadata.generation_latency_ms == 100.0
    # 15. Total latency is measured.
    assert metadata.total_latency_ms > 0
    
    # 7. exactly 5 results are used.
    # 16. Retrieved chunk metadata is preserved.
    assert len(metadata.retrieved_chunks) == 5
    assert metadata.retrieved_chunks == mock_chunks

def test_fixed_rag_configuration_changes_k(mock_config):
    mock_config.retrieval.baseline_k = 7
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    
    pipeline = Pipeline(config=mock_config, retriever=mock_retriever, generator=mock_generator, baseline="fixed_rag")
    pipeline.run("Query")
    
    # 6. Changing the configuration changes the K passed to Retriever.
    mock_retriever.retrieve.assert_called_once_with("Query", top_k=7)
