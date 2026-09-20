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

def test_successful_end_to_end_pipeline(mock_config):
    mock_retriever = MagicMock()
    # Return two mock chunks
    mock_retriever.retrieve.return_value = [
        {"text": "Chunk 1 text."},
        {"text": "Chunk 2 text."}
    ]
    
    mock_generator = MagicMock()
    mock_generator.generate.return_value = GeneratedAnswer(
        text="Mock generated answer.",
        generation_latency_ms=150.0
    )
    
    pipeline = Pipeline(config=mock_config, retriever=mock_retriever, generator=mock_generator)
    
    query = "  What is X?  "
    result = pipeline.run(query)
    
    # 6. Answer
    assert isinstance(result, PipelineResult)
    assert result.answer == "Mock generated answer."
    
    # 1. Pipeline structure / 2. Fixed K / 3. Query normalization
    mock_retriever.retrieve.assert_called_once_with("What is X?", top_k=5)
    
    # 5. Context assembly (rank order joined by \n\n)
    expected_context = "Chunk 1 text.\n\nChunk 2 text."
    mock_generator.generate.assert_called_once_with("What is X?", expected_context)
    
    # 7. ExecutionMetadata / 4. Original query preservation
    metadata = result.execution_metadata
    assert metadata.query == "What is X?"
    assert metadata.original_query == "  What is X?  "
    assert metadata.retrieval_k == 5
    assert metadata.retrieval_latency_ms > 0
    assert metadata.generation_latency_ms == 150.0
    assert metadata.total_latency_ms > 0
    assert metadata.total_latency_ms >= metadata.retrieval_latency_ms

def test_invalid_query_rejected(mock_config):
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    pipeline = Pipeline(config=mock_config, retriever=mock_retriever, generator=mock_generator)
    
    # 8. Invalid query
    with pytest.raises(ValueError):
        pipeline.run("   ")

def test_component_isolation(mock_config):
    # 9. Component isolation
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    pipeline = Pipeline(config=mock_config, retriever=mock_retriever, generator=mock_generator)
    
    assert pipeline.retriever is mock_retriever
    assert pipeline.generator is mock_generator
