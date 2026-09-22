import pytest
from unittest.mock import MagicMock
from src.pipeline.adaptive_rag import Pipeline, PipelineResult
from src.utils.config import Config
from src.generator.response_generator import GeneratedAnswer
from src.optimizer.context_optimizer import OptimizedContext

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.retrieval = MagicMock()
    config.retrieval.k_complex = 10
    config.retrieval.baseline_k = 5
    return config

def test_always_compress_pipeline_execution(mock_config):
    mock_retriever = MagicMock()
    mock_chunks = [
        {"text": f"Chunk {i} text.", "document_id": f"doc_{i}", "score": 0.9 - (0.05 * i)} 
        for i in range(10)
    ]
    mock_retriever.retrieve.return_value = mock_chunks
    
    mock_compressor = MagicMock()
    mock_compressor.compress.return_value = OptimizedContext(
        text="Compressed context",
        original_tokens=100,
        compressed_tokens=40,
        compression_ratio=2.5,
        latency_ms=50.0,
        status="SUCCESS"
    )
    
    mock_generator = MagicMock()
    mock_generator.generate.return_value = GeneratedAnswer(
        text="Mock generated answer.",
        generation_latency_ms=100.0
    )
    
    pipeline = Pipeline(config=mock_config, retriever=mock_retriever, generator=mock_generator, baseline="always_compress", compressor=mock_compressor)
    
    query = "What is X?"
    result = pipeline.run(query)
    
    assert isinstance(result, PipelineResult)
    assert result.answer == "Mock generated answer."
    
    # 1. K=10 should be requested
    mock_retriever.retrieve.assert_called_once_with("What is X?", top_k=10)
    
    expected_original_context = [c["text"] for c in mock_chunks]
    
    # 2. Compressor is invoked
    mock_compressor.compress.assert_called_once_with("What is X?", expected_original_context)
    
    # 3. Generator receives compressed context
    mock_generator.generate.assert_called_once_with("What is X?", "Compressed context")
    
    metadata = result.execution_metadata
    
    # 4. Metadata includes compression metrics
    assert metadata.retrieval_k == 10
    assert metadata.compression_applied is True
    assert metadata.original_context_tokens == 100
    assert metadata.compressed_context_tokens == 40
    assert metadata.compression_ratio == 2.5
    assert metadata.compression_latency_ms == 50.0
    assert metadata.error_status is None
    
def test_compression_failure_fallback_in_pipeline(mock_config):
    mock_retriever = MagicMock()
    mock_chunks = [
        {"text": f"Chunk {i} text."} for i in range(10)
    ]
    mock_retriever.retrieve.return_value = mock_chunks
    
    mock_compressor = MagicMock()
    expected_original_context = "\n\n".join([c["text"] for c in mock_chunks])
    # Simulate a FAILED response from ContextOptimizer
    mock_compressor.compress.return_value = OptimizedContext(
        text=expected_original_context,
        original_tokens=len(expected_original_context.split()),
        compressed_tokens=None,
        compression_ratio=None,
        latency_ms=20.0,
        status="FAILED"
    )
    
    mock_generator = MagicMock()
    mock_generator.generate.return_value = GeneratedAnswer(
        text="Mock generated answer.",
        generation_latency_ms=100.0
    )
    
    pipeline = Pipeline(config=mock_config, retriever=mock_retriever, generator=mock_generator, baseline="always_compress", compressor=mock_compressor)
    
    result = pipeline.run("Test query")
    
    # 1. Compressor was called
    mock_compressor.compress.assert_called_once()
    
    # 2. Generator receives original uncompressed context due to fallback
    mock_generator.generate.assert_called_once_with("Test query", expected_original_context)
    
    # 3. Metadata records the failure
    metadata = result.execution_metadata
    assert metadata.compression_applied is False
    assert metadata.error_status == "compression_failed"
    assert metadata.compressed_context_tokens is None
