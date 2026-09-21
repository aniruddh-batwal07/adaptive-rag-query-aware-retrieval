import pytest
from unittest.mock import MagicMock
from src.optimizer.context_optimizer import ContextOptimizer, OptimizedContext
from src.utils.config import Config

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.compression = MagicMock()
    config.compression.budget = 0.5
    return config

def test_successful_compression(mock_config):
    mock_compressor = MagicMock()
    mock_compressor.compress_prompt.return_value = {
        "compressed_prompt": "compressed text",
        "origin_tokens": 100,
        "compressed_tokens": 40
    }
    
    optimizer = ContextOptimizer(config=mock_config, compressor=mock_compressor)
    
    query = "What is the answer?"
    retrieved_context = "This is a very long text " * 20
    
    result = optimizer.compress(query, retrieved_context)
    
    # 1. Successful compression
    assert result.status == "SUCCESS"
    assert result.text == "compressed text"
    
    # 2. Token counts
    assert result.original_tokens == 100
    assert result.compressed_tokens == 40
    assert result.original_tokens > result.compressed_tokens
    
    # 3. Compression ratio
    assert abs(result.compression_ratio - (100 / 40)) < 1e-6
    
    # 4. Compression latency
    assert result.latency_ms > 0
    
    # 5. Query/context handling
    mock_compressor.compress_prompt.assert_called_once_with(
        context=[retrieved_context],
        instruction="",
        question=query,
        rate=0.5,
        rank_method="llmlingua"
    )
    
    # 7. Metadata completeness
    assert isinstance(result, OptimizedContext)

def test_empty_input_handling(mock_config):
    optimizer = ContextOptimizer(config=mock_config, compressor=MagicMock())
    
    result = optimizer.compress("query", "   ")
    
    assert result.text == ""
    assert result.original_tokens == 0
    assert result.compressed_tokens == 0
    assert result.compression_ratio == 1.0
    assert result.latency_ms == 0.0
    assert result.status == "SUCCESS"

def test_invalid_input(mock_config):
    optimizer = ContextOptimizer(config=mock_config, compressor=MagicMock())
    
    with pytest.raises(ValueError):
        optimizer.compress(None, "context")
        
    with pytest.raises(ValueError):
        optimizer.compress("query", None)

def test_deterministic_behavior(mock_config):
    mock_compressor = MagicMock()
    mock_compressor.compress_prompt.return_value = {
        "compressed_prompt": "same output",
        "origin_tokens": 50,
        "compressed_tokens": 25
    }
    optimizer = ContextOptimizer(config=mock_config, compressor=mock_compressor)
    
    res1 = optimizer.compress("q", "c")
    res2 = optimizer.compress("q", "c")
    
    assert res1.text == res2.text
    assert res1.original_tokens == res2.original_tokens
    assert res1.compressed_tokens == res2.compressed_tokens
    assert res1.compression_ratio == res2.compression_ratio

def test_error_propagation(mock_config):
    mock_compressor = MagicMock()
    mock_compressor.compress_prompt.side_effect = RuntimeError("Compression error")
    
    optimizer = ContextOptimizer(config=mock_config, compressor=mock_compressor)
    
    with pytest.raises(RuntimeError, match="Compression error"):
        optimizer.compress("query", "context")

def test_division_by_zero(mock_config):
    mock_compressor = MagicMock()
    mock_compressor.compress_prompt.return_value = {
        "compressed_prompt": "",
        "origin_tokens": 100,
        "compressed_tokens": 0
    }
    
    optimizer = ContextOptimizer(config=mock_config, compressor=mock_compressor)
    result = optimizer.compress("q", "c")
    assert result.compression_ratio == float('inf')
