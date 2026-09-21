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

def test_llm_only_pipeline_execution(mock_config):
    mock_generator = MagicMock()
    mock_generator.generate.return_value = GeneratedAnswer(
        text="Mock generated answer.",
        generation_latency_ms=100.0
    )
    
    # Do not pass a retriever, and verify it's not initialized internally either
    pipeline = Pipeline(config=mock_config, generator=mock_generator, baseline="llm_only")
    
    # 4, 5, 6: Retriever is NOT called, VectorStore is NOT called, Embedder is NOT called.
    assert pipeline.retriever is None
    
    query = "  What is X?  "
    result = pipeline.run(query)
    
    # 1. LLM-only execution succeeds.
    assert isinstance(result, PipelineResult)
    # 9. Generator output becomes pipeline answer.
    assert result.answer == "Mock generated answer."
    
    # 2. Preprocessor is called (implied by query normalization).
    # 3. Generator is called.
    # 8. Empty/minimal context is passed to Generator.
    mock_generator.generate.assert_called_once_with("What is X?", "")
    
    metadata = result.execution_metadata
    
    # 7. retrieval_k == 0.
    assert metadata.retrieval_k == 0
    # 10. generation_latency_ms is populated.
    assert metadata.generation_latency_ms == 100.0
    # 11. total_latency_ms is populated.
    assert metadata.total_latency_ms > 0
    # 12. No fake retrieval latency is reported.
    assert metadata.retrieval_latency_ms is None
