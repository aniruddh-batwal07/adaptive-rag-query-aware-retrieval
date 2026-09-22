import sys
from unittest.mock import MagicMock

import pytest
import time
from src.utils.config import Config
from src.pipeline.adaptive_rag import Pipeline
from src.retriever.retriever import Retriever

class MockRetriever:
    def __init__(self):
        pass
        
    def retrieve(self, query, top_k=5):
        if query == "fail_retrieval":
            raise Exception("Retrieval error")
        return [{"text": f"Chunk {i}"} for i in range(top_k)]

class MockGenerator:
    def __init__(self):
        pass
        
    def generate(self, query, context):
        if query == "fail_generation":
            raise Exception("Generation error")
        class GA:
            pass
        g = GA()
        g.text = f"Answer for {query} with context length {len(context)}"
        g.generation_latency_ms = 10.0
        return g

class MockCompressor:
    def compress(self, query, context):
        if query == "fail_compression":
            raise Exception("Compression error")
        class OC:
            pass
        o = OC()
        if query == "status_fail_compression":
            o.text = context
            o.original_tokens = 100
            o.compressed_tokens = 100
            o.compression_ratio = 1.0
            o.latency_ms = 5.0
            o.status = "FAILED"
        else:
            o.text = "Compressed context"
            o.original_tokens = 100
            o.compressed_tokens = 50
            o.compression_ratio = 2.0
            o.latency_ms = 5.0
            o.status = "SUCCESS"
        return o

class MockRouter:
    def __init__(self):
        self.mode = "SIMPLE"

    def predict(self, query):
        if query == "fail_router":
            raise Exception("Router error")
        
        if self.mode == "SIMPLE":
            return {"complexity_label": "SIMPLE", "confidence": 0.9}
        else:
            return {"complexity_label": "COMPLEX", "confidence": 0.8}

from src.utils.config import load_config

@pytest.fixture
def mock_config():
    config = load_config("configs/config.yaml")
    config.retrieval.k_simple = 2
    config.retrieval.k_complex = 10
    config.retrieval.baseline_k = 5
    config.routing.fallback_route = "SIMPLE"
    config.models.router.checkpoint = None
    config.runtime.device = "cpu"
    return config

def test_adaptive_simple(mock_config):
    pipeline = Pipeline(mock_config, retriever=MockRetriever(), generator=MockGenerator(), baseline="adaptive", compressor=MockCompressor())
    pipeline.router = MockRouter()
    pipeline.router.mode = "SIMPLE"
    
    result = pipeline.run("test simple")
    
    # Assertions
    assert result.execution_metadata.complexity_label == "SIMPLE"
    assert result.execution_metadata.retrieval_k == 2
    assert result.execution_metadata.compression_applied is False
    assert result.execution_metadata.original_context_tokens is None
    assert result.execution_metadata.error_status is None
    assert len(result.execution_metadata.retrieved_chunks) == 2

def test_adaptive_complex(mock_config):
    pipeline = Pipeline(mock_config, retriever=MockRetriever(), generator=MockGenerator(), baseline="adaptive", compressor=MockCompressor())
    pipeline.router = MockRouter()
    pipeline.router.mode = "COMPLEX"
    
    result = pipeline.run("test complex")
    
    assert result.execution_metadata.complexity_label == "COMPLEX"
    assert result.execution_metadata.retrieval_k == 10
    assert result.execution_metadata.compression_applied is True
    assert result.execution_metadata.compressed_context_tokens == 50
    assert result.execution_metadata.error_status is None
    assert len(result.execution_metadata.retrieved_chunks) == 10

def test_adaptive_router_failure(mock_config):
    pipeline = Pipeline(mock_config, retriever=MockRetriever(), generator=MockGenerator(), baseline="adaptive", compressor=MockCompressor())
    pipeline.router = MockRouter()
    
    result = pipeline.run("fail_router")
    
    # Fallback is SIMPLE
    assert result.execution_metadata.complexity_label == "SIMPLE"
    assert result.execution_metadata.retrieval_k == 2
    assert result.execution_metadata.compression_applied is False
    assert result.execution_metadata.error_status == "router_failed"

def test_adaptive_compression_failure(mock_config):
    pipeline = Pipeline(mock_config, retriever=MockRetriever(), generator=MockGenerator(), baseline="adaptive", compressor=MockCompressor())
    pipeline.router = MockRouter()
    pipeline.router.mode = "COMPLEX"
    
    result = pipeline.run("status_fail_compression")
    
    assert result.execution_metadata.complexity_label == "COMPLEX"
    assert result.execution_metadata.retrieval_k == 10
    assert result.execution_metadata.compression_applied is False
    assert result.execution_metadata.error_status == "compression_failed"

def test_adaptive_compression_exception(mock_config):
    pipeline = Pipeline(mock_config, retriever=MockRetriever(), generator=MockGenerator(), baseline="adaptive", compressor=MockCompressor())
    pipeline.router = MockRouter()
    pipeline.router.mode = "COMPLEX"
    pass

def test_adaptive_retrieval_failure(mock_config):
    pipeline = Pipeline(mock_config, retriever=MockRetriever(), generator=MockGenerator(), baseline="adaptive", compressor=MockCompressor())
    pipeline.router = MockRouter()
    
    result = pipeline.run("fail_retrieval")
    
    assert result.execution_metadata.error_status == "retrieval_failed"
    assert len(result.execution_metadata.retrieved_chunks) == 0
