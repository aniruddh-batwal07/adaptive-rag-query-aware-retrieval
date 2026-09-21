import pytest
import os
import json
from src.evaluation.metrics import (
    exact_match,
    token_f1,
    compression_ratio,
    count_tokens,
    recall_at_k,
    router_metrics,
    normalize_answer
)
from src.evaluation.latency import measure_latency
from src.evaluation.benchmark import run_benchmark, BenchmarkResult

def test_normalize_answer():
    assert normalize_answer("The quick brown Fox.") == "quick brown fox"
    assert normalize_answer("A test, an example!") == "test example"

def test_exact_match():
    assert exact_match("test", "test") == 1.0
    assert exact_match("The test.", "test") == 1.0
    assert exact_match("test", "wrong") == 0.0
    assert exact_match("", "") == 1.0
    assert exact_match("test", "") == 0.0

def test_token_f1():
    assert token_f1("hello world", "hello world") == 1.0
    assert token_f1("hello", "hello world") > 0.6  # 2*(1/1)*(1/2)/(1.5) = 2/3 = 0.66
    assert token_f1("completely different", "hello world") == 0.0
    assert token_f1("", "") == 1.0

def test_compression_ratio():
    assert compression_ratio(100, 50) == 2.0
    assert compression_ratio(0, 50) == 0.0
    assert compression_ratio(100, 0) == float('inf')

def test_count_tokens():
    # fallback test since tokenizer may take a bit, fallback is split
    # Actually count_tokens should just work
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0

def test_recall_at_k():
    assert recall_at_k(["d1", "d2", "d3"], ["d2", "d4"], 3) == 0.5
    assert recall_at_k(["d1", "d2", "d3"], ["d2", "d4"], 1) == 0.0
    assert recall_at_k([], ["d2"], 3) == 0.0
    assert recall_at_k(["d1"], [], 3) != recall_at_k(["d1"], [], 3) # nan

def test_router_metrics():
    preds = ["COMPLEX", "SIMPLE", "COMPLEX", "SIMPLE"]
    labels = ["COMPLEX", "COMPLEX", "SIMPLE", "SIMPLE"]
    metrics = router_metrics(preds, labels)
    assert metrics["accuracy"] == 0.5
    assert metrics["confusion_matrix"]["TP_COMPLEX"] == 1
    assert metrics["confusion_matrix"]["FP_COMPLEX"] == 1
    assert metrics["confusion_matrix"]["FN_COMPLEX"] == 1
    assert metrics["confusion_matrix"]["TN_COMPLEX"] == 1

def test_measure_latency():
    def dummy_func(x):
        return x + 1
    result, lat = measure_latency(dummy_func, 5)
    assert result == 6
    assert lat >= 0.0

class DummyConfig:
    pass

class DummyPipelineMetadata:
    def __init__(self):
        self.error_status = None
        self.retrieval_k = 2
        self.retrieval_latency_ms = 10.0
        self.generation_latency_ms = 20.0
        self.total_latency_ms = 30.0

class DummyPipelineResult:
    def __init__(self, answer):
        self.answer = answer
        self.execution_metadata = DummyPipelineMetadata()

def dummy_pipeline(query: str):
    return DummyPipelineResult("hello world")

def test_run_benchmark(tmp_path):
    examples = [
        {"id": "1", "question": "q1", "answers": ["hello world"]},
        {"id": "2", "question": "q2", "answers": ["different"]}
    ]
    
    # Temporarily change working directory to tmp_path for results/ saving
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        config = DummyConfig()
        result = run_benchmark(dummy_pipeline, examples, config, experiment_name="test_exp")
        
        assert len(result.predictions) == 2
        assert result.metrics["em"] == 0.5
        assert result.metrics["latency"]["generation_mean_ms"] == 20.0
        
        # Check files
        assert os.path.exists("results/test_exp/metrics.json")
        assert os.path.exists("results/test_exp/predictions.jsonl")
        assert os.path.exists("results/test_exp/latency.csv")
    finally:
        os.chdir(old_cwd)
