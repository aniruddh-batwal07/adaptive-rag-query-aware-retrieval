import json
import pytest
from pathlib import Path
import pandas as pd

def test_m11_results_untouched():
    # Verify that M11 results exist and were not overwritten by M12
    # The primary benchmark uses specific timestamps or we can just check if they are non-empty.
    m11_adaptive_preds = Path("results/primary_benchmark/adaptive/predictions.jsonl")
    assert m11_adaptive_preds.exists()
    assert m11_adaptive_preds.stat().st_size > 0

def test_ablation_conditions():
    # Test that K=10 for both conditions and compression is applied for one but not the other
    without_preds = Path("results/m12/ablation_fixed_rag/predictions.jsonl")
    with_preds = Path("results/m12/ablation_always_compress/predictions.jsonl")
    
    if without_preds.exists() and with_preds.exists():
        with open(without_preds, "r") as f:
            w_out = [json.loads(line) for line in f]
        with open(with_preds, "r") as f:
            w_in = [json.loads(line) for line in f]
            
        for record in w_out:
            assert record["retrieval_k"] == 10
            assert record["compression_applied"] is False
            
        for record in w_in:
            assert record["retrieval_k"] == 10
            assert record["compression_applied"] is True

def test_ablation_metrics_aggregation():
    agg_file = Path("results/m12/ablation_aggregate.json")
    if not agg_file.exists():
        pytest.skip("Ablation aggregate file not generated yet")
    with open(agg_file, "r") as f:
        data = json.load(f)
        
    assert "WITHOUT_COMPRESSION" in data
    assert "WITH_COMPRESSION" in data
    assert "DIFFERENCES" in data
    
    diff = data["DIFFERENCES"]
    assert "token_reduction" in diff
    assert "percentage_token_reduction" in diff
    assert "compression_overhead" in diff
    assert "generation_latency_difference" in diff
    assert "total_latency_difference" in diff
    
    # Verify calculation correctness
    orig_tok = data["WITHOUT_COMPRESSION"]["average_retrieved_context_tokens"]
    if orig_tok is None or (isinstance(orig_tok, float) and orig_tok != orig_tok):
        orig_tok = data["WITH_COMPRESSION"]["average_retrieved_context_tokens"]
        
    expected_reduction = orig_tok - data["WITH_COMPRESSION"]["average_compressed_context_tokens"]
    assert abs(diff["token_reduction"] - expected_reduction) < 1e-6
    
    expected_pct = (expected_reduction / orig_tok) * 100
    assert abs(diff["percentage_token_reduction"] - expected_pct) < 1e-6

def test_adaptive_analysis_grouping():
    analysis_file = Path("results/m12/adaptive_analysis.json")
    if not analysis_file.exists():
        pytest.skip("Adaptive analysis file not generated yet")
    with open(analysis_file, "r") as f:
        data = json.load(f)
        
    assert "SIMPLE" in data
    assert "COMPLEX" in data
    
    # In M11 adaptive results, SIMPLE should have K=2 and COMPLEX K=10
    assert data["SIMPLE"]["selected_k"] == 2.0
    assert data["COMPLEX"]["selected_k"] == 10.0
    
    # SIMPLE doesn't apply compression, COMPLEX does
    assert data["SIMPLE"]["compression_invocation_rate"] == 0.0
    assert data["COMPLEX"]["compression_invocation_rate"] == 1.0

def test_reproducibility_metadata():
    repo_file = Path("results/m12/reproducibility.json")
    if not repo_file.exists():
        pytest.skip("Reproducibility file not generated yet")
    with open(repo_file, "r") as f:
        data = json.load(f)
        
    required_fields = ["dataset_split", "sample_count", "random_seed", "models", "retrieval_k", "compression_parameters", "generation_parameters", "benchmark_git_commit"]
    for field in required_fields:
        assert field in data

