import pytest
import os
import torch
from src.router.query_classifier import QueryClassifier
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_classifier():
    # Use a tiny model for fast testing, or mock the HuggingFace components
    # We will just mock the actual predictions for unit testing to avoid downloads in CI
    classifier = QueryClassifier("hf-internal-testing/tiny-random-distilbert", device="cpu")
    return classifier

def test_router_initialization(mock_classifier):
    assert mock_classifier.model_name_or_path == "hf-internal-testing/tiny-random-distilbert"
    assert mock_classifier.device.type == "cpu"
    assert mock_classifier.id2label[0] == "SIMPLE"
    assert mock_classifier.id2label[1] == "COMPLEX"

def test_predict_single(mock_classifier):
    res = mock_classifier.predict("What is the capital of France?")
    assert "complexity_label" in res
    assert "confidence" in res
    assert res["complexity_label"] in ["SIMPLE", "COMPLEX"]
    assert 0.0 <= res["confidence"] <= 1.0

def test_predict_batch(mock_classifier):
    queries = ["Who wrote Hamlet?", "What is the meaning of life, the universe, and everything?"]
    results = mock_classifier.predict_batch(queries)
    assert len(results) == 2
    for res in results:
        assert res["complexity_label"] in ["SIMPLE", "COMPLEX"]
        assert 0.0 <= res["confidence"] <= 1.0

def test_predict_empty(mock_classifier):
    results = mock_classifier.predict_batch([])
    assert results == []

def test_model_save(mock_classifier, tmp_path):
    save_dir = tmp_path / "saved_model"
    mock_classifier.save(str(save_dir))
    
    assert os.path.exists(save_dir)
    assert os.path.exists(save_dir / "config.json")
    
    # Reload and test
    loaded_classifier = QueryClassifier(str(save_dir), device="cpu")
    res = loaded_classifier.predict("Test query")
    assert res["complexity_label"] in ["SIMPLE", "COMPLEX"]
