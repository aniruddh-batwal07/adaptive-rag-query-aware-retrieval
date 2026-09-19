import sys
from unittest.mock import MagicMock, patch
import pytest
import tempfile
import yaml
from pathlib import Path

# Inject a mock for sentence_transformers to avoid importing it during tests
mock_sentence_transformers = MagicMock()
mock_SentenceTransformer = MagicMock()
mock_sentence_transformers.SentenceTransformer = mock_SentenceTransformer
sys.modules["sentence_transformers"] = mock_sentence_transformers

from src.retriever.embeddings import Embedder

@pytest.fixture
def temp_config():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        config_data = {
            "models": {
                "router": {"name": "test"},
                "embeddings": {"name": "BAAI/test-model"},
                "compressor": {"name": "test"},
                "generator": {"name": "test"}
            },
            "retrieval": {
                "k_simple": 2, "k_complex": 10, "baseline_k": 5,
                "chunk_size": 10, "chunk_overlap": 2
            },
            "routing": {"fallback_route": "SIMPLE"},
            "compression": {"enabled": False},
            "generation": {"temperature": 0.1, "max_new_tokens": 256, "seed": 42},
            "evaluation": {
                "dataset": "mixed", "split": "test", "sample_limit": 100,
                "split_ratios": [0.7, 0.15, 0.15]
            },
            "runtime": {"device": "cpu", "batch_size": 1}
        }
        
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
            
        yield str(config_path)

def test_embed_single(temp_config):
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [0.1, 0.2, 0.3])
    mock_SentenceTransformer.return_value = mock_model
    
    embedder = Embedder(temp_config)
    
    assert embedder.model_name == "BAAI/test-model"
    assert embedder.device == "cpu"
    
    # Test valid input
    vec = embedder.embed("Hello world")
    assert vec == [0.1, 0.2, 0.3]
    mock_model.encode.assert_called_once_with("Hello world", convert_to_numpy=True, show_progress_bar=False)
    
    # Test invalid input
    with pytest.raises(ValueError):
        embedder.embed("")
        
    with pytest.raises(ValueError):
        embedder.embed("   ")

def test_embed_batch(temp_config):
    mock_model = MagicMock()
    mock_model.encode.return_value = MagicMock(tolist=lambda: [[0.1], [0.2]])
    mock_SentenceTransformer.return_value = mock_model
    
    embedder = Embedder(temp_config)
    
    vecs = embedder.embed_batch(["text1", "text2"])
    assert len(vecs) == 2
    assert vecs == [[0.1], [0.2]]
    mock_model.encode.assert_called_once_with(["text1", "text2"], convert_to_numpy=True, show_progress_bar=False)
    
    assert embedder.embed_batch([]) == []
    
    with pytest.raises(ValueError):
        embedder.embed_batch(["valid", ""])

@patch("src.retriever.embeddings.torch")
def test_device_cuda_unavailable(mock_torch, temp_config):
    # Set config to cuda
    with open(temp_config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["runtime"]["device"] = "cuda"
    with open(temp_config, "w") as f:
        yaml.dump(cfg, f)
        
    mock_torch.cuda.is_available.return_value = False
    embedder = Embedder(temp_config)
    
    with pytest.raises(RuntimeError, match="CUDA was requested but is not available"):
        embedder._load()
