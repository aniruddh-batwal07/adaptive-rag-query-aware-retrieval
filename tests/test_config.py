import pytest
import yaml
import tempfile
import os
from src.utils.config import load_config

@pytest.fixture
def valid_yaml_content():
    return """
models:
  router:
    name: "microsoft/deberta-v3-small"
    checkpoint: null
  embeddings:
    name: "BAAI/bge-small-en-v1.5"
  compressor:
    name: "llmlingua"
  generator:
    name: "microsoft/Phi-3-mini-4k-instruct"

retrieval:
  k_simple: 2
  k_complex: 10
  baseline_k: 5
  chunk_size: 512
  chunk_overlap: 50

routing:
  threshold: null
  fallback_route: "SIMPLE"

compression:
  enabled: true
  budget: null

generation:
  temperature: 0.1
  max_new_tokens: 256
  seed: 42

evaluation:
  dataset: "mixed"
  split: "test"
  sample_limit: 1000
  split_ratios: [0.7, 0.15, 0.15]

runtime:
  device: "auto"
  batch_size: 1
"""

def test_valid_config_loads(valid_yaml_content):
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write(valid_yaml_content)
        temp_path = f.name
    try:
        config = load_config(temp_path)
        assert config.retrieval.k_simple == 2
        assert config.retrieval.k_complex == 10
        assert config.retrieval.baseline_k == 5
    finally:
        os.remove(temp_path)

def test_missing_required_key(valid_yaml_content):
    data = yaml.safe_load(valid_yaml_content)
    del data['retrieval']['k_simple']
    
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        yaml.dump(data, f)
        temp_path = f.name
    try:
        with pytest.raises(KeyError, match="Missing required field"):
            load_config(temp_path)
    finally:
        os.remove(temp_path)

def test_invalid_type_or_value(valid_yaml_content):
    data = yaml.safe_load(valid_yaml_content)
    data['retrieval']['chunk_size'] = -10
    
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        yaml.dump(data, f)
        temp_path = f.name
    try:
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            load_config(temp_path)
    finally:
        os.remove(temp_path)

def test_chunk_overlap_ge_chunk_size(valid_yaml_content):
    data = yaml.safe_load(valid_yaml_content)
    data['retrieval']['chunk_overlap'] = 512
    data['retrieval']['chunk_size'] = 512
    
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        yaml.dump(data, f)
        temp_path = f.name
    try:
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            load_config(temp_path)
    finally:
        os.remove(temp_path)

def test_malformed_yaml():
    with tempfile.NamedTemporaryFile('w', delete=False) as f:
        f.write("models:\n  router:\n   name: missing_quote\n  unmatched: [")
        temp_path = f.name
    try:
        with pytest.raises(ValueError, match="Malformed YAML"):
            load_config(temp_path)
    finally:
        os.remove(temp_path)
