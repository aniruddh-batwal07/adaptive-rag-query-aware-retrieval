import pytest
import json
import tempfile
import yaml
from pathlib import Path
from datasets.build_corpus import build_corpus, chunk_text, get_stable_id

def test_chunk_text_short():
    text = "word1 word2 word3"
    chunks = chunk_text(text, 5, 2)
    assert len(chunks) == 1
    assert chunks[0] == "word1 word2 word3"

def test_chunk_text_long_overlap():
    text = "w1 w2 w3 w4 w5 w6"
    chunks = chunk_text(text, 4, 2)
    # step is 4 - 2 = 2
    # chunk 0: w1 w2 w3 w4
    # chunk 1: w3 w4 w5 w6
    assert len(chunks) == 2
    assert chunks[0] == "w1 w2 w3 w4"
    assert chunks[1] == "w3 w4 w5 w6"

def test_chunk_text_empty():
    assert chunk_text("   ", 5, 2) == []
    assert chunk_text("", 5, 2) == []

@pytest.fixture
def temp_setup():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Create a mock config
        config_data = {
            "models": {
                "router": {"name": "test"},
                "embeddings": {"name": "test"},
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
            "runtime": {"device": "auto", "batch_size": 1}
        }
        
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        config_path = config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
            
        base_dir = tmp_path / "datasets"
        raw_dir = base_dir / "raw"
        raw_dir.mkdir(parents=True)
        
        long_text = " ".join(f"word{i}" for i in range(20))
        # Create mock data
        hotpotqa_records = [
            {
                "query": "hotpotqa query 1", 
                "answer": "ans 1", 
                "dataset_source": "hotpotqa", 
                "split": "validation",
                "supporting_docs": [
                    {"title": "Doc 1", "text": "This is a short doc."},
                    {"title": "Doc 2", "text": long_text},
                    {"title": "Doc 3", "text": ""} # Empty doc
                ]
            },
            {
                "query": "hotpotqa query 2", 
                "answer": "ans 2", 
                "dataset_source": "hotpotqa", 
                "split": "validation",
                "supporting_docs": [
                    {"title": "Doc 1", "text": "This is a short doc."} # Duplicate doc
                ]
            }
        ]
        
        popqa_records = [
            {
                "query": "popqa query 1",
                "answer": "ans popqa",
                "dataset_source": "popqa",
                "split": "test",
                "supporting_docs": None
            }
        ]
        
        with open(raw_dir / "popqa.jsonl", "w") as f:
            for r in popqa_records:
                f.write(json.dumps(r) + "\n")
                
        with open(raw_dir / "hotpotqa.jsonl", "w") as f:
            for r in hotpotqa_records:
                f.write(json.dumps(r) + "\n")
                
        yield tmp_path, config_path, base_dir

def test_build_corpus(temp_setup):
    tmp_path, config_path, base_dir = temp_setup
    
    stats = build_corpus(config_path, base_dir)
    
    # 4 source docs total: [Doc 1, Doc 2, Doc 3 (empty)] in query 1, and [Doc 1] in query 2
    # Popqa has 0 source docs
    assert stats["hotpotqa"]["source_docs"] == 4
    
    # Unique non-empty docs: Doc 1, Doc 2 (2 unique)
    assert stats["hotpotqa"]["unique_docs"] == 2
    assert stats["total"]["unique_docs"] == 2
    
    # Doc 1 chunk count: 1 (5 words < 10)
    # Doc 2 chunk count: 20 words, chunk_size=10, overlap=2, step=8
    #   i=0: words 0..9
    #   i=8: words 8..17
    #   i=16: words 16..19
    # Total chunks = 3
    # Total overall chunks = 4
    assert stats["total"]["chunks"] == 4
    
    corpus_file = base_dir / "corpus" / "chunks.jsonl"
    assert corpus_file.exists()
    
    chunks = []
    with open(corpus_file, "r") as f:
        for line in f:
            chunks.append(json.loads(line))
            
    assert len(chunks) == 4
    
    for c in chunks:
        assert "document_id" in c
        assert "chunk_id" in c
        assert "text" in c
        assert "source_metadata" in c
        
        assert "hotpotqa query 1" not in c["text"] # no query injection
        assert "ans 1" not in c["text"] # no answer injection
        
        assert c["source_metadata"]["dataset_source"] == "hotpotqa"
        assert "title" in c["source_metadata"]
        assert "source_split" in c["source_metadata"]

def test_determinism(temp_setup):
    tmp_path, config_path, base_dir = temp_setup
    build_corpus(config_path, base_dir)
    
    corpus_file = base_dir / "corpus" / "chunks.jsonl"
    with open(corpus_file, "r") as f:
        run1_chunks = f.read()
        
    build_corpus(config_path, base_dir)
    with open(corpus_file, "r") as f:
        run2_chunks = f.read()
        
    assert run1_chunks == run2_chunks
