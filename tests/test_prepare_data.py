import pytest
import json
import tempfile
import yaml
from pathlib import Path
from datasets.prepare_data import create_splits

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
                "chunk_size": 512, "chunk_overlap": 50
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
        
        # Create mock data
        popqa_records = [
            {"query": f"popqa query {i}", "answer": f"ans {i}", "dataset_source": "popqa", "split": "test"} 
            for i in range(50)
        ]
        hotpotqa_records = [
            {"query": f"hotpotqa query {i}", "answer": f"ans {i}", "dataset_source": "hotpotqa", "split": "validation"} 
            for i in range(50)
        ]
        
        # Add a duplicate query
        hotpotqa_records.append({"query": "popqa query 0", "answer": "diff ans", "dataset_source": "hotpotqa", "split": "validation"})
        
        with open(raw_dir / "popqa.jsonl", "w") as f:
            for r in popqa_records:
                f.write(json.dumps(r) + "\n")
                
        with open(raw_dir / "hotpotqa.jsonl", "w") as f:
            for r in hotpotqa_records:
                f.write(json.dumps(r) + "\n")
                
        yield tmp_path, config_path, base_dir

def test_split_generation(temp_setup):
    tmp_path, config_path, base_dir = temp_setup
    
    metadata = create_splits(config_path, base_dir)
    
    splits_dir = base_dir / "splits"
    
    assert (splits_dir / "train.jsonl").exists()
    assert (splits_dir / "val.jsonl").exists()
    assert (splits_dir / "test.jsonl").exists()
    assert (splits_dir / "metadata.json").exists()
    
    assert metadata["input_counts"]["total_raw"] == 101
    assert metadata["input_counts"]["total_deduped"] == 100
    
    # Check counts
    train_count = metadata["output_counts"]["train"]
    val_count = metadata["output_counts"]["val"]
    test_count = metadata["output_counts"]["test"]
    
    assert train_count == 70
    assert val_count == 15
    assert test_count == 15
    
    # Load and verify contents
    seen_queries = set()
    splits_records = {}
    
    for split_name in ["train", "val", "test"]:
        records = []
        with open(splits_dir / f"{split_name}.jsonl", "r") as f:
            for line in f:
                r = json.loads(line)
                records.append(r)
                
                # Verify labels and preservation
                assert r["project_split"] == split_name
                assert r["dataset_source"] in ["popqa", "hotpotqa"]
                assert "split" in r
                
                # Check overlaps
                q = r["query"]
                assert q not in seen_queries
                seen_queries.add(q)
                
        splits_records[split_name] = records
        
def test_determinism(temp_setup):
    tmp_path, config_path, base_dir = temp_setup
    
    create_splits(config_path, base_dir)
    with open(base_dir / "splits" / "train.jsonl", "r") as f:
        run1_train = [json.loads(line) for line in f]
        
    # Run again with same seed
    create_splits(config_path, base_dir)
    with open(base_dir / "splits" / "train.jsonl", "r") as f:
        run2_train = [json.loads(line) for line in f]
        
    assert [r["query"] for r in run1_train] == [r["query"] for r in run2_train]
    
    # Change seed
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["generation"]["seed"] = 999
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
        
    create_splits(config_path, base_dir)
    with open(base_dir / "splits" / "train.jsonl", "r") as f:
        run3_train = [json.loads(line) for line in f]
        
    assert [r["query"] for r in run1_train] != [r["query"] for r in run3_train]

def test_empty_input(temp_setup):
    tmp_path, config_path, base_dir = temp_setup
    
    raw_dir = base_dir / "raw"
    with open(raw_dir / "popqa.jsonl", "w") as f:
        pass
    with open(raw_dir / "hotpotqa.jsonl", "w") as f:
        pass
        
    with pytest.raises(ValueError, match="No input records found"):
        create_splits(config_path, base_dir)
