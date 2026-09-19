import pytest
import json
import tempfile
from datasets.load_dataset import normalize_popqa, normalize_hotpotqa

def test_normalize_popqa():
    raw_record = {
        "question": "What is the capital of Japan?",
        "answers": ["Tokyo", "Tokyo city"]
    }
    norm = normalize_popqa(raw_record)
    
    assert norm["query"] == "What is the capital of Japan?"
    assert norm["answer"] == "Tokyo"
    assert norm["supporting_docs"] is None
    assert norm["dataset_source"] == "popqa"
    assert norm["split"] == "test"

def test_normalize_popqa_leakage(caplog):
    raw_record = {
        "question": "The capital of Japan is Tokyo",
        "answers": ["Tokyo"]
    }
    with caplog.at_level("WARNING"):
        norm = normalize_popqa(raw_record)
        assert "Potential leakage" in caplog.text

def test_normalize_hotpotqa():
    raw_record = {
        "question": "Who was born first, Einstein or Newton?",
        "answer": "Newton",
        "context": {
            "title": ["Albert Einstein", "Isaac Newton"],
            "sentences": [
                ["Einstein was born in 1879."],
                ["Newton was born in 1643."]
            ]
        }
    }
    norm = normalize_hotpotqa(raw_record)
    
    assert norm["query"] == "Who was born first, Einstein or Newton?"
    assert norm["answer"] == "Newton"
    assert norm["dataset_source"] == "hotpotqa"
    assert len(norm["supporting_docs"]) == 2
    assert norm["supporting_docs"][0]["title"] == "Albert Einstein"
    assert "1879" in norm["supporting_docs"][0]["text"]
    assert norm["supporting_docs"][1]["title"] == "Isaac Newton"

def test_normalize_hotpotqa_leakage(caplog):
    raw_record = {
        "question": "Newton was born first.",
        "answer": "Newton"
    }
    with caplog.at_level("WARNING"):
        norm = normalize_hotpotqa(raw_record)
        assert "Potential leakage" in caplog.text

def test_serialization():
    raw_record = {
        "question": "Test?",
        "answers": ["Yes"]
    }
    norm = normalize_popqa(raw_record)
    json_str = json.dumps(norm)
    assert "Test?" in json_str
