import pytest
import tempfile
import json
from pathlib import Path
from datasets.fixtures_loader import load_fixtures

def test_load_simple_fixtures():
    data = load_fixtures("simple")
    assert isinstance(data, list)
    assert len(data) >= 10
    for item in data:
        assert "query" in item
        assert "reference_answer" in item
        assert item["complexity_label"] == "SIMPLE"
        assert item["query"].strip() != ""
        assert item["reference_answer"].strip() != ""

def test_load_complex_fixtures():
    data = load_fixtures("complex")
    assert isinstance(data, list)
    assert len(data) >= 10
    for item in data:
        assert "query" in item
        assert "reference_answer" in item
        assert item["complexity_label"] == "COMPLEX"
        assert item["query"].strip() != ""
        assert item["reference_answer"].strip() != ""

def test_determinism():
    data1 = load_fixtures("simple")
    data2 = load_fixtures("simple")
    assert data1 == data2

def test_unknown_fixture_type():
    with pytest.raises(ValueError, match="Unknown fixture_type"):
        load_fixtures("invalid_type")

def test_missing_file():
    with tempfile.TemporaryDirectory() as tmp:
        bdir = Path(tmp)
        with pytest.raises(FileNotFoundError):
            load_fixtures("simple", bdir)

def test_malformed_json_and_schema():
    with tempfile.TemporaryDirectory() as tmp:
        bdir = Path(tmp)
        
        malformed_file = bdir / "simple_queries.json"
        malformed_file.write_text("not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_fixtures("simple", bdir)
            
        malformed_file.write_text(json.dumps([{"query": "a", "complexity_label": "SIMPLE"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="missing required key 'reference_answer'"):
            load_fixtures("simple", bdir)
            
        malformed_file.write_text(json.dumps([{"query": "a", "reference_answer": "b", "complexity_label": "COMPLEX"}]), encoding="utf-8")
        with pytest.raises(ValueError, match="expected 'SIMPLE'"):
            load_fixtures("simple", bdir)
