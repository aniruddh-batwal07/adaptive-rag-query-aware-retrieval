import json
from pathlib import Path
from typing import List, Dict, Any, Optional

def load_fixtures(fixture_type: str, base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """
    Loads local JSON fixtures for development.
    fixture_type should be 'simple' or 'complex'.
    """
    if fixture_type not in ["simple", "complex"]:
        raise ValueError(f"Unknown fixture_type: {fixture_type}. Must be 'simple' or 'complex'.")
        
    if base_dir is None:
        base_dir = Path(__file__).parent / "fixtures"
        
    file_path = base_dir / f"{fixture_type}_queries.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {file_path}")
        
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON in fixture {file_path}: {e}")
        
    if not isinstance(data, list):
        raise ValueError("Fixture data must be a JSON array (list).")
        
    expected_label = "SIMPLE" if fixture_type == "simple" else "COMPLEX"
    
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} is not a JSON object.")
        for key in ["query", "reference_answer", "complexity_label"]:
            if key not in item:
                raise ValueError(f"Item {idx} is missing required key '{key}'.")
            if not item[key] or not isinstance(item[key], str) or str(item[key]).strip() == "":
                raise ValueError(f"Item {idx} key '{key}' must be a non-empty string.")
        
        if item["complexity_label"] != expected_label:
            raise ValueError(
                f"Item {idx} has complexity_label '{item['complexity_label']}', "
                f"expected '{expected_label}'."
            )
            
    return data
