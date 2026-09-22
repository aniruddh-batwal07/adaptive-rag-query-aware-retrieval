import json
import os

def test_mini_10_dataset():
    test_file = "datasets/splits/test.jsonl"
    mini_test_file = "datasets/splits/test_mini_10.jsonl"
    
    assert os.path.exists(test_file), f"{test_file} not found"
    assert os.path.exists(mini_test_file), f"{mini_test_file} not found"
    
    original_queries = set()
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            original_queries.add(record['query'])
            
    mini_records = []
    with open(mini_test_file, 'r', encoding='utf-8') as f:
        for line in f:
            mini_records.append(json.loads(line))
            
    assert len(mini_records) == 10, f"Expected exactly 10 queries, found {len(mini_records)}"
    
    popqa_count = 0
    hotpotqa_count = 0
    queries = set()
    
    for record in mini_records:
        assert record['query'] in original_queries, f"Query not found in original test.jsonl: {record['query']}"
        assert record['query'] not in queries, f"Duplicate query found: {record['query']}"
        queries.add(record['query'])
        
        if record.get('dataset_source') == 'popqa':
            popqa_count += 1
        elif record.get('dataset_source') == 'hotpotqa':
            hotpotqa_count += 1
            
    assert popqa_count == 5, f"Expected 5 popqa queries, found {popqa_count}"
    assert hotpotqa_count == 5, f"Expected 5 hotpotqa queries, found {hotpotqa_count}"
