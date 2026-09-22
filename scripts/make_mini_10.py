import json
import random

def create_mini_test():
    input_file = "datasets/splits/test.jsonl"
    output_file = "datasets/splits/test_mini_10.jsonl"
    
    popqa_records = []
    hotpotqa_records = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            if record.get('dataset_source') == 'popqa':
                popqa_records.append(record)
            elif record.get('dataset_source') == 'hotpotqa':
                hotpotqa_records.append(record)
                
    random.seed(42)
    selected_popqa = random.sample(popqa_records, 5)
    selected_hotpotqa = random.sample(hotpotqa_records, 5)
    
    combined = selected_popqa + selected_hotpotqa
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in combined:
            f.write(json.dumps(record) + '\n')

if __name__ == "__main__":
    create_mini_test()
