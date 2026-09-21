import json
import os
import random

def prepare_router_data():
    splits_dir = "datasets/splits"
    out_dir = "datasets/router"
    os.makedirs(out_dir, exist_ok=True)
    
    for split in ["train", "val"]:
        in_path = os.path.join(splits_dir, f"{split}.jsonl")
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        
        if not os.path.exists(in_path):
            continue
            
        with open(in_path, 'r', encoding='utf-8') as f_in, open(out_path, 'w', encoding='utf-8') as f_out:
            records = []
            for line in f_in:
                item = json.loads(line)
                ds_source = item.get("dataset_source", "")
                
                if "popqa" in ds_source.lower():
                    label = "SIMPLE"
                elif "hotpotqa" in ds_source.lower():
                    label = "COMPLEX"
                else:
                    continue # Skip unknown
                    
                router_item = {
                    "query": item["query"],
                    "label": label,
                    "dataset_source": ds_source # ONLY for analysis, NOT a feature
                }
                records.append(router_item)
                
            # Shuffle so we don't have all SIMPLE then all COMPLEX
            random.seed(42)
            random.shuffle(records)
            
            for r in records:
                f_out.write(json.dumps(r) + "\n")
        print(f"Prepared {len(records)} items for {split} split in {out_path}")

if __name__ == "__main__":
    prepare_router_data()
