import json
import random
from pathlib import Path
import sys
import os

# Ensure we can import 'src' when running as a script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("prepare_data")

def load_jsonl(path: Path):
    records = []
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records

def deduplicate_records(records):
    """
    Remove records with identical queries to prevent leakage.
    Sorts deterministically before deduplication to ensure reproducibility.
    """
    # Deterministic sort based on dataset_source, then query, then answer
    sorted_records = sorted(
        records, 
        key=lambda x: (x.get("dataset_source", ""), x.get("query", ""), x.get("answer", ""))
    )
    
    seen_queries = set()
    deduped = []
    for r in sorted_records:
        q = r.get("query", "").strip().lower()
        if q not in seen_queries:
            seen_queries.add(q)
            deduped.append(r)
            
    dropped = len(records) - len(deduped)
    if dropped > 0:
        logger.info(f"Dropped {dropped} duplicate queries.")
        
    return deduped

def create_splits(config_path="configs/config.yaml", base_dir=None):
    config = load_config(config_path)
    seed = config.generation.seed
    ratios = config.evaluation.split_ratios
    
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    raw_dir = base_dir / "raw"
    splits_dir = base_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    popqa_file = raw_dir / "popqa.jsonl"
    hotpotqa_file = raw_dir / "hotpotqa.jsonl"
    
    pop_records = load_jsonl(popqa_file)
    hotpot_records = load_jsonl(hotpotqa_file)
    
    all_records = pop_records + hotpot_records
    if not all_records:
        logger.error("No input records found.")
        raise ValueError("No input records found")
        
    logger.info(f"Loaded {len(pop_records)} PopQA and {len(hotpot_records)} HotpotQA records.")
    
    # 1. Deduplicate
    deduped = deduplicate_records(all_records)
    
    # 2. Deterministic Shuffle
    rng = random.Random(seed)
    rng.shuffle(deduped)
    
    # 3. Calculate split indices
    total = len(deduped)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    
    splits = {
        "train": deduped[:train_end],
        "val": deduped[train_end:val_end],
        "test": deduped[val_end:]
    }
    
    # 4. Write to files
    out_counts = {}
    for split_name, records in splits.items():
        out_file = splits_dir / f"{split_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for r in records:
                # Assign project_split without overwriting original split
                r_copy = dict(r)
                r_copy["project_split"] = split_name
                f.write(json.dumps(r_copy) + "\n")
        out_counts[split_name] = len(records)
        logger.info(f"Wrote {len(records)} records to {out_file}")
        
    # 5. Write metadata
    metadata = {
        "seed": seed,
        "split_ratios": ratios,
        "input_files": ["popqa.jsonl", "hotpotqa.jsonl"],
        "input_counts": {
            "popqa": len(pop_records),
            "hotpotqa": len(hotpot_records),
            "total_raw": len(all_records),
            "total_deduped": total
        },
        "output_counts": out_counts
    }
    
    with open(splits_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info("Splitting complete.")
    return metadata

if __name__ == "__main__":
    create_splits()
