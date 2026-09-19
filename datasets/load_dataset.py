import json
import pandas as pd
from pathlib import Path
import sys
import os

# Ensure we can import 'src' when running as a script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("load_dataset")

def normalize_popqa(record):
    """
    Normalizes a PopQA record.
    PopQA typically contains 'question' and 'possible_answers' (a JSON string of list).
    """
    query = record.get("question", "")
    ans_str = ""
    
    if "possible_answers" in record:
        try:
            answers_list = json.loads(record["possible_answers"])
            if isinstance(answers_list, list) and len(answers_list) > 0:
                ans_str = str(answers_list[0])
        except Exception:
            ans_str = str(record["possible_answers"])
    elif "answers" in record:
        answers = record["answers"]
        if isinstance(answers, list) and len(answers) > 0:
            ans_str = str(answers[0])
        else:
            ans_str = str(answers)
    elif "answer" in record:
        ans_str = str(record["answer"])
        
    if ans_str and query and ans_str.lower() in query.lower():
        logger.warning(f"Potential leakage: answer found in query for PopQA record: {query}")
        
    return {
        "query": query,
        "answer": ans_str,
        "supporting_docs": None,
        "dataset_source": "popqa",
        "split": "test"
    }

def normalize_hotpotqa(record, split="validation"):
    """
    Normalizes a HotpotQA record.
    HotpotQA typically contains 'question', 'answer', and 'context'.
    """
    query = record.get("question", "")
    ans_str = record.get("answer", "")
    
    supporting_docs = []
    context_raw = record.get("context", {})
    
    if isinstance(context_raw, dict) and 'title' in context_raw and 'sentences' in context_raw:
        titles = context_raw['title']
        sentences_lists = context_raw['sentences']
        if len(titles) == len(sentences_lists):
            for t, s_list in zip(titles, sentences_lists):
                text = " ".join(s_list)
                supporting_docs.append({"title": t, "text": text})
                
    if ans_str and query and ans_str.lower() in query.lower():
        logger.warning(f"Potential leakage: answer found in query for HotpotQA record: {query}")
        
    return {
        "query": query,
        "answer": ans_str,
        "supporting_docs": supporting_docs,
        "dataset_source": "hotpotqa",
        "split": split
    }

def process_and_save(dataset_name, hf_url, split, normalize_fn, output_path, sample_limit, seed):
    logger.info(f"Loading {dataset_name} from HuggingFace via {hf_url}...")
    try:
        if hf_url.endswith(".parquet"):
            df = pd.read_parquet(hf_url)
        else:
            df = pd.read_csv(hf_url, sep='\t')
    except Exception as e:
        logger.error(f"Failed to load {dataset_name}: {e}")
        return 0, 0
        
    total_records = len(df)
    logger.info(f"Loaded {total_records} records for {dataset_name}.")
    
    if sample_limit and sample_limit < total_records:
        df = df.sample(n=sample_limit, random_state=seed)
        logger.info(f"Sampled {sample_limit} records with seed {seed}.")
    else:
        sample_limit = total_records
        
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    valid_count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = row.to_dict()
            norm_rec = normalize_fn(record)
            
            if not norm_rec["query"] or not norm_rec["answer"]:
                logger.warning("Skipping record with empty query or answer")
                continue
                
            f.write(json.dumps(norm_rec) + "\n")
            valid_count += 1
            
    logger.info(f"Saved {valid_count} normalized records to {output_path}")
    return total_records, valid_count

def run_loading():
    config = load_config("configs/config.yaml")
    base_dir = Path(__file__).parent
    raw_dir = base_dir / "raw"
    seed = config.generation.seed
    limit = config.evaluation.sample_limit
    
    popqa_url = "hf://datasets/akariasai/PopQA/test.tsv"
    popqa_out = raw_dir / "popqa.jsonl"
    pop_total, pop_count = process_and_save(
        "PopQA", popqa_url, "test", normalize_popqa, popqa_out, limit, seed
    )
    
    hotpot_url = "hf://datasets/hotpotqa/hotpot_qa/distractor/validation-00000-of-00001.parquet"
    hotpot_out = raw_dir / "hotpotqa.jsonl"
    hotpot_total, hotpot_count = process_and_save(
        "HotpotQA", hotpot_url, "validation", 
        lambda r: normalize_hotpotqa(r, "validation"), 
        hotpot_out, limit, seed
    )
    
    return pop_count, hotpot_count

if __name__ == "__main__":
    run_loading()
