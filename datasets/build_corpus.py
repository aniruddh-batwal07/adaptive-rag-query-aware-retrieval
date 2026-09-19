import json
import hashlib
from pathlib import Path
import sys
import os

# Ensure we can import 'src' when running as a script
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("build_corpus")

def get_stable_id(text: str, title: str = "") -> str:
    """Generate a deterministic ID based on document title and content."""
    content_to_hash = f"{title}|||{text}".encode("utf-8")
    return hashlib.md5(content_to_hash).hexdigest()

def chunk_text(text: str, chunk_size: int, chunk_overlap: int):
    """
    Fixed-window chunking using whitespace-separated words.
    This avoids unnecessary tokenizer dependencies.
    """
    words = text.split()
    if not words:
        return []
        
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = 1 # fallback if overlap is larger than size
        
    chunks = []
    for i in range(0, len(words), step):
        chunk_words = words[i:i+chunk_size]
        chunks.append(" ".join(chunk_words))
        
        # Stop if we've reached the end of the words
        if i + chunk_size >= len(words):
            break
            
    return chunks

def build_corpus(config_path="configs/config.yaml", base_dir=None):
    config = load_config(config_path)
    chunk_size = config.retrieval.chunk_size
    chunk_overlap = config.retrieval.chunk_overlap
    
    if base_dir is None:
        base_dir = Path(__file__).parent
        
    raw_dir = base_dir / "raw"
    corpus_dir = base_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_process = {
        "popqa": raw_dir / "popqa.jsonl",
        "hotpotqa": raw_dir / "hotpotqa.jsonl"
    }
    
    stats = {}
    
    # Store unique documents to prevent duplicates
    unique_docs = {} # mapping doc_id -> doc_dict
    
    for dataset_name, filepath in files_to_process.items():
        ds_stats = {"source_docs": 0, "unique_docs": 0, "chunks": 0}
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            stats[dataset_name] = ds_stats
            continue
            
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                # Check for supporting_docs
                docs = record.get("supporting_docs")
                if not docs:
                    continue
                    
                for doc in docs:
                    ds_stats["source_docs"] += 1
                    
                    title = doc.get("title", "")
                    text = doc.get("text", "")
                    
                    if not text.strip():
                        continue # Skip empty documents
                        
                    doc_id = get_stable_id(text, title)
                    
                    if doc_id not in unique_docs:
                        unique_docs[doc_id] = {
                            "document_id": doc_id,
                            "dataset_source": dataset_name,
                            "title": title,
                            "text": text,
                            "source_split": record.get("split", "unknown")
                        }
                        ds_stats["unique_docs"] += 1
                        
        stats[dataset_name] = ds_stats

    # Now, perform chunking on unique documents
    output_chunks = []
    
    # Sort keys for determinism
    sorted_doc_ids = sorted(unique_docs.keys())
    
    for doc_id in sorted_doc_ids:
        doc = unique_docs[doc_id]
        
        chunks = chunk_text(doc["text"], chunk_size, chunk_overlap)
        
        for i, chunk_text_str in enumerate(chunks):
            if not chunk_text_str.strip():
                continue
                
            chunk_id = f"{doc_id}_{i}"
            chunk_record = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "text": chunk_text_str,
                "source_metadata": {
                    "dataset_source": doc["dataset_source"],
                    "title": doc["title"],
                    "source_split": doc["source_split"]
                }
            }
            output_chunks.append(chunk_record)
            stats[doc["dataset_source"]]["chunks"] += 1
            
    # Write to corpus file
    corpus_file = corpus_dir / "chunks.jsonl"
    with open(corpus_file, "w", encoding="utf-8") as f:
        for chunk in output_chunks:
            f.write(json.dumps(chunk) + "\n")
            
    stats["total"] = {
        "unique_docs": len(unique_docs),
        "chunks": len(output_chunks)
    }
    
    logger.info(f"Corpus building complete. Wrote {len(output_chunks)} chunks to {corpus_file}")
    
    return stats

if __name__ == "__main__":
    stats = build_corpus()
    print(json.dumps(stats, indent=2))
