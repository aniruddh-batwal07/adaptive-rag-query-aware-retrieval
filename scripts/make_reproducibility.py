import json
import subprocess
from pathlib import Path

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        return None

def main():
    repo_data = {
        "experiment_name": "M12_Compression_Ablation",
        "dataset_split": "datasets/splits/test_mini_10.jsonl",
        "sample_count": 10,
        "random_seed": 42,
        "models": {
            "generator": "HuggingFaceTB/SmolLM-135M-Instruct",
            "retriever_embeddings": "BAAI/bge-small-en-v1.5",
            "compressor": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            "router": "distilbert-base-uncased"
        },
        "retrieval_k": {
            "condition_a_without_compression": 10,
            "condition_b_with_compression": 10
        },
        "compression_parameters": {
            "budget": 0.5
        },
        "generation_parameters": {
            "temperature": 0.1,
            "max_new_tokens": 256
        },
        "benchmark_git_commit": get_git_commit(),
        "software_environment": "Python 3.14.5, PyTorch, HuggingFace Transformers, ChromaDB"
    }
    
    Path("results/m12").mkdir(parents=True, exist_ok=True)
    with open("results/m12/reproducibility.json", "w") as f:
        json.dump(repo_data, f, indent=2)

if __name__ == "__main__":
    main()
