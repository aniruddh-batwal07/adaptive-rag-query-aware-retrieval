import argparse
import sys
import os

# Ensure src is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.pipeline.adaptive_rag import Pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the minimal AdaptiveRAG pipeline.")
    parser.add_argument("--query", type=str, required=True, help="The query to answer.")
    args = parser.parse_args()
    
    print("Loading config...")
    config = load_config("configs/config.yaml")
    
    print("Initializing pipeline (loading retriever and generator)...")
    pipeline = Pipeline(config)
    
    print(f"\nRunning query: '{args.query}'\n")
    result = pipeline.run(args.query)
    
    print("================ ANSWER ================")
    print(result.answer)
    print("========================================")
    print("METADATA:")
    print(f"  Query                : {result.execution_metadata.query}")
    print(f"  Retrieval K          : {result.execution_metadata.retrieval_k}")
    print(f"  Retrieval Latency    : {result.execution_metadata.retrieval_latency_ms:.2f} ms")
    print(f"  Generation Latency   : {result.execution_metadata.generation_latency_ms:.2f} ms")
    print(f"  Total Latency        : {result.execution_metadata.total_latency_ms:.2f} ms")

if __name__ == "__main__":
    main()
