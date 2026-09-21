import argparse
import sys
import os
import json

# Ensure src is importable
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.pipeline.adaptive_rag import Pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the minimal AdaptiveRAG pipeline.")
    parser.add_argument("--query", type=str, help="The query to answer.")
    parser.add_argument("--baseline", type=str, choices=["fixed_rag", "llm_only", "always_compress"], default="fixed_rag",
                        help="Select the baseline to run. Use 'fixed_rag' for Baseline B, 'llm_only' for Baseline A, or 'always_compress' for Baseline C.")
    parser.add_argument("--fixtures", action="store_true", help="Run the 5 development fixture queries and save results.")
    args = parser.parse_args()
    
    if not args.query and not args.fixtures:
        parser.error("Must provide either --query or --fixtures")
    
    print("Loading config...")
    config = load_config("configs/config.yaml")
    
    print(f"Initializing pipeline with baseline='{args.baseline}'...")
    pipeline = Pipeline(config, baseline=args.baseline)
    
    if args.query:
        print(f"\nRunning query: '{args.query}'\n")
        result = pipeline.run(args.query)
        
        print("================ ANSWER ================")
        print(result.answer)
        print("========================================")
        print("METADATA:")
        print(f"  Query                : {result.execution_metadata.query}")
        print(f"  Retrieval K          : {result.execution_metadata.retrieval_k}")
        lat = result.execution_metadata.retrieval_latency_ms
        if lat is not None:
            print(f"  Retrieval Latency    : {lat:.2f} ms")
        else:
            print(f"  Retrieval Latency    : N/A")

        print(f"  Compression Applied  : {result.execution_metadata.compression_applied}")
        if result.execution_metadata.compression_applied:
            print(f"  Original Tokens      : {result.execution_metadata.original_context_tokens}")
            print(f"  Compressed Tokens    : {result.execution_metadata.compressed_context_tokens}")
            c_lat = result.execution_metadata.compression_latency_ms
            if c_lat is not None:
                print(f"  Compression Latency  : {c_lat:.2f} ms")

        print(f"  Generation Latency   : {result.execution_metadata.generation_latency_ms:.2f} ms")
        print(f"  Total Latency        : {result.execution_metadata.total_latency_ms:.2f} ms")
        if result.execution_metadata.error_status:
            print(f"  Error Status         : {result.execution_metadata.error_status}")

    if args.fixtures:
        print("\nRunning fixtures...")
        fixtures_path = "datasets/fixtures/simple_queries.json"
        with open(fixtures_path, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        # Take first 5 dev fixture queries
        queries_data = queries_data[:5]
        
        if args.baseline == "llm_only":
            results_dir = "results/baseline_a"
        elif args.baseline == "always_compress":
            results_dir = "results/baseline_c"
        else:
            results_dir = "results/baseline_b"
            
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, "results.jsonl")
        
        print(f"Saving results to {results_file}")
        with open(results_file, 'w', encoding='utf-8') as out_f:
            for item in queries_data:
                q = item["query"]
                print(f"Processing: {q}")
                result = pipeline.run(q)
                
                res_dict = {
                    "query": result.execution_metadata.original_query,
                    "answer": result.answer,
                    "baseline": args.baseline,
                    "retrieval_k": result.execution_metadata.retrieval_k,
                    "retrieved_chunks": result.execution_metadata.retrieved_chunks,
                    "generation_latency_ms": result.execution_metadata.generation_latency_ms,
                    "total_latency_ms": result.execution_metadata.total_latency_ms,
                    "metadata": {
                        "normalized_query": result.execution_metadata.query,
                        "retrieval_latency_ms": result.execution_metadata.retrieval_latency_ms,
                        "compression_applied": result.execution_metadata.compression_applied,
                        "original_context_tokens": result.execution_metadata.original_context_tokens,
                        "compressed_context_tokens": result.execution_metadata.compressed_context_tokens,
                        "compression_ratio": result.execution_metadata.compression_ratio,
                        "compression_latency_ms": result.execution_metadata.compression_latency_ms,
                        "error_status": result.execution_metadata.error_status,
                    }
                }
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if type(obj).__name__ == 'float32' or type(obj).__name__ == 'float64':
                            return float(obj)
                        if type(obj).__name__ == 'int64' or type(obj).__name__ == 'int32':
                            return int(obj)
                        if hasattr(obj, 'item'):
                            return obj.item()
                        return super(NumpyEncoder, self).default(obj)

                try:
                    out_f.write(json.dumps(res_dict, cls=NumpyEncoder) + "\n")
                    out_f.flush()
                except Exception as e:
                    print(f"FAILED TO DUMPS: {e}")
                    raise
        print("Done running fixtures.")

if __name__ == "__main__":
    main()
