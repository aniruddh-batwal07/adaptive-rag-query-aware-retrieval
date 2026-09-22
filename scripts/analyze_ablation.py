import json
import pandas as pd
from pathlib import Path

def load_metrics(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

def main():
    without_comp_preds = "results/m12/ablation_fixed_rag/predictions.jsonl"
    with_comp_preds = "results/m12/ablation_always_compress/predictions.jsonl"
    
    records_without = []
    with open(without_comp_preds, "r") as f:
        for line in f:
            records_without.append(json.loads(line))
            
    records_with = []
    with open(with_comp_preds, "r") as f:
        for line in f:
            records_with.append(json.loads(line))
            
    df_without = pd.DataFrame(records_without)
    df_with = pd.DataFrame(records_with)
    
    # Aggregate Metrics
    agg = {
        "WITHOUT_COMPRESSION": {
            "em": df_without["em"].mean(),
            "token_f1": df_without["f1"].mean(),
            "average_retrieved_context_tokens": df_without["original_context_tokens"].mean(),
            "average_compressed_context_tokens": df_without["compressed_context_tokens"].mean(),
            "average_retrieval_latency": df_without["retrieval_latency_ms"].mean(),
            "average_compression_latency": df_without["compression_latency_ms"].mean(),
            "average_generation_latency": df_without["generation_latency_ms"].mean(),
            "average_total_latency": df_without["total_latency_ms"].mean(),
        },
        "WITH_COMPRESSION": {
            "em": df_with["em"].mean(),
            "token_f1": df_with["f1"].mean(),
            "average_retrieved_context_tokens": df_with["original_context_tokens"].mean(),
            "average_compressed_context_tokens": df_with["compressed_context_tokens"].mean(),
            "average_retrieval_latency": df_with["retrieval_latency_ms"].mean(),
            "average_compression_latency": df_with["compression_latency_ms"].mean(),
            "average_generation_latency": df_with["generation_latency_ms"].mean(),
            "average_total_latency": df_with["total_latency_ms"].mean(),
        }
    }
    
    # Calculate differences
    orig_tokens = agg["WITHOUT_COMPRESSION"]["average_retrieved_context_tokens"]
    if pd.isna(orig_tokens):
        orig_tokens = agg["WITH_COMPRESSION"]["average_retrieved_context_tokens"]
        
    comp_tokens = agg["WITH_COMPRESSION"]["average_compressed_context_tokens"]
    
    agg["DIFFERENCES"] = {
        "token_reduction": orig_tokens - comp_tokens,
        "percentage_token_reduction": (orig_tokens - comp_tokens) / orig_tokens * 100,
        "compression_overhead": agg["WITH_COMPRESSION"]["average_compression_latency"],
        "generation_latency_difference": agg["WITHOUT_COMPRESSION"]["average_generation_latency"] - agg["WITH_COMPRESSION"]["average_generation_latency"],
        "total_latency_difference": agg["WITHOUT_COMPRESSION"]["average_total_latency"] - agg["WITH_COMPRESSION"]["average_total_latency"]
    }
    
    with open("results/m12/ablation_aggregate.json", "w") as f:
        json.dump(agg, f, indent=2)

if __name__ == "__main__":
    main()
