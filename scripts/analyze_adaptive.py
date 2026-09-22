import json
import pandas as pd
from pathlib import Path

def main():
    preds_file = "results/primary_benchmark/adaptive/predictions.jsonl"
    
    records = []
    with open(preds_file, "r") as f:
        for line in f:
            records.append(json.loads(line))
            
    df = pd.DataFrame(records)
    
    # Group by complexity_label
    groups = df.groupby("complexity_label")
    
    analysis = {}
    
    for label, group in groups:
        metrics = {
            "count": len(group),
            "selected_k": group["retrieval_k"].mean(),
            "compression_invocation_rate": group["compression_applied"].mean(),
            "average_retrieved_context_tokens": group["original_context_tokens"].mean(),
            "average_compressed_context_tokens": group["compressed_context_tokens"].mean(),
            "average_retrieval_latency": group["retrieval_latency_ms"].mean(),
            "average_compression_latency": group["compression_latency_ms"].mean(),
            "average_generation_latency": group["generation_latency_ms"].mean(),
            "average_total_latency": group["total_latency_ms"].mean(),
            "em": group["em"].mean(),
            "token_f1": group["f1"].mean()
        }
        analysis[label] = metrics
        
    Path("results/m12").mkdir(parents=True, exist_ok=True)
    with open("results/m12/adaptive_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

if __name__ == "__main__":
    main()
