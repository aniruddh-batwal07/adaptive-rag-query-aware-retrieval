import json
import csv

distil_path = "results/primary_benchmark/adaptive/predictions.jsonl"
bert_path = "results/router_comparison/adaptive_bert/predictions.jsonl"

def load_preds(path):
    preds = {}
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            # Use query as ID since query_id is "unknown"
            preds[data["query"]] = data
    return preds

distil_preds = load_preds(distil_path)
bert_preds = load_preds(bert_path)

comparison = {
    "routing": {
        "distilbert_simple_count": 0,
        "distilbert_complex_count": 0,
        "bert_simple_count": 0,
        "bert_complex_count": 0,
        "mismatches": 0,
        "bert_false_negatives": 0,
    },
    "execution": {
        "distilbert": {"k_sum": 0, "compress_count": 0, "ret_lat": 0, "comp_lat": 0, "gen_lat": 0, "tot_lat": 0},
        "bert": {"k_sum": 0, "compress_count": 0, "ret_lat": 0, "comp_lat": 0, "gen_lat": 0, "tot_lat": 0}
    },
    "quality": {
        "distilbert": {"em": 0, "f1": 0},
        "bert": {"em": 0, "f1": 0}
    },
    "disagreements": []
}

n = len(distil_preds)

for query in distil_preds:
    d = distil_preds[query]
    b = bert_preds[query]
    
    # Routing
    d_label = d["complexity_label"]
    b_label = b["complexity_label"]
    proxy = d.get("true_label") or b.get("true_label", "unknown")
    
    if d_label == "SIMPLE": comparison["routing"]["distilbert_simple_count"] += 1
    else: comparison["routing"]["distilbert_complex_count"] += 1
        
    if b_label == "SIMPLE": comparison["routing"]["bert_simple_count"] += 1
    else: comparison["routing"]["bert_complex_count"] += 1
        
    disagree = (d_label != b_label)
    if disagree:
        comparison["routing"]["mismatches"] += 1
        # False negative: proxy is COMPLEX (or d_label is COMPLEX) and b_label is SIMPLE
        if d_label == "COMPLEX" and b_label == "SIMPLE":
            comparison["routing"]["bert_false_negatives"] += 1
            
        comparison["disagreements"].append({
            "query": query,
            "proxy_complexity": proxy,
            "distilbert": {
                "label": d_label,
                "k": d["retrieval_k"],
                "compression": d["compression_applied"],
                "f1": d.get("f1", 0),
                "tot_lat": d["total_latency_ms"]
            },
            "bert": {
                "label": b_label,
                "k": b["retrieval_k"],
                "compression": b["compression_applied"],
                "f1": b.get("f1", 0),
                "tot_lat": b["total_latency_ms"]
            },
            "k_changed": d["retrieval_k"] != b["retrieval_k"],
            "compression_changed": d["compression_applied"] != b["compression_applied"],
            "output_changed": d["prediction"] != b["prediction"]
        })
        
    # Execution
    comparison["execution"]["distilbert"]["k_sum"] += d["retrieval_k"]
    comparison["execution"]["bert"]["k_sum"] += b["retrieval_k"]
    if d["compression_applied"]: comparison["execution"]["distilbert"]["compress_count"] += 1
    if b["compression_applied"]: comparison["execution"]["bert"]["compress_count"] += 1
    
    comparison["execution"]["distilbert"]["ret_lat"] += d["retrieval_latency_ms"] or 0
    comparison["execution"]["bert"]["ret_lat"] += b["retrieval_latency_ms"] or 0
    comparison["execution"]["distilbert"]["comp_lat"] += d["compression_latency_ms"] or 0
    comparison["execution"]["bert"]["comp_lat"] += b["compression_latency_ms"] or 0
    comparison["execution"]["distilbert"]["gen_lat"] += d["generation_latency_ms"] or 0
    comparison["execution"]["bert"]["gen_lat"] += b["generation_latency_ms"] or 0
    comparison["execution"]["distilbert"]["tot_lat"] += d["total_latency_ms"] or 0
    comparison["execution"]["bert"]["tot_lat"] += b["total_latency_ms"] or 0
    
    # Quality
    comparison["quality"]["distilbert"]["em"] += d.get("em", 0)
    comparison["quality"]["bert"]["em"] += b.get("em", 0)
    comparison["quality"]["distilbert"]["f1"] += d.get("f1", 0)
    comparison["quality"]["bert"]["f1"] += b.get("f1", 0)

# Averages
for model in ["distilbert", "bert"]:
    comparison["execution"][model]["avg_k"] = comparison["execution"][model]["k_sum"] / n
    comparison["execution"][model]["compression_rate"] = comparison["execution"][model]["compress_count"] / n
    for k in ["ret_lat", "comp_lat", "gen_lat", "tot_lat"]:
        comparison["execution"][model]["avg_" + k] = comparison["execution"][model][k] / n
        del comparison["execution"][model][k]
    for k in ["em", "f1"]:
        comparison["quality"][model][k] /= n

with open("results/router_comparison/adaptive_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

with open("results/router_comparison/adaptive_comparison.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "DistilBERT", "BERT", "Difference"])
    writer.writerow(["EM", comparison["quality"]["distilbert"]["em"], comparison["quality"]["bert"]["em"], comparison["quality"]["bert"]["em"] - comparison["quality"]["distilbert"]["em"]])
    writer.writerow(["Token F1", comparison["quality"]["distilbert"]["f1"], comparison["quality"]["bert"]["f1"], comparison["quality"]["bert"]["f1"] - comparison["quality"]["distilbert"]["f1"]])
    writer.writerow(["Avg K", comparison["execution"]["distilbert"]["avg_k"], comparison["execution"]["bert"]["avg_k"], comparison["execution"]["bert"]["avg_k"] - comparison["execution"]["distilbert"]["avg_k"]])
    writer.writerow(["Compression Rate", comparison["execution"]["distilbert"]["compression_rate"], comparison["execution"]["bert"]["compression_rate"], comparison["execution"]["bert"]["compression_rate"] - comparison["execution"]["distilbert"]["compression_rate"]])
    writer.writerow(["Avg Total Latency (ms)", comparison["execution"]["distilbert"]["avg_tot_lat"], comparison["execution"]["bert"]["avg_tot_lat"], comparison["execution"]["bert"]["avg_tot_lat"] - comparison["execution"]["distilbert"]["avg_tot_lat"]])

print("Comparison generated!")
