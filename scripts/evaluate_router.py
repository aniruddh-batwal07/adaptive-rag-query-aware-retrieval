import json
import os
import time
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.evaluation.latency import measure_latency
from src.router.query_classifier import QueryClassifier

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="models/router")
    parser.add_argument("--val_data", type=str, default="datasets/router/val.jsonl")
    parser.add_argument("--out_dir", type=str, default="results/router_eval")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"Loading model from {args.model_dir}...")
    classifier = QueryClassifier(args.model_dir)
    
    true_labels = []
    pred_labels = []
    latencies = []
    errors = []
    
    print(f"Evaluating on {args.val_data}...")
    with open(args.val_data, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            query = item["query"]
            true_label = item["label"]
            
            res, latency = measure_latency(classifier.predict, query)
            
            pred_label = res["complexity_label"]
            confidence = res["confidence"]
            
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            latencies.append(latency)
            
            if true_label != pred_label:
                errors.append({
                    "query": query,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "confidence": confidence,
                    "dataset_source": item.get("dataset_source", "")
                })

    # Metrics
    # Map to binary: SIMPLE = 0, COMPLEX = 1
    y_true = [1 if l == "COMPLEX" else 0 for l in true_labels]
    y_pred = [1 if l == "COMPLEX" else 0 for l in pred_labels]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    
    avg_latency = sum(latencies) / len(latencies)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": {
            "true_simple_pred_simple": cm[0][0],
            "true_simple_pred_complex": cm[0][1],
            "true_complex_pred_simple": cm[1][0],
            "true_complex_pred_complex": cm[1][1]
        },
        "latency_ms": {
            "average": avg_latency,
            "min": min(latencies),
            "max": max(latencies)
        },
        "sample_count": len(true_labels),
        "device": str(classifier.device),
        "model_dir": args.model_dir
    }
    
    with open(os.path.join(args.out_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    with open(os.path.join(args.out_dir, "errors.jsonl"), 'w') as f:
        for err in errors:
            f.write(json.dumps(err) + "\n")
            
    print(f"Saved results to {args.out_dir}")

if __name__ == "__main__":
    evaluate()
