import json
import os
import sys
import argparse
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.evaluation.latency import measure_latency
from src.router.query_classifier import QueryClassifier

def eval_model(model_dir, val_data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Loading model from {model_dir}...")
    classifier = QueryClassifier(model_dir)
    
    true_labels = []
    pred_labels = []
    latencies = []
    
    print(f"Evaluating on {val_data}...")
    with open(val_data, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            query = item["query"]
            true_label = item["label"]
            
            res, latency = measure_latency(classifier.predict, query)
            
            pred_label = res["complexity_label"]
            
            true_labels.append(true_label)
            pred_labels.append(pred_label)
            latencies.append(latency)

    y_true = [1 if l == "COMPLEX" else 0 for l in true_labels]
    y_pred = [1 if l == "COMPLEX" else 0 for l in pred_labels]
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    
    avg_latency = sum(latencies) / len(latencies)
    
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
        "avg_inference_latency_ms": avg_latency,
        "num_samples": len(true_labels)
    }
    
    with open(os.path.join(out_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
        
    return metrics

def compare():
    val_data = "datasets/router/val.jsonl"
    distilbert_dir = "models/router"
    bert_dir = "models/router_bert"
    
    comp_dir = "results/router_comparison"
    distilbert_out = os.path.join(comp_dir, "distilbert")
    bert_out = os.path.join(comp_dir, "bert")
    
    print("Evaluating DistilBERT...")
    m_distilbert = eval_model(distilbert_dir, val_data, distilbert_out)
    
    print("Evaluating BERT...")
    m_bert = eval_model(bert_dir, val_data, bert_out)
    
    # Comparison
    comparison = {
        "distilbert": m_distilbert,
        "bert": m_bert,
        "differences": {
            "accuracy": m_bert["accuracy"] - m_distilbert["accuracy"],
            "precision": m_bert["precision"] - m_distilbert["precision"],
            "recall": m_bert["recall"] - m_distilbert["recall"],
            "f1": m_bert["f1"] - m_distilbert["f1"],
            "avg_inference_latency_ms": m_bert["avg_inference_latency_ms"] - m_distilbert["avg_inference_latency_ms"]
        }
    }
    
    with open(os.path.join(comp_dir, "comparison.json"), 'w') as f:
        json.dump(comparison, f, indent=2)
        
    # CSV
    metrics_list = ["accuracy", "precision", "recall", "f1", "avg_inference_latency_ms"]
    csv_file = os.path.join(comp_dir, "comparison.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "distilbert", "bert", "difference"])
        for met in metrics_list:
            writer.writerow([
                met, 
                m_distilbert[met], 
                m_bert[met], 
                comparison["differences"][met]
            ])
            
    print(f"Saved comparison results to {comp_dir}")

if __name__ == "__main__":
    compare()
