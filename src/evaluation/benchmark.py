import json
import csv
import os
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Dict, Any, Optional

from src.utils.config import Config
from src.evaluation.metrics import (
    exact_match,
    token_f1,
    compression_ratio,
    count_tokens,
    recall_at_k,
    router_metrics
)

@dataclass
class BenchmarkResult:
    metrics: Dict[str, Any]
    predictions: List[Dict[str, Any]]

def run_benchmark(pipeline_fn: Callable[[str], Any], examples: List[Dict[str, Any]], config: Config, experiment_name: str = "default_experiment") -> BenchmarkResult:
    """
    Runs a pipeline on a list of examples and computes evaluation metrics.
    
    Args:
        pipeline_fn: Callable that takes a query string and returns a PipelineResult.
        examples: List of dataset examples (e.g., from test.jsonl). Must contain 'question' and 'answers' (or 'answer').
                  Optional: 'supporting_facts' for HotpotQA evidence.
        config: System configuration.
        experiment_name: Name of the output directory.
        
    Returns:
        BenchmarkResult containing aggregated metrics and per-example predictions.
    """
    predictions_out = []
    results_dir = Path("results") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = results_dir / "predictions.jsonl"
    with open(predictions_file, "w") as f:
        pass # Clear file before appending
    
    # Aggregators
    em_scores = []
    f1_scores = []
    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []
    
    # Optional aggregators
    compression_latencies = []
    router_latencies = []
    original_tokens_list = []
    compressed_tokens_list = []
    router_labels_true = []
    router_labels_pred = []
    
    # Retrieval recall aggregators
    recalls = []

    from tqdm import tqdm
    for ex in tqdm(examples, desc="Running benchmark"):
        query = ex.get('question', ex.get('query', ''))
        reference_answers = ex.get('answers', [ex.get('answer', '')])
        if isinstance(reference_answers, str):
            reference_answers = [reference_answers]
        
        # Ground truth labels if available
        # Proxy labels: popqa -> SIMPLE, hotpotqa -> COMPLEX (for testing router)
        # We can assume 'source' field or infer from dataset
        true_label = "SIMPLE" if ex.get('dataset_source', ex.get('source', '')) == "popqa" else "COMPLEX"
        
        # Execute pipeline
        try:
            result = pipeline_fn(query)
            answer = result.answer
            metadata = result.execution_metadata
            status = getattr(metadata, 'error_status', 'SUCCESS') or 'SUCCESS'
        except Exception as e:
            answer = ""
            metadata = None
            status = f"FAILED: {str(e)}"
        
        # Compare with references (take max over all references)
        best_em = 0.0
        best_f1 = 0.0
        for ref in reference_answers:
            em = exact_match(answer, ref)
            f1 = token_f1(answer, ref)
            if em > best_em:
                best_em = em
            if f1 > best_f1:
                best_f1 = f1
                
        # Extract metadata
        record = {
            "query_id": ex.get("id", ex.get("_id", "unknown")),
            "query": query,
            "reference_answers": reference_answers,
            "prediction": answer,
            "em": best_em,
            "f1": best_f1,
            "status": status,
        }
        
        if metadata:
            record["retrieval_k"] = getattr(metadata, "retrieval_k", None)
            
            # Latencies
            ret_lat = getattr(metadata, "retrieval_latency_ms", None)
            gen_lat = getattr(metadata, "generation_latency_ms", None)
            tot_lat = getattr(metadata, "total_latency_ms", None)
            comp_lat = getattr(metadata, "compression_latency_ms", None)
            rout_lat = getattr(metadata, "router_latency_ms", None)
            
            record["retrieval_latency_ms"] = ret_lat
            record["generation_latency_ms"] = gen_lat
            record["total_latency_ms"] = tot_lat
            record["compression_latency_ms"] = comp_lat
            record["router_latency_ms"] = rout_lat
            
            # Tokens & Compression
            record["compression_applied"] = getattr(metadata, "compression_applied", False)
            orig_toks = getattr(metadata, "original_context_tokens", None)
            comp_toks = getattr(metadata, "compressed_context_tokens", None)
            
            record["original_context_tokens"] = orig_toks
            record["compressed_context_tokens"] = comp_toks
            record["compression_ratio"] = getattr(metadata, "compression_ratio", None)
            
            # Router
            comp_label = getattr(metadata, "complexity_label", None)
            record["complexity_label"] = comp_label
            record["router_confidence"] = getattr(metadata, "router_confidence", None)
            record["true_label"] = true_label
                
            # Retrieval evaluation
            retrieved_chunks = getattr(metadata, "retrieved_chunks", [])
            retrieved_ids = [c.get("document_id") for c in retrieved_chunks if "document_id" in c]
            # fallback to chunk_id if document_id is not present
            if not retrieved_ids and retrieved_chunks:
                retrieved_ids = [c.get("chunk_id", "") for c in retrieved_chunks]
                
            ref_docs = ex.get('supporting_facts', [])
            if ref_docs and retrieved_ids:
                # HotpotQA format: list of [title, sentence_index]
                ref_ids = list(set([doc[0] for doc in ref_docs if isinstance(doc, list) and len(doc) > 0]))
                k = getattr(metadata, "retrieval_k", 5)
                rec_at_k = recall_at_k(retrieved_ids, ref_ids, k)
                record["recall_at_k"] = rec_at_k
            else:
                record["recall_at_k"] = None

        predictions_out.append(record)
        
        with open(predictions_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Aggregate Metrics from file
    em_scores = []
    f1_scores = []
    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []
    compression_latencies = []
    router_latencies = []
    original_tokens_list = []
    compressed_tokens_list = []
    router_labels_true = []
    router_labels_pred = []
    recalls = []

    with open(predictions_file, "r") as f:
        for line in f:
            p = json.loads(line)
            if "em" in p: em_scores.append(p["em"])
            if "f1" in p: f1_scores.append(p["f1"])

            if p.get("retrieval_latency_ms") is not None: retrieval_latencies.append(p["retrieval_latency_ms"])
            if p.get("generation_latency_ms") is not None: generation_latencies.append(p["generation_latency_ms"])
            if p.get("total_latency_ms") is not None: total_latencies.append(p["total_latency_ms"])
            if p.get("compression_latency_ms") is not None: compression_latencies.append(p["compression_latency_ms"])
            if p.get("router_latency_ms") is not None: router_latencies.append(p["router_latency_ms"])

            if p.get("original_context_tokens") is not None: original_tokens_list.append(p["original_context_tokens"])
            if p.get("compressed_context_tokens") is not None: compressed_tokens_list.append(p["compressed_context_tokens"])

            if p.get("complexity_label") is not None:
                router_labels_pred.append(p["complexity_label"])
                router_labels_true.append(p.get("true_label", "COMPLEX"))

            if p.get("recall_at_k") is not None: recalls.append(p["recall_at_k"])

    def safe_mean(l):
        return sum(l) / len(l) if len(l) > 0 else None
        
    metrics = {
        "em": safe_mean(em_scores),
        "f1": safe_mean(f1_scores),
        "latency": {
            "retrieval_mean_ms": safe_mean(retrieval_latencies),
            "generation_mean_ms": safe_mean(generation_latencies),
            "total_mean_ms": safe_mean(total_latencies),
            "compression_mean_ms": safe_mean(compression_latencies) if compression_latencies else None,
            "router_mean_ms": safe_mean(router_latencies) if router_latencies else None,
        },
        "tokens": {
            "original_mean": safe_mean(original_tokens_list) if original_tokens_list else None,
            "compressed_mean": safe_mean(compressed_tokens_list) if compressed_tokens_list else None,
        },
        "retrieval": {
            "recall_mean": safe_mean(recalls) if recalls else None
        }
    }
    
    if router_labels_pred:
        metrics["router"] = router_metrics(router_labels_pred, router_labels_true)
        
    # Persist results
    results_dir = Path("results") / experiment_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. metrics.json
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # 3. latency.csv
    with open(results_dir / "latency.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "retrieval_ms", "compression_ms", "router_ms", "generation_ms", "total_ms"])
        for p in predictions_out:
            writer.writerow([
                p.get("query_id"),
                p.get("retrieval_latency_ms"),
                p.get("compression_latency_ms"),
                p.get("router_latency_ms"),
                p.get("generation_latency_ms"),
                p.get("total_latency_ms")
            ])
            
    return BenchmarkResult(metrics=metrics, predictions=predictions_out)
