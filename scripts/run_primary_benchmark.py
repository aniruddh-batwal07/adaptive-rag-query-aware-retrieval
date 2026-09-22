import argparse
import json
import logging
import csv
import pandas as pd
from pathlib import Path
from src.utils.config import load_config
from src.pipeline.adaptive_rag import Pipeline
from src.evaluation.benchmark import run_benchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Primary Benchmark for all Baselines")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default="datasets/splits/test.jsonl", help="Path to test JSONL")
    parser.add_argument("--limit", type=int, default=1000, help="Limit number of evaluation examples")
    args = parser.parse_args()

    config = load_config(args.config)

    logger.info(f"Loading dataset: {args.dataset}")
    examples = []
    try:
        with open(args.dataset, "r") as f:
            for line in f:
                examples.append(json.loads(line))
                if args.limit and len(examples) >= args.limit:
                    break
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {args.dataset}")
        return

    logger.info(f"Loaded {len(examples)} examples.")
    
    baselines = ["llm_only", "fixed_rag", "always_compress", "adaptive"]
    summary_data = []

    logger.info("Initializing global models to prevent reloading...")
    from src.retriever.retriever import Retriever
    from src.generator.response_generator import ResponseGenerator
    from src.optimizer.context_optimizer import ContextOptimizer
    from src.router.query_classifier import QueryClassifier
    from src.router.routing_logic import AdaptiveController

    retriever = Retriever()
    generator = ResponseGenerator(config)
    compressor = ContextOptimizer(config)
    router_path = config.models.router.checkpoint or "models/router"
    router = QueryClassifier(router_path, device=config.runtime.device)
    controller = AdaptiveController(config)
    logger.info("All global models initialized.")

    for baseline in baselines:
        logger.info(f"=== Starting Baseline: {baseline} ===")
        exp_name = f"primary_benchmark/{baseline}"
        pipeline = Pipeline(
            config=config, 
            baseline=baseline,
            retriever=retriever,
            generator=generator,
            compressor=compressor,
            router=router,
            controller=controller
        )

        result = run_benchmark(
            pipeline_fn=pipeline.run, 
            examples=examples, 
            config=config,
            experiment_name=exp_name
        )
        
        m = result.metrics
        lat = m.get("latency", {})
        tok = m.get("tokens", {})
        
        # Save baseline configuration in config.yaml
        config_path = Path("results") / exp_name / "config.yaml"
        with open(config_path, "w") as f:
            import yaml, dataclasses
            yaml.dump(dataclasses.asdict(config), f)

        # Save metadata reproducibility file
        metadata_path = Path("results") / exp_name / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "dataset": args.dataset,
                "sample_count": len(examples),
                "model_identifiers": {
                    "generator": config.models.generator.name,
                    "retriever_embeddings": config.models.embeddings.name,
                    "compressor": config.models.compressor.name,
                    "router": config.models.router.name
                },
                "retrieval_k": {
                    "fixed_rag": getattr(config.retrieval, 'k_baseline', 5),
                    "always_compress": config.retrieval.k_complex,
                    "adaptive": f"SIMPLE: {config.retrieval.k_simple}, COMPLEX: {config.retrieval.k_complex}"
                }.get(baseline, None),
                "random_seed": 42
            }, f, indent=2)

        summary_data.append({
            "Baseline": baseline,
            "EM": m.get("em"),
            "Token_F1": m.get("f1"),
            "Recall_Mean": m.get("retrieval", {}).get("recall_mean"),
            "Mean_Total_Latency_ms": lat.get("total_mean_ms"),
            "Mean_Gen_Latency_ms": lat.get("generation_mean_ms"),
            "Mean_Ret_Latency_ms": lat.get("retrieval_mean_ms"),
            "Mean_Comp_Latency_ms": lat.get("compression_mean_ms"),
            "Mean_Rout_Latency_ms": lat.get("router_mean_ms"),
            "Original_Context_Tokens": tok.get("original_mean"),
            "Compressed_Context_Tokens": tok.get("compressed_mean")
        })
        logger.info(f"Finished Baseline: {baseline}")
        
        if "router" in m:
            logger.info("ROUTER METRICS FOR TEST SET:")
            logger.info(json.dumps(m["router"], indent=2))
        
    # Write summary.csv
    summary_path = Path("results/primary_benchmark/summary.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(summary_data)
    df.to_csv(summary_path, index=False)
    logger.info(f"Summary written to {summary_path}")
    print("\nBenchmark Summary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
