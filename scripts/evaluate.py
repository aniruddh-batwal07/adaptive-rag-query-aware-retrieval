import argparse
import json
import logging
from src.utils.config import load_config
from src.pipeline.adaptive_rag import Pipeline
from src.evaluation.benchmark import run_benchmark

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Evaluate AdaptiveRAG Baseline")
    parser.add_argument("--baseline", type=str, default="fixed_rag", choices=["llm_only", "fixed_rag", "always_compress", "adaptive"])
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default="datasets/splits/test.jsonl", help="Path to test JSONL")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of evaluation examples")
    parser.add_argument("--exp-name", type=str, default=None, help="Experiment name (default: baseline_name)")
    args = parser.parse_args()

    config = load_config(args.config)
    exp_name = args.exp_name if args.exp_name else f"baseline_{args.baseline}"

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
    logger.info(f"Initializing pipeline for baseline: {args.baseline}")
    
    pipeline = Pipeline(config=config, baseline=args.baseline)

    logger.info(f"Running benchmark...")
    
    # Run benchmark handles everything and writes artifacts
    result = run_benchmark(
        pipeline_fn=pipeline.run, 
        examples=examples, 
        config=config,
        experiment_name=exp_name
    )

    logger.info("\n========== BENCHMARK SUMMARY ==========")
    logger.info(f"Baseline: {args.baseline}")
    logger.info(f"Examples Evaluated: {len(result.predictions)}")
    
    m = result.metrics
    logger.info(f"Exact Match (EM): {m.get('em'):.4f}" if m.get('em') is not None else "Exact Match (EM): N/A")
    logger.info(f"Token F1:         {m.get('f1'):.4f}" if m.get('f1') is not None else "Token F1:         N/A")
    
    lat = m.get("latency", {})
    logger.info(f"Mean Gen Latency: {lat.get('generation_mean_ms'):.2f} ms" if lat.get('generation_mean_ms') else "Mean Gen Latency: N/A")
    logger.info(f"Mean Tot Latency: {lat.get('total_mean_ms'):.2f} ms" if lat.get('total_mean_ms') else "Mean Tot Latency: N/A")
    
    logger.info(f"Results saved to results/{exp_name}/")

if __name__ == "__main__":
    main()
