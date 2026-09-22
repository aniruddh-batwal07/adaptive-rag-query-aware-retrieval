import json
import logging
from pathlib import Path
from src.utils.config import load_config
from src.pipeline.adaptive_rag import Pipeline
from src.evaluation.benchmark import run_benchmark
from src.retriever.retriever import Retriever
from src.generator.response_generator import ResponseGenerator
from src.optimizer.context_optimizer import ContextOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    config = load_config("configs/config.yaml")
    
    # We want to force fixed_rag to use k=10 for condition A
    config.retrieval.baseline_k = 10
    config.retrieval.k_complex = 10 # should already be 10

    dataset_path = "datasets/splits/test_mini_10.jsonl"
    examples = []
    with open(dataset_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
            
    logger.info(f"Loaded {len(examples)} examples for M12 ablation.")

    retriever = Retriever()
    generator = ResponseGenerator(config)
    compressor = ContextOptimizer(config)

    baselines = ["fixed_rag", "always_compress"]
    
    for baseline in baselines:
        exp_name = f"m12/ablation_{baseline}"
        logger.info(f"Running condition: {baseline}")
        
        pipeline = Pipeline(
            config=config,
            baseline=baseline,
            retriever=retriever,
            generator=generator,
            compressor=compressor
        )
        
        run_benchmark(
            pipeline_fn=pipeline.run,
            examples=examples,
            config=config,
            experiment_name=exp_name
        )
        
        metadata_path = Path("results") / exp_name / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump({
                "dataset": dataset_path,
                "sample_count": len(examples),
                "condition": "WITHOUT compression" if baseline == "fixed_rag" else "WITH compression",
                "k_value": 10
            }, f, indent=2)

if __name__ == "__main__":
    main()
