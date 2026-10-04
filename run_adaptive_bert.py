import json
import logging
from pathlib import Path
from src.utils.config import load_config
from src.pipeline.adaptive_rag import Pipeline
from src.evaluation.benchmark import run_benchmark
from src.retriever.retriever import Retriever
from src.generator.response_generator import ResponseGenerator
from src.optimizer.context_optimizer import ContextOptimizer
from src.router.query_classifier import QueryClassifier
from src.router.routing_logic import AdaptiveController

logging.basicConfig(level=logging.INFO)

def main():
    config = load_config("configs/config_bert_router.yaml")
    dataset_path = "datasets/splits/test_mini_10.jsonl"
    
    examples = []
    with open(dataset_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
            
    print(f"Loaded {len(examples)} examples.")
    
    retriever = Retriever()
    generator = ResponseGenerator(config)
    compressor = ContextOptimizer(config)
    router_path = config.models.router.checkpoint
    print(f"Router path: {router_path}")
    router = QueryClassifier(router_path, device=config.runtime.device)
    controller = AdaptiveController(config)
    
    pipeline = Pipeline(
        config=config, 
        baseline="adaptive",
        retriever=retriever,
        generator=generator,
        compressor=compressor,
        router=router,
        controller=controller
    )

    exp_name = "router_comparison/adaptive_bert"
    result = run_benchmark(
        pipeline_fn=pipeline.run, 
        examples=examples, 
        config=config,
        experiment_name=exp_name
    )
    
    print("Benchmark complete!")
    print(json.dumps(result.metrics, indent=2))

if __name__ == "__main__":
    main()
