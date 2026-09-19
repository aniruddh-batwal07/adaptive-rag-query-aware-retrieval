import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class RouterConfig:
    name: str
    checkpoint: Optional[str]

@dataclass
class EmbeddingsConfig:
    name: str

@dataclass
class CompressorConfig:
    name: str

@dataclass
class GeneratorConfig:
    name: str

@dataclass
class ModelsConfig:
    router: RouterConfig
    embeddings: EmbeddingsConfig
    compressor: CompressorConfig
    generator: GeneratorConfig

@dataclass
class RetrievalConfig:
    k_simple: int
    k_complex: int
    baseline_k: int
    chunk_size: int
    chunk_overlap: int

@dataclass
class RoutingConfig:
    threshold: Optional[float]
    fallback_route: str

@dataclass
class CompressionConfig:
    enabled: bool
    budget: Optional[float]

@dataclass
class GenerationConfig:
    temperature: float
    max_new_tokens: int
    seed: int

@dataclass
class EvaluationConfig:
    dataset: str
    split: str
    sample_limit: int
    split_ratios: list[float]

@dataclass
class RuntimeConfig:
    device: str
    batch_size: int

@dataclass
class Config:
    models: ModelsConfig
    retrieval: RetrievalConfig
    routing: RoutingConfig
    compression: CompressionConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig

def validate_config(config: Config):
    # k values
    if config.retrieval.k_simple <= 0:
        raise ValueError("retrieval.k_simple must be positive")
    if config.retrieval.k_complex <= 0:
        raise ValueError("retrieval.k_complex must be positive")
    if config.retrieval.baseline_k <= 0:
        raise ValueError("retrieval.baseline_k must be positive")
    
    # chunk parameters
    if config.retrieval.chunk_size <= 0:
        raise ValueError("retrieval.chunk_size must be positive")
    if config.retrieval.chunk_overlap < 0:
        raise ValueError("retrieval.chunk_overlap must be non-negative")
    if config.retrieval.chunk_overlap >= config.retrieval.chunk_size:
        raise ValueError("retrieval.chunk_overlap must be less than chunk_size")
    
    # generation
    if config.generation.temperature < 0:
        raise ValueError("generation.temperature must be non-negative")
    if config.generation.max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be positive")
    
    # evaluation
    if config.evaluation.sample_limit <= 0:
        raise ValueError("evaluation.sample_limit must be positive")
    if not isinstance(config.evaluation.split_ratios, list) or len(config.evaluation.split_ratios) != 3:
        raise ValueError("evaluation.split_ratios must be a list of 3 floats")
    if not all(isinstance(x, (int, float)) and x >= 0 for x in config.evaluation.split_ratios):
        raise ValueError("evaluation.split_ratios must contain non-negative numbers")
    if abs(sum(config.evaluation.split_ratios) - 1.0) > 1e-6:
        raise ValueError("evaluation.split_ratios must sum to 1.0")
    
    # runtime
    if config.runtime.batch_size <= 0:
        raise ValueError("runtime.batch_size must be positive")

    # type checks
    if not isinstance(config.models.router.name, str):
        raise TypeError("models.router.name must be a string")
    if not isinstance(config.models.embeddings.name, str):
        raise TypeError("models.embeddings.name must be a string")
    if not isinstance(config.models.compressor.name, str):
        raise TypeError("models.compressor.name must be a string")
    if not isinstance(config.models.generator.name, str):
        raise TypeError("models.generator.name must be a string")
    if not isinstance(config.compression.enabled, bool):
        raise TypeError("compression.enabled must be a boolean")
    if not isinstance(config.routing.fallback_route, str):
        raise TypeError("routing.fallback_route must be a string")

def load_config(path: str) -> Config:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Malformed YAML in configuration file: {e}")

    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML dictionary")

    # Required top-level sections
    required_sections = [
        'models', 'retrieval', 'routing', 'compression', 'generation',
        'evaluation', 'runtime'
    ]
    for section in required_sections:
        if section not in data:
            raise KeyError(f"Missing required configuration section: {section}")
            
    # Models
    models_data = data['models']
    for k in ['router', 'embeddings', 'compressor', 'generator']:
        if k not in models_data:
            raise KeyError(f"Missing required field: models.{k}")
        if 'name' not in models_data[k]:
            raise KeyError(f"Missing required field: models.{k}.name")
            
    router_config = RouterConfig(name=models_data['router']['name'], checkpoint=models_data['router'].get('checkpoint'))
    embeddings_config = EmbeddingsConfig(name=models_data['embeddings']['name'])
    compressor_config = CompressorConfig(name=models_data['compressor']['name'])
    generator_config = GeneratorConfig(name=models_data['generator']['name'])
    
    models_config = ModelsConfig(
        router=router_config,
        embeddings=embeddings_config,
        compressor=compressor_config,
        generator=generator_config
    )
    
    # Retrieval
    ret_data = data['retrieval']
    for k in ['k_simple', 'k_complex', 'baseline_k', 'chunk_size', 'chunk_overlap']:
        if k not in ret_data:
            raise KeyError(f"Missing required field: retrieval.{k}")
    retrieval_config = RetrievalConfig(**ret_data)
    
    # Routing
    rout_data = data['routing']
    for k in ['fallback_route']:
        if k not in rout_data:
            raise KeyError(f"Missing required field: routing.{k}")
    routing_config = RoutingConfig(threshold=rout_data.get('threshold'), fallback_route=rout_data['fallback_route'])
    
    # Compression
    comp_data = data['compression']
    for k in ['enabled']:
        if k not in comp_data:
            raise KeyError(f"Missing required field: compression.{k}")
    compression_config = CompressionConfig(enabled=comp_data['enabled'], budget=comp_data.get('budget'))
    
    # Generation
    gen_data = data['generation']
    for k in ['temperature', 'max_new_tokens', 'seed']:
        if k not in gen_data:
            raise KeyError(f"Missing required field: generation.{k}")
    generation_config = GenerationConfig(**gen_data)
    
    # Evaluation
    eval_data = data['evaluation']
    for k in ['dataset', 'split', 'sample_limit', 'split_ratios']:
        if k not in eval_data:
            raise KeyError(f"Missing required field: evaluation.{k}")
    evaluation_config = EvaluationConfig(
        dataset=eval_data['dataset'],
        split=eval_data['split'],
        sample_limit=eval_data['sample_limit'],
        split_ratios=eval_data['split_ratios']
    )
    
    # Runtime
    run_data = data['runtime']
    for k in ['device', 'batch_size']:
        if k not in run_data:
            raise KeyError(f"Missing required field: runtime.{k}")
    runtime_config = RuntimeConfig(**run_data)
    
    config = Config(
        models=models_config,
        retrieval=retrieval_config,
        routing=routing_config,
        compression=compression_config,
        generation=generation_config,
        evaluation=evaluation_config,
        runtime=runtime_config
    )
    
    validate_config(config)
    return config
