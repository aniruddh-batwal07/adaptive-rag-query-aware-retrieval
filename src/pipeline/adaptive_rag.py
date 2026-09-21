import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from src.utils.logger import get_logger
from src.utils.config import Config
from src.utils.preprocessor import preprocess_query, PreprocessedQuery
from src.retriever.retriever import Retriever
from src.generator.response_generator import ResponseGenerator, GeneratedAnswer

logger = get_logger(__name__)

@dataclass
class ExecutionMetadata:
    query: str
    retrieval_k: int
    retrieval_latency_ms: Optional[float]
    generation_latency_ms: float
    total_latency_ms: float
    original_query: str

@dataclass
class PipelineResult:
    answer: str
    execution_metadata: ExecutionMetadata

class Pipeline:
    def __init__(self, config: Config, retriever: Retriever = None, generator: ResponseGenerator = None, baseline: str = "default"):
        """
        Initializes the pipeline with required components.
        If retriever or generator are not provided, they are instantiated based on config.
        """
        self.config = config
        self.baseline = baseline
        if self.baseline != "llm_only":
            self.retriever = retriever if retriever is not None else Retriever()
        else:
            self.retriever = None
        self.generator = generator if generator is not None else ResponseGenerator(config)
        self.baseline_k = config.retrieval.baseline_k
        
    def run(self, query: str) -> PipelineResult:
        """
        Runs the end-to-end pipeline:
        Preprocessor -> Retriever -> Context Assembly -> Generator
        """
        start_time = time.time()
        
        # 1. Preprocess
        preprocessed: PreprocessedQuery = preprocess_query(query)
        normalized_query = preprocessed.normalized_query
        
        if self.baseline == "llm_only":
            retrieval_latency_ms = None
            retrieval_k = 0
            context = ""
        else:
            # 2. Retrieval
            retrieval_start = time.time()
            retrieved_chunks = self.retriever.retrieve(normalized_query, top_k=self.baseline_k)
            retrieval_latency_ms = (time.time() - retrieval_start) * 1000.0
            retrieval_k = self.baseline_k
            
            # 3. Context Assembly
            # Construct the context string by joining chunk texts deterministically in rank order
            context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
            
        # 4. Generation
        # Note: The generator's generate method will build the prompt and run inference
        generated_answer: GeneratedAnswer = self.generator.generate(normalized_query, context)
        
        total_latency_ms = (time.time() - start_time) * 1000.0
        
        metadata = ExecutionMetadata(
            query=normalized_query,
            retrieval_k=retrieval_k,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generated_answer.generation_latency_ms,
            total_latency_ms=total_latency_ms,
            original_query=preprocessed.original_query
        )
        
        return PipelineResult(
            answer=generated_answer.text,
            execution_metadata=metadata
        )
