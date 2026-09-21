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
    original_query: str
    complexity_label: Optional[str]
    router_confidence: Optional[float]
    retrieval_k: int
    retrieval_latency_ms: Optional[float]
    compression_applied: bool
    original_context_tokens: Optional[int]
    compressed_context_tokens: Optional[int]
    compression_ratio: Optional[float]
    compression_latency_ms: Optional[float]
    generation_latency_ms: float
    total_latency_ms: float
    router_latency_ms: Optional[float]
    error_status: Optional[str]
    retrieved_chunks: List[Dict[str, Any]]

@dataclass
class PipelineResult:
    answer: str
    execution_metadata: ExecutionMetadata

class Pipeline:
    def __init__(self, config: Config, retriever: Retriever = None, generator: ResponseGenerator = None, baseline: str = "fixed_rag", compressor=None):
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

        if self.baseline == "always_compress":
            self.baseline_k = config.retrieval.k_complex
            if compressor is not None:
                self.compressor = compressor
            else:
                from src.optimizer.context_optimizer import ContextOptimizer
                self.compressor = ContextOptimizer(config)
        else:
            self.baseline_k = config.retrieval.baseline_k
            self.compressor = None
        
    def run(self, query: str) -> PipelineResult:
        """
        Runs the end-to-end pipeline:
        Preprocessor -> Retriever -> Context Assembly -> (Compressor) -> Generator
        """
        start_time = time.time()
        
        # 1. Preprocess
        preprocessed: PreprocessedQuery = preprocess_query(query)
        normalized_query = preprocessed.normalized_query
        
        complexity_label = None
        router_confidence = None
        router_latency_ms = None
        compression_applied = False
        original_context_tokens = None
        compressed_context_tokens = None
        compression_ratio = None
        compression_latency_ms = None
        error_status = None

        if self.baseline == "llm_only":
            retrieval_latency_ms = None
            retrieval_k = 0
            context = ""
            retrieved_chunks = []
        else:
            # 2. Retrieval
            retrieval_start = time.time()
            retrieved_chunks = self.retriever.retrieve(normalized_query, top_k=self.baseline_k)
            retrieval_latency_ms = (time.time() - retrieval_start) * 1000.0
            retrieval_k = self.baseline_k
            
            # 3. Context Assembly
            # Construct the context string by joining chunk texts deterministically in rank order
            context = "\n\n".join([chunk["text"] for chunk in retrieved_chunks])
            
            if self.baseline == "always_compress" and len(retrieved_chunks) > 0:
                optimized_context = self.compressor.compress(normalized_query, context)
                context = optimized_context.text
                original_context_tokens = optimized_context.original_tokens
                compressed_context_tokens = optimized_context.compressed_tokens
                compression_ratio = optimized_context.compression_ratio
                compression_latency_ms = optimized_context.latency_ms
                if optimized_context.status == "FAILED":
                    error_status = "compression_failed"
                    compression_applied = False
                else:
                    compression_applied = True

        # 4. Generation
        # Note: The generator's generate method will build the prompt and run inference
        try:
            generated_answer: GeneratedAnswer = self.generator.generate(normalized_query, context)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            error_status = "generation_failed" if error_status is None else f"{error_status}|generation_failed"
            generated_answer = GeneratedAnswer(text="", generation_latency_ms=0.0)
        
        total_latency_ms = (time.time() - start_time) * 1000.0
        
        metadata = ExecutionMetadata(
            query=normalized_query,
            original_query=preprocessed.original_query,
            complexity_label=complexity_label,
            router_confidence=router_confidence,
            retrieval_k=retrieval_k,
            retrieval_latency_ms=retrieval_latency_ms,
            compression_applied=compression_applied,
            original_context_tokens=original_context_tokens,
            compressed_context_tokens=compressed_context_tokens,
            compression_ratio=compression_ratio,
            compression_latency_ms=compression_latency_ms,
            generation_latency_ms=generated_answer.generation_latency_ms,
            total_latency_ms=total_latency_ms,
            router_latency_ms=router_latency_ms,
            error_status=error_status,
            retrieved_chunks=retrieved_chunks
        )
        
        return PipelineResult(
            answer=generated_answer.text,
            execution_metadata=metadata
        )
