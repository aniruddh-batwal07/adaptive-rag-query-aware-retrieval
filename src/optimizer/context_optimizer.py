import time
from dataclasses import dataclass
from typing import Optional

from llmlingua import PromptCompressor
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

@dataclass
class OptimizedContext:
    text: str
    original_tokens: int
    compressed_tokens: Optional[int]
    compression_ratio: Optional[float]
    latency_ms: float
    status: str

class ContextOptimizer:
    def __init__(self, config: Config, compressor=None):
        """
        Initializes the ContextOptimizer.
        If compressor is not provided, it instantiates LLMLingua's PromptCompressor.
        """
        self.config = config
        
        # Read the compression budget from configuration, defaulting to 0.5
        budget_config = getattr(config, 'compression', None)
        self.budget = getattr(budget_config, 'budget', 0.5) if budget_config else 0.5
        if self.budget is None:
            self.budget = 0.5
            
        if compressor is not None:
            self.compressor = compressor
        else:
            model_name = config.models.compressor.name
            if config.runtime.device == "cpu":
                device = "cpu"
            else:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
            logger.info(f"Initializing LLMLingua PromptCompressor with model '{model_name}' on {device}...")
            use_llmlingua2 = True
            
            self.compressor = PromptCompressor(
                model_name=model_name,
                device_map=device,
                use_llmlingua2=use_llmlingua2
            )
            logger.info("LLMLingua PromptCompressor initialized successfully.")

    def compress(self, query: str, retrieved_context: str) -> OptimizedContext:
        """
        Compress the retrieved context based on the query using LLMLingua.
        """
        if not isinstance(query, str) or not isinstance(retrieved_context, str):
            raise ValueError("query and retrieved_context must be strings.")
            
        if not retrieved_context.strip():
            return OptimizedContext(
                text="",
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                latency_ms=0.0,
                status="SUCCESS"
            )
            
        start_time = time.time()
        
        try:
            results = self.compressor.compress_prompt(
                context=[retrieved_context],
                instruction="",
                question=query,
                rate=self.budget,
                rank_method="llmlingua"
            )
            
            compressed_text = results.get("compressed_prompt", "")
            
            # Retrieve token counts from LLMLingua's output
            original_tokens = results.get("origin_tokens", len(retrieved_context.split()))
            compressed_tokens = results.get("compressed_tokens", len(compressed_text.split()))
            
            # Ensure safe division
            if compressed_tokens == 0:
                compression_ratio = 0.0 if original_tokens == 0 else float('inf')
            else:
                compression_ratio = original_tokens / compressed_tokens
                
            latency_ms = (time.time() - start_time) * 1000.0
            
            return OptimizedContext(
                text=compressed_text,
                original_tokens=original_tokens,
                compressed_tokens=compressed_tokens,
                compression_ratio=compression_ratio,
                latency_ms=latency_ms,
                status="SUCCESS"
            )
            
        except Exception as e:
            logger.error(f"Context compression failed: {e}")
            latency_ms = (time.time() - start_time) * 1000.0

            return OptimizedContext(
                text=retrieved_context,
                original_tokens=len(retrieved_context.split()),
                compressed_tokens=None,
                compression_ratio=None,
                latency_ms=latency_ms,
                status="FAILED"
            )
