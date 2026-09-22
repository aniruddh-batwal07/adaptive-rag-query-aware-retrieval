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

    def compress(self, query: str, retrieved_context) -> OptimizedContext:
        """
        Compress the retrieved context based on the query using LLMLingua.
        Accepts either a single string or a list of strings (chunks).
        """
        if not isinstance(query, str):
            raise ValueError("query must be a string.")

        if isinstance(retrieved_context, str):
            chunks = [retrieved_context] if retrieved_context.strip() else []
        elif isinstance(retrieved_context, list):
            chunks = [c for c in retrieved_context if str(c).strip()]
        else:
            raise ValueError("retrieved_context must be a string or a list of strings.")

        if not chunks:
            return OptimizedContext(
                text="",
                original_tokens=0,
                compressed_tokens=0,
                compression_ratio=1.0,
                latency_ms=0.0,
                status="SUCCESS"
            )

        start_time = time.time()

        total_original_tokens = 0
        total_compressed_tokens = 0
        compressed_chunks = []

        try:
            for chunk in chunks:
                results = self.compressor.compress_prompt(
                    context=[chunk],
                    instruction="",
                    question=query,
                    rate=self.budget,
                    rank_method="llmlingua"
                )

                compressed_text = results.get("compressed_prompt", "")
                orig_toks = results.get("origin_tokens", len(chunk.split()))
                comp_toks = results.get("compressed_tokens", len(compressed_text.split()))

                compressed_chunks.append(compressed_text)
                total_original_tokens += orig_toks
                total_compressed_tokens += comp_toks

            final_text = "\n\n".join(compressed_chunks)

            # Ensure safe division
            if total_compressed_tokens == 0:
                compression_ratio = 0.0 if total_original_tokens == 0 else float('inf')
            else:
                compression_ratio = total_original_tokens / total_compressed_tokens

            latency_ms = (time.time() - start_time) * 1000.0

            return OptimizedContext(
                text=final_text,
                original_tokens=total_original_tokens,
                compressed_tokens=total_compressed_tokens,
                compression_ratio=compression_ratio,
                latency_ms=latency_ms,
                status="SUCCESS"
            )

        except Exception as e:
            logger.error(f"Context compression failed: {e}")
            latency_ms = (time.time() - start_time) * 1000.0

            fallback_text = "\n\n".join(chunks)

            return OptimizedContext(
                text=fallback_text,
                original_tokens=len(fallback_text.split()),
                compressed_tokens=None,
                compression_ratio=None,
                latency_ms=latency_ms,
                status="FAILED"
            )
