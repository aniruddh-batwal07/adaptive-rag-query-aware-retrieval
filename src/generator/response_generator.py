"""
Response Generator for AdaptiveRAG
====================================
Generates grounded answers by feeding optimized context and the
user query to an instruction-tuned LLM.

Status: NOT IMPLEMENTED — contains only class skeleton and TODOs.
"""

from typing import Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ResponseGenerator:
    """Generate answers using an instruction-tuned language model.

    Takes a user query and a retrieved (optionally compressed) context,
    constructs an augmented prompt, and produces a grounded answer.

    Attributes:
        model_name: HuggingFace model identifier for the generator LLM.
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature for generation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the ResponseGenerator.

        Args:
            config: ``llm`` section of the project configuration.
        """
        self.model_name: str = config.get("model_name", "microsoft/Phi-3-mini-4k-instruct")
        self.max_new_tokens: int = config.get("max_new_tokens", 256)
        self.temperature: float = config.get("temperature", 0.3)
        self.model = None      # TODO: Load the language model
        self.tokenizer = None  # TODO: Load the tokenizer

        logger.info(
            "ResponseGenerator initialized (model=%s) — PENDING",
            self.model_name,
        )

    def load_model(self) -> None:
        """Load the LLM and tokenizer from HuggingFace.

        TODO:
            - Load the tokenizer with ``AutoTokenizer``.
            - Load the model with ``AutoModelForCausalLM``.
            - Move to the configured device.
            - Set model to evaluation mode.
        """
        raise NotImplementedError("ResponseGenerator.load_model() is not yet implemented.")

    def generate(self, query: str, context: str) -> str:
        """Generate an answer grounded in the provided context.

        Args:
            query: The user's natural-language question.
            context: Retrieved (and optionally compressed) document text.

        Returns:
            The model-generated answer string.

        TODO:
            - Construct the augmented prompt with context and query.
            - Tokenize and run inference.
            - Decode and return the generated text.
        """
        raise NotImplementedError("ResponseGenerator.generate() is not yet implemented.")

    def build_prompt(self, query: str, context: str) -> str:
        """Construct the instruction-augmented prompt.

        Args:
            query: The user's question.
            context: Supporting context text.

        Returns:
            A formatted prompt string ready for tokenization.

        TODO:
            - Define the prompt template.
            - Insert query and context into the template.
        """
        raise NotImplementedError("ResponseGenerator.build_prompt() is not yet implemented.")
