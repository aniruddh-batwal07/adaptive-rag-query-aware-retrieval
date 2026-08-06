"""
AdaptiveRAG — Project Entry Point
==================================
Initializes the pipeline, loads configuration, and orchestrates all
placeholder modules. Full implementation is pending.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path for clean imports
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.config import load_config
from src.utils.logger import get_logger

# Initialize the project-wide logger
logger = get_logger("AdaptiveRAG")


def main() -> None:
    """Entry point for the AdaptiveRAG pipeline.

    Steps executed:
        1. Print project information.
        2. Load configuration from ``configs/config.yaml``.
        3. Initialize placeholder modules (not yet implemented).
        4. Report readiness status.
    """
    # ── Step 1: Project Information ──────────────────────────────
    logger.info("=" * 60)
    logger.info("AdaptiveRAG — Adaptive Retrieval-Augmented Generation")
    logger.info("Query Complexity and Context Optimization")
    logger.info("=" * 60)

    # ── Step 2: Load Configuration ───────────────────────────────
    logger.info("Loading configuration from configs/config.yaml ...")
    config = load_config()
    logger.info(
        "Configuration loaded — project: %s v%s",
        config["project"]["name"],
        config["project"]["version"],
    )
    logger.info("LLM            : %s", config["llm"]["model_name"])
    logger.info("Embedding Model: %s", config["embedding"]["model_name"])
    logger.info("Vector DB      : %s", config["vector_db"]["provider"])

    # ── Step 3: Initialize Placeholder Modules ───────────────────
    logger.info("-" * 60)
    logger.info("Initializing pipeline modules ...")

    # TODO: Initialize QueryClassifier from src.router.query_classifier
    logger.warning("[PENDING] QueryClassifier — not yet implemented.")

    # TODO: Initialize AdaptiveRouter from src.router.routing_logic
    logger.warning("[PENDING] AdaptiveRouter  — not yet implemented.")

    # TODO: Initialize Retriever from src.retriever.retriever
    logger.warning("[PENDING] Retriever       — not yet implemented.")

    # TODO: Initialize VectorStore from src.retriever.vector_store
    logger.warning("[PENDING] VectorStore     — not yet implemented.")

    # TODO: Initialize ContextOptimizer from src.optimizer.context_optimizer
    logger.warning("[PENDING] ContextOptimizer— not yet implemented.")

    # TODO: Initialize ResponseGenerator from src.generator.response_generator
    logger.warning("[PENDING] ResponseGenerator— not yet implemented.")

    # ── Step 4: Readiness Status ─────────────────────────────────
    logger.info("-" * 60)
    logger.info(
        "All module stubs loaded. Implementation is pending. "
        "Run `pytest` to verify the project skeleton."
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
