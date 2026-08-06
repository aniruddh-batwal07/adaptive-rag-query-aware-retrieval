"""
Evaluation Metrics for AdaptiveRAG
===================================
Placeholder functions for the evaluation metrics used to benchmark
retrieval quality, answer accuracy, and system efficiency.

Status: NOT IMPLEMENTED — all functions raise ``NotImplementedError``.
"""

from typing import List


def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int,
) -> float:
    """Compute Recall@K for a single query.

    Measures the fraction of relevant documents that appear within
    the top-K retrieved results.

    Args:
        retrieved_ids: Ordered list of retrieved document identifiers.
        relevant_ids: Ground-truth relevant document identifiers.
        k: Cut-off rank.

    Returns:
        Recall@K score in [0.0, 1.0].
    """
    raise NotImplementedError("recall_at_k() is not yet implemented.")


def mean_reciprocal_rank(
    retrieved_ids: List[List[str]],
    relevant_ids: List[List[str]],
) -> float:
    """Compute Mean Reciprocal Rank (MRR) over a set of queries.

    MRR is the average of the reciprocal ranks of the first relevant
    document across all queries.

    Args:
        retrieved_ids: List of retrieved-ID lists, one per query.
        relevant_ids: List of relevant-ID lists, one per query.

    Returns:
        MRR score in [0.0, 1.0].
    """
    raise NotImplementedError("mean_reciprocal_rank() is not yet implemented.")


def ndcg(
    retrieved_ids: List[str],
    relevance_scores: List[float],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain (nDCG) at rank K.

    Evaluates the ranking quality by discounting the relevance of
    results found at lower positions.

    Args:
        retrieved_ids: Ordered list of retrieved document identifiers.
        relevance_scores: Graded relevance scores for each retrieved doc.
        k: Cut-off rank.

    Returns:
        nDCG@K score in [0.0, 1.0].
    """
    raise NotImplementedError("ndcg() is not yet implemented.")


def measure_latency(start_time: float, end_time: float) -> float:
    """Compute end-to-end latency in milliseconds.

    Args:
        start_time: Timestamp (seconds) at query ingestion.
        end_time: Timestamp (seconds) at response completion.

    Returns:
        Latency in milliseconds.
    """
    raise NotImplementedError("measure_latency() is not yet implemented.")


def compute_token_usage(prompt: str, response: str) -> dict:
    """Compute token usage statistics for a single query–response pair.

    Args:
        prompt: The full prompt string sent to the LLM.
        response: The generated response string.

    Returns:
        A dictionary with keys ``prompt_tokens``, ``response_tokens``,
        and ``total_tokens``.
    """
    raise NotImplementedError("compute_token_usage() is not yet implemented.")
