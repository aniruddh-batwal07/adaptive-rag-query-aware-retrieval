from dataclasses import dataclass
import re

@dataclass
class PreprocessedQuery:
    original_query: str
    normalized_query: str

def preprocess_query(query: str) -> PreprocessedQuery:
    """
    Validates and normalizes the input query.
    
    Args:
        query (str): The raw input query.
        
    Returns:
        PreprocessedQuery: A dataclass containing both the original and normalized query.
        
    Raises:
        TypeError: If the query is not a string.
        ValueError: If the query is empty or consists only of whitespace.
    """
    if not isinstance(query, str):
        raise TypeError("Query must be a string.")
    
    stripped = query.strip()
    
    if not stripped:
        raise ValueError("Query cannot be empty or whitespace only.")
        
    # Replace any sequence of whitespace characters with a single space
    normalized = re.sub(r'\s+', ' ', stripped)
    
    return PreprocessedQuery(
        original_query=query,
        normalized_query=normalized
    )
