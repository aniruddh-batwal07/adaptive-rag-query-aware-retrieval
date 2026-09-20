import pytest
from src.utils.preprocessor import preprocess_query, PreprocessedQuery

def test_normal_query():
    query = "What is the capital of France?"
    result = preprocess_query(query)
    
    assert isinstance(result, PreprocessedQuery)
    assert result.original_query == query
    assert result.normalized_query == "What is the capital of France?"

def test_leading_trailing_whitespace():
    query = "   What is the capital of France?   "
    result = preprocess_query(query)
    
    assert result.original_query == query
    assert result.normalized_query == "What is the capital of France?"

def test_repeated_internal_whitespace():
    query = "What   is \t the \n capital of France?"
    result = preprocess_query(query)
    
    assert result.original_query == query
    assert result.normalized_query == "What is the capital of France?"

def test_empty_string():
    with pytest.raises(ValueError, match="Query cannot be empty or whitespace only."):
        preprocess_query("")

def test_whitespace_only_string():
    with pytest.raises(ValueError, match="Query cannot be empty or whitespace only."):
        preprocess_query("   \t\n  ")

def test_non_string_input():
    with pytest.raises(TypeError, match="Query must be a string."):
        preprocess_query(123)
        
    with pytest.raises(TypeError, match="Query must be a string."):
        preprocess_query(None)
        
    with pytest.raises(TypeError, match="Query must be a string."):
        preprocess_query(["What is X?"])

def test_original_query_preservation():
    query = "   Preserve  me! \t "
    result = preprocess_query(query)
    assert result.original_query == query

def test_determinism():
    query = "  What   is   the capital of France?  "
    result1 = preprocess_query(query)
    result2 = preprocess_query(query)
    
    assert result1.original_query == result2.original_query
    assert result1.normalized_query == result2.normalized_query

def test_semantic_preservation():
    # Only formatting changes, words/punctuation remain
    query = "\t  Hello ,   world ! \n How are you ?  "
    result = preprocess_query(query)
    
    assert result.normalized_query == "Hello , world ! How are you ?"
    assert result.original_query == query
