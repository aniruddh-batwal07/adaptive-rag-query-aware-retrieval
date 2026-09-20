import pytest
from unittest.mock import patch, MagicMock
from src.generator.response_generator import ResponseGenerator, GeneratedAnswer
from src.utils.config import Config, ModelsConfig, GenerationConfig, RuntimeConfig, GeneratorConfig

@pytest.fixture
def mock_config():
    return Config(
        models=ModelsConfig(
            router=MagicMock(),
            embeddings=MagicMock(),
            compressor=MagicMock(),
            generator=GeneratorConfig(name="mock/model-name")
        ),
        retrieval=MagicMock(),
        routing=MagicMock(),
        compression=MagicMock(),
        generation=GenerationConfig(
            temperature=0.7,
            max_new_tokens=100,
            seed=42
        ),
        evaluation=MagicMock(),
        runtime=RuntimeConfig(
            device="cpu",
            batch_size=1
        )
    )

@patch("src.generator.response_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.generator.response_generator.AutoTokenizer.from_pretrained")
def test_generator_init(mock_tokenizer, mock_model, mock_config):
    generator = ResponseGenerator(mock_config)
    
    assert generator.model_name == "mock/model-name"
    mock_model.assert_called_once_with("mock/model-name")
    mock_tokenizer.assert_called_once_with("mock/model-name")

@patch("src.generator.response_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.generator.response_generator.AutoTokenizer.from_pretrained")
def test_generation_output(mock_tokenizer, mock_model, mock_config):
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance
    
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer.return_value = mock_tokenizer_instance
    mock_tokenizer_instance.pad_token = "[PAD]"
    mock_tokenizer_instance.pad_token_id = 0
    mock_tokenizer_instance.eos_token_id = 1
    
    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 5)
    mock_inputs = {"input_ids": mock_input_ids}
    mock_tokenizer_instance.return_value = MagicMock(to=MagicMock(return_value=mock_inputs))
    
    mock_model_instance.generate.return_value = [list(range(10))]
    mock_tokenizer_instance.decode.return_value = "Mocked answer"
    
    generator = ResponseGenerator(mock_config)
    
    ans = generator.generate("What is X?", "X is 42.")
    
    assert isinstance(ans, GeneratedAnswer)
    assert ans.text == "Mocked answer"
    assert ans.generation_latency_ms > 0
    
    mock_model_instance.generate.assert_called_once()
    kwargs = mock_model_instance.generate.call_args[1]
    assert kwargs["max_new_tokens"] == 100
    assert kwargs["temperature"] == 0.7
    assert kwargs["do_sample"] is True
    
    ans2 = generator.generate("What is Y?", "Y is 43.")
    assert mock_model.call_count == 1

@patch("src.generator.response_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.generator.response_generator.AutoTokenizer.from_pretrained")
def test_invalid_input(mock_tokenizer, mock_model, mock_config):
    generator = ResponseGenerator(mock_config)
    
    with pytest.raises(TypeError):
        generator.generate(123, "Context")
        
    with pytest.raises(TypeError):
        generator.generate("Query", 123)

@patch("src.generator.response_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.generator.response_generator.AutoTokenizer.from_pretrained")
def test_error_propagation(mock_tokenizer, mock_model, mock_config):
    mock_tokenizer.side_effect = Exception("Tokenizer failed")
    
    with pytest.raises(Exception, match="Tokenizer failed"):
        ResponseGenerator(mock_config)

@patch("src.generator.response_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.generator.response_generator.AutoTokenizer.from_pretrained")
def test_build_prompt_exact_assembly(mock_tokenizer, mock_model, mock_config):
    generator = ResponseGenerator(mock_config)
    
    query = "What is the capital of France?"
    context = "Paris is the capital of France."
    
    expected_prompt = "Question:\nWhat is the capital of France?\n\nContext:\nParis is the capital of France."
    actual_prompt = generator.build_prompt(query, context)
    
    assert actual_prompt == expected_prompt

@patch("src.generator.response_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.generator.response_generator.AutoTokenizer.from_pretrained")
def test_build_prompt_preservation(mock_tokenizer, mock_model, mock_config):
    generator = ResponseGenerator(mock_config)
    
    query = "Query with $pecial chars & numbers 123"
    context = "Context with \n newlines and \t tabs"
    
    actual_prompt = generator.build_prompt(query, context)
    
    assert query in actual_prompt
    assert context in actual_prompt

@patch("src.generator.response_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.generator.response_generator.AutoTokenizer.from_pretrained")
def test_build_prompt_empty_context(mock_tokenizer, mock_model, mock_config):
    generator = ResponseGenerator(mock_config)
    
    query = "What is X?"
    context = ""
    
    expected_prompt = "Question:\nWhat is X?\n\nContext:\n"
    actual_prompt = generator.build_prompt(query, context)
    
    assert actual_prompt == expected_prompt

@patch("src.generator.response_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.generator.response_generator.AutoTokenizer.from_pretrained")
def test_build_prompt_determinism(mock_tokenizer, mock_model, mock_config):
    generator = ResponseGenerator(mock_config)
    
    query = "What is X?"
    context = "X is 42."
    
    prompt1 = generator.build_prompt(query, context)
    prompt2 = generator.build_prompt(query, context)
    
    assert prompt1 == prompt2

@patch("src.generator.response_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.generator.response_generator.AutoTokenizer.from_pretrained")
def test_build_prompt_simple_vs_complex_invariance(mock_tokenizer, mock_model, mock_config):
    generator = ResponseGenerator(mock_config)
    
    # Simulate SIMPLE path (short context)
    query_simple = "What is X?"
    context_simple = "X is 42."
    prompt_simple = generator.build_prompt(query_simple, context_simple)
    
    # Simulate COMPLEX path (long context)
    query_complex = "What is X?"
    context_complex = "X is 42. It is a number. " * 10
    prompt_complex = generator.build_prompt(query_complex, context_complex)
    
    # Verify the template structure is identical (the prefixes are exactly the same)
    assert prompt_simple.startswith(f"Question:\n{query_simple}\n\nContext:\n")
    assert prompt_complex.startswith(f"Question:\n{query_complex}\n\nContext:\n")

@patch("src.generator.response_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.generator.response_generator.AutoTokenizer.from_pretrained")
def test_generator_integration_with_prompt_builder(mock_tokenizer, mock_model, mock_config):
    generator = ResponseGenerator(mock_config)
    
    mock_tokenizer_instance = mock_tokenizer.return_value
    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 5)
    mock_tokenizer_instance.return_value = MagicMock(to=MagicMock(return_value={"input_ids": mock_input_ids}))
    
    mock_model_instance = mock_model.return_value
    mock_model_instance.generate.return_value = [[0] * 10]
    mock_tokenizer_instance.decode.return_value = "Ans"
    
    query = "Q"
    context = "C"
    
    # Spy on build_prompt
    with patch.object(generator, 'build_prompt', wraps=generator.build_prompt) as spy_build_prompt:
        generator.generate(query, context)
        
        # Verify build_prompt was called with correct arguments
        spy_build_prompt.assert_called_once_with(query, context)
        
        # Verify tokenizer was called with the assembled prompt
        expected_prompt = generator.build_prompt(query, context)
        mock_tokenizer_instance.assert_any_call(expected_prompt, return_tensors="pt")
