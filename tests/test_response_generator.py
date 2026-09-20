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
        
    mock_tokenizer_instance = mock_tokenizer.return_value
    mock_input_ids = MagicMock()
    mock_input_ids.shape = (1, 5)
    mock_tokenizer_instance.return_value = MagicMock(to=MagicMock(return_value={"input_ids": mock_input_ids}))
    mock_model_instance = mock_model.return_value
    mock_model_instance.generate.return_value = [[0] * 10]
    mock_tokenizer_instance.decode.return_value = "Ans"
    
    ans = generator.generate("Query", "")
    assert ans.text == "Ans"

@patch("src.generator.response_generator.AutoModelForCausalLM.from_pretrained")
@patch("src.generator.response_generator.AutoTokenizer.from_pretrained")
def test_error_propagation(mock_tokenizer, mock_model, mock_config):
    mock_tokenizer.side_effect = Exception("Tokenizer failed")
    
    with pytest.raises(Exception, match="Tokenizer failed"):
        ResponseGenerator(mock_config)
