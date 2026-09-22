import pytest
from src.utils.config import Config
from src.router.routing_logic import AdaptiveController

from src.utils.config import load_config

from src.utils.config import load_config

@pytest.fixture
def mock_config():
    config = load_config("configs/config.yaml")
    config.retrieval.k_simple = 2
    config.retrieval.k_complex = 10
    config.routing.fallback_route = "SIMPLE"
    return config

def test_controller_simple(mock_config):
    controller = AdaptiveController(mock_config)
    decision = controller.get_decision({"complexity_label": "SIMPLE", "confidence": 0.9})
    assert decision.complexity == "SIMPLE"
    assert decision.retrieval_k == 2
    assert decision.compression_enabled is False

def test_controller_complex(mock_config):
    controller = AdaptiveController(mock_config)
    decision = controller.get_decision({"complexity_label": "COMPLEX", "confidence": 0.9})
    assert decision.complexity == "COMPLEX"
    assert decision.retrieval_k == 10
    assert decision.compression_enabled is True

def test_controller_fallback(mock_config):
    controller = AdaptiveController(mock_config)
    decision = controller.get_decision({"complexity_label": None, "confidence": None})
    assert decision.complexity == "SIMPLE"
    assert decision.retrieval_k == 2
    assert decision.compression_enabled is False

def test_controller_fallback_complex(mock_config):
    mock_config.routing.fallback_route = "COMPLEX"
    controller = AdaptiveController(mock_config)
    decision = controller.get_decision({"complexity_label": None, "confidence": None})
    assert decision.complexity == "COMPLEX"
    assert decision.retrieval_k == 10
    assert decision.compression_enabled is True
