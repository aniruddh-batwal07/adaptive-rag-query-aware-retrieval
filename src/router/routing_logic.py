from dataclasses import dataclass
from typing import Dict, Union

from src.utils.config import Config

@dataclass
class RoutingDecision:
    complexity: str
    retrieval_k: int
    compression_enabled: bool

class AdaptiveController:
    def __init__(self, config: Config):
        self.config = config
        
    def get_decision(self, complexity_result: Dict[str, Union[str, float]]) -> RoutingDecision:
        label = complexity_result.get("complexity_label")
        
        if label == "COMPLEX":
            return RoutingDecision(
                complexity="COMPLEX",
                retrieval_k=self.config.retrieval.k_complex,
                compression_enabled=True
            )
        elif label == "SIMPLE":
            return RoutingDecision(
                complexity="SIMPLE",
                retrieval_k=self.config.retrieval.k_simple,
                compression_enabled=False
            )
        else:
            # Fallback
            fallback = self.config.routing.fallback_route
            if fallback == "COMPLEX":
                return RoutingDecision(
                    complexity="COMPLEX",
                    retrieval_k=self.config.retrieval.k_complex,
                    compression_enabled=True
                )
            else:
                return RoutingDecision(
                    complexity="SIMPLE",
                    retrieval_k=self.config.retrieval.k_simple,
                    compression_enabled=False
                )
