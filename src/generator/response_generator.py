import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

@dataclass
class GeneratedAnswer:
    text: str
    generation_latency_ms: float

class ResponseGenerator:
    def __init__(self, config: Config):
        self.model_name = config.models.generator.name
        self.temperature = config.generation.temperature
        self.max_new_tokens = config.generation.max_new_tokens
        self.seed = config.generation.seed
        
        if config.runtime.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.runtime.device
            
        logger.info(f"Loading generator model {self.model_name} on {self.device}...")
        start_time = time.time()
        
        torch.manual_seed(self.seed)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # If cuda, device_map="auto" will efficiently map it. If cpu, load directly.
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto"
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            
        self.model.eval()
        
        load_time = time.time() - start_time
        logger.info(f"Successfully loaded generator in {load_time * 1000:.2f} ms.")

    def build_prompt(self, query: str, context: str) -> str:
        return f"Question:\n{query}\n\nContext:\n{context}"

    def generate(self, query: str, context: str) -> GeneratedAnswer:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        if not isinstance(context, str):
            raise TypeError("Context must be a string")
            
        prompt = self.build_prompt(query, context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        start_time = time.time()
        
        # Set random seed right before generation to ensure determinism
        torch.manual_seed(self.seed)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=(self.temperature > 0.0),
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        latency = (time.time() - start_time) * 1000.0
        
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        if not text:
            logger.warning("Generator produced an empty string.")
            text = "No answer generated."
            
        return GeneratedAnswer(text=text, generation_latency_ms=latency)
