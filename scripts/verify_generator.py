import sys
import os

# Add parent dir to path to import src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.generator.response_generator import ResponseGenerator
import time

def main():
    print("Loading config...")
    config = load_config("configs/config.yaml")
    
    print("Initializing Generator...")
    start_load = time.time()
    generator = ResponseGenerator(config)
    load_latency = time.time() - start_load
    print(f"Generator loaded in {load_latency:.2f} seconds.")
    
    query = "What is the capital of France?"
    context = "France is a country in Western Europe. Its capital is Paris, known for the Eiffel Tower."
    
    print("\n--- First Generation ---")
    ans1 = generator.generate(query, context)
    print(f"Text: {ans1.text}")
    print(f"Latency: {ans1.generation_latency_ms:.2f} ms")
    
    print("\n--- Second Generation (Determinism Check) ---")
    ans2 = generator.generate(query, context)
    print(f"Text: {ans2.text}")
    print(f"Latency: {ans2.generation_latency_ms:.2f} ms")
    
    if ans1.text == ans2.text:
        print("\nDeterminism Check: PASSED")
    else:
        print("\nDeterminism Check: FAILED (Outputs differ)")

if __name__ == "__main__":
    main()
