import time
from typing import Callable, Tuple, Any

def measure_latency(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Tuple[Any, float]:
    """
    Wraps any callable, executing it and returning the result along with the latency in milliseconds.
    Uses time.perf_counter() for high-resolution timing.
    """
    start_time = time.perf_counter()
    result = fn(*args, **kwargs)
    end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000.0
    return result, latency_ms
