import torch
from typing import List, Optional
import sys
import os
from pathlib import Path

# Ensure 'src' is importable if run as a script directly
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("embeddings")

class Embedder:
    def __init__(self, config_path: str = "configs/config.yaml"):
        # Resolve path relative to project root if possible
        if not Path(config_path).is_absolute():
            config_path = str(Path(parent_dir) / config_path)
            
        config = load_config(config_path)
        self.model_name = config.models.embeddings.name
        self.device = config.runtime.device
        self._model = None
        
    def _load(self):
        if self._model is not None:
            return
            
        target_device = self.device
        if target_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available.")
        elif target_device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"Loading embedding model '{self.model_name}' on device '{target_device}'...")
        try:
            import sys
            import os
            # Temporarily remove project root from sys.path to avoid shadowing 'datasets'
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            removed = []
            for p in list(sys.path):
                if p == project_root or p == '' or p == os.getcwd():
                    sys.path.remove(p)
                    removed.append(p)
                    
            from sentence_transformers import SentenceTransformer
            
            for p in reversed(removed):
                sys.path.insert(0, p)
                
            # show_progress_bar is suppressed by default for encode, but loading might show it
            self._model = SentenceTransformer(self.model_name, device=target_device)
            logger.info("Embedding model loaded successfully.")
        except ImportError:
            raise RuntimeError("sentence-transformers is not installed.")
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}")
            
    def embed(self, text: str) -> List[float]:
        """Encodes a single text string into a numerical vector."""
        if not text or not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
            
        self._load()
        vector = self._model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vector.tolist()
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Encodes a list of text strings into a list of numerical vectors."""
        if not texts:
            return []
            
        if not isinstance(texts, list):
            raise TypeError("Input must be a list of strings.")
            
        for t in texts:
            if not t or not isinstance(t, str) or not t.strip():
                raise ValueError("Input batch contains an invalid or empty string.")
                
        self._load()
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()
        
    @property
    def dimensionality(self) -> int:
        self._load()
        return self._model.get_sentence_embedding_dimension()

# Default singleton for simple module-level interface
_default_embedder = None

def get_default_embedder():
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder

def embed(text: str) -> List[float]:
    return get_default_embedder().embed(text)

def embed_batch(texts: List[str]) -> List[List[float]]:
    return get_default_embedder().embed_batch(texts)
