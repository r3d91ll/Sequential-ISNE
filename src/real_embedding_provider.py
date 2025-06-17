"""
Real embedding provider using sentence transformers for Sequential-ISNE.
"""

import numpy as np
from typing import List
from src.embeddings import EmbeddingProvider
from src.cpu_embedder import CPUEmbedder


class RealEmbeddingProvider(EmbeddingProvider):
    """
    Real embedding provider using CPUEmbedder with sentence transformers.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self._model_name = model_name
        self._embedder = CPUEmbedder({'model_name': model_name})
        self._dimension = self._embedder.get_embedding_dimension()
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        embeddings = self._embedder.embed_batch(texts)
        return np.array(embeddings)
    
    @property
    def embedding_dimension(self) -> int:
        """Return the dimensionality of embeddings."""
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Return the model name for reproducibility."""
        return self._model_name