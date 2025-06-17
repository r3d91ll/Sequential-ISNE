#!/usr/bin/env python3
"""
Simple Embedding Interface for Sequential-ISNE

Provides a clean interface for generating semantic embeddings that will be
combined with ISNE structural embeddings in our dual-embedding architecture.

Supports multiple embedding models with a focus on reproducibility and
academic benchmarking.
"""

import logging
import numpy as np
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from pathlib import Path

# Optional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimensionality of embeddings."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name for reproducibility."""
        pass


class SimpleEmbeddingProvider(EmbeddingProvider):
    """
    Simple embedding provider using sentence-transformers.
    
    Provides reproducible semantic embeddings for academic research.
    Default model is optimized for general-purpose semantic similarity.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "auto"):
        """
        Initialize embedding provider.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
            device: Device to run on ("cpu", "cuda", or "auto")
        """
        self._model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        
        logger.info(f"Initializing embedding provider: {model_name} on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if HAS_TORCH:
                return "cuda" if torch.cuda.is_available() else "cpu"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self._model_name, device=self.device)
                logger.info(f"Loaded model {self._model_name} on {self.device}")
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. Install with: pip install sentence-transformers"
                )
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if not texts:
            return np.array([])
        
        self._load_model()
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
            batch_size=32
        )
        
        logger.debug(f"Generated embeddings for {len(texts)} texts: {embeddings.shape}")
        return embeddings
    
    @property
    def embedding_dimension(self) -> int:
        """Return embedding dimension (384 for all-MiniLM-L6-v2)."""
        dimension_map = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "multi-qa-MiniLM-L6-cos-v1": 384,
            "paraphrase-MiniLM-L6-v2": 384
        }
        return dimension_map.get(self._model_name, 384)  # Default to 384
    
    @property
    def model_name(self) -> str:
        """Return model name for reproducibility."""
        return self._model_name


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock embedding provider for testing and development.
    
    Generates random but consistent embeddings based on text hash.
    Useful for testing the Sequential-ISNE architecture without
    requiring heavy embedding models.
    """
    
    def __init__(self, dimension: int = 384, seed: int = 42):
        self.dimension = dimension
        self.seed = seed
        self._model_name = f"mock-{dimension}d"
        
        # Use consistent random generator
        self.rng = np.random.RandomState(seed)
        logger.info(f"Initialized mock embedding provider: {dimension}D, seed={seed}")
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate consistent mock embeddings based on text hash."""
        if not texts:
            return np.array([])
        
        embeddings = []
        for text in texts:
            # Create consistent hash-based seed for each text
            text_hash = hash(text) % (2**31)
            text_rng = np.random.RandomState(text_hash + self.seed)
            
            # Generate unit-normalized random embedding
            embedding = text_rng.randn(self.dimension)
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        result = np.array(embeddings)
        logger.debug(f"Generated mock embeddings for {len(texts)} texts: {result.shape}")
        return result
    
    @property
    def embedding_dimension(self) -> int:
        return self.dimension
    
    @property
    def model_name(self) -> str:
        return self._model_name


class EmbeddingManager:
    """
    Manages embedding generation for Sequential-ISNE chunks.
    
    Handles batching, caching, and provides a simple interface for
    the StreamingChunkProcessor to add semantic embeddings to chunks.
    """
    
    def __init__(
        self, 
        provider: EmbeddingProvider,
        batch_size: int = 32,
        cache_embeddings: bool = True
    ):
        self.provider = provider
        self.batch_size = batch_size
        self.cache_embeddings = cache_embeddings
        
        # Simple in-memory cache
        self._embedding_cache: Dict[str, np.ndarray] = {}
        
        logger.info(
            f"Initialized EmbeddingManager with {provider.model_name}, "
            f"batch_size={batch_size}, caching={cache_embeddings}"
        )
    
    def embed_chunk_contents(self, chunks: List['StreamingChunk']) -> None:
        """
        Add semantic embeddings to chunks in-place.
        
        Args:
            chunks: List of StreamingChunk objects to embed
        """
        if not chunks:
            return
        
        # Extract content for embedding
        texts_to_embed = []
        chunk_indices = []
        
        for i, chunk in enumerate(chunks):
            # Only embed content chunks
            if chunk.metadata.chunk_type == "content":
                content_key = self._get_cache_key(chunk.content)
                
                if self.cache_embeddings and content_key in self._embedding_cache:
                    # Use cached embedding
                    chunk.semantic_embedding = self._embedding_cache[content_key].tolist()
                else:
                    # Need to generate embedding
                    texts_to_embed.append(chunk.content)
                    chunk_indices.append(i)
        
        if not texts_to_embed:
            logger.debug("All embeddings found in cache")
            return
        
        logger.info(f"Generating embeddings for {len(texts_to_embed)} chunks")
        
        # Generate embeddings in batches
        all_embeddings = []
        for start_idx in range(0, len(texts_to_embed), self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_texts = texts_to_embed[start_idx:end_idx]
            
            batch_embeddings = self.provider.embed_texts(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Combine all batches
        if all_embeddings:
            combined_embeddings = np.vstack(all_embeddings)
        else:
            combined_embeddings = np.array([])
        
        # Assign embeddings to chunks and cache
        for i, chunk_idx in enumerate(chunk_indices):
            embedding = combined_embeddings[i]
            chunks[chunk_idx].semantic_embedding = embedding.tolist()
            
            # Cache if enabled
            if self.cache_embeddings:
                content_key = self._get_cache_key(chunks[chunk_idx].content)
                self._embedding_cache[content_key] = embedding
        
        logger.info(f"Added semantic embeddings to {len(chunk_indices)} chunks")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text content."""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding setup."""
        return {
            "provider": self.provider.model_name,
            "dimension": self.provider.embedding_dimension,
            "batch_size": self.batch_size,
            "cache_enabled": self.cache_embeddings,
            "cached_embeddings": len(self._embedding_cache)
        }
    
    def save_cache(self, cache_path: str) -> None:
        """Save embedding cache to disk."""
        if not self.cache_embeddings or not self._embedding_cache:
            logger.warning("No cache to save")
            return
        
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "provider_info": {
                "model_name": self.provider.model_name,
                "dimension": self.provider.embedding_dimension
            },
            "embeddings": {
                key: embedding.tolist() 
                for key, embedding in self._embedding_cache.items()
            }
        }
        
        import json
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        logger.info(f"Saved {len(self._embedding_cache)} cached embeddings to {cache_file}")
    
    def load_cache(self, cache_path: str) -> None:
        """Load embedding cache from disk."""
        cache_file = Path(cache_path)
        if not cache_file.exists():
            logger.warning(f"Cache file not found: {cache_file}")
            return
        
        import json
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        # Validate provider compatibility
        saved_model = cache_data.get("provider_info", {}).get("model_name")
        if saved_model != self.provider.model_name:
            logger.warning(
                f"Cache model mismatch: saved={saved_model}, current={self.provider.model_name}"
            )
            return
        
        # Load embeddings
        self._embedding_cache = {
            key: np.array(embedding) 
            for key, embedding in cache_data.get("embeddings", {}).items()
        }
        
        logger.info(f"Loaded {len(self._embedding_cache)} cached embeddings from {cache_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo embedding generation
    test_texts = [
        "def authenticate_user(username, password):",
        "JWT token validation process",
        "Authentication module documentation",
        "Test cases for token validation"
    ]
    
    # Test with mock provider
    print("Testing with MockEmbeddingProvider:")
    mock_provider = MockEmbeddingProvider(dimension=384)
    mock_embeddings = mock_provider.embed_texts(test_texts)
    print(f"Generated embeddings shape: {mock_embeddings.shape}")
    print(f"Sample embedding norm: {np.linalg.norm(mock_embeddings[0]):.3f}")
    
    # Test embedding manager
    print("\nTesting EmbeddingManager:")
    from streaming_processor import StreamingChunk, ChunkMetadata
    
    # Create sample chunks
    chunks = [
        StreamingChunk(
            chunk_id=i,
            content=text,
            metadata=ChunkMetadata(
                chunk_id=i,
                chunk_type="content",
                doc_path=f"test_{i}.py",
                directory="test",
                processing_order=i
            )
        )
        for i, text in enumerate(test_texts)
    ]
    
    manager = EmbeddingManager(mock_provider)
    manager.embed_chunk_contents(chunks)
    
    # Check results
    embedded_count = sum(1 for chunk in chunks if chunk.semantic_embedding is not None)
    print(f"Embedded {embedded_count}/{len(chunks)} chunks")
    print(f"Embedding info: {manager.get_embedding_info()}")