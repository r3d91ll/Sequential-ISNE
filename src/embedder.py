"""
Simplified embedder for generating text embeddings.

This is a minimal adaptation of the HADES CPU embedder, focusing on
sentence transformer models for embedding generation.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.doc_types import EmbeddingInput, EmbeddingOutput, ChunkEmbedding


class TextEmbedder:
    """
    Simplified text embedder using sentence transformers.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize text embedder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Configuration settings
        self._model_name = self._config.get('model_name', 'all-MiniLM-L6-v2')
        self._batch_size = self._config.get('batch_size', 32)
        self._max_length = self._config.get('max_length', 512)
        self._normalize_embeddings = self._config.get('normalize_embeddings', True)
        
        # Model components
        self._model = None
        self._model_loaded = False
        
        # Statistics tracking
        self._stats = {
            "total_embeddings_created": 0,
            "successful_embeddings": 0,
            "failed_embeddings": 0,
            "total_processing_time": 0.0
        }
        
        # Check if sentence transformers is available
        self._sentence_transformers_available = self._check_sentence_transformers()
        
        if self._sentence_transformers_available:
            self.logger.info(f"Initialized embedder with model: {self._model_name}")
        else:
            self.logger.warning("sentence-transformers not available - using fallback")
    
    def embed(self, input_data: EmbeddingInput) -> EmbeddingOutput:
        """
        Generate embeddings for text chunks.
        
        Args:
            input_data: Input data with chunks to embed
            
        Returns:
            Output data with embeddings
        """
        errors = []
        
        try:
            start_time = datetime.now()
            
            # Initialize model if needed
            if not self._model_loaded:
                self._initialize_model()
            
            if not self._model_loaded:
                raise ValueError("Model not initialized")
            
            # Extract texts from chunks
            texts = [chunk.text for chunk in input_data.chunks]
            chunk_ids = [chunk.id for chunk in input_data.chunks]
            
            # Generate embeddings
            embedding_vectors = self._embed_texts(texts)
            
            # Convert to output format
            embeddings = []
            for i, (chunk_id, vector) in enumerate(zip(chunk_ids, embedding_vectors)):
                embedding = ChunkEmbedding(
                    chunk_id=chunk_id,
                    embedding=vector.tolist() if hasattr(vector, 'tolist') else vector,
                    embedding_dimension=len(vector),
                    model_name=input_data.model_name or self._model_name,
                    confidence=1.0,
                    metadata=input_data.metadata
                )
                embeddings.append(embedding)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self._stats["total_embeddings_created"] += len(embeddings)
            self._stats["successful_embeddings"] += 1
            self._stats["total_processing_time"] += processing_time
            
            return EmbeddingOutput(
                embeddings=embeddings,
                embedding_stats={
                    "processing_time": processing_time,
                    "embedding_count": len(embeddings),
                    "total_texts": len(texts),
                    "batch_size": self._batch_size,
                    "model_name": self._model_name
                },
                model_info={
                    "model_name": self._model_name,
                    "embedding_dimension": len(embedding_vectors[0]) if embedding_vectors else 0,
                    "max_length": self._max_length,
                    "device": "cpu"
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            # Update error stats
            self._stats["failed_embeddings"] += 1
            
            return EmbeddingOutput(
                embeddings=[],
                embedding_stats={},
                model_info={},
                errors=errors
            )
    
    def estimate_embedding_time(self, input_data: EmbeddingInput) -> float:
        """
        Estimate time to generate embeddings.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated processing time in seconds
        """
        try:
            num_chunks = len(input_data.chunks)
            # Simple estimation: ~0.1 seconds per chunk on CPU
            return max(0.1, num_chunks * 0.1)
        except Exception:
            return 1.0
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        model_dims = {
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768,
            'all-distilroberta-v1': 768,
            'paraphrase-MiniLM-L6-v2': 384
        }
        
        for model, dim in model_dims.items():
            if model in self._model_name.lower():
                return dim
        
        return 384  # Default dimension
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        return self._stats.copy()
    
    def _check_sentence_transformers(self) -> bool:
        """Check if sentence-transformers is available."""
        try:
            import sentence_transformers
            return True
        except ImportError:
            return False
    
    def _initialize_model(self) -> None:
        """Initialize the sentence transformer model."""
        try:
            if self._sentence_transformers_available:
                from sentence_transformers import SentenceTransformer
                
                self.logger.info(f"Loading sentence transformer model: {self._model_name}")
                self._model = SentenceTransformer(self._model_name)
                self._model_loaded = True
                self.logger.info("Model loaded successfully")
            else:
                # Use fallback implementation
                self._model = "fallback"
                self._model_loaded = True
                self.logger.info("Using fallback embedding implementation")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            self._model = "fallback"
            self._model_loaded = True
            self.logger.info("Using fallback embedding implementation")
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not self._model_loaded:
            raise ValueError("Model not initialized")
        
        try:
            if self._sentence_transformers_available and hasattr(self._model, 'encode'):
                # Use sentence transformers
                embeddings = self._model.encode(
                    texts,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    normalize_embeddings=self._normalize_embeddings
                )
                
                # Convert to list format
                if hasattr(embeddings, 'tolist'):
                    return embeddings.tolist()
                else:
                    return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]
            
            else:
                # Use fallback implementation
                return self._embed_texts_fallback(texts)
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            # Return fallback embeddings
            return self._embed_texts_fallback(texts)
    
    def _embed_texts_fallback(self, texts: List[str]) -> List[List[float]]:
        """Fallback embedding method using basic text features."""
        embeddings = []
        
        for text in texts:
            # Simple feature-based embedding
            features = self._extract_text_features(text)
            embeddings.append(features)
        
        return embeddings
    
    def _extract_text_features(self, text: str) -> List[float]:
        """Extract basic text features for fallback embedding."""
        import re
        import math
        
        # Basic text statistics (normalized to create embedding-like vector)
        features = []
        
        # Length features
        features.append(min(len(text) / 1000.0, 1.0))  # Normalized length
        features.append(min(len(text.split()) / 100.0, 1.0))  # Normalized word count
        
        # Character distribution features
        char_counts = {}
        for char in text.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Add features for common characters
        common_chars = 'abcdefghijklmnopqrstuvwxyz0123456789 '
        for char in common_chars:
            freq = char_counts.get(char, 0) / max(len(text), 1)
            features.append(freq)
        
        # Pad or truncate to fixed dimension
        target_dim = self.get_embedding_dimension()
        while len(features) < target_dim:
            features.append(0.0)
        
        features = features[:target_dim]
        
        # Normalize if requested
        if self._normalize_embeddings:
            norm = math.sqrt(sum(f * f for f in features))
            if norm > 0:
                features = [f / norm for f in features]
        
        return features


class MockEmbedder:
    """Mock embedder for testing that generates consistent fake embeddings."""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate mock embeddings for a batch of texts."""
        embeddings = []
        for i, text in enumerate(texts):
            # Create deterministic embedding based on text length and position
            np.random.seed(len(text) + i)
            embedding = np.random.randn(self.embedding_dim).tolist()
            embeddings.append(embedding)
        return embeddings
    
    def embed(self, text: str) -> List[float]:
        """Generate mock embedding for a single text."""
        return self.embed_batch([text])[0]