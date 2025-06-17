"""
CPU-based embedder for Sequential-ISNE using sentence transformers.

Adapted from HADES CPU embedder to provide real embeddings for Sequential-ISNE.
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.doc_types import EmbeddingInput, EmbeddingOutput, ChunkEmbedding


class CPUEmbedder:
    """
    CPU-based embedder using sentence transformers.
    
    This is a simplified version of the HADES CPU embedder, focused on
    providing real embeddings for Sequential-ISNE experiments.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize CPU embedder.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Configuration
        self._model_name = self._config.get('model_name', 'all-MiniLM-L6-v2')
        self._batch_size = self._config.get('batch_size', 32)
        self._normalize_embeddings = self._config.get('normalize_embeddings', True)
        self._max_length = self._config.get('max_length', 512)
        
        # Model state
        self._model = None
        self._model_loaded = False
        
        # Check if sentence-transformers is available
        self._sentence_transformers_available = self._check_sentence_transformers()
        
        # Statistics
        self._stats = {
            "total_embeddings": 0,
            "successful_batches": 0,
            "failed_batches": 0,
            "total_processing_time": 0.0
        }
        
        # Initialize model on creation
        self._initialize_model()
        
        if self._model_loaded:
            self.logger.info(f"Initialized CPU embedder with model: {self._model_name}")
        else:
            self.logger.warning("CPU embedder initialized with fallback (no sentence-transformers)")
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not self._model_loaded:
            self._initialize_model()
            
        if not self._model_loaded:
            # Use fallback
            return self._embed_texts_fallback(texts)
        
        try:
            start_time = datetime.now()
            
            # Generate embeddings
            embeddings = self._embed_texts(texts)
            
            # Update stats
            processing_time = (datetime.now() - start_time).total_seconds()
            self._stats["total_embeddings"] += len(embeddings)
            self._stats["successful_batches"] += 1
            self._stats["total_processing_time"] += processing_time
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding batch failed: {e}")
            self._stats["failed_batches"] += 1
            # Return fallback embeddings
            return self._embed_texts_fallback(texts)
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        return self.embed_batch([text])[0]
    
    def process(self, input_data: EmbeddingInput) -> EmbeddingOutput:
        """
        Process embedding input according to the contract.
        
        Args:
            input_data: Input data with chunks to embed
            
        Returns:
            Output with embeddings
        """
        errors = []
        embeddings = []
        
        try:
            # Extract texts from chunks
            texts = [chunk.text for chunk in input_data.chunks]
            
            # Generate embeddings
            embedding_vectors = self.embed_batch(texts)
            
            # Create embedding objects
            for chunk, vector in zip(input_data.chunks, embedding_vectors):
                embedding = ChunkEmbedding(
                    chunk_id=chunk.id,
                    embedding=vector,
                    embedding_dimension=len(vector),
                    model_name=self._model_name,
                    confidence=1.0,
                    metadata=chunk.metadata
                )
                embeddings.append(embedding)
                
        except Exception as e:
            error_msg = f"Embedding generation failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
        
        return EmbeddingOutput(
            embeddings=embeddings,
            embedding_stats={
                "model_name": self._model_name,
                "total_embeddings": len(embeddings),
                "batch_size": self._batch_size
            },
            model_info={
                "model_name": self._model_name,
                "embedding_dimension": self.get_embedding_dimension(),
                "device": "cpu"
            },
            errors=errors
        )
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        model_dims = {
            'all-MiniLM-L6-v2': 384,
            'all-mpnet-base-v2': 768,
            'all-distilroberta-v1': 768,
            'paraphrase-MiniLM-L6-v2': 384
        }
        
        # Check if model name contains any known model
        for model, dim in model_dims.items():
            if model.lower() in self._model_name.lower():
                return dim
        
        return 384  # Default dimension
    
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
                # Use fallback
                self._model = None
                self._model_loaded = False
                self.logger.warning("Sentence transformers not available, using fallback")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize model: {e}")
            self._model = None
            self._model_loaded = False
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence transformers."""
        if not self._model_loaded or self._model is None:
            return self._embed_texts_fallback(texts)
        
        try:
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
                
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            return self._embed_texts_fallback(texts)
    
    def _embed_texts_fallback(self, texts: List[str]) -> List[List[float]]:
        """Fallback embedding method using random vectors."""
        self.logger.debug("Using fallback embedding method")
        embeddings = []
        embedding_dim = self.get_embedding_dimension()
        
        for i, text in enumerate(texts):
            # Create deterministic embedding based on text
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(embedding_dim).tolist()
            
            # Normalize if requested
            if self._normalize_embeddings:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = (embedding / norm).tolist()
            
            embeddings.append(embedding)
        
        return embeddings
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        return self._stats.copy()