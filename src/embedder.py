#!/usr/bin/env python3
"""
CodeBERT Embedder for Sequential-ISNE

Single embedder using CodeBERT for both text and code modalities.
CodeBERT can handle both natural language (research papers) and programming code,
ensuring embedding alignment for theory-practice bridge detection.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Union, Optional

# Optional transformers import
try:
    from transformers import AutoTokenizer, AutoModel  # type: ignore
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class CodeBERTEmbedder:
    """
    Unified embedder using CodeBERT for both text and code content.
    
    CodeBERT is pre-trained on both natural language and programming languages,
    making it ideal for creating comparable embeddings across modalities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Model configuration
        self.model_name = self.config.get('model_name', 'microsoft/codebert-base')
        self.max_length = self.config.get('max_length', 512)
        self.normalize_embeddings = self.config.get('normalize', True)
        self.device = self.config.get('device', 'cpu')
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
        logger.info(f"CodeBERT embedder initialized: {self.model_name}")
        logger.info(f"Device: {self.device}, Max length: {self.max_length}")
    
    def _initialize_model(self):
        """Initialize CodeBERT model and tokenizer."""
        if not HAS_TRANSFORMERS:
            logger.warning("Transformers not available, using fallback embeddings")
            return
        
        try:
            # Set device
            if self.device == 'auto':
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded CodeBERT model on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load CodeBERT model: {e}")
            self.model = None
            self.tokenizer = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        Works for both natural language and code.
        """
        if not self.model or not self.tokenizer:
            return self._fallback_embedding(text)
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            
            # Normalize if requested
            if self.normalize_embeddings:
                embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.warning(f"CodeBERT embedding failed for text: {e}")
            return self._fallback_embedding(text)
    
    def embed_batch(self, texts: List[str], batch_size: int = 8) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts.
        Processes in smaller batches for memory efficiency.
        """
        if not texts:
            return []
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch_internal(batch)
            embeddings.extend(batch_embeddings)
            
            if i % (batch_size * 10) == 0:
                logger.debug(f"Processed {i + len(batch)}/{len(texts)} embeddings")
        
        logger.info(f"Generated {len(embeddings)} CodeBERT embeddings")
        return embeddings
    
    def _embed_batch_internal(self, texts: List[str]) -> List[np.ndarray]:
        """Internal batch embedding method."""
        if not self.model or not self.tokenizer:
            return [self._fallback_embedding(text) for text in texts]
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embeddings
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Normalize if requested
            if self.normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            
            return [embedding for embedding in embeddings]
            
        except Exception as e:
            logger.warning(f"Batch CodeBERT embedding failed: {e}")
            return [self._fallback_embedding(text) for text in texts]
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to chunk dictionaries.
        Handles both code and text chunks with the same CodeBERT model.
        """
        if not chunks:
            return chunks
        
        # Extract content for embedding
        texts = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_batch(texts)
        
        # Add embeddings to chunks
        enhanced_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            enhanced_chunk = chunk.copy()
            enhanced_chunk['embedding'] = embedding
            enhanced_chunk['embedding_model'] = self.model_name
            enhanced_chunk['embedding_dimension'] = len(embedding)
            enhanced_chunks.append(enhanced_chunk)
        
        logger.info(f"Added CodeBERT embeddings to {len(enhanced_chunks)} chunks")
        return enhanced_chunks
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Fallback embedding when CodeBERT is not available.
        Creates a simple hash-based embedding for consistency.
        """
        # Use text hash for reproducible fallback embedding
        import hashlib
        
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to numerical values
        embedding_dim = 768  # Match CodeBERT dimension
        np.random.seed(int(text_hash[:8], 16))  # Reproducible seed from hash
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        
        # Normalize
        if self.normalize_embeddings:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """Get information about the embedding configuration."""
        return {
            'model_name': self.model_name,
            'embedding_dimension': 768,  # CodeBERT standard dimension
            'max_length': self.max_length,
            'normalize_embeddings': self.normalize_embeddings,
            'device': self.device,
            'model_available': self.model is not None,
            'handles_code': True,
            'handles_text': True,
            'cross_modal_comparable': True
        }
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Ensure embeddings are normalized
        if self.normalize_embeddings:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 > 0 and norm2 > 0:
                embedding1 = embedding1 / norm1
                embedding2 = embedding2 / norm2
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def find_similar_chunks(
        self, 
        query_chunk: Dict[str, Any], 
        candidate_chunks: List[Dict[str, Any]], 
        top_k: int = 5,
        similarity_threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find chunks similar to a query chunk using CodeBERT embeddings.
        Perfect for theory-practice bridge detection.
        """
        if 'embedding' not in query_chunk:
            logger.warning("Query chunk missing embedding")
            return []
        
        query_embedding = query_chunk['embedding']
        similarities = []
        
        for candidate in candidate_chunks:
            if 'embedding' not in candidate:
                continue
            
            similarity = self.calculate_similarity(query_embedding, candidate['embedding'])
            
            if similarity >= similarity_threshold:
                result = candidate.copy()
                result['similarity_score'] = similarity
                similarities.append(result)
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
        return similarities[:top_k]


if __name__ == "__main__":
    # Demo the CodeBERT embedder
    logging.basicConfig(level=logging.INFO)
    
    embedder = CodeBERTEmbedder()
    
    # Test with both code and text
    code_sample = '''
def calculate_sequential_isne(chunks, relationships):
    """Calculate Sequential-ISNE embeddings for streaming chunks."""
    embeddings = {}
    for chunk in chunks:
        embedding = process_chunk_sequential(chunk)
        embeddings[chunk.id] = embedding
    return embeddings
'''
    
    text_sample = '''
Sequential-ISNE is a novel approach to graph neural network embeddings that processes 
chunks in sequential order. The algorithm maintains global consistency while enabling 
streaming updates, making it suitable for large-scale document and code analysis.
'''
    
    print("=== CodeBERT Embedding Demo ===")
    
    # Generate embeddings
    code_embedding = embedder.embed_text(code_sample)
    text_embedding = embedder.embed_text(text_sample)
    
    print(f"Code embedding shape: {code_embedding.shape}")
    print(f"Text embedding shape: {text_embedding.shape}")
    
    # Calculate cross-modal similarity
    similarity = embedder.calculate_similarity(code_embedding, text_embedding)
    print(f"Code-Text similarity: {similarity:.3f}")
    
    # Show embedding info
    info = embedder.get_embedding_info()
    print("\n=== Embedding Configuration ===")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Test batch processing
    print("\n=== Batch Processing ===")
    batch_texts = [code_sample, text_sample, "Another test text", "def test(): pass"]
    batch_embeddings = embedder.embed_batch(batch_texts)
    print(f"Generated {len(batch_embeddings)} batch embeddings")