"""
Simplified text chunker for document processing pipeline.

This is a minimal adaptation of the HADES CPU chunker, focusing on
sentence-aware chunking functionality.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.doc_types import ChunkingInput, ChunkingOutput, TextChunk


class TextChunker:
    """
    Simplified text chunker with sentence-aware chunking.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize text chunker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Configuration settings
        self._chunk_size = self._config.get('chunk_size', 512)
        self._chunk_overlap = self._config.get('chunk_overlap', 50)
        self._min_chunk_size = self._config.get('min_chunk_size', 100)
        self._preserve_sentence_boundaries = self._config.get('preserve_sentence_boundaries', True)
        
        # Statistics tracking
        self._stats = {
            "total_chunks_created": 0,
            "successful_chunks": 0,
            "failed_chunks": 0,
            "total_processing_time": 0.0
        }
        
        self.logger.info(f"Initialized text chunker with chunk_size={self._chunk_size}")
    
    def chunk(self, input_data: ChunkingInput) -> ChunkingOutput:
        """
        Chunk text into smaller pieces.
        
        Args:
            input_data: Input data with text to chunk
            
        Returns:
            Output data with text chunks
        """
        errors = []
        chunks = []
        
        try:
            start_time = datetime.now()
            
            # Use input parameters
            chunk_size = input_data.chunk_size
            chunk_overlap = input_data.chunk_overlap
            text = input_data.text
            
            # Perform chunking
            if self._preserve_sentence_boundaries:
                chunks = self._sentence_aware_chunking(text, input_data)
            else:
                chunks = self._fixed_chunking(text, input_data)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self._stats["total_chunks_created"] += len(chunks)
            self._stats["successful_chunks"] += len(chunks)
            self._stats["total_processing_time"] += processing_time
            
            return ChunkingOutput(
                chunks=chunks,
                processing_stats={
                    "processing_time": processing_time,
                    "chunks_created": len(chunks),
                    "characters_processed": len(text),
                    "chunk_size_used": chunk_size,
                    "chunk_overlap_used": chunk_overlap
                },
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Chunking failed: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            
            # Track error statistics
            self._stats["failed_chunks"] += 1
            
            return ChunkingOutput(
                chunks=[],
                processing_stats={},
                errors=errors
            )
    
    def estimate_chunks(self, input_data: ChunkingInput) -> int:
        """
        Estimate number of chunks that will be generated.
        
        Args:
            input_data: Input data to estimate for
            
        Returns:
            Estimated number of chunks
        """
        try:
            text_length = len(input_data.text)
            chunk_size = input_data.chunk_size
            chunk_overlap = input_data.chunk_overlap
            
            effective_chunk_size = chunk_size - chunk_overlap
            
            if effective_chunk_size <= 0:
                return 1
            
            estimated_chunks = max(1, text_length // effective_chunk_size)
            return estimated_chunks
            
        except Exception as e:
            self.logger.error(f"Failed to estimate chunks: {e}")
            return 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chunking statistics."""
        return self._stats.copy()
    
    def _sentence_aware_chunking(self, text: str, input_data: ChunkingInput) -> List[TextChunk]:
        """Sentence-aware chunking that respects sentence boundaries."""
        sentences = self._split_sentences(text)
        
        chunks = []
        chunk_id = 0
        current_chunk_sentences = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > input_data.chunk_size and current_chunk_sentences:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk_sentences)
                start_idx = text.find(current_chunk_sentences[0])
                end_idx = start_idx + len(chunk_text)
                
                chunk = TextChunk(
                    id=f"{input_data.document_id}_chunk_{chunk_id}",
                    text=chunk_text,
                    start_index=start_idx,
                    end_index=end_idx,
                    chunk_index=chunk_id,
                    metadata={
                        "chunking_method": "sentence_aware",
                        "sentence_count": len(current_chunk_sentences)
                    }
                )
                
                chunks.append(chunk)
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk_sentences, input_data.chunk_overlap)
                current_chunk_sentences = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk_sentences)
            else:
                current_chunk_sentences.append(sentence)
                current_length += sentence_length
        
        # Add remaining sentences as final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            start_idx = text.find(current_chunk_sentences[0]) if current_chunk_sentences[0] in text else 0
            end_idx = start_idx + len(chunk_text)
            
            chunk = TextChunk(
                id=f"{input_data.document_id}_chunk_{chunk_id}",
                text=chunk_text,
                start_index=start_idx,
                end_index=end_idx,
                chunk_index=chunk_id,
                metadata={
                    "chunking_method": "sentence_aware",
                    "sentence_count": len(current_chunk_sentences)
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _fixed_chunking(self, text: str, input_data: ChunkingInput) -> List[TextChunk]:
        """Fixed-size chunking with character boundaries."""
        chunks = []
        chunk_id = 0
        start_idx = 0
        
        while start_idx < len(text):
            end_idx = min(start_idx + input_data.chunk_size, len(text))
            chunk_text = text[start_idx:end_idx]
            
            if len(chunk_text.strip()) < self._min_chunk_size and start_idx > 0:
                # Merge small chunks with previous one
                if chunks:
                    chunks[-1].text += " " + chunk_text
                    chunks[-1].end_index = end_idx
                break
            
            chunk = TextChunk(
                id=f"{input_data.document_id}_chunk_{chunk_id}",
                text=chunk_text,
                start_index=start_idx,
                end_index=end_idx,
                chunk_index=chunk_id,
                metadata={
                    "chunking_method": "fixed",
                    "chunk_size": input_data.chunk_size,
                    "overlap": input_data.chunk_overlap
                }
            )
            
            chunks.append(chunk)
            chunk_id += 1
            
            # Move to next chunk with overlap
            start_idx += (input_data.chunk_size - input_data.chunk_overlap)
            
            if end_idx >= len(text):
                break
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex."""
        # Basic sentence boundary detection
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_size: int) -> List[str]:
        """Get sentences for overlap based on overlap size."""
        if not sentences:
            return []
        
        # Calculate how many sentences to include in overlap
        overlap_length = 0
        overlap_sentences = []
        
        # Start from the end and work backwards
        for sentence in reversed(sentences):
            if overlap_length + len(sentence) <= overlap_size:
                overlap_sentences.insert(0, sentence)
                overlap_length += len(sentence)
            else:
                break
        
        return overlap_sentences