"""
Simplified document processing pipeline combining document processing, 
chunking, and embedding generation.

This provides a unified interface for processing PDFs and generating embeddings
suitable for ISNE testing.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from document_processor import DocumentProcessor
from chunker import TextChunker
from embedder import TextEmbedder
from doc_types import (
    ProcessedDocument, ChunkingInput, EmbeddingInput, 
    ChunkEmbedding, TextChunk
)


class DocumentProcessingPipeline:
    """
    Unified pipeline for document processing, chunking, and embedding generation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document processing pipeline.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Initialize components
        doc_config = self._config.get('document_processor', {})
        chunk_config = self._config.get('chunker', {})
        embed_config = self._config.get('embedder', {})
        
        self.document_processor = DocumentProcessor(doc_config)
        self.chunker = TextChunker(chunk_config)
        self.embedder = TextEmbedder(embed_config)
        
        # Pipeline statistics
        self._pipeline_stats = {
            "total_files_processed": 0,
            "successful_files": 0,
            "failed_files": 0,
            "total_chunks_created": 0,
            "total_embeddings_created": 0,
            "total_pipeline_time": 0.0
        }
        
        self.logger.info("Initialized document processing pipeline")
    
    def process_file(
        self, 
        file_path: Union[str, Path],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single file through the complete pipeline.
        
        Args:
            file_path: Path to the file to process
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            model_name: Name of the embedding model to use
            
        Returns:
            Dictionary containing processing results
        """
        start_time = datetime.now()
        self._pipeline_stats["total_files_processed"] += 1
        
        try:
            # Step 1: Document processing
            self.logger.info(f"Processing document: {file_path}")
            document = self.document_processor.process_document(file_path)
            
            if document.error:
                self._pipeline_stats["failed_files"] += 1
                return {
                    "success": False,
                    "error": document.error,
                    "file_path": str(file_path),
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Step 2: Text chunking
            self.logger.info(f"Chunking document content ({len(document.content)} characters)")
            chunking_input = ChunkingInput(
                text=document.content,
                document_id=document.id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            chunking_output = self.chunker.chunk(chunking_input)
            
            if chunking_output.errors:
                self._pipeline_stats["failed_files"] += 1
                return {
                    "success": False,
                    "error": f"Chunking failed: {'; '.join(chunking_output.errors)}",
                    "file_path": str(file_path),
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Step 3: Embedding generation
            self.logger.info(f"Generating embeddings for {len(chunking_output.chunks)} chunks")
            embedding_input = EmbeddingInput(
                chunks=chunking_output.chunks,
                model_name=model_name,
                metadata={"source_file": str(file_path)}
            )
            
            embedding_output = self.embedder.embed(embedding_input)
            
            if embedding_output.errors:
                self._pipeline_stats["failed_files"] += 1
                return {
                    "success": False,
                    "error": f"Embedding failed: {'; '.join(embedding_output.errors)}",
                    "file_path": str(file_path),
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Calculate total processing time
            total_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self._pipeline_stats["successful_files"] += 1
            self._pipeline_stats["total_chunks_created"] += len(chunking_output.chunks)
            self._pipeline_stats["total_embeddings_created"] += len(embedding_output.embeddings)
            self._pipeline_stats["total_pipeline_time"] += total_time
            
            # Return complete results
            return {
                "success": True,
                "file_path": str(file_path),
                "document": {
                    "id": document.id,
                    "content_length": len(document.content),
                    "content_type": document.content_type,
                    "format": document.format,
                    "metadata": document.metadata
                },
                "chunks": [
                    {
                        "id": chunk.id,
                        "text": chunk.text,
                        "start_index": chunk.start_index,
                        "end_index": chunk.end_index,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata
                    }
                    for chunk in chunking_output.chunks
                ],
                "embeddings": [
                    {
                        "chunk_id": emb.chunk_id,
                        "embedding": emb.embedding,
                        "dimension": emb.embedding_dimension,
                        "model_name": emb.model_name
                    }
                    for emb in embedding_output.embeddings
                ],
                "processing_stats": {
                    "total_processing_time": total_time,
                    "document_processing_time": document.processing_time,
                    "chunking_stats": chunking_output.processing_stats,
                    "embedding_stats": embedding_output.embedding_stats,
                    "model_info": embedding_output.model_info
                }
            }
            
        except Exception as e:
            error_msg = f"Pipeline processing failed: {str(e)}"
            self.logger.error(error_msg)
            self._pipeline_stats["failed_files"] += 1
            
            return {
                "success": False,
                "error": error_msg,
                "file_path": str(file_path),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    def process_files(
        self,
        file_paths: List[Union[str, Path]],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple files through the pipeline.
        
        Args:
            file_paths: List of file paths to process
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            model_name: Name of the embedding model to use
            
        Returns:
            List of processing results for each file
        """
        results = []
        
        for file_path in file_paths:
            result = self.process_file(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model_name=model_name
            )
            results.append(result)
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline processing statistics."""
        stats = self._pipeline_stats.copy()
        
        # Add component-specific stats
        stats["component_stats"] = {
            "document_processor": self.document_processor.get_stats(),
            "chunker": self.chunker.get_stats(),
            "embedder": self.embedder.get_stats()
        }
        
        # Calculate additional metrics
        if stats["total_files_processed"] > 0:
            stats["success_rate"] = stats["successful_files"] / stats["total_files_processed"]
            stats["avg_processing_time"] = stats["total_pipeline_time"] / stats["total_files_processed"]
        else:
            stats["success_rate"] = 0.0
            stats["avg_processing_time"] = 0.0
        
        if stats["successful_files"] > 0:
            stats["avg_chunks_per_file"] = stats["total_chunks_created"] / stats["successful_files"]
            stats["avg_embeddings_per_file"] = stats["total_embeddings_created"] / stats["successful_files"]
        else:
            stats["avg_chunks_per_file"] = 0.0
            stats["avg_embeddings_per_file"] = 0.0
        
        return stats
    
    def check_health(self) -> Dict[str, Any]:
        """Check the health of all pipeline components."""
        return {
            "pipeline_healthy": True,
            "components": {
                "document_processor": {
                    "docling_available": self.document_processor._docling_available,
                    "supported_formats": self.document_processor.get_supported_formats()
                },
                "chunker": {
                    "initialized": True,
                    "chunk_size": self.chunker._chunk_size,
                    "chunk_overlap": self.chunker._chunk_overlap
                },
                "embedder": {
                    "model_loaded": self.embedder._model_loaded,
                    "model_name": self.embedder._model_name,
                    "sentence_transformers_available": self.embedder._sentence_transformers_available,
                    "embedding_dimension": self.embedder.get_embedding_dimension()
                }
            }
        }
    
    def extract_embeddings_for_isne(
        self, 
        file_path: Union[str, Path],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        model_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract embeddings in a format suitable for ISNE processing.
        
        Args:
            file_path: Path to the file to process
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
            model_name: Name of the embedding model to use
            
        Returns:
            Dictionary with embeddings and metadata for ISNE, or None if failed
        """
        result = self.process_file(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=model_name
        )
        
        if not result["success"]:
            self.logger.error(f"Failed to process file for ISNE: {result.get('error')}")
            return None
        
        # Extract embeddings and create adjacency information
        embeddings = []
        texts = []
        chunk_ids = []
        
        for i, (chunk, embedding) in enumerate(zip(result["chunks"], result["embeddings"])):
            embeddings.append(embedding["embedding"])
            texts.append(chunk["text"])
            chunk_ids.append(chunk["id"])
        
        # Create adjacency matrix based on chunk overlap
        # Adjacent chunks (consecutive) are connected
        adjacency_matrix = []
        for i in range(len(embeddings)):
            row = [0] * len(embeddings)
            # Connect to previous and next chunks
            if i > 0:
                row[i-1] = 1
            if i < len(embeddings) - 1:
                row[i+1] = 1
            adjacency_matrix.append(row)
        
        return {
            "source_file": str(file_path),
            "embeddings": embeddings,
            "texts": texts,
            "chunk_ids": chunk_ids,
            "adjacency_matrix": adjacency_matrix,
            "embedding_dimension": result["embeddings"][0]["dimension"] if embeddings else 0,
            "model_name": result["embeddings"][0]["model_name"] if embeddings else "",
            "num_nodes": len(embeddings),
            "processing_stats": result["processing_stats"]
        }