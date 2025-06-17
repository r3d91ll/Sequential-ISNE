"""
Simplified type definitions for document processing pipeline.

These are minimal versions of the HADES contracts, focusing on the essential
functionality needed for PDF processing and ISNE testing.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


# ===== Enums =====

class ContentCategory(str, Enum):
    """Content category enumeration."""
    TEXT = "text"
    DOCUMENT = "document"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Processing status enumeration."""
    SUCCESS = "success"
    ERROR = "error"


# ===== Document Processing Types =====

@dataclass
class ProcessedDocument:
    """Processed document structure."""
    id: str
    content: str
    content_type: str
    format: str
    content_category: ContentCategory
    metadata: Dict[str, Any]
    error: Optional[str] = None
    processing_time: Optional[float] = None


# ===== Chunking Types =====

@dataclass
class TextChunk:
    """Text chunk structure."""
    id: str
    text: str
    start_index: int
    end_index: int
    chunk_index: int
    metadata: Dict[str, Any]


@dataclass
class ChunkingInput:
    """Input for chunking operations."""
    text: str
    document_id: str
    chunk_size: int = 512
    chunk_overlap: int = 50


@dataclass
class ChunkingOutput:
    """Output from chunking operations."""
    chunks: List[TextChunk]
    processing_stats: Dict[str, Any]
    errors: List[str]


# ===== Embedding Types =====

@dataclass
class ChunkEmbedding:
    """Chunk embedding structure."""
    chunk_id: str
    embedding: List[float]
    embedding_dimension: int
    model_name: str
    confidence: float = 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingInput:
    """Input for embedding operations."""
    chunks: List[TextChunk]
    model_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class EmbeddingOutput:
    """Output from embedding operations."""
    embeddings: List[ChunkEmbedding]
    embedding_stats: Dict[str, Any]
    model_info: Dict[str, Any]
    errors: List[str]