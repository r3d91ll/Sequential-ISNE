#!/usr/bin/env python3
"""
Document Types for Sequential-ISNE

Simple document type definitions for the Sequential-ISNE pipeline.
Minimal types extracted from HADES for compatibility.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path


class ContentCategory(Enum):
    """Content category enumeration."""
    DOCUMENTATION = "documentation"
    CODE = "code"
    PDF = "pdf"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass
class ProcessedDocument:
    """Processed document representation."""
    file_path: str
    file_type: str
    content: str
    metadata: Dict[str, Any]
    category: ContentCategory = ContentCategory.UNKNOWN
    error: Optional[str] = None
    processed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'file_path': self.file_path,
            'file_type': self.file_type,
            'content': self.content,
            'metadata': self.metadata,
            'category': self.category.value if self.category else 'unknown',
            'error': self.error,
            'processed_at': self.processed_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessedDocument':
        """Create from dictionary."""
        category = ContentCategory.UNKNOWN
        if data.get('category'):
            try:
                category = ContentCategory(data['category'])
            except ValueError:
                category = ContentCategory.UNKNOWN
        
        return cls(
            file_path=data.get('file_path', ''),
            file_type=data.get('file_type', 'unknown'),
            content=data.get('content', ''),
            metadata=data.get('metadata', {}),
            category=category,
            error=data.get('error'),
            processed_at=data.get('processed_at')
        )


def classify_file_category(file_path: Path) -> ContentCategory:
    """Classify file category based on extension."""
    extension = file_path.suffix.lower()
    
    if extension == '.py':
        return ContentCategory.CODE
    elif extension in {'.md', '.txt', '.rst'}:
        return ContentCategory.DOCUMENTATION
    elif extension == '.pdf':
        return ContentCategory.PDF
    elif extension in {'.doc', '.docx', '.txt'}:
        return ContentCategory.TEXT
    else:
        return ContentCategory.UNKNOWN