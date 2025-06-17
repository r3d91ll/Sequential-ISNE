"""
Simple document processor with fallbacks for text files.

This processor can handle text files (.py, .md, .txt) without external dependencies,
and gracefully degrades for PDFs when docling is not available.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from src.doc_types import ProcessedDocument, ContentCategory


class SimpleDocumentProcessor:
    """
    Simple document processor with text file support and PDF fallbacks.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize simple document processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Try to import Docling for PDF support
        self._docling_available = self._check_docling_availability()
        
        # Statistics tracking
        self._stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_processing_time": 0.0
        }
        
        if self._docling_available:
            self.logger.info("Initialized simple document processor with Docling support")
        else:
            self.logger.info("Initialized simple document processor (text files only, no PDF support)")
    
    def _check_docling_availability(self) -> bool:
        """Check if Docling is available for PDF processing."""
        try:
            import docling
            return True
        except ImportError:
            return False
    
    def process_document(
        self, 
        file_path: Union[str, Path], 
        **kwargs: Any
    ) -> ProcessedDocument:
        """
        Process a document file (text or PDF).
        
        Args:
            file_path: Path to the document file
            **kwargs: Additional processing arguments
            
        Returns:
            ProcessedDocument with content and metadata
        """
        start_time = datetime.now()
        self._stats["total_documents"] += 1
        
        file_path = Path(file_path)
        
        try:
            # Determine file type and process accordingly
            if file_path.suffix.lower() == '.pdf':
                result = self._process_pdf(file_path)
            elif file_path.suffix.lower() in ['.py', '.md', '.txt', '.yaml', '.yml', '.json']:
                result = self._process_text_file(file_path)
            else:
                # Try as text file
                result = self._process_text_file(file_path)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._stats["total_processing_time"] += processing_time
            
            if result.error:
                self._stats["failed_documents"] += 1
            else:
                self._stats["successful_documents"] += 1
            
            return result
            
        except Exception as e:
            self._stats["failed_documents"] += 1
            self.logger.error(f"Error processing {file_path}: {e}")
            
            return ProcessedDocument(
                id=str(file_path),
                content="",
                content_type="error",
                format="unknown",
                content_category=ContentCategory.UNKNOWN,
                metadata={
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size if file_path.exists() else 0,
                    "processing_timestamp": datetime.now().isoformat()
                },
                error=f"Processing failed: {e}"
            )
    
    def _process_text_file(self, file_path: Path) -> ProcessedDocument:
        """
        Process a text-based file (.py, .md, .txt, etc.).
        
        Args:
            file_path: Path to the text file
            
        Returns:
            ProcessedDocument with text content
        """
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine content type
            content_type = self._determine_content_type(file_path)
            
            # Get file metadata
            stat = file_path.stat()
            
            return ProcessedDocument(
                id=str(file_path),
                content=content,
                content_type=content_type,
                format=file_path.suffix.lower(),
                content_category=self._determine_content_category(file_path),
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "file_size": stat.st_size,
                    "modification_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "processing_timestamp": datetime.now().isoformat(),
                    "content_length": len(content),
                    "line_count": content.count('\n') + 1 if content else 0
                },
                error=None
            )
            
        except UnicodeDecodeError as e:
            return ProcessedDocument(
                id=str(file_path),
                content="",
                content_type="error",
                format=file_path.suffix.lower(),
                content_category=ContentCategory.UNKNOWN,
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "processing_timestamp": datetime.now().isoformat()
                },
                error=f"Unicode decode error: {e}"
            )
        except Exception as e:
            return ProcessedDocument(
                id=str(file_path),
                content="",
                content_type="error",
                format=file_path.suffix.lower(),
                content_category=ContentCategory.UNKNOWN,
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "processing_timestamp": datetime.now().isoformat()
                },
                error=f"File reading error: {e}"
            )
    
    def _process_pdf(self, file_path: Path) -> ProcessedDocument:
        """
        Process a PDF file using Docling (following HADES approach).
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ProcessedDocument with PDF content or error
        """
        if not self._docling_available:
            return ProcessedDocument(
                id=str(file_path),
                content="",
                content_type="error",
                format="pdf",
                content_category=ContentCategory.DOCUMENT,
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "processing_timestamp": datetime.now().isoformat()
                },
                error="Docling not available for PDF processing"
            )
        
        try:
            # Import docling here to avoid import errors
            from docling.document_converter import DocumentConverter
            
            # Create converter (following HADES approach)
            converter = DocumentConverter()
            
            # Convert document (works for PDF, DOCX, PPTX, HTML, images, etc.)
            conversion_result = converter.convert_single(str(file_path))
            
            # Extract content - use markdown as primary content (HADES approach)
            markdown_content = conversion_result.render_as_markdown()
            
            # Extract metadata (following HADES metadata extraction)
            doc_metadata = {
                'processed_by': 'docling',
                'conversion_status': 'SUCCESS',
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'processed_at': datetime.now().isoformat()
            }
            
            # Add document-specific metadata if available (docling result has a document attribute)
            if hasattr(conversion_result, 'document'):
                doc = conversion_result.document
                if hasattr(doc, 'pages') and doc.pages:
                    doc_metadata['num_pages'] = len(doc.pages)
                if hasattr(doc, 'title') and doc.title:
                    doc_metadata['title'] = doc.title
                if hasattr(doc, 'author') and doc.author:
                    doc_metadata['author'] = doc.author
                if hasattr(doc, 'creation_date') and doc.creation_date:
                    doc_metadata['creation_date'] = str(doc.creation_date)
            
            # Get file metadata
            stat = file_path.stat()
            
            return ProcessedDocument(
                id=file_path.stem,  # Use stem like HADES
                content=markdown_content,
                content_type="application/pdf",
                format="pdf",
                content_category=ContentCategory.DOCUMENT,
                metadata={
                    **doc_metadata,
                    "file_name": file_path.name,
                    "modification_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "processing_timestamp": datetime.now().isoformat(),
                    "content_length": len(markdown_content)
                },
                error=None
            )
            
        except Exception as e:
            return ProcessedDocument(
                id=str(file_path),
                content="",
                content_type="error",
                format="pdf",
                content_category=ContentCategory.DOCUMENT,
                metadata={
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "processing_timestamp": datetime.now().isoformat()
                },
                error=f"PDF processing failed: {e}"
            )
    
    def _determine_content_type(self, file_path: Path) -> str:
        """
        Determine content type based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Content type string
        """
        extension = file_path.suffix.lower()
        
        if extension == '.py':
            return "code/python"
        elif extension in ['.md', '.markdown']:
            return "text/markdown"
        elif extension in ['.txt', '.text']:
            return "text/plain"
        elif extension in ['.yaml', '.yml']:
            return "text/yaml"
        elif extension == '.json':
            return "application/json"
        else:
            return "text/plain"
    
    def _determine_content_category(self, file_path: Path) -> ContentCategory:
        """
        Determine content category based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            ContentCategory enum value
        """
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return ContentCategory.DOCUMENT
        elif extension in ['.md', '.markdown']:
            return ContentCategory.MARKDOWN
        elif extension in ['.py', '.txt', '.yaml', '.yml', '.json']:
            return ContentCategory.TEXT
        else:
            return ContentCategory.UNKNOWN
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()
    
    def reset_stats(self):
        """Reset processing statistics."""
        for key in self._stats:
            self._stats[key] = 0