"""
Simplified document processor using Docling for PDF processing.

This is a minimal adaptation of the HADES DoclingDocumentProcessor,
focused on essential functionality for PDF processing.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
from doc_types import ProcessedDocument, ContentCategory


class DocumentProcessor:
    """
    Simplified document processor using Docling for PDF and document processing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize document processor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self._config = config or {}
        
        # Try to import Docling
        self._docling_available = self._check_docling_availability()
        
        # Statistics tracking
        self._stats = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_processing_time": 0.0
        }
        
        if self._docling_available:
            self.logger.info("Initialized document processor with Docling support")
        else:
            self.logger.warning("Docling not available - processor will return errors")
    
    def process_document(
        self, 
        file_path: Union[str, Path], 
        **kwargs: Any
    ) -> ProcessedDocument:
        """Process a single document using Docling."""
        start_time = datetime.now()
        
        # Update statistics
        self._stats["total_documents"] += 1
        
        if not self._docling_available:
            self._stats["failed_documents"] += 1
            return self._create_error_document(file_path, "Docling not available")
        
        try:
            # Import Docling
            from docling.document_converter import DocumentConverter
            
            # Create converter
            converter = DocumentConverter()
            
            # Convert document (works for PDF, DOCX, PPTX, HTML, images, etc.)
            conversion_result = converter.convert_single(str(file_path))
            
            # Extract content - use markdown as primary content
            markdown_content = conversion_result.render_as_markdown()
            
            # Extract metadata
            doc_metadata = {
                'processed_by': 'docling',
                'conversion_status': 'SUCCESS',
                'file_path': str(file_path),
                'file_size': Path(file_path).stat().st_size,
                'processed_at': datetime.now().isoformat()
            }
            
            # Add document-specific metadata if available
            if hasattr(conversion_result, 'document'):
                doc = conversion_result.document
                if hasattr(doc, 'pages') and doc.pages:
                    doc_metadata['num_pages'] = len(doc.pages)
                if hasattr(doc, 'title') and doc.title:
                    doc_metadata['title'] = doc.title
                
            # Determine content type and category based on file extension
            file_ext = Path(file_path).suffix.lower()
            content_type_map = {
                '.pdf': 'application/pdf',
                '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                '.doc': 'application/msword',
                '.txt': 'text/plain',
                '.md': 'text/markdown'
            }
            
            content_type = content_type_map.get(file_ext, 'application/octet-stream')
            
            # Calculate processing time and update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._stats["successful_documents"] += 1
            self._stats["total_processing_time"] += processing_time
            
            # Return processed document
            return ProcessedDocument(
                id=Path(file_path).stem,
                content=markdown_content,
                content_type=content_type,
                format=file_ext.lstrip('.'),
                content_category=ContentCategory.DOCUMENT,
                metadata=doc_metadata,
                error=None,
                processing_time=processing_time
            )
            
        except Exception as e:
            # Calculate processing time and update error statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._stats["failed_documents"] += 1
            self._stats["total_processing_time"] += processing_time
            
            error_msg = f"Docling processing failed for {file_path}: {e}"
            self.logger.error(error_msg)
            
            error_doc = self._create_error_document(file_path, str(e))
            error_doc.processing_time = processing_time
            return error_doc
    
    def process_documents(
        self, 
        file_paths: List[Union[str, Path]], 
        **kwargs: Any
    ) -> List[ProcessedDocument]:
        """Process multiple documents."""
        results = []
        for file_path in file_paths:
            result = self.process_document(file_path, **kwargs)
            results.append(result)
        return results
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats that Docling can process."""
        if self._docling_available:
            return [
                '.pdf', '.docx', '.doc', '.pptx', '.ppt', 
                '.xlsx', '.xls', '.html', '.htm', '.txt', '.md'
            ]
        return []
    
    def can_process(self, file_path: str) -> bool:
        """Check if this processor can handle the file."""
        if not self._docling_available:
            return False
        
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.get_supported_formats()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self._stats.copy()
    
    def _check_docling_availability(self) -> bool:
        """Check if Docling is available."""
        try:
            import docling
            return True
        except ImportError:
            return False
    
    def _create_error_document(self, file_path: Union[str, Path], error: str) -> ProcessedDocument:
        """Create an error document."""
        return ProcessedDocument(
            id=f"error_{Path(file_path).name}",
            content="",
            content_type="text/plain",
            format="unknown",
            content_category=ContentCategory.UNKNOWN,
            metadata={"file_path": str(file_path)},
            error=error
        )