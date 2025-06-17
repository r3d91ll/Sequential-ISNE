"""
Simplified File Processor for Sequential-ISNE

Routes files to appropriate processors and normalizes output to JSON format.
Stripped down from HADES components to just the essentials.
"""

import ast
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from src.doc_types import ProcessedDocument, ContentCategory


class FileProcessor:
    """
    Simple file processor that routes files to appropriate handlers and normalizes output.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize docling if available
        self._docling_available = self._check_docling()
        if self._docling_available:
            from docling.document_converter import DocumentConverter
            self._docling_converter = DocumentConverter()
        
        # File type routing
        self._supported_types = {
            'python': ['.py'],
            'documents': ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.html', '.htm'],
            'text': ['.md', '.txt', '.yaml', '.yml', '.json', '.rst'],
            'images': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
        }
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a file and return normalized JSON format.
        
        Args:
            file_path: Path to file to process
            
        Returns:
            Dictionary with normalized content and metadata
        """
        file_path = Path(file_path)
        file_type = self._detect_file_type(file_path)
        
        try:
            if file_type == 'python':
                return self._process_python_file(file_path)
            elif file_type == 'documents' or file_type == 'images':
                return self._process_document_file(file_path)
            elif file_type == 'text':
                return self._process_text_file(file_path)
            else:
                return self._create_error_result(file_path, f"Unsupported file type: {file_path.suffix}")
                
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            return self._create_error_result(file_path, str(e))
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type based on extension."""
        ext = file_path.suffix.lower()
        
        for file_type, extensions in self._supported_types.items():
            if ext in extensions:
                return file_type
        
        return 'unknown'
    
    def _process_python_file(self, file_path: Path) -> Dict[str, Any]:
        """Process Python file using AST."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            # Parse AST
            tree = ast.parse(source_code)
            
            # Extract basic information
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node),
                        'args': [arg.arg for arg in node.args.args]
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node),
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    })
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'name': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append({
                            'name': f"{node.module}.{alias.name}" if node.module else alias.name,
                            'alias': alias.asname,
                            'line': node.lineno,
                            'from_module': node.module
                        })
            
            # Create chunks for different elements
            chunks = []
            
            # Add file-level chunk
            chunks.append({
                'type': 'file_overview',
                'content': source_code[:500],  # First 500 chars
                'metadata': {
                    'functions_count': len(functions),
                    'classes_count': len(classes),
                    'imports_count': len(imports)
                }
            })
            
            # Add function chunks
            for func in functions:
                chunks.append({
                    'type': 'function',
                    'content': f"Function: {func['name']}\nArgs: {func['args']}\nDocstring: {func['docstring'] or 'None'}",
                    'metadata': func
                })
            
            # Add class chunks
            for cls in classes:
                chunks.append({
                    'type': 'class',
                    'content': f"Class: {cls['name']}\nMethods: {cls['methods']}\nDocstring: {cls['docstring'] or 'None'}",
                    'metadata': cls
                })
            
            return {
                'file_path': str(file_path),
                'file_type': 'python',
                'processing_method': 'ast',
                'content_format': 'structured',
                'original_content': source_code,
                'processed_content': {
                    'functions': functions,
                    'classes': classes,
                    'imports': imports
                },
                'chunks': chunks,
                'metadata': {
                    'file_size': len(source_code),
                    'line_count': source_code.count('\n') + 1,
                    'processed_at': datetime.now().isoformat()
                },
                'error': None
            }
            
        except Exception as e:
            return self._create_error_result(file_path, f"Python AST parsing failed: {e}")
    
    def _process_document_file(self, file_path: Path) -> Dict[str, Any]:
        """Process document file using docling."""
        if not self._docling_available:
            return self._create_error_result(file_path, "Docling not available for document processing")
        
        try:
            # Convert with docling
            result = self._docling_converter.convert_single(str(file_path))
            markdown_content = result.render_as_markdown()
            
            # Simple chunking - split by paragraphs or sections
            chunks = []
            paragraphs = [p.strip() for p in markdown_content.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 50:  # Only meaningful chunks
                    chunks.append({
                        'type': 'paragraph',
                        'content': paragraph,
                        'metadata': {
                            'chunk_index': i,
                            'char_count': len(paragraph)
                        }
                    })
            
            return {
                'file_path': str(file_path),
                'file_type': 'document',
                'processing_method': 'docling',
                'content_format': 'markdown',
                'original_content': None,  # Don't store binary content
                'processed_content': markdown_content,
                'chunks': chunks,
                'metadata': {
                    'file_size': file_path.stat().st_size,
                    'content_length': len(markdown_content),
                    'chunk_count': len(chunks),
                    'processed_at': datetime.now().isoformat()
                },
                'error': None
            }
            
        except Exception as e:
            return self._create_error_result(file_path, f"Document processing failed: {e}")
    
    def _process_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Process text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple chunking by lines or paragraphs
            chunks = []
            if file_path.suffix.lower() in ['.md', '.rst']:
                # For markdown, split by headers or paragraphs
                sections = [s.strip() for s in content.split('\n\n') if s.strip()]
            else:
                # For other text, split by lines
                sections = [line.strip() for line in content.split('\n') if line.strip()]
            
            for i, section in enumerate(sections):
                if len(section) > 20:  # Only meaningful chunks
                    chunks.append({
                        'type': 'text_section',
                        'content': section,
                        'metadata': {
                            'chunk_index': i,
                            'char_count': len(section)
                        }
                    })
            
            return {
                'file_path': str(file_path),
                'file_type': 'text',
                'processing_method': 'text_parser',
                'content_format': 'text',
                'original_content': content,
                'processed_content': content,
                'chunks': chunks,
                'metadata': {
                    'file_size': len(content),
                    'line_count': content.count('\n') + 1,
                    'chunk_count': len(chunks),
                    'processed_at': datetime.now().isoformat()
                },
                'error': None
            }
            
        except Exception as e:
            return self._create_error_result(file_path, f"Text processing failed: {e}")
    
    def _create_error_result(self, file_path: Path, error_msg: str) -> Dict[str, Any]:
        """Create error result."""
        return {
            'file_path': str(file_path),
            'file_type': 'unknown',
            'processing_method': 'none',
            'content_format': 'none',
            'original_content': None,
            'processed_content': None,
            'chunks': [],
            'metadata': {
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'processed_at': datetime.now().isoformat()
            },
            'error': error_msg
        }
    
    def _check_docling(self) -> bool:
        """Check if docling is available."""
        try:
            import docling
            return True
        except ImportError:
            self.logger.warning("Docling not available - PDF processing will be limited")
            return False
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of all supported file extensions."""
        extensions = []
        for ext_list in self._supported_types.values():
            extensions.extend(ext_list)
        return extensions