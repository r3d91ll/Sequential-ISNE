#!/usr/bin/env python3
"""
Unified Chunker for Sequential-ISNE

Combines Chonky for text/markdown chunking and custom AST for Python code chunking.
This demonstrates our complete chunking contribution as part of the data normalization pipeline.
"""

import ast
import logging
from typing import List, Dict, Any
from pathlib import Path

# Optional Chonky import for text chunking
try:
    from chonkie import TokenChunker
    HAS_CHONKY = True
except ImportError:
    HAS_CHONKY = False

logger = logging.getLogger(__name__)


class UnifiedChunker:
    """
    Unified chunker that routes content to appropriate chunking strategy:
    - Python code → AST-based chunking
    - Text/Markdown → Chonky token-based chunking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Text chunking configuration
        self.text_chunk_size = self.config.get('text_chunk_size', 512)
        self.text_overlap = self.config.get('text_overlap', 50)
        
        # Python chunking configuration
        self.python_chunk_functions = self.config.get('python_chunk_functions', True)
        self.python_chunk_classes = self.config.get('python_chunk_classes', True)
        
        # Initialize Chonky if available
        if HAS_CHONKY:
            self.text_chunker = TokenChunker(
                chunk_size=self.text_chunk_size,
                chunk_overlap=self.text_overlap
            )
            logger.info("Initialized Chonky text chunker")
        else:
            self.text_chunker = None
            logger.warning("Chonky not available, using fallback text chunking")
    
    def chunk_content(self, content: str, file_type: str, source_file: str = "") -> List[Dict[str, Any]]:
        """
        Chunk content based on file type.
        
        Args:
            content: Raw content to chunk
            file_type: 'python' or 'text'/'markdown'
            source_file: Original file path for metadata
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        if file_type == 'python':
            return self._chunk_python_code(content, source_file)
        else:
            return self._chunk_text_content(content, source_file)
    
    def _chunk_python_code(self, code: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Chunk Python code using AST analysis.
        Extracts functions, classes, and top-level statements as separate chunks.
        """
        chunks = []
        
        try:
            tree = ast.parse(code)
            
            # Track processed lines to avoid duplication
            processed_lines = set()
            
            # Extract classes and functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        start_line = node.lineno
                        end_line = node.end_lineno or start_line
                        
                        # Skip if already processed
                        if any(line in processed_lines for line in range(start_line, end_line + 1)):
                            continue
                        
                        # Extract source lines
                        code_lines = code.split('\n')
                        if start_line <= len(code_lines):
                            chunk_content = '\n'.join(code_lines[start_line-1:end_line])
                            
                            chunk_type = 'function' if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else 'class'
                            
                            chunks.append({
                                'content': chunk_content.strip(),
                                'metadata': {
                                    'source_file': source_file,
                                    'chunk_type': chunk_type,
                                    'name': node.name,
                                    'start_line': start_line,
                                    'end_line': end_line,
                                    'chunking_method': 'ast'
                                }
                            })
                            
                            # Mark lines as processed
                            for line in range(start_line, end_line + 1):
                                processed_lines.add(line)
            
            # Extract remaining top-level code (imports, constants, etc.)
            code_lines = code.split('\n')
            unprocessed_content = []
            
            for i, line in enumerate(code_lines, 1):
                if i not in processed_lines and line.strip():
                    unprocessed_content.append(line)
            
            if unprocessed_content:
                chunks.append({
                    'content': '\n'.join(unprocessed_content).strip(),
                    'metadata': {
                        'source_file': source_file,
                        'chunk_type': 'module_level',
                        'name': 'module_imports_and_constants',
                        'chunking_method': 'ast'
                    }
                })
            
            logger.debug(f"AST chunking extracted {len(chunks)} chunks from {source_file}")
            
        except SyntaxError as e:
            logger.warning(f"Failed to parse Python code in {source_file}: {e}")
            # Fallback to text chunking for malformed Python
            return self._chunk_text_content(code, source_file)
        
        return chunks
    
    def _chunk_text_content(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Chunk text/markdown content using Chonky or fallback method.
        """
        chunks = []
        
        if HAS_CHONKY and self.text_chunker:
            # Use Chonky for intelligent text chunking
            try:
                chonky_chunks = self.text_chunker.chunk(text)
                
                for i, chunk in enumerate(chonky_chunks):
                    chunks.append({
                        'content': chunk.text,
                        'metadata': {
                            'source_file': source_file,
                            'chunk_type': 'text',
                            'chunk_index': i,
                            'start_index': chunk.start_index,
                            'end_index': chunk.end_index,
                            'chunking_method': 'chonky'
                        }
                    })
                
                logger.debug(f"Chonky chunking created {len(chunks)} chunks from {source_file}")
                
            except Exception as e:
                logger.warning(f"Chonky chunking failed for {source_file}: {e}, using fallback")
                chunks = self._fallback_text_chunking(text, source_file)
        else:
            # Fallback text chunking
            chunks = self._fallback_text_chunking(text, source_file)
        
        return chunks
    
    def _fallback_text_chunking(self, text: str, source_file: str) -> List[Dict[str, Any]]:
        """
        Simple fallback text chunking by sentences and paragraphs.
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, start new chunk
            if len(current_chunk) + len(paragraph) > self.text_chunk_size and current_chunk:
                chunks.append({
                    'content': current_chunk.strip(),
                    'metadata': {
                        'source_file': source_file,
                        'chunk_type': 'text',
                        'chunk_index': chunk_index,
                        'chunking_method': 'fallback'
                    }
                })
                current_chunk = paragraph
                chunk_index += 1
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': {
                    'source_file': source_file,
                    'chunk_type': 'text',
                    'chunk_index': chunk_index,
                    'chunking_method': 'fallback'
                }
            })
        
        logger.debug(f"Fallback chunking created {len(chunks)} chunks from {source_file}")
        return chunks
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunking results."""
        if not chunks:
            return {'total_chunks': 0}
        
        stats = {
            'total_chunks': len(chunks),
            'chunking_methods': {},
            'chunk_types': {},
            'avg_chunk_length': 0,
            'total_content_length': 0
        }
        
        total_length = 0
        for chunk in chunks:
            content_length = len(chunk['content'])
            total_length += content_length
            
            method = chunk['metadata'].get('chunking_method', 'unknown')
            chunk_type = chunk['metadata'].get('chunk_type', 'unknown')
            
            stats['chunking_methods'][method] = stats['chunking_methods'].get(method, 0) + 1
            stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
        
        stats['total_content_length'] = total_length
        stats['avg_chunk_length'] = total_length / len(chunks) if chunks else 0
        
        return stats


if __name__ == "__main__":
    # Demo the unified chunker
    logging.basicConfig(level=logging.INFO)
    
    chunker = UnifiedChunker()
    
    # Test Python code chunking
    python_code = '''
import os
import sys

def hello_world():
    """A simple greeting function."""
    return "Hello, World!"

class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

if __name__ == "__main__":
    calc = Calculator()
    print(calc.add(2, 3))
'''
    
    print("=== Python Code Chunking ===")
    python_chunks = chunker.chunk_content(python_code, 'python', 'demo.py')
    for i, chunk in enumerate(python_chunks):
        print(f"Chunk {i+1} ({chunk['metadata']['chunk_type']}):")
        print(f"  Content: {chunk['content'][:100]}...")
        print(f"  Method: {chunk['metadata']['chunking_method']}")
        print()
    
    # Test text chunking
    text_content = '''
# Sequential-ISNE: A Novel Approach

Sequential-ISNE represents a breakthrough in graph neural network embeddings. This method processes chunks in sequential order, maintaining global consistency while enabling streaming updates.

## Key Innovations

The algorithm introduces several key innovations:

1. **Streaming Processing**: Chunks are processed as they arrive
2. **Global Sequential IDs**: Maintains consistent node mapping
3. **Theory-Practice Bridges**: Links research concepts to code implementations

## Applications

This approach has applications in code analysis, document understanding, and knowledge graph construction.
'''
    
    print("=== Text Chunking ===")
    text_chunks = chunker.chunk_content(text_content, 'text', 'paper.md')
    for i, chunk in enumerate(text_chunks):
        print(f"Chunk {i+1}:")
        print(f"  Content: {chunk['content'][:100]}...")
        print(f"  Method: {chunk['metadata']['chunking_method']}")
        print()
    
    # Show statistics
    all_chunks = python_chunks + text_chunks
    stats = chunker.get_chunking_stats(all_chunks)
    print("=== Chunking Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")