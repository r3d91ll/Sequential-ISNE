"""
Data Normalizer for Sequential-ISNE

Takes processed file data and normalizes it into the standardized JSON format
that Sequential-ISNE requires for consistent chunk processing.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from src.file_processor import FileProcessor


class DataNormalizer:
    """
    Normalizes all file types into a consistent JSON format for Sequential-ISNE.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.file_processor = FileProcessor()
    
    def normalize_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Normalize a single file into Sequential-ISNE format.
        
        Args:
            file_path: Path to file to normalize
            
        Returns:
            Normalized data structure
        """
        # Process the file
        processed = self.file_processor.process_file(file_path)
        
        # Normalize to Sequential-ISNE format
        normalized = {
            'document_id': self._generate_document_id(file_path),
            'source': {
                'file_path': processed['file_path'],
                'file_name': Path(file_path).name,
                'file_type': processed['file_type'],
                'file_extension': Path(file_path).suffix.lower(),
                'processing_method': processed['processing_method']
            },
            'content': {
                'format': processed['content_format'],
                'raw_content': processed.get('original_content'),
                'processed_content': processed['processed_content'],
                'content_summary': self._create_content_summary(processed)
            },
            'chunks': self._normalize_chunks(processed['chunks'], file_path),
            'metadata': {
                **processed['metadata'],
                'normalization_timestamp': datetime.now().isoformat(),
                'supported_file_type': processed['file_type'] in ['python', 'document', 'text'],
                'chunk_count': len(processed['chunks'])
            },
            'error': processed.get('error')
        }
        
        return normalized
    
    def normalize_directory(self, directory_path: Union[str, Path], 
                          recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Normalize all supported files in a directory.
        
        Args:
            directory_path: Path to directory
            recursive: Whether to process subdirectories
            
        Returns:
            List of normalized documents
        """
        directory_path = Path(directory_path)
        results = []
        
        # Get all files
        if recursive:
            files = directory_path.rglob('*')
        else:
            files = directory_path.glob('*')
        
        supported_extensions = self.file_processor.get_supported_extensions()
        
        for file_path in files:
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    normalized = self.normalize_file(file_path)
                    results.append(normalized)
                except Exception as e:
                    self.logger.error(f"Failed to normalize {file_path}: {e}")
                    # Add error document
                    results.append(self._create_error_document(file_path, str(e)))
        
        return results
    
    def _generate_document_id(self, file_path: Union[str, Path]) -> str:
        """Generate a unique document ID."""
        file_path = Path(file_path)
        # Use relative path as ID for consistency
        return str(file_path).replace('/', '_').replace('\\', '_')
    
    def _create_content_summary(self, processed: Dict[str, Any]) -> Dict[str, Any]:
        """Create a content summary for the document."""
        summary = {
            'has_content': bool(processed.get('processed_content')),
            'content_length': 0,
            'chunk_count': len(processed.get('chunks', [])),
            'file_type': processed['file_type']
        }
        
        # Add type-specific summary info
        if processed['file_type'] == 'python':
            content = processed.get('processed_content', {})
            summary.update({
                'functions': len(content.get('functions', [])),
                'classes': len(content.get('classes', [])),
                'imports': len(content.get('imports', []))
            })
        elif processed['file_type'] in ['document', 'text']:
            content = processed.get('processed_content', '')
            if isinstance(content, str):
                summary['content_length'] = len(content)
                summary['estimated_reading_time'] = len(content) // 1000  # rough estimate
        
        return summary
    
    def _normalize_chunks(self, chunks: List[Dict[str, Any]], 
                         file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Normalize chunks to Sequential-ISNE format."""
        normalized_chunks = []
        
        for i, chunk in enumerate(chunks):
            normalized_chunk = {
                'chunk_id': f"{self._generate_document_id(file_path)}_chunk_{i}",
                'chunk_index': i,
                'chunk_type': chunk.get('type', 'content'),
                'content': chunk['content'],
                'metadata': {
                    **chunk.get('metadata', {}),
                    'source_file': str(file_path),
                    'chunk_size': len(chunk['content']),
                    'normalization_timestamp': datetime.now().isoformat()
                }
            }
            normalized_chunks.append(normalized_chunk)
        
        return normalized_chunks
    
    def _create_error_document(self, file_path: Union[str, Path], 
                             error_msg: str) -> Dict[str, Any]:
        """Create error document for failed processing."""
        return {
            'document_id': self._generate_document_id(file_path),
            'source': {
                'file_path': str(file_path),
                'file_name': Path(file_path).name,
                'file_type': 'unknown',
                'file_extension': Path(file_path).suffix.lower(),
                'processing_method': 'failed'
            },
            'content': {
                'format': 'none',
                'raw_content': None,
                'processed_content': None,
                'content_summary': {'has_content': False, 'content_length': 0}
            },
            'chunks': [],
            'metadata': {
                'normalization_timestamp': datetime.now().isoformat(),
                'supported_file_type': False,
                'chunk_count': 0
            },
            'error': error_msg
        }
    
    def save_normalized_data(self, normalized_data: List[Dict[str, Any]], 
                           output_path: Union[str, Path]) -> None:
        """Save normalized data to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(normalized_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(normalized_data)} normalized documents to {output_path}")
    
    def load_normalized_data(self, input_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load normalized data from JSON file."""
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.logger.info(f"Loaded {len(data)} normalized documents from {input_path}")
        return data