#!/usr/bin/env python3
"""
Hierarchical Directory-Aware Processing for Sequential-ISNE

Implements intelligent directory processing that leverages hierarchical
co-location patterns to create stronger semantic-structural bridges.

Key innovation: Process documentation before code within each directory
to create natural DOC → CODE relationships that mirror human understanding.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from dataclasses import dataclass
from enum import Enum
import re

from streaming_processor import StreamingChunkProcessor, ProcessingOrder, StreamingChunk

logger = logging.getLogger(__name__)


class FileType(Enum):
    """File type classification for hierarchical processing."""
    RESEARCH_PAPER = "research_paper"      # .pdf research papers
    DOCUMENTATION = "documentation"        # .md, .rst, .txt
    SOURCE_CODE = "source_code"           # .py, .js, .ts, etc.
    TEST_CODE = "test_code"               # test_*.py, *_test.js
    CONFIG = "config"                     # .yaml, .json, .toml
    DATA = "data"                         # .csv, .json data files
    UNKNOWN = "unknown"


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical processing."""
    # Processing order within directories
    type_priority: Dict[FileType, int] = None
    
    # Cross-document relationship strength
    same_directory_bonus: float = 0.2
    hierarchical_bonus: float = 0.3  # DOC → CODE bonus
    
    # Research paper co-location detection
    detect_research_papers: bool = True
    paper_extensions: List[str] = None
    
    def __post_init__(self):
        if self.type_priority is None:
            self.type_priority = {
                FileType.RESEARCH_PAPER: 0,     # Papers provide theoretical context
                FileType.DOCUMENTATION: 1,      # Documentation provides usage context  
                FileType.CONFIG: 2,             # Config shows how it's used
                FileType.SOURCE_CODE: 3,        # Source implements the concepts
                FileType.TEST_CODE: 4,          # Tests validate implementation
                FileType.DATA: 5,               # Data files last
                FileType.UNKNOWN: 6
            }
        
        if self.paper_extensions is None:
            self.paper_extensions = ['.pdf', '.tex', '.bib']


class HierarchicalProcessor(StreamingChunkProcessor):
    """
    Directory-aware streaming processor that creates stronger semantic-structural bridges.
    
    Extends StreamingChunkProcessor with hierarchical intelligence:
    1. Detects research paper co-location
    2. Processes files in theory → practice order
    3. Creates enhanced relationship weights for co-located content
    4. Enables dual-modality learning (semantic + structural)
    """
    
    def __init__(self, hierarchical_config: Optional[HierarchicalConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.hierarchical_config = hierarchical_config or HierarchicalConfig()
        
        # Track hierarchical relationships
        self.directory_analysis: Dict[str, Dict[str, Any]] = {}
        self.research_paper_locations: Dict[str, List[str]] = {}  # directory -> paper_paths
        self.file_type_mapping: Dict[str, FileType] = {}
        
        logger.info("Initialized HierarchicalProcessor with directory-aware processing")
    
    def _classify_file_type(self, file_path: str) -> FileType:
        """Classify file type for hierarchical processing."""
        path = Path(file_path)
        extension = path.suffix.lower()
        name = path.name.lower()
        
        # Research papers (key innovation!)
        if extension in self.hierarchical_config.paper_extensions:
            return FileType.RESEARCH_PAPER
        
        # Documentation
        if extension in ['.md', '.rst', '.txt', '.adoc']:
            return FileType.DOCUMENTATION
        
        # Source code
        if extension in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.rs', '.go']:
            # Check if it's test code
            if (name.startswith('test_') or name.endswith('_test') or 
                'test' in str(path.parent).lower()):
                return FileType.TEST_CODE
            return FileType.SOURCE_CODE
        
        # Configuration
        if extension in ['.yaml', '.yml', '.json', '.toml', '.ini', '.cfg']:
            return FileType.CONFIG
        
        # Data files
        if extension in ['.csv', '.tsv', '.jsonl', '.parquet']:
            return FileType.DATA
        
        return FileType.UNKNOWN
    
    def _analyze_directory_structure(self, file_paths: List[str]) -> None:
        """Analyze directory structure for hierarchical relationships."""
        logger.info("Analyzing directory structure for hierarchical processing")
        
        # Group files by directory
        directory_files: Dict[str, List[str]] = {}
        for file_path in file_paths:
            directory = str(Path(file_path).parent)
            if directory not in directory_files:
                directory_files[directory] = []
            directory_files[directory].append(file_path)
        
        # Analyze each directory
        for directory, files in directory_files.items():
            analysis = self._analyze_single_directory(directory, files)
            self.directory_analysis[directory] = analysis
            
            # Track research paper co-location
            research_papers = [f for f in files if self._classify_file_type(f) == FileType.RESEARCH_PAPER]
            if research_papers:
                self.research_paper_locations[directory] = research_papers
                logger.info(f"Found {len(research_papers)} research papers in {directory}")
        
        logger.info(f"Analyzed {len(directory_files)} directories")
    
    def _analyze_single_directory(self, directory: str, files: List[str]) -> Dict[str, Any]:
        """Analyze a single directory for co-location patterns."""
        analysis = {
            'total_files': len(files),
            'file_types': {},
            'has_research_papers': False,
            'has_documentation': False,
            'has_source_code': False,
            'hierarchical_opportunities': []
        }
        
        # Classify all files in directory
        type_counts = {}
        for file_path in files:
            file_type = self._classify_file_type(file_path)
            self.file_type_mapping[file_path] = file_type
            
            type_counts[file_type] = type_counts.get(file_type, 0) + 1
            
            # Track key combinations
            if file_type == FileType.RESEARCH_PAPER:
                analysis['has_research_papers'] = True
            elif file_type == FileType.DOCUMENTATION:
                analysis['has_documentation'] = True
            elif file_type == FileType.SOURCE_CODE:
                analysis['has_source_code'] = True
        
        analysis['file_types'] = {ft.value: count for ft, count in type_counts.items()}
        
        # Identify hierarchical opportunities
        if analysis['has_research_papers'] and analysis['has_source_code']:
            analysis['hierarchical_opportunities'].append('research_to_implementation')
        
        if analysis['has_documentation'] and analysis['has_source_code']:
            analysis['hierarchical_opportunities'].append('documentation_to_code')
        
        return analysis
    
    def _sort_files_hierarchically(self, file_paths: List[str]) -> List[str]:
        """Sort files using hierarchical directory-aware strategy."""
        # First analyze directory structure
        self._analyze_directory_structure(file_paths)
        
        # Group by directory
        directory_files: Dict[str, List[str]] = {}
        for file_path in file_paths:
            directory = str(Path(file_path).parent)
            if directory not in directory_files:
                directory_files[directory] = []
            directory_files[directory].append(file_path)
        
        # Sort directories (could be enhanced with dependency analysis)
        sorted_directories = sorted(directory_files.keys())
        
        # Process each directory hierarchically
        final_order = []
        for directory in sorted_directories:
            dir_files = directory_files[directory]
            sorted_dir_files = self._sort_directory_files(dir_files)
            final_order.extend(sorted_dir_files)
        
        logger.info(f"Hierarchical sorting: {len(final_order)} files across {len(sorted_directories)} directories")
        return final_order
    
    def _sort_directory_files(self, files: List[str]) -> List[str]:
        """Sort files within a directory using hierarchical priorities."""
        def file_sort_key(file_path: str):
            file_type = self.file_type_mapping.get(file_path, FileType.UNKNOWN)
            priority = self.hierarchical_config.type_priority.get(file_type, 999)
            
            # Secondary sort by specific patterns
            path = Path(file_path)
            secondary = 0
            
            # Research papers first
            if file_type == FileType.RESEARCH_PAPER:
                secondary = 0
            # README files very early in documentation
            elif path.name.lower().startswith('readme'):
                secondary = 0
            # Main module files before utilities
            elif 'main' in path.name.lower() or path.name == '__init__.py':
                secondary = 1
            else:
                secondary = 2
            
            return (priority, secondary, path.name)
        
        return sorted(files, key=file_sort_key)
    
    def process_documents(self, file_paths: List[str]) -> Iterator[StreamingChunk]:
        """
        Process documents with hierarchical directory awareness.
        
        Creates enhanced relationships that bridge theory → practice.
        """
        logger.info(f"Starting hierarchical processing of {len(file_paths)} files")
        
        # Use hierarchical sorting instead of simple directory-first
        if self.processing_order == ProcessingOrder.DIRECTORY_FIRST:
            sorted_files = self._sort_files_hierarchically(file_paths)
        else:
            sorted_files = self._sort_files_by_strategy(file_paths)
        
        # Log hierarchical analysis
        for directory, analysis in self.directory_analysis.items():
            opportunities = analysis.get('hierarchical_opportunities', [])
            if opportunities:
                logger.info(f"Directory {directory}: {opportunities}")
        
        # Process files using parent class with sorted order
        current_directory = None
        for file_path in sorted_files:
            directory = str(Path(file_path).parent)
            
            # Add enhanced directory markers for hierarchical directories
            if self.add_directory_markers and directory != current_directory:
                if current_directory is not None:
                    logger.debug(f"Hierarchical transition: {current_directory} → {directory}")
                
                # Create enhanced directory marker with hierarchical info
                yield self._create_enhanced_directory_marker(directory)
                current_directory = directory
                self.stats['total_directories'] += 1
            
            # Process individual file (reuse parent logic)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                continue
            
            # Enhanced document processing with type awareness
            yield from self._process_file_hierarchically(file_path, content, directory)
    
    def _create_enhanced_directory_marker(self, directory: str) -> StreamingChunk:
        """Create directory marker with hierarchical analysis."""
        analysis = self.directory_analysis.get(directory, {})
        opportunities = analysis.get('hierarchical_opportunities', [])
        
        # Enhanced content for directory markers
        content_parts = [f"<DIRECTORY_START:{directory}>"]
        if opportunities:
            content_parts.append(f"<HIERARCHICAL_OPPORTUNITIES:{','.join(opportunities)}>")
        
        if directory in self.research_paper_locations:
            papers = self.research_paper_locations[directory]
            content_parts.append(f"<RESEARCH_PAPERS:{len(papers)}>")
        
        enhanced_content = "\n".join(content_parts)
        
        # Use parent method but with enhanced content
        marker_chunk = self._create_directory_marker(directory)
        marker_chunk.content = enhanced_content
        
        return marker_chunk
    
    def _process_file_hierarchically(self, file_path: str, content: str, directory: str) -> Iterator[StreamingChunk]:
        """Process a single file with hierarchical awareness."""
        file_type = self.file_type_mapping.get(file_path, FileType.UNKNOWN)
        
        # Document start with type information
        if self.add_boundary_markers:
            start_chunk = self._create_document_boundary(file_path, "doc_start")
            # Enhance with file type information
            start_chunk.content += f"\n<FILE_TYPE:{file_type.value}>"
            yield start_chunk
            doc_start_chunk_id = start_chunk.chunk_id
        else:
            doc_start_chunk_id = None
        
        # Process content chunks with type-aware chunking
        content_chunks = self._chunk_document_content(content, file_path)
        content_chunk_ids = []
        
        for i, chunk_content in enumerate(content_chunks):
            metadata = self._create_enhanced_chunk_metadata(
                file_path, chunk_content, directory, file_type, doc_start_chunk_id
            )
            
            chunk = StreamingChunk(
                chunk_id=self.current_chunk_id,
                content=chunk_content,
                metadata=metadata
            )
            
            # Track for enhanced relationship discovery
            self.directory_chunks[directory].append(self.current_chunk_id)
            content_chunk_ids.append(self.current_chunk_id)
            self.chunk_registry[self.current_chunk_id] = chunk
            
            self.stats['total_chunks'] += 1
            self.stats['content_chunks'] += 1
            self.current_chunk_id += 1
            
            yield chunk
        
        # Document end
        if self.add_boundary_markers:
            end_chunk = self._create_document_boundary(file_path, "doc_end")
            yield end_chunk
            doc_end_chunk_id = end_chunk.chunk_id
        else:
            doc_end_chunk_id = None
        
        # Track document boundaries
        if doc_start_chunk_id is not None and doc_end_chunk_id is not None:
            self.document_boundaries[file_path] = (doc_start_chunk_id, doc_end_chunk_id)
        
        self.stats['total_documents'] += 1
        logger.debug(f"Processed {file_path} ({file_type.value}): {len(content_chunks)} chunks")
    
    def _create_enhanced_chunk_metadata(self, file_path, chunk_content, directory, file_type, doc_start_chunk_id):
        """Create enhanced metadata with hierarchical information."""
        from streaming_processor import ChunkMetadata
        import hashlib
        
        return ChunkMetadata(
            chunk_id=self.current_chunk_id,
            chunk_type="content",
            doc_path=file_path,
            directory=directory,
            processing_order=self.current_chunk_id,
            doc_start_chunk_id=doc_start_chunk_id,
            content_hash=hashlib.md5(chunk_content.encode()).hexdigest()[:8],
            file_extension=Path(file_path).suffix
        )
    
    def get_hierarchical_relationships(self) -> List[Dict[str, Any]]:
        """
        Generate enhanced relationships using hierarchical analysis.
        
        Creates stronger relationships for:
        1. Research papers → Implementation code
        2. Documentation → Source code  
        3. Configuration → Usage code
        4. Tests → Source code
        """
        base_relationships = self.get_sequential_relationships()
        enhanced_relationships = []
        
        # Enhance base relationships with hierarchical bonuses
        for rel in base_relationships:
            enhanced_rel = rel.copy()
            
            from_chunk = self.chunk_registry.get(rel['from_chunk_id'])
            to_chunk = self.chunk_registry.get(rel['to_chunk_id'])
            
            if from_chunk and to_chunk:
                from_type = self.file_type_mapping.get(from_chunk.metadata.doc_path, FileType.UNKNOWN)
                to_type = self.file_type_mapping.get(to_chunk.metadata.doc_path, FileType.UNKNOWN)
                
                # Apply hierarchical bonuses
                bonus = 0.0
                
                # Research paper → Code relationships
                if from_type == FileType.RESEARCH_PAPER and to_type == FileType.SOURCE_CODE:
                    bonus = self.hierarchical_config.hierarchical_bonus
                    enhanced_rel['relationship_type'] = 'research_to_implementation'
                    enhanced_rel['context'] += ' [RESEARCH→CODE]'
                
                # Documentation → Code relationships
                elif from_type == FileType.DOCUMENTATION and to_type == FileType.SOURCE_CODE:
                    bonus = self.hierarchical_config.hierarchical_bonus * 0.8
                    enhanced_rel['relationship_type'] = 'documentation_to_code'
                    enhanced_rel['context'] += ' [DOC→CODE]'
                
                # Same directory bonus
                if from_chunk.metadata.directory == to_chunk.metadata.directory:
                    bonus += self.hierarchical_config.same_directory_bonus
                
                # Apply bonus
                enhanced_rel['confidence'] = min(1.0, rel['confidence'] + bonus)
                enhanced_rel['hierarchical_bonus'] = bonus
            
            enhanced_relationships.append(enhanced_rel)
        
        logger.info(f"Enhanced {len(enhanced_relationships)} relationships with hierarchical analysis")
        return enhanced_relationships
    
    def get_hierarchical_statistics(self) -> Dict[str, Any]:
        """Get comprehensive hierarchical processing statistics."""
        base_stats = self.get_processing_statistics()
        
        hierarchical_stats = {
            'directory_analysis': self.directory_analysis,
            'research_paper_locations': {
                k: len(v) for k, v in self.research_paper_locations.items()
            },
            'file_type_distribution': {},
            'hierarchical_opportunities': []
        }
        
        # File type distribution
        type_counts = {}
        for file_type in self.file_type_mapping.values():
            type_counts[file_type.value] = type_counts.get(file_type.value, 0) + 1
        hierarchical_stats['file_type_distribution'] = type_counts
        
        # Collect all hierarchical opportunities
        for directory, analysis in self.directory_analysis.items():
            opportunities = analysis.get('hierarchical_opportunities', [])
            hierarchical_stats['hierarchical_opportunities'].extend([
                {'directory': directory, 'opportunity': opp} for opp in opportunities
            ])
        
        return {**base_stats, 'hierarchical': hierarchical_stats}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo hierarchical processing
    print("=== Hierarchical Processing Demo ===")
    
    # Sample files with research papers co-located with code
    sample_files = [
        "src/pathrag/PathRAG_Paper.pdf",      # Research paper co-location!
        "src/pathrag/README.md",
        "src/pathrag/pathrag_implementation.py",
        "src/pathrag/test_pathrag.py",
        "src/isne/ISNE_Paper.pdf",            # Research paper co-location!
        "src/isne/README.md", 
        "src/isne/isne_model.py",
        "src/isne/test_isne.py",
        "docs/architecture.md",
        "config/model_config.yaml"
    ]
    
    processor = HierarchicalProcessor()
    
    # Show sorting strategy
    sorted_files = processor._sort_files_hierarchically(sample_files)
    
    print("\n📁 Hierarchical Processing Order:")
    for i, file_path in enumerate(sorted_files):
        file_type = processor.file_type_mapping.get(file_path, FileType.UNKNOWN)
        print(f"  {i+1:2d}. {file_path} ({file_type.value})")
    
    print(f"\n🧠 Hierarchical Opportunities:")
    for directory, analysis in processor.directory_analysis.items():
        opportunities = analysis.get('hierarchical_opportunities', [])
        if opportunities:
            print(f"  {directory}: {opportunities}")
    
    print(f"\n📑 Research Paper Co-locations:")
    for directory, papers in processor.research_paper_locations.items():
        print(f"  {directory}: {papers}")
    
    print("\nThis creates natural THEORY → PRACTICE relationships!")
    print("Research papers provide theoretical context for implementation code.")