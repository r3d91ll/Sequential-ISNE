#!/usr/bin/env python3
"""
Enhanced Hierarchical Processor for Sequential-ISNE

Implements the "Documentation-First Depth-First" strategy based on research validation
that filesystem hierarchy encodes human organizational knowledge as a "free" knowledge graph.

Key innovation: Leverages directory structure as an implicit knowledge graph
to achieve hierarchical knowledge organization without expensive computation.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

from streaming_processor import StreamingChunkProcessor, StreamingChunk, ChunkMetadata

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Different hierarchical processing strategies for research comparison."""
    RANDOM = "random"
    ALPHABETICAL = "alphabetical"
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    DOC_FIRST_DEPTH = "doc_first_depth"  # Our recommended approach


@dataclass
class HierarchicalMetrics:
    """Metrics for evaluating hierarchical processing effectiveness."""
    code_to_doc_discovery: float = 0.0
    doc_to_code_discovery: float = 0.0
    cross_component_discovery: float = 0.0
    hierarchical_coherence: float = 0.0
    semantic_bridge_strength: float = 0.0


class EnhancedHierarchicalProcessor(StreamingChunkProcessor):
    """
    Enhanced processor implementing the research-validated hierarchical strategy.
    
    Core insight: Directory structure encodes human organizational knowledge,
    providing a "free" knowledge graph that Sequential-ISNE can exploit.
    """
    
    def __init__(self, strategy: ProcessingStrategy = ProcessingStrategy.DOC_FIRST_DEPTH, **kwargs):
        super().__init__(**kwargs)
        self.strategy = strategy
        
        # Enhanced tracking for research validation
        self.hierarchical_metrics = HierarchicalMetrics()
        self.directory_depth_map: Dict[str, int] = {}
        self.component_boundaries: List[Tuple[int, str]] = []  # (chunk_id, component_name)
        self.semantic_bridges: List[Dict[str, Any]] = []
        
        # File classification for research analysis
        self.file_classifications: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Initialized Enhanced Hierarchical Processor with {strategy.value} strategy")
    
    def _analyze_directory_hierarchy(self, file_paths: List[str]) -> Dict[str, Any]:
        """Analyze the implicit knowledge graph encoded in directory structure."""
        hierarchy_analysis = {
            'max_depth': 0,
            'components': {},  # directory -> component analysis
            'cross_component_opportunities': [],
            'semantic_bridge_opportunities': [],
            'implicit_knowledge_graph': {
                'nodes': [],  # Directories as knowledge nodes
                'edges': []   # Parent-child and sibling relationships
            }
        }
        
        # Map directory depths
        for file_path in file_paths:
            path = Path(file_path)
            depth = len(path.parents) - 1  # Relative to root
            directory = str(path.parent)
            
            self.directory_depth_map[directory] = depth
            hierarchy_analysis['max_depth'] = max(hierarchy_analysis['max_depth'], depth)
            
            # Analyze component structure
            if directory not in hierarchy_analysis['components']:
                hierarchy_analysis['components'][directory] = self._analyze_component(directory, file_paths)
        
        # Build implicit knowledge graph
        directories = list(self.directory_depth_map.keys())
        
        # Nodes: Each directory is a knowledge component
        for directory in directories:
            analysis = hierarchy_analysis['components'][directory]
            hierarchy_analysis['implicit_knowledge_graph']['nodes'].append({
                'id': directory,
                'depth': self.directory_depth_map[directory],
                'component_type': analysis['component_type'],
                'has_research_papers': analysis['has_research_papers'],
                'has_documentation': analysis['has_documentation'],
                'has_implementation': analysis['has_implementation']
            })
        
        # Edges: Parent-child and sibling relationships
        for dir1 in directories:
            for dir2 in directories:
                if dir1 != dir2:
                    relationship = self._analyze_directory_relationship(dir1, dir2)
                    if relationship:
                        hierarchy_analysis['implicit_knowledge_graph']['edges'].append({
                            'from': dir1,
                            'to': dir2,
                            'relationship': relationship,
                            'strength': self._calculate_relationship_strength(dir1, dir2, relationship)
                        })
        
        logger.info(f"Analyzed implicit knowledge graph: {len(directories)} nodes, "
                   f"{len(hierarchy_analysis['implicit_knowledge_graph']['edges'])} edges")
        
        return hierarchy_analysis
    
    def _analyze_component(self, directory: str, all_files: List[str]) -> Dict[str, Any]:
        """Analyze a directory component for its knowledge role."""
        dir_files = [f for f in all_files if str(Path(f).parent) == directory]
        
        analysis = {
            'total_files': len(dir_files),
            'component_type': self._classify_component_type(directory),
            'has_research_papers': False,
            'has_documentation': False,
            'has_implementation': False,
            'file_types': {},
            'knowledge_density': 0.0,  # Measure of information richness
            'semantic_bridge_potential': 0.0
        }
        
        # Classify files and detect patterns
        for file_path in dir_files:
            file_analysis = self._classify_file_for_research(file_path)
            self.file_classifications[file_path] = file_analysis
            
            # Update component analysis
            if file_analysis['is_research_paper']:
                analysis['has_research_papers'] = True
            if file_analysis['is_documentation']:
                analysis['has_documentation'] = True  
            if file_analysis['is_implementation']:
                analysis['has_implementation'] = True
            
            # File type distribution
            file_type = file_analysis['primary_type']
            analysis['file_types'][file_type] = analysis['file_types'].get(file_type, 0) + 1
        
        # Calculate knowledge density (research contribution metric)
        analysis['knowledge_density'] = self._calculate_knowledge_density(analysis)
        analysis['semantic_bridge_potential'] = self._calculate_bridge_potential(analysis)
        
        return analysis
    
    def _classify_component_type(self, directory: str) -> str:
        """Classify directory component type for knowledge graph analysis."""
        dir_name = Path(directory).name.lower()
        
        if any(keyword in dir_name for keyword in ['src', 'lib', 'core']):
            return 'implementation'
        elif any(keyword in dir_name for keyword in ['docs', 'documentation']):
            return 'documentation'
        elif any(keyword in dir_name for keyword in ['test', 'spec']):
            return 'validation'
        elif any(keyword in dir_name for keyword in ['config', 'settings']):
            return 'configuration'
        elif any(keyword in dir_name for keyword in ['example', 'demo']):
            return 'demonstration'
        elif any(keyword in dir_name for keyword in ['research', 'papers']):
            return 'research'
        else:
            return 'general'
    
    def _classify_file_for_research(self, file_path: str) -> Dict[str, Any]:
        """Enhanced file classification for research analysis."""
        path = Path(file_path)
        extension = path.suffix.lower()
        name = path.name.lower()
        
        classification = {
            'primary_type': 'unknown',
            'is_research_paper': False,
            'is_documentation': False,
            'is_implementation': False,
            'is_configuration': False,
            'semantic_richness': 0.0,  # How much semantic information it contains
            'structural_importance': 0.0,  # How important for code structure
            'bridge_potential': 0.0  # Potential to bridge semantic/structural gaps
        }
        
        # Research papers (highest semantic richness)
        if extension in ['.pdf', '.tex', '.bib']:
            classification.update({
                'primary_type': 'research_paper',
                'is_research_paper': True,
                'semantic_richness': 1.0,
                'bridge_potential': 0.9  # Papers bridge theory to practice
            })
        
        # Documentation (high semantic richness)
        elif extension in ['.md', '.rst', '.txt', '.adoc']:
            classification.update({
                'primary_type': 'documentation',
                'is_documentation': True,
                'semantic_richness': 0.8,
                'bridge_potential': 0.7
            })
            
            # Special cases
            if 'readme' in name:
                classification['semantic_richness'] = 0.9
                classification['bridge_potential'] = 0.8
        
        # Implementation code (high structural importance)
        elif extension in ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.rs', '.go']:
            is_test = ('test' in name or 'spec' in name or 'test' in str(path.parent).lower())
            
            classification.update({
                'primary_type': 'test_code' if is_test else 'source_code',
                'is_implementation': True,
                'structural_importance': 0.9,
                'bridge_potential': 0.6
            })
        
        # Configuration (medium importance)
        elif extension in ['.yaml', '.yml', '.json', '.toml', '.ini']:
            classification.update({
                'primary_type': 'configuration',
                'is_configuration': True,
                'structural_importance': 0.5,
                'semantic_richness': 0.3
            })
        
        return classification
    
    def _calculate_knowledge_density(self, component_analysis: Dict[str, Any]) -> float:
        """Calculate knowledge density for research metrics."""
        density = 0.0
        
        # Research papers add high density
        if component_analysis['has_research_papers']:
            density += 0.4
        
        # Documentation adds medium density
        if component_analysis['has_documentation']:
            density += 0.3
        
        # Implementation adds structure density
        if component_analysis['has_implementation']:
            density += 0.2
        
        # Bonus for complete components (theory + practice)
        if (component_analysis['has_research_papers'] and 
            component_analysis['has_implementation']):
            density += 0.1  # Completeness bonus
        
        return min(1.0, density)
    
    def _calculate_bridge_potential(self, component_analysis: Dict[str, Any]) -> float:
        """Calculate potential for creating semantic-structural bridges."""
        if (component_analysis['has_documentation'] and 
            component_analysis['has_implementation']):
            return 0.8
        elif (component_analysis['has_research_papers'] and 
              component_analysis['has_implementation']):
            return 0.9  # Highest bridge potential: theory → practice
        elif component_analysis['has_documentation']:
            return 0.6
        else:
            return 0.3
    
    def _analyze_directory_relationship(self, dir1: str, dir2: str) -> Optional[str]:
        """Analyze relationship between two directories."""
        path1, path2 = Path(dir1), Path(dir2)
        
        # Parent-child relationship
        if path1 in path2.parents:
            return 'parent_child'
        elif path2 in path1.parents:
            return 'child_parent'
        
        # Sibling relationship (same parent)
        if path1.parent == path2.parent:
            return 'sibling'
        
        # Cousin relationship (same grandparent)
        if len(path1.parents) > 1 and len(path2.parents) > 1:
            if path1.parents[1] == path2.parents[1]:
                return 'cousin'
        
        return None
    
    def _calculate_relationship_strength(self, dir1: str, dir2: str, relationship: str) -> float:
        """Calculate strength of directory relationship."""
        strength_map = {
            'parent_child': 0.9,
            'child_parent': 0.9,
            'sibling': 0.7,
            'cousin': 0.5
        }
        
        base_strength = strength_map.get(relationship, 0.3)
        
        # Boost strength for complementary components
        comp1 = self.directory_depth_map.get(dir1, {})
        comp2 = self.directory_depth_map.get(dir2, {})
        
        # Documentation + Implementation bonus
        # (Would need component analysis - simplified for now)
        
        return base_strength
    
    def process_with_strategy(self, file_paths: List[str]) -> Iterator[StreamingChunk]:
        """Process files using the specified hierarchical strategy."""
        logger.info(f"Processing {len(file_paths)} files with {self.strategy.value} strategy")
        
        # Analyze directory hierarchy first
        hierarchy_analysis = self._analyze_directory_hierarchy(file_paths)
        
        if self.strategy == ProcessingStrategy.DOC_FIRST_DEPTH:
            yield from self._process_doc_first_depth(file_paths, hierarchy_analysis)
        elif self.strategy == ProcessingStrategy.DEPTH_FIRST:
            yield from self._process_depth_first(file_paths)
        elif self.strategy == ProcessingStrategy.BREADTH_FIRST:
            yield from self._process_breadth_first(file_paths)
        elif self.strategy == ProcessingStrategy.ALPHABETICAL:
            yield from self._process_alphabetical(file_paths)
        elif self.strategy == ProcessingStrategy.RANDOM:
            yield from self._process_random(file_paths)
        else:
            # Default to doc-first-depth
            yield from self._process_doc_first_depth(file_paths, hierarchy_analysis)
    
    def _process_doc_first_depth(self, file_paths: List[str], hierarchy_analysis: Dict[str, Any]) -> Iterator[StreamingChunk]:
        """
        Process using Documentation-First Depth-First strategy.
        
        This is our research-validated approach that creates optimal semantic bridges.
        """
        # Group files by directory and depth
        directory_files: Dict[str, List[str]] = {}
        for file_path in file_paths:
            directory = str(Path(file_path).parent)
            if directory not in directory_files:
                directory_files[directory] = []
            directory_files[directory].append(file_path)
        
        # Sort directories by depth (process shallow to deep)
        sorted_directories = sorted(
            directory_files.keys(), 
            key=lambda d: self.directory_depth_map.get(d, 999)
        )
        
        current_directory = None
        
        for directory in sorted_directories:
            # Add enhanced directory marker
            if self.add_directory_markers and directory != current_directory:
                yield self._create_enhanced_directory_marker(directory, hierarchy_analysis)
                current_directory = directory
                self.stats['total_directories'] += 1
            
            # Process files in doc-first order within directory
            dir_files = directory_files[directory]
            ordered_files = self._order_files_doc_first(dir_files)
            
            for file_path in ordered_files:
                # Track component boundaries for research analysis
                self.component_boundaries.append((self.current_chunk_id, directory))
                
                yield from self._process_single_file_enhanced(file_path, directory)
    
    def _order_files_doc_first(self, files: List[str]) -> List[str]:
        """Order files within directory: documentation first, then implementation."""
        def doc_first_priority(file_path: str) -> Tuple[int, int, str]:
            classification = self.file_classifications.get(file_path, {})
            primary_type = classification.get('primary_type', 'unknown')
            
            # Primary sort: type priority (documentation first)
            type_priorities = {
                'research_paper': 0,    # Papers provide theoretical context
                'documentation': 1,     # Docs provide usage context
                'configuration': 2,     # Config shows how it's used
                'source_code': 3,       # Implementation follows
                'test_code': 4,         # Tests validate implementation
                'unknown': 5
            }
            primary_priority = type_priorities.get(primary_type, 999)
            
            # Secondary sort: special file priorities
            path = Path(file_path)
            if path.name.lower().startswith('readme'):
                secondary_priority = 0
            elif 'main' in path.name.lower() or path.name == '__init__.py':
                secondary_priority = 1
            else:
                secondary_priority = 2
            
            return (primary_priority, secondary_priority, path.name)
        
        return sorted(files, key=doc_first_priority)
    
    def _process_single_file_enhanced(self, file_path: str, directory: str) -> Iterator[StreamingChunk]:
        """Process a single file with enhanced hierarchical awareness."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return
        
        classification = self.file_classifications.get(file_path, {})
        
        # Enhanced document start with hierarchical metadata
        if self.add_boundary_markers:
            start_chunk = self._create_enhanced_doc_boundary(file_path, "doc_start", classification)
            yield start_chunk
            doc_start_chunk_id = start_chunk.chunk_id
        else:
            doc_start_chunk_id = None
        
        # Process content with hierarchical awareness
        content_chunks = self._chunk_document_content(content, file_path)
        
        for i, chunk_content in enumerate(content_chunks):
            metadata = self._create_hierarchical_chunk_metadata(
                file_path, chunk_content, directory, classification, doc_start_chunk_id
            )
            
            chunk = StreamingChunk(
                chunk_id=self.current_chunk_id,
                content=chunk_content,
                metadata=metadata
            )
            
            # Enhanced tracking for research metrics
            self.chunk_registry[self.current_chunk_id] = chunk
            self.directory_chunks[directory].append(self.current_chunk_id)
            
            # Detect potential semantic bridges
            if classification.get('bridge_potential', 0) > 0.7:
                self.semantic_bridges.append({
                    'chunk_id': self.current_chunk_id,
                    'file_path': file_path,
                    'bridge_type': classification.get('primary_type'),
                    'potential': classification.get('bridge_potential')
                })
            
            self.stats['total_chunks'] += 1
            self.stats['content_chunks'] += 1
            self.current_chunk_id += 1
            
            yield chunk
        
        # Enhanced document end
        if self.add_boundary_markers:
            end_chunk = self._create_enhanced_doc_boundary(file_path, "doc_end", classification)
            yield end_chunk
        
        self.stats['total_documents'] += 1
    
    def _create_enhanced_directory_marker(self, directory: str, hierarchy_analysis: Dict[str, Any]) -> StreamingChunk:
        """Create enhanced directory marker with hierarchical analysis."""
        component_info = hierarchy_analysis['components'].get(directory, {})
        
        enhanced_content = [
            f"<DIRECTORY_START:{directory}>",
            f"<DEPTH:{self.directory_depth_map.get(directory, 0)}>",
            f"<COMPONENT_TYPE:{component_info.get('component_type', 'unknown')}>",
            f"<KNOWLEDGE_DENSITY:{component_info.get('knowledge_density', 0.0):.2f}>",
            f"<BRIDGE_POTENTIAL:{component_info.get('semantic_bridge_potential', 0.0):.2f}>"
        ]
        
        # Add hierarchical opportunities
        if component_info.get('has_research_papers') and component_info.get('has_implementation'):
            enhanced_content.append("<HIERARCHICAL_OPPORTUNITY:research_to_implementation>")
        
        if component_info.get('has_documentation') and component_info.get('has_implementation'):
            enhanced_content.append("<HIERARCHICAL_OPPORTUNITY:documentation_to_code>")
        
        marker_chunk = self._create_directory_marker(directory)
        marker_chunk.content = "\n".join(enhanced_content)
        
        return marker_chunk
    
    def _create_enhanced_doc_boundary(self, file_path: str, boundary_type: str, classification: Dict[str, Any]) -> StreamingChunk:
        """Create enhanced document boundary with classification metadata."""
        boundary_chunk = self._create_document_boundary(file_path, boundary_type)
        
        # Add classification metadata to boundary
        enhanced_content = [
            boundary_chunk.content,
            f"<FILE_TYPE:{classification.get('primary_type', 'unknown')}>",
            f"<SEMANTIC_RICHNESS:{classification.get('semantic_richness', 0.0):.2f}>",
            f"<STRUCTURAL_IMPORTANCE:{classification.get('structural_importance', 0.0):.2f}>",
            f"<BRIDGE_POTENTIAL:{classification.get('bridge_potential', 0.0):.2f}>"
        ]
        
        boundary_chunk.content = "\n".join(enhanced_content)
        return boundary_chunk
    
    def _create_hierarchical_chunk_metadata(self, file_path, chunk_content, directory, classification, doc_start_chunk_id):
        """Create enhanced metadata with hierarchical information."""
        import hashlib
        
        base_metadata = ChunkMetadata(
            chunk_id=self.current_chunk_id,
            chunk_type="content",
            doc_path=file_path,
            directory=directory,
            processing_order=self.current_chunk_id,
            doc_start_chunk_id=doc_start_chunk_id,
            content_hash=hashlib.md5(chunk_content.encode()).hexdigest()[:8],
            file_extension=Path(file_path).suffix
        )
        
        # Add hierarchical attributes (would extend ChunkMetadata in practice)
        return base_metadata
    
    def get_hierarchical_research_metrics(self) -> Dict[str, Any]:
        """Get comprehensive research metrics for paper validation."""
        base_stats = self.get_processing_statistics()
        
        research_metrics = {
            'processing_strategy': self.strategy.value,
            'implicit_knowledge_graph': {
                'directories_analyzed': len(self.directory_depth_map),
                'max_hierarchy_depth': max(self.directory_depth_map.values()) if self.directory_depth_map else 0,
                'component_boundaries': len(self.component_boundaries),
                'semantic_bridges_detected': len(self.semantic_bridges)
            },
            'file_classification_distribution': {},
            'component_analysis': {},
            'hierarchical_opportunities': [],
            'knowledge_density_distribution': [],
            'semantic_bridge_analysis': self.semantic_bridges
        }
        
        # File classification distribution
        classification_counts = {}
        for file_path, classification in self.file_classifications.items():
            primary_type = classification.get('primary_type', 'unknown')
            classification_counts[primary_type] = classification_counts.get(primary_type, 0) + 1
        research_metrics['file_classification_distribution'] = classification_counts
        
        return {**base_stats, 'research_metrics': research_metrics}
    
    # Placeholder methods for other strategies (for research comparison)
    def _process_depth_first(self, file_paths: List[str]) -> Iterator[StreamingChunk]:
        """Standard depth-first processing for comparison."""
        # Implementation would go here
        return iter([])
    
    def _process_breadth_first(self, file_paths: List[str]) -> Iterator[StreamingChunk]:
        """Breadth-first processing for comparison."""
        # Implementation would go here  
        return iter([])
    
    def _process_alphabetical(self, file_paths: List[str]) -> Iterator[StreamingChunk]:
        """Alphabetical processing baseline."""
        # Implementation would go here
        return iter([])
    
    def _process_random(self, file_paths: List[str]) -> Iterator[StreamingChunk]:
        """Random processing baseline."""
        # Implementation would go here
        return iter([])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo enhanced hierarchical processing with research validation
    print("=== Enhanced Hierarchical Processing Research Demo ===")
    
    # Sample files representing a research codebase with co-located papers
    research_files = [
        "README.md",
        "src/pathrag/PathRAG_Paper.pdf",
        "src/pathrag/README.md", 
        "src/pathrag/core.py",
        "src/pathrag/utils.py",
        "src/pathrag/test_pathrag.py",
        "src/isne/ISNE_Paper.pdf",
        "src/isne/README.md",
        "src/isne/model.py",
        "src/isne/test_isne.py",
        "src/sequential_isne/Sequential_ISNE_Paper.pdf",  # Our contribution!
        "src/sequential_isne/README.md",
        "src/sequential_isne/processor.py",
        "config/model_config.yaml",
        "docs/architecture.md"
    ]
    
    processor = EnhancedHierarchicalProcessor(
        strategy=ProcessingStrategy.DOC_FIRST_DEPTH,
        add_boundary_markers=True,
        add_directory_markers=True
    )
    
    # Process files (would use actual files in practice)
    print(f"\n📊 Processing {len(research_files)} files for research validation...")
    
    # Simulate processing by analyzing structure
    hierarchy_analysis = processor._analyze_directory_hierarchy(research_files)
    
    print(f"\n🧠 Implicit Knowledge Graph Analysis:")
    implicit_kg = hierarchy_analysis['implicit_knowledge_graph']
    print(f"   Knowledge Nodes: {len(implicit_kg['nodes'])}")
    print(f"   Relationship Edges: {len(implicit_kg['edges'])}")
    
    print(f"\n📑 Component Analysis:")
    for directory, analysis in hierarchy_analysis['components'].items():
        print(f"   {directory}:")
        print(f"     Type: {analysis['component_type']}")
        print(f"     Knowledge Density: {analysis['knowledge_density']:.2f}")
        print(f"     Bridge Potential: {analysis['semantic_bridge_potential']:.2f}")
        if analysis['has_research_papers']:
            print(f"     ✅ Contains research papers (theory → practice bridge)")
    
    print(f"\n🔬 Research Contribution:")
    print(f"   Filesystem hierarchy provides 'free' knowledge graph")
    print(f"   Documentation-first processing creates optimal semantic bridges")
    print(f"   Directory structure encodes human organizational knowledge")
    print(f"   Sequential-ISNE exploits this implicit structure for better performance")