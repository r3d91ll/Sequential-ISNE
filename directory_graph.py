#!/usr/bin/env python3
"""
Directory-Informed Graph Bootstrap for Sequential-ISNE

Creates initial graph structure from filesystem directory organization,
then extends with imports, semantic similarity, and other signals.
This provides the proper graph structure that ISNE requires.
"""

import logging
import networkx as nx
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


class DirectoryGraph:
    """
    Bootstrap graph construction from directory structure.
    
    Uses filesystem organization as implicit knowledge graph:
    - Co-located files have stronger relationships
    - Directory hierarchy provides semantic grouping
    - Import statements create directed edges
    - Documentation-code proximity suggests theory-practice bridges
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.file_to_node = {}  # file_path -> node_id mapping
        self.node_to_file = {}  # node_id -> file_path mapping
        self.current_node_id = 0
        
        # Track different edge types
        self.edge_types = {
            'co_located': 0,
            'parent_child': 0,
            'imports': 0,
            'semantic': 0,
            'sequential': 0
        }
        
        logger.info("Initialized DirectoryGraph for bootstrap")
    
    def add_file_node(self, file_path: Path, content: str = "", file_type: str = "unknown") -> int:
        """Add a file as a node in the graph."""
        file_path = Path(file_path)
        
        if str(file_path) in self.file_to_node:
            return self.file_to_node[str(file_path)]
        
        node_id = self.current_node_id
        self.current_node_id += 1
        
        # Add node with metadata
        self.graph.add_node(node_id, 
                           file_path=str(file_path),
                           file_name=file_path.name,
                           directory=str(file_path.parent),
                           file_type=file_type,
                           extension=file_path.suffix,
                           content_preview=content[:200],  # First 200 chars
                           content_length=len(content))
        
        # Update mappings
        self.file_to_node[str(file_path)] = node_id
        self.node_to_file[node_id] = str(file_path)
        
        return node_id
    
    def bootstrap_from_directory(self, root_path: Path, file_contents: Dict[str, str] = None) -> None:
        """
        Bootstrap initial graph from directory structure.
        
        Args:
            root_path: Root directory to analyze
            file_contents: Optional mapping of file_path -> content
        """
        root_path = Path(root_path)
        file_contents = file_contents or {}
        
        logger.info(f"Bootstrapping graph from directory: {root_path}")
        
        # Phase 1: Add all files as nodes
        all_files = []
        for file_path in root_path.rglob('*'):
            if file_path.is_file() and self._should_include_file(file_path):
                content = file_contents.get(str(file_path), "")
                file_type = self._classify_file_type(file_path)
                node_id = self.add_file_node(file_path, content, file_type)
                all_files.append((node_id, file_path))
        
        logger.info(f"Added {len(all_files)} file nodes")
        
        # Phase 2: Add directory co-location edges (strong signal)
        self._add_colocation_edges(all_files)
        
        # Phase 3: Add parent-child directory edges (medium signal)
        self._add_hierarchy_edges(all_files)
        
        # Phase 4: Add import/reference edges (strong directed signal)
        self._add_import_edges(all_files, file_contents)
        
        logger.info(f"Bootstrap complete: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        logger.info(f"Edge types: {self.edge_types}")
    
    def _should_include_file(self, file_path: Path) -> bool:
        """Determine if file should be included in graph."""
        # Skip hidden files, binary files, etc.
        if file_path.name.startswith('.'):
            return False
        
        # Include code files
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.go'}
        if file_path.suffix.lower() in code_extensions:
            return True
        
        # Include documentation
        doc_extensions = {'.md', '.txt', '.rst', '.pdf'}
        if file_path.suffix.lower() in doc_extensions:
            return True
        
        # Include config files
        config_extensions = {'.json', '.yaml', '.yml', '.toml', '.ini'}
        if file_path.suffix.lower() in config_extensions:
            return True
        
        # Include specific filenames
        special_files = {'README', 'LICENSE', 'CHANGELOG', 'Makefile', 'Dockerfile'}
        if file_path.name.upper() in special_files:
            return True
        
        return False
    
    def _classify_file_type(self, file_path: Path) -> str:
        """Classify file type for different processing."""
        extension = file_path.suffix.lower()
        
        if extension == '.py':
            return 'python'
        elif extension in {'.js', '.ts', '.java', '.cpp', '.c', '.h', '.rs', '.go'}:
            return 'code'
        elif extension in {'.md', '.txt', '.rst'}:
            return 'documentation'
        elif extension == '.pdf':
            return 'pdf_document'
        elif extension in {'.json', '.yaml', '.yml', '.toml', '.ini'}:
            return 'config'
        else:
            return 'unknown'
    
    def _add_colocation_edges(self, all_files: List[Tuple[int, Path]]) -> None:
        """Add edges between files in the same directory (strong signal)."""
        # Group files by directory
        directory_groups = defaultdict(list)
        for node_id, file_path in all_files:
            directory_groups[file_path.parent].append(node_id)
        
        # Add edges within each directory
        for directory, node_ids in directory_groups.items():
            for i, node_a in enumerate(node_ids):
                for node_b in node_ids[i+1:]:
                    # Bidirectional co-location edge
                    self.graph.add_edge(node_a, node_b, 
                                      edge_type='co_located',
                                      weight=1.0,
                                      source='directory_colocation')
                    self.graph.add_edge(node_b, node_a, 
                                      edge_type='co_located',
                                      weight=1.0,
                                      source='directory_colocation')
                    self.edge_types['co_located'] += 2
        
        logger.info(f"Added {self.edge_types['co_located']} co-location edges")
    
    def _add_hierarchy_edges(self, all_files: List[Tuple[int, Path]]) -> None:
        """Add edges between parent and child directories (medium signal)."""
        # Group files by directory depth
        files_by_depth = defaultdict(list)
        for node_id, file_path in all_files:
            depth = len(file_path.parts)
            files_by_depth[depth].append((node_id, file_path))
        
        # Connect files in parent-child directories
        for depth in sorted(files_by_depth.keys()):
            if depth + 1 not in files_by_depth:
                continue
            
            parent_files = files_by_depth[depth]
            child_files = files_by_depth[depth + 1]
            
            for parent_node, parent_path in parent_files:
                for child_node, child_path in child_files:
                    # Check if child is actually in parent directory
                    if child_path.parent == parent_path.parent:
                        self.graph.add_edge(parent_node, child_node,
                                          edge_type='parent_child',
                                          weight=0.8,
                                          source='directory_hierarchy')
                        self.edge_types['parent_child'] += 1
        
        logger.info(f"Added {self.edge_types['parent_child']} hierarchy edges")
    
    def _add_import_edges(self, all_files: List[Tuple[int, Path]], file_contents: Dict[str, str]) -> None:
        """Add directed edges based on import statements (strong signal)."""
        python_files = [(node_id, file_path) for node_id, file_path in all_files 
                       if file_path.suffix == '.py']
        
        for node_id, file_path in python_files:
            content = file_contents.get(str(file_path), "")
            if not content:
                continue
            
            try:
                # Parse Python AST to find imports
                tree = ast.parse(content)
                imports = self._extract_imports_from_ast(tree)
                
                # Find target files for imports
                for import_name in imports:
                    target_node = self._find_import_target(import_name, file_path, all_files)
                    if target_node is not None:
                        self.graph.add_edge(node_id, target_node,
                                          edge_type='imports',
                                          weight=0.9,
                                          source='python_import',
                                          import_name=import_name)
                        self.edge_types['imports'] += 1
                        
            except (SyntaxError, Exception) as e:
                logger.debug(f"Failed to parse imports from {file_path}: {e}")
        
        logger.info(f"Added {self.edge_types['imports']} import edges")
    
    def _extract_imports_from_ast(self, tree: ast.AST) -> Set[str]:
        """Extract import statements from Python AST."""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    for alias in node.names:
                        imports.add(f"{node.module}.{alias.name}")
        
        return imports
    
    def _find_import_target(self, import_name: str, source_file: Path, all_files: List[Tuple[int, Path]]) -> Optional[int]:
        """Find the target file for an import statement."""
        # Convert import name to potential file paths
        potential_paths = [
            import_name.replace('.', '/') + '.py',
            import_name.replace('.', '/') + '/__init__.py',
            import_name + '.py'
        ]
        
        # Search for matching files
        for node_id, file_path in all_files:
            file_str = str(file_path)
            for potential in potential_paths:
                if file_str.endswith(potential):
                    return node_id
        
        return None
    
    def extend_with_semantic_similarity(self, embeddings: Dict[int, List[float]], threshold: float = 0.7) -> None:
        """Extend graph with semantic similarity edges (weak signal)."""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        if not embeddings:
            logger.warning("No embeddings provided for semantic extension")
            return
        
        logger.info(f"Extending graph with semantic similarity (threshold: {threshold})")
        
        node_ids = list(embeddings.keys())
        embedding_matrix = np.array([embeddings[node_id] for node_id in node_ids])
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(embedding_matrix)
        
        # Add edges for high similarity pairs (not already connected)
        added_edges = 0
        for i, node_a in enumerate(node_ids):
            for j, node_b in enumerate(node_ids[i+1:], i+1):
                similarity = similarities[i][j]
                
                if similarity >= threshold and not self.graph.has_edge(node_a, node_b):
                    # Check if different file types (cross-modal)
                    type_a = self.graph.nodes[node_a].get('file_type', 'unknown')
                    type_b = self.graph.nodes[node_b].get('file_type', 'unknown')
                    
                    if type_a != type_b:  # Cross-modal relationship
                        self.graph.add_edge(node_a, node_b,
                                          edge_type='semantic',
                                          weight=similarity,
                                          source='semantic_similarity',
                                          cross_modal=True)
                        self.graph.add_edge(node_b, node_a,
                                          edge_type='semantic', 
                                          weight=similarity,
                                          source='semantic_similarity',
                                          cross_modal=True)
                        added_edges += 2
                        self.edge_types['semantic'] += 2
        
        logger.info(f"Added {added_edges} semantic similarity edges")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics."""
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'is_connected': nx.is_weakly_connected(self.graph),
            'edge_types': self.edge_types.copy()
        }
        
        # File type distribution
        file_types = defaultdict(int)
        for node_id in self.graph.nodes():
            file_type = self.graph.nodes[node_id].get('file_type', 'unknown')
            file_types[file_type] += 1
        stats['file_type_distribution'] = dict(file_types)
        
        # Directory distribution
        directories = defaultdict(int)
        for node_id in self.graph.nodes():
            directory = self.graph.nodes[node_id].get('directory', 'unknown')
            directories[directory] += 1
        stats['directory_distribution'] = dict(directories)
        
        return stats
    
    def find_theory_practice_bridges(self) -> List[Dict[str, Any]]:
        """Find potential theory-practice bridges in the graph."""
        bridges = []
        
        # Look for documentation-code connections
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            if node_data.get('file_type') in ['documentation', 'pdf_document']:
                # Find connected code files
                for neighbor_id in self.graph.neighbors(node_id):
                    neighbor_data = self.graph.nodes[neighbor_id]
                    if neighbor_data.get('file_type') in ['python', 'code']:
                        edge_data = self.graph.edges[node_id, neighbor_id]
                        
                        bridges.append({
                            'theory_file': node_data['file_path'],
                            'practice_file': neighbor_data['file_path'],
                            'theory_node': node_id,
                            'practice_node': neighbor_id,
                            'connection_type': edge_data.get('edge_type', 'unknown'),
                            'strength': edge_data.get('weight', 0.0),
                            'source': edge_data.get('source', 'unknown')
                        })
        
        # Sort by strength
        bridges.sort(key=lambda x: x['strength'], reverse=True)
        return bridges
    
    def save_graph(self, output_path: str) -> None:
        """Save the graph structure."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as GraphML for visualization
        nx.write_graphml(self.graph, str(output_file.with_suffix('.graphml')))
        
        # Save statistics as JSON
        import json
        stats = self.get_graph_statistics()
        with open(output_file.with_suffix('.stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved DirectoryGraph to {output_file}")


if __name__ == "__main__":
    # Demo the DirectoryGraph
    logging.basicConfig(level=logging.INFO)
    
    # Test with current directory
    graph = DirectoryGraph()
    
    # Mock file contents for testing
    test_contents = {
        "demo.py": "from embedder import CodeBERTEmbedder\nfrom chunker import UnifiedChunker\n",
        "embedder.py": "class CodeBERTEmbedder:\n    def __init__(self):\n        pass\n",
        "README.md": "# Sequential-ISNE\nThis project implements graph embeddings."
    }
    
    graph.bootstrap_from_directory(Path('.'), test_contents)
    
    print("=== DirectoryGraph Demo ===")
    stats = graph.get_graph_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n=== Theory-Practice Bridges ===")
    bridges = graph.find_theory_practice_bridges()
    for i, bridge in enumerate(bridges[:5]):
        print(f"{i+1}. {bridge['theory_file']} ↔ {bridge['practice_file']} (strength: {bridge['strength']:.2f})")