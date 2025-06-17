#!/usr/bin/env python3
"""
Directory-Informed ISNE: Focused Demonstration

Shows how to bootstrap a graph from directory structure and enhance it with ISNE.
Clear 3-step process: Bootstrap → Train → Compare

Usage: python simple_demo.py /path/to/directory
"""

import sys
import ast
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict

import networkx as nx
import numpy as np

# Suppress verbose logging for demo clarity
logging.basicConfig(level=logging.WARNING)

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class DirectoryGraph:
    """Bootstrap graph from directory structure."""
    
    def __init__(self, root_path: Path):
        self.graph = nx.Graph()  # Undirected for simplicity
        self.file_to_node = {}
        self.node_to_file = {}
        self.node_counter = 0
        
        print(f"📁 Analyzing directory: {root_path}")
        self._build_from_directory(root_path)
    
    def _build_from_directory(self, root_path: Path):
        """Build graph from filesystem structure."""
        
        # Step 1: Add files as nodes
        python_files = list(root_path.rglob("*.py"))
        doc_files = list(root_path.rglob("*.md")) + list(root_path.rglob("*.txt"))
        
        all_files = python_files + doc_files
        for file_path in all_files:
            if not file_path.name.startswith('.'):
                self._add_file_node(file_path)
        
        print(f"   📄 Added {len(all_files)} files as nodes")
        
        # Step 2: Add co-location edges (files in same directory)
        self._add_colocation_edges(all_files)
        
        # Step 3: Add import edges (Python imports)
        self._add_import_edges(python_files)
    
    def _add_file_node(self, file_path: Path):
        """Add file as graph node."""
        node_id = self.node_counter
        self.node_counter += 1
        
        file_type = "code" if file_path.suffix == ".py" else "docs"
        
        self.graph.add_node(node_id, 
                           file_path=str(file_path),
                           file_name=file_path.name,
                           file_type=file_type)
        
        self.file_to_node[str(file_path)] = node_id
        self.node_to_file[node_id] = str(file_path)
    
    def _add_colocation_edges(self, all_files: List[Path]):
        """Connect files in same directory (selectively for demo)."""
        # Group by directory
        dir_groups = defaultdict(list)
        for file_path in all_files:
            if str(file_path) in self.file_to_node:
                dir_groups[file_path.parent].append(self.file_to_node[str(file_path)])
        
        # Connect within directories (but not fully connected for demo)
        edges_added = 0
        for nodes in dir_groups.values():
            # Only connect each file to 2-3 neighbors (not all)
            for i, node_a in enumerate(nodes):
                max_connections = min(3, len(nodes) - 1)
                for j in range(1, max_connections + 1):
                    if i + j < len(nodes):
                        node_b = nodes[i + j]
                        self.graph.add_edge(node_a, node_b, edge_type="colocation", weight=1.0)
                        edges_added += 1
        
        print(f"   🔗 Added {edges_added} co-location edges")
    
    def _add_import_edges(self, python_files: List[Path]):
        """Connect files through import statements."""
        edges_added = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                tree = ast.parse(content)
                imports = self._extract_imports(tree)
                
                source_node = self.file_to_node.get(str(file_path))
                if not source_node:
                    continue
                
                for import_name in imports:
                    target_node = self._find_import_target(import_name, python_files)
                    if target_node and target_node != source_node:
                        self.graph.add_edge(source_node, target_node, 
                                          edge_type="import", weight=0.8)
                        edges_added += 1
            except:
                continue  # Skip files with syntax errors
        
        print(f"   📥 Added {edges_added} import edges")
    
    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract import names from AST."""
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        return imports
    
    def _find_import_target(self, import_name: str, python_files: List[Path]) -> int:
        """Find target file for import."""
        # Simple heuristic: look for matching filename
        for file_path in python_files:
            if file_path.stem == import_name or import_name in str(file_path):
                return self.file_to_node.get(str(file_path))
        return None
    
    def get_metrics(self) -> Dict[str, float]:
        """Get basic graph metrics."""
        if not self.graph.nodes:
            return {}
        
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_clustering": nx.average_clustering(self.graph),
            "connected_components": nx.number_connected_components(self.graph)
        }


class SimpleISNE:
    """Minimal ISNE implementation for demonstration."""
    
    def __init__(self, graph: nx.Graph, embedding_dim: int = 64):
        self.graph = graph
        self.embedding_dim = embedding_dim
        self.node_embeddings = {}
        
    def train(self, iterations: int = 10) -> Dict[str, float]:
        """Train ISNE embeddings (simplified version)."""
        print(f"🎯 Training ISNE for {iterations} iterations...")
        
        # Initialize random embeddings
        np.random.seed(42)
        for node in self.graph.nodes():
            self.node_embeddings[node] = np.random.randn(self.embedding_dim)
        
        # Simple neighborhood aggregation (ISNE core idea)
        for iteration in range(iterations):
            new_embeddings = {}
            
            for node in self.graph.nodes():
                neighbors = list(self.graph.neighbors(node))
                
                if neighbors:
                    # Aggregate neighbor embeddings (ISNE principle)
                    neighbor_embeds = [self.node_embeddings[n] for n in neighbors]
                    aggregated = np.mean(neighbor_embeds, axis=0)
                    
                    # Combine with current embedding
                    new_embeddings[node] = 0.7 * self.node_embeddings[node] + 0.3 * aggregated
                else:
                    new_embeddings[node] = self.node_embeddings[node]
            
            self.node_embeddings = new_embeddings
        
        # Normalize embeddings
        for node in self.node_embeddings:
            norm = np.linalg.norm(self.node_embeddings[node])
            if norm > 0:
                self.node_embeddings[node] /= norm
        
        return {"iterations": iterations, "nodes_trained": len(self.node_embeddings)}
    
    def get_enhanced_graph(self) -> nx.Graph:
        """Create enhanced graph with discovered relationships."""
        enhanced = self.graph.copy()
        edges_added = 0
        
        # Find new relationships based on embedding similarity
        nodes = list(self.graph.nodes())
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                if not enhanced.has_edge(node_a, node_b):
                    # Calculate embedding similarity
                    emb_a = self.node_embeddings[node_a]
                    emb_b = self.node_embeddings[node_b]
                    similarity = np.dot(emb_a, emb_b)
                    
                    # Add edge if similarity is high (threshold 0.5)
                    if similarity > 0.5:
                        enhanced.add_edge(node_a, node_b, 
                                        edge_type="isne_discovered", 
                                        weight=similarity)
                        edges_added += 1
        
        print(f"   ✨ ISNE discovered {edges_added} new relationships")
        return enhanced


def analyze_graph(graph: nx.Graph, name: str) -> Dict[str, float]:
    """Analyze graph and print metrics."""
    if not graph.nodes:
        return {}
    
    metrics = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": nx.density(graph),
        "avg_clustering": nx.average_clustering(graph) if not graph.is_directed() else nx.average_clustering(graph.to_undirected()),
        "connected_components": nx.number_connected_components(graph) if not graph.is_directed() else nx.number_weakly_connected_components(graph)
    }
    
    print(f"\n📊 {name} Metrics:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.3f}")
    
    return metrics


def find_theory_practice_bridges(graph: nx.Graph, directory_graph: DirectoryGraph) -> List[Tuple[str, str]]:
    """Find connections between documentation and code files."""
    bridges = []
    
    for node_a, node_b in graph.edges():
        type_a = graph.nodes[node_a].get('file_type')
        type_b = graph.nodes[node_b].get('file_type')
        
        if type_a != type_b:  # Cross-modal connection
            file_a = directory_graph.node_to_file[node_a]
            file_b = directory_graph.node_to_file[node_b]
            bridges.append((Path(file_a).name, Path(file_b).name))
    
    return bridges


def visualize_comparison(basic_graph: nx.Graph, enhanced_graph: nx.Graph, output_dir: Path = Path(".")):
    """Create before/after visualization if matplotlib available."""
    if not HAS_MATPLOTLIB:
        print("📊 Matplotlib not available - skipping visualization")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Basic graph
    pos1 = nx.spring_layout(basic_graph, seed=42)
    nx.draw(basic_graph, pos1, ax=ax1, node_size=30, node_color='lightblue', 
            edge_color='gray', alpha=0.7)
    ax1.set_title(f"Before ISNE\n{basic_graph.number_of_nodes()} nodes, {basic_graph.number_of_edges()} edges")
    
    # Enhanced graph  
    pos2 = nx.spring_layout(enhanced_graph, seed=42)
    edge_colors = ['red' if enhanced_graph.edges[u,v].get('edge_type') == 'isne_discovered' 
                   else 'gray' for u, v in enhanced_graph.edges()]
    nx.draw(enhanced_graph, pos2, ax=ax2, node_size=30, node_color='lightgreen',
            edge_color=edge_colors, alpha=0.7)
    ax2.set_title(f"After ISNE\n{enhanced_graph.number_of_nodes()} nodes, {enhanced_graph.number_of_edges()} edges\nRed = ISNE discovered")
    
    plt.tight_layout()
    output_file = output_dir / "isne_comparison.png"
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: {output_file}")
    plt.close()


def main():
    """Main demonstration."""
    if len(sys.argv) != 2:
        print("Usage: python simple_demo.py /path/to/directory")
        sys.exit(1)
    
    root_path = Path(sys.argv[1])
    if not root_path.exists():
        print(f"Error: Directory {root_path} does not exist")
        sys.exit(1)
    
    print("🚀 Directory-Informed ISNE Demonstration")
    print("=" * 50)
    
    # Step 1: Bootstrap graph from directory
    print("\n📁 STEP 1: Bootstrap Graph from Directory Structure")
    directory_graph = DirectoryGraph(root_path)
    basic_metrics = analyze_graph(directory_graph.graph, "Basic Directory Graph")
    
    # Step 2: Train ISNE
    print("\n🎯 STEP 2: Train ISNE on Directory Graph")
    isne = SimpleISNE(directory_graph.graph)
    training_results = isne.train(iterations=10)
    print(f"   ✅ Trained embeddings for {training_results['nodes_trained']} nodes")
    
    # Step 3: Get enhanced graph
    print("\n✨ STEP 3: Generate Enhanced Graph")
    enhanced_graph = isne.get_enhanced_graph()
    enhanced_metrics = analyze_graph(enhanced_graph, "ISNE-Enhanced Graph")
    
    # Step 4: Compare results
    print("\n📈 STEP 4: Comparison Results")
    print("-" * 30)
    print(f"Edges before ISNE: {basic_metrics.get('edges', 0)}")
    print(f"Edges after ISNE:  {enhanced_metrics.get('edges', 0)}")
    improvement = enhanced_metrics.get('edges', 0) - basic_metrics.get('edges', 0)
    print(f"New relationships:  {improvement}")
    
    # Step 5: Find theory-practice bridges
    bridges = find_theory_practice_bridges(enhanced_graph, directory_graph)
    print(f"\n🌉 Theory-Practice Bridges Found: {len(bridges)}")
    for i, (doc, code) in enumerate(bridges[:5]):
        print(f"   {i+1}. {doc} ↔ {code}")
    
    # Step 6: Create visualization
    print("\n📊 STEP 6: Create Visualization")
    visualize_comparison(directory_graph.graph, enhanced_graph)
    
    print(f"\n🎉 Demo Complete!")
    print(f"📊 Summary: Started with {basic_metrics.get('edges', 0)} relationships, ")
    print(f"           ISNE discovered {improvement} additional relationships")


if __name__ == "__main__":
    main()