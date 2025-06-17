#!/usr/bin/env python3
"""
Directory-Informed ISNE: Focused Demonstration

Shows how to bootstrap a graph from directory structure and enhance it with ISNE.
Clear 3-step process: Bootstrap → Train → Compare

Usage: python simple_demo.py /path/to/directory
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import networkx as nx
import numpy as np

# Import DirectoryGraph from src
from src.directory_graph import DirectoryGraph

# Suppress verbose logging for demo clarity
logging.basicConfig(level=logging.WARNING)

# Try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


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
        print("Example: python simple_demo.py .")
        sys.exit(1)
    
    root_path = Path(sys.argv[1])
    if not root_path.exists():
        print(f"Error: Directory {root_path} does not exist")
        sys.exit(1)
    
    print("🚀 Directory-Informed ISNE Demonstration")
    print("=" * 50)
    
    # Step 1: Bootstrap graph from directory
    print("\n📁 STEP 1: Bootstrap Graph from Directory Structure")
    directory_graph = DirectoryGraph()
    directory_graph.bootstrap_from_directory(root_path)
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