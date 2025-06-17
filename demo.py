#!/usr/bin/env python3
"""
Sequential-ISNE Demo

User-friendly demonstration of Sequential-ISNE that can be pointed at any directory.
This will bootstrap a knowledge graph from the directory structure and enhance it
using ISNE to discover new relationships.

Usage: python demo.py /path/to/directory [--epochs 20] [--embedding-dim 128]

Examples:
    python demo.py .                           # Test current directory
    python demo.py ~/my-project --epochs 30   # Test with more training
    python demo.py /path/to/code --embedding-dim 256  # Larger embeddings
"""

import argparse
import json
import logging
import time
import networkx as nx
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

from src.directory_graph import DirectoryGraph
from src.sequential_isne import SequentialISNE, TrainingConfig


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


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'demo_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class SequentialISNEDemo:
    """User-friendly Sequential-ISNE demonstration."""
    
    def __init__(self, target_directory: Path, epochs: int = 20, embedding_dim: int = 128):
        self.target_directory = target_directory
        self.epochs = epochs
        self.embedding_dim = embedding_dim
        self.start_time = time.time()
        self.results: Dict[str, Any] = {
            'test_type': 'user_demo',
            'started_at': datetime.now().isoformat(),
            'target_directory': str(target_directory),
            'parameters': {
                'epochs': epochs,
                'embedding_dim': embedding_dim
            },
            'phases': {},
            'final_metrics': {}
        }
        
        print("🚀 Sequential-ISNE Demo")
        print("=" * 60)
        print(f"📁 Target Directory: {target_directory}")
        print(f"🎯 Training Epochs: {epochs}")
        print(f"📊 Embedding Dimension: {embedding_dim}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
    
    def run_demo(self) -> Dict[str, Any]:
        """Run the complete Sequential-ISNE demonstration."""
        
        # Phase 1: Analyze target directory
        self._phase_1_directory_analysis()
        
        # Phase 2: Bootstrap directory graph
        self._phase_2_bootstrap_graph()
        
        # Phase 3: Train ISNE embeddings
        self._phase_3_train_isne()
        
        # Phase 4: Generate enhanced graph
        self._phase_4_enhance_graph()
        
        # Phase 5: Analyze results
        self._phase_5_analyze_results()
        
        # Phase 6: Export results
        self._phase_6_export_results()
        
        return self.results
    
    def _phase_1_directory_analysis(self):
        """Phase 1: Analyze the target directory structure."""
        print("🔍 PHASE 1: Directory Analysis")
        print("-" * 40)
        
        phase_start = time.time()
        
        # Count files by type
        file_counts = {
            'python': 0, 'docs': 0, 'configs': 0, 'other': 0, 'total': 0
        }
        
        all_files = []
        for file_path in self.target_directory.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                all_files.append(file_path)
                file_counts['total'] += 1
                
                if file_path.suffix == '.py':
                    file_counts['python'] += 1
                elif file_path.suffix in {'.md', '.txt', '.rst', '.pdf'}:
                    file_counts['docs'] += 1
                elif file_path.suffix in {'.json', '.yaml', '.yml', '.toml'}:
                    file_counts['configs'] += 1
                else:
                    file_counts['other'] += 1
        
        # Calculate directory depth
        if all_files:
            max_depth = max(len(f.relative_to(self.target_directory).parts) for f in all_files)
        else:
            max_depth = 0
        
        phase_time = time.time() - phase_start
        
        print(f"   📊 Total files: {file_counts['total']}")
        print(f"   🐍 Python files: {file_counts['python']}")
        print(f"   📄 Documentation: {file_counts['docs']}")
        print(f"   ⚙️  Config files: {file_counts['configs']}")
        print(f"   📁 Max depth: {max_depth}")
        print(f"   🕒 Analysis time: {phase_time:.1f}s")
        
        self.results['phases']['phase_1'] = {
            'duration': phase_time,
            'file_counts': file_counts,
            'max_depth': max_depth,
            'total_files_found': len(all_files)
        }
    
    def _phase_2_bootstrap_graph(self):
        """Phase 2: Bootstrap knowledge graph from directory structure."""
        print(f"\n🏗️  PHASE 2: Bootstrap Knowledge Graph")
        print("-" * 40)
        
        phase_start = time.time()
        
        print("   🔄 Creating graph from directory structure...")
        
        # Create directory graph
        self.directory_graph = DirectoryGraph()
        self.directory_graph.bootstrap_from_directory(self.target_directory)
        basic_metrics = analyze_graph(self.directory_graph.graph, "Directory Graph")
        
        phase_time = time.time() - phase_start
        
        print(f"   ✅ Graph created!")
        print(f"   📊 Nodes: {basic_metrics.get('nodes', 0)}")
        print(f"   🔗 Edges: {basic_metrics.get('edges', 0)}")
        print(f"   📈 Density: {basic_metrics.get('density', 0):.4f}")
        print(f"   🕒 Bootstrap time: {phase_time:.1f}s")
        
        self.results['phases']['phase_2'] = {
            'duration': phase_time,
            'basic_metrics': basic_metrics
        }
    
    def _phase_3_train_isne(self):
        """Phase 3: Train ISNE embeddings on the knowledge graph."""
        print(f"\n🎯 PHASE 3: Train ISNE Embeddings")
        print("-" * 40)
        
        phase_start = time.time()
        
        print(f"   🔄 Training ISNE with {self.epochs} epochs...")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                print("   💻 Using CPU (GPU not available)")
        except ImportError:
            print("   ⚠️  PyTorch not available, using fallback training")
        
        # Configure Sequential-ISNE
        config = TrainingConfig(
            embedding_dim=self.embedding_dim,
            epochs=self.epochs,
            batch_size=32,
            learning_rate=0.001,
            device="auto"
        )
        self.isne = SequentialISNE(config)
        
        print(f"   🏃 Training on {self.directory_graph.graph.number_of_nodes()} nodes...")
        
        # Convert directory graph to chunks
        chunks = self._create_chunks_from_directory_graph()
        
        # Build graph and train
        training_start = time.time()
        self.isne.build_graph_from_directory_graph(self.directory_graph, chunks)
        training_results = self.isne.train_embeddings()
        training_time = time.time() - training_start
        
        # Save the trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/demo_isne_model_{timestamp}.json"
        self.isne.save_model(model_path)
        print(f"   💾 Model saved to: {model_path}")
        
        phase_time = time.time() - phase_start
        
        print(f"   ✅ Training complete!")
        print(f"   📊 Nodes trained: {training_results.get('nodes_trained', 0)}")
        print(f"   🔄 Epochs completed: {training_results.get('epochs_completed', 0)}")
        print(f"   🕒 Training time: {training_time:.1f}s")
        
        self.results['phases']['phase_3'] = {
            'duration': phase_time,
            'training_time': training_time,
            'training_results': training_results
        }
    
    def _phase_4_enhance_graph(self):
        """Phase 4: Generate enhanced graph with ISNE discoveries."""
        print(f"\n✨ PHASE 4: Enhance Graph with ISNE")
        print("-" * 40)
        
        phase_start = time.time()
        
        print("   🔄 Finding new relationships with ISNE...")
        
        # Create enhanced graph using ISNE similarity discoveries
        self.enhanced_graph = self._create_enhanced_graph_from_isne()
        enhanced_metrics = analyze_graph(self.enhanced_graph, "ISNE-Enhanced Graph")
        
        phase_time = time.time() - phase_start
        
        basic_edges = self.directory_graph.graph.number_of_edges()
        enhanced_edges = self.enhanced_graph.number_of_edges()
        improvement = enhanced_edges - basic_edges
        improvement_pct = (improvement / basic_edges) * 100 if basic_edges > 0 else 0
        
        print(f"   ✅ Enhancement complete!")
        print(f"   📊 Original edges: {basic_edges}")
        print(f"   📊 Enhanced edges: {enhanced_edges}")
        print(f"   📈 New relationships: +{improvement} (+{improvement_pct:.1f}%)")
        print(f"   🕒 Enhancement time: {phase_time:.1f}s")
        
        self.results['phases']['phase_4'] = {
            'duration': phase_time,
            'enhanced_metrics': enhanced_metrics,
            'improvement': {
                'new_edges': improvement,
                'improvement_percentage': improvement_pct
            }
        }
    
    def _phase_5_analyze_results(self):
        """Phase 5: Analyze and interpret results."""
        print(f"\n🔬 PHASE 5: Analyze Results")
        print("-" * 40)
        
        phase_start = time.time()
        
        print("   🔄 Finding theory-practice bridges...")
        
        # Find theory-practice bridges
        bridges = find_theory_practice_bridges(self.enhanced_graph, self.directory_graph)
        
        # Categorize files by type
        file_categories = {'code': 0, 'docs': 0, 'configs': 0, 'other': 0}
        for node in self.directory_graph.graph.nodes():
            file_type = self.directory_graph.graph.nodes[node].get('file_type', 'other')
            if file_type in file_categories:
                file_categories[file_type] += 1
            else:
                file_categories['other'] += 1
        
        phase_time = time.time() - phase_start
        total_time = time.time() - self.start_time
        
        print(f"   ✅ Analysis complete!")
        print(f"   🌉 Theory-practice bridges: {len(bridges)}")
        print(f"   📊 File categories: {file_categories}")
        print(f"   🕒 Total demo time: {total_time:.1f}s")
        
        # Show sample bridges
        if bridges:
            print("\n   🔍 Sample bridges discovered:")
            for i, (theory, practice) in enumerate(bridges[:5]):
                print(f"      {i+1}. {theory} ↔ {practice}")
        
        self.bridges = bridges
        self.results['phases']['phase_5'] = {
            'duration': phase_time,
            'total_bridges': len(bridges),
            'file_categories': file_categories,
            'sample_bridges': bridges[:10]
        }
        
        self.results['final_metrics'] = {
            'total_duration': total_time,
            'nodes_processed': self.directory_graph.graph.number_of_nodes(),
            'new_relationships': self.results['phases']['phase_4']['improvement']['new_edges'],
            'bridges_found': len(bridges)
        }
    
    def _phase_6_export_results(self):
        """Phase 6: Export results and create summary."""
        print(f"\n💾 PHASE 6: Export Results")
        print("-" * 40)
        
        phase_start = time.time()
        
        # Finalize results
        self.results['completed_at'] = datetime.now().isoformat()
        self.results['total_duration'] = time.time() - self.start_time
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON results
        json_file = Path(f"demo_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Markdown summary
        md_file = Path(f"demo_summary_{timestamp}.md")
        self._create_markdown_summary(md_file)
        
        phase_time = time.time() - phase_start
        total_time = self.results['total_duration']
        
        print(f"   ✅ Results exported!")
        print(f"   📄 JSON results: {json_file}")
        print(f"   📄 Markdown summary: {md_file}")
        print(f"   🕒 Export time: {phase_time:.1f}s")
        
        print(f"\n🎉 DEMO COMPLETE!")
        print("=" * 60)
        print(f"⏰ Total time: {total_time:.1f}s")
        print(f"📊 Files processed: {self.results['phases']['phase_1']['file_counts']['total']}")
        print(f"🔗 New relationships: +{self.results['final_metrics']['new_relationships']}")
        print(f"🌉 Theory-practice bridges: {self.results['final_metrics']['bridges_found']}")
        print("=" * 60)
        
        self.results['phases']['phase_6'] = {
            'duration': phase_time,
            'exports': [str(json_file), str(md_file)]
        }
    
    def _create_chunks_from_directory_graph(self) -> List[Dict[str, Any]]:
        """Create chunk representations from directory graph nodes."""
        chunks = []
        
        for node_id in self.directory_graph.graph.nodes():
            node_data = self.directory_graph.graph.nodes[node_id]
            file_path = node_data.get('file_path', 'unknown')
            
            # Read file content if possible
            content = ""
            try:
                if Path(file_path).exists() and Path(file_path).is_file():
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()[:2000]  # Limit content size for demo
            except:
                content = f"File: {Path(file_path).name}"
            
            chunk = {
                'chunk_id': node_id,
                'content': content,
                'document_metadata': {
                    'file_path': file_path,
                    'file_name': node_data.get('file_name', ''),
                    'file_type': node_data.get('file_type', 'unknown')
                },
                'embedding': []  # Will be generated during training
            }
            chunks.append(chunk)
        
        return chunks
    
    def _create_enhanced_graph_from_isne(self):
        """Create enhanced graph with ISNE similarity discoveries."""
        # Start with the original directory graph
        enhanced = self.directory_graph.graph.copy()
        edges_added = 0
        
        if self.isne.trained_embeddings is None:
            print("   ⚠️  No trained embeddings available, returning original graph")
            return enhanced
        
        # Use top-k similarity approach
        chunk_ids = list(self.isne.node_to_index.keys())
        k_similar = min(5, len(chunk_ids) - 1)  # Adaptive k for smaller graphs
        threshold = 0.8
        
        print(f"   🎯 Finding top-{k_similar} similar chunks (threshold: {threshold})...")
        
        for i, chunk_a in enumerate(chunk_ids):
            if i % max(1, len(chunk_ids) // 10) == 0:  # Progress updates
                print(f"   📊 Progress: {i}/{len(chunk_ids)} chunks processed...")
            
            # Find top-k similar chunks
            similar_chunks = self.isne.find_similar_chunks(
                chunk_a, 
                k=k_similar, 
                similarity_threshold=threshold
            )
            
            # Add edges for each similar chunk found
            for similar_chunk_id, similarity in similar_chunks:
                if not enhanced.has_edge(chunk_a, similar_chunk_id):
                    enhanced.add_edge(chunk_a, similar_chunk_id, 
                                    edge_type="isne_discovered",
                                    weight=similarity,
                                    source="sequential_isne_demo")
                    edges_added += 1
        
        print(f"   ✨ ISNE discovered {edges_added} new relationships")
        return enhanced
    
    def _create_markdown_summary(self, output_file: Path):
        """Create markdown summary of demo results."""
        summary = f"""# Sequential-ISNE Demo Results

## Demo Overview
- **Directory**: {self.results['target_directory']}
- **Started**: {self.results['started_at']}
- **Completed**: {self.results['completed_at']}
- **Duration**: {self.results['total_duration']:.1f} seconds

## Parameters
- **Epochs**: {self.results['parameters']['epochs']}
- **Embedding Dimension**: {self.results['parameters']['embedding_dim']}

## Results Summary
- **Files Processed**: {self.results['phases']['phase_1']['file_counts']['total']}
- **Python Files**: {self.results['phases']['phase_1']['file_counts']['python']}
- **Documentation**: {self.results['phases']['phase_1']['file_counts']['docs']}

## Graph Enhancement
- **Original Edges**: {self.results['phases']['phase_2']['basic_metrics']['edges']}
- **Enhanced Edges**: {self.results['phases']['phase_4']['enhanced_metrics']['edges']}
- **New Relationships**: +{self.results['phases']['phase_4']['improvement']['new_edges']}
- **Improvement**: {self.results['phases']['phase_4']['improvement']['improvement_percentage']:.1f}%

## ISNE Training
- **Training Time**: {self.results['phases']['phase_3']['training_time']:.1f}s
- **Epochs Completed**: {self.results['phases']['phase_3']['training_results']['epochs_completed']}
- **Nodes Trained**: {self.results['phases']['phase_3']['training_results']['nodes_trained']}

## Discovery Results
- **Theory-Practice Bridges**: {self.results['phases']['phase_5']['total_bridges']}

This demo validates Sequential-ISNE's ability to discover meaningful relationships in your codebase.
"""
        
        with open(output_file, 'w') as f:
            f.write(summary)


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(description='Sequential-ISNE Demo')
    parser.add_argument('directory', type=Path, help='Directory to analyze')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs (default: 20)')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension (default: 128)')
    parser.add_argument('--output-dir', type=Path, default=Path('.'), help='Output directory for results (default: current directory)')
    
    args = parser.parse_args()
    
    if not args.directory.exists():
        print(f"❌ Directory not found: {args.directory}")
        return 1
    
    if not args.directory.is_dir():
        print(f"❌ Path is not a directory: {args.directory}")
        return 1
    
    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to output directory for results
    import os
    original_dir = os.getcwd()
    os.chdir(args.output_dir)
    
    try:
        # Run demo
        demo = SequentialISNEDemo(
            target_directory=args.directory,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim
        )
        demo.run_demo()
        return 0
    finally:
        # Return to original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    exit(main())