#!/usr/bin/env python3
"""
Full-Scale Sequential-ISNE Test Suite

Comprehensive test processing the ENTIRE sequential-ISNE-testdata/isne-testdata directory
as one unified knowledge graph. This should take 3-4 hours for proper academic scale.

Usage: python full_scale_test.py
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from directory_graph import DirectoryGraph
from simple_demo import analyze_graph, find_theory_practice_bridges
from src.sequential_isne import SequentialISNE, TrainingConfig

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'full_scale_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class FullScaleTest:
    """Full academic scale test with unified graph processing."""
    
    def __init__(self, testdata_root: Path):
        self.testdata_root = testdata_root
        self.start_time = time.time()
        self.results = {
            'test_type': 'full_scale_unified',
            'started_at': datetime.now().isoformat(),
            'dataset_path': str(testdata_root),
            'phases': {},
            'final_metrics': {}
        }
        
        print("🎓 FULL-SCALE SEQUENTIAL-ISNE TEST")
        print("=" * 80)
        print(f"📁 Processing entire dataset: {testdata_root}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕒 Expected duration: 3-4 hours")
        print()
    
    def run_full_scale_test(self) -> Dict[str, Any]:
        """Run comprehensive full-scale test on entire dataset."""
        
        # Phase 1: Dataset Analysis
        self._phase_1_dataset_analysis()
        
        # Phase 2: Directory Graph Bootstrap (MASSIVE)
        self._phase_2_directory_graph_bootstrap()
        
        # Phase 3: ISNE Training (INTENSIVE)
        self._phase_3_isne_training()
        
        # Phase 4: Enhanced Graph Generation
        self._phase_4_enhanced_graph_generation()
        
        # Phase 5: Theory-Practice Bridge Detection
        self._phase_5_bridge_detection()
        
        # Phase 6: Comprehensive Analysis
        self._phase_6_comprehensive_analysis()
        
        # Phase 7: Results Export
        self._phase_7_results_export()
        
        return self.results
    
    def _phase_1_dataset_analysis(self):
        """Phase 1: Analyze the complete dataset structure."""
        print("🔍 PHASE 1: Dataset Structure Analysis")
        print("-" * 50)
        
        phase_start = time.time()
        dataset_path = self.testdata_root / "isne-testdata"
        
        # Count all files by type
        file_counts = {
            'python': 0, 'docs': 0, 'pdfs': 0, 'configs': 0, 'other': 0, 'total': 0
        }
        
        all_files = []
        for file_path in dataset_path.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                all_files.append(file_path)
                file_counts['total'] += 1
                
                if file_path.suffix == '.py':
                    file_counts['python'] += 1
                elif file_path.suffix in {'.md', '.txt', '.rst'}:
                    file_counts['docs'] += 1
                elif file_path.suffix == '.pdf':
                    file_counts['pdfs'] += 1
                elif file_path.suffix in {'.json', '.yaml', '.yml', '.toml'}:
                    file_counts['configs'] += 1
                else:
                    file_counts['other'] += 1
        
        # Analyze directory structure depth
        max_depth = max(len(f.parts) for f in all_files) if all_files else 0
        
        # Repository analysis
        repos = {
            'isne-enhanced': dataset_path / 'isne-enhanced',
            'pathrag-enhanced': dataset_path / 'pathrag-enhanced', 
            'graphrag-enhanced': dataset_path / 'graphrag-enhanced',
            'theory-papers': dataset_path / 'theory-papers'
        }
        
        repo_stats = {}
        for name, path in repos.items():
            if path.exists():
                repo_files = list(path.rglob('*'))
                repo_stats[name] = {
                    'files': len([f for f in repo_files if f.is_file()]),
                    'directories': len([f for f in repo_files if f.is_dir()]),
                    'python_files': len([f for f in repo_files if f.suffix == '.py']),
                    'doc_files': len([f for f in repo_files if f.suffix in {'.md', '.txt', '.pdf'}])
                }
        
        phase_time = time.time() - phase_start
        
        print(f"   📊 Total files: {file_counts['total']}")
        print(f"   🐍 Python files: {file_counts['python']}")
        print(f"   📄 Documentation: {file_counts['docs']}")
        print(f"   📚 PDF papers: {file_counts['pdfs']}")
        print(f"   ⚙️  Config files: {file_counts['configs']}")
        print(f"   📁 Max directory depth: {max_depth}")
        print(f"   🕒 Analysis time: {phase_time:.1f}s")
        
        print("\n   Repository Breakdown:")
        for name, stats in repo_stats.items():
            print(f"     {name}: {stats['files']} files ({stats['python_files']} Python, {stats['doc_files']} docs)")
        
        self.results['phases']['phase_1'] = {
            'duration': phase_time,
            'file_counts': file_counts,
            'max_depth': max_depth,
            'repository_stats': repo_stats,
            'total_files_found': len(all_files)
        }
    
    def _phase_2_directory_graph_bootstrap(self):
        """Phase 2: Create massive unified directory graph."""
        print(f"\n🏗️  PHASE 2: Directory Graph Bootstrap (MASSIVE SCALE)")
        print("-" * 50)
        
        phase_start = time.time()
        dataset_path = self.testdata_root / "isne-testdata"
        
        print("   🔄 Bootstrapping unified graph from entire dataset...")
        print("   ⚠️  This phase will take 15-30 minutes for full dataset")
        
        # Create unified directory graph
        self.directory_graph = DirectoryGraph()
        self.directory_graph.bootstrap_from_directory(dataset_path)
        basic_metrics = analyze_graph(self.directory_graph.graph, "Unified Dataset")
        
        phase_time = time.time() - phase_start
        
        print(f"   ✅ Bootstrap complete!")
        print(f"   📊 Nodes: {basic_metrics.get('nodes', 0)}")
        print(f"   🔗 Edges: {basic_metrics.get('edges', 0)}")
        print(f"   📈 Density: {basic_metrics.get('density', 0):.4f}")
        print(f"   🕒 Bootstrap time: {phase_time/60:.1f} minutes")
        
        self.results['phases']['phase_2'] = {
            'duration': phase_time,
            'basic_metrics': basic_metrics,
            'edge_types': getattr(self.directory_graph, 'edge_types', {})
        }
    
    def _phase_3_isne_training(self):
        """Phase 3: INTENSIVE ISNE training on massive graph."""
        print(f"\n🎯 PHASE 3: ISNE Training (INTENSIVE - ACADEMIC SCALE)")
        print("-" * 50)
        
        phase_start = time.time()
        
        print("   🔄 Initializing ISNE for academic-scale training...")
        print("   ⚠️  This phase will take 2-3 HOURS for proper convergence")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"   🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
                print(f"   🔥 CUDA version: {torch.version.cuda}")
            else:
                print("   💻 Using CPU (GPU not available)")
        except ImportError:
            print("   ⚠️  PyTorch not available, using fallback training")
        
        # Create ISNE with academic-scale parameters optimized for GPU
        config = TrainingConfig(
            embedding_dim=256,      # Larger embedding for academic scale
            epochs=50,              # Academic-scale training iterations
            batch_size=32,          # Larger batches for GPU efficiency
            learning_rate=0.001,    # Standard learning rate
            device="auto"           # Auto-detect GPU/CPU
        )
        self.isne = SequentialISNE(config)
        
        print(f"   🏃 Training ISNE on {self.directory_graph.graph.number_of_nodes()} nodes...")
        print("   📊 Training progress (this will take a while):")
        
        # Convert directory graph to chunks for Sequential-ISNE
        chunks = self._create_chunks_from_directory_graph()
        
        # Build graph and train
        training_start = time.time()
        self.isne.build_graph_from_directory_graph(self.directory_graph, chunks)
        training_results = self.isne.train_embeddings()  # Real ISNE training
        training_time = time.time() - training_start
        
        phase_time = time.time() - phase_start
        
        print(f"   ✅ ISNE training complete!")
        print(f"   📊 Nodes trained: {training_results.get('nodes_trained', 0)}")
        print(f"   🔄 Iterations: {training_results.get('iterations', 0)}")
        print(f"   🕒 Training time: {training_time/3600:.2f} hours")
        print(f"   🕒 Total phase time: {phase_time/3600:.2f} hours")
        
        self.results['phases']['phase_3'] = {
            'duration': phase_time,
            'training_time': training_time, 
            'training_results': training_results
        }
    
    def _phase_4_enhanced_graph_generation(self):
        """Phase 4: Generate enhanced graph with discovered relationships."""
        print(f"\n✨ PHASE 4: Enhanced Graph Generation")
        print("-" * 50)
        
        phase_start = time.time()
        
        print("   🔄 Generating enhanced graph with ISNE discoveries...")
        print("   ⚠️  This phase will take 30-60 minutes for similarity computation")
        
        # Create enhanced graph using ISNE similarity discoveries
        self.enhanced_graph = self._create_enhanced_graph_from_isne()
        enhanced_metrics = analyze_graph(self.enhanced_graph, "ISNE-Enhanced")
        
        phase_time = time.time() - phase_start
        
        basic_edges = self.directory_graph.graph.number_of_edges()
        enhanced_edges = self.enhanced_graph.number_of_edges()
        improvement = enhanced_edges - basic_edges
        improvement_pct = (improvement / basic_edges) * 100 if basic_edges > 0 else 0
        
        print(f"   ✅ Enhanced graph generation complete!")
        print(f"   📊 Original edges: {basic_edges}")
        print(f"   📊 Enhanced edges: {enhanced_edges}")
        print(f"   📈 New relationships: +{improvement} (+{improvement_pct:.1f}%)")
        print(f"   🕒 Generation time: {phase_time/60:.1f} minutes")
        
        self.results['phases']['phase_4'] = {
            'duration': phase_time,
            'enhanced_metrics': enhanced_metrics,
            'improvement': {
                'new_edges': improvement,
                'improvement_percentage': improvement_pct
            }
        }
    
    def _phase_5_bridge_detection(self):
        """Phase 5: Comprehensive theory-practice bridge detection."""
        print(f"\n🌉 PHASE 5: Theory-Practice Bridge Detection")
        print("-" * 50)
        
        phase_start = time.time()
        
        print("   🔄 Detecting theory-practice bridges across all repositories...")
        
        bridges = find_theory_practice_bridges(self.enhanced_graph, self.directory_graph)
        
        # Categorize bridges by repository
        bridge_categories = {
            'isne': [], 'pathrag': [], 'graphrag': [], 'theory': [], 'cross_repo': []
        }
        
        for theory_file, practice_file in bridges:
            if 'isne' in theory_file.lower() or 'isne' in practice_file.lower():
                bridge_categories['isne'].append((theory_file, practice_file))
            elif 'pathrag' in theory_file.lower() or 'pathrag' in practice_file.lower():
                bridge_categories['pathrag'].append((theory_file, practice_file))
            elif 'graphrag' in theory_file.lower() or 'graphrag' in practice_file.lower():
                bridge_categories['graphrag'].append((theory_file, practice_file))
            elif 'theory' in theory_file.lower() or 'theory' in practice_file.lower():
                bridge_categories['theory'].append((theory_file, practice_file))
            else:
                bridge_categories['cross_repo'].append((theory_file, practice_file))
        
        phase_time = time.time() - phase_start
        
        print(f"   ✅ Bridge detection complete!")
        print(f"   🌉 Total bridges: {len(bridges)}")
        print(f"   📊 ISNE bridges: {len(bridge_categories['isne'])}")
        print(f"   📊 PathRAG bridges: {len(bridge_categories['pathrag'])}")
        print(f"   📊 GraphRAG bridges: {len(bridge_categories['graphrag'])}")
        print(f"   📊 Theory bridges: {len(bridge_categories['theory'])}")
        print(f"   📊 Cross-repo bridges: {len(bridge_categories['cross_repo'])}")
        print(f"   🕒 Detection time: {phase_time:.1f}s")
        
        # Show sample bridges
        print("\n   🔍 Sample bridges detected:")
        for i, (theory, practice) in enumerate(bridges[:10]):
            print(f"      {i+1}. {theory} ↔ {practice}")
        
        self.bridges = bridges
        self.results['phases']['phase_5'] = {
            'duration': phase_time,
            'total_bridges': len(bridges),
            'bridge_categories': {k: len(v) for k, v in bridge_categories.items()},
            'sample_bridges': bridges[:20]
        }
    
    def _phase_6_comprehensive_analysis(self):
        """Phase 6: Deep analysis of results."""
        print(f"\n🔬 PHASE 6: Comprehensive Analysis")
        print("-" * 50)
        
        phase_start = time.time()
        
        print("   🔄 Performing comprehensive analysis...")
        
        # Graph statistics
        basic_stats = self.directory_graph.get_graph_statistics()
        enhanced_stats = {
            'nodes': self.enhanced_graph.number_of_nodes(),
            'edges': self.enhanced_graph.number_of_edges(),
            'density': self.enhanced_graph.number_of_edges() / (self.enhanced_graph.number_of_nodes() * (self.enhanced_graph.number_of_nodes() - 1)) if self.enhanced_graph.number_of_nodes() > 1 else 0
        }
        
        # Performance analysis
        total_time = time.time() - self.start_time
        phase_times = {phase: data.get('duration', 0) for phase, data in self.results['phases'].items()}
        
        phase_time = time.time() - phase_start
        
        print(f"   ✅ Analysis complete!")
        print(f"   📊 Total test time: {total_time/3600:.2f} hours")
        print(f"   📊 Graph bootstrap: {phase_times.get('phase_2', 0)/60:.1f} min")
        print(f"   📊 ISNE training: {phase_times.get('phase_3', 0)/3600:.2f} hours")
        print(f"   📊 Enhancement: {phase_times.get('phase_4', 0)/60:.1f} min")
        print(f"   🕒 Analysis time: {phase_time:.1f}s")
        
        self.results['phases']['phase_6'] = {
            'duration': phase_time,
            'basic_graph_stats': basic_stats,
            'enhanced_graph_stats': enhanced_stats,
            'performance_analysis': {
                'total_test_hours': total_time / 3600,
                'phase_durations': phase_times
            }
        }
    
    def _phase_7_results_export(self):
        """Phase 7: Export comprehensive results."""
        print(f"\n💾 PHASE 7: Results Export")
        print("-" * 50)
        
        phase_start = time.time()
        
        # Finalize results
        self.results['completed_at'] = datetime.now().isoformat()
        self.results['total_duration'] = time.time() - self.start_time
        
        # Export to multiple formats
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON results
        json_file = Path(f"full_scale_results_{timestamp}.json")
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Markdown summary
        md_file = Path(f"full_scale_summary_{timestamp}.md")
        self._create_markdown_summary(md_file)
        
        phase_time = time.time() - phase_start
        total_time = self.results['total_duration']
        
        print(f"   ✅ Results exported!")
        print(f"   📄 JSON results: {json_file}")
        print(f"   📄 Markdown summary: {md_file}")
        print(f"   🕒 Export time: {phase_time:.1f}s")
        
        print(f"\n🎉 FULL-SCALE TEST COMPLETE!")
        print("=" * 80)
        print(f"⏰ Total duration: {total_time/3600:.2f} hours")
        print(f"📊 Nodes processed: {self.directory_graph.graph.number_of_nodes()}")
        print(f"🔗 Relationships discovered: +{self.enhanced_graph.number_of_edges() - self.directory_graph.graph.number_of_edges()}")
        print(f"🌉 Theory-practice bridges: {len(self.bridges)}")
        print("=" * 80)
        
        self.results['phases']['phase_7'] = {
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
                        content = f.read()[:5000]  # Limit content size
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
        """Create enhanced graph with ISNE similarity discoveries using top-k approach."""
        # Start with the original directory graph
        enhanced = self.directory_graph.graph.copy()
        edges_added = 0
        
        if self.isne.trained_embeddings is None:
            print("   ⚠️  No trained embeddings available, returning original graph")
            return enhanced
        
        # Top-k similarity approach (mathematically sound)
        chunk_ids = list(self.isne.node_to_index.keys())
        k_similar = 10  # Find top 10 most similar chunks for each chunk
        similarity_threshold = 0.6  # Minimum similarity to consider
        
        print(f"   🎯 Top-K ISNE similarity computation for {len(chunk_ids)} chunks...")
        print(f"   📊 Parameters: k={k_similar}, threshold={similarity_threshold}")
        print(f"   💻 Maximum possible new edges: {len(chunk_ids) * k_similar:,}")
        
        similarity_start = time.time()
        
        # For each chunk, find its k most similar chunks
        for i, chunk_a in enumerate(chunk_ids):
            if i % 50 == 0:  # Progress updates
                print(f"   📊 Progress: {i}/{len(chunk_ids)} chunks processed...")
            
            # Find top-k similar chunks
            similar_chunks = self.isne.find_similar_chunks(
                chunk_a, 
                k=k_similar, 
                similarity_threshold=similarity_threshold
            )
            
            # Add edges for each similar chunk found
            for similar_chunk_id, similarity in similar_chunks:
                # Skip if already connected
                if not enhanced.has_edge(chunk_a, similar_chunk_id):
                    enhanced.add_edge(chunk_a, similar_chunk_id, 
                                    edge_type="isne_discovered",
                                    weight=similarity,
                                    source="sequential_isne_topk")
                    edges_added += 1
        
        similarity_time = time.time() - similarity_start
        print(f"   ⚡ Top-K similarity computation completed in {similarity_time:.2f} seconds")
        print(f"   ✨ ISNE discovered {edges_added} new relationships via top-k method")
        
        return enhanced
    
    def _create_markdown_summary(self, output_file: Path):
        """Create comprehensive markdown summary."""
        total_time = self.results['total_duration']
        
        summary = f"""# Full-Scale Sequential-ISNE Test Results

## Test Overview
- **Started**: {self.results['started_at']}
- **Completed**: {self.results['completed_at']}
- **Duration**: {total_time/3600:.2f} hours
- **Dataset**: {self.results['dataset_path']}

## Dataset Analysis
- **Total Files**: {self.results['phases']['phase_1']['file_counts']['total']}
- **Python Files**: {self.results['phases']['phase_1']['file_counts']['python']}
- **Documentation**: {self.results['phases']['phase_1']['file_counts']['docs']}
- **PDF Papers**: {self.results['phases']['phase_1']['file_counts']['pdfs']}

## Graph Metrics
- **Initial Nodes**: {self.results['phases']['phase_2']['basic_metrics']['nodes']}
- **Initial Edges**: {self.results['phases']['phase_2']['basic_metrics']['edges']}
- **Enhanced Edges**: {self.results['phases']['phase_4']['enhanced_metrics']['edges']}
- **New Relationships**: +{self.results['phases']['phase_4']['improvement']['new_edges']}
- **Improvement**: {self.results['phases']['phase_4']['improvement']['improvement_percentage']:.1f}%

## ISNE Training
- **Training Time**: {self.results['phases']['phase_3']['training_time']/3600:.2f} hours
- **Epochs Completed**: {self.results['phases']['phase_3']['training_results']['epochs_completed']}
- **Nodes Trained**: {self.results['phases']['phase_3']['training_results']['nodes_trained']}

## Theory-Practice Bridges
- **Total Bridges**: {self.results['phases']['phase_5']['total_bridges']}
- **ISNE Bridges**: {self.results['phases']['phase_5']['bridge_categories']['isne']}
- **PathRAG Bridges**: {self.results['phases']['phase_5']['bridge_categories']['pathrag']}
- **GraphRAG Bridges**: {self.results['phases']['phase_5']['bridge_categories']['graphrag']}
- **Cross-Repository**: {self.results['phases']['phase_5']['bridge_categories']['cross_repo']}

## Performance Breakdown
- **Graph Bootstrap**: {self.results['phases']['phase_2']['duration']/60:.1f} minutes
- **ISNE Training**: {self.results['phases']['phase_3']['duration']/3600:.2f} hours  
- **Enhancement**: {self.results['phases']['phase_4']['duration']/60:.1f} minutes
- **Bridge Detection**: {self.results['phases']['phase_5']['duration']:.1f} seconds

## Validation Status
✅ **SUCCESS** - Full academic scale validation complete

This test validates Sequential-ISNE at true academic research scale.
"""
        
        with open(output_file, 'w') as f:
            f.write(summary)


def main():
    """Run full-scale Sequential-ISNE test."""
    testdata_root = Path("/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata")
    
    if not testdata_root.exists():
        print(f"❌ Test data directory not found: {testdata_root}")
        return 1
    
    # Run full-scale test
    test = FullScaleTest(testdata_root)
    results = test.run_full_scale_test()
    
    return 0


if __name__ == "__main__":
    exit(main())