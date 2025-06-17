#!/usr/bin/env python3
"""
Academic Test Suite for Directory-Informed ISNE

Comprehensive validation using the sequential-ISNE-testdata academic dataset.
Tests the core hypothesis: directory structure as implicit knowledge graph.

Usage: python academic_test.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import time

from simple_demo import DirectoryGraph, SimpleISNE, analyze_graph, find_theory_practice_bridges

# Suppress verbose logging for clean test output
logging.basicConfig(level=logging.WARNING)


class AcademicTestSuite:
    """Comprehensive academic validation test suite."""
    
    def __init__(self, testdata_root: Path):
        self.testdata_root = testdata_root
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_suite': 'academic_validation',
            'datasets': {},
            'summary': {}
        }
        
        print("🎓 Sequential-ISNE Academic Test Suite")
        print("=" * 60)
        print(f"📁 Test Data Root: {testdata_root}")
        print(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def run_full_validation(self) -> Dict[str, Any]:
        """Run complete academic validation suite."""
        
        # Test datasets in order of complexity
        test_datasets = [
            ("isne-original", "ISNE Original Repository"),
            ("isne-enhanced", "ISNE Enhanced with Theory Papers"),
            ("pathrag-original", "PathRAG Original Repository"), 
            ("pathrag-enhanced", "PathRAG Enhanced with Theory Papers"),
            ("graphrag-enhanced", "GraphRAG Enhanced with Theory Papers"),
        ]
        
        for dataset_name, description in test_datasets:
            dataset_path = self.testdata_root / "isne-testdata" / dataset_name
            if dataset_path.exists():
                print(f"\n📊 Testing: {description}")
                print("-" * 50)
                self._test_dataset(dataset_name, dataset_path, description)
            else:
                print(f"⚠️  Dataset not found: {dataset_path}")
        
        # Generate comparative analysis
        self._generate_comparative_analysis()
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _test_dataset(self, name: str, path: Path, description: str):
        """Test a single dataset."""
        start_time = time.time()
        
        try:
            # Step 1: Bootstrap directory graph
            print("   📁 Bootstrapping directory graph...")
            directory_graph = DirectoryGraph(path)
            basic_metrics = analyze_graph(directory_graph.graph, f"Basic {name}")
            
            # Step 2: Train ISNE
            print("   🎯 Training ISNE...")
            isne = SimpleISNE(directory_graph.graph, embedding_dim=128)  # Larger for academic scale
            training_results = isne.train(iterations=15)  # More iterations for academic datasets
            
            # Step 3: Get enhanced graph
            print("   ✨ Generating enhanced graph...")
            enhanced_graph = isne.get_enhanced_graph()
            enhanced_metrics = analyze_graph(enhanced_graph, f"Enhanced {name}")
            
            # Step 4: Find theory-practice bridges
            bridges = find_theory_practice_bridges(enhanced_graph, directory_graph)
            
            # Calculate key metrics
            improvement = enhanced_metrics.get('edges', 0) - basic_metrics.get('edges', 0)
            improvement_pct = (improvement / basic_metrics.get('edges', 1)) * 100
            
            # File type analysis
            file_types = self._analyze_file_types(directory_graph)
            
            # Store results
            dataset_results = {
                'description': description,
                'path': str(path),
                'processing_time': time.time() - start_time,
                'basic_metrics': basic_metrics,
                'enhanced_metrics': enhanced_metrics,
                'improvement': {
                    'new_relationships': improvement,
                    'improvement_percentage': improvement_pct,
                    'theory_practice_bridges': len(bridges)
                },
                'file_analysis': file_types,
                'bridges_sample': bridges[:10],  # First 10 bridges for inspection
                'training_results': training_results
            }
            
            self.results['datasets'][name] = dataset_results
            
            # Print summary
            print(f"   ✅ Completed in {time.time() - start_time:.1f}s")
            print(f"   📊 Relationships: {basic_metrics.get('edges', 0)} → {enhanced_metrics.get('edges', 0)} (+{improvement})")
            print(f"   🌉 Theory-Practice Bridges: {len(bridges)}")
            print(f"   📄 Files: {file_types.get('total', 0)} ({file_types.get('code', 0)} code, {file_types.get('docs', 0)} docs)")
            
        except Exception as e:
            print(f"   ❌ Error testing {name}: {e}")
            self.results['datasets'][name] = {
                'error': str(e),
                'description': description,
                'path': str(path)
            }
    
    def _analyze_file_types(self, directory_graph: DirectoryGraph) -> Dict[str, int]:
        """Analyze file type distribution."""
        file_types = {'code': 0, 'docs': 0, 'total': 0}
        
        for node_id in directory_graph.graph.nodes():
            file_type = directory_graph.graph.nodes[node_id].get('file_type', 'unknown')
            if file_type == 'code':
                file_types['code'] += 1
            elif file_type == 'docs':
                file_types['docs'] += 1
            file_types['total'] += 1
        
        return file_types
    
    def _generate_comparative_analysis(self):
        """Generate comparative analysis across datasets."""
        datasets = self.results['datasets']
        
        if not datasets:
            self.results['summary']['error'] = "No datasets successfully processed"
            return
        
        # Calculate aggregate metrics
        total_basic_edges = sum(d.get('basic_metrics', {}).get('edges', 0) for d in datasets.values() if 'basic_metrics' in d)
        total_enhanced_edges = sum(d.get('enhanced_metrics', {}).get('edges', 0) for d in datasets.values() if 'enhanced_metrics' in d)
        total_bridges = sum(d.get('improvement', {}).get('theory_practice_bridges', 0) for d in datasets.values() if 'improvement' in d)
        
        # Find best performing dataset
        best_improvement = 0
        best_dataset = None
        for name, data in datasets.items():
            if 'improvement' in data:
                improvement = data['improvement'].get('new_relationships', 0)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_dataset = name
        
        # Enhanced vs Original comparison
        enhanced_comparison = self._compare_enhanced_vs_original(datasets)
        
        self.results['summary'] = {
            'total_datasets_tested': len([d for d in datasets.values() if 'basic_metrics' in d]),
            'aggregate_metrics': {
                'total_basic_relationships': total_basic_edges,
                'total_enhanced_relationships': total_enhanced_edges,
                'total_new_relationships': total_enhanced_edges - total_basic_edges,
                'total_theory_practice_bridges': total_bridges
            },
            'best_performing_dataset': {
                'name': best_dataset,
                'improvement': best_improvement
            },
            'enhanced_vs_original': enhanced_comparison,
            'validation_status': 'SUCCESS' if total_bridges > 0 else 'PARTIAL'
        }
        
        print(f"\n🏆 ACADEMIC VALIDATION SUMMARY")
        print("=" * 60)
        print(f"📊 Datasets Tested: {self.results['summary']['total_datasets_tested']}")
        print(f"🔗 Total Relationships: {total_basic_edges} → {total_enhanced_edges} (+{total_enhanced_edges - total_basic_edges})")
        print(f"🌉 Theory-Practice Bridges: {total_bridges}")
        print(f"🥇 Best Performer: {best_dataset} (+{best_improvement} relationships)")
        print(f"✅ Validation Status: {self.results['summary']['validation_status']}")
    
    def _compare_enhanced_vs_original(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """Compare enhanced vs original repositories."""
        comparisons = {}
        
        # ISNE comparison
        isne_original = datasets.get('isne-original', {})
        isne_enhanced = datasets.get('isne-enhanced', {})
        if 'basic_metrics' in isne_original and 'basic_metrics' in isne_enhanced:
            comparisons['isne'] = {
                'original_relationships': isne_original['basic_metrics'].get('edges', 0),
                'enhanced_relationships': isne_enhanced['basic_metrics'].get('edges', 0),
                'original_bridges': isne_original['improvement'].get('theory_practice_bridges', 0),
                'enhanced_bridges': isne_enhanced['improvement'].get('theory_practice_bridges', 0)
            }
        
        # PathRAG comparison  
        pathrag_original = datasets.get('pathrag-original', {})
        pathrag_enhanced = datasets.get('pathrag-enhanced', {})
        if 'basic_metrics' in pathrag_original and 'basic_metrics' in pathrag_enhanced:
            comparisons['pathrag'] = {
                'original_relationships': pathrag_original['basic_metrics'].get('edges', 0),
                'enhanced_relationships': pathrag_enhanced['basic_metrics'].get('edges', 0),
                'original_bridges': pathrag_original['improvement'].get('theory_practice_bridges', 0),
                'enhanced_bridges': pathrag_enhanced['improvement'].get('theory_practice_bridges', 0)
            }
        
        return comparisons
    
    def _save_results(self):
        """Save test results to file."""
        output_dir = Path("academic_test_results")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"academic_validation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        # Also create a summary report
        summary_file = output_dir / f"validation_summary_{timestamp}.md"
        self._create_summary_report(summary_file)
        print(f"📄 Summary report: {summary_file}")
    
    def _create_summary_report(self, output_file: Path):
        """Create markdown summary report."""
        summary = self.results['summary']
        
        report = f"""# Academic Validation Report

Generated: {self.results['timestamp']}

## Test Results Summary

- **Datasets Tested**: {summary.get('total_datasets_tested', 0)}
- **Validation Status**: {summary.get('validation_status', 'UNKNOWN')}

## Aggregate Metrics

- **Basic Relationships**: {summary.get('aggregate_metrics', {}).get('total_basic_relationships', 0)}
- **Enhanced Relationships**: {summary.get('aggregate_metrics', {}).get('total_enhanced_relationships', 0)}
- **New Relationships**: {summary.get('aggregate_metrics', {}).get('total_new_relationships', 0)}
- **Theory-Practice Bridges**: {summary.get('aggregate_metrics', {}).get('total_theory_practice_bridges', 0)}

## Best Performing Dataset

- **Name**: {summary.get('best_performing_dataset', {}).get('name', 'None')}
- **Improvement**: +{summary.get('best_performing_dataset', {}).get('improvement', 0)} relationships

## Dataset Details

"""
        
        for name, data in self.results['datasets'].items():
            if 'basic_metrics' in data:
                report += f"""### {name}

- **Description**: {data.get('description', 'N/A')}
- **Processing Time**: {data.get('processing_time', 0):.1f}s
- **Relationships**: {data.get('basic_metrics', {}).get('edges', 0)} → {data.get('enhanced_metrics', {}).get('edges', 0)}
- **New Relationships**: +{data.get('improvement', {}).get('new_relationships', 0)}
- **Theory-Practice Bridges**: {data.get('improvement', {}).get('theory_practice_bridges', 0)}
- **Files**: {data.get('file_analysis', {}).get('total', 0)} total ({data.get('file_analysis', {}).get('code', 0)} code, {data.get('file_analysis', {}).get('docs', 0)} docs)

"""
        
        with open(output_file, 'w') as f:
            f.write(report)


def main():
    """Run academic test suite."""
    testdata_root = Path("/home/todd/ML-Lab/Olympus/sequential-ISNE-testdata")
    
    if not testdata_root.exists():
        print(f"❌ Test data directory not found: {testdata_root}")
        print("Please ensure the sequential-ISNE-testdata directory exists.")
        return 1
    
    # Run test suite
    test_suite = AcademicTestSuite(testdata_root)
    results = test_suite.run_full_validation()
    
    # Final status
    validation_status = results['summary'].get('validation_status', 'FAILED')
    if validation_status == 'SUCCESS':
        print(f"\n🎉 ACADEMIC VALIDATION COMPLETE: {validation_status}")
        return 0
    else:
        print(f"\n⚠️  ACADEMIC VALIDATION: {validation_status}")
        return 1


if __name__ == "__main__":
    exit(main())